# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-memory, direct-to-TP-shard loader for DASH-Q safetensors."""

import json
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.transformers_utils.dashq import (
    DashQTensorReader,
    _unique_object,
    load_matrix,
    validate_metadata,
)
from vllm.transformers_utils.repo_utils import hf_api

logger = init_logger(__name__)


class DashQModelLoader(BaseModelLoader):
    def _folder(self, config):
        if Path(config.model).is_dir():
            return config.model
        return hf_api().snapshot_download(
            config.model,
            revision=config.revision,
            cache_dir=self.load_config.download_dir,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model"],
        )

    def download_model(self, model_config):
        self._folder(model_config)

    def create_model(self, vllm_config, model_config, prefix=""):
        p = vllm_config.parallel_config
        if (
            p.pipeline_parallel_size != 1
            or p.tensor_parallel_size not in (1, 2)
            or p.data_parallel_size != 1
            or p.enable_expert_parallel
            or p.use_sequence_parallel_moe
            or p.enable_eplb
        ):
            raise ValueError("DASH-Q supports TP=1/2, PP=DP=1 and no EP/SP/EPLB")
        if model_config.quantization != "dashq":
            raise ValueError("--load-format dashq requires --quantization dashq")
        if getattr(model_config.hf_config, "model_type", None) != "nemotron_h":
            raise ValueError("DASH-Q currently supports NemotronH only")
        if vllm_config.speculative_config is not None or vllm_config.lora_config:
            raise ValueError("DASH-Q does not yet support speculation or LoRA")
        return super().create_model(vllm_config, model_config, prefix)

    def load_weights(self, model, model_config):
        folder = self._folder(model_config)
        with (Path(folder) / "dashq_config.json").open() as f:
            metadata = validate_metadata(json.load(f, object_pairs_hook=_unique_object))
        reader = DashQTensorReader(folder)
        consumed, sources, quant_params = set(), set(), set()
        try:
            with torch.no_grad():
                for layer in model.modules():
                    for source, prefix, expert, slices, offset in getattr(
                        layer, "dashq_sources", []
                    ):
                        if source in sources:
                            raise ValueError(f"DASH-Q module loaded twice: {source}")
                        sources.add(source)
                        for suffix, attr in (
                            ("W_q_packed", "qweight"),
                            ("scale", "scale"),
                            ("zero", "zero"),
                        ):
                            param = getattr(layer, prefix + attr)
                            quant_params.add(id(param))
                            dest = param if expert is None else param[expert]
                            load_matrix(
                                reader,
                                source,
                                metadata[source],
                                dest,
                                suffix,
                                slices,
                                offset,
                            )
                            consumed.add(f"{source}.{suffix}")
                if sources != metadata.keys():
                    raise ValueError(
                        "DASH-Q metadata/model coverage mismatch: "
                        f"{sorted(metadata.keys() - sources)[:8]}"
                    )

                def remaining():
                    for name in reader.index:
                        if name not in consumed:
                            consumed.add(name)
                            yield name, reader.tensor(name)

                loaded = model.load_weights(remaining())
                expected = {
                    name
                    for name, p in model.named_parameters()
                    if id(p) not in quant_params
                }
                missing = expected - loaded
                if missing:
                    raise ValueError(
                        f"Uninitialized DASH-Q model parameters: {missing}"
                    )
                if consumed != reader.index.keys():
                    raise ValueError("Unconsumed DASH-Q checkpoint tensors")
        finally:
            reader.close()
        total = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.info(
            "DASH-Q loaded %d modules, %d tensors; rank weights %.3f GiB; "
            "packed input staging <=64 MiB",
            len(sources),
            len(consumed),
            total / 2**30,
        )
