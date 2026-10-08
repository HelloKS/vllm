# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose Motif's weight layout before allocating routed expert parameters."""

import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class MotifRoutedExperts(RoutedExperts):
    def __init__(
        self,
        *args,
        hidden_clamp,
        polynorm_output_scale,
        polynorm_sigmoid_weight,
        use_identity_allreduce,
        **kwargs,
    ):
        self._motif_options = dict(
            hidden_clamp=hidden_clamp,
            polynorm_output_scale=polynorm_output_scale,
            polynorm_sigmoid_weight=polynorm_sigmoid_weight,
            use_identity_allreduce=use_identity_allreduce,
        )
        super().__init__(*args, **kwargs)
        from vllm.model_executor.models.motif_weight_utils import log_motif_load_memory

        log_motif_load_memory("allocated", self)

    def _get_quant_method(self, prefix, quant_config, moe_config):
        from vllm.model_executor.layers.fused_moe.motif_experts import MotifMoEMethod
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptBlockFp8Config,
            ModelOptMxFp8Config,
            ModelOptNvFp4DynamicConfig,
        )

        options = dict(self._motif_options)
        method_cls = MotifMoEMethod
        if isinstance(quant_config, ModelOptNvFp4DynamicConfig):
            from vllm.model_executor.layers.fused_moe.motif_nvfp4_experts import (
                MotifNvfp4MoEMethod,
            )

            method_cls = MotifNvfp4MoEMethod
            options["direct_load"] = quant_config.direct_load
        elif isinstance(quant_config, ModelOptBlockFp8Config):
            from vllm.model_executor.layers.fused_moe.motif_blockfp8_experts import (
                MotifBlockFp8MoEMethod,
            )

            method_cls = MotifBlockFp8MoEMethod
        elif isinstance(quant_config, ModelOptMxFp8Config):
            from vllm.model_executor.layers.fused_moe.motif_marlin_experts import (
                MotifMxfp8MarlinMoEMethod,
            )

            method_cls = MotifMxfp8MarlinMoEMethod
            options["serialized"] = quant_config.is_checkpoint_mxfp8_serialized
        elif quant_config is not None and quant_config.get_name() == "fp8":
            from vllm.model_executor.layers.fused_moe.motif_blockfp8_experts import (
                MotifBlockFp8MoEMethod,
            )

            if quant_config.weight_block_size != [128, 128]:
                raise ValueError("Motif FP8 requires weight_block_size=[128, 128]")
            method_cls = MotifBlockFp8MoEMethod
            options["serialized"] = quant_config.is_checkpoint_fp8_serialized
        elif quant_config is not None and quant_config.get_name() in (
            "auto_awq",
            "auto_gptq",
            "awq",
            "awq_marlin",
            "gptq",
            "gptq_marlin",
        ):
            from vllm.model_executor.layers.fused_moe.motif_marlin_experts import (
                MotifInt4MoEMethod,
            )

            method_cls = MotifInt4MoEMethod
            options["quant_config"] = quant_config
        elif quant_config is not None:
            raise NotImplementedError(
                f"Motif PolyNorm experts do not support {quant_config.get_name()}"
            )

        self.act_fn_weight = torch.nn.Parameter(
            torch.empty(self.local_num_experts, 3, dtype=torch.float32),
            requires_grad=False,
        )
        self.act_fn_bias = torch.nn.Parameter(
            torch.empty(self.local_num_experts, 1, dtype=torch.float32),
            requires_grad=False,
        )
        return method_cls(
            moe=moe_config,
            poly_norm_weight=self.act_fn_weight,
            poly_norm_bias=self.act_fn_bias,
            **options,
        )
