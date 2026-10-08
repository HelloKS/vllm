# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DASH-Q INT2/g32 for native NemotronH Linear and non-gated MoE layers."""

from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.dashq_ops import (
    dashq_linear,
    dashq_moe,
)
from vllm.transformers_utils.dashq import tp_slices, validate_metadata


def _allocate(layer, prefix, n, k, experts=None):
    if k % 32:
        raise ValueError("DASH-Q local K must divide group size 32")
    lead = () if experts is None else (experts,)
    for suffix, shape, dtype in (
        ("qweight", (*lead, k // 16, n), torch.int32),
        ("scale", (*lead, k // 32, n), torch.bfloat16),
        ("zero", (*lead, k // 32, n), torch.bfloat16),
    ):
        layer.register_parameter(
            prefix + suffix,
            nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False),
        )


class DashQConfig(QuantizationConfig):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.modules = validate_metadata(config)

    def get_name(self):
        return "dashq"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return ["dashq_config.json"]

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            return DashQMoEMethod(self, layer.moe_config, prefix)
        if isinstance(layer, LinearBase):
            sources = [prefix]
            if prefix.endswith(".qkv_proj"):
                sources = [prefix[:-8] + s + "_proj" for s in ("q", "k", "v")]
            present = [s in self.modules for s in sources]
            if any(present) and not all(present):
                raise ValueError(f"{prefix}: mixed quantized/unquantized QKV")
            if all(present):
                return DashQLinearMethod(self, sources)
            return UnquantizedLinearMethod()
        return None


class DashQLinearMethod(LinearMethodBase):
    def __init__(self, config, sources):
        self.config = config
        self.sources = sources

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("DASH-Q requires --dtype bfloat16")
        n, k = sum(output_partition_sizes), input_size_per_partition
        _allocate(layer, "dashq_", n, k)
        layer.dashq_sources = []
        out_offset = 0
        for source in self.sources:
            meta = self.config.modules[source]
            full_n, full_k = meta["out_features"], meta["in_features"]
            if isinstance(layer, ReplicatedLinear):
                kind = "replicated"
            elif isinstance(layer, RowParallelLinear):
                kind = "row"
            else:
                kind = "column"
            sizes = getattr(layer, "output_sizes", None)
            if len(self.sources) > 1:
                sizes = None
            slices = tp_slices(
                full_n, full_k, layer.tp_rank, layer.tp_size, kind, sizes
            )
            local_n = sum(s.n_stop - s.n_start for s in slices)
            if any(s.k_stop - s.k_start != k for s in slices):
                raise ValueError(f"{source}: checkpoint K does not match layer")
            layer.dashq_sources.append((source, "dashq_", None, slices, out_offset))
            out_offset += local_n
        if out_offset != n:
            raise ValueError("DASH-Q output partitions do not match layer")

    def apply(self, layer, x, bias=None):
        return dashq_linear(
            x, layer.dashq_qweight, layer.dashq_scale, layer.dashq_zero, bias
        )


class DashQMoEMethod(FusedMoEMethodBase):
    def __init__(self, config, moe, prefix):
        super().__init__(moe)
        self.config, self.prefix = config, prefix
        p = moe.moe_parallel_config
        if (
            p.use_ep
            or p.dp_size != 1
            or p.pcp_size != 1
            or p.sp_size != 1
            or p.tp_size not in (1, 2)
        ):
            raise ValueError("DASH-Q MoE supports TP=1/2, DP=PP=1, no EP/SP")
        if moe.activation != MoEActivation.RELU2_NO_MUL:
            raise ValueError("DASH-Q MoE requires non-gated ReLU squared")

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16 or layer.apply_router_weight_on_input:
            raise ValueError("DASH-Q requires BF16 and output routing weights")
        p = self.moe.moe_parallel_config
        h, i = hidden_size, intermediate_size_per_partition
        _allocate(layer, "w13_", i, h, num_experts)
        _allocate(layer, "w2_", h, i, num_experts)
        layer.dashq_sources = []
        for expert in range(num_experts):
            for projection, target, kind, shape in (
                ("up", "w13_", "column", (i * p.tp_size, h)),
                ("down", "w2_", "row", (h, i * p.tp_size)),
            ):
                source = f"{self.prefix}.{projection}_proj_list.{expert}"
                meta = self.config.modules.get(source)
                if meta is None or (meta["out_features"], meta["in_features"]) != shape:
                    raise ValueError(f"{source}: missing/incompatible expert metadata")
                slices = tp_slices(*shape, p.tp_rank, p.tp_size, kind)
                layer.dashq_sources.append((source, target, expert, slices, 0))

    def get_fused_moe_quant_config(self, layer):
        return None  # The custom expert kernel consumes its own packed buffers.

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        # The existing runner owns routing, shared experts, latent transforms,
        # routed scaling and TP reductions. This returns local routed output only.
        return dashq_moe(
            x,
            layer.w13_qweight,
            layer.w13_scale,
            layer.w13_zero,
            layer.w2_qweight,
            layer.w2_scale,
            layer.w2_zero,
            topk_weights,
            topk_ids,
        )
