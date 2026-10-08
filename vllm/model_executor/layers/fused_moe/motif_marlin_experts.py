# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin W4A16/W8A16 experts with Motif's PolyNorm activation."""

import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.motif_experts import (
    _MotifPolyNormExpertsBase,
    _MotifPolyNormMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
from vllm.model_executor.layers.quantization.auto_awq import (
    AutoAWQConfig,
    AutoAWQMoEMethod,
)
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod
from vllm.scalar_type import scalar_types


class MotifMarlinExperts(MarlinExperts):
    def __init__(self, *, motif_method, **kwargs):
        super().__init__(**kwargs)
        self.input_dtype = None
        for name in (
            "poly_norm_weight",
            "poly_norm_bias",
            "hidden_clamp",
            "polynorm_output_scale",
            "polynorm_sigmoid_weight",
        ):
            setattr(self, name, getattr(motif_method, name))
        self.eps = 1e-6

    def activation(self, activation, output, input, *, topk_ids=None, expert_map=None):
        if topk_ids is None:
            raise ValueError("Motif Marlin requires expert IDs for PolyNorm")
        gate, up = input.chunk(2, dim=-1)
        result = _MotifPolyNormExpertsBase._grouped_polynorm_activation(
            self,
            gate.contiguous(),
            up.contiguous(),
            topk_ids.to(torch.int32),
            expert_map.to(torch.int32) if expert_map is not None else None,
            topk_ids.shape[-1],
        )
        output.copy_(result)


class MotifInt4MoEMethod(_MotifPolyNormMoEMethodBase):
    def __init__(self, *, quant_config, **kwargs):
        if quant_config.weight_bits != 4 or quant_config.group_size not in (
            32,
            64,
            128,
        ):
            raise ValueError(
                "Motif INT4 supports group sizes 32, 64, 128 and 4-bit weights"
            )
        self.is_awq = isinstance(quant_config, AutoAWQConfig)
        if not self.is_awq and (quant_config.desc_act or not quant_config.is_sym):
            raise ValueError("Motif GPTQ requires symmetric INT4 with desc_act=False")
        if self.is_awq and not quant_config.zero_point:
            raise ValueError("Motif AWQ requires zero_point=True")
        if getattr(quant_config, "dynamic", None):
            raise ValueError("Motif INT4 does not support per-layer dynamic overrides")
        super().__init__(**kwargs)
        self.quant_config = quant_config
        self.wna16_moe_backend = WNA16MoEBackend.MARLIN
        self.quant_type = scalar_types.uint4 if self.is_awq else scalar_types.uint4b8
        self.input_dtype = None
        self.use_marlin = True

    @property
    def _storage_method(self):
        return AutoAWQMoEMethod if self.is_awq else AutoGPTQMoEMethod

    def create_weights(self, layer, *args, **kwargs):
        self._storage_method.create_weights(self, layer, *args, **kwargs)
        layer.w13_scales.data.zero_()
        layer.w2_scales.data.zero_()

    def process_weights_after_loading(self, layer):
        for name in ("w13_scales", "w2_scales"):
            for scale in getattr(layer, name):
                if not bool((torch.isfinite(scale) & (scale > 0)).all()):
                    raise ValueError(f"Missing or invalid Motif INT4 scales: {name}")
        self._storage_method.process_weights_after_loading(self, layer)

    def _setup_kernel(self, layer):
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self._init_motif_kernel(layer)

    def get_fused_moe_quant_config(self, layer):
        return self._storage_method.get_fused_moe_quant_config(self, layer)

    def select_gemm_impl(self, prepare_finalize, layer):
        return MotifMarlinExperts(
            motif_method=self, moe_config=self.moe, quant_config=self.moe_quant_config
        )


class MotifMxfp8MarlinMoEMethod(_MotifPolyNormMoEMethodBase):
    def __init__(self, *, serialized=False, **kwargs):
        super().__init__(**kwargs)
        self.serialized = serialized

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            FusedMoeWeightScaleSupported,
        )
        from vllm.model_executor.utils import set_weight_attrs

        for name, n, k in (
            ("w13_weight", 2 * intermediate_size_per_partition, hidden_size),
            ("w2_weight", hidden_size, intermediate_size_per_partition),
        ):
            if n % 128 or k % 128:
                raise ValueError(
                    "Motif MXFP8 Marlin requires dimensions divisible by 128"
                )
            w = torch.nn.Parameter(
                torch.empty(num_experts, n, k, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            sf = torch.nn.Parameter(
                torch.full((num_experts, n, k // 32), 255, dtype=torch.uint8),
                requires_grad=False,
            )
            set_weight_attrs(w, extra_weight_attrs)
            set_weight_attrs(
                sf,
                {
                    **extra_weight_attrs,
                    "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,
                },
            )
            layer.register_parameter(name, w)
            layer.register_parameter(name + "_scale", sf)
        layer.weight_block_size = [1, 32]

    def load_expert_tensor(self, layer, wname, weight):
        if self.serialized:
            if weight.dtype != torch.float8_e4m3fn:
                raise ValueError("Serialized MXFP8 requires E4M3 expert weights")
            return False
        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Dynamic MXFP8 requires BF16/FP16 expert weights")
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )

        dst = getattr(layer, wname)
        sf = getattr(layer, wname + "_scale")
        ids = (
            layer._expert_map.tolist()
            if layer._expert_map is not None
            else range(weight.shape[0])
        )
        if len(ids) != weight.shape[0]:
            raise ValueError("Checkpoint expert count does not match the expert map")
        for ge, le in enumerate(ids):
            if le < 0:
                continue
            for row in range(0, weight.shape[1], 128):
                chunk = weight[ge, row : row + 128].to(dst.device)
                q, scales = mxfp8_e4m3_quantize(chunk)
                dst.data[le, row : row + 128].copy_(q)
                sf.data[le, row : row + 128].copy_(scales)
        return True

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_mxfp8_moe_layer_for_marlin,
        )
        from vllm.model_executor.utils import replace_parameter

        w13, w2 = layer.w13_weight.data, layer.w2_weight.data
        s13, s2 = layer.w13_weight_scale.data, layer.w2_weight_scale.data
        for scales in (s13, s2):
            for expert in scales:
                if bool((expert == 255).any()):
                    raise ValueError("MXFP8 expert scales are missing or contain NaN")
        e, n, k = w13.shape
        inter = w2.shape[2]
        packed13 = w13.view(torch.int32).reshape(e, k // 16, n * 4)
        packed2 = w2.view(torch.int32).reshape(e, inter // 16, k * 4)
        # Repack expert-by-expert into the existing weight allocation.
        scale13 = torch.empty(
            (e, k // 32, n), dtype=layer.params_dtype, device=w13.device
        )
        scale2 = torch.empty(
            (e, inter // 32, k), dtype=layer.params_dtype, device=w2.device
        )
        for idx in range(e):
            a, b, sa, sb = prepare_mxfp8_moe_layer_for_marlin(
                layer,
                w13[idx : idx + 1],
                w2[idx : idx + 1],
                s13[idx : idx + 1],
                s2[idx : idx + 1],
            )
            packed13[idx].copy_(a[0])
            packed2[idx].copy_(b[0])
            scale13[idx].copy_(sa[0])
            scale2[idx].copy_(sb[0])
        replace_parameter(layer, "w13_weight", packed13)
        replace_parameter(layer, "w2_weight", packed2)
        replace_parameter(layer, "w13_weight_scale", scale13)
        replace_parameter(layer, "w2_weight_scale", scale2)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self._init_motif_kernel(layer)

    def get_fused_moe_quant_config(self, layer):
        from vllm.model_executor.layers.fused_moe.config import (
            fp8_w8a16_moe_quant_config,
        )

        return fp8_w8a16_moe_quant_config(
            layer.w13_weight_scale, layer.w2_weight_scale, block_shape=[1, 32]
        )

    def select_gemm_impl(self, prepare_finalize, layer):
        return MotifMarlinExperts(
            motif_method=self, moe_config=self.moe, quant_config=self.moe_quant_config
        )
