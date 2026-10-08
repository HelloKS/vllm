# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif block-FP8 storage with a portable Triton W8A8 execution path."""

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.motif_experts import (
    MotifTritonExperts,
    _MotifPolyNormMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import set_weight_attrs


class MotifBlockFp8MoEMethod(_MotifPolyNormMoEMethodBase):
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
        for name, n, k in (
            ("w13_weight", 2 * intermediate_size_per_partition, hidden_size),
            ("w2_weight", hidden_size, intermediate_size_per_partition),
        ):
            if n % 128 or k % 128:
                raise ValueError("Motif block FP8 requires dimensions divisible by 128")
            w = torch.nn.Parameter(
                torch.empty(num_experts, n, k, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            scale = torch.nn.Parameter(
                torch.zeros(num_experts, n // 128, k // 128, dtype=torch.float32),
                requires_grad=False,
            )
            set_weight_attrs(w, extra_weight_attrs)
            set_weight_attrs(
                scale,
                {
                    **extra_weight_attrs,
                    "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,
                },
            )
            layer.register_parameter(name, w)
            layer.register_parameter(name + "_scale", scale)

    def load_expert_tensor(self, layer, wname, weight):
        if self.serialized:
            if weight.dtype != torch.float8_e4m3fn:
                raise ValueError("Serialized block FP8 expects E4M3 expert weights")
            return False
        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Dynamic block FP8 expects BF16/FP16 expert weights")
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
            n, k = weight[ge].shape
            # 128-row chunks bound FP32 quantization temporaries independently of E.
            for row in range(0, n, 128):
                block = weight[ge, row : row + 128].to(
                    device=dst.device, dtype=torch.float32
                )
                block = block.reshape(128, k // 128, 128)
                scale = block.abs().amax(dim=(0, 2)).clamp_min(1e-12) / 448.0
                quant = (block / scale[None, :, None]).clamp(-448, 448)
                dst.data[le, row : row + 128].copy_(quant.reshape(128, k).to(dst.dtype))
                sf.data[le, row // 128].copy_(scale)
        return True

    def process_weights_after_loading(self, layer):
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name)
            if not bool((torch.isfinite(scale) & (scale > 0)).all()):
                raise ValueError(f"Missing or invalid Motif FP8 scales: {name}")
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self._init_motif_kernel(layer)

    def get_fused_moe_quant_config(self, layer):
        # Transport remains BF16; activation quantization is inside each GEMM.
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(shape=GroupShape(128, 128)),
            _a2=FusedMoEQuantDesc(shape=GroupShape(128, 128)),
            _w1=FusedMoEQuantDesc(
                dtype=torch.float8_e4m3fn,
                shape=GroupShape(128, 128),
                scale=layer.w13_weight_scale,
            ),
            _w2=FusedMoEQuantDesc(
                dtype=torch.float8_e4m3fn,
                shape=GroupShape(128, 128),
                scale=layer.w2_weight_scale,
            ),
        )

    def select_gemm_impl(self, prepare_finalize, layer):
        return MotifTritonExperts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            poly_norm_weight=self.poly_norm_weight,
            poly_norm_bias=self.poly_norm_bias,
            hidden_clamp=self.hidden_clamp,
            polynorm_output_scale=self.polynorm_output_scale,
            polynorm_sigmoid_weight=self.polynorm_sigmoid_weight,
        )
