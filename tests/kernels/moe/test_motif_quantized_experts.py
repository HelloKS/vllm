# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise packed loading, repacking and PolyNorm through both expert GEMMs."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("scheme", ["block_fp8", "mxfp8", "awq", "gptq"])
@torch.inference_mode()
def test_quantized_experts_preserve_polynorm_and_routing(scheme):
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.motif_blockfp8_experts import (
        MotifBlockFp8MoEMethod,
    )
    from vllm.model_executor.layers.fused_moe.motif_experts import MotifMoEMethod
    from vllm.model_executor.layers.fused_moe.motif_marlin_experts import (
        MotifInt4MoEMethod,
        MotifMxfp8MarlinMoEMethod,
    )
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    if scheme == "block_fp8" and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("Native FP8 tensor cores required")
    torch.manual_seed(13)
    e, h, inter, tokens, topk = 4, 256, 256, 17, 2
    dtype = torch.bfloat16
    config = make_dummy_moe_config(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=h,
        intermediate_size=inter,
        in_dtype=dtype,
    )
    options = dict(
        moe=config,
        poly_norm_weight=torch.randn(e, 3, device="cuda"),
        poly_norm_bias=torch.randn(e, 1, device="cuda") * 0.1,
        hidden_clamp=6.0,
        polynorm_output_scale=0.7,
        polynorm_sigmoid_weight=True,
        use_identity_allreduce=True,
    )
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(dtype):
        if scheme == "block_fp8":
            method = MotifBlockFp8MoEMethod(**options)
        elif scheme == "mxfp8":
            method = MotifMxfp8MarlinMoEMethod(**options)
        else:
            quant = (
                AutoAWQConfig(4, 32, True, False)
                if scheme == "awq"
                else AutoGPTQConfig(4, 32, False, True, False, {}, {})
            )
            method = MotifInt4MoEMethod(quant_config=quant, **options)
        layer = torch.nn.Module()
        layer.params_dtype = dtype
        layer.intermediate_size_per_partition = inter
        layer._loaded_expert_biases = set()
        layer._expert_map = None
        layer._expert_routing_tables = lambda: None
        with torch.device("cuda"):
            method.create_weights(layer, e, h, inter, dtype)
        w13 = torch.randn(e, 2 * inter, h, device="cuda", dtype=dtype) * 0.03
        w2 = torch.randn(e, h, inter, device="cuda", dtype=dtype) * 0.03
        if scheme in ("block_fp8", "mxfp8"):
            method.load_expert_tensor(layer, "w13_weight", w13)
            method.load_expert_tensor(layer, "w2_weight", w2)
        else:
            # Constant nibbles are independent of AWQ/GPTQ packing permutations.
            # AWQ: (1 - 0) * scale. Symmetric GPTQ: (0 - 8) * scale.
            for prefix, ref in (("w13", w13), ("w2", w2)):
                getattr(layer, prefix + "_qweight").fill_(
                    0x11111111 if scheme == "awq" else 0
                )
                scales = getattr(layer, prefix + "_scales")
                for idx in range(e):
                    scales[idx].fill_(0.01 * (idx + 1))
                    ref[idx].fill_(
                        float(scales[idx, 0, 0]) * (1 if scheme == "awq" else -8)
                    )
                zeros = getattr(layer, prefix + "_qzeros", None)
                if zeros is not None:
                    zeros.zero_()
        method.process_weights_after_loading(layer)
        reference_layer = torch.nn.Module()
        reference_layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        reference_layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        reference_layer._expert_routing_tables = lambda: None
        reference = MotifMoEMethod(**options)
        reference.process_weights_after_loading(reference_layer)

        x = torch.randn(tokens, h, device="cuda", dtype=dtype)
        ids = torch.stack((torch.arange(tokens) % e, (torch.arange(tokens) + 1) % e), 1)
        ids = ids.to(device="cuda", dtype=torch.int32)
        weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
        weights /= weights.sum(1, keepdim=True)

        def run(m, loaded):
            expert = m.moe_kernel.fused_experts
            _, _, n, _, _ = expert.moe_problem_size(
                x, loaded.w13_weight, loaded.w2_weight, ids
            )
            shapes = expert.workspace_shapes(
                tokens, n, h, topk, e, e, None, MoEActivation.SILU
            )
            a, b, output = [torch.empty(s, device="cuda", dtype=dtype) for s in shapes]
            expert.apply(
                output,
                x,
                loaded.w13_weight,
                loaded.w2_weight,
                weights,
                ids,
                MoEActivation.SILU,
                e,
                None,
                None,
                None,
                a,
                b,
                None,
                False,
            )
            return output.float()

        actual, expected = run(method, layer), run(reference, reference_layer)
        assert torch.isfinite(actual).all()
        relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-6)
        assert relative_error < (0.12 if "fp8" in scheme else 0.03)


@pytest.mark.parametrize("scheme", ["block_fp8", "mxfp8"])
@torch.inference_mode()
def test_dynamic_fp8_only_loads_rank_local_experts(scheme):
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.motif_blockfp8_experts import (
        MotifBlockFp8MoEMethod,
    )
    from vllm.model_executor.layers.fused_moe.motif_marlin_experts import (
        MotifMxfp8MarlinMoEMethod,
    )

    cls = MotifBlockFp8MoEMethod if scheme == "block_fp8" else MotifMxfp8MarlinMoEMethod
    with set_current_vllm_config(VllmConfig()):
        method = cls(
            moe=make_dummy_moe_config(
                num_experts=4,
                num_local_experts=2,
                hidden_dim=256,
                intermediate_size=128,
            ),
            poly_norm_weight=torch.ones(2, 3),
            poly_norm_bias=torch.zeros(2, 1),
            hidden_clamp=None,
            polynorm_output_scale=1.0,
            polynorm_sigmoid_weight=False,
        )
        layer = torch.nn.Module()
        layer._expert_map = torch.tensor([-1, 1, -1, 0])
        with torch.device("cuda"):
            method.create_weights(layer, 2, 256, 128, torch.bfloat16)
        source = torch.randn(4, 256, 256, dtype=torch.bfloat16)
        # Nonlocal experts must never be quantized or copied into local slots.
        source[0].fill_(torch.nan)
        source[2].fill_(torch.nan)
        method.load_expert_tensor(layer, "w13_weight", source)
        scale = layer.w13_weight_scale
        if scheme == "block_fp8":
            restored = layer.w13_weight.float() * scale.repeat_interleave(
                128, 1
            ).repeat_interleave(128, 2)
        else:
            restored = layer.w13_weight.float() * torch.exp2(
                scale.float() - 127
            ).repeat_interleave(32, -1)
        expected = source[[3, 1]].to("cuda").float()
        assert (restored - expected).norm() / expected.norm() < 0.05
