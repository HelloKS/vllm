# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif MoE expert kernel for NVFP4 (dynamic bf16->NVFP4 or direct load).

Mirrors ``motif_mxfp8_experts.py`` structurally, but targets the CUTLASS
NVFP4 grouped GEMM (``cutlass_fp4_moe_mm``): E2M1 values packed two-per-byte,
1x16 E4M3 blockscales (swizzled 128x4 layout) and a per-expert fp32 global
scale folded into the GEMM epilogue alpha.

Two loading modes, selected by ``MotifNvfp4MoEMethod(direct_load=...)``:

* **Dynamic** (bf16 checkpoint): the upstream unquantized method allocates
  bf16 expert weights and ``process_weights_after_loading`` quantizes them
  in place via ``_quantize_experts_to_nvfp4``.
* **Direct** (pre-quantized checkpoint from
  ``tools/motif_nvfp4_quantize_ckpt.py``): ``convert_layer_for_direct_load``
  re-registers the expert parameters as packed uint8 weights + linear E4M3
  blockscales + per-expert fp32 ``weight_scale_2``; the checkpoint tensors are
  loaded straight into them and ``process_weights_after_loading`` only
  swizzles the blockscales and folds the epilogue alphas. Both modes produce
  bit-identical runtime tensors for the same source weights (the offline tool
  uses the same quantization recipe).

**Activation global scale (a_gscale).** Uncalibrated, it defaults to 1.0: the
E4M3 blockscale then stores ``block_amax / 6`` directly (safe up to
``448 * 6 = 2688``, above motif's ``hidden_clamp``), but small-magnitude blocks
(``amax < ~0.09``) push the blockscale into E4M3 subnormals and lose precision.
A calibration pass (see ``dump_nvfp4_act_scales`` and
``examples/offline_inference/motif_nvfp4_calibrate.py``) measures a per-expert
activation amax and writes a sidecar; on the next load the a_gscale becomes
``448 * 6 / amax`` (per GEMM input), lifting those blockscales back into the
E4M3 normal range.

**Epilogue alpha fold (critical).** The CUTLASS FP4 mm's only channel to undo a
global scale is the epilogue alpha, so the alpha MUST be
``1 / (a_gscale * w_gscale)``. With a_gscale == 1 this reduces to the pure
weight term ``1 / w_gscale = amax_w / 2688``; with a calibrated a_gscale the
weight alpha is divided by a_gscale at load (and, in RL, at every refit).

Flow mirrors ``MotifMxfp8Experts.apply``: moe_permute → scaled_fp4_experts_quant
→ cutlass_fp4_moe_mm (GEMM1) → grouped PolyNorm (permuted layout, fake
top_k=1 via searchsorted) → scaled_fp4_experts_quant → cutlass_fp4_moe_mm
(GEMM2) → moe_unpermute (folds topk weights + reduce).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.motif_experts import (
    _MotifPolyNormExpertsBase,
    _MotifPolyNormMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

logger = init_logger(__name__)

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
FLOAT4_E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16
# global_scale that lands the largest block's E4M3 scale at 448 (full range).
_SF_FULL_RANGE = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX  # 2688.0

# Calibration / sidecar plumbing.
_CALIB_ENV = "VLLM_MOTIF_NVFP4_CALIBRATE"  # "1" -> accumulate act amax in apply
_CALIB_DIAG_ENV = (
    "VLLM_MOTIF_NVFP4_CALIB_DIAG"  # "1" -> also collect a2 channel/row stats
)
_FUSED_ENV = (
    "VLLM_MOTIF_NVFP4_FUSED"  # default on; "0" -> unfused routing/PolyNorm (see apply)
)
_ACT_SCALES_ENV = "VLLM_MOTIF_NVFP4_ACT_SCALES"  # explicit sidecar path override
_ACT_SCALES_FILENAME = "nvfp4_act_scales.safetensors"  # default name in model dir

# Loaded sidecar cache: path -> {f"{layer}.a13_gscale": tensor, ...} (or None).
_act_scales_cache: dict[str, dict | None] = {}


def _calibration_enabled() -> bool:
    return os.environ.get(_CALIB_ENV, "0") == "1"


def _calib_diag_enabled() -> bool:
    """a2 채널성/토큰성 진단 수집 (캘리브 모드에서만 유효)."""
    return _calibration_enabled() and os.environ.get(_CALIB_DIAG_ENV, "0") == "1"


def _resolve_act_scales_path() -> str | None:
    """Sidecar path: explicit env override, else ``<model_dir>/<filename>``."""
    env_path = os.environ.get(_ACT_SCALES_ENV)
    if env_path:
        return env_path if os.path.isfile(env_path) else None
    try:
        from vllm.config import get_current_vllm_config

        model = get_current_vllm_config().model_config.model
    except Exception:
        return None
    if not model or not os.path.isdir(model):
        return None
    cand = os.path.join(model, _ACT_SCALES_FILENAME)
    return cand if os.path.isfile(cand) else None


def _get_act_scales() -> dict | None:
    """Load (and cache) the flat sidecar dict, or None if absent/unreadable."""
    path = _resolve_act_scales_path()
    if path is None:
        return None
    if path not in _act_scales_cache:
        try:
            from safetensors.torch import load_file

            _act_scales_cache[path] = load_file(path)
            logger.info("Loaded NVFP4 activation scales from %s", path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load NVFP4 act scales %s: %s", path, e)
            _act_scales_cache[path] = None
    return _act_scales_cache[path]


def _allocate_nvfp4_activation(
    rows: int, hidden_size: int, topk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed-value + swizzled-scale buffers for one fused quantization.

    The scale buffer is sized for the worst case rather than for ``rows``: the
    per-expert scale slices start on 128-row boundaries, so the swizzled layout
    is addressed by expert offset, not by the actual row count. A static shape
    also keeps the allocation CUDA-graph friendly.
    """
    max_rows = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk
    assert rows <= max_rows
    values = torch.empty((rows, hidden_size // 2), device=device, dtype=torch.uint8)
    scales = torch.empty(
        (max_rows, (hidden_size // 16 + 3) // 4),
        device=device,
        dtype=torch.int32,
    )
    return values, scales


def _quantize_permuted_nvfp4(
    hidden_states: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    permuted_idx: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather routed rows and quantize them to NVFP4 in one pass.

    Replaces ``moe_permute``'s bf16 scratch buffer plus a separate
    ``scaled_fp4_experts_quant``: the permuted activations never materialize.
    """
    output, output_scales = _allocate_nvfp4_activation(
        permuted_idx.numel(), hidden_states.shape[1], topk, hidden_states.device
    )
    ops.scaled_fp4_experts_quant_permuted(
        output,
        output_scales,
        hidden_states,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
        permuted_idx,
        inv_permuted_idx,
        topk,
    )
    return output, output_scales.view(torch.float8_e4m3fn)


def _make_route_maps(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routing metadata only — no permuted activation buffer is produced.

    ``use_small_sort`` picks the bounded stable counting sort for decode-sized
    routing; larger batches fall back to the general path, which needs the
    explicit source-row table.
    """
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    num_rows = num_tokens * topk
    assert (hidden_size * hidden_states.element_size()) % 16 == 0

    use_small_sort = num_rows <= 1024 and global_num_experts <= 512
    source_rows = (
        torch.empty(0, dtype=torch.int32, device=hidden_states.device)
        if use_small_sort
        else torch.arange(
            num_rows, dtype=torch.int32, device=hidden_states.device
        ).reshape(num_tokens, topk)
    )
    expert_offsets = torch.empty(
        local_num_experts + 1, dtype=torch.int64, device=hidden_states.device
    )
    # ``skip_input_permute`` bypasses the kernel that fills ``inv_permuted_idx``;
    # the first quantizer writes it instead, but only for the rows it actually
    # processes. Under EP the (token, k) pairs whose expert lives on another
    # rank are never processed, so their entries must already hold a skip
    # sentinel — ``moe_unpermute`` drops entries at or past the last local
    # expert offset. Leaving them uninitialized makes those tokens accumulate
    # an arbitrary other row's expert output: the text stays plausible but the
    # logits shift, which showed up as MTP acceptance collapsing from ~92% to
    # 7% under load. ``num_rows`` is always >= that offset, so it is a safe
    # sentinel, and it matches what the unfused wrapper pre-fills.
    inv_permuted_idx = torch.full(
        (num_tokens, topk),
        num_rows,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    permuted_idx = torch.full(
        (num_rows,), num_rows, dtype=torch.int32, device=hidden_states.device
    )
    ops.moe_permute(
        hidden_states,
        topk_ids.to(torch.int32),
        source_rows,
        expert_map,
        global_num_experts,
        local_num_experts,
        topk,
        torch.empty(0, dtype=hidden_states.dtype, device=hidden_states.device),
        expert_offsets,
        inv_permuted_idx,
        permuted_idx,
        True,
    )
    return expert_offsets, inv_permuted_idx.flatten(), permuted_idx


def _gscale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Per-expert global scale that maps the calibrated amax to E4M3 max (448).

    Experts with no calibration signal (amax ~ 0: never routed during the
    pass) fall back to the uncalibrated default gscale of 1.0 — the naive
    ``2688 / eps`` would zero that expert's activations at serving. NaNs are
    sanitized the same way (a NaN would otherwise be silently masked to 0 by
    the cross-rank MAX all-reduce, which uses fmaxf semantics).
    """
    amax = torch.nan_to_num(amax.to(torch.float32), nan=0.0)
    gscale = _SF_FULL_RANGE / amax.clamp(min=1e-8)
    return torch.where(amax > 1e-6, gscale, torch.ones_like(gscale))


def _local_gscales_for_layer(
    layer_name: str,
    e_local: int,
    expert_map: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Slice this rank's local per-expert (a13, a2) global scales from the
    per-GLOBAL sidecar, or None if the layer is absent. ``expert_map`` maps
    global->local (-1 == not on this rank), matching weight loading.
    """
    scales = _get_act_scales()
    if scales is None:
        return None
    key13 = f"{layer_name}.a13_gscale"
    key2 = f"{layer_name}.a2_gscale"
    if key13 not in scales or key2 not in scales:
        return None
    g13 = scales[key13].to(device=device, dtype=torch.float32)
    g2 = scales[key2].to(device=device, dtype=torch.float32)
    if expert_map is None:
        # Single-rank / replicated: global == local.
        return g13[:e_local].contiguous(), g2[:e_local].contiguous()
    em = expert_map.to(device)
    global_idx = (em >= 0).nonzero(as_tuple=True)[0]
    local_idx = em[global_idx]
    out13 = torch.ones(e_local, dtype=torch.float32, device=device)
    out2 = torch.ones(e_local, dtype=torch.float32, device=device)
    out13[local_idx] = g13[global_idx]
    out2[local_idx] = g2[global_idx]
    return out13, out2


def _quantize_experts_to_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE weight ``[E, X, K]`` (bf16) to NVFP4.

    Returns:
      * packed E2M1 values ``[E, X, K // 2]`` (uint8, two fp4 per byte),
      * swizzled E4M3 blockscales ``[E, X, K // 16]`` (128x4 tiled layout
        expected by ``cutlass_fp4_moe_mm``; X and K // 16 are already
        128- / 4-aligned for motif dims so no padding occurs),
      * per-expert epilogue alphas ``[E]`` fp32 (``amax / (448 * 6)``, the
        weight dequant multiplier; the activation side contributes 1.0).

    Runs once at load time, so a simple per-expert loop over
    ``ops.scaled_fp4_quant`` is fine.

    """
    assert weight.dim() == 3, f"weight must be 3D, got {weight.shape}"
    E, X, K = weight.shape
    assert K % NVFP4_BLOCK_SIZE == 0, f"K={K} % {NVFP4_BLOCK_SIZE} != 0"
    assert X % 128 == 0, f"X={X} must be 128-aligned for the NVFP4 sf layout"
    assert (K // NVFP4_BLOCK_SIZE) % 4 == 0, (
        f"K={K} must give 4-aligned scale columns for the NVFP4 sf layout"
    )

    weight_fp4 = torch.empty(E, X, K // 2, dtype=torch.uint8, device=weight.device)
    weight_scale = torch.empty(
        E,
        X,
        K // NVFP4_BLOCK_SIZE,
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )
    alphas = torch.empty(E, dtype=torch.float32, device=weight.device)
    for e in range(E):
        amax = weight[e].abs().amax().float().clamp(min=1e-12)
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
        q, sf = ops.scaled_fp4_quant(
            weight[e], global_scale, is_sf_swizzled_layout=True
        )
        weight_fp4[e].copy_(q)
        weight_scale[e].copy_(sf.reshape(X, K // NVFP4_BLOCK_SIZE))
        alphas[e] = 1.0 / global_scale
    return weight_fp4, weight_scale, alphas


def _encode_e2m1(x: torch.Tensor) -> torch.Tensor:
    """fp32 values in [-6, 6] -> E2M1 4-bit codes (sign << 3 | magnitude).

    Magnitude buckets mirror ``cast_to_fp4`` (see also the fakequant tool):
    the E2M1 value grid is [0, 0.5, 1, 1.5, 2, 3, 4, 6] with ties at exact
    midpoints resolved by the boundary conditions below. ``signbit`` (not
    ``< 0``) keeps -0.0 inputs on the negative branch.
    """
    a = x.abs()
    code = (
        (a > 0.25).to(torch.uint8)
        + (a >= 0.75).to(torch.uint8)
        + (a > 1.25).to(torch.uint8)
        + (a >= 1.75).to(torch.uint8)
        + (a > 2.5).to(torch.uint8)
        + (a >= 3.5).to(torch.uint8)
        + (a > 5.0).to(torch.uint8)
    )
    return code | (torch.signbit(x).to(torch.uint8) << 3)


def quantize_experts_to_nvfp4_kernel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA-kernel NVFP4 quantizer for stacked MoE weights ``[E, X, K]``.

    Same per-expert loop over ``ops.scaled_fp4_quant`` as
    ``_quantize_experts_to_nvfp4`` but returns **linear-layout** blockscales
    and ``weight_scale_2`` instead of swizzled scales and alphas — the
    serialization layout of ``tools/motif_nvfp4_quantize_ckpt.py``. Because
    the packing kernel is the same, a checkpoint produced by this function
    reproduces the dynamic path bit-for-bit after the load-time swizzle.
    Requires a CUDA device with FP4 support.
    """
    assert weight.dim() == 3, f"weight must be 3D, got {tuple(weight.shape)}"
    E, _, K = weight.shape
    assert K % NVFP4_BLOCK_SIZE == 0, f"K={K} % {NVFP4_BLOCK_SIZE} != 0"

    amax = weight.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
    global_scales = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax  # [E]

    q_list, sf_list = [], []
    for e in range(E):
        q, sf = ops.scaled_fp4_quant(
            weight[e], global_scales[e], is_sf_swizzled_layout=False
        )
        q_list.append(q)
        sf_list.append(sf)
    return (
        torch.stack(q_list),
        torch.stack(sf_list),
        (1.0 / global_scales).to(torch.float32),
    )


def quantize_experts_to_nvfp4_ref(
    weight: torch.Tensor, chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch NVFP4 quantizer for stacked MoE weights ``[E, X, K]``.

    Same recipe as ``_quantize_experts_to_nvfp4`` (per-expert global scale
    ``448*6 / amax``, 1x16 E4M3 blockscales along K, E2M1 values) but without
    the CUDA kernel, so it runs on any device. NOT guaranteed bit-identical
    to the kernel: E2M1 rounding at exact bucket boundaries can differ by one
    grid step (dequant-level difference is bounded by one E2M1 ulp times the
    blockscale). Prefer ``quantize_experts_to_nvfp4_kernel`` on CUDA — the
    offline tool does. This is what
    ``tools/motif_nvfp4_quantize_ckpt.py`` serializes. Returns:

      * packed E2M1 values ``[E, X, K // 2]`` (uint8, low nibble = element 2i),
      * **linear-layout** E4M3 blockscales ``[E, X, K // 16]`` (the loader
        swizzles after TP/EP sharding),
      * per-expert ``weight_scale_2`` ``[E]`` fp32 (``amax / (448 * 6)``, the
        pure weight epilogue-alpha term — identical to the dynamic path's
        ``alphas``).

    Chunked over experts to bound the fp32 working set.
    """
    assert weight.dim() == 3, f"weight must be 3D, got {tuple(weight.shape)}"
    E, X, K = weight.shape
    assert K % NVFP4_BLOCK_SIZE == 0, f"K={K} % {NVFP4_BLOCK_SIZE} != 0"

    packed = torch.empty(E, X, K // 2, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(
        E,
        X,
        K // NVFP4_BLOCK_SIZE,
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )
    scale_2 = torch.empty(E, dtype=torch.float32, device=weight.device)

    for s in range(0, E, chunk):
        sl = slice(s, min(s + chunk, E))
        x = weight[sl].to(torch.float32)
        n = x.shape[0]
        amax = x.abs().amax(dim=(1, 2)).clamp(min=1e-12)  # [n]
        gscale = _SF_FULL_RANGE / amax  # [n]

        xb = x.reshape(n, X, K // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
        vmax = xb.abs().amax(dim=-1)  # [n, X, K/16]
        sf = (
            (gscale.view(n, 1, 1) * (vmax / FLOAT4_E2M1_MAX))
            .clamp(-FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        # Per-block dequant multiplier at the E4M3-rounded scale.
        deq = sf.to(torch.float32) / gscale.view(n, 1, 1)
        q = (xb / deq.clamp_min(1e-20).unsqueeze(-1)).clamp(
            -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX
        )
        codes = _encode_e2m1(q)  # [n, X, K/16, 16]
        codes = codes.reshape(n, X, K)
        packed[sl] = codes[..., 0::2] | (codes[..., 1::2] << 4)
        scales[sl] = sf
        scale_2[sl] = 1.0 / gscale

    return packed, scales, scale_2


class MotifNvfp4Experts(_MotifPolyNormExpertsBase):
    """Motif MoE experts using CUTLASS NVFP4 grouped MM.

    Per-expert weight layout (after process_weights_after_loading):
      ``w1`` (gate||up): ``[E_local, 2*I, K/2]`` uint8 + swizzled e4m3 scales
      ``w2`` (down):     ``[E_local, K, I/2]``   uint8 + swizzled e4m3 scales

    Unlike ``CutlassExpertsFp4`` (which requires ep_size == 1 and permutes via
    ``get_cutlass_moe_mm_data``), this uses ``moe_permute``/``moe_unpermute``
    so the TP-sharded (identity-allreduce) and EP paths with ``expert_map``
    keep working — same approach as ``MotifMxfp8Experts``.

    When ``calib_amax`` is set (a ``(a13_amax, a2_amax)`` pair of ``[E_local]``
    buffers owned by the layer), ``apply`` also accumulates a per-expert running
    amax of the two GEMM inputs — the statistic ``dump_nvfp4_act_scales`` turns
    into activation global scales. The fp4 forward still runs normally (at the
    current a_gscale), so calibration output stays valid.
    """

    def __init__(self, *args, calib_amax=None, **kwargs):
        super().__init__(*args, **kwargs)
        # (a13_amax, a2_amax) [E_local] buffers, or None when not calibrating.
        self._calib_amax = calib_amax

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(120)
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # N = 2*I (w1's logical output dim; fp4 packing is on the K dim).
        activation_out_dim = N // 2
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(activation_out_dim, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert hidden_states.is_contiguous()
        assert hidden_states.dim() == 2
        assert hidden_states.dtype in (torch.float16, torch.bfloat16)
        assert not apply_router_weight_on_input, (
            "MotifNvfp4Experts does not support apply_router_weight_on_input"
        )
        assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8

        # The fused kernels never materialize the two bf16 intermediates the
        # amax accumulators read, so calibration always runs the unfused path.
        # It is an offline pass over a few hundred prompts, so the cost is moot.
        #
        # Serving defaults to fused. The EP divergence that previously forced
        # unfused (MTP acceptance collapsing to ~1% under dp8 + expert
        # parallel, while the generated text still looked clean) was two
        # defects in how EP-dropped rows were handled, both fixed here:
        #   - inv_permuted_idx was left uninitialized when skip_input_permute
        #     bypassed the kernel that writes it, so dropped (token, k) entries
        #     kept garbage and moe_unpermute scattered to the wrong rows;
        #   - the experts-quant binary search fell through to expert 0 row 0
        #     for rows past the last local expert offset, so every dropped row
        #     overwrote that block's scales.
        # Unpermute output is now bit-identical to the unfused path on both the
        # small-sort and general routes, acceptance holds at 96-100% under
        # load, and AA-Omniscience metric is 0.4996 vs 0.4857 unfused.
        # Opt out with VLLM_MOTIF_NVFP4_FUSED=0.
        if (
            current_platform.is_device_capability_family(100)
            and self._calib_amax is None
            and os.environ.get(_FUSED_ENV, "1").lower()
            in (
                "1",
                "true",
            )
        ):
            self._apply_fused(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                workspace13,
                workspace2,
            )
        else:
            self._apply_unfused(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                workspace13,
                workspace2,
            )

    def _apply_fused(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> None:
        """Serving path: gather+quantize and PolyNorm+quantize are fused.

        Neither the permuted bf16 activations nor the bf16 PolyNorm output is
        written to memory; the routing metadata, per-expert offsets and
        block-scale offsets are produced by the launches that already need
        them. The fused PolyNorm kernel skips rows past the last local expert
        offset, so EP-dropped tail rows cost nothing.
        """
        logger.info_once("NVFP4 CUTLASS fp4 expert path active, fused (first forward)")
        E_local, two_intermediate_size, packed_K = w1.shape
        intermediate_size = two_intermediate_size // 2
        K = packed_K * 2
        device = hidden_states.device
        topk = topk_ids.size(1)
        M_total = hidden_states.size(0) * topk

        if global_num_experts == -1:
            global_num_experts = E_local

        expert_first_token_offset, inv_permuted_idx, permuted_idx = _make_route_maps(
            hidden_states,
            topk_ids,
            global_num_experts,
            E_local,
            expert_map,
        )

        problem_sizes1 = torch.empty((E_local, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((E_local, 3), dtype=torch.int32, device=device)
        expert_offsets = torch.empty((E_local + 1,), dtype=torch.int32, device=device)
        blockscale_offsets = torch.empty(
            (E_local + 1,), dtype=torch.int32, device=device
        )
        ops.get_cutlass_moe_mm_problem_sizes_and_nvfp4_offsets(
            expert_first_token_offset,
            problem_sizes1,
            problem_sizes2,
            expert_offsets,
            blockscale_offsets,
            intermediate_size,
            K,
        )

        a_perm_fp4, a_perm_scale = _quantize_permuted_nvfp4(
            hidden_states,
            self.a1_gscale,
            expert_offsets,
            blockscale_offsets,
            permuted_idx,
            inv_permuted_idx,
            topk,
        )
        gemm1_out = _resize_cache(workspace13, (M_total, two_intermediate_size))
        ops.cutlass_fp4_moe_mm(
            gemm1_out,
            a_perm_fp4,
            w1,
            a_perm_scale,
            self.w1_scale,
            self.g1_alphas,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
        del a_perm_fp4, a_perm_scale

        gate, up = gemm1_out.split(intermediate_size, dim=-1)
        act_fp4, act_scale = self._fused_polynorm_nvfp4(
            gate,
            up,
            expert_first_token_offset,
            blockscale_offsets,
            topk,
        )

        gemm2_out = _resize_cache(workspace2, (M_total, K))
        ops.cutlass_fp4_moe_mm(
            gemm2_out,
            act_fp4,
            w2,
            act_scale,
            self.w2_scale,
            self.g2_alphas,
            problem_sizes2,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
        del act_fp4, act_scale

        moe_unpermute(
            out=output,
            permuted_hidden_states=gemm2_out,
            topk_weights=topk_weights,
            inv_permuted_idx=inv_permuted_idx,
            expert_first_token_offset=expert_first_token_offset,
        )

    def _fused_polynorm_nvfp4(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        expert_first_token_offset: torch.Tensor,
        blockscale_offsets: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PolyNorm + GEMM2-input quantization in a single pass.

        Host-side parameter preparation mirrors
        ``_grouped_polynorm_activation``: the kernel takes fp32 weights and
        expects the sigmoid already applied, and folds the output clamp and
        output scale that the unfused path applies after the kernel returns.
        """
        weight = self.poly_norm_weight.float()
        if self.polynorm_sigmoid_weight:
            weight = torch.sigmoid(weight)
        bias = self.poly_norm_bias.float()
        hc = float(self.hidden_clamp) if self.hidden_clamp is not None else -1.0
        out, out_scales = _allocate_nvfp4_activation(
            gate.shape[0], gate.shape[1], topk, gate.device
        )
        ops.grouped_poly_norm_nvfp4_quant(
            out,
            out_scales,
            gate,
            up,
            weight,
            bias,
            expert_first_token_offset,
            blockscale_offsets,
            self.a2_gscale,
            self.eps,
            hc,
            self.polynorm_output_scale,
        )
        return out, out_scales.view(torch.float8_e4m3fn)

    def _apply_unfused(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> None:
        """Calibration path: keeps the bf16 GEMM inputs the amax hooks read."""
        logger.info_once(
            "NVFP4 CUTLASS fp4 expert path active, unfused/calibration (first forward)"
        )
        E_local, two_intermediate_size, _ = w1.shape
        intermediate_size = two_intermediate_size // 2
        K = hidden_states.size(1)
        device = hidden_states.device
        M = hidden_states.size(0)
        topk = topk_ids.size(1)
        M_total = M * topk

        if global_num_experts == -1:
            global_num_experts = E_local

        # a_perm uses a fresh buffer (workspace2 is reserved for gemm2_out;
        # mm2_out must NOT alias output, which shares memory with workspace13
        # in the non-chunked path).
        permuted_buf = torch.empty(
            (M_total, K), dtype=hidden_states.dtype, device=device
        )
        a_perm, _, expert_first_token_offset, inv_permuted_idx, _ = moe_permute(
            hidden_states,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=global_num_experts,
            n_local_expert=E_local,
            expert_map=expert_map,
            permuted_hidden_states=permuted_buf,
        )

        problem_sizes1 = torch.empty((E_local, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((E_local, 3), dtype=torch.int32, device=device)
        # ps1 = [m_e, 2*I, K], ps2 = [m_e, K, I] — the same convention
        # get_cutlass_moe_mm_data produces for the FP4 grouped GEMM
        # (swap_ab must be False on the FP4 path).
        ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
            expert_first_token_offset,
            problem_sizes1,
            problem_sizes2,
            intermediate_size,
            K,
            False,
        )
        # scaled_fp4_experts_quant consumes the full (E+1) offset arrays;
        # cutlass_fp4_moe_mm takes the leading E entries.
        expert_offsets = expert_first_token_offset.to(torch.int32)
        # Per-expert NVFP4 sf slice is align(m_e, 128) rows (the swizzled
        # layout's tile height); offsets are the cumulative aligned counts.
        # Same CPU-sync-free construction as the MXFP8 path.
        counts = (expert_first_token_offset[1:] - expert_first_token_offset[:-1]).to(
            torch.int32
        )
        aligned = ((counts + 127) // 128) * 128
        blockscale_offsets = torch.zeros(
            (E_local + 1,), dtype=torch.int32, device=device
        )
        blockscale_offsets[1:] = aligned.cumsum(0).to(torch.int32)

        # Quant-view offsets with a SENTINEL expert appended: the fp4 experts
        # quant kernel is row-driven and its expert-interval scan has no miss
        # guard (nvfp4_experts_quant.cu) — EP-dropped tail rows past
        # expert_first_token_offset[-1] fall through with expert_idx=0 /
        # rowIdx_in_expert=0 and race-clobber expert 0 row 0's scale factors
        # with values quantized from uninitialized memory. Mapping the tail to
        # a sentinel expert routes those writes into a scratch sf region past
        # the real experts' slices. Offsets are padded to a multiple of 16
        # intervals with empty [M_total, M_total) plateaus so the kernel's
        # 16-wide chunked offset loads never read past the tensor. The GEMMs
        # keep the real E-entry offsets/problem sizes, so results for valid
        # rows are bit-identical.
        E_pad = ((E_local + 1) + 15) // 16 * 16
        expert_offsets_q = torch.full(
            (E_pad + 1,), M_total, dtype=torch.int32, device=device
        )
        expert_offsets_q[: E_local + 1] = expert_offsets
        blockscale_offsets_q = torch.empty(
            (E_pad + 1,), dtype=torch.int32, device=device
        )
        blockscale_offsets_q[: E_local + 1] = blockscale_offsets
        tail_scratch_end = (
            blockscale_offsets[E_local]
            + ((M_total - expert_offsets[E_local] + 127) // 128) * 128
        )
        blockscale_offsets_q[E_local + 1 :] = tail_scratch_end
        # The kernel may index the per-expert gscale with the sentinel id;
        # keep a padded copy so that read stays in-bounds (values unused —
        # sentinel sf writes land in the scratch region).
        gq = getattr(self, "_gscale_q_pad", None)
        if gq is None or gq[0].numel() != E_pad:
            pad = E_pad - E_local
            gq = (
                torch.cat([self.a1_gscale, self.a1_gscale.new_ones(pad)]),
                torch.cat([self.a2_gscale, self.a2_gscale.new_ones(pad)]),
            )
            self._gscale_q_pad = gq
        a1_gscale_q, a2_gscale_q = gq

        # Calibration: accumulate per-expert amax of the GEMM1 input (a_perm,
        # permuted per-expert layout). Only the first
        # ``expert_first_token_offset[-1]`` rows are real activations —
        # everything past that is EP-dropped and never written by the permute
        # kernel. Folding those rows into the last local expert (as a plain
        # clamp does) poisons its amax with uninitialized memory; NaNs there
        # survive the scatter-amax and the dump's cross-rank MAX all-reduce
        # then masks them to 0, so the poisoning shows up as a silent
        # amax == 0 for every rank's last local expert. The .item() sync is
        # fine — calibration is a rare offline pass, not the serving path.
        if self._calib_amax is not None:
            n_valid = int(expert_first_token_offset[-1].item())
            if n_valid > 0:
                seg = (
                    torch.searchsorted(
                        expert_first_token_offset,
                        torch.arange(n_valid, device=device, dtype=torch.int64),
                        right=True,
                    )
                    - 1
                ).clamp_(0, E_local - 1)
                self._accum_amax(self._calib_amax[0], a_perm[:n_valid], seg)

        a_perm_fp4, a_perm_scale = ops.scaled_fp4_experts_quant(
            a_perm,
            a1_gscale_q,
            expert_offsets_q,
            blockscale_offsets_q,
            topk,
        )

        gemm1_out = _resize_cache(workspace13, (M_total, two_intermediate_size))
        ops.cutlass_fp4_moe_mm(
            gemm1_out,
            a_perm_fp4,
            w1,
            a_perm_scale,
            self.w1_scale,
            self.g1_alphas,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )

        # PolyNorm operates on the permuted layout. We hand it a fake
        # top_k=1 with a per-row local expert id derived from the cumulative
        # expert_first_token_offset (searchsorted instead of repeat_interleave
        # to stay CPU-sync-free during CUDA graph capture).
        all_rows = torch.arange(M_total, device=device, dtype=torch.int64)
        permuted_topk_ids = (
            (torch.searchsorted(expert_first_token_offset, all_rows, right=True) - 1)
            .clamp_(0, E_local - 1)
            .to(torch.int32)
            .unsqueeze(1)
        )

        gate, up = gemm1_out.split(intermediate_size, dim=-1)
        act_out = self._grouped_polynorm_activation(
            gate.contiguous(), up.contiguous(), permuted_topk_ids, None, 1
        )

        # Calibration: accumulate per-expert amax of the GEMM2 input (PolyNorm
        # output); reuse permuted_topk_ids as the per-row local expert id.
        # Same valid-prefix restriction as the GEMM1 accumulation: past
        # expert_first_token_offset[-1] the GEMMs never wrote gemm1_out, so
        # act_out there is PolyNorm over stale workspace memory (finite but
        # garbage — it inflated a2 amax stats by orders of magnitude).
        if self._calib_amax is not None:
            n_valid = int(expert_first_token_offset[-1].item())
            if n_valid > 0:
                # Recompute segment ids fresh instead of reusing
                # ``permuted_topk_ids``: reusing that buffer after the grouped
                # PolyNorm kernel consumed it lost exactly the LAST local
                # expert's a2 amax on every rank (observed as 8 experts/layer
                # x 51 layers = 408 gscale==1.0 fallback holes in the sidecar,
                # while the a13 stats — which use a fresh searchsorted — were
                # complete). Fresh recomputation is immune to any aliasing or
                # in-kernel mutation of the shared ids buffer.
                seg2 = (
                    torch.searchsorted(
                        expert_first_token_offset,
                        torch.arange(n_valid, device=device, dtype=torch.int64),
                        right=True,
                    )
                    - 1
                ).clamp_(0, E_local - 1)
                self._accum_amax(self._calib_amax[1], act_out[:n_valid], seg2)
                diag = self._calib_amax[2] if len(self._calib_amax) > 2 else None
                if diag is not None:
                    a_abs = act_out[:n_valid].abs().float()
                    diag["ch_amax"].index_reduce_(
                        0, seg2, a_abs, "amax", include_self=True
                    )
                    diag["ch_abssum"].index_add_(0, seg2, a_abs)
                    diag["row_cnt"].index_add_(
                        0,
                        seg2,
                        torch.ones_like(seg2, dtype=torch.float32),
                    )
                    rmax = a_abs.amax(dim=1)
                    hbin = (
                        (rmax.clamp(min=1e-8).log2() * 4)
                        .floor()
                        .long()
                        .add_(32)
                        .clamp_(0, 111)
                    )
                    diag["row_hist"].view(-1).index_add_(
                        0, seg2 * 112 + hbin, torch.ones_like(rmax)
                    )

        act_fp4, act_scale = ops.scaled_fp4_experts_quant(
            act_out.contiguous(),
            a2_gscale_q,
            expert_offsets_q,
            blockscale_offsets_q,
            topk,
        )

        # gemm2_out MUST use workspace2 (not workspace13) — workspace13
        # aliases ``output`` in the non-chunked path and moe_unpermute would
        # then read its own writes.
        gemm2_out = _resize_cache(workspace2, (M_total, K))
        ops.cutlass_fp4_moe_mm(
            gemm2_out,
            act_fp4,
            w2,
            act_scale,
            self.w2_scale,
            self.g2_alphas,
            problem_sizes2,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )

        # moe_unpermute folds topk_weights + reduces over the topk dim.
        # CHECK_SKIPPED branch handles the EP sentinel in inv_permuted_idx.
        moe_unpermute(
            out=output,
            permuted_hidden_states=gemm2_out,
            topk_weights=topk_weights,
            inv_permuted_idx=inv_permuted_idx,
            expert_first_token_offset=expert_first_token_offset,
        )

    @staticmethod
    def _accum_amax(buf: torch.Tensor, x: torch.Tensor, seg_ids: torch.Tensor) -> None:
        """Running per-expert amax: buf[e] = max(buf[e], max|x[rows of e]|).

        ``buf`` is ``[E_local]``; ``x`` is ``[M_total, D]``; ``seg_ids`` is the
        per-row local expert id ``[M_total]``. scatter_reduce(amax) is
        graph-safe and needs no host sync.
        """
        row_amax = x.detach().abs().amax(dim=1).to(torch.float32)
        buf.scatter_reduce_(
            0,
            seg_ids.to(torch.int64),
            row_amax,
            reduce="amax",
            include_self=True,
        )


class MotifNvfp4MoEMethod(_MotifPolyNormMoEMethodBase):
    """Packed NVFP4 experts, allocated before loading on SM100/SM120.

    Serialized checkpoints copy packed weights and linear blockscales directly.
    Dynamic loading quantizes rank-local experts into the same final allocation.
    The SM120 family uses separate PolyNorm and activation quantization kernels.
    """

    def __init__(self, *, direct_load: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.direct_load = direct_load

    @property
    def supports_eplb(self) -> bool:
        return False

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
        from vllm.model_executor.models.motif_weight_utils import (
            allocate_nvfp4_expert_tensors,
        )
        from vllm.model_executor.utils import set_weight_attrs

        if (
            current_platform.is_device_capability_family(120)
            and params_dtype != torch.bfloat16
        ):
            raise ValueError("SM120 Motif NVFP4 grouped GEMM requires --dtype bfloat16")
        tensors = allocate_nvfp4_expert_tensors(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )
        for name, tensor in tensors.items():
            param = nn.Parameter(tensor, requires_grad=False)
            attrs = dict(extra_weight_attrs)
            if name.endswith("_scale"):
                attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
            set_weight_attrs(param, attrs)
            layer.register_parameter(name, param)

    def load_expert_tensor(self, layer, wname, weight):
        """Quantize only rank-local experts; never stage a BF16 MoE layer."""
        if self.direct_load:
            return False
        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Dynamic Motif NVFP4 requires BF16/FP16 expert weights")
        dst = getattr(layer, wname)
        sf = getattr(layer, wname + "_scale")
        alpha = getattr(layer, wname + "_scale_2")
        mapping = layer._expert_map
        local_ids = (
            mapping.tolist() if mapping is not None else list(range(weight.shape[0]))
        )
        if len(local_ids) != weight.shape[0]:
            raise ValueError("Checkpoint expert count does not match the expert map")
        for ge, le in enumerate(local_ids):
            if le < 0:
                continue
            if weight[ge].numel() * weight.element_size() > 128 * 1024**2:
                raise ValueError("Motif expert exceeds the 128 MiB staging budget")
            source = weight[ge].to(device=dst.device)
            q, scales, scale_2 = _quantize_experts_to_nvfp4(source.unsqueeze(0))
            dst.data[le].copy_(q[0])
            sf.data[le].copy_(scales[0])
            alpha.data[le].copy_(scale_2[0])
            del source, q, scales, scale_2
        return True

    def _finalize_direct_loaded_weights(
        self, layer: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate direct-loaded tensors and swizzle the blockscales.

        Returns the pure weight epilogue alphas ``(w13, w2)`` — for NVFP4
        these are exactly the loaded per-expert ``weight_scale_2``.
        """
        for wname in ("w13_weight", "w2_weight"):
            w = getattr(layer, wname)
            s = getattr(layer, wname + "_scale")
            s2 = getattr(layer, wname + "_scale_2")
            assert w.dtype == torch.uint8, (
                f"direct load expects packed uint8 {wname} in the "
                f"checkpoint, got {w.dtype} — was the checkpoint produced "
                "by tools/motif_nvfp4_quantize_ckpt.py?"
            )
            assert s.dtype == torch.float8_e4m3fn, (
                f"{wname}_scale must be float8_e4m3fn, got {s.dtype}"
            )
            assert s2.dtype == torch.float32
            assert bool((torch.isfinite(s2.data) & (s2.data > 0)).all()), (
                f"{wname}_scale_2 must be positive finite — the checkpoint "
                "is missing the per-expert NVFP4 global scales"
            )
            # Bound validation/swizzling temporaries to one expert.
            for e in range(s.shape[0]):
                values = s.data[e].float()
                if not bool(torch.isfinite(values).all()):
                    raise ValueError(f"{wname}_scale: missing or invalid expert {e}")
                del values
                if self.direct_load:
                    s.data[e : e + 1].copy_(swizzle_blockscale(s.data[e : e + 1]))
        return layer.w13_weight_scale_2.data, layer.w2_weight_scale_2.data

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if not (
            current_platform.is_cuda()
            and (
                current_platform.is_device_capability_family(100)
                or current_platform.is_device_capability_family(120)
            )
        ):
            raise RuntimeError(
                "Motif NVFP4 requires a compiled SM100 or SM120-family kernel."
            )

        w13_alpha_w, w2_alpha_w = self._finalize_direct_loaded_weights(layer)
        E_local = layer.w13_weight.shape[0]
        device = layer.w13_weight.device

        # Activation global scales: calibrated sidecar (per-GLOBAL expert,
        # sliced to this rank's locals) if present, else 1.0. a_gscale == 1.0
        # stores block_amax/6 directly in the E4M3 blockscale (safe up to 2688,
        # above hidden_clamp) but wastes precision on small blocks; calibration
        # lifts them into the E4M3 normal range. See module docstring.
        loaded = _local_gscales_for_layer(
            getattr(layer, "layer_name", ""),
            E_local,
            getattr(layer, "_expert_map", None),
            device,
        )
        if loaded is not None:
            a13_gscale, a2_gscale = loaded
            logger.info_once(
                "NVFP4: using calibrated activation scales for MoE experts"
            )
        else:
            a13_gscale = torch.ones(E_local, dtype=torch.float32, device=device)
            a2_gscale = torch.ones(E_local, dtype=torch.float32, device=device)
        layer.register_parameter(
            "a13_input_gscale",
            nn.Parameter(a13_gscale, requires_grad=False),
        )
        layer.register_parameter(
            "a2_input_gscale",
            nn.Parameter(a2_gscale, requires_grad=False),
        )

        # Epilogue alpha MUST undo BOTH global scales: alpha = 1/(a_gscale *
        # w_gscale). w*_alpha_w is 1/w_gscale, so divide by a_gscale.
        layer.register_parameter(
            "w13_alpha",
            nn.Parameter(w13_alpha_w / a13_gscale, requires_grad=False),
        )
        layer.register_parameter(
            "w2_alpha",
            nn.Parameter(w2_alpha_w / a2_gscale, requires_grad=False),
        )

        logger.info_once(
            "NVFP4 MoE weights ready (%s): %d local experts",
            "direct-load" if self.direct_load else "dynamic-quantized",
            E_local,
        )

        # Calibration mode: register per-local-expert amax accumulators that
        # apply() fills; dump_nvfp4_act_scales reads them back.
        if _calibration_enabled():
            layer.register_buffer(
                "a13_amax",
                torch.zeros(E_local, dtype=torch.float32, device=device),
                persistent=False,
            )
            layer.register_buffer(
                "a2_amax",
                torch.zeros(E_local, dtype=torch.float32, device=device),
                persistent=False,
            )
            if _calib_diag_enabled():
                # a2 진단: per-channel amax/abs-sum(채널성 판별)와 row-amax
                # 로그히스토그램(quarter-log2, 112 bins — percentile 계산용).
                w2s = layer.w2_weight
                inter = w2s.shape[2] * (2 if w2s.dtype == torch.uint8 else 1)
                for nm, shape in (
                    ("a2_ch_amax", (E_local, inter)),
                    ("a2_ch_abssum", (E_local, inter)),
                    ("a2_row_cnt", (E_local,)),
                    ("a2_row_hist", (E_local, 112)),
                ):
                    layer.register_buffer(
                        nm,
                        torch.zeros(shape, dtype=torch.float32, device=device),
                        persistent=False,
                    )

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self._init_motif_kernel(layer)

    def get_fused_moe_quant_config(
        self, layer: nn.Module
    ) -> FusedMoEQuantConfig | None:
        # _a1/_a2 dtype=None: the prepare/dispatch flow ships bf16 tokens
        # untouched; MotifNvfp4Experts.apply quantizes them inline via
        # scaled_fp4_experts_quant (after the permute, so the sf layout
        # matches the grouped GEMM's per-expert 128-row tiles). The gscales
        # and alphas ride along on the descs (a1_gscale/g1_alphas etc. are
        # FusedMoEQuantConfig properties reading alpha_or_gscale).
        block_shape = GroupShape(1, NVFP4_BLOCK_SIZE)
        a1_desc = FusedMoEQuantDesc(alpha_or_gscale=layer.a13_input_gscale)
        a2_desc = FusedMoEQuantDesc(alpha_or_gscale=layer.a2_input_gscale)
        w1_desc = FusedMoEQuantDesc(
            dtype="nvfp4",
            shape=block_shape,
            scale=layer.w13_weight_scale,
            alpha_or_gscale=layer.w13_alpha,
        )
        w2_desc = FusedMoEQuantDesc(
            dtype="nvfp4",
            shape=block_shape,
            scale=layer.w2_weight_scale,
            alpha_or_gscale=layer.w2_alpha,
        )
        return FusedMoEQuantConfig(
            _a1=a1_desc,
            _a2=a2_desc,
            _w1=w1_desc,
            _w2=w2_desc,
        )

    def select_gemm_impl(self, prepare_finalize, layer):
        assert prepare_finalize.activation_format == FusedMoEActivationFormat.Standard
        assert self.moe_quant_config is not None
        logger.info_once(
            "NVFP4 MoE modular kernel: MotifNvfp4Experts via %s",
            prepare_finalize.__class__.__name__,
        )
        calib_amax = None
        if hasattr(layer, "a13_amax"):
            diag = None
            if hasattr(layer, "a2_ch_amax"):
                diag = {
                    "ch_amax": layer.a2_ch_amax,
                    "ch_abssum": layer.a2_ch_abssum,
                    "row_cnt": layer.a2_row_cnt,
                    "row_hist": layer.a2_row_hist,
                }
            calib_amax = (layer.a13_amax, layer.a2_amax, diag)
        return MotifNvfp4Experts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            poly_norm_weight=self.poly_norm_weight,
            poly_norm_bias=self.poly_norm_bias,
            hidden_clamp=self.hidden_clamp,
            polynorm_output_scale=self.polynorm_output_scale,
            polynorm_sigmoid_weight=self.polynorm_sigmoid_weight,
            calib_amax=calib_amax,
        )


def dump_nvfp4_act_scales(model: nn.Module, path: str) -> int:
    """Convert calibration amax accumulators into a per-GLOBAL-expert activation
    global-scale sidecar and write it (safetensors) on rank 0.

    Walks ``model`` for Motif MoE experts that carry ``a13_amax``/``a2_amax``
    (registered when ``VLLM_MOTIF_NVFP4_CALIBRATE=1``), maps each rank's local
    amax to the global expert index via ``_expert_map``, all-reduces the max
    across the distributed world (so EP shards and DP replicas combine), then
    stores ``gscale = 2688 / amax`` keyed by ``<layer_name>.a13_gscale`` /
    ``.a2_gscale``. Load it back by pointing ``VLLM_MOTIF_NVFP4_ACT_SCALES`` at
    the file (or dropping it in the model dir). Returns the number of MoE layers
    written.
    """
    import torch.distributed as dist

    world = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if world else 0

    flat: dict[str, torch.Tensor] = {}
    diag_flat: dict[str, torch.Tensor] = {}
    p999_flat: dict[str, torch.Tensor] = {}
    n_layers = 0
    for name, mod in model.named_modules():
        a13 = getattr(mod, "a13_amax", None)
        a2 = getattr(mod, "a2_amax", None)
        if a13 is None or a2 is None:
            continue
        n_layers += 1
        layer_name = getattr(mod, "layer_name", name)
        expert_map = getattr(mod, "_expert_map", None)

        if expert_map is None:
            g13_amax = a13.clone()
            g2_amax = a2.clone()
        else:
            em = expert_map.to(a13.device)
            e_global = int(em.numel())
            g13_amax = torch.zeros(e_global, dtype=torch.float32, device=a13.device)
            g2_amax = torch.zeros(e_global, dtype=torch.float32, device=a13.device)
            global_idx = (em >= 0).nonzero(as_tuple=True)[0]
            local_idx = em[global_idx]
            g13_amax[global_idx] = a13[local_idx].float()
            g2_amax[global_idx] = a2[local_idx].float()

        if world:
            # Same expert may be held (EP) or seen (DP) on multiple ranks: max.
            dist.all_reduce(g13_amax, op=dist.ReduceOp.MAX)
            dist.all_reduce(g2_amax, op=dist.ReduceOp.MAX)

        # 진단 버퍼(diag 모드): local -> global 매핑 후 집계.
        g_diag = None
        if getattr(mod, "a2_ch_amax", None) is not None:

            def _to_global(
                t: torch.Tensor,
                expert_map=expert_map,
                e_global=e_global,
                global_idx=global_idx,
                local_idx=local_idx,
            ) -> torch.Tensor:
                if expert_map is None:
                    return t.float().clone()
                shape = (e_global,) + tuple(t.shape[1:])
                g = torch.zeros(shape, dtype=torch.float32, device=t.device)
                g[global_idx] = t[local_idx].float()
                return g

            g_diag = {
                "ch_amax": _to_global(mod.a2_ch_amax),
                "ch_abssum": _to_global(mod.a2_ch_abssum),
                "row_cnt": _to_global(mod.a2_row_cnt),
                "row_hist": _to_global(mod.a2_row_hist),
            }
            if world:
                dist.all_reduce(g_diag["ch_amax"], op=dist.ReduceOp.MAX)
                dist.all_reduce(g_diag["ch_abssum"])
                dist.all_reduce(g_diag["row_cnt"])
                dist.all_reduce(g_diag["row_hist"])

        if rank == 0:
            # Self-diagnosis: an expert that was routed (a13 amax > 0) must
            # also have a GEMM2-input amax. A hole here means the a2
            # accumulation lost rows (the 408-hole bug shipped exactly this
            # signature: every rank's last local expert, all layers).
            hole = (g13_amax > 0) & (g2_amax <= 0)
            if bool(hole.any()):
                logger.warning(
                    "NVFP4 calib: %d expert(s) in %s have a13 amax but no a2 "
                    "amax (indices %s) — a2 accumulation lost rows; those "
                    "experts fall back to gscale=1.0.",
                    int(hole.sum()),
                    layer_name,
                    hole.nonzero(as_tuple=True)[0].tolist()[:12],
                )
            flat[f"{layer_name}.a13_gscale"] = _gscale_from_amax(g13_amax).cpu()
            flat[f"{layer_name}.a2_gscale"] = _gscale_from_amax(g2_amax).cpu()

            if g_diag is not None:
                cnt = g_diag["row_cnt"].clamp(min=1.0)
                diag_flat[f"{layer_name}.a2_ch_amax"] = g_diag["ch_amax"].cpu()
                diag_flat[f"{layer_name}.a2_ch_mean"] = (
                    g_diag["ch_abssum"] / cnt.unsqueeze(1)
                ).cpu()
                diag_flat[f"{layer_name}.a2_row_cnt"] = g_diag["row_cnt"].cpu()
                diag_flat[f"{layer_name}.a2_row_hist"] = g_diag["row_hist"].cpu()
                # p99.9 row-amax (quarter-log2 bin 상단 경계, 실측 max 이하로 캡).
                hist = g_diag["row_hist"]
                tot = hist.sum(dim=1)
                cum = hist.cumsum(dim=1)
                tgt = (tot * 0.999).unsqueeze(1)
                binidx = (cum >= tgt).float().argmax(dim=1)
                amax_p999 = torch.pow(
                    torch.tensor(2.0, device=hist.device),
                    (binidx.float() + 1.0 - 32.0) / 4.0,
                )
                amax_p999 = torch.minimum(amax_p999, g2_amax)
                amax_p999 = torch.where(tot > 0, amax_p999, torch.zeros_like(amax_p999))
                p999_flat[f"{layer_name}.a13_gscale"] = _gscale_from_amax(
                    g13_amax
                ).cpu()
                p999_flat[f"{layer_name}.a2_gscale"] = _gscale_from_amax(
                    amax_p999
                ).cpu()

    if rank == 0 and flat:
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_file(flat, path)
        if diag_flat:
            save_file(diag_flat, path.replace(".safetensors", ".diag.safetensors"))
            save_file(
                p999_flat,
                os.path.join(
                    os.path.dirname(os.path.abspath(path)),
                    "nvfp4_act_scales_p999.safetensors",
                ),
            )
            logger.info("Wrote a2 diag + p99.9 sidecar next to %s", path)
        logger.info(
            "Wrote NVFP4 activation scales for %d MoE layer(s) to %s",
            n_layers,
            path,
        )
    return n_layers
