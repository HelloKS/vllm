# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native NoPE sparse MLA Triton backend for GLM on SM12x.

The GLM cache uses the 656-byte ``fp8_ds_mla`` physical entry, but its
attention is logically 512-D and has no RoPE component.  This kernel reads
only the 512 FP8 NoPE bytes and their four FP32 block scales.  In particular,
it does not manufacture a 64-D RoPE tail to select a fixed FlashInfer ABI.
"""

import torch

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

_GLM_NOPE_DIM = 512
_GLM_QUANT_BLOCK = 128
_GLM_ENTRY_BYTES = 656
_GLM_QUERY_CHUNK = 128
_GLM_CANDIDATE_CHUNK = 1024
_GLM_HEAD_BLOCK = 16
_GLM_SCORE_BLOCK = 64
_GLM_VALUE_BLOCK = 128


@triton.jit
def _glm_fp8ds_d512_score_kernel(
    q_ptr,
    kv_cache_ptr,
    slot_ids_ptr,
    lens_ptr,
    scores_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_cache_block,
    stride_cache_token,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    cache_block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_candidates: tl.constexpr,
    candidate_offset: tl.constexpr,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    candidate_block_idx = tl.program_id(2)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    candidate_offsets = candidate_block_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    global_candidates = candidate_offset + candidate_offsets
    dim_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < num_heads
    valid_len = tl.minimum(
        tl.load(lens_ptr + token_idx), candidate_offset + num_candidates
    )
    candidate_mask = (candidate_offsets < num_candidates) & (
        global_candidates < valid_len
    )

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None],
        other=0.0,
    )
    slot_ids = tl.load(
        slot_ids_ptr + token_idx * stride_slot_t + global_candidates * stride_slot_c,
        mask=candidate_mask,
        other=-1,
    )
    valid_slots = candidate_mask & (slot_ids >= 0)
    block_indices = slot_ids // cache_block_size
    positions = slot_ids % cache_block_size
    entry_ptrs = (
        kv_cache_ptr
        + block_indices.to(tl.int64) * stride_cache_block
        + positions * stride_cache_token
    )
    packed = tl.load(
        entry_ptrs[None, :] + dim_offsets[:, None],
        mask=valid_slots[None, :],
        other=0,
    )
    fp8_values = packed.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale_ptrs = (entry_ptrs + HEAD_DIM).to(tl.pointer_type(tl.float32))
    block_scales = tl.load(
        scale_ptrs[None, :] + (dim_offsets // QUANT_BLOCK)[:, None],
        mask=valid_slots[None, :],
        other=0.0,
    )
    kv = (fp8_values * block_scales).to(tl.bfloat16)
    scores = tl.dot(q, kv) * scale

    tl.store(
        scores_ptr
        + token_idx * stride_scores_t
        + head_offsets[:, None] * stride_scores_h
        + candidate_offsets[None, :] * stride_scores_c,
        scores,
        mask=head_mask[:, None] & candidate_mask[None, :],
    )


@triton.jit
def _glm_d512_stats_kernel(
    scores_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    num_candidates: tl.constexpr,
    candidate_offset: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    candidate_offsets = tl.arange(0, BLOCK_C)
    valid_len = tl.load(lens_ptr + token_idx) - candidate_offset
    candidate_mask = candidate_offsets < tl.minimum(
        tl.maximum(valid_len, 0), num_candidates
    )
    scores = tl.load(
        scores_ptr
        + token_idx * stride_scores_t
        + head_idx * stride_scores_h
        + candidate_offsets * stride_scores_c,
        mask=candidate_mask,
        other=-float("inf"),
    ).to(tl.float32)
    chunk_max = tl.max(scores, axis=0)
    safe_max = tl.where(valid_len > 0, chunk_max, 0.0)
    weights = tl.where(candidate_mask, tl.exp(scores - safe_max), 0.0)
    chunk_denom = tl.sum(weights, axis=0)

    tl.store(
        max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        chunk_max,
    )
    tl.store(
        denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        chunk_denom,
    )


@triton.jit
def _glm_fp8ds_d512_value_kernel(
    scores_ptr,
    kv_cache_ptr,
    slot_ids_ptr,
    lens_ptr,
    max_score_ptr,
    acc_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_cache_block,
    stride_cache_token,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_candidates: tl.constexpr,
    candidate_offset: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    candidate_offsets = tl.arange(0, BLOCK_C)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < HEAD_DIM
    local_valid_len = tl.minimum(
        tl.maximum(tl.load(lens_ptr + token_idx) - candidate_offset, 0),
        num_candidates,
    )
    max_score = tl.load(
        max_score_ptr + token_idx * stride_state_t + head_offsets * stride_state_h,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    safe_max = tl.where(local_valid_len > 0, max_score, 0.0)
    acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    for local_start in range(0, num_candidates, BLOCK_C):
        candidates = local_start + candidate_offsets
        candidate_mask = candidates < local_valid_len
        global_candidates = candidate_offset + candidates
        slot_ids = tl.load(
            slot_ids_ptr
            + token_idx * stride_slot_t
            + global_candidates * stride_slot_c,
            mask=candidate_mask,
            other=-1,
        )
        valid_slots = candidate_mask & (slot_ids >= 0)
        scores = tl.load(
            scores_ptr
            + token_idx * stride_scores_t
            + head_offsets[:, None] * stride_scores_h
            + candidates[None, :] * stride_scores_c,
            mask=head_mask[:, None] & candidate_mask[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        weights = tl.where(
            candidate_mask[None, :],
            tl.exp(scores - safe_max[:, None]),
            0.0,
        )
        block_indices = slot_ids // cache_block_size
        positions = slot_ids % cache_block_size
        entry_ptrs = (
            kv_cache_ptr
            + block_indices.to(tl.int64) * stride_cache_block
            + positions * stride_cache_token
        )
        packed = tl.load(
            entry_ptrs[:, None] + dim_offsets[None, :],
            mask=valid_slots[:, None] & dim_mask[None, :],
            other=0,
        )
        fp8_values = packed.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_ptrs = (entry_ptrs + HEAD_DIM).to(tl.pointer_type(tl.float32))
        block_scales = tl.load(
            scale_ptrs[:, None] + (dim_offsets // QUANT_BLOCK)[None, :],
            mask=valid_slots[:, None] & dim_mask[None, :],
            other=0.0,
        )
        values = (fp8_values * block_scales).to(tl.bfloat16)
        acc += tl.dot(weights.to(tl.bfloat16), values)

    tl.store(
        acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _glm_d512_merge_acc_kernel(
    max_score_ptr,
    acc_ptr,
    chunk_max_score_ptr,
    chunk_acc_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < HEAD_DIM
    running_max = tl.load(
        max_score_ptr + token_idx * stride_state_t + head_offsets * stride_state_h,
        mask=head_mask,
        other=-float("inf"),
    ).to(tl.float32)
    chunk_max = tl.load(
        chunk_max_score_ptr
        + token_idx * stride_state_t
        + head_offsets * stride_state_h,
        mask=head_mask,
        other=-float("inf"),
    ).to(tl.float32)
    next_max = tl.maximum(running_max, chunk_max)
    running_scale = tl.where(
        running_max != -float("inf"), tl.exp(running_max - next_max), 0.0
    )
    chunk_scale = tl.where(
        chunk_max != -float("inf"), tl.exp(chunk_max - next_max), 0.0
    )
    running_acc = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    chunk_acc = tl.load(
        chunk_acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        running_acc * running_scale[:, None] + chunk_acc * chunk_scale[:, None],
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _glm_d512_merge_state_kernel(
    max_score_ptr,
    denom_ptr,
    chunk_max_score_ptr,
    chunk_denom_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    running_max = tl.load(
        max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h
    ).to(tl.float32)
    running_denom = tl.load(
        denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h
    ).to(tl.float32)
    chunk_max = tl.load(
        chunk_max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h
    ).to(tl.float32)
    chunk_denom = tl.load(
        chunk_denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h
    ).to(tl.float32)
    next_max = tl.maximum(running_max, chunk_max)
    running_scale = tl.where(
        running_max != -float("inf"), tl.exp(running_max - next_max), 0.0
    )
    chunk_scale = tl.where(
        chunk_max != -float("inf"), tl.exp(chunk_max - next_max), 0.0
    )
    tl.store(
        max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        next_max,
    )
    tl.store(
        denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        running_denom * running_scale + chunk_denom * chunk_scale,
    )


@triton.jit
def _glm_d512_finalize_kernel(
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    output_ptr,
    lse_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    denom = tl.load(
        denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h
    ).to(tl.float32)
    max_score = tl.load(
        max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h
    ).to(tl.float32)
    acc = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + dim_offsets * stride_acc_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    result = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + dim_offsets * stride_output_d,
        result,
        mask=dim_mask,
    )
    if dim_block_idx == 0:
        lse = tl.where(denom > 0.0, max_score + tl.log(denom), -float("inf"))
        tl.store(lse_ptr + token_idx * stride_lse_t + head_idx * stride_lse_h, lse)


def glm_fp8ds_nope_sparse_mla(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run native 512-D GLM sparse MLA over global physical cache slots."""
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if kv_cache.dim() == 4:
        assert kv_cache.shape[1] == 1
        kv_cache = kv_cache[:, 0]

    assert q.dim() == 3 and q.shape[-1] == _GLM_NOPE_DIM
    assert kv_cache.dim() == 3 and kv_cache.dtype == torch.uint8
    assert kv_cache.shape[-1] == _GLM_ENTRY_BYTES
    assert slot_ids.dim() == 2 and slot_ids.shape[0] == q.shape[0]
    assert lens.dim() == 1 and lens.shape[0] == q.shape[0]
    assert q.is_cuda and kv_cache.is_cuda and slot_ids.is_cuda and lens.is_cuda

    num_tokens, num_heads, _ = q.shape
    num_candidates = slot_ids.shape[1]
    output = q.new_empty((num_tokens, num_heads, _GLM_NOPE_DIM))
    lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)

    # Adapt the indexed D512 split family to packed fp8_ds_mla. The score and
    # value stages dequantize directly from the paged cache, avoiding a full
    # BF16 KV materialization. Query and candidate chunking bounds the shared
    # scratch independently of max_num_batched_tokens and preserves every
    # valid entry in the 2176-wide kpool+tail buffer.
    query_chunk = min(num_tokens, _GLM_QUERY_CHUNK)
    score_width = min(num_candidates, _GLM_CANDIDATE_CHUNK)
    workspace_specs = (
        ((query_chunk, num_heads, score_width), torch.float32),
        ((query_chunk, num_heads), torch.float32),
        ((query_chunk, num_heads), torch.float32),
        ((query_chunk, num_heads, _GLM_NOPE_DIM), torch.float32),
        ((query_chunk, num_heads), torch.float32),
        ((query_chunk, num_heads), torch.float32),
        ((query_chunk, num_heads, _GLM_NOPE_DIM), torch.float32),
    )
    if is_workspace_manager_initialized():
        workspace = current_workspace_manager().get_simultaneous(*workspace_specs)
    else:
        workspace = [
            torch.empty(shape, dtype=dtype, device=q.device)
            for shape, dtype in workspace_specs
        ]
    (
        scores_buffer,
        max_score_buffer,
        denom_buffer,
        acc_buffer,
        chunk_max_buffer,
        chunk_denom_buffer,
        chunk_acc_buffer,
    ) = workspace

    for token_start in range(0, num_tokens, _GLM_QUERY_CHUNK):
        token_end = min(token_start + _GLM_QUERY_CHUNK, num_tokens)
        chunk_tokens = token_end - token_start
        q_chunk = q[token_start:token_end]
        slots_chunk = slot_ids[token_start:token_end]
        lens_chunk = lens[token_start:token_end]
        out_chunk = output[token_start:token_end]
        lse_chunk = lse[token_start:token_end]
        scores = scores_buffer[:chunk_tokens]
        running_max = max_score_buffer[:chunk_tokens]
        running_denom = denom_buffer[:chunk_tokens]
        running_acc = acc_buffer[:chunk_tokens]
        chunk_max = chunk_max_buffer[:chunk_tokens]
        chunk_denom = chunk_denom_buffer[:chunk_tokens]
        chunk_acc = chunk_acc_buffer[:chunk_tokens]
        running_max.fill_(float("-inf"))
        running_denom.zero_()
        running_acc.zero_()

        for candidate_start in range(0, num_candidates, _GLM_CANDIDATE_CHUNK):
            candidate_count = min(
                _GLM_CANDIDATE_CHUNK, num_candidates - candidate_start
            )
            score_grid = (
                chunk_tokens,
                triton.cdiv(num_heads, _GLM_HEAD_BLOCK),
                triton.cdiv(candidate_count, _GLM_SCORE_BLOCK),
            )
            _glm_fp8ds_d512_score_kernel[score_grid](
                q_chunk,
                kv_cache,
                slots_chunk,
                lens_chunk,
                scores,
                q_chunk.stride(0),
                q_chunk.stride(1),
                q_chunk.stride(2),
                kv_cache.stride(0),
                kv_cache.stride(1),
                slots_chunk.stride(0),
                slots_chunk.stride(1),
                scores.stride(0),
                scores.stride(1),
                scores.stride(2),
                block_size,
                num_heads,
                candidate_count,
                candidate_start,
                float(scale),
                HEAD_BLOCK=_GLM_HEAD_BLOCK,
                BLOCK_C=_GLM_SCORE_BLOCK,
                HEAD_DIM=_GLM_NOPE_DIM,
                QUANT_BLOCK=_GLM_QUANT_BLOCK,
                num_warps=8,
                # The 16x512 by 512x64 score tile consumes 49,152 bytes per
                # pipeline stage. Three stages require 147,456 bytes, above
                # SM120's 101,376-byte shared-memory limit; two use 98,304.
                num_stages=2,
            )

            stats_grid = (chunk_tokens, num_heads)
            _glm_d512_stats_kernel[stats_grid](
                scores,
                lens_chunk,
                chunk_max,
                chunk_denom,
                scores.stride(0),
                scores.stride(1),
                scores.stride(2),
                chunk_max.stride(0),
                chunk_max.stride(1),
                candidate_count,
                candidate_start,
                BLOCK_C=next_power_of_2(candidate_count),
                num_warps=4,
                num_stages=3,
            )

            value_grid = (
                chunk_tokens,
                triton.cdiv(num_heads, _GLM_HEAD_BLOCK),
                triton.cdiv(_GLM_NOPE_DIM, _GLM_VALUE_BLOCK),
            )
            _glm_fp8ds_d512_value_kernel[value_grid](
                scores,
                kv_cache,
                slots_chunk,
                lens_chunk,
                chunk_max,
                chunk_acc,
                scores.stride(0),
                scores.stride(1),
                scores.stride(2),
                kv_cache.stride(0),
                kv_cache.stride(1),
                slots_chunk.stride(0),
                slots_chunk.stride(1),
                chunk_max.stride(0),
                chunk_max.stride(1),
                chunk_acc.stride(0),
                chunk_acc.stride(1),
                chunk_acc.stride(2),
                block_size,
                num_heads,
                candidate_count,
                candidate_start,
                HEAD_BLOCK=_GLM_HEAD_BLOCK,
                BLOCK_C=_GLM_SCORE_BLOCK,
                BLOCK_D=_GLM_VALUE_BLOCK,
                HEAD_DIM=_GLM_NOPE_DIM,
                QUANT_BLOCK=_GLM_QUANT_BLOCK,
                num_warps=4,
                num_stages=3,
            )

            merge_acc_grid = (
                chunk_tokens,
                triton.cdiv(num_heads, _GLM_HEAD_BLOCK),
                triton.cdiv(_GLM_NOPE_DIM, _GLM_VALUE_BLOCK),
            )
            _glm_d512_merge_acc_kernel[merge_acc_grid](
                running_max,
                running_acc,
                chunk_max,
                chunk_acc,
                running_max.stride(0),
                running_max.stride(1),
                running_acc.stride(0),
                running_acc.stride(1),
                running_acc.stride(2),
                num_heads,
                HEAD_BLOCK=_GLM_HEAD_BLOCK,
                BLOCK_D=_GLM_VALUE_BLOCK,
                HEAD_DIM=_GLM_NOPE_DIM,
                num_warps=4,
                num_stages=3,
            )
            _glm_d512_merge_state_kernel[stats_grid](
                running_max,
                running_denom,
                chunk_max,
                chunk_denom,
                running_max.stride(0),
                running_max.stride(1),
                num_warps=4,
                num_stages=3,
            )

        finalize_grid = (
            chunk_tokens,
            num_heads,
            triton.cdiv(_GLM_NOPE_DIM, _GLM_VALUE_BLOCK),
        )
        _glm_d512_finalize_kernel[finalize_grid](
            running_max,
            running_denom,
            running_acc,
            out_chunk,
            lse_chunk,
            running_max.stride(0),
            running_max.stride(1),
            running_acc.stride(0),
            running_acc.stride(1),
            running_acc.stride(2),
            out_chunk.stride(0),
            out_chunk.stride(1),
            out_chunk.stride(2),
            lse_chunk.stride(0),
            lse_chunk.stride(1),
            num_heads,
            BLOCK_D=_GLM_VALUE_BLOCK,
            HEAD_DIM=_GLM_NOPE_DIM,
            num_warps=4,
        )
    return output, lse


class GlmTritonMLASparseMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    """FlashInfer-compatible metadata plus capture-stable D512 scratch."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if not is_workspace_manager_initialized():
            return
        num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        current_workspace_manager().get_simultaneous(
            ((_GLM_QUERY_CHUNK, num_heads, _GLM_CANDIDATE_CHUNK), torch.float32),
            ((_GLM_QUERY_CHUNK, num_heads), torch.float32),
            ((_GLM_QUERY_CHUNK, num_heads), torch.float32),
            ((_GLM_QUERY_CHUNK, num_heads, _GLM_NOPE_DIM), torch.float32),
            ((_GLM_QUERY_CHUNK, num_heads), torch.float32),
            ((_GLM_QUERY_CHUNK, num_heads), torch.float32),
            ((_GLM_QUERY_CHUNK, num_heads, _GLM_NOPE_DIM), torch.float32),
        )


class GlmTritonMLASparseSM120Backend(AttentionBackend):
    """SM120 backend for GLM's native 512-D, RoPE-free sparse MLA."""

    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes: list[CacheDType] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_GLM_SM120"

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return GlmTritonMLASparseSM120Impl

    @staticmethod
    def get_builder_cls() -> type[FlashInferMLASparseMetadataBuilder]:
        return GlmTritonMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_GLM_NOPE_DIM]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if dtype != torch.bfloat16:
            return "dtype not supported"
        if (
            kv_cache_dtype not in cls.supported_kv_cache_dtypes
            and kv_cache_dtype is not None
        ):
            return "kv_cache_dtype not supported"

        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        if config.model_config is None:
            return "GLM sparse MLA requires model configuration"
        hf_config = config.model_config.hf_text_config
        if int(getattr(hf_config, "qk_rope_head_dim", 64)) != 0:
            return "GLM Triton sparse MLA requires qk_rope_head_dim=0"
        if int(getattr(hf_config, "kv_lora_rank", 0)) != _GLM_NOPE_DIM:
            return "GLM Triton sparse MLA requires kv_lora_rank=512"
        if not hasattr(hf_config, "index_topk"):
            return "GLM Triton sparse MLA requires index_topk"
        return None


class GlmTritonMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """GLM-specific adapter around the native 512-D Triton kernel."""

    is_sparse = True
    supports_dense_mha_prefill = False
    can_return_lse_for_decode = True
    lse_base_on_e = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer=None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "TRITON_MLA_SPARSE_GLM_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "TRITON_MLA_SPARSE_GLM_SM120 only supports decoder self-attention"
            )
        if kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "TRITON_MLA_SPARSE_GLM_SM120 requires fp8_ds_mla KV cache"
            )
        if int(mla_args["kv_lora_rank"]) != _GLM_NOPE_DIM:
            raise NotImplementedError("GLM Triton sparse MLA requires kv_lora_rank=512")
        if int(mla_args["qk_rope_head_dim"]) != 0:
            raise NotImplementedError("GLM Triton sparse MLA requires native NoPE")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        if self.topk_indices_buffer is None:
            raise RuntimeError("GLM Triton sparse MLA requires a top-k index buffer")
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_tokens = q.shape[0]
        assert self.topk_indices_buffer is not None
        request_indices = self.topk_indices_buffer[:num_tokens]
        physical_indices, valid_lens = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            request_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=request_indices.shape[1],
            return_valid_counts=True,
        )
        output, lse = glm_fp8ds_nope_sparse_mla(
            q=q,
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8),
            slot_ids=physical_indices,
            lens=valid_lens,
            block_size=attn_metadata.block_size,
            scale=self.scale,
        )
        return output, lse if self.need_to_return_lse_for_decode else None
