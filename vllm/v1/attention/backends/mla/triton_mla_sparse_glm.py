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

_GLM_NOPE_DIM = 512
_GLM_QUANT_BLOCK = 128
_GLM_ENTRY_BYTES = 656


@triton.jit
def _glm_fp8ds_nope_sparse_mla_kernel(
    q_ptr,
    kv_cache_ptr,
    slot_ids_ptr,
    lens_ptr,
    output_ptr,
    lse_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_cache_block,
    stride_cache_token,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
    cache_block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_candidates: tl.constexpr,
    head_dim: tl.constexpr,
    quant_block: tl.constexpr,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, head_dim)
    head_mask = head_offsets < num_heads
    matrix_mask = head_mask[:, None]

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((HEAD_BLOCK,), -float("inf"), tl.float32)
    running_denom = tl.zeros((HEAD_BLOCK,), tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, head_dim), tl.float32)
    valid_len = tl.minimum(tl.load(lens_ptr + token_idx), num_candidates)

    for candidate_idx in range(0, valid_len):
        slot_id = tl.load(
            slot_ids_ptr + token_idx * stride_slot_t + candidate_idx * stride_slot_c
        )
        if slot_id >= 0:
            block_idx = slot_id // cache_block_size
            pos_in_block = slot_id % cache_block_size
            entry_ptr = (
                kv_cache_ptr
                + block_idx.to(tl.int64) * stride_cache_block
                + pos_in_block * stride_cache_token
            )

            packed = tl.load(entry_ptr + dim_offsets)
            fp8_values = packed.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            scale_ptr = (entry_ptr + head_dim).to(tl.pointer_type(tl.float32))
            block_scales = tl.load(scale_ptr + scale_offsets)
            kv = fp8_values * block_scales

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            old_weight = tl.exp(running_max - next_max)
            new_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * old_weight[:, None] + kv[None, :] * new_weight[:, None]
            )
            running_denom = running_denom * old_weight + new_weight
            running_max = next_max

    has_tokens = running_denom > 0.0
    inv_denom = tl.where(has_tokens, 1.0 / running_denom, 0.0)
    result = running_acc * inv_denom[:, None]
    lse = tl.where(
        has_tokens,
        running_max + tl.log(running_denom),
        -float("inf"),
    )

    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_offsets[:, None] * stride_output_h
        + dim_offsets[None, :] * stride_output_d,
        result,
        mask=matrix_mask,
    )
    tl.store(
        lse_ptr + token_idx * stride_lse_t + head_offsets * stride_lse_h,
        lse,
        mask=head_mask,
    )


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
    output = q.new_empty((num_tokens, num_heads, _GLM_NOPE_DIM))
    lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)

    # This follows the head grouping used by the portable D512 family: keep
    # latency-sensitive decode at one head/program, and share each dequantized
    # KV row across a few heads for larger token batches.
    head_block = 1 if num_tokens <= 4 else 2 if num_tokens < 16 else 4
    grid = (num_tokens, triton.cdiv(num_heads, head_block))
    _glm_fp8ds_nope_sparse_mla_kernel[grid](
        q,
        kv_cache,
        slot_ids,
        lens,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        slot_ids.stride(0),
        slot_ids.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        block_size,
        num_heads,
        slot_ids.shape[1],
        _GLM_NOPE_DIM,
        _GLM_QUANT_BLOCK,
        float(scale),
        HEAD_BLOCK=head_block,
        num_warps=8,
        num_stages=3,
    )
    return output, lse


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
        return FlashInferMLASparseMetadataBuilder

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
