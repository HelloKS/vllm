# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


_SM120_GLM_TOPK = 2048


def _normalize_lse(
    lse: torch.Tensor,
    num_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    if lse.dim() == 3:
        if lse.shape[-1] == 1:
            lse = lse.squeeze(-1)
        elif lse.shape[1] == 1:
            lse = lse.squeeze(1)
    if lse.shape != (num_tokens, num_heads):
        raise RuntimeError(
            "Unexpected FlashInfer SM120 sparse MLA LSE shape: "
            f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
        )
    return lse


def _merge_log2_attention_partitions(
    first_out: torch.Tensor,
    first_lse: torch.Tensor,
    second_out: torch.Tensor,
    second_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently normalized attention partitions.

    FlashInfer's SM120 sparse MLA kernels expose LSE in log2 space. Compute
    the merge in FP32; this is the same split-softmax identity used by the
    kernel's internal split-K merge.
    """
    merged_lse = torch.logaddexp2(first_lse, second_lse)
    finite = torch.isfinite(merged_lse)
    safe_lse = torch.where(finite, merged_lse, torch.zeros_like(merged_lse))
    first_weight = torch.where(
        torch.isfinite(first_lse), torch.exp2(first_lse - safe_lse), 0.0
    )
    second_weight = torch.where(
        torch.isfinite(second_lse), torch.exp2(second_lse - safe_lse), 0.0
    )
    merged_out = (
        first_out.float() * first_weight.unsqueeze(-1)
        + second_out.float() * second_weight.unsqueeze(-1)
    ).to(first_out.dtype)
    return merged_out, merged_lse


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False

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
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if self.qk_rope_head_dim != 0 and not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
        runtime_num_heads = q.shape[1]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        topk_indices_physical, sparse_topk_lens = (
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        )

        if self.qk_rope_head_dim == 0:
            from vllm.v1.attention.backends.mla.triton_mla_sparse_glm import (
                glm_fp8ds_nope_sparse_mla,
            )

            out, lse = glm_fp8ds_nope_sparse_mla(
                q=q,
                kv_cache=kv_c_and_k_pe_cache.view(torch.uint8),
                slot_ids=topk_indices_physical,
                lens=sparse_topk_lens,
                block_size=attn_metadata.block_size,
                scale=self.scale,
            )
            return out, lse if self.need_to_return_lse_for_decode else None

        kernel_q = q
        kernel_qk_rope_head_dim = self.qk_rope_head_dim

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        def run_partition(
            indices: torch.Tensor,
            lengths: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            empty_rows = lengths == 0
            indices[:, 0] = indices[:, 0].masked_fill(empty_rows, 0)
            output = q.new_empty(
                (num_actual_toks, runtime_num_heads, self.kv_lora_rank),
                dtype=q.dtype,
            )
            kernel_out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
                query=kernel_q.unsqueeze(1),
                kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
                workspace_buffer=self._workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=kernel_qk_rope_head_dim,
                block_tables=indices.unsqueeze(1),
                seq_lens=lengths.clamp(min=1),
                max_seq_len=_SM120_GLM_TOPK,
                out=output.unsqueeze(1),
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                sparse_mla_top_k=_SM120_GLM_TOPK,
                return_lse=True,
                kv_scale_format=self.kv_scale_format,
            )
            if not isinstance(kernel_out, tuple):
                raise RuntimeError(
                    "FlashInfer SM120 sparse MLA did not return LSE when requested."
                )
            out, lse = kernel_out
            out = out.squeeze(1)
            lse = _normalize_lse(lse, num_actual_toks, runtime_num_heads)
            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
            return out, lse

        # GLM kpool deliberately emits index_topk history tokens plus the
        # incomplete trailing pool (up to kpool - 1 tokens). Its 2176-column
        # buffer is capacity padding, while the SM120 GLM kernel is compiled
        # only for exactly 2048 columns. Preserve the original attention set by
        # evaluating history and tail as separate fixed-shape partitions and
        # merging them through their LSEs. A plain 2176 -> 2048 truncation would
        # silently drop the newest tail tokens.
        # A column slice of the 2176-wide kpool buffer retains row stride 2176
        # and is therefore not contiguous when there is more than one query.
        # FlashInfer's FFI requires a dense [tokens, 2048] indices tensor.
        main_indices = topk_indices_physical[:, :_SM120_GLM_TOPK].contiguous()
        if main_indices.shape[1] < _SM120_GLM_TOPK:
            main_indices = torch.nn.functional.pad(
                main_indices, (0, _SM120_GLM_TOPK - main_indices.shape[1]), value=-1
            )
        main_lens = sparse_topk_lens.clamp(max=_SM120_GLM_TOPK)
        main_out, main_lse = run_partition(main_indices, main_lens)

        if topk_indices_physical.shape[1] > _SM120_GLM_TOPK:
            tail_indices = torch.full(
                (num_actual_toks, _SM120_GLM_TOPK),
                -1,
                dtype=topk_indices_physical.dtype,
                device=topk_indices_physical.device,
            )
            tail_width = min(
                topk_indices_physical.shape[1] - _SM120_GLM_TOPK,
                _SM120_GLM_TOPK,
            )
            tail_indices[:, :tail_width] = topk_indices_physical[
                :, _SM120_GLM_TOPK : _SM120_GLM_TOPK + tail_width
            ]
            tail_lens = (sparse_topk_lens - _SM120_GLM_TOPK).clamp(
                min=0, max=tail_width
            )
            tail_out, tail_lse = run_partition(tail_indices, tail_lens)
            out, lse = _merge_log2_attention_partitions(
                main_out, main_lse, tail_out, tail_lse
            )
        else:
            out, lse = main_out, main_lse

        empty_rows = sparse_topk_lens == 0
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse if self.need_to_return_lse_for_decode else None
