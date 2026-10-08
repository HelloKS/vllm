# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif DiffKV fallback using 16x32 tiles and separate QK/V dimensions."""

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.ops.motif_attention import motif_attention
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)


class TritonDiffKVBackend(TritonAttentionBackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16", "float16"]

    @classmethod
    def supports_non_causal(cls):
        return False

    @classmethod
    def supports_mm_prefix(cls):
        return False

    @classmethod
    def supports_rswa(cls):
        return False

    @staticmethod
    def get_name():
        return "TRITON_DIFFKV"

    @staticmethod
    def get_impl_cls():
        return TritonDiffKVImpl

    @classmethod
    def supports_attn_type(cls, attn_type):
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls):
        return False


class TritonDiffKVImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        unsupported = []
        if self.kv_cache_dtype not in TritonDiffKVBackend.supported_kv_cache_dtypes:
            unsupported.append(f"kv_cache_dtype={self.kv_cache_dtype!r}")
        if self.attn_type != AttentionType.DECODER:
            unsupported.append(f"attn_type={self.attn_type!r}")
        if self.alibi_slopes is not None:
            unsupported.append("alibi_slopes is set")
        if self.sinks is not None:
            unsupported.append("sinks is set")
        if self.logits_soft_cap:
            unsupported.append(f"logits_soft_cap={self.logits_soft_cap!r}")
        if self.chunk_lookback != -1:
            unsupported.append(f"chunk_lookback={self.chunk_lookback!r}")
        if self.use_alibi_sqrt:
            unsupported.append("use_alibi_sqrt=True")
        if unsupported:
            hint = (
                " Use --kv-cache-dtype bfloat16 with --dtype bfloat16; "
                "NVFP4 checkpoint weights do not require quantized KV cache."
                if self.kv_cache_dtype
                not in TritonDiffKVBackend.supported_kv_cache_dtypes
                else ""
            )
            raise NotImplementedError(
                "Motif DiffKV does not support " + ", ".join(unsupported) + "." + hint
            )
        self.supports_quant_query_input = False

    def fused_output_quant_supported(self, quant_key):
        return False

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache.transpose(1, 2),
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Motif DiffKV does not quantize output")
        if attn_metadata is None:
            return output.zero_()
        n = attn_metadata.num_actual_tokens
        k, v = kv_cache.transpose(1, 2).split(
            [self.head_size, output.shape[-1]],
            dim=-1,
        )
        motif_attention(
            query[:n],
            k,
            v,
            attn_metadata.query_start_loc,
            None,
            attn_metadata.max_query_len,
            self.scale,
            causal=True,
            window=self.sliding_window[0],
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            out=output[:n],
        )
        return output
