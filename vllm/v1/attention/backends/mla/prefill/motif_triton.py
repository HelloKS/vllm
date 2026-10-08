# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions, MLAPrefillBackend
from vllm.v1.attention.ops.motif_attention import motif_attention


class MotifTritonPrefillBackend(MLAPrefillBackend):
    supported_mla_dimensions = [
        MLADimensions(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
    ]

    @staticmethod
    def get_name():
        return "MOTIF_TRITON"

    @classmethod
    def supports_compute_capability(cls, device_capability):
        return device_capability.major == 12

    def supports_out(self):
        return True

    def run_prefill_new_tokens(
        self, q, k, v, return_softmax_lse, out=None, output_scale=None
    ):
        if output_scale is not None:
            raise NotImplementedError("Motif Triton prefill does not quantize output")
        meta = self._prefill_metadata
        result = motif_attention(
            q,
            k,
            v,
            meta.query_start_loc,
            meta.query_start_loc,
            meta.max_query_len,
            self.scale,
            causal=True,
            out=out,
        )
        return result if return_softmax_lse else result[0]

    def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
        return motif_attention(
            q,
            k,
            v,
            chunk.query_start_loc,
            chunk.cu_seq_lens,
            chunk.max_query_len,
            self.scale,
            causal=False,
            out=out,
        )
