# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check DiffKV, sliding-window boundaries, chunk LSE and graph replay."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "paged,causal,window,page_padding",
    [
        (False, False, -1, 0),
        (False, True, -1, 0),
        (True, True, 128, 0),
        (True, True, 128, 4096),  # FP8 MLA + BF16 SWA, TP=2.
        (True, True, 128, 40960),  # BF16 MLA + BF16 SWA, TP=2.
    ],
)
@pytest.mark.parametrize("query_len,kv_len", [(1, 129), (17, 257), (33, 33)])
def test_motif_diffkv_matches_reference(
    paged, causal, window, page_padding, query_len, kv_len
):
    from vllm.v1.attention.ops.motif_attention import motif_attention

    torch.manual_seed(9)
    device = "cuda"
    q = torch.randn(query_len, 40, 192, device=device, dtype=torch.bfloat16)
    k = torch.randn(kv_len, 8, 192, device=device, dtype=q.dtype)
    v = torch.randn(kv_len, 8, 128, device=device, dtype=q.dtype)
    qs = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    ks = torch.tensor([0, kv_len], device=device, dtype=torch.int32)
    extra = {}
    if paged:
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash_diffkv,
        )

        pages = (kv_len + 127) // 128
        # Reverse physical pages to catch accidental contiguous-cache reads.
        table = torch.arange(pages - 1, -1, -1, device=device, dtype=torch.int32)[None]
        page_elements = 128 * 8 * 320
        storage = torch.zeros(
            pages, page_elements + page_padding, device=device, dtype=q.dtype
        )
        cache = storage.as_strided(
            (pages, 128, 8, 320), (storage.stride(0), 8 * 320, 320, 1)
        )
        positions = torch.arange(kv_len, device=device, dtype=torch.int64)
        slots = table[0, positions // 128].long() * 128 + positions % 128
        scale = torch.ones((), device=device, dtype=torch.float32)
        triton_reshape_and_cache_flash_diffkv(k, v, cache, slots, "auto", scale, scale)
        kc, vc = cache.split([192, 128], dim=-1)
        assert torch.count_nonzero(storage[:, page_elements:]) == 0
        extra = dict(block_table=table, seq_lens=ks[1:])
    else:
        kc, vc = k, v
    out, lse = motif_attention(
        q, kc, vc, qs, ks, query_len, 192**-0.5, causal=causal, window=window, **extra
    )
    scores = (
        torch.einsum("qhd,khd->hqk", q.float(), k.float().repeat_interleave(5, 1))
        / 192**0.5
    )
    pos = torch.arange(query_len, device=device) + kv_len - query_len
    cols = torch.arange(kv_len, device=device)
    if causal:
        mask = cols[None, :] <= pos[:, None]
        if window >= 0:
            mask &= cols[None, :] >= pos[:, None] - window
        scores.masked_fill_(~mask, -torch.inf)
    ref = torch.einsum(
        "hqk,khd->qhd", scores.softmax(-1), v.float().repeat_interleave(5, 1)
    )
    torch.testing.assert_close(out.float(), ref, atol=0.025, rtol=0.025)
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=0.01, rtol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = motif_attention(
            q,
            kc,
            vc,
            qs,
            ks,
            query_len,
            192**-0.5,
            causal=causal,
            window=window,
            **extra,
        )
    graph.replay()
    torch.testing.assert_close(captured, out)


def test_motif_prefill_empty_context_has_zero_output_and_negative_infinite_lse():
    from vllm.v1.attention.ops.motif_attention import motif_attention

    q = torch.randn(5, 4, 192, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(0, 4, 192, device="cuda", dtype=q.dtype)
    v = torch.empty(0, 4, 128, device="cuda", dtype=q.dtype)
    qs = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    ks = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    out, lse = motif_attention(q, k, v, qs, ks, 5, 192**-0.5, causal=False)
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()


def test_motif_ragged_context_lse_keeps_sequence_and_head_offsets():
    from vllm.v1.attention.ops.motif_attention import motif_attention

    torch.manual_seed(17)
    q = torch.randn(20, 10, 192, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(65, 2, 192, device="cuda", dtype=q.dtype)
    v = torch.randn(65, 2, 128, device="cuda", dtype=q.dtype)
    qs = torch.tensor([0, 3, 20], device="cuda", dtype=torch.int32)
    ks = torch.tensor([0, 0, 65], device="cuda", dtype=torch.int32)
    out, lse = motif_attention(q, k, v, qs, ks, 17, 192**-0.5, causal=False)
    assert torch.count_nonzero(out[:3]) == 0
    assert torch.isneginf(lse[:, :3]).all()
    scores = (
        torch.einsum("qhd,khd->hqk", q[3:].float(), k.float().repeat_interleave(5, 1))
        / 192**0.5
    )
    expected = torch.einsum(
        "hqk,khd->qhd", scores.softmax(-1), v.float().repeat_interleave(5, 1)
    )
    torch.testing.assert_close(out[3:].float(), expected, atol=0.025, rtol=0.025)
    torch.testing.assert_close(lse[:, 3:], scores.logsumexp(-1), atol=0.01, rtol=0.01)
