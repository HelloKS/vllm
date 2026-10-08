# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-tile DiffKV attention for Motif prefill and sliding-window decode."""

import torch

from vllm.triton_utils import tl, triton

_VALIDATED_KERNELS: set[tuple] = set()


@triton.jit
def _attention(
    Q,
    K,
    V,
    OUT,
    LSE,
    Q_START,
    K_START,
    SEQ_LENS,
    BLOCK_TABLE,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    os0: tl.constexpr,
    os1: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    HQ: tl.constexpr,
    GROUP: tl.constexpr,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    BQ: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGED: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
):
    tile, head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_start = tl.load(Q_START + seq)
    q_len = tl.load(Q_START + seq + 1) - q_start
    rows = tile * BM + tl.arange(0, BM)
    dq = tl.arange(0, BQ)
    dv = tl.arange(0, BV)
    if PAGED:
        k_start = 0
        k_len = tl.load(SEQ_LENS + seq)
    else:
        k_start = tl.load(K_START + seq)
        k_len = tl.load(K_START + seq + 1) - k_start
    q = tl.load(
        Q + (q_start + rows[:, None]) * qs0 + head * qs1 + dq[None, :],
        (rows[:, None] < q_len) & (dq[None, :] < DQ),
        0,
    )
    m = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BV), tl.float32)
    end = k_len
    begin = 0
    if CAUSAL:
        end = tl.minimum(k_len, k_len - q_len + (tile + 1) * BM)
        if WINDOW >= 0:
            begin = tl.maximum(0, k_len - q_len + tile * BM - WINDOW)
            begin = begin // BN * BN
    for start in range(begin, end, BN):
        cols = start + tl.arange(0, BN)
        if PAGED:
            page = tl.load(
                BLOCK_TABLE + seq * TABLE_STRIDE + cols // PAGE, cols < k_len, 0
            )
            k_base = page * ks0 + (cols % PAGE) * ks1 + (head // GROUP) * ks2
            v_base = page * vs0 + (cols % PAGE) * vs1 + (head // GROUP) * vs2
        else:
            k_base = (k_start + cols) * ks0 + (head // GROUP) * ks1
            v_base = (k_start + cols) * vs0 + (head // GROUP) * vs1
        k = tl.load(
            K + k_base[None, :] + dq[:, None],
            (cols[None, :] < k_len) & (dq[:, None] < DQ),
            0,
        )
        v = tl.load(
            V + v_base[:, None] + dv[None, :],
            (cols[:, None] < k_len) & (dv[None, :] < DV),
            0,
        )
        scores = tl.dot(q, k).to(tl.float32) * SCALE
        valid = (rows[:, None] < q_len) & (cols[None, :] < k_len)
        if CAUSAL:
            pos = k_len - q_len + rows
            valid &= cols[None, :] <= pos[:, None]
            if WINDOW >= 0:
                valid &= cols[None, :] >= pos[:, None] - WINDOW
        scores = tl.where(valid, scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(scores - safe_m[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        denom = denom * alpha + tl.sum(p, 1)
        m = new_m
    acc /= tl.where(denom > 0, denom, 1.0)[:, None]
    tl.store(
        OUT + (q_start + rows[:, None]) * os0 + head * os1 + dv[None, :],
        acc,
        (rows[:, None] < q_len) & (dv[None, :] < DV),
    )
    tl.store(
        LSE + head * tl.load(Q_START + tl.num_programs(2)) + q_start + rows,
        tl.where(denom > 0, m + tl.log(denom), -float("inf")),
        rows < q_len,
    )


def motif_attention(
    q,
    k,
    v,
    q_start,
    k_start,
    max_query_len,
    scale,
    *,
    causal,
    window=-1,
    block_table=None,
    seq_lens=None,
    out=None,
):
    """Return output and natural-log LSE [heads, tokens], including empty KV."""
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("Motif Triton attention requires matching BF16/FP16 Q/K/V")
    paged = block_table is not None
    kv_heads = k.shape[-2]
    if q.shape[1] % kv_heads or q.shape[-1] != k.shape[-1]:
        raise ValueError("Invalid Motif GQA head dimensions")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("Motif attention requires contiguous head dimensions")
    if out is None:
        out = torch.empty((*q.shape[:2], v.shape[-1]), device=q.device, dtype=q.dtype)
    lse = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
    if q.shape[0] == 0:
        return out, lse
    grid = (triton.cdiv(max_query_len, 16), q.shape[1], q_start.numel() - 1)
    args = (
        q,
        k,
        v,
        out,
        lse,
        q_start,
        k_start,
        seq_lens,
        block_table,
        *q.stride()[:2],
        *k.stride()[:3],
        *v.stride()[:3],
        *out.stride()[:2],
        block_table.stride(0) if paged else 0,
        k.shape[1] if paged else 1,
        q.shape[1],
        q.shape[1] // kv_heads,
        q.shape[-1],
        v.shape[-1],
        triton.next_power_of_2(q.shape[-1]),
        triton.next_power_of_2(v.shape[-1]),
        scale,
        causal,
        window,
        paged,
    )
    signature = (q.device, q.dtype, args[9:])
    if signature not in _VALIDATED_KERNELS:
        from vllm.utils.mem_utils import get_max_shared_memory_bytes

        compiled = _attention.warmup(*args, grid=grid, num_warps=4, num_stages=1)
        available = get_max_shared_memory_bytes(q.device.index)
        required = compiled.metadata.shared
        if required > available:
            raise RuntimeError(
                f"Motif DiffKV requires {required} bytes shared memory; "
                f"device permits {available} bytes"
            )
        _VALIDATED_KERNELS.add(signature)
    _attention[grid](*args, num_warps=4, num_stages=1)
    return out, lse
