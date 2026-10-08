# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT2/g32 kernels. Weights remain packed throughout prefill and decode.

All launches have shape-derived bounds. Routing stays on device, and scratch
belongs to the invocation (and hence to the capturing CUDA graph pool).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gemv(
    X,
    Q,
    S,
    Z,
    IDS,
    PART,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    MODE: tl.constexpr,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 256,
):
    pn, pair, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = pn * BN + tl.arange(0, BN)
    k = split * BK + tl.arange(0, BK)
    expert = 0
    row = pair
    if MODE != 0:
        expert = tl.load(IDS + pair).to(tl.int64)
    if MODE == 1:
        row = pair // TOPK
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    mask = (k[:, None] < K) & (n[None, :] < N)
    word = tl.load(
        Q + expert * (K // 16) * N + (k[:, None] // 16) * N + n[None, :], mask, 0
    )
    q = (word >> (2 * (k[:, None] % 16))) & 3
    g = expert * (K // 32) * N + (k[:, None] // 32) * N + n[None, :]
    scale = tl.load(S + g, mask, 0).to(tl.float32)
    zero = tl.load(Z + g, mask, 0).to(tl.float32)
    w = (q.to(tl.float32) - zero) * scale
    y = tl.sum(x[:, None] * w, axis=0)
    tl.store(PART + (pair * SPLITS + split) * N + n, y, n < N)


@triton.jit
def _reduce_parts(
    PART,
    OUT,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    RS: tl.constexpr,
    RELU2: tl.constexpr,
    BN: tl.constexpr = 128,
):
    pair, pn = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, RS)
    n = pn * BN + tl.arange(0, BN)
    v = tl.load(
        PART + (pair * SPLITS + s[:, None]) * N + n[None, :],
        (s[:, None] < SPLITS) & (n[None, :] < N),
        0,
    )
    y = tl.sum(v, axis=0)
    if RELU2:
        # Match BF16 Linear output followed by BF16 ReLU squared.
        y = tl.maximum(y.to(OUT.dtype.element_ty).to(tl.float32), 0)
        y = y * y
    tl.store(OUT + pair * N + n, y, n < N)


@triton.jit
def _linear_gemm(
    X,
    Q,
    S,
    Z,
    PART,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
):
    pm, pn, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m, n = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(split, K // 32, SPLITS):
        k = group * 32 + tl.arange(0, 32)
        a = tl.load(X + m[:, None] * K + k[None, :], m[:, None] < M, 0)
        word = tl.load(Q + (k[:, None] // 16) * N + n[None, :], n[None, :] < N, 0)
        q = (word >> (2 * (k[:, None] % 16))) & 3
        s = tl.load(S + group * N + n, n < N, 0).to(tl.float32)
        z = tl.load(Z + group * N + n, n < N, 0).to(tl.float32)
        b = ((q.to(tl.float32) - z[None, :]) * s[None, :]).to(a.dtype)
        acc += tl.dot(a, b)
    tl.store(
        PART + (m[:, None] * SPLITS + split) * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _grouped_gemm(
    X,
    Q,
    S,
    Z,
    SORTED,
    EXPERT,
    PADDED,
    OUT,
    PAIRS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    UP: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
):
    block, pn = tl.program_id(0), tl.program_id(1)
    if block * BM >= tl.load(PADDED):
        return
    expert = tl.load(EXPERT + block).to(tl.int64)
    pairs = tl.load(SORTED + block * BM + tl.arange(0, BM))
    rows = pairs
    if UP:
        rows = pairs // TOPK
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 32):
        k = group * 32 + tl.arange(0, 32)
        a = tl.load(X + rows[:, None] * K + k[None, :], pairs[:, None] < PAIRS, 0)
        word = tl.load(
            Q + expert * (K // 16) * N + (k[:, None] // 16) * N + n[None, :],
            n[None, :] < N,
            0,
        )
        q = (word >> (2 * (k[:, None] % 16))) & 3
        g = (expert * (K // 32) + group) * N + n
        s = tl.load(S + g, n < N, 0).to(tl.float32)
        z = tl.load(Z + g, n < N, 0).to(tl.float32)
        b = ((q.to(tl.float32) - z[None, :]) * s[None, :]).to(a.dtype)
        acc += tl.dot(a, b)
    if UP:
        acc = tl.maximum(acc.to(OUT.dtype.element_ty).to(tl.float32), 0)
        acc = acc * acc
    tl.store(
        OUT + pairs[:, None] * N + n[None, :],
        acc,
        (pairs[:, None] < PAIRS) & (n[None, :] < N),
    )


@triton.jit
def _combine(
    X,
    ROUTE,
    Y,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    RT: tl.constexpr,
    BN: tl.constexpr = 128,
):
    token, pn = tl.program_id(0), tl.program_id(1)
    t, n = tl.arange(0, RT), pn * BN + tl.arange(0, BN)
    x = tl.load(
        X + (token * TOPK + t[:, None]) * N + n[None, :],
        (t[:, None] < TOPK) & (n[None, :] < N),
        0,
    ).to(tl.float32)
    r = tl.load(ROUTE + token * TOPK + t, t < TOPK, 0).to(tl.float32)
    y = tl.sum(x * r[:, None], axis=0)
    tl.store(Y + token * N + n, y, n < N)


def _check_input(x):
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("DASH-Q kernels require CUDA BF16 activations")


def _gemv_forward(x, q, s, z, ids, pairs, mode, topk, relu2=False):
    n, k = q.shape[-1], q.shape[-2] * 16
    splits = triton.cdiv(k, 256)
    partial = torch.empty((pairs, splits, n), device=x.device, dtype=torch.float32)
    output = torch.empty((pairs, n), device=x.device, dtype=x.dtype)
    _gemv[(triton.cdiv(n, 32), pairs, splits)](
        x, q, s, z, ids, partial, n, k, topk, splits, mode, num_warps=4
    )
    _reduce_parts[(pairs, triton.cdiv(n, 128))](
        partial,
        output,
        n,
        splits,
        triton.next_power_of_2(splits),
        relu2,
        num_warps=4,
    )
    return output


def dashq_linear(x, q, scale, zero, bias=None):
    _check_input(x)
    shape = x.shape
    k, n = q.shape[0] * 16, q.shape[1]
    if shape[-1] != k:
        raise ValueError("DASH-Q input dimension mismatch")
    x = x.reshape(-1, k).contiguous()
    m = x.shape[0]
    if m == 0:
        return x.new_empty((*shape[:-1], n))
    if m == 1:
        y = _gemv_forward(x, q, scale, zero, x, 1, 0, 1)
    else:
        splits = min(4, k // 32) if m <= 32 else 1
        partial = torch.empty((m, splits, n), dtype=torch.float32, device=x.device)
        y = x.new_empty((m, n))
        _linear_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 64), splits)](
            x,
            q,
            scale,
            zero,
            partial,
            m,
            n,
            k,
            splits,
            num_warps=4,
        )
        _reduce_parts[(m, triton.cdiv(n, 128))](
            partial,
            y,
            n,
            splits,
            triton.next_power_of_2(splits),
            False,
        )
    if bias is not None:
        y = y + bias
    return y.reshape(*shape[:-1], n)


def dashq_moe(x, q1, s1, z1, q2, s2, z2, topk_weights, topk_ids):
    _check_input(x)
    x = x.contiguous()
    ids, weights = topk_ids.contiguous(), topk_weights.contiguous()
    m, h = x.shape
    topk = ids.shape[1]
    pairs, intermediate = m * topk, q1.shape[-1]
    if m == 0:
        return torch.empty_like(x)
    if m <= 2:
        up = _gemv_forward(x, q1, s1, z1, ids, pairs, 1, topk, True)
        down = _gemv_forward(up, q2, s2, z2, ids, pairs, 2, topk)
    else:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        sorted_ids, expert_ids, padded = moe_align_block_size(
            ids, 16, q1.shape[0], pad_sorted_ids=True
        )
        up = x.new_empty((pairs, intermediate))
        down = x.new_empty((pairs, h))
        for inp, q, s, z, out, is_up in (
            (x, q1, s1, z1, up, True),
            (up, q2, s2, z2, down, False),
        ):
            n, k = q.shape[-1], q.shape[-2] * 16
            _grouped_gemm[(sorted_ids.numel() // 16, triton.cdiv(n, 64))](
                inp,
                q,
                s,
                z,
                sorted_ids,
                expert_ids,
                padded,
                out,
                pairs,
                n,
                k,
                topk,
                is_up,
                num_warps=4,
            )
    y = torch.empty_like(x)
    _combine[(m, triton.cdiv(h, 128))](
        down,
        weights,
        y,
        h,
        topk,
        triton.next_power_of_2(topk),
    )
    return y
