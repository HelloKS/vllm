# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.triton_mla_sparse_glm import (
    glm_fp8ds_nope_sparse_mla,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_glm_fp8ds_nope_sparse_mla_matches_torch_reference() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    block_size = 64
    num_blocks = 2
    num_tokens = 2
    num_heads = 2

    # Build an fp8_ds_mla entry with unit per-128 scales. The final 128-byte
    # physical RoPE region is deliberately random: the native NoPE kernel must
    # never consume it.
    cache = torch.empty((num_blocks, block_size, 656), dtype=torch.uint8, device=device)
    kv_fp8 = (
        torch.randn(num_blocks, block_size, 512, device=device)
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn)
    )
    cache[..., :512].copy_(kv_fp8.view(torch.uint8))
    cache[..., 512:528].view(torch.float32).fill_(1.0)
    cache[..., 528:].random_(0, 256)

    q = torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device=device)
    slot_ids = torch.tensor(
        [[0, 3, 65, -1], [7, 68, -1, -1]], dtype=torch.int32, device=device
    )
    lens = torch.tensor([3, 2], dtype=torch.int32, device=device)
    scale = 512**-0.5

    output, lse = glm_fp8ds_nope_sparse_mla(q, cache, slot_ids, lens, block_size, scale)

    expected_output = torch.zeros_like(output, dtype=torch.float32)
    expected_lse = torch.empty_like(lse)
    kv_flat = kv_fp8.reshape(-1, 512).float()
    for token in range(num_tokens):
        keys = kv_flat[slot_ids[token, : lens[token]].long()]
        logits = torch.einsum("hd,kd->hk", q[token].float(), keys) * scale
        probs = logits.softmax(dim=-1)
        expected_output[token] = torch.einsum("hk,kd->hd", probs, keys)
        expected_lse[token] = torch.logsumexp(logits, dim=-1)

    torch.testing.assert_close(output.float(), expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-3, atol=2e-3)
