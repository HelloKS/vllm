# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backend import MLAAttentionImpl


def test_fp8_ds_mla_nope_cache_update_pads_reserved_rope(monkeypatch):
    captured_k_pe = None

    def fake_concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    ):
        nonlocal captured_k_pe
        captured_k_pe = k_pe

    monkeypatch.setattr(ops, "concat_and_cache_mla", fake_concat_and_cache_mla)

    num_tokens = 3
    kv_c = torch.randn(num_tokens, 512, dtype=torch.bfloat16)
    k_pe = torch.empty(num_tokens, 1, 0, dtype=torch.bfloat16)
    kv_cache = torch.empty(1, 1, 656, dtype=torch.uint8)
    slot_mapping = torch.zeros(num_tokens, dtype=torch.int64)
    scale = torch.ones(1, dtype=torch.float32)

    MLAAttentionImpl.do_kv_cache_update(
        None, kv_c, k_pe, kv_cache, slot_mapping, "fp8_ds_mla", scale
    )

    assert captured_k_pe is not None
    assert captured_k_pe.shape == (num_tokens, 64)
    assert torch.count_nonzero(captured_k_pe) == 0
