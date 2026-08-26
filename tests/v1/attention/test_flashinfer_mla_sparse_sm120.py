# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120


def test_nope_forward_passes_sparse_topk_lens(monkeypatch):
    impl = object.__new__(sm120.FlashInferMLASparseSM120Impl)
    impl.num_heads = 2
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 256
    impl.qk_rope_head_dim = 0
    impl.scale = 0.5
    impl.kv_scale_format = "arbitrary_fp32"
    impl.topk_indices_buffer = torch.tensor(
        [[3, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32
    )
    impl._workspace_buffer = torch.empty(1, dtype=torch.uint8)

    valid_lens = torch.tensor([1, 0], dtype=torch.int32)

    def fake_convert(*args, **kwargs):
        assert kwargs["return_valid_counts"] is True
        return impl.topk_indices_buffer.clone(), valid_lens

    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        kwargs["out"].fill_(7)
        return kwargs["out"]

    monkeypatch.setattr(
        sm120, "triton_convert_req_index_to_global_index", fake_convert
    )
    monkeypatch.setattr(
        "vllm.utils.flashinfer.flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        fake_decode,
    )

    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        block_size=64,
        topk_tokens=2,
    )
    q = torch.randn(2, 2, 512, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 656, dtype=torch.uint8)

    out, lse = impl.forward_mqa(q, kv_cache, metadata, layer=None)

    assert lse is None
    assert captured["sparse_mla_top_k"] == 4
    assert captured["max_seq_len"] == 4
    assert captured["qk_rope_head_dim"] == 64
    assert captured["query"].shape == (2, 1, 2, 576)
    assert torch.equal(captured["query"][:, 0, :, :512], q)
    assert torch.count_nonzero(captured["query"][..., 512:]) == 0
    assert torch.equal(captured["seq_lens"], valid_lens)
    assert torch.equal(
        captured["sparse_mla_top_k_lens"],
        torch.tensor([1, 1], dtype=torch.int32),
    )
    assert captured["block_tables"][1, 0, 0] == 0
    assert torch.count_nonzero(out[0] - 7) == 0
    assert torch.count_nonzero(out[1]) == 0
