# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120


def test_nope_forward_uses_native_d512_triton_kernel(monkeypatch):
    impl = object.__new__(sm120.FlashInferMLASparseSM120Impl)
    impl.num_heads = 2
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 256
    impl.qk_rope_head_dim = 0
    impl.scale = 0.5
    impl.kv_scale_format = "arbitrary_fp32"
    impl.need_to_return_lse_for_decode = False
    impl.topk_indices_buffer = torch.tensor(
        [[3, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32
    )
    impl._workspace_buffer = None

    valid_lens = torch.tensor([1, 0], dtype=torch.int32)

    def fake_convert(*args, **kwargs):
        assert kwargs["return_valid_counts"] is True
        return impl.topk_indices_buffer.clone(), valid_lens

    captured = {}

    def fake_triton(**kwargs):
        captured.update(kwargs)
        out = torch.full((2, 2, 512), 7, dtype=torch.bfloat16)
        lse = torch.zeros(2, 2, dtype=torch.float32)
        return out, lse

    monkeypatch.setattr(sm120, "triton_convert_req_index_to_global_index", fake_convert)
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.triton_mla_sparse_glm."
        "glm_fp8ds_nope_sparse_mla",
        fake_triton,
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
    assert captured["q"].shape == (2, 2, 512)
    assert torch.equal(captured["q"], q)
    assert torch.equal(captured["lens"], valid_lens)
    assert captured["slot_ids"].shape == (2, 4)
    assert captured["block_size"] == 64
    assert torch.count_nonzero(out[0] - 7) == 0
    assert torch.count_nonzero(out[1] - 7) == 0


def test_kpool_tail_uses_second_fixed_topk_partition(monkeypatch):
    impl = object.__new__(sm120.FlashInferMLASparseSM120Impl)
    impl.num_heads = 1
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 512
    impl.qk_rope_head_dim = 64
    impl.scale = 0.5
    impl.kv_scale_format = "arbitrary_fp32"
    impl.need_to_return_lse_for_decode = True
    impl.topk_indices_buffer = torch.stack(
        (
            torch.arange(2176, dtype=torch.int32),
            torch.arange(3000, 5176, dtype=torch.int32),
        )
    )
    impl._workspace_buffer = torch.empty(1, dtype=torch.uint8)

    def fake_convert(*args, **kwargs):
        return impl.topk_indices_buffer.clone(), torch.tensor(
            [2051, 2050], dtype=torch.int32
        )

    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        assert kwargs["block_tables"].is_contiguous()
        partition = len(calls)
        kwargs["out"].fill_(2 if partition == 1 else 8)
        # Partition softmax masses are 2 and 1, respectively.
        lse = torch.full((2, 1), 1.0 if partition == 1 else 0.0)
        return kwargs["out"], lse

    monkeypatch.setattr(sm120, "triton_convert_req_index_to_global_index", fake_convert)
    monkeypatch.setattr(
        "vllm.utils.flashinfer.flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        fake_decode,
    )

    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        block_size=64,
    )
    q = torch.randn(2, 1, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 64, 656, dtype=torch.uint8)

    out, lse = impl.forward_mqa(q, kv_cache, metadata, layer=None)

    assert len(calls) == 2
    assert calls[0]["block_tables"].shape == (2, 1, 2048)
    assert calls[1]["block_tables"].shape == (2, 1, 2048)
    assert calls[0]["seq_lens"].tolist() == [2048, 2048]
    assert calls[1]["seq_lens"].tolist() == [3, 2]
    assert calls[1]["block_tables"][0, 0, :3].tolist() == [2048, 2049, 2050]
    torch.testing.assert_close(out.float(), torch.full_like(out.float(), 4.0))
    torch.testing.assert_close(lse, torch.log2(torch.full((2, 1), 3.0)))


def test_merge_log2_attention_partitions_matches_direct_softmax():
    logits = torch.tensor([[[1.0, 2.0, -0.5, 3.0]]], dtype=torch.float32)
    values = torch.tensor(
        [[[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [-1.0, 4.0]]]],
        dtype=torch.float32,
    )
    first_logits, second_logits = logits[..., :3], logits[..., 3:]
    first_out = torch.einsum(
        "bhn,bhnd->bhd", first_logits.softmax(-1), values[..., :3, :]
    )
    second_out = values[..., 3, :]
    first_lse = torch.logsumexp(first_logits, -1) / torch.log(torch.tensor(2.0))
    second_lse = second_logits.squeeze(-1) / torch.log(torch.tensor(2.0))

    merged_out, merged_lse = sm120._merge_log2_attention_partitions(
        first_out, first_lse, second_out, second_lse
    )
    expected_out = torch.einsum("bhn,bhnd->bhd", logits.softmax(-1), values)
    expected_lse = torch.logsumexp(logits, -1) / torch.log(torch.tensor(2.0))

    torch.testing.assert_close(merged_out, expected_out)
    torch.testing.assert_close(merged_lse, expected_lse)
