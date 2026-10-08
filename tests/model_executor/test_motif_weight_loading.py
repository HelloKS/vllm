# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard packed allocation and TP head mapping without requiring CUDA."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.motif_weight_utils import (
    allocate_nvfp4_expert_tensors,
    load_motif_kv_b_shard,
)


def test_nvfp4_allocates_checkpoint_storage_without_bf16(monkeypatch):
    """The TP=2 expert allocation must never construct a BF16 weight."""
    dtypes = []
    empty = torch.empty

    def record_empty(*args, **kwargs):
        dtypes.append(kwargs.get("dtype", torch.get_default_dtype()))
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", record_empty)
    with torch.device("meta"):
        tensors = allocate_nvfp4_expert_tensors(192, 4096, 1280)
    assert torch.bfloat16 not in dtypes and torch.float16 not in dtypes
    assert tensors["w13_weight"].shape == (192, 2560, 2048)
    assert tensors["w2_weight_scale"].shape == (192, 4096, 80)
    actual = sum(t.numel() * t.element_size() for t in tensors.values())
    expected = 192 * 3 * 4096 * 1280 * 9 // 16 + 2 * 192 * 4
    assert actual == expected


def test_nvfp4_missing_global_scales_are_detectable():
    tensors = allocate_nvfp4_expert_tensors(2, 128, 128)
    assert torch.count_nonzero(tensors["w13_weight_scale_2"]) == 0
    assert tensors["w13_weight_scale"].dtype == torch.float8_e4m3fn
    assert torch.isnan(tensors["w13_weight_scale"].float()).all()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "axis,dtype,rows",
    [
        (0, torch.bfloat16, 256),
        (0, torch.float32, 2),
        (1, torch.int32, 32),
    ],
)
def test_kv_b_shard_matches_gqa_for_weight_scale_and_packed_layout(
    rank, axis, dtype, rows
):
    """Rank-local expansion must match expansion before TP slicing."""
    source = torch.arange(16 * rows * 4).reshape(16 * rows, 4).to(dtype)
    if axis == 1:
        source = source.t().contiguous()
    expanded = source.movedim(axis, 0).reshape(16, rows, -1).repeat_interleave(5, 0)
    expanded = expanded.reshape(80 * rows, -1).movedim(0, axis)
    expected = expanded.narrow(axis, rank * 40 * rows, 40 * rows)
    param = torch.nn.Parameter(torch.empty_like(expected), requires_grad=False)
    param.output_dim = axis
    attn = SimpleNamespace(
        num_heads=80,
        num_kv_heads=16,
        kv_b_proj=SimpleNamespace(tp_size=2, tp_rank=rank),
    )
    assert load_motif_kv_b_shard(param, source, attn)
    torch.testing.assert_close(param, expected)


def test_kv_b_rejects_layout_without_head_boundaries():
    param = torch.nn.Parameter(torch.empty(100, 4))
    param.output_dim = 0
    attn = SimpleNamespace(
        num_heads=80, num_kv_heads=16, kv_b_proj=SimpleNamespace(tp_size=2, tp_rank=0)
    )
    with pytest.raises(ValueError, match="head boundaries"):
        load_motif_kv_b_shard(param, torch.empty(40, 4), attn)
