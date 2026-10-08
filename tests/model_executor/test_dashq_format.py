# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for actual checkpoint I/O, packing and rank-local placement.

Can also run without the GPU test conftest:
pytest --confcutdir=tests/model_executor tests/model_executor/test_dashq_format.py
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.transformers_utils import dashq


def metadata(n=18, k=128):
    return dict(
        nbits=2,
        group_size=32,
        packing="int2_packed_u32",
        scale_zero_dtype="float16",
        linear_dtype="bfloat16",
        in_features=k,
        quant_in_features=k,
        out_features=n,
        num_groups=n * (k // 32),
    )


def pack(q):
    groups = q.to(torch.int64).reshape(q.shape[0], -1, 16)
    shifts = torch.arange(16, dtype=torch.int64) * 2
    return (groups << shifts).sum(-1).to(torch.int32)


@pytest.mark.parametrize(
    "kind,parts",
    [
        ("row", None),
        ("column", None),
        ("column", [8, 4, 6]),
        ("replicated", None),
    ],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_shards_retain_codes_and_fractional_zero(
    tmp_path, monkeypatch, kind, parts, rank
):
    n, k = 18, 128
    q = torch.arange(n * k).reshape(n, k) % 4
    q[:, -1] = 3  # Exercise the sign bit of serialized int32 words.
    words = pack(q)
    s = torch.linspace(0.1, 0.5, n * k // 32).half().reshape(n, -1)
    z = torch.linspace(-0.375, 3.375, n * k // 32).half().reshape(n, -1)
    tensors = {
        "layer.W_q_packed": words.flatten(),
        "layer.scale": s.reshape(-1, 1),
        "layer.zero": z.reshape(-1, 1),
    }
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in tensors}})
    )
    # Force multiple chunks to catch output offsets and row-strided K slicing.
    monkeypatch.setattr(dashq, "CHUNK_BYTES", 64)
    slices = dashq.tp_slices(n, k, rank, 2, kind, parts)
    local_n = sum(p.n_stop - p.n_start for p in slices)
    local_k = slices[0].k_stop - slices[0].k_start
    reader = dashq.DashQTensorReader(tmp_path)
    try:
        for suffix, original, divisor, dtype in (
            ("W_q_packed", words, 16, torch.int32),
            ("scale", s, 32, torch.bfloat16),
            ("zero", z, 32, torch.bfloat16),
        ):
            # A nonzero target offset also models stacked QKV placement.
            dest = torch.full((local_k // divisor, local_n + 3), -17, dtype=dtype)
            dashq.load_matrix(
                reader, "layer", metadata(n, k), dest, suffix, slices, out_offset=3
            )
            expected = (
                torch.cat(
                    [
                        original[
                            p.n_start : p.n_stop,
                            p.k_start // divisor : p.k_stop // divisor,
                        ]
                        for p in slices
                    ]
                )
                .t()
                .to(dtype)
            )
            torch.testing.assert_close(dest[:, 3:], expected, rtol=0, atol=0)
            assert (dest[:, :3] == -17).all()
    finally:
        reader.close()


def test_mamba_splits_each_projection_not_the_flat_output():
    widths = [16384, 16384, 1024, 1024, 256]
    rank0 = dashq.tp_slices(sum(widths), 8192, 0, 2, "column", widths)
    rank1 = dashq.tp_slices(sum(widths), 8192, 1, 2, "column", widths)
    assert [p.n_stop - p.n_start for p in rank0] == [8192, 8192, 512, 512, 128]
    for left, right in zip(rank0, rank1):
        assert left.n_stop == right.n_start
        assert left.out_start == right.out_start
    assert rank0[1].n_start == 16384
    assert rank1[-1].n_stop == 35072


@pytest.mark.parametrize(
    "change",
    [
        {"nbits": 4},
        {"group_size": 64},
        {"num_groups": 1},
        {"quant_in_features": 256},
        {"scale_zero_dtype": "int32"},
        {"linear_dtype": "float16"},
        {"packing": "gptq"},
    ],
)
def test_reject_incompatible_metadata(change):
    meta = metadata() | change
    with pytest.raises(ValueError):
        dashq.validate_metadata(
            {
                "format": "dashq-packed-linear",
                "format_version": 1,
                "quantized_modules": {"layer": meta},
            }
        )


def test_reject_group_crossing_tp_partition():
    with pytest.raises(ValueError, match="group"):
        dashq.tp_slices(8, 96, 0, 2, "row")


def test_index_rejects_duplicate_tensor_names(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        dashq.DashQTensorReader(tmp_path)


def test_reader_rejects_shape_mismatch(tmp_path):
    save_file(
        {"x": torch.zeros(16, dtype=torch.int32)}, str(tmp_path / "a.safetensors")
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"a.safetensors"}}'
    )
    reader = dashq.DashQTensorReader(tmp_path)
    try:
        with pytest.raises(ValueError, match="expected"):
            reader.matrix("x", 8, 16, True)
    finally:
        reader.close()
