# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict dynamic-shape export must not specialize the 32768-token warmup.

These tests exercise the real operator schemas and fake implementations on CPU;
they do not execute or stand in for the CUDA kernels.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.model_executor.layers.quantization.utils.dashq_ops import (
    dashq_linear,
    dashq_moe,
)


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("q", torch.zeros((4, 32), dtype=torch.int32))
        self.register_buffer("s", torch.ones((2, 32), dtype=torch.bfloat16))
        self.register_buffer("z", torch.zeros_like(self.s))
        self.register_buffer("bias", torch.zeros(32, dtype=torch.bfloat16))

    def forward(self, x):
        return dashq_linear(x, self.q, self.s, self.z, self.bias)


class MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("q", torch.zeros((4, 4, 64), dtype=torch.int32))
        self.register_buffer("s", torch.ones((4, 2, 64), dtype=torch.bfloat16))
        self.register_buffer("z", torch.zeros_like(self.s))

    def forward(self, x, route, ids):
        return dashq_moe(x, self.q, self.s, self.z, self.q, self.s, self.z, route, ids)


@pytest.mark.parametrize("kind", ["linear", "moe"])
def test_strict_dynamic_export_preserves_tokens(kind):
    tokens = torch.export.Dim("tokens", min=1, max=32768)
    model = Linear() if kind == "linear" else MoE()

    def inputs(m):
        x = torch.empty((m, 64), dtype=torch.bfloat16)
        if kind == "linear":
            return (x,)
        return x, torch.empty((m, 2)), torch.empty((m, 2), dtype=torch.int32)

    args = inputs(32768)
    exported = torch.export.export(
        model, args, dynamic_shapes=tuple({0: tokens} for _ in args), strict=True
    )
    op = getattr(torch.ops.vllm, f"dashq_{kind}").default
    assert any(n.target == op for n in exported.graph.nodes)
    with FakeTensorMode(allow_non_fake_inputs=True):
        for m in (1, 2, 31, 32, 33, 128, 32768):
            output = exported.module()(*inputs(m))
            assert output.shape == (m, 32 if kind == "linear" else 64)
            assert output.dtype == torch.bfloat16
