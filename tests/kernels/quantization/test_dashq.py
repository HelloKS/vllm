# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CUDA kernels, including changing routes during graph replay."""

import pytest
import torch

pytest.importorskip("triton")

from vllm.model_executor.layers.quantization.utils.dashq_ops import (
    dashq_linear,
    dashq_moe,
)
from vllm.model_executor.layers.quantization.utils.dashq_triton import (
    dashq_linear as triton_linear,
)
from vllm.model_executor.layers.quantization.utils.dashq_triton import (
    dashq_moe as triton_moe,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def weights(n, k, experts=None):
    lead = () if experts is None else (experts,)
    codes = torch.randint(0, 4, (*lead, k, n), device="cuda", dtype=torch.int32)
    words = torch.zeros((*lead, k // 16, n), device="cuda", dtype=torch.int32)
    for bit in range(16):
        words |= codes[..., bit::16, :] << (2 * bit)
    scale = (torch.rand((*lead, k // 32, n), device="cuda") * 0.02 + 0.01).bfloat16()
    zero = (torch.rand_like(scale.float()) * 3 - 0.25).bfloat16()
    ref = codes.float() - zero.float().repeat_interleave(32, -2)
    ref *= scale.float().repeat_interleave(32, -2)
    return words, scale, zero, ref


def assert_error(actual, reference):
    assert torch.isfinite(actual).all()
    relative = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative < 1e-2, relative.item()


@pytest.mark.parametrize("m", [1, 2, 31, 32, 33, 128])
@pytest.mark.parametrize("n,k", [(130, 96), (256, 2048), (2560, 2048)])
def test_linear_matches_dequantized_reference(m, n, k):
    torch.manual_seed(14)
    q, s, z, ref = weights(n, k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=x.dtype) * 0.01
    expected = (x.float() @ ref).bfloat16() if m == 1 else x @ ref.bfloat16()
    expected = expected + bias
    assert_error(dashq_linear(x, q, s, z, bias), expected)


@pytest.mark.parametrize("m", [1, 2, 33])
def test_linear_inductor_compilation(m):
    """Inductor must receive tile constexprs on GEMV and both GEMM paths."""
    q, s, z, ref = weights(130, 96)
    x = torch.randn((m, 96), device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(triton_linear, backend="inductor", fullgraph=True)
    expected = (x.float() @ ref).bfloat16() if m == 1 else x @ ref.bfloat16()
    assert_error(compiled(x, q, s, z), expected)


def reference_moe(x, w1, w2, route, ids):
    y = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = ids[row, slot]
            up = (
                (x[row].float() @ w1[e]).bfloat16()
                if x.shape[0] <= 2
                else x[row] @ w1[e].bfloat16()
            )
            up = up.relu().square()
            down = (
                (up.float() @ w2[e]).bfloat16()
                if x.shape[0] <= 2
                else up @ w2[e].bfloat16()
            )
            y[row] += down.float() * route[row, slot]
    return y.bfloat16()


@pytest.mark.parametrize("m", [1, 2, 33, 128])
def test_moe_empty_and_shared_experts(m):
    torch.manual_seed(81)
    q1, s1, z1, w1 = weights(128, 64, 4)
    q2, s2, z2, w2 = weights(64, 128, 4)
    x = torch.randn((m, 64), device="cuda", dtype=torch.bfloat16)
    ids = (
        torch.tensor([0, 2], device="cuda", dtype=torch.int32).expand(m, 2).contiguous()
    )
    route = torch.softmax(torch.randn((m, 2), device="cuda"), -1)
    actual = dashq_moe(x, q1, s1, z1, q2, s2, z2, route, ids)
    assert_error(actual, reference_moe(x, w1, w2, route, ids))


@pytest.mark.parametrize("m", [1, 2, 33])
def test_moe_inductor_compilation(m):
    """Cover explicit tile arguments in routed GEMV, grouped GEMM and combine."""
    q1, s1, z1, w1 = weights(128, 64, 4)
    q2, s2, z2, w2 = weights(64, 128, 4)
    x = torch.randn((m, 64), device="cuda", dtype=torch.bfloat16)
    ids = torch.rand((m, 4), device="cuda").argsort(-1)[:, :2].int()
    route = torch.softmax(torch.randn((m, 2), device="cuda"), -1)
    compiled = torch.compile(triton_moe, backend="inductor", fullgraph=True)
    actual = compiled(x, q1, s1, z1, q2, s2, z2, route, ids)
    assert_error(actual, reference_moe(x, w1, w2, route, ids))


def test_linear_dynamic_inductor_compilation():
    """A strict dynamic warmup at 32768 tokens must also serve small batches."""
    q, s, z, ref = weights(32, 64)
    compiled = torch.compile(
        dashq_linear, backend="inductor", fullgraph=True, dynamic=True
    )
    for m in (32768, 2, 31, 32, 33, 128, 1):
        x = torch.randn((m, 64), device="cuda", dtype=torch.bfloat16)
        if m == 32768:
            torch._dynamo.mark_dynamic(x, 0, min=1, max=32768)
        expected = (x.float() @ ref).bfloat16() if m == 1 else x @ ref.bfloat16()
        assert_error(compiled(x, q, s, z), expected)


def test_moe_dynamic_inductor_compilation():
    """Keep both dispatch branches opaque while routes and token counts change."""
    q1, s1, z1, w1 = weights(128, 64, 4)
    q2, s2, z2, w2 = weights(64, 128, 4)
    compiled = torch.compile(
        dashq_moe, backend="inductor", fullgraph=True, dynamic=True
    )
    for m in (128, 2, 33, 1):
        x = torch.randn((m, 64), device="cuda", dtype=torch.bfloat16)
        ids = torch.rand((m, 4), device="cuda").argsort(-1)[:, :2].int()
        route = torch.softmax(torch.randn((m, 2), device="cuda"), -1)
        if m == 128:
            for tensor in (x, ids, route):
                torch._dynamo.mark_dynamic(tensor, 0, min=1, max=32768)
        actual = compiled(x, q1, s1, z1, q2, s2, z2, route, ids)
        assert_error(actual, reference_moe(x, w1, w2, route, ids))


@pytest.mark.parametrize("m", [1, 2])
def test_linear_cuda_graph_replay(m):
    q, s, z, _ = weights(256, 2048)
    x = torch.randn((m, 2048), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            dashq_linear(x, q, s, z)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dashq_linear(x, q, s, z)
    for _ in range(10):
        x.normal_()
        expected = dashq_linear(x, q, s, z)
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 2, 33])
def test_moe_top22_varying_routes(m):
    torch.manual_seed(61)
    q1, s1, z1, w1 = weights(128, 64, 24)
    q2, s2, z2, w2 = weights(64, 128, 24)
    x = torch.randn((m, 64), device="cuda", dtype=torch.bfloat16)
    ids = torch.rand((m, 24), device="cuda").argsort(-1)[:, :22].int()
    route = torch.softmax(torch.randn((m, 22), device="cuda"), -1)
    actual = dashq_moe(x, q1, s1, z1, q2, s2, z2, route, ids)
    assert_error(actual, reference_moe(x, w1, w2, route, ids))


@pytest.mark.parametrize("m", [1, 2])
def test_cuda_graph_replay_changes_routing_without_stale_scratch(m):
    torch.manual_seed(32)
    q1, s1, z1, _ = weights(128, 64, 4)
    q2, s2, z2, _ = weights(64, 128, 4)
    x = torch.randn((m, 64), device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros((m, 2), device="cuda", dtype=torch.int32)
    route = torch.full((m, 2), 0.5, device="cuda")

    def run():
        return dashq_moe(x, q1, s1, z1, q2, s2, z2, route, ids)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for step in range(100):
        x.normal_()
        ids.fill_(step % 4)
        expected = run()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
