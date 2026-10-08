# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opaque DASH-Q ops preserve dynamic token counts in vLLM's compiled graph.

Batch-dependent dispatch and Triton constexpr specialization happen at runtime,
inside the op. The fake implementations propagate shapes without specializing
the token dimension. Imports of CUDA kernels are deferred to actual execution.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _linear(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.dashq_triton import (
        dashq_linear,
    )

    return dashq_linear(x, q, scale, zero, bias)


def _linear_fake(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], q.shape[-1]))


def _moe(
    x: torch.Tensor,
    q1: torch.Tensor,
    s1: torch.Tensor,
    z1: torch.Tensor,
    q2: torch.Tensor,
    s2: torch.Tensor,
    z2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.dashq_triton import dashq_moe

    return dashq_moe(x, q1, s1, z1, q2, s2, z2, topk_weights, topk_ids)


def _moe_fake(
    x: torch.Tensor,
    q1: torch.Tensor,
    s1: torch.Tensor,
    z1: torch.Tensor,
    q2: torch.Tensor,
    s2: torch.Tensor,
    z2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


direct_register_custom_op(
    "dashq_linear", _linear, fake_impl=_linear_fake, dispatch_key="CUDA"
)
direct_register_custom_op("dashq_moe", _moe, fake_impl=_moe_fake, dispatch_key="CUDA")


def dashq_linear(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.dashq_linear(x, q, scale, zero, bias)


def dashq_moe(
    x: torch.Tensor,
    q1: torch.Tensor,
    s1: torch.Tensor,
    z1: torch.Tensor,
    q2: torch.Tensor,
    s2: torch.Tensor,
    z2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.dashq_moe(x, q1, s1, z1, q2, s2, z2, topk_weights, topk_ids)
