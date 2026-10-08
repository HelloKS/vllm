# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation and sharding helpers independent of GPU kernel imports."""

import torch


def allocate_nvfp4_expert_tensors(num_experts, hidden_size, intermediate_size):
    tensors = {}
    for name, n, k in (
        ("w13_weight", 2 * intermediate_size, hidden_size),
        ("w2_weight", hidden_size, intermediate_size),
    ):
        if n % 128 or k % 64:
            raise ValueError("Motif NVFP4 requires N divisible by 128 and K by 64")
        tensors[name] = torch.empty(num_experts, n, k // 2, dtype=torch.uint8)
        tensors[name + "_scale"] = torch.full(
            (num_experts, n, k // 16), torch.nan, dtype=torch.float8_e4m3fn
        )
        tensors[name + "_scale_2"] = torch.zeros(num_experts, dtype=torch.float32)
    return tensors


def load_motif_kv_b_shard(param, source, attention):
    """Expand GQA per query head directly into this TP rank's packed layout."""
    axis = getattr(param, "output_dim", None)
    if axis is None or source.ndim == 0:
        return False
    nq, nkv = attention.num_heads, attention.num_kv_heads
    if nq == nkv:
        return False
    linear = attention.kv_b_proj
    expected_global = param.shape[axis] * linear.tp_size
    if source.shape[axis] == expected_global:
        return False
    group = nq // nkv
    if (
        source.shape[axis] * group != expected_global
        or source.shape[axis] % nkv
        or nq % linear.tp_size
    ):
        raise ValueError("Motif kv_b_proj layout cannot preserve GQA head boundaries")
    head_rows = source.shape[axis] // nkv
    local_heads = nq // linear.tp_size
    for head in range(local_heads):
        src_head = (linear.tp_rank * local_heads + head) // group
        param.data.narrow(axis, head * head_rows, head_rows).copy_(
            source.narrow(axis, src_head * head_rows, head_rows)
        )
    return True


def log_motif_load_memory(stage, layer):
    """Optional per-worker load diagnostics; CUDA and RSS may overlap on GB10."""
    import os

    if os.getenv("VLLM_MOTIF_LOAD_PROFILE") != "1":
        return
    import psutil

    from vllm.logger import init_logger

    stats = {
        "rss": psutil.Process().memory_info().rss,
        "system_available": psutil.virtual_memory().available,
    }
    if torch.cuda.is_available():
        stats.update(
            allocated=torch.accelerator.memory_allocated(),
            reserved=torch.accelerator.memory_reserved(),
            peak_allocated=torch.accelerator.max_memory_allocated(),
        )
    init_logger(__name__).info(
        "MOTIF_LOAD %s %s bytes=%s", stage, getattr(layer, "layer_name", ""), stats
    )
