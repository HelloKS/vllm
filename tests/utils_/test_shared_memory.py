# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-memory limits must be byte counts for the requested device.

Keep these tests independent of the CUDA profiling dependencies in test_mem_utils.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.utils import mem_utils


def test_shared_memory_uses_optin_bytes_per_device(monkeypatch):
    properties = [
        SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376),
        SimpleNamespace(
            multi_processor_count=132, shared_memory_per_block_optin=232448
        ),
    ]
    getter = Mock(side_effect=lambda gpu: properties[gpu])
    monkeypatch.setattr(torch.cuda, "get_device_properties", getter)
    monkeypatch.setattr(
        mem_utils, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    mem_utils.get_max_shared_memory_bytes.cache_clear()
    try:
        assert mem_utils.get_max_shared_memory_bytes(0) == 101376
        assert mem_utils.get_max_shared_memory_bytes(1) == 232448
        assert mem_utils.get_max_shared_memory_bytes(0) == 101376
        assert getter.call_count == 2
    finally:
        mem_utils.get_max_shared_memory_bytes.cache_clear()


@pytest.mark.parametrize("limit", [0, -1])
def test_shared_memory_rejects_invalid_limit(monkeypatch, limit):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda gpu: SimpleNamespace(shared_memory_per_block_optin=limit),
    )
    monkeypatch.setattr(
        mem_utils, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    mem_utils.get_max_shared_memory_bytes.cache_clear()
    try:
        with pytest.raises(AssertionError, match="max_shared_mem"):
            mem_utils.get_max_shared_memory_bytes(0)
    finally:
        mem_utils.get_max_shared_memory_bytes.cache_clear()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Requires the native CUDA extension and an NVIDIA GPU",
)
@pytest.mark.parametrize("shared_first", [False, True])
def test_native_device_attributes_do_not_share_one_cached_value(shared_first):
    from vllm import _custom_ops as ops

    for device in range(torch.accelerator.device_count()):
        props = torch.cuda.get_device_properties(device)
        queries = [
            # cudaDevAttrMultiProcessorCount = 16.
            (
                partial(ops.get_device_attribute, 16, device),
                props.multi_processor_count,
            ),
            (
                partial(ops.get_max_shared_memory_per_block_device_attribute, device),
                props.shared_memory_per_block_optin,
            ),
        ]
        if shared_first:
            queries.reverse()
        for query, expected in queries:
            assert query() == expected
