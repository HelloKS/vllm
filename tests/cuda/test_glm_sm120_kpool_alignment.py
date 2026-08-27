# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability


def _config(index_kpool: int):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_kpool=index_kpool)
        )
    )


def test_sm120_kpool_alignment_uses_deepgemm_64_entry_page() -> None:
    with patch.object(
        CudaPlatform,
        "get_device_capability",
        return_value=DeviceCapability(12, 0),
    ):
        assert CudaPlatform._get_indexer_block_alignment(_config(4)) == 256


def test_sm100_kpool_alignment_keeps_generic_32_entry_page() -> None:
    with patch.object(
        CudaPlatform,
        "get_device_capability",
        return_value=DeviceCapability(10, 0),
    ):
        assert CudaPlatform._get_indexer_block_alignment(_config(4)) == 128
