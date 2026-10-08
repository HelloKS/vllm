# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif-specific KV alignment, window eviction and prefix reuse."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_kv_cache_utils import _grouping_config, make_request
from vllm.model_executor.models.motif_kv_cache import MotifKVCacheSpecMixin
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_groups,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowSpec

pytestmark = pytest.mark.cpu_test


class _SpecLayer:
    def __init__(self, spec):
        self.spec = spec

    def get_kv_cache_spec(self, config):
        return replace(self.spec, block_size=config.cache_config.block_size)


class _MotifLayer(MotifKVCacheSpecMixin, _SpecLayer):
    pass


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _motif_cache_specs(block_size, mla_dtype, tp_size=2):
    mla = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=mla_dtype,
    )
    swa = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=16 // tp_size,
        head_size=192,
        head_size_v=128,
        dtype=torch.bfloat16,
        sliding_window=129,
    )
    config = _grouping_config()
    config.cache_config.block_size = block_size
    layers = {f"layer.{i}": _MotifLayer(mla if i % 4 == 0 else swa) for i in range(53)}
    config.compilation_config = SimpleNamespace(static_forward_context=layers)
    return {name: layer.get_kv_cache_spec(config) for name, layer in layers.items()}


def test_motif_alignment_ignores_other_models_and_refreshes_block_size():
    mla = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=192,
        head_size_v=128,
        dtype=torch.bfloat16,
        sliding_window=129,
    )
    config = _grouping_config()
    config.cache_config.block_size = 16
    local = _MotifLayer(swa)
    unrelated = _SpecLayer(replace(mla, head_size=9999))
    config.compilation_config = SimpleNamespace(
        static_forward_context={
            "motif.mla": _MotifLayer(mla),
            "motif.swa": local,
            "other.mla": unrelated,
        }
    )
    assert local.get_kv_cache_spec(config).page_size_bytes == 92160
    assert unrelated.get_kv_cache_spec(config).page_size_padded is None
    config.cache_config.block_size = 128
    aligned = local.get_kv_cache_spec(config)
    assert aligned.block_size == 128
    assert aligned.page_size_bytes == 737280
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    assert local.get_kv_cache_spec(config).page_size_padded is None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    del config.compilation_config.static_forward_context["motif.mla"]
    assert local.get_kv_cache_spec(config).page_size_padded is None


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("mla_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_motif_mla_swa_pages_preserve_window_allocation(block_size, mla_dtype, tp_size):
    specs = _motif_cache_specs(block_size, mla_dtype, tp_size)
    original_specs = specs.copy()
    groups = get_kv_cache_groups(_grouping_config(), specs)

    assert len(groups) == 4
    assert {name for group in groups for name in group.layer_names} == set(specs)
    assert specs == original_specs
    assert len({g.kv_cache_spec.page_size_bytes for g in groups}) == 1
    mla = groups[0].kv_cache_spec
    assert isinstance(mla, MLAAttentionSpec)
    assert mla.page_size_padded is None
    assert mla.block_size % block_size == 0
    assert mla.page_size_bytes == mla.block_size * mla.state_content_size_bytes
    max_original_page = max(s.page_size_bytes for s in specs.values())
    assert (
        max_original_page
        <= mla.page_size_bytes
        < (max_original_page + specs["layer.0"].page_size_bytes)
    )

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(max_model_len=131072),
        max_in_flight_tokens=2048,
    )
    for group in groups[1:]:
        swa = group.kv_cache_spec
        assert isinstance(swa, SlidingWindowSpec)
        assert swa.block_size == block_size
        assert swa.sliding_window == 129
        config.model_config.max_model_len = 131072
        window_bytes = swa.max_memory_usage_bytes(config)
        config.model_config.max_model_len *= 2
        assert swa.max_memory_usage_bytes(config) == window_bytes
        assert window_bytes < swa.page_size_bytes * (131072 // block_size)


@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("mla_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("enable_caching", [False, True])
def test_motif_swa_recycles_blocks_and_reuses_prefix(
    block_size, mla_dtype, enable_caching
):
    config = _grouping_config()
    config.attention_config = SimpleNamespace(hisparse_config=None)
    groups = get_kv_cache_groups(config, _motif_cache_specs(block_size, mla_dtype))
    mla = groups[0].kv_cache_spec
    cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
        config, groups, available_memory=mla.page_size_bytes * 14 * 256
    )
    # The worker's views must include SWA padding but keep MLA rows contiguous.
    for tensor, group in zip(cache_config.kv_cache_tensors, groups):
        assert tensor.block_stride == group.kv_cache_spec.page_size_bytes
        assert tensor.layer_stride == tensor.block_stride * cache_config.num_blocks

    prompt_length = 20 * mla.block_size
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([cache_config]),
        max_model_len=prompt_length + 32,
        scheduler_block_size=mla.block_size,
        hash_block_size=block_size,
        max_in_flight_tokens=512,
        enable_caching=enable_caching,
    )
    tokens = list(range(prompt_length + 1))
    request = make_request("fill", tokens, block_size=block_size, hash_fn=sha256)
    while request.num_computed_tokens < prompt_length:
        # Chunks need not align with the larger MLA block or prefix-hit boundary.
        chunk_size = min(512, prompt_length - request.num_computed_tokens)
        assert manager.allocate_slots(request, chunk_size) is not None
        request.num_computed_tokens += chunk_size
        manager.remove_skipped_blocks(request.request_id, request.num_computed_tokens)
        held = manager.get_blocks(request.request_id).blocks
        assert all(not b.is_null for b in held[0])
        for blocks in held[1:]:
            assert sum(not b.is_null for b in blocks) <= (128 + block_size - 1) // (
                block_size
            )

    assert len(held[0]) == 20
    assert all(any(b.is_null for b in blocks) for blocks in held[1:])
    manager.free(request)
    replay = make_request("replay", tokens, block_size=block_size, hash_fn=sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(replay)
    if enable_caching:
        assert num_computed == prompt_length
        assert (
            manager.allocate_slots(replay, 1, num_computed, computed_blocks) is not None
        )
        replay.num_computed_tokens = num_computed + 1
        for token in range(16):
            replay.append_output_token_ids(token)
            assert manager.allocate_slots(replay, 1) is not None
            replay.num_computed_tokens += 1
        manager.remove_skipped_blocks(replay.request_id, replay.num_computed_tokens)
        for blocks in manager.get_blocks(replay.request_id).blocks[1:]:
            assert sum(not b.is_null for b in blocks) <= 128 // block_size + 1
    else:
        assert num_computed == 0
