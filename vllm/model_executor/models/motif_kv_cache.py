# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align Motif's SWA pages with its latent MLA cache before core grouping."""

from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec, SlidingWindowSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MotifKVCacheSpecMixin:
    def _get_unaligned_kv_cache_spec(self, vllm_config: "VllmConfig") -> KVCacheSpec:
        spec = super().get_kv_cache_spec(vllm_config)  # type: ignore[misc]
        assert spec is not None, "Motif decoder attention must have a KV cache spec"
        return spec

    def get_kv_cache_spec(self, vllm_config: "VllmConfig") -> KVCacheSpec:
        spec = self._get_unaligned_kv_cache_spec(vllm_config)
        if (
            type(spec) is not SlidingWindowSpec
            or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
        ):
            return spec

        # Specs are queried after model loading, when block size and per-layer
        # cache dtypes are resolved. Read base specs to avoid recursive alignment.
        layers = vllm_config.compilation_config.static_forward_context.values()
        specs = [
            layer._get_unaligned_kv_cache_spec(vllm_config)
            for layer in layers
            if isinstance(layer, MotifKVCacheSpecMixin)
        ]
        if not all(
            type(s) in (MLAAttentionSpec, SlidingWindowSpec)
            and getattr(s, "page_size_padded", None) is None
            for s in specs
        ):
            return spec
        mla_pages = {
            s.page_size_bytes for s in specs if isinstance(s, MLAAttentionSpec)
        }
        if len(mla_pages) != 1:
            return spec

        page_size = round_up(max(s.page_size_bytes for s in specs), mla_pages.pop())
        # Core grouping can grow unpadded MLA blocks by an integer factor.
        # Only SWA needs padding; its block size and eviction window stay intact.
        return replace(spec, page_size_padded=page_size)
