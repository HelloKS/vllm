# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from einops import rearrange
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import (
    FusedRMSNormGated,
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)
from vllm.transformers_utils.configs.solar_open2 import SolarOpen2Config
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from ...linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from ..mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from ..ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from ..ops.gather_initial_states import gather_initial_states

logger = init_logger(__name__)


@eager_break_during_capture
def solar_open2_kda_attention(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(
        q_proj_states=q_proj_states,
        k_proj_states=k_proj_states,
        v_proj_states=v_proj_states,
        g1=g1,
        beta=beta,
        core_attn_out=core_attn_out,
    )


def solar_open2_kda_attention_fake(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="solar_open2_kda_attention",
    op_func=solar_open2_kda_attention,
    mutates_args=["q_proj_states", "k_proj_states", "v_proj_states", "core_attn_out"],
    fake_impl=solar_open2_kda_attention_fake,
)


class SolarOpen2KimiDeltaAttention(GatedDeltaNetAttention):
    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: SolarOpen2Config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        if (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.method != "dspark"
        ):
            raise NotImplementedError(
                "Solar Open 2 currently supports only DSpark speculative decoding"
            )

        # Flipped True after the first V1 profile run warms up the autotuned KDA
        # prefill kernels (see _warmup_prefill_kernels / _forward profile branch).
        self._prefill_kernels_warmed_up = False

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = kda_config["short_conv_kernel_size"]

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_a_proj",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.b_proj",
        )

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # FLA triton kernels (chunk_kda, solve_tril) need a PyTorch-backed
        # allocator for scratch memory required by triton >= 3.x autotuner.
        set_triton_allocator(current_platform.current_device())
        self.use_full_proj = config.kda_use_full_proj
        self.allow_neg_eigval = config.kda_allow_neg_eigval

        if self.use_full_proj:
            projection_size = self.head_dim * self.num_heads
            quant_config = self.quant_config
            prefix = self.prefix

            del self.f_a_proj
            del self.f_b_proj
            self.f_proj = ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_proj",
            )

            del self.g_a_proj
            del self.g_b_proj
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]

        beta = self.b_proj(hidden_states)[0].float().sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        if self.use_full_proj:
            g1 = self.f_proj(hidden_states)[0]
        else:
            g1 = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        g1 = fused_kda_gate(g1, self.A_log, self.head_dim, g_bias=self.dt_bias)
        beta = beta.unsqueeze(0)
        g1 = g1.unsqueeze(0)

        if self.use_full_proj:
            g_proj_states = self.g_proj(hidden_states)[0]
        else:
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.solar_open2_kda_attention(
            q,
            k,
            v,
            g1,
            beta,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _warmup_prefill_kernels(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        """Autotune chunked prefill during profiling, before allocating the cache."""
        if self._prefill_kernels_warmed_up:
            return
        T = 64
        if q_proj_states.size(0) < T:
            return

        try:
            q, k, v = [
                rearrange(x[:T], "n (h d) -> 1 n h d", d=self.head_dim)
                for x in (q_proj_states, k_proj_states, v_proj_states)
            ]
            *_, state_dtype = self.get_state_dtype()
            initial_state = torch.zeros(
                1,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
                device=q.device,
                dtype=state_dtype,
            )
            cu_seqlens = torch.tensor([0, T], device=q.device, dtype=torch.int32)
            # Quiesce the device before and after autotuning. The profile forward
            # leaves prior ops in flight; concurrently autotuning the KDA
            # chunked-prefill kernels across TP ranks from a non-quiescent state
            # intermittently faults under CUDA forward-compat. Bracketing the call
            # with syncs makes the warmup deterministic.
            torch.accelerator.synchronize()
            chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g1[:, :T],
                beta=beta[:, :T],
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            torch.accelerator.synchronize()
            self._prefill_kernels_warmed_up = True
        except Exception:
            logger.warning(
                "KDA prefill-kernel warmup (T=%d) failed for layer %s; the first "
                "real request may JIT-autotune and stall/crash under CUDA "
                "forward-compat.",
                T,
                self.prefix,
                exc_info=True,
            )
        finally:
            torch.accelerator.empty_cache()

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(
                q_proj_states, k_proj_states, v_proj_states, g1, beta
            )
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            return
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        m = attn_metadata_narrowed
        num_actual_tokens = m.num_actual_tokens
        projections = [
            x[:num_actual_tokens] for x in (q_proj_states, k_proj_states, v_proj_states)
        ]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_states = conv_state.split(self.local_num_heads * self.head_dim, dim=-2)
        conv_layers = (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        conv_weights = [layer.weight.squeeze(1) for layer in conv_layers]

        if m.spec_sequence_masks is not None:
            if m.num_prefills == 0 and m.num_decodes == 0:
                spec_projections = projections
                spec_g, spec_beta = g1, beta
                projections = []
            else:
                assert m.spec_token_indx is not None
                assert m.non_spec_token_indx is not None
                spec_projections = [
                    x.index_select(0, m.spec_token_indx) for x in projections
                ]
                projections = [
                    x.index_select(0, m.non_spec_token_indx) for x in projections
                ]
                spec_g = g1.index_select(1, m.spec_token_indx)
                spec_beta = beta.index_select(1, m.spec_token_indx)
                g1 = g1.index_select(1, m.non_spec_token_indx)
                beta = beta.index_select(1, m.non_spec_token_indx)

            assert m.spec_state_indices_tensor is not None
            assert m.spec_query_start_loc is not None
            assert m.num_accepted_tokens is not None
            spec_indices = m.spec_state_indices_tensor
            q, k, v = [
                rearrange(
                    causal_conv1d_update(
                        x,
                        state,
                        weight,
                        layer.bias,
                        activation="silu",
                        conv_state_indices=spec_indices[:, 0][: m.num_spec_decodes],
                        num_accepted_tokens=m.num_accepted_tokens,
                        query_start_loc=m.spec_query_start_loc,
                        max_query_len=spec_indices.size(-1),
                        validate_data=False,
                    ),
                    "n (h d) -> 1 n h d",
                    d=self.head_dim,
                )
                for x, state, weight, layer in zip(
                    spec_projections, conv_states, conv_weights, conv_layers
                )
            ]
            # Read the previous accepted state and save each candidate's state
            # so the next verification can discard a rejected suffix.
            spec_out, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=spec_g,
                beta=spec_beta,
                initial_state=recurrent_state,
                cu_seqlens=m.spec_query_start_loc[: m.num_spec_decodes + 1],
                ssm_state_indices=spec_indices,
                num_accepted_tokens=m.num_accepted_tokens,
            )
            if projections:
                core_attn_out.index_copy_(1, m.spec_token_indx, spec_out)
            else:
                # The recurrent kernel leaves graph-padding rows unwritten.
                core_attn_out[:, : m.num_spec_decode_tokens] = spec_out[
                    :, : m.num_spec_decode_tokens
                ]

        if not projections:
            return

        state_indices = m.non_spec_state_indices_tensor
        query_start_loc = m.non_spec_query_start_loc
        assert state_indices is not None
        assert query_start_loc is not None
        if m.num_prefills > 0:
            q, k, v = [
                causal_conv1d_fn(
                    x.transpose(0, 1),
                    weight,
                    layer.bias,
                    activation="silu",
                    conv_states=state,
                    has_initial_state=m.has_initial_state,
                    cache_indices=state_indices,
                    query_start_loc=query_start_loc,
                    metadata=m,
                ).transpose(0, 1)
                for x, state, weight, layer in zip(
                    projections, conv_states, conv_weights, conv_layers
                )
            ]
        else:
            q, k, v = [
                causal_conv1d_update(
                    x,
                    state,
                    weight,
                    layer.bias,
                    activation="silu",
                    conv_state_indices=state_indices[: x.size(0)],
                    validate_data=True,
                )
                for x, state, weight, layer in zip(
                    projections, conv_states, conv_weights, conv_layers
                )
            ]
        q, k, v = [
            rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in (q, k, v)
        ]

        if m.num_prefills > 0:
            assert m.has_initial_state is not None
            initial_state = gather_initial_states(
                recurrent_state, state_indices, m.has_initial_state
            )
            non_spec_out, last_recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
            )
            recurrent_state[state_indices] = last_recurrent_state
        else:
            non_spec_out, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=recurrent_state,
                cu_seqlens=query_start_loc[: m.num_decodes + 1],
                ssm_state_indices=state_indices,
            )
        if m.spec_sequence_masks is not None:
            assert m.non_spec_token_indx is not None
            core_attn_out.index_copy_(1, m.non_spec_token_indx, non_spec_out)
        else:
            num_non_spec_tokens = m.num_prefill_tokens + m.num_decode_tokens
            core_attn_out[:, :num_non_spec_tokens] = non_spec_out[
                :, :num_non_spec_tokens
            ]
