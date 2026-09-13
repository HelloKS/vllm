# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig


class SolarOpen2Config(PretrainedConfig):
    """Configuration for Solar Open 2 models served by vLLM."""

    model_type = "solar_open2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 196608,
        hidden_size: int = 4096,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 64,
        head_dim: int = 128,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1_048_576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        moe_intermediate_size: int = 1280,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_routed_experts: int = 320,
        routed_scaling_factor: float = 1.0,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        first_k_dense_replace: int = 0,
        norm_topk_prob: bool = True,
        use_qk_norm: bool = False,
        use_rope: bool = False,
        gqa_interval: int = 4,
        gqa_layers: list[int] | None = None,
        use_gqa_gate: bool = True,
        use_gqa_gate_bias: bool = False,
        linear_attn_config: dict | None = None,
        kda_use_full_proj: bool = False,
        kda_gate_lower_bound: float = -5.0,
        kda_allow_neg_eigval: bool = True,
        layer_types: list[str] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if gqa_interval <= 0:
            raise ValueError("gqa_interval must be greater than zero")

        if gqa_layers is not None:
            if len(set(gqa_layers)) != len(gqa_layers) or any(
                layer < 0 or layer >= num_hidden_layers for layer in gqa_layers
            ):
                raise ValueError(
                    "gqa_layers must contain unique layer indices within the model"
                )
            gqa_layers = list(gqa_layers)

        if layer_types is None:
            full_attention_layers = (
                set(gqa_layers)
                if gqa_layers is not None
                else {
                    layer
                    for layer in range(num_hidden_layers)
                    if (layer + 1) % gqa_interval == 0
                }
            )
            layer_types = [
                "full_attention"
                if layer in full_attention_layers
                else "linear_attention"
                for layer in range(num_hidden_layers)
            ]
        else:
            if len(layer_types) != num_hidden_layers or any(
                layer_type not in {"full_attention", "linear_attention"}
                for layer_type in layer_types
            ):
                raise ValueError(
                    "layer_types must contain one valid attention type per layer"
                )
            layer_types = list(layer_types)
            full_attention_layers = {
                layer
                for layer, layer_type in enumerate(layer_types)
                if layer_type == "full_attention"
            }
            if gqa_layers is None:
                gqa_layers = sorted(full_attention_layers)
            elif set(gqa_layers) != full_attention_layers:
                raise ValueError(
                    "gqa_layers and layer_types must describe the same layers"
                )

        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_theta = kwargs.pop("rope_theta", 10000.0)
        if rope_parameters is None:
            rope_parameters = dict(rope_scaling or {})
            if "type" in rope_parameters and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters.pop("type")
            rope_parameters.setdefault("rope_type", "default")
            rope_parameters.setdefault("rope_theta", rope_theta)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = 1 if n_group is None else n_group
        self.topk_group = 1 if topk_group is None else topk_group
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob

        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.gqa_interval = gqa_interval
        self.gqa_layers = gqa_layers
        self.use_gqa_gate = use_gqa_gate
        self.use_gqa_gate_bias = use_gqa_gate_bias
        self.linear_attn_config = linear_attn_config or {
            "short_conv_kernel_size": 4,
            "head_dim": head_dim,
            "num_heads": num_attention_heads,
            "num_kv_heads": None,
        }
        self.kda_use_full_proj = kda_use_full_proj
        self.kda_gate_lower_bound = kda_gate_lower_bound
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.layer_types = layer_types


__all__ = ["SolarOpen2Config"]
