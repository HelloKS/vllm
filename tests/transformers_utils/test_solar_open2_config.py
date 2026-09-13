# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from transformers import AutoConfig

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs import SolarOpen2Config


@pytest.fixture
def solar_open2_config_dict():
    return {
        "model_type": "solar_open2",
        "architectures": ["SolarOpen2ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 48,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 1_048_576,
        "rope_theta": 10000,
        "gqa_interval": 3,
        "gqa_layers": list(range(0, 48, 4)),
        "n_routed_experts": 320,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "linear_attn_config": {
            "short_conv_kernel_size": 4,
            "head_dim": 128,
            "num_heads": 64,
            "num_kv_heads": None,
        },
    }


def test_get_config_without_remote_code(tmp_path, solar_open2_config_dict):
    (tmp_path / "config.json").write_text(json.dumps(solar_open2_config_dict))

    config = get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config, SolarOpen2Config)
    assert config.gqa_layers == list(range(0, 48, 4))
    assert config.layer_types.count("full_attention") == 12
    assert config.layer_types[0] == "full_attention"
    assert config.layer_types[1] == "linear_attention"
    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 10000,
    }

    assert isinstance(AutoConfig.from_pretrained(tmp_path), SolarOpen2Config)


def test_gqa_interval_derives_layer_types():
    config = SolarOpen2Config(num_hidden_layers=8, gqa_interval=4)

    assert config.gqa_layers is None
    assert config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]


def test_rejects_inconsistent_attention_patterns():
    with pytest.raises(ValueError, match="must describe the same layers"):
        SolarOpen2Config(
            num_hidden_layers=4,
            gqa_layers=[0],
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
