# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models import solar_open2
from vllm.model_executor.models.solar_open2 import (
    SolarOpen2ForCausalLM,
    SolarOpen2Model,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)


@pytest.mark.parametrize("capture_aux", [False, True])
def test_solar_dspark_captures_post_layer_residual_before_final_norm(
    monkeypatch, capture_aux
):
    """The Sionic draft consumes layers 3/15/27/39/47 before final RMSNorm."""
    model = SolarOpen2Model.__new__(SolarOpen2Model)
    torch.nn.Module.__init__(model)
    target = SolarOpen2ForCausalLM.__new__(SolarOpen2ForCausalLM)
    torch.nn.Module.__init__(target)
    target.model = model
    model.start_layer, model.end_layer = 0, 48
    model.layers = [
        Mock(return_value=(torch.full((2, 4), float(i)), torch.ones(2, 4)))
        for i in range(48)
    ]
    model.norm = Mock(return_value=(torch.zeros(2, 4), None))
    draft_config = SimpleNamespace(
        dflash_config={"target_layer_ids": [3, 15, 27, 39, 47]}
    )
    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=draft_config)
    )
    if capture_aux:
        target.set_aux_hidden_state_layers(
            get_eagle3_aux_layers_from_config(spec_config)
        )
    monkeypatch.setattr(
        solar_open2,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    result = model.forward(
        input_ids=None,
        positions=torch.arange(2),
        inputs_embeds=torch.zeros(2, 4),
    )
    if capture_aux:
        output, aux = result
        assert len(aux) == 5
        for tap, expected in zip(aux, [4, 16, 28, 40, 48]):
            torch.testing.assert_close(tap, torch.full((2, 4), float(expected)))
    else:
        output = result
    torch.testing.assert_close(output, torch.zeros(2, 4))


@pytest.mark.parametrize("num_spec", [0, 7])
def test_solar_layer_and_model_allocate_same_speculative_cache(num_spec):
    """The per-layer allocation must include the draft convolution history."""
    layer = SimpleNamespace(
        tp_size=2, num_heads=64, head_dim=128, conv_size=4, num_spec=num_spec
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                linear_attn_config={
                    "num_heads": 64,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                }
            )
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec),
    )
    layer_shape = solar_open2.SolarOpen2KimiDeltaAttention.get_state_shape(layer)
    model_shape = SolarOpen2ForCausalLM.get_mamba_state_shape_from_config(config)
    assert layer_shape == model_shape
    conv_shape, recurrent_shape = layer_shape
    assert sorted(conv_shape) == [3 + num_spec, 3 * 32 * 128]
    assert recurrent_shape == (32, 128, 128)
