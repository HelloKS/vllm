# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif admission hooks must read the configured model runner."""

from types import SimpleNamespace

import pytest

from vllm import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("ratio", [None, 0.5])
def test_request_admission_uses_configured_runner(monkeypatch, use_v2, ratio):
    # Keep tokenization/model loading out of this admission regression while
    # exercising both validation and request-local budget resolution.
    monkeypatch.setattr(SamplingParams, "verify", lambda *args: None)
    monkeypatch.setattr(
        InputProcessor,
        "_build_logits_processors_params_validator",
        lambda self: lambda params: None,
    )
    monkeypatch.setenv("VLLM_THINK_BUDGET_RATIO", "0")
    monkeypatch.setenv("VLLM_THINK_BUDGET_FORCE_STR", "")
    monkeypatch.setenv("VLLM_REP_MODE", "off")
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_config=SimpleNamespace(
            try_get_generation_config=lambda: {},
            supports_multimodal_inputs=False,
            is_diffusion=False,
            return_sampling_mask=False,
            max_model_len=128,
            runner_type="generate",
        ),
        cache_config=None,
        lora_config=None,
        scheduler_config=None,
        speculative_config=None,
        structured_outputs_config=None,
        observability_config=None,
        diffusion_config=None,
        reasoning_config=SimpleNamespace(enabled=True),
        watermark_config=None,
        _check_supports_watermarking=lambda params: False,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        ),
    )
    renderer = SimpleNamespace(
        tokenizer=None,
        _executor=None,
        get_eos_token_id=lambda: None,
        validate_token_ids=lambda ids: None,
    )
    processor = InputProcessor(config, renderer=renderer)
    params = SamplingParams(
        max_tokens=16,
        extra_args={"think_budget_ratio": ratio} if ratio is not None else None,
    )
    prompt = {"type": "token", "prompt_token_ids": [1, 2, 3]}
    if use_v2 and ratio is not None:
        with pytest.raises(
            ValueError, match="not yet supported by the V2 model runner"
        ):
            processor.process_inputs("test", prompt, params, ("generate",))
        return

    request = processor.process_inputs("test", prompt, params, ("generate",))
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.sampling_params.thinking_token_budget == (8 if ratio else None)
    assert request.sampling_params.watermarking is False
    assert params.thinking_token_budget is None
    assert params.watermarking is None
