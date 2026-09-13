# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.solar_open2 import (
    _DEFAULT_LEADING_NEWLINE_IDS,
    DEFAULT_REASONING_BUDGET,
    DISABLE_EXTRA_ARG,
    REASONING_BUDGET_ENV,
    REASONING_BUDGET_EXTRA_ARG,
    THINK_LEADING_FORBIDDEN_IDS_ENV,
    SolarOpen2TemplateLogitsProcessor,
    SolarOpen2TokenFSMEnforcer,
    SolarOpen2TokenIds,
    _normalize_reasoning_budget,
    _resolve_token_ids_from_tokenizer_dirs,
    resolve_solar_open2_reasoning_budget_default,
    resolve_solar_open2_think_leading_forbidden_ids,
)

IDS = SolarOpen2TokenIds(
    im_start=128,
    im_end=129,
    think_start=130,
    think_end=131,
    im_content=132,
    tool_start=133,
    tool_end=134,
    tool_call_start=135,
    tool_call_end=136,
    tool_arg_start=137,
    tool_arg_value=138,
    tool_arg_end=139,
    tool_response_start=140,
    tool_response_end=141,
)
REGULAR_TOKEN = 99
VOCAB_SIZE = 160


def _logits() -> torch.Tensor:
    return torch.arange(VOCAB_SIZE, dtype=torch.float32)


def _apply(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    has_structured_outputs: bool = False,
) -> torch.Tensor:
    enforcer = SolarOpen2TokenFSMEnforcer(
        prompt_token_ids,
        IDS,
        has_structured_outputs=has_structured_outputs,
    )
    return enforcer(prompt_token_ids, output_token_ids, _logits())


def _is_masked(logits: torch.Tensor, token_id: int | None) -> bool:
    assert token_id is not None
    return logits[token_id].item() == float("-inf")


def _is_available(logits: torch.Tensor, token_id: int | None) -> bool:
    assert token_id is not None
    return logits[token_id].item() != float("-inf")


def test_open_reasoning_prompt_allows_only_think_end_special():
    logits = _apply([IDS.think_start], [])

    assert _is_available(logits, IDS.think_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_end)
    assert _is_masked(logits, IDS.think_start)
    assert _is_masked(logits, IDS.tool_call_start)
    assert _is_masked(logits, IDS.tool_arg_start)
    assert _is_masked(logits, IDS.tool_response_start)


def test_empty_reasoning_transitions_to_content_state():
    logits = _apply([IDS.think_start], [IDS.think_end])

    assert _is_available(logits, IDS.tool_call_start)
    # Fresh CONTENT (no content token yet): ending the turn is forbidden
    # until the model produces at least one content token or a tool call.
    assert _is_masked(logits, IDS.im_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_start)
    assert _is_masked(logits, IDS.im_content)
    assert _is_masked(logits, IDS.think_start)
    assert _is_masked(logits, IDS.think_end)
    assert _is_masked(logits, IDS.tool_start)
    assert _is_masked(logits, IDS.tool_end)
    assert _is_masked(logits, IDS.tool_arg_value)


def test_prompt_with_closed_think_block_starts_in_content_state():
    logits = _apply([IDS.think_start, IDS.think_end], [])

    assert _is_available(logits, IDS.tool_call_start)
    # Empty-content guard applies from the start for prompt-closed pairs.
    assert _is_masked(logits, IDS.im_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.think_start)
    assert _is_masked(logits, IDS.think_end)


def test_structured_outputs_do_not_mask_content_logits():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [],
        has_structured_outputs=True,
    )

    for token_id in (
        IDS.think_start,
        IDS.think_end,
        IDS.im_start,
        IDS.im_end,
        IDS.tool_call_start,
        IDS.tool_arg_value,
        REGULAR_TOKEN,
    ):
        assert _is_available(logits, token_id)


def test_tool_call_begin_requires_a_regular_function_name_token():
    logits = _apply([IDS.think_start, IDS.think_end], [IDS.tool_call_start])

    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_end)
    assert _is_masked(logits, IDS.tool_arg_start)
    assert _is_masked(logits, IDS.tool_call_end)


def test_tool_call_name_allows_args_or_call_end_specials():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [IDS.tool_call_start, REGULAR_TOKEN],
    )

    assert _is_available(logits, IDS.tool_arg_start)
    assert _is_available(logits, IDS.tool_call_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.tool_arg_value)
    assert _is_masked(logits, IDS.think_end)


def test_tool_arg_name_allows_only_value_marker_special():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_arg_start, REGULAR_TOKEN],
    )

    assert _is_available(logits, IDS.tool_arg_value)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.tool_arg_start)
    assert _is_masked(logits, IDS.tool_call_end)


def test_tool_arg_end_allows_separator_text_or_next_marker():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [
            IDS.tool_call_start,
            REGULAR_TOKEN,
            IDS.tool_arg_start,
            REGULAR_TOKEN,
            IDS.tool_arg_value,
            REGULAR_TOKEN,
            IDS.tool_arg_end,
        ],
    )

    assert _is_available(logits, IDS.tool_arg_start)
    assert _is_available(logits, IDS.tool_call_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.tool_arg_value)


def test_tool_call_end_allows_regular_stop_text_or_another_tool_call():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_call_end],
    )

    assert _is_available(logits, IDS.tool_call_start)
    assert _is_available(logits, IDS.im_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.tool_arg_start)
    assert _is_masked(logits, IDS.think_end)
    assert _is_masked(logits, IDS.tool_response_end)


def test_incremental_state_resets_to_prompt_state_when_output_shrinks():
    enforcer = SolarOpen2TokenFSMEnforcer([IDS.think_start], IDS)

    logits = enforcer([IDS.think_start], [IDS.think_end], _logits())
    assert _is_masked(logits, IDS.think_end)
    logits = enforcer([IDS.think_start], [], _logits())

    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, IDS.tool_call_start)


def test_tokenizer_json_resolution_reads_added_tokens(tmp_path: Path):
    (tmp_path / "tokenizer_config.json").write_text(
        """
        {
          "added_tokens_decoder": {
            "128": {"content": "<|im:start|>"},
            "129": {"content": "<|im:end|>"},
            "130": {"content": "<|think:start|>"},
            "131": {"content": "<|think:end|>"},
            "132": {"content": "<|im:content|>"},
            "133": {"content": "<|tool:start|>"},
            "134": {"content": "<|tool:end|>"},
            "135": {"content": "<|tool_call:start|>"}
          }
        }
        """,
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text(
        """
        {
          "added_tokens": [
            {"id": 136, "content": "<|tool_call:end|>"},
            {"id": 137, "content": "<|tool_arg:start|>"},
            {"id": 138, "content": "<|tool_arg:value|>"},
            {"id": 139, "content": "<|tool_arg:end|>"},
            {"id": 140, "content": "<|tool_response:start|>"},
            {"id": 141, "content": "<|tool_response:end|>"}
          ]
        }
        """,
        encoding="utf-8",
    )

    assert _resolve_token_ids_from_tokenizer_dirs([tmp_path]) == {
        "im_start": 128,
        "im_end": 129,
        "think_start": 130,
        "think_end": 131,
        "im_content": 132,
        "tool_start": 133,
        "tool_end": 134,
        "tool_call_start": 135,
        "tool_call_end": 136,
        "tool_arg_start": 137,
        "tool_arg_value": 138,
        "tool_arg_end": 139,
        "tool_response_start": 140,
        "tool_response_end": 141,
    }


def _validate(extra_args):
    SolarOpen2TemplateLogitsProcessor.validate_params(
        SamplingParams(extra_args=extra_args)
    )


def test_validate_params_accepts_integer_disable_flag():
    # The per-request disable flag is an integer (non-zero disables; 0 or
    # omitted keeps the processor on). int transports natively over the HTTP
    # vllm_xargs field. None / {} / 0 / non-zero / bool are all valid, and an
    # unrelated extra_arg must not trip the disable validation.
    _validate(None)
    _validate({})
    _validate({DISABLE_EXTRA_ARG: 0})
    _validate({DISABLE_EXTRA_ARG: 1})
    _validate({DISABLE_EXTRA_ARG: True})
    _validate({"other": "x"})


def test_validate_params_rejects_non_integer_disable_flag():
    for bad in ("1", 1.5, [1]):
        with pytest.raises(ValueError):
            _validate({DISABLE_EXTRA_ARG: bad})


# ── Non-sentinel EOS enforcement (template-illegal turn endings) ──────────────

EOS_ID = 2  # e.g. <|endoftext|> — listed in generation_config eos_token_id


def _apply_with_eos(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
) -> torch.Tensor:
    enforcer = SolarOpen2TokenFSMEnforcer(
        prompt_token_ids,
        IDS,
        extra_eos_token_ids=frozenset({EOS_ID}),
    )
    return enforcer(prompt_token_ids, output_token_ids, _logits())


def test_reasoning_masks_non_sentinel_eos():
    # Open think block in prompt -> REASONING: the model must not be able to
    # end its turn before emitting <|think:end|>.
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], [])
    assert _is_masked(logits, EOS_ID)
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, IDS.im_end)


def test_content_allows_non_sentinel_eos():
    # CONTENT with at least one content token -> the turn may legally end,
    # so EOS stays available (no behavior change for successful turns).
    logits = _apply_with_eos(
        [IDS.im_content, IDS.think_start], [IDS.think_end, REGULAR_TOKEN]
    )
    assert _is_available(logits, EOS_ID)
    assert _is_available(logits, IDS.im_end)


def test_tool_call_states_mask_non_sentinel_eos():
    # Mid tool-call: ending the turn would leave an unterminated call.
    output = [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_arg_start]
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], output)
    assert _is_masked(logits, EOS_ID)


def test_tool_call_end_allows_non_sentinel_eos():
    output = [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_call_end]
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], output)
    assert _is_available(logits, EOS_ID)
    assert _is_available(logits, IDS.im_end)


def test_no_extra_eos_preserves_legacy_behavior():
    # Without resolved extra EOS ids the enforcer behaves exactly as before.
    logits = _apply([IDS.im_content, IDS.think_start], [])
    assert _is_available(logits, EOS_ID)


def test_resolve_extra_eos_ids_from_generation_config(tmp_path: Path):
    from vllm.v1.sample.logits_processor.solar_open2 import (
        resolve_solar_open2_extra_eos_ids,
    )

    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id": [2, 129]}', encoding="utf-8"
    )

    class _ModelConfig:
        tokenizer = str(tmp_path)
        model = str(tmp_path)
        hf_config_path = None
        hf_config = None

    class _VllmConfig:
        model_config = _ModelConfig()

    extra = resolve_solar_open2_extra_eos_ids(_VllmConfig(), IDS)
    # 129 is the <|im:end|> sentinel -> excluded; only the bare EOS remains.
    assert extra == frozenset({2})


def test_resolve_extra_eos_ids_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from vllm.v1.sample.logits_processor.solar_open2 import (
        EOS_IDS_ENV,
        resolve_solar_open2_extra_eos_ids,
    )

    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id": [2]}', encoding="utf-8"
    )
    monkeypatch.setenv(EOS_IDS_ENV, "[5, 129]")

    class _ModelConfig:
        tokenizer = str(tmp_path)
        model = str(tmp_path)
        hf_config_path = None
        hf_config = None

    class _VllmConfig:
        model_config = _ModelConfig()

    extra = resolve_solar_open2_extra_eos_ids(_VllmConfig(), IDS)
    assert extra == frozenset({5})


# ── Empty-content guard: the turn must produce content before it may end ─────


def test_fresh_content_masks_turn_end_until_progress():
    # Right after <|think:end|>: no content yet -> both <|im:end|> and bare
    # EOS are masked; regular tokens and tool calls remain available.
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], [IDS.think_end])
    assert _is_masked(logits, IDS.im_end)
    assert _is_masked(logits, EOS_ID)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.tool_call_start)


def test_content_progress_reenables_turn_end():
    logits = _apply_with_eos(
        [IDS.im_content, IDS.think_start], [IDS.think_end, REGULAR_TOKEN]
    )
    assert _is_available(logits, IDS.im_end)
    assert _is_available(logits, EOS_ID)


def test_completed_tool_call_counts_as_content_progress():
    # think_end -> tool call -> back in CONTENT: the completed call is
    # sufficient content, so the turn may end without extra text.
    output = [
        IDS.think_end,
        IDS.tool_call_start,
        REGULAR_TOKEN,
        IDS.tool_call_end,
        REGULAR_TOKEN,
    ]
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], output)
    assert _is_available(logits, IDS.im_end)
    assert _is_available(logits, EOS_ID)


# ── Batched apply parity (vectorized masking) ─────────────────────────────────


def _make_processor(
    extra_eos: frozenset[int] = frozenset({EOS_ID}),
) -> SolarOpen2TemplateLogitsProcessor:
    # Bypass __init__ (it resolves token ids from a real model dir); wire the
    # fields the masking path uses directly.
    proc = object.__new__(SolarOpen2TemplateLogitsProcessor)
    proc.req_info = {}
    proc.token_ids = IDS
    proc.extra_eos_token_ids = extra_eos
    # Mirror the real __init__ contract. The batched apply() path reads leading
    # ids off each factory (not the processor), so this is for completeness /
    # future tests that go through new_req_logits_processor.
    proc.think_leading_forbidden_ids = frozenset({NEWLINE_ID})
    return proc


def _factory(has_structured_outputs: bool = False):
    from vllm.v1.sample.logits_processor.solar_open2 import (
        _SolarOpen2RequestFactory,
    )

    return _SolarOpen2RequestFactory(
        IDS,
        has_structured_outputs=has_structured_outputs,
        extra_eos_token_ids=frozenset({EOS_ID}),
    )


_BATCH_SCENARIOS: list[tuple[list[int], list[int], bool]] = [
    # (prompt, output, has_structured_outputs)
    ([IDS.think_start], [], False),  # REASONING
    ([IDS.think_start], [IDS.think_end], False),  # CONTENT fresh
    ([IDS.think_start], [IDS.think_end, REGULAR_TOKEN], False),  # CONTENT progress
    ([IDS.think_start], [IDS.think_end, IDS.tool_call_start], False),
    (
        [IDS.think_start],
        [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_arg_start],
        False,
    ),
    (
        [IDS.think_start],
        [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_call_end],
        False,
    ),
    ([IDS.think_start], [IDS.think_end, REGULAR_TOKEN], True),  # structured: no mask
    ([IDS.think_start], [], True),  # structured but REASONING: masked
]


def _batch_logits(num_rows: int) -> torch.Tensor:
    return torch.arange(num_rows * VOCAB_SIZE, dtype=torch.float32).reshape(
        num_rows, VOCAB_SIZE
    )


def test_batched_apply_matches_per_row_path():
    from functools import partial

    proc = _make_processor()
    outputs = []
    for req_idx, (prompt, output, structured) in enumerate(_BATCH_SCENARIOS):
        output = list(output)
        outputs.append(output)
        factory = _factory(structured)
        proc.req_info[req_idx] = partial(factory, prompt, output)

    batched = proc.apply(_batch_logits(len(_BATCH_SCENARIOS)))

    reference = _batch_logits(len(_BATCH_SCENARIOS))
    for req_idx, (prompt, output, structured) in enumerate(_BATCH_SCENARIOS):
        factory = _factory(structured)
        reference[req_idx] = factory(prompt, list(output), reference[req_idx])

    assert torch.equal(batched, reference)


def test_batched_apply_tracks_incremental_output_growth():
    from functools import partial

    proc = _make_processor()
    output: list[int] = []
    factory = _factory()
    proc.req_info[0] = partial(factory, [IDS.think_start], output)

    # Step 1: REASONING — only think_end among sentinels stays available.
    step1 = proc.apply(_batch_logits(1))
    assert _is_masked(step1[0], IDS.tool_call_start)
    assert _is_masked(step1[0], EOS_ID)
    assert _is_available(step1[0], IDS.think_end)

    # The adapter keeps a live reference to output ids; appending tokens
    # must advance the FSM on the next apply call.
    output.extend([IDS.think_end, REGULAR_TOKEN])
    step2 = proc.apply(_batch_logits(1))
    assert _is_available(step2[0], IDS.im_end)
    assert _is_available(step2[0], EOS_ID)
    assert _is_masked(step2[0], IDS.think_start)


def test_batched_apply_groups_rows_sharing_state():
    from functools import partial

    proc = _make_processor()
    # Two REASONING rows + one CONTENT-progress row -> two mask groups.
    scenarios = [
        ([IDS.think_start], []),
        ([IDS.think_start], []),
        ([IDS.think_start], [IDS.think_end, REGULAR_TOKEN]),
    ]
    for req_idx, (prompt, output) in enumerate(scenarios):
        proc.req_info[req_idx] = partial(_factory(), prompt, list(output))

    result = proc.apply(_batch_logits(3))
    for row in (0, 1):
        assert _is_masked(result[row], IDS.im_end)
        assert _is_masked(result[row], EOS_ID)
        assert _is_available(result[row], IDS.think_end)
    assert _is_available(result[2], IDS.im_end)
    assert _is_available(result[2], EOS_ID)
    assert _is_available(result[2], IDS.tool_call_start)


# ── Reasoning-token budget: cap reasoning, then force <|think:end|> ───────────


def _apply_budget(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    reasoning_budget: int | None,
    extra_eos: bool = False,
) -> torch.Tensor:
    enforcer = SolarOpen2TokenFSMEnforcer(
        prompt_token_ids,
        IDS,
        reasoning_budget=reasoning_budget,
        extra_eos_token_ids=frozenset({EOS_ID}) if extra_eos else frozenset(),
    )
    return enforcer(prompt_token_ids, output_token_ids, _logits())


def test_reasoning_budget_below_cap_allows_normal_reasoning():
    # 2 of a 3-token budget consumed: regular reasoning text and <|think:end|>
    # remain available; nothing is force-emitted yet.
    logits = _apply_budget(
        [IDS.think_start], [REGULAR_TOKEN, REGULAR_TOKEN], reasoning_budget=3
    )
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, IDS.im_end)


def test_reasoning_budget_at_cap_forces_think_end():
    # 3rd of a 3-token budget consumed: the next step must be <|think:end|>;
    # every other token (regular text and all sentinels) is masked.
    logits = _apply_budget(
        [IDS.think_start],
        [REGULAR_TOKEN, REGULAR_TOKEN, REGULAR_TOKEN],
        reasoning_budget=3,
    )
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_end)
    assert _is_masked(logits, IDS.tool_call_start)


def test_reasoning_budget_one_forces_after_single_token():
    logits = _apply_budget([IDS.think_start], [REGULAR_TOKEN], reasoning_budget=1)
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, REGULAR_TOKEN)


def test_reasoning_budget_none_is_unbounded():
    # Legacy behavior: without a budget, reasoning text stays available no
    # matter how many reasoning tokens have been emitted.
    logits = _apply_budget(
        [IDS.think_start], [REGULAR_TOKEN] * 50, reasoning_budget=None
    )
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)


def test_reasoning_budget_zero_disables_cap():
    # 0 normalizes to "no cap" so callers can opt out of a server default.
    logits = _apply_budget([IDS.think_start], [REGULAR_TOKEN] * 5, reasoning_budget=0)
    assert _is_available(logits, REGULAR_TOKEN)


def test_reasoning_budget_does_not_apply_in_content():
    # After <|think:end|> the FSM is in CONTENT; the budget never force-closes
    # there. Regular content tokens stay available (fresh-content guard still
    # masks <|im:end|> until progress, independent of the budget).
    logits = _apply_budget(
        [IDS.think_start], [REGULAR_TOKEN, IDS.think_end], reasoning_budget=1
    )
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_end)


def test_reasoning_budget_resets_when_output_shrinks():
    # Recomputation (output shorter than last seen) resets the counter to the
    # prompt state, so the budget is re-evaluated from scratch.
    enforcer = SolarOpen2TokenFSMEnforcer([IDS.think_start], IDS, reasoning_budget=3)
    forced = enforcer(
        [IDS.think_start],
        [REGULAR_TOKEN, REGULAR_TOKEN, REGULAR_TOKEN],
        _logits(),
    )
    assert _is_masked(forced, REGULAR_TOKEN)  # at cap

    rewound = enforcer([IDS.think_start], [REGULAR_TOKEN], _logits())
    assert _is_available(rewound, REGULAR_TOKEN)  # counter reset, 1 < 3


def test_reasoning_budget_restarts_on_new_think_block():
    # A fresh <|think:start|> resets the counter mid-turn. Here the prompt opens
    # a (later closed) block; the model reopens reasoning, and the budget counts
    # only tokens of the new block.
    logits = _apply_budget(
        [IDS.think_start, IDS.think_end],
        [REGULAR_TOKEN, IDS.think_start, REGULAR_TOKEN],
        reasoning_budget=2,
    )
    # New block has 1 of 2 tokens -> still below cap.
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)


def test_reasoning_budget_then_empty_content_guard_chain():
    # End-to-end: budget truncates reasoning, the model emits the forced
    # <|think:end|>, and the empty-content guard then forbids ending the turn
    # until real content appears (both <|im:end|> and bare EOS masked).
    forced = _apply_budget(
        [IDS.im_content, IDS.think_start],
        [REGULAR_TOKEN, REGULAR_TOKEN],
        reasoning_budget=2,
        extra_eos=True,
    )
    assert _is_available(forced, IDS.think_end)
    assert _is_masked(forced, REGULAR_TOKEN)

    after_close = _apply_budget(
        [IDS.im_content, IDS.think_start],
        [REGULAR_TOKEN, REGULAR_TOKEN, IDS.think_end],
        reasoning_budget=2,
        extra_eos=True,
    )
    assert _is_masked(after_close, IDS.im_end)
    assert _is_masked(after_close, EOS_ID)
    assert _is_available(after_close, REGULAR_TOKEN)


def test_reasoning_budget_unenforceable_without_think_end_id():
    # If <|think:end|> is unresolved we cannot force a close; the budget is a
    # no-op rather than masking the whole vocab into a dead end.
    ids_no_end = SolarOpen2TokenIds(think_start=130, think_end=None, im_end=129)
    enforcer = SolarOpen2TokenFSMEnforcer([130], ids_no_end, reasoning_budget=1)
    logits = enforcer([130], [REGULAR_TOKEN, REGULAR_TOKEN], _logits())
    assert _is_available(logits, REGULAR_TOKEN)


def test_reasoning_budget_with_structured_outputs_still_caps_reasoning():
    # Structured outputs only bypass masking in CONTENT; reasoning is still
    # FSM-governed, so the budget force-closes <|think:end|> as usual.
    enforcer = SolarOpen2TokenFSMEnforcer(
        [IDS.think_start], IDS, has_structured_outputs=True, reasoning_budget=2
    )
    logits = enforcer([IDS.think_start], [REGULAR_TOKEN, REGULAR_TOKEN], _logits())
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, REGULAR_TOKEN)


def test_normalize_reasoning_budget_values():
    assert _normalize_reasoning_budget(None) is None
    assert _normalize_reasoning_budget(0) is None
    assert _normalize_reasoning_budget(5) == 5
    for bad in (True, False, "5", 1.5, [1]):
        with pytest.raises(ValueError):
            _normalize_reasoning_budget(bad)
    with pytest.raises(ValueError):
        _normalize_reasoning_budget(-1)


def test_validate_params_accepts_and_rejects_reasoning_budget():
    SolarOpen2TemplateLogitsProcessor.validate_params(
        SamplingParams(extra_args={REASONING_BUDGET_EXTRA_ARG: 0})
    )
    SolarOpen2TemplateLogitsProcessor.validate_params(
        SamplingParams(extra_args={REASONING_BUDGET_EXTRA_ARG: 4096})
    )
    for bad in ("4096", 1.5, -1, True):
        with pytest.raises(ValueError):
            SolarOpen2TemplateLogitsProcessor.validate_params(
                SamplingParams(extra_args={REASONING_BUDGET_EXTRA_ARG: bad})
            )


def test_resolve_reasoning_budget_default_from_env(monkeypatch: pytest.MonkeyPatch):
    # Unset => the 128K default cap (not unbounded).
    assert DEFAULT_REASONING_BUDGET == 128 * 1024
    monkeypatch.delenv(REASONING_BUDGET_ENV, raising=False)
    assert resolve_solar_open2_reasoning_budget_default() == DEFAULT_REASONING_BUDGET

    monkeypatch.setenv(REASONING_BUDGET_ENV, "2048")
    assert resolve_solar_open2_reasoning_budget_default() == 2048

    # Explicit 0 disables the cap server-wide.
    monkeypatch.setenv(REASONING_BUDGET_ENV, "0")
    assert resolve_solar_open2_reasoning_budget_default() is None

    monkeypatch.setenv(REASONING_BUDGET_ENV, "not-an-int")
    with pytest.raises(ValueError):
        resolve_solar_open2_reasoning_budget_default()


# ── Batched apply: reasoning-budget force path ───────────────────────────────


def _budget_factory(reasoning_budget: int | None):
    from vllm.v1.sample.logits_processor.solar_open2 import (
        _SolarOpen2RequestFactory,
    )

    return _SolarOpen2RequestFactory(
        IDS,
        has_structured_outputs=False,
        extra_eos_token_ids=frozenset({EOS_ID}),
        reasoning_budget=reasoning_budget,
    )


def test_batched_apply_forces_think_end_for_budget_exhausted_rows():
    from functools import partial

    proc = _make_processor()
    # Row 0: budget=2 exhausted (2 reasoning tokens) -> force <|think:end|>.
    # Row 1: no budget, still REASONING -> standard mask.
    proc.req_info[0] = partial(
        _budget_factory(2), [IDS.think_start], [REGULAR_TOKEN, REGULAR_TOKEN]
    )
    proc.req_info[1] = partial(
        _budget_factory(None), [IDS.think_start], [REGULAR_TOKEN]
    )

    result = proc.apply(_batch_logits(2))
    # Forced row: only think_end survives; regular text and sentinels masked.
    assert _is_available(result[0], IDS.think_end)
    assert _is_masked(result[0], REGULAR_TOKEN)
    assert _is_masked(result[0], IDS.im_end)
    assert _is_masked(result[0], EOS_ID)
    # Normal REASONING row: think_end and regular text stay available.
    assert _is_available(result[1], IDS.think_end)
    assert _is_available(result[1], REGULAR_TOKEN)
    assert _is_masked(result[1], IDS.im_end)


def test_batched_apply_budget_matches_per_row_path():
    from functools import partial

    scenarios: list[tuple[list[int], list[int], int | None]] = [
        ([IDS.think_start], [REGULAR_TOKEN, REGULAR_TOKEN], 2),  # forced
        ([IDS.think_start], [REGULAR_TOKEN], 2),  # below cap
        ([IDS.think_start], [REGULAR_TOKEN] * 4, None),  # unbounded
    ]
    proc = _make_processor()
    for req_idx, (prompt, output, budget) in enumerate(scenarios):
        proc.req_info[req_idx] = partial(_budget_factory(budget), prompt, list(output))
    batched = proc.apply(_batch_logits(len(scenarios)))

    reference = _batch_logits(len(scenarios))
    for req_idx, (prompt, output, budget) in enumerate(scenarios):
        reference[req_idx] = _budget_factory(budget)(
            prompt, list(output), reference[req_idx]
        )
    assert torch.equal(batched, reference)


def test_batched_apply_budget_force_falls_back_when_think_end_out_of_range():
    from functools import partial

    # Budget exhausted, but the vocab is too small to contain think_end (131):
    # the batched force path must fall back to normal REASONING masking, exactly
    # like the per-row __call__ does (no silently-unmasked row).
    proc = _make_processor()
    proc.req_info[0] = partial(_budget_factory(1), [IDS.think_start], [REGULAR_TOKEN])
    small_vocab = 130  # excludes think_end=131
    batched = proc.apply(
        torch.arange(small_vocab, dtype=torch.float32).reshape(1, small_vocab)
    )

    reference = torch.arange(small_vocab, dtype=torch.float32).reshape(1, small_vocab)
    reference[0] = _budget_factory(1)([IDS.think_start], [REGULAR_TOKEN], reference[0])
    assert torch.equal(batched, reference)
    # REASONING fallback masks im_end; regular text stays available.
    assert _is_masked(batched[0], IDS.im_end)
    assert _is_available(batched[0], REGULAR_TOKEN)


# ── Leading-newline ban: forbid "\n\n" right after <|think:start|> ────────────
# A leading "\n\n" flips the model into a second-person, user-facing register
# inside its reasoning block, leaking user-directed phrasing. Only the first
# reasoning token (right after <|think:start|>) is constrained.

NEWLINE_ID = 150  # stand-in for the "\n\n" token id (< VOCAB_SIZE, not a sentinel)


def _apply_leading(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    leading_ids: frozenset[int] = frozenset({NEWLINE_ID}),
) -> torch.Tensor:
    enforcer = SolarOpen2TokenFSMEnforcer(
        prompt_token_ids,
        IDS,
        leading_forbidden_ids=leading_ids,
    )
    return enforcer(prompt_token_ids, output_token_ids, _logits())


def test_leading_newline_masked_right_after_think_start():
    # Prompt ends with <|think:start|> -> the first generated token is the
    # leading reasoning token: the newline is masked while ordinary reasoning
    # text and <|think:end|> stay available (normal REASONING mask preserved).
    logits = _apply_leading([IDS.think_start], [])
    assert _is_masked(logits, NEWLINE_ID)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, IDS.im_end)


def test_leading_newline_allowed_after_first_reasoning_token():
    # One reasoning token already emitted -> past the leading position, so the
    # newline is available again (only the *leading* newline triggers the leak).
    logits = _apply_leading([IDS.think_start], [REGULAR_TOKEN])
    assert _is_available(logits, NEWLINE_ID)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)


def test_leading_newline_masked_when_model_emits_think_start():
    # <|think:start|> generated mid-stream (not in the prompt): the token right
    # after it is still the leading position.
    logits = _apply_leading([IDS.im_content], [IDS.think_start])
    assert _is_masked(logits, NEWLINE_ID)
    assert _is_available(logits, REGULAR_TOKEN)


def test_leading_newline_not_masked_mid_prompt_reasoning():
    # Prompt opens reasoning AND already contains a reasoning token: the next
    # token is a continuation, not the leading position -> newline available.
    logits = _apply_leading([IDS.think_start, REGULAR_TOKEN], [])
    assert _is_available(logits, NEWLINE_ID)


def test_leading_newline_no_ban_without_config():
    # Default (no leading ids configured): behavior is unchanged, so the newline
    # stays available even right after <|think:start|>.
    logits = _apply_leading([IDS.think_start], [], leading_ids=frozenset())
    assert _is_available(logits, NEWLINE_ID)


def test_leading_newline_reset_on_output_shrink():
    # Rewind (output shorter than last seen) restores the leading flag from the
    # prompt so the ban re-applies on recomputation.
    enforcer = SolarOpen2TokenFSMEnforcer(
        [IDS.think_start], IDS, leading_forbidden_ids=frozenset({NEWLINE_ID})
    )
    advanced = enforcer([IDS.think_start], [REGULAR_TOKEN], _logits())
    assert _is_available(advanced, NEWLINE_ID)  # past the leading position
    rewound = enforcer([IDS.think_start], [], _logits())
    assert _is_masked(rewound, NEWLINE_ID)  # leading position restored


def test_leading_newline_bypass_closed_with_single_and_double():
    # Banning both the single ("\n") and double ("\n\n") newline ids at the
    # leading position closes the two-token bypass: the model can neither emit
    # "\n\n" directly nor start a "\n" + "\n" run.
    single_nl, double_nl = 150, 151
    logits = _apply_leading(
        [IDS.think_start], [], leading_ids=frozenset({single_nl, double_nl})
    )
    assert _is_masked(logits, single_nl)
    assert _is_masked(logits, double_nl)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)


def test_batched_apply_masks_leading_newline():
    from functools import partial

    from vllm.v1.sample.logits_processor.solar_open2 import _SolarOpen2RequestFactory

    def leading_factory():
        return _SolarOpen2RequestFactory(
            IDS,
            has_structured_outputs=False,
            extra_eos_token_ids=frozenset({EOS_ID}),
            leading_forbidden_ids=frozenset({NEWLINE_ID}),
        )

    proc = _make_processor()
    # Row 0: leading reasoning (prompt ends with think_start) -> newline masked.
    # Row 1: one reasoning token in -> newline available (not leading). The two
    # rows therefore land in distinct mask groups.
    proc.req_info[0] = partial(leading_factory(), [IDS.think_start], [])
    proc.req_info[1] = partial(leading_factory(), [IDS.think_start], [REGULAR_TOKEN])

    result = proc.apply(_batch_logits(2))
    assert _is_masked(result[0], NEWLINE_ID)
    assert _is_available(result[0], IDS.think_end)
    assert _is_available(result[1], NEWLINE_ID)

    # Batched masking must match the per-row __call__ path exactly.
    reference = _batch_logits(2)
    reference[0] = leading_factory()([IDS.think_start], [], reference[0])
    reference[1] = leading_factory()([IDS.think_start], [REGULAR_TOKEN], reference[1])
    assert torch.equal(result, reference)


def _leading_vllm_config(tokenizer_dir: str | None):
    class _ModelConfig:
        tokenizer = tokenizer_dir
        model = tokenizer_dir
        hf_config_path = None
        hf_config = None

    class _VllmConfig:
        model_config = _ModelConfig()

    return _VllmConfig()


def test_resolve_think_leading_forbidden_ids_from_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Every pure newline-run token ("Ċ", "ĊĊ", "ĊĊĊ") is resolved from the vocab;
    # mixed tokens (a trailing newline after text) are NOT, so the two-single-"\n"
    # bypass is closed by default while ordinary tokens stay available.
    monkeypatch.delenv(THINK_LEADING_FORBIDDEN_IDS_ENV, raising=False)
    (tmp_path / "tokenizer.json").write_text(
        '{"model": {"vocab": {'
        '"\\u010a": 4294, "\\u010a\\u010a": 4372, "\\u010a\\u010a\\u010a": 7183, '
        '"the\\u010a": 5000, "good": 6000'
        "}}}",
        encoding="utf-8",
    )
    resolved = resolve_solar_open2_think_leading_forbidden_ids(
        _leading_vllm_config(str(tmp_path))
    )
    assert resolved == frozenset({4294, 4372, 7183})


def test_resolve_think_leading_forbidden_ids_verbatim_newline_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A non-byte-level tokenizer that stores newlines verbatim ("\n", "\n\n") is
    # also covered.
    monkeypatch.delenv(THINK_LEADING_FORBIDDEN_IDS_ENV, raising=False)
    (tmp_path / "tokenizer.json").write_text(
        '{"model": {"vocab": {"\\n": 10, "\\n\\n": 11, "x": 12}}}',
        encoding="utf-8",
    )
    resolved = resolve_solar_open2_think_leading_forbidden_ids(
        _leading_vllm_config(str(tmp_path))
    )
    assert resolved == frozenset({10, 11})


def test_resolve_think_leading_forbidden_ids_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    # No tokenizer dir and no env override -> the shipped-tokenizer default
    # ("\n" and "\n\n").
    monkeypatch.delenv(THINK_LEADING_FORBIDDEN_IDS_ENV, raising=False)
    resolved = resolve_solar_open2_think_leading_forbidden_ids(
        _leading_vllm_config(None)
    )
    assert resolved == _DEFAULT_LEADING_NEWLINE_IDS


def test_resolve_think_leading_forbidden_ids_env_override(
    monkeypatch: pytest.MonkeyPatch,
):
    # Env override wins over vocab/default; "[]" disables the ban; malformed
    # values fail loudly.
    monkeypatch.setenv(THINK_LEADING_FORBIDDEN_IDS_ENV, "[4372, 4294]")
    assert resolve_solar_open2_think_leading_forbidden_ids(
        _leading_vllm_config(None)
    ) == frozenset({4372, 4294})

    monkeypatch.setenv(THINK_LEADING_FORBIDDEN_IDS_ENV, "[]")
    assert (
        resolve_solar_open2_think_leading_forbidden_ids(_leading_vllm_config(None))
        == frozenset()
    )

    monkeypatch.setenv(THINK_LEADING_FORBIDDEN_IDS_ENV, "not-json")
    with pytest.raises(ValueError):
        resolve_solar_open2_think_leading_forbidden_ids(_leading_vllm_config(None))
