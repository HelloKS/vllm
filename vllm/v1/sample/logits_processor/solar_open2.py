# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from copy import copy
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
)
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsProcessor as V2LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

NEG_INF = float("-inf")
TOKEN_IDS_ENV = "SOLAR_OPEN2_TOKEN_IDS"
EOS_IDS_ENV = "SOLAR_OPEN2_EOS_TOKEN_IDS"
REASONING_BUDGET_ENV = "SOLAR_REASONING_BUDGET"
DISABLE_EXTRA_ARG = "disable_solar_open2_logits_processor"
REASONING_BUDGET_EXTRA_ARG = "solar_open2_reasoning_budget"
THINK_LEADING_FORBIDDEN_IDS_ENV = "SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS"
# Server-wide reasoning-token cap applied when SOLAR_REASONING_BUDGET is unset.
# 128K tokens (128 * 1024). Set the env to 0 to disable the cap.
DEFAULT_REASONING_BUDGET = 128 * 1024

# Byte-level (GPT-2 style) encoding of a newline run. The Solar Open2 tokenizer
# is a byte-level BPE, so each '\n' (0x0A) is stored in the vocab as 'Ċ'
# (U+010A); a run of N newlines is the single vocab token "Ċ" * N (e.g. "ĊĊ"
# == "\n\n"). A vocab token is a pure newline run iff its only character is 'Ċ'.
_NEWLINE_BYTELEVEL = "Ċ"  # 'Ċ' == byte-level "\n"
# Fallback leading-newline ids used when tokenizer metadata is unavailable.
_DEFAULT_LEADING_NEWLINE_IDS = frozenset({4294, 4372})
# Sentinel returned by ``advance_mask_ids`` when the reasoning budget is
# exhausted and the step must force ``<|think:end|>`` (mask the whole row except
# think_end) — not expressible as a small forbidden-id tuple, so the masking
# layer handles it specially.
_FORCE_THINK_END = "force_think_end"


@dataclass(frozen=True)
class SolarOpen2TokenIds:
    """Single-token control IDs for the Solar Open2 chat envelope.

    Defaults match the Solar Open2 tokenizer layout. Runtime tokenizer metadata
    and SOLAR_OPEN2_TOKEN_IDS override them.
    """

    im_start: int | None = 128
    im_end: int | None = 129
    think_start: int | None = 130
    think_end: int | None = 131
    im_content: int | None = 132
    tool_start: int | None = 133
    tool_end: int | None = 134
    tool_call_start: int | None = 135
    tool_call_end: int | None = 136
    tool_arg_start: int | None = 137
    tool_arg_value: int | None = 138
    tool_arg_end: int | None = 139
    tool_response_start: int | None = 140
    tool_response_end: int | None = 141


_TOKEN_TEXT_BY_FIELD = {
    "im_start": "<|im:start|>",
    "im_end": "<|im:end|>",
    "think_start": "<|think:start|>",
    "think_end": "<|think:end|>",
    "im_content": "<|im:content|>",
    "tool_start": "<|tool:start|>",
    "tool_end": "<|tool:end|>",
    "tool_call_start": "<|tool_call:start|>",
    "tool_call_end": "<|tool_call:end|>",
    "tool_arg_start": "<|tool_arg:start|>",
    "tool_arg_value": "<|tool_arg:value|>",
    "tool_arg_end": "<|tool_arg:end|>",
    "tool_response_start": "<|tool_response:start|>",
    "tool_response_end": "<|tool_response:end|>",
}


class SolarOpen2State(Enum):
    REASONING = "reasoning"
    CONTENT = "content"
    TOOL_CALL_BEGIN = "tool_call_begin"
    TOOL_CALL_NAME = "tool_call_name"
    TOOL_ARG_BEGIN = "tool_arg_begin"
    TOOL_ARG_NAME = "tool_arg_name"
    TOOL_ARG_VALUE_BEGIN = "tool_arg_value_begin"
    TOOL_ARG_VALUE = "tool_arg_value"
    TOOL_ARG_END = "tool_arg_end"
    TOOL_CALL_END = "tool_call_end"


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid token id")
    return int(value)


def _read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse %s for Solar Open2 token IDs: %s", path, e)
        return None


def _token_ids_from_json(data: Any) -> dict[str, int]:
    """Extract token text -> id mappings from common HF tokenizer JSON shapes."""
    token_to_id: dict[str, int] = {}
    if not isinstance(data, dict):
        return token_to_id

    added_decoder = data.get("added_tokens_decoder")
    if isinstance(added_decoder, dict):
        for raw_id, raw_token in added_decoder.items():
            if isinstance(raw_token, dict):
                content = raw_token.get("content")
            else:
                content = raw_token
            if isinstance(content, str):
                token_to_id[content] = int(raw_id)

    added_tokens = data.get("added_tokens")
    if isinstance(added_tokens, list):
        for raw_token in added_tokens:
            if not isinstance(raw_token, dict):
                continue
            content = raw_token.get("content")
            raw_id = raw_token.get("id")
            if isinstance(content, str) and raw_id is not None:
                token_to_id[content] = int(raw_id)

    model = data.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        for token, raw_id in vocab.items():
            if isinstance(token, str) and raw_id is not None:
                token_to_id[token] = int(raw_id)

    return token_to_id


def _candidate_tokenizer_dirs(vllm_config: "VllmConfig") -> list[Path]:
    model_config = vllm_config.model_config
    candidates: list[Path] = []
    for attr in ("tokenizer", "model", "hf_config_path"):
        value = getattr(model_config, attr, None)
        if not value:
            continue
        path = Path(str(value))
        if path.is_file():
            path = path.parent
        if path.is_dir() and path not in candidates:
            candidates.append(path)
    return candidates


def _resolve_token_ids_from_tokenizer_dirs(
    tokenizer_dirs: list[Path],
) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for tokenizer_dir in tokenizer_dirs:
        token_to_id: dict[str, int] = {}
        for name in (
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
        ):
            token_to_id.update(_token_ids_from_json(_read_json(tokenizer_dir / name)))
        for field_name, token_text in _TOKEN_TEXT_BY_FIELD.items():
            if field_name not in resolved and token_text in token_to_id:
                resolved[field_name] = token_to_id[token_text]
    return resolved


def _resolve_token_ids_from_env() -> dict[str, int | None]:
    raw = os.environ.get(TOKEN_IDS_ENV)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{TOKEN_IDS_ENV} must be a JSON object") from e
    if not isinstance(data, dict):
        raise ValueError(f"{TOKEN_IDS_ENV} must be a JSON object")

    by_token_text = {
        _TOKEN_TEXT_BY_FIELD[field_name]: field_name
        for field_name in _TOKEN_TEXT_BY_FIELD
    }
    resolved: dict[str, int | None] = {}
    valid_fields = {field.name for field in fields(SolarOpen2TokenIds)}
    for key, value in data.items():
        field_name = by_token_text.get(key, key)
        if field_name not in valid_fields:
            raise ValueError(f"Unknown Solar Open2 token id field: {key}")
        resolved[field_name] = _int_or_none(value)
    return resolved


def _coerce_eos_ids(value: Any) -> set[int]:
    """Normalize an ``eos_token_id`` value (int | list | None) to a set."""
    if value is None or isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        return {int(v) for v in value if isinstance(v, int) and not isinstance(v, bool)}
    return set()


def resolve_solar_open2_extra_eos_ids(
    vllm_config: "VllmConfig",
    token_ids: SolarOpen2TokenIds,
) -> frozenset[int]:
    """Resolve turn-terminating (EOS) token ids that are *not* chat sentinels.

    Solar Open2's ``generation_config.json`` declares ``eos_token_id`` as a
    list — e.g. ``[2, 129]`` = ``<|endoftext|>`` + ``<|im:end|>``. The FSM
    enforcer manages the sentinel ids (``<|im:end|>`` etc.) per state, but a
    bare EOS like ``<|endoftext|>`` is not a sentinel, so the model could end
    its turn from *any* state (e.g. mid-reasoning, before emitting
    ``<|think:end|>``). The non-streaming parser hides this template
    violation by falling back to "whole output is content", while the
    streaming parser cannot retroactively reclassify already-streamed
    reasoning deltas — yielding an empty assistant message. Masking these
    EOS ids in states where ending the turn is template-illegal removes the
    divergence at the source.

    Sources (union): model ``hf_config.eos_token_id``, every candidate dir's
    ``generation_config.json``, and the tokenizer's ``eos_token`` text mapped
    through the tokenizer files. ``SOLAR_OPEN2_EOS_TOKEN_IDS`` (JSON list of
    ints) overrides everything when set. Sentinel ids are always excluded —
    they remain state-managed by the enforcer itself.
    """
    raw_env = os.environ.get(EOS_IDS_ENV)
    if raw_env:
        try:
            data = json.loads(raw_env)
        except json.JSONDecodeError as e:
            raise ValueError(f"{EOS_IDS_ENV} must be a JSON list of ints") from e
        if not isinstance(data, list):
            raise ValueError(f"{EOS_IDS_ENV} must be a JSON list of ints")
        env_ids = {
            value for value in (_int_or_none(v) for v in data) if value is not None
        }
        return frozenset(env_ids - _sentinel_id_set(token_ids))

    eos_ids: set[int] = set()

    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    eos_ids |= _coerce_eos_ids(getattr(hf_config, "eos_token_id", None))

    eos_token_texts: set[str] = set()
    for tokenizer_dir in _candidate_tokenizer_dirs(vllm_config):
        gen_cfg = _read_json(tokenizer_dir / "generation_config.json")
        if isinstance(gen_cfg, dict):
            eos_ids |= _coerce_eos_ids(gen_cfg.get("eos_token_id"))
        for name in ("tokenizer_config.json", "special_tokens_map.json"):
            cfg = _read_json(tokenizer_dir / name)
            if not isinstance(cfg, dict):
                continue
            eos_token = cfg.get("eos_token")
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content")
            if isinstance(eos_token, str):
                eos_token_texts.add(eos_token)
        if eos_token_texts:
            token_to_id: dict[str, int] = {}
            for name in (
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
            ):
                token_to_id.update(
                    _token_ids_from_json(_read_json(tokenizer_dir / name))
                )
            for text in eos_token_texts:
                if text in token_to_id:
                    eos_ids.add(token_to_id[text])

    extra = frozenset(eos_ids - _sentinel_id_set(token_ids))
    if not extra:
        logger.warning(
            "Solar Open2 logits processor resolved no non-sentinel EOS token "
            "ids; the model may be able to end its turn mid-reasoning "
            "(template violation). Set %s to override.",
            EOS_IDS_ENV,
        )
    else:
        logger.info(
            "Solar Open2 logits processor masking non-sentinel EOS ids %s in "
            "states where ending the turn is template-illegal.",
            sorted(extra),
        )
    return extra


def _sentinel_id_set(token_ids: SolarOpen2TokenIds) -> set[int]:
    return {
        getattr(token_ids, field.name)
        for field in fields(SolarOpen2TokenIds)
        if getattr(token_ids, field.name) is not None
    }


# Per-state mask specification: (allowed sentinel field names, eos_masked).
# ``eos_masked`` mirrors the turn-end legality rule: bare EOS is forbidden
# wherever ending the turn is template-illegal — everywhere except
# CONTENT-with-progress and TOOL_CALL_END.
# CONTENT is keyed separately by the content-progress flag (see
# ``_forbidden_table``); all other states ignore the flag.
_MASK_SPEC_BY_STATE: dict[SolarOpen2State, tuple[tuple[str, ...], bool]] = {
    SolarOpen2State.REASONING: (("think_end",), True),
    SolarOpen2State.TOOL_CALL_BEGIN: ((), True),
    SolarOpen2State.TOOL_CALL_NAME: (("tool_arg_start", "tool_call_end"), True),
    SolarOpen2State.TOOL_ARG_BEGIN: ((), True),
    SolarOpen2State.TOOL_ARG_NAME: (("tool_arg_value",), True),
    SolarOpen2State.TOOL_ARG_VALUE_BEGIN: (("tool_arg_end",), True),
    SolarOpen2State.TOOL_ARG_VALUE: (("tool_arg_end",), True),
    SolarOpen2State.TOOL_ARG_END: (("tool_arg_start", "tool_call_end"), True),
    SolarOpen2State.TOOL_CALL_END: (("tool_call_start", "im_end"), False),
}
_MASK_SPEC_CONTENT: dict[bool, tuple[tuple[str, ...], bool]] = {
    # content_progress=True: the turn may legally end -> EOS stays available.
    True: (("tool_call_start", "im_end"), False),
    # Fresh CONTENT (no content yet): turn end (im_end + bare EOS) forbidden.
    False: (("tool_call_start",), True),
}


@lru_cache(maxsize=8)
def _forbidden_table(
    token_ids: SolarOpen2TokenIds,
    extra_eos_token_ids: frozenset[int],
) -> dict[tuple[SolarOpen2State, bool], tuple[int, ...]]:
    """Precompute, per (state, content_progress), the forbidden token ids.

    The forbidden set for a decoding step is fully determined by the FSM
    state (plus the content-progress flag in CONTENT), so it can be built
    once per (token_ids, extra_eos) configuration instead of re-deriving
    Python lists on every step of every request.
    """
    all_controls = {
        getattr(token_ids, field.name)
        for field in fields(SolarOpen2TokenIds)
        if getattr(token_ids, field.name) is not None
    }

    def build(allowed_fields: tuple[str, ...], eos_masked: bool) -> tuple[int, ...]:
        allowed = {
            token_id
            for token_id in (getattr(token_ids, name) for name in allowed_fields)
            if token_id is not None
        }
        forbidden = all_controls - allowed
        if eos_masked:
            forbidden |= extra_eos_token_ids
        return tuple(sorted(forbidden))

    table: dict[tuple[SolarOpen2State, bool], tuple[int, ...]] = {}
    for state, (allowed_fields, eos_masked) in _MASK_SPEC_BY_STATE.items():
        ids = build(allowed_fields, eos_masked)
        table[(state, False)] = ids
        table[(state, True)] = ids
    for progress, (allowed_fields, eos_masked) in _MASK_SPEC_CONTENT.items():
        table[(SolarOpen2State.CONTENT, progress)] = build(allowed_fields, eos_masked)
    return table


# Device-resident forbidden-index tensors, keyed by (ids, vocab_size, device).
# The id tuples come from ``_forbidden_table`` so there are only a handful of
# distinct entries per model; caching removes the per-step
# list -> tensor (H2D copy) conversion that dominated masking cost.
_MASK_TENSOR_CACHE: dict[tuple[tuple[int, ...], int, torch.device], torch.Tensor] = {}


def _mask_index_tensor(
    forbidden_ids: tuple[int, ...],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    key = (forbidden_ids, vocab_size, device)
    tensor = _MASK_TENSOR_CACHE.get(key)
    if tensor is None:
        in_vocab = [token_id for token_id in forbidden_ids if token_id < vocab_size]
        tensor = torch.tensor(in_vocab, dtype=torch.long, device=device)
        _MASK_TENSOR_CACHE[key] = tensor
    return tensor


def resolve_solar_open2_token_ids(vllm_config: "VllmConfig") -> SolarOpen2TokenIds:
    values: dict[str, int | None] = {
        field.name: getattr(SolarOpen2TokenIds(), field.name)
        for field in fields(SolarOpen2TokenIds)
    }
    file_values = _resolve_token_ids_from_tokenizer_dirs(
        _candidate_tokenizer_dirs(vllm_config)
    )
    values.update(file_values)
    values.update(_resolve_token_ids_from_env())
    token_ids = SolarOpen2TokenIds(**values)

    unresolved = [
        field.name
        for field in fields(SolarOpen2TokenIds)
        if getattr(token_ids, field.name) is None
    ]
    if unresolved:
        logger.warning(
            "Solar Open2 logits processor has unresolved token IDs: %s. "
            "Those control tokens will not be masked.",
            ", ".join(unresolved),
        )
    return token_ids


def _normalize_reasoning_budget(value: Any) -> int | None:
    """Coerce a budget value to a positive int, or ``None`` (= no cap).

    Accepts ``int`` (``bool`` rejected — a budget is a token count, not a
    toggle). ``<= 0`` collapses to ``None`` so callers can disable the cap with
    ``0`` (e.g. a per-request override of a server-wide default).
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{REASONING_BUDGET_EXTRA_ARG} must be a non-negative integer "
            "(reasoning-token count; 0 or omitted disables the cap)"
        )
    if value < 0:
        raise ValueError(
            f"{REASONING_BUDGET_EXTRA_ARG} must be non-negative, got {value}"
        )
    return value if value > 0 else None


def resolve_solar_open2_reasoning_budget_default() -> int | None:
    """Server-wide default reasoning-token budget from the environment.

    ``SOLAR_REASONING_BUDGET`` (a non-negative integer) caps how many tokens
    the model may emit inside a ``<|think:start|>`` / ``<|think:end|>`` block
    before the FSM force-emits ``<|think:end|>``. A per-request
    ``solar_open2_reasoning_budget`` xarg overrides it. When the env is unset
    the default is ``DEFAULT_REASONING_BUDGET`` (128K tokens); set the env to
    ``0`` to disable the cap server-wide.
    """
    raw = os.environ.get(REASONING_BUDGET_ENV)
    if not raw:
        return DEFAULT_REASONING_BUDGET
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"{REASONING_BUDGET_ENV} must be an integer number of tokens"
        ) from e
    return _normalize_reasoning_budget(value)


def resolve_solar_open2_think_leading_forbidden_ids(
    vllm_config: "VllmConfig",
) -> frozenset[int]:
    r"""Resolve token ids forbidden immediately after ``<|think:start|>``.

    Leading newline-run tokens can produce malformed reasoning output, so they
    are blocked only at the start of a reasoning block.

    Resolution order:
    1. ``SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS`` (JSON list of ints) overrides
       everything -- e.g. ``"[4372]"`` to forbid only ``"\n\n"``, or ``"[]"`` to
       disable the ban entirely.
    2. Every vocab token whose byte-level text is a pure newline run (``"Ċ"``,
       ``"ĊĊ"``, ...), looked up in the tokenizer files. A verbatim ``"\n"`` /
       ``"\n\n"`` (non-byte-level tokenizers) is included too.
    3. ``_DEFAULT_LEADING_NEWLINE_IDS`` as a last-resort fallback.
    """
    raw = os.environ.get(THINK_LEADING_FORBIDDEN_IDS_ENV)
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{THINK_LEADING_FORBIDDEN_IDS_ENV} must be a JSON list of ints"
            ) from e
        if not isinstance(data, list):
            raise ValueError(
                f"{THINK_LEADING_FORBIDDEN_IDS_ENV} must be a JSON list of ints"
            )
        ids = {value for value in (_int_or_none(v) for v in data) if value is not None}
        return frozenset(ids)

    for tokenizer_dir in _candidate_tokenizer_dirs(vllm_config):
        token_to_id: dict[str, int] = {}
        for name in ("tokenizer.json", "tokenizer_config.json"):
            token_to_id.update(_token_ids_from_json(_read_json(tokenizer_dir / name)))
        newline_ids = {
            token_id
            for token, token_id in token_to_id.items()
            if token and set(token) == {_NEWLINE_BYTELEVEL}
        }
        for literal in ("\n", "\n\n"):
            if literal in token_to_id:
                newline_ids.add(token_to_id[literal])
        if newline_ids:
            return frozenset(newline_ids)

    return _DEFAULT_LEADING_NEWLINE_IDS


class SolarOpen2TokenFSMEnforcer:
    """Request-level Solar Open2 token finite-state-machine enforcer.

    Masks structurally invalid single-token sentinels for the current phase.
    Optionally enforces a reasoning-token budget: once ``reasoning_budget``
    tokens have been emitted inside a ``<|think:start|>`` / ``<|think:end|>``
    block, the FSM force-emits ``<|think:end|>`` to end reasoning. ``None`` (the
    default) disables the budget — pure structural enforcement.
    """

    def __init__(
        self,
        prompt_token_ids: list[int],
        token_ids: SolarOpen2TokenIds,
        has_structured_outputs: bool = False,
        extra_eos_token_ids: frozenset[int] = frozenset(),
        reasoning_budget: int | None = None,
        leading_forbidden_ids: frozenset[int] = frozenset(),
    ) -> None:
        self.token_ids = token_ids
        self._initial_prompt_state = self._initial_state(prompt_token_ids)
        self._state = self._initial_prompt_state
        self._last_processed_len = 0
        # Whether the token immediately preceding the next decode step is
        # <|think:start|> (i.e. the upcoming token is the *leading* reasoning
        # token). Seeded from the prompt's last token so a template-appended
        # <|think:start|> is caught on the very first generated token, then kept
        # in sync as output tokens are processed.
        think_start = token_ids.think_start
        self._initial_prev_is_think_start = (
            think_start is not None
            and bool(prompt_token_ids)
            and prompt_token_ids[-1] == think_start
        )
        self._prev_was_think_start = self._initial_prev_is_think_start
        self._has_structured_outputs = has_structured_outputs
        # Max reasoning tokens before the FSM force-emits <|think:end|>. None or
        # <= 0 disables the cap. Only counts tokens generated in the current
        # turn while in REASONING; the think_start / think_end sentinels are
        # excluded, and reasoning already present in the prompt is not counted
        # (the cap bounds *new* reasoning this generation adds).
        self._reasoning_budget = _normalize_reasoning_budget(reasoning_budget)
        self._reasoning_token_count = 0
        # Non-sentinel EOS ids (e.g. <|endoftext|>): allowed only in states
        # where ending the turn is template-legal (see __call__). Without
        # this, the model can stop mid-reasoning / mid-tool-call, which the
        # non-streaming parser papers over but the streaming parser cannot.
        self._extra_eos_token_ids = extra_eos_token_ids
        # True once the turn has produced actual content: at least one
        # regular token while in CONTENT, or a completed tool call. Until
        # then, ending the turn from CONTENT is forbidden — prevents
        # ``<|think:end|><|im:end|>`` (empty assistant message).
        self._content_progress = False
        self._all_control_token_ids = frozenset(
            token_id
            for token_id in (
                token_ids.im_start,
                token_ids.im_end,
                token_ids.think_start,
                token_ids.think_end,
                token_ids.im_content,
                token_ids.tool_start,
                token_ids.tool_end,
                token_ids.tool_call_start,
                token_ids.tool_call_end,
                token_ids.tool_arg_start,
                token_ids.tool_arg_value,
                token_ids.tool_arg_end,
                token_ids.tool_response_start,
                token_ids.tool_response_end,
            )
            if token_id is not None
        )
        # Forbidden-id table shared per (token_ids, eos) configuration.
        self._forbidden_by_key = _forbidden_table(token_ids, extra_eos_token_ids)
        # Forbidden set for the *leading* reasoning token (right after
        # <|think:start|>): the normal REASONING mask plus the leak-triggering
        # newline ids. None when no leading ids are configured, so the leading
        # step is indistinguishable from any other REASONING step.
        base_reasoning = self._forbidden_by_key[(SolarOpen2State.REASONING, False)]
        self._reasoning_leading_forbidden: tuple[int, ...] | None = (
            tuple(sorted(set(base_reasoning) | set(leading_forbidden_ids)))
            if leading_forbidden_ids
            else None
        )
        # token id -> next state for state-changing sentinels. ``setdefault``
        # preserves the original elif-chain precedence if two fields were
        # misconfigured with the same id.
        transitions: dict[int, SolarOpen2State] = {}
        for token_id, next_state in (
            (token_ids.think_start, SolarOpen2State.REASONING),
            (token_ids.think_end, SolarOpen2State.CONTENT),
            (token_ids.tool_call_start, SolarOpen2State.TOOL_CALL_BEGIN),
            (token_ids.tool_call_end, SolarOpen2State.TOOL_CALL_END),
            (token_ids.tool_arg_start, SolarOpen2State.TOOL_ARG_BEGIN),
            (token_ids.tool_arg_value, SolarOpen2State.TOOL_ARG_VALUE_BEGIN),
            (token_ids.tool_arg_end, SolarOpen2State.TOOL_ARG_END),
        ):
            if token_id is not None:
                transitions.setdefault(token_id, next_state)
        self._sentinel_transitions = transitions

    def _initial_state(self, prompt_token_ids: list[int]) -> SolarOpen2State:
        think_start = self.token_ids.think_start
        think_end = self.token_ids.think_end
        if think_start is None or think_end is None:
            return SolarOpen2State.CONTENT

        last_start = _rindex(prompt_token_ids, think_start)
        if last_start is None:
            return SolarOpen2State.CONTENT
        last_end = _rindex(prompt_token_ids, think_end)
        if last_end is None or last_start > last_end:
            return SolarOpen2State.REASONING
        return SolarOpen2State.CONTENT

    def _reset_to_prompt_state(self) -> None:
        self._state = self._initial_prompt_state
        self._content_progress = False
        self._reasoning_token_count = 0
        self._last_processed_len = 0
        self._prev_was_think_start = self._initial_prev_is_think_start

    def _process_token(self, token_id: int) -> None:
        ids = self.token_ids
        prev_state = self._state
        next_state = self._sentinel_transitions.get(token_id)
        if next_state is not None:
            self._state = next_state
        elif self._state == SolarOpen2State.TOOL_CALL_BEGIN:
            self._state = SolarOpen2State.TOOL_CALL_NAME
        elif self._state == SolarOpen2State.TOOL_ARG_BEGIN:
            self._state = SolarOpen2State.TOOL_ARG_NAME
        elif self._state == SolarOpen2State.TOOL_ARG_VALUE_BEGIN:
            self._state = SolarOpen2State.TOOL_ARG_VALUE
        elif self._state == SolarOpen2State.TOOL_CALL_END:
            self._state = SolarOpen2State.CONTENT

        # Reasoning-budget accounting: count tokens emitted *inside* a reasoning
        # block this turn. <|think:start|> (re)opens the block and resets the
        # counter; only tokens that keep the FSM in REASONING count (the closing
        # <|think:end|> transitions to CONTENT and is excluded).
        if ids.think_start is not None and token_id == ids.think_start:
            self._reasoning_token_count = 0
        elif (
            prev_state == SolarOpen2State.REASONING
            and self._state == SolarOpen2State.REASONING
        ):
            self._reasoning_token_count += 1

        if ids.tool_call_end is not None and token_id == ids.tool_call_end:
            # A completed tool call counts as turn content.
            self._content_progress = True
        elif (
            self._state == SolarOpen2State.CONTENT
            and token_id not in self._all_control_token_ids
        ):
            self._content_progress = True

        # Track whether *this* token is <|think:start|> so the next decode step
        # can recognize the leading reasoning position (the token right after it).
        self._prev_was_think_start = (
            ids.think_start is not None and token_id == ids.think_start
        )

    def _update_state_incremental(self, output_token_ids: list[int]) -> None:
        current_len = len(output_token_ids)
        if current_len < self._last_processed_len:
            self._reset_to_prompt_state()
        for idx in range(self._last_processed_len, current_len):
            self._process_token(output_token_ids[idx])
        self._last_processed_len = current_len

    def _reasoning_budget_exhausted(self) -> bool:
        """True when the reasoning-token cap has been reached this turn."""
        return (
            self._reasoning_budget is not None
            and self._reasoning_token_count >= self._reasoning_budget
        )

    @staticmethod
    def _force_single_token(logits: torch.Tensor, token_id: int) -> None:
        """Mask the entire vocabulary except ``token_id`` (force its emission)."""
        keep = logits[token_id].item()
        logits.fill_(NEG_INF)
        logits[token_id] = keep

    def advance_mask_ids(
        self, output_token_ids: list[int]
    ) -> tuple[int, ...] | str | None:
        """Advance the FSM and return this step's masking directive.

        Returns one of:
        - a forbidden-id tuple: the precomputed union of (a) control sentinels
          structurally invalid for the current state and (b) bare EOS ids
          (e.g. ``<|endoftext|>``) wherever ending the turn is template-illegal
          — anywhere except CONTENT-with-progress and right after a closed tool
          call — forcing the model to emit ``<|think:end|>`` (and close any open
          tool call) before it can stop, so streaming and non-streaming parses
          stay equivalent;
        - ``_FORCE_THINK_END``: the reasoning budget is exhausted, so this step
          must force ``<|think:end|>`` (mask everything except think_end);
        - ``None``: no masking applies (structured outputs own the CONTENT
          phase).
        """
        self._update_state_incremental(output_token_ids)
        return self._mask_ids()

    def _mask_ids(self) -> tuple[int, ...] | str | None:
        state = self._state
        if self._has_structured_outputs and state == SolarOpen2State.CONTENT:
            return None
        if state == SolarOpen2State.REASONING:
            if (
                self._reasoning_budget_exhausted()
                and self.token_ids.think_end is not None
            ):
                return _FORCE_THINK_END
            # Leading reasoning token (the one right after <|think:start|>):
            # forbid the leak-triggering newline ids on top of the normal
            # REASONING sentinel/EOS mask. Deeper reasoning tokens are untouched.
            if (
                self._prev_was_think_start
                and self._reasoning_leading_forbidden is not None
            ):
                return self._reasoning_leading_forbidden
            return self._forbidden_by_key[(SolarOpen2State.REASONING, False)]
        progress = state == SolarOpen2State.CONTENT and self._content_progress
        return self._forbidden_by_key[(state, progress)]

    def __call__(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        del prompt_token_ids
        directive = self.advance_mask_ids(output_token_ids)
        if directive is None:
            return logits
        if isinstance(directive, str):
            # Reasoning-budget cap: force <|think:end|> so the model stops
            # reasoning. If think_end is unresolved/out of range the cap is
            # unenforceable, so fall back to normal reasoning masking.
            think_end = self.token_ids.think_end
            if think_end is not None and think_end < logits.shape[-1]:
                self._force_single_token(logits, think_end)
                return logits
            forbidden_ids = self._forbidden_by_key[(SolarOpen2State.REASONING, False)]
        else:
            forbidden_ids = directive
        if forbidden_ids:
            indices = _mask_index_tensor(forbidden_ids, logits.shape[-1], logits.device)
            if indices.numel():
                logits[indices] = NEG_INF
        return logits


def _rindex(values: list[int], needle: int) -> int | None:
    for idx in range(len(values) - 1, -1, -1):
        if values[idx] == needle:
            return idx
    return None


def _has_structured_outputs(params: SamplingParams) -> bool:
    return (
        params.structured_outputs is not None
        and not params.structured_outputs.all_constraints_none()
    )


class SolarOpen2TemplateLogitsProcessor(AdapterLogitsProcessor, V2LogitsProcessor):
    """Solar Open2 token FSM adapter for both model runner interfaces."""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        if not params.extra_args:
            return
        value = params.extra_args.get(DISABLE_EXTRA_ARG)
        # Integer flag: any non-zero value disables the processor for this
        # request; 0 (or omitted) keeps it on. ``int`` transports natively over
        # the HTTP ``vllm_xargs`` field, so no protocol change is needed.
        # ``bool`` is an ``int`` subclass and stays accepted for in-process
        # callers.
        if value is not None and not isinstance(value, int):
            raise ValueError(
                f"{DISABLE_EXTRA_ARG} must be an integer "
                "(non-zero disables; 0 or omitted keeps the processor on)"
            )
        # Per-request reasoning-token cap. Validated eagerly so a malformed
        # value fails the request instead of being silently ignored.
        _normalize_reasoning_budget(params.extra_args.get(REASONING_BUDGET_EXTRA_ARG))

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device | LogitsProcRequestState,
        is_pin_memory: bool = False,
    ):
        self._v2_req_states = (
            device if isinstance(device, LogitsProcRequestState) else None
        )
        self._v2_requests: dict[int, _SolarOpen2RequestFactory] = {}
        if isinstance(device, LogitsProcRequestState):
            device = device.device
        super().__init__(vllm_config, device, is_pin_memory)
        self.token_ids = resolve_solar_open2_token_ids(vllm_config)
        self.extra_eos_token_ids = resolve_solar_open2_extra_eos_ids(
            vllm_config, self.token_ids
        )
        self.default_reasoning_budget = resolve_solar_open2_reasoning_budget_default()
        self.think_leading_forbidden_ids = (
            resolve_solar_open2_think_leading_forbidden_ids(vllm_config)
        )
        if self.think_leading_forbidden_ids:
            logger.info(
                "Solar Open2 logits processor forbidding leading newline ids %s "
                "right after <|think:start|>. Set %s to override (or '[]' to "
                "disable). Ids >= vocab_size are ignored at masking time.",
                sorted(self.think_leading_forbidden_ids),
                THINK_LEADING_FORBIDDEN_IDS_ENV,
            )
        else:
            logger.info(
                "Solar Open2 logits processor: leading-newline ban disabled "
                "(no forbidden ids resolved)."
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> "_SolarOpen2RequestFactory | None":
        self.validate_params(params)
        extra_args = params.extra_args or {}
        disable = extra_args.get(DISABLE_EXTRA_ARG)
        if disable is not None and disable != 0:
            return None
        # Per-request budget overrides the server-wide env default. An explicit
        # 0 opts this request out of a configured default (normalized to None).
        budget = extra_args.get(REASONING_BUDGET_EXTRA_ARG)
        if budget is None:
            budget = self.default_reasoning_budget
        return _SolarOpen2RequestFactory(
            self.token_ids,
            has_structured_outputs=_has_structured_outputs(params),
            extra_eos_token_ids=self.extra_eos_token_ids,
            reasoning_budget=budget,
            leading_forbidden_ids=self.think_leading_forbidden_ids,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        factory = self.new_req_logits_processor(sampling_params)
        self._v2_requests.pop(req_idx, None)
        if factory is None:
            return False
        self._v2_requests[req_idx] = factory
        return True

    def _apply_v2(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        req_states = self._v2_req_states
        assert req_states is not None
        active_slots = [int(idx) for idx in ctx.idx_mapping_np]
        if not any(idx in self._v2_requests for idx in active_slots):
            return logits

        # Only committed tokens advance persistent state. Draft prefixes use a
        # temporary FSM so rejected drafts cannot leak into the next step.
        slots, local_pos, positions, input_ids = torch.stack(
            (ctx.expanded_idx_mapping, ctx.expanded_local_pos, ctx.pos, ctx.input_ids)
        ).tolist()
        total_lens = req_states.total_len.gpu[ctx.idx_mapping].tolist()
        rows_by_slot: dict[int, list[int]] = {}
        for row, slot in enumerate(slots):
            rows_by_slot.setdefault(slot, []).append(row)
        rows_by_ids: dict[tuple[int, ...], list[int]] = {}
        force_rows: list[int] = []
        for slot, total_len in zip(active_slots, total_lens):
            factory = self._v2_requests.get(slot)
            if factory is None:
                continue
            rows = sorted(rows_by_slot[slot], key=lambda row: local_pos[row])
            prompt_len = int(req_states.prompt_len.np[slot])
            history = req_states.all_token_ids.gpu[slot]
            if factory._enforcer is None:
                factory._get_enforcer(history[:prompt_len].tolist())
            enforcer = factory._enforcer
            assert enforcer is not None
            committed_len = min(total_len, positions[rows[0]] + 1)
            output_len = max(0, committed_len - prompt_len)
            if output_len < enforcer._last_processed_len:
                enforcer._reset_to_prompt_state()
            start = prompt_len + enforcer._last_processed_len
            for token_id in history[start:committed_len].tolist():
                enforcer._process_token(token_id)
            enforcer._last_processed_len = output_len
            draft_enforcer = copy(enforcer)
            for row in rows:
                if local_pos[row] > 0:
                    draft_enforcer._process_token(input_ids[row])
                directive = draft_enforcer._mask_ids()
                if isinstance(directive, str):
                    force_rows.append(row)
                elif directive is not None:
                    rows_by_ids.setdefault(directive, []).append(row)
        return self._apply_masks(logits, rows_by_ids, force_rows)

    def apply(
        self, logits: torch.Tensor, ctx: LogitsContext | None = None
    ) -> torch.Tensor:
        """Batched masking: group rows sharing a forbidden set, mask once.

        The base ``AdapterLogitsProcessor.apply`` invokes the per-request
        callable row by row, which costs two small advanced-indexing kernels
        (plus a host->device index copy) per request per decoding step. The
        forbidden set only depends on each request's FSM state, so rows are
        grouped by their (precomputed) forbidden-id tuple and masked with a
        single 2-D indexed fill per distinct set — at most a handful of
        kernels per step regardless of batch size. The per-row result is
        bit-identical to the base implementation.
        """
        if ctx is not None:
            return self._apply_v2(logits, ctx)
        if not self.req_info:
            return logits
        rows_by_ids: dict[tuple[int, ...], list[int]] = {}
        force_rows: list[int] = []
        for req_idx, req_lp in self.req_info.items():
            factory = getattr(req_lp, "func", None)
            if isinstance(factory, _SolarOpen2RequestFactory):
                directive = factory.advance_mask_ids(*req_lp.args)
                if directive is None:
                    continue
                if isinstance(directive, str):
                    # Reasoning-budget cap: force <|think:end|> for this row.
                    force_rows.append(req_idx)
                else:
                    rows_by_ids.setdefault(directive, []).append(req_idx)
            else:
                # Defensive fallback: unknown per-request callable (should
                # not happen — new_req_logits_processor only returns the
                # factory). Preserve base adapter semantics.
                req_logits = logits[req_idx]
                new_logits = req_lp(req_logits)
                if new_logits is not req_logits:
                    logits[req_idx] = new_logits
        return self._apply_masks(logits, rows_by_ids, force_rows)

    def _apply_masks(
        self,
        logits: torch.Tensor,
        rows_by_ids: dict[tuple[int, ...], list[int]],
        force_rows: list[int],
    ) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        device = logits.device
        for forbidden_ids, rows in rows_by_ids.items():
            cols = _mask_index_tensor(forbidden_ids, vocab_size, device)
            if not cols.numel():
                continue
            row_indices = torch.tensor(rows, dtype=torch.long, device=device)
            logits[row_indices.unsqueeze(1), cols.unsqueeze(0)] = NEG_INF
        if force_rows:
            rows_t = torch.tensor(force_rows, dtype=torch.long, device=device)
            think_end = self.token_ids.think_end
            if think_end is not None and think_end < vocab_size:
                # Budget-exhausted rows: mask everything except <|think:end|>
                # with a single batched fill+restore (rare — at most one step
                # per request).
                saved = logits[rows_t, think_end].clone()
                logits[rows_t, :] = NEG_INF
                logits[rows_t, think_end] = saved
            else:
                # think_end unresolved/out of range: the cap is unenforceable,
                # so fall back to normal reasoning masking for these rows
                # (mirrors the per-row __call__ path, keeping the two paths
                # equivalent).
                forbidden_ids = _forbidden_table(
                    self.token_ids, self.extra_eos_token_ids
                )[(SolarOpen2State.REASONING, False)]
                cols = _mask_index_tensor(forbidden_ids, vocab_size, device)
                if cols.numel():
                    logits[rows_t.unsqueeze(1), cols.unsqueeze(0)] = NEG_INF
        return logits


class _SolarOpen2RequestFactory:
    def __init__(
        self,
        token_ids: SolarOpen2TokenIds,
        has_structured_outputs: bool,
        extra_eos_token_ids: frozenset[int] = frozenset(),
        reasoning_budget: int | None = None,
        leading_forbidden_ids: frozenset[int] = frozenset(),
    ) -> None:
        self.token_ids = token_ids
        self.has_structured_outputs = has_structured_outputs
        self.extra_eos_token_ids = extra_eos_token_ids
        self.reasoning_budget = reasoning_budget
        self.leading_forbidden_ids = leading_forbidden_ids
        self._enforcer: SolarOpen2TokenFSMEnforcer | None = None

    def _get_enforcer(self, prompt_token_ids: list[int]) -> SolarOpen2TokenFSMEnforcer:
        if self._enforcer is None:
            self._enforcer = SolarOpen2TokenFSMEnforcer(
                prompt_token_ids,
                self.token_ids,
                has_structured_outputs=self.has_structured_outputs,
                extra_eos_token_ids=self.extra_eos_token_ids,
                reasoning_budget=self.reasoning_budget,
                leading_forbidden_ids=self.leading_forbidden_ids,
            )
        return self._enforcer

    def advance_mask_ids(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
    ) -> tuple[int, ...] | str | None:
        """FSM step for the batched ``apply`` path (no tensor work here)."""
        return self._get_enforcer(prompt_token_ids).advance_mask_ids(output_token_ids)

    def __call__(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return self._get_enforcer(prompt_token_ids)(
            prompt_token_ids, output_token_ids, logits
        )
