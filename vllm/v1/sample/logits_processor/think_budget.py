# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOTIF: think-budget request-admission helpers.

The v0.20 fork carried a full ThinkingTokenBudgetLogitsProcessor in this
module; upstream v0.26 ships the equivalent state machine natively
(``ThinkingBudgetStateHolder``, wired through ``gpu_input_batch`` into the
sampler), so only the Motif-specific admission surface remains here:

* ratio budgets: ``SamplingParams.thinking_token_budget`` (absolute, most
  specific) > ``vllm_xargs.think_budget_ratio`` > ``VLLM_THINK_BUDGET_RATIO``
  env > off. The ratio applies to the request's actual completion budget
  ``min(max_tokens, max_model_len - prompt_len)``, so "think may use 60% of
  my response" is ``0.6`` verbatim -- no back-solving against the model
  context. It is resolved to an absolute ``thinking_token_budget`` at
  admission (``resolve_thinking_budget_for_request``), so the worker-side
  holder only ever sees absolute budgets.
* force sequence: ``vllm_xargs.think_budget_force_str`` >
  ``VLLM_THINK_BUDGET_FORCE_STR`` env > bare reasoning end token ids. The
  string must END with the reasoning end text (e.g. ``</think>``); it is
  encoded at admission and shipped to the holder via
  ``extra_args[THINK_BUDGET_FORCE_IDS_KEY]``.
"""

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

# vllm_xargs keys (test/experimentation surface).
_XARGS_BUDGET_RATIO = "think_budget_ratio"
_XARGS_FORCE_STR = "think_budget_force_str"

# extra_args key carrying the admission-resolved force token ids to the
# worker-side ThinkingBudgetStateHolder (internal plumbing, not user API).
THINK_BUDGET_FORCE_IDS_KEY = "_think_budget_force_ids"

# The historical answer-space floor: at least this many tokens (or a quarter
# of the completion budget for small requests) are always kept for the answer.
DEFAULT_ANSWER_RESERVE = 4096

# Env vars are VLLM_-prefixed (upstream-PR friendly); the fork-internal
# MOTIF_-prefixed spellings that predate the unification keep working with
# a one-time deprecation warning.
_warned_legacy_env: set[str] = set()


def env_with_legacy(name: str) -> str:
    """Read env var ``name`` (VLLM_-prefixed), falling back to the legacy
    MOTIF_-prefixed spelling."""
    value = os.environ.get(name, "")
    if value:
        return value
    legacy = "MOTIF_" + name.removeprefix("VLLM_")
    value = os.environ.get(legacy, "")
    if value and legacy not in _warned_legacy_env:
        _warned_legacy_env.add(legacy)
        logger.debug("%s is deprecated; use %s (same semantics).", legacy, name)
    return value


_FORCE_STR_ENCODE_CACHE: dict[str, list[int]] = {}


def encode_force_str(force_str: str, model_config: "ModelConfig") -> list[int]:
    """Tokenize a force string, cached (single tokenizer per server)."""
    ids = _FORCE_STR_ENCODE_CACHE.get(force_str)
    if ids is None:
        from vllm.tokenizers import cached_tokenizer_from_config

        tokenizer = cached_tokenizer_from_config(model_config)
        ids = list(tokenizer.encode(force_str, add_special_tokens=False))
        _FORCE_STR_ENCODE_CACHE[force_str] = ids
    return list(ids)


def resolve_think_budget_ratio(params: SamplingParams) -> float | None:
    """Per-request budget ratio: vllm_xargs > env > None (off).

    Raises ValueError on malformed xargs values (called by the input
    processor during request validation, so clients get a 400).
    """
    extra = params.extra_args or {}
    raw = extra.get(_XARGS_BUDGET_RATIO)
    if raw is not None:
        try:
            ratio = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"vllm_xargs.{_XARGS_BUDGET_RATIO} must be a float, got {raw!r}."
            ) from None
        if not 0.0 < ratio <= 1.0:
            raise ValueError(
                f"vllm_xargs.{_XARGS_BUDGET_RATIO} must be in (0, 1], got {ratio}."
            )
        return ratio
    env_ratio = float(env_with_legacy("VLLM_THINK_BUDGET_RATIO") or 0)
    return env_ratio if 0.0 < env_ratio <= 1.0 else None


def resolve_force_str(params: SamplingParams) -> str | None:
    """Per-request force string: vllm_xargs > env > None (bare end ids).

    Raises ValueError on malformed xargs values.
    """
    extra = params.extra_args or {}
    raw = extra.get(_XARGS_FORCE_STR)
    if raw is not None:
        if not isinstance(raw, str) or not raw:
            raise ValueError(
                f"vllm_xargs.{_XARGS_FORCE_STR} must be a non-empty string."
            )
        return raw
    return env_with_legacy("VLLM_THINK_BUDGET_FORCE_STR") or None


def validate_think_budget_xargs(
    params: SamplingParams,
    reasoning_enabled: bool,
    v2_model_runner: bool = False,
) -> None:
    """Request-admission validation of the think-budget vllm_xargs keys.

    Only xargs-origin values are checked: a server-wide env ratio on a
    server without reasoning stays a startup warning (existing behavior),
    never a per-request error. Raises ValueError (-> HTTP 400).
    """
    extra = params.extra_args or {}
    has_ratio = extra.get(_XARGS_BUDGET_RATIO) is not None
    has_force = extra.get(_XARGS_FORCE_STR) is not None
    if not has_ratio and not has_force:
        return
    if has_ratio:
        resolve_think_budget_ratio(params)
    if has_force:
        resolve_force_str(params)
    if not reasoning_enabled:
        raise ValueError(
            "vllm_xargs think_budget_ratio / think_budget_force_str require "
            "reasoning to be enabled (--reasoning-parser)."
        )
    if v2_model_runner:
        raise ValueError(
            "vllm_xargs think_budget_ratio / think_budget_force_str are not "
            "yet supported by the V2 model runner. Run vLLM with "
            "VLLM_USE_V2_MODEL_RUNNER=0 to use them."
        )


def resolve_thinking_budget_for_request(
    params: SamplingParams,
    prompt_len: int,
    model_config: "ModelConfig",
    reasoning_enabled: bool,
    v2_model_runner: bool,
) -> None:
    """Admission-time resolution of the MOTIF think-budget surface.

    Mutates ``params`` in place: fills ``thinking_token_budget`` from the
    ratio chain when the absolute field is unset, and stashes the resolved
    force-sequence ids under ``extra_args[THINK_BUDGET_FORCE_IDS_KEY]``.
    Malformed / misconfigured xargs were already 400'd by
    ``validate_think_budget_xargs``; anything raising here fails closed, and
    env-origin misconfig degrades to a one-time warning (a server-wide env
    default must not fail every request).
    """
    if not reasoning_enabled or v2_model_runner:
        try:
            misconfigured = (
                resolve_think_budget_ratio(params) is not None
                or resolve_force_str(params) is not None
            )
        except ValueError:
            misconfigured = True
        if misconfigured:
            logger.warning_once(
                "[think-budget] a think-budget ratio / force string is set "
                "but reasoning is not enabled (pass --reasoning-parser) or "
                "the V2 model runner is active; think-budget is a no-op."
            )
        return

    try:
        force_str = resolve_force_str(params)
    except ValueError:
        force_str = None
    if force_str:
        force_ids = encode_force_str(force_str, model_config)
        if force_ids:
            extra = dict(params.extra_args or {})
            extra[THINK_BUDGET_FORCE_IDS_KEY] = force_ids
            params.extra_args = extra

    if params.thinking_token_budget is not None:
        return
    try:
        ratio = resolve_think_budget_ratio(params)
    except ValueError:
        ratio = None
    if ratio is None:
        return
    # The ratio applies to the request's actual completion budget --
    # avail = min(max_tokens, max_model_len - prompt). process_inputs has
    # already defaulted max_tokens to the post-prompt space when unset, so
    # offline requests degrade to the old post-prompt-space behavior. The
    # answer reserve floor shrinks to avail/4 for small budgets, else the
    # historical 4096 floor would zero the think budget entirely.
    avail = model_config.max_model_len - prompt_len
    if params.max_tokens:
        avail = min(avail, params.max_tokens)
    if avail <= 0:
        return
    reserve_floor = min(DEFAULT_ANSWER_RESERVE, max(1, avail // 4))
    answer_reserve = max(reserve_floor, int(avail * (1.0 - ratio)))
    params.thinking_token_budget = max(0, avail - answer_reserve)
