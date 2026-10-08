# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif reasoning and Hermes-format tool calls, including JSON repair."""

import functools
import json
import re
from collections.abc import Callable

from vllm.entrypoints.generate.base.protocol import ExtractedToolCallInformation
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX. Any other backslash
# (e.g. shell ``\&`` ``\$``, regex ``\s`` ``\[``) is invalid JSON and gets
# dropped. The alternation consumes escapes left to right so the trailing
# backslash of a valid ``\\`` pair is never re-read as the start of the next
# escape: a lookahead-only sub turned ``\\[`` (escaped backslash, then ``[``)
# into the invalid ``\[`` by dropping its second backslash.
_ESCAPE_OR_LONE_BACKSLASH_RE = re.compile(r'(\\["\\/bfnrtu])|\\')

# Keys whose values are ``list[str]`` in the Motif tool schemas. Scoped to
# known keys on purpose: blindly wrapping any string value in ``[`` would
# let the bracket balancer "fix" unrelated breakage into wrong JSON.
_ARRAY_VALUE_KEY_RE = re.compile(r'"(queries|urls)"(\s*:\s*)"')


def _normalize_invalid_escapes(text: str) -> str:
    """Drop backslashes that do not form a valid JSON escape."""
    return _ESCAPE_OR_LONE_BACKSLASH_RE.sub(lambda m: m.group(1) or "", text)


def _coerce_arguments_wrapper(obj: dict) -> dict:
    """Wrap flat calls: ``{"name": .., "x": ..}`` -> ``{"name", "arguments"}``."""
    if isinstance(obj, dict) and "name" in obj and "arguments" not in obj:
        args = {k: v for k, v in obj.items() if k != "name"}
        return {"name": obj["name"], "arguments": args}
    return obj


def _escape_quotes_in_strings(block: str) -> str:
    """R-quote: escape unescaped quotes inside string values.

    A ``"`` inside a string is treated as a closing quote only when the next
    non-whitespace character is a JSON structural character (``,]}:``) or
    end of input; otherwise it is content and becomes ``\\"``.
    """
    out: list[str] = []
    in_str = False
    i, n = 0, len(block)
    while i < n:
        ch = block[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(block[i + 1])
            i += 2
            continue
        if ch == '"':
            j = i + 1
            while j < n and block[j] in " \t\r\n":
                j += 1
            next_ch = block[j] if j < n else ""
            if next_ch in ",]}:" or j >= n:
                in_str = False
                out.append(ch)
            else:
                out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _quote_repair_candidates(block: str, budget: int = 64):
    """R-backtrack: enumerate close-vs-content interpretations of quotes.

    :func:`_escape_quotes_in_strings` decides locally: an unescaped ``"``
    inside a string closes it iff the next non-whitespace character is a JSON
    structural character (``,]}:``). That rule is fooled by string content
    that *looks like* JSON (e.g. a comment ``{"@type":"MusicRecording"}``
    inside a ``cmd``), where a mid-content quote is followed by ``:``.

    Such quotes are genuinely ambiguous, so treat each as a choice point and
    DFS over interpretations, yielding fully rendered candidate blocks. The
    close interpretation is explored first, so the first candidate equals the
    output of :func:`_escape_quotes_in_strings`. The caller accepts the first
    candidate that parses (and passes the schema oracle); ``budget`` bounds
    the number of leaves, keeping the worst case (2^choice_points) small.
    """
    n = len(block)
    yielded = 0
    # frame: (index, in_str, rendered-so-far)
    stack: list[tuple[int, bool, str]] = [(0, False, "")]
    while stack and yielded < budget:
        i, in_str, acc = stack.pop()
        buf: list[str] = []
        branched = False
        while i < n:
            ch = block[i]
            if not in_str:
                buf.append(ch)
                if ch == '"':
                    in_str = True
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(block[i + 1])
                i += 2
                continue
            if ch == '"':
                j = i + 1
                while j < n and block[j] in " \t\r\n":
                    j += 1
                if j >= n:  # end of input: the string must terminate here
                    in_str = False
                    buf.append(ch)
                    i += 1
                    continue
                if block[j] in ",]}:":
                    # Ambiguous: branch. LIFO, so push content-escape first
                    # and the close interpretation (stock heuristic) pops
                    # first.
                    base = acc + "".join(buf)
                    stack.append((i + 1, True, base + '\\"'))
                    stack.append((i + 1, False, base + '"'))
                    branched = True
                    break
                buf.append('\\"')  # mid-content quote: forced escape
                i += 1
                continue
            buf.append(ch)
            i += 1
        if not branched:
            yielded += 1
            yield acc + "".join(buf)


def _tool_arg_specs(tools) -> dict[str, tuple[frozenset[str] | None, bool]]:
    """Extract ``name -> (property names, additionalProperties)`` from tools.

    Tolerates ChatCompletionToolsParam / FunctionTool objects and plain
    dicts (with or without the ``{"function": ...}`` wrapper).
    """
    specs: dict[str, tuple[frozenset[str] | None, bool]] = {}
    for tool in tools or []:
        fn = (
            tool.get("function")
            if isinstance(tool, dict)
            else getattr(tool, "function", None)
        )
        src = tool if fn is None else fn
        if isinstance(src, dict):
            name, params = src.get("name"), src.get("parameters")
        else:
            name, params = getattr(src, "name", None), getattr(src, "parameters", None)
        if not name:
            continue
        props: frozenset[str] | None = None
        additional = True
        if isinstance(params, dict):
            if isinstance(params.get("properties"), dict):
                props = frozenset(params["properties"])
            additional = bool(params.get("additionalProperties", True))
        specs[name] = (props, additional)
    return specs


def _schema_oracle(tools):
    """Acceptance predicate for backtracking candidates.

    A repaired candidate is only trusted when its tool name is registered
    and (for closed schemas) its argument keys are a subset of the declared
    properties — a wrong close interpretation that happens to parse tends to
    invent argument keys out of string content (e.g. ``"@type"``), which no
    registered schema declares. Without tool information every parsed
    candidate is accepted (matching the schema-less ladder rungs).
    """
    specs = _tool_arg_specs(tools)
    if not specs:
        return lambda obj: True

    def accept(obj: dict) -> bool:
        name = obj.get("name")
        if name not in specs:
            return False
        props, additional = specs[name]
        args = obj.get("arguments")
        if isinstance(args, dict) and props is not None and not additional:
            return frozenset(args) <= props
        return True

    return accept


def _open_string_array(block: str) -> str:
    """R-array: ``"queries": "a", "b"`` -> ``"queries": ["a", "b"``.

    Only inserts the opening ``[``; the closing ``]`` is supplied by
    :func:`_balance_brackets`.
    """
    return _ARRAY_VALUE_KEY_RE.sub(r'"\1"\2["', block, count=1)


def _balance_brackets(block: str) -> str:
    """R-bracket: balance ``[]`` / ``{}`` counts near the end of the block.

    Handles duplicated ``]]``, excess trailing ``]``, missing ``]`` (inserted
    right before the trailing ``}`` run), and over/under-closed braces.
    """
    t = block.rstrip()
    if "]]" in t and t.count("]") > t.count("["):
        t = t.replace("]]", "]", 1)
    while t.count("]") > t.count("[") and re.search(r"\]\s*\}*\s*$", t):
        t = re.sub(r"\](\s*\}*\s*)$", r"\1", t, count=1)
    missing = t.count("[") - t.count("]")
    if missing > 0:
        t = re.sub(r"(\}+)\s*$", "]" * missing + r"\1", t, count=1)
    open_braces, close_braces = t.count("{"), t.count("}")
    if open_braces > close_braces:
        t = t + "}" * (open_braces - close_braces)
    else:
        while t.count("}") > t.count("{") and t.endswith("}"):
            t = t[:-1].rstrip()
    return t


# Streaming repair-cache bound. Completed blocks add one or two keys each,
# but a trailing body that stays valid JSON while growing (e.g. a bare
# number) would add a new key per delta; past the cap, repairs simply run
# uncached so memory stays bounded.
_REPAIR_CACHE_MAX = 128

# Tried in order; the first variant that yields a dict wins. Ordered by
# measured corpus coverage so already-valid blocks exit on the first rung.
_REPAIR_LADDER: tuple[Callable[[str], str], ...] = (
    lambda block: block,
    _balance_brackets,
    lambda block: _balance_brackets(_open_string_array(block)),
    _escape_quotes_in_strings,
    lambda block: _balance_brackets(_escape_quotes_in_strings(block)),
    lambda block: _balance_brackets(
        _open_string_array(_escape_quotes_in_strings(block))
    ),
)


def _try_load(block: str) -> dict | None:
    """Parse with idiom fixes: invalid escapes, trailing ``}``, control chars."""
    bases = [block, _normalize_invalid_escapes(block)]
    for base in bases:
        trimmed = base.rstrip()
        variants = [trimmed]
        candidate = trimmed
        for _ in range(3):
            if candidate.endswith("}"):
                candidate = candidate[:-1].rstrip()
                variants.append(candidate)
        for variant in variants:
            # strict=False accepts raw control characters inside strings.
            for strict in (True, False):
                try:
                    obj = json.loads(variant, strict=strict)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    return obj
    return None


def _repair_block(block: str, tools=None) -> str | None:
    """Return the block as a valid JSON string, or None if unrecoverable.

    The deterministic ladder runs first (unchanged semantics). If every rung
    fails, R-backtrack searches over ambiguous-quote interpretations; those
    candidates are additionally gated by the schema oracle when ``tools``
    are known, because a wrong interpretation can still parse as JSON.
    """
    for repair in _REPAIR_LADDER:
        obj = _try_load(repair(block))
        if obj is not None:
            return json.dumps(_coerce_arguments_wrapper(obj), ensure_ascii=False)
    accept = _schema_oracle(tools)
    seen: set[str] = set()
    for cand in _quote_repair_candidates(block):
        for variant in (cand, _balance_brackets(cand)):
            if variant in seen:
                continue
            seen.add(variant)
            obj = _try_load(variant)
            if obj is None:
                continue
            coerced = _coerce_arguments_wrapper(obj)
            if accept(coerced):
                return json.dumps(coerced, ensure_ascii=False)
    return None


def sanitize_model_output(text: str, tools=None) -> str:
    """Rewrite each ``<tool_call>`` block as valid JSON where possible.

    Prose around the blocks is left untouched; unrecoverable blocks are kept
    verbatim so the downstream Hermes parser behaves exactly like stock.
    A trailing block whose ``</tool_call>`` never arrived is repaired and
    closed when its body can be made valid. ``tools`` (when known) gate the
    backtracking repair candidates via the schema oracle.
    """
    if _TOOL_CALL_START not in text:
        return text

    def _sub_block(match: re.Match) -> str:
        fixed = _repair_block(match.group(1), tools)
        if fixed is None:
            return match.group(0)
        return f"{_TOOL_CALL_START}\n{fixed}\n{_TOOL_CALL_END}"

    text = _TOOL_CALL_BLOCK_RE.sub(_sub_block, text)

    open_pos = text.rfind(_TOOL_CALL_START)
    if open_pos != -1 and text.find(_TOOL_CALL_END, open_pos) == -1:
        inner = text[open_pos + len(_TOOL_CALL_START) :]
        fixed = _repair_block(inner.strip(), tools)
        if fixed is not None:
            text = text[:open_pos] + f"{_TOOL_CALL_START}\n{fixed}\n{_TOOL_CALL_END}"
    return text


@functools.cache
def motif_config(thinking: bool = True) -> ParserEngineConfig:
    transitions = {
        (ParserState.CONTENT, "TOOL_START"): Transition(
            ParserState.TOOL_ARGS, (EventType.TOOL_CALL_START,)
        ),
        (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
            ParserState.CONTENT, (EventType.TOOL_CALL_END,)
        ),
    }
    terminals = {"TOOL_START": _TOOL_CALL_START, "TOOL_END": _TOOL_CALL_END}
    if thinking:
        terminals.update(THINK_START="<think>", THINK_END="</think>")
        transitions.update(
            {
                (ParserState.REASONING, "THINK_START"): Transition(
                    ParserState.REASONING
                ),
                (ParserState.REASONING, "THINK_END"): Transition(
                    ParserState.CONTENT, (EventType.REASONING_END,)
                ),
            }
        )
    return ParserEngineConfig(
        name="motif",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        terminals=terminals,
        token_id_terminals=terminals,
        transitions=transitions,
        preserve_tokens=frozenset(("<think>", "</think>")),
        tool_args_json=False,
        strip_trailing_reasoning_whitespace=False,
        strip_content_whitespace_with_tools=False,
    )


class MotifParser(ParserEngine):
    def __init__(self, tokenizer, tools=None, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        self._repair_cache: dict[str, str | None] = {}
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=motif_config(self.thinking_enabled),
            **kwargs,
        )

    def _cached_repair(self, block: str) -> str | None:
        if block in self._repair_cache:
            return self._repair_cache[block]
        fixed = _repair_block(block, self._tools)
        if len(self._repair_cache) < _REPAIR_CACHE_MAX:
            self._repair_cache[block] = fixed
        return fixed

    def _extract_name_and_args(self, raw_body: str) -> tuple[str, str]:
        fixed = self._cached_repair(raw_body)
        if fixed is None:
            return "", ""
        return super()._extract_name_and_args(fixed)

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        # Repairs can change earlier bytes. Stream arguments only once the
        # body is valid JSON or its closing marker has arrived.
        slot = self._tool_slots[idx]
        try:
            json.loads(slot.args)
        except (ValueError, TypeError):
            return None
        return self._flush_arg_converter(idx)

    def _flush_arg_converter(self, idx: int) -> str | None:
        slot = self._tool_slots[idx]
        _, arguments = self._extract_name_and_args(slot.args)
        previous = slot.streamed_json
        if arguments and arguments.startswith(previous):
            slot.streamed_json = arguments
            return arguments[len(previous) :] or None
        return None

    def extract_tool_calls_from_content(self, content, request):
        result = super().extract_tool_calls_from_content(content, request)
        if not result.tools_called:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=content or None
            )
        return result
