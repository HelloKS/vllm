# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.solar_open2_tool_parser import SolarOpen2ToolParser

# A local Solar Open2 tokenizer directory can be supplied for integration
# coverage. Parser-only tests remain offline when the public model is not local.
MODEL = os.environ.get(
    "SOLAR_OPEN2_TEST_MODEL",
    "upstage/Solar-Open2-250B",
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(scope="module")
def tokenizer():
    if os.path.isdir(MODEL):
        return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)
    return SimpleNamespace(get_vocab=lambda: {})


@pytest.fixture
def parser(tokenizer):
    return SolarOpen2ToolParser(tokenizer)


@pytest.fixture
def typed_request():
    """ChatCompletionRequest with a tools schema covering every coercion path."""
    return ChatCompletionRequest(
        model="solar-open2",
        messages=[],
        tools=[
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer"},
                        },
                    },
                },
            ),
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "do_all_types",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "s": {"type": "string"},
                            "i": {"type": "integer"},
                            "n": {"type": "number"},
                            "b": {"type": "boolean"},
                            "arr": {"type": "array"},
                            "obj": {"type": "object"},
                            "maybe_int": {"type": ["integer", "null"]},
                        },
                    },
                },
            ),
        ],
    )


class TestExtractToolCalls:
    """Non-streaming tool call extraction tests."""

    def test_no_tool_calls(self, parser):
        model_output = "This is a regular response."
        result = parser.extract_tool_calls(model_output, None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a regular response."

    def test_single_tool_no_content(self, parser):
        model_output = (
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Seoul"}
        assert result.content is None

    def test_single_tool_with_content_prefix(self, parser):
        model_output = (
            "Let me check the weather."
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.content == "Let me check the weather."

    def test_multiple_tool_calls(self, parser):
        model_output = (
            "I'll check both."
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>\n"
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Tokyo<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Seoul"}
        assert result.tool_calls[1].function.name == "get_weather"
        assert json.loads(result.tool_calls[1].function.arguments) == {"city": "Tokyo"}
        assert result.content == "I'll check both."

    def test_multiple_arguments_without_schema(self, parser):
        """With no schema to consult, values fall back to strings."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>vLLM tutorial<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"query": "vLLM tutorial", "max_results": "5"}

    def test_multiple_arguments_with_schema(self, parser, typed_request):
        """With a schema, ``integer`` args are coerced to int."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>vLLM tutorial<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, typed_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"query": "vLLM tutorial", "max_results": 5}

    def test_no_arguments(self, parser):
        model_output = "<|tool_call:start|>get_time\n<|tool_call:end|>"
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_time"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_json_like_value_in_arg(self, parser):
        model_output = (
            "<|tool_call:start|>process_data\n"
            '<|tool_arg:start|>data<|tool_arg:value|>{"key": "value"}<|tool_arg:end|>\n'
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"data": '{"key": "value"}'}

    def test_reasoning_tags_with_tool_calls(self, parser):
        """The tool parser receives raw model output which may include
        reasoning tags. Content should be everything before the first
        tool_call:start, including reasoning tags."""
        model_output = (
            "<|think:start|>I should use a tool<|think:end|>"
            "Sure, let me help."
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        # Content is everything before the first tool call
        assert result.content == (
            "<|think:start|>I should use a tool<|think:end|>Sure, let me help."
        )


class TestTypeCoercion:
    """Schema-driven type coercion for tool call arguments."""

    def _build(self, arg_name: str, arg_value: str) -> str:
        return (
            "<|tool_call:start|>do_all_types\n"
            f"<|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )

    def test_integer(self, parser, typed_request):
        result = parser.extract_tool_calls(self._build("i", "42"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"i": 42}

    def test_number_float(self, parser, typed_request):
        result = parser.extract_tool_calls(self._build("n", "3.14"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"n": 3.14}

    def test_number_downcasts_to_int_when_fractional_is_zero(
        self, parser, typed_request
    ):
        """Matches qwen3coder / step3p5 / seed_oss behavior."""
        result = parser.extract_tool_calls(self._build("n", "7.0"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"n": 7}

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_boolean(self, parser, typed_request, raw, expected):
        result = parser.extract_tool_calls(self._build("b", raw), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"b": expected}

    def test_array(self, parser, typed_request):
        result = parser.extract_tool_calls(
            self._build("arr", "[1, 2, 3]"), typed_request
        )
        assert json.loads(result.tool_calls[0].function.arguments) == {"arr": [1, 2, 3]}

    def test_object(self, parser, typed_request):
        result = parser.extract_tool_calls(
            self._build("obj", '{"k": "v"}'), typed_request
        )
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "obj": {"k": "v"}
        }

    def test_union_with_null_prefers_non_null_type(self, parser, typed_request):
        """``["integer", "null"]`` should still coerce numeric values to int."""
        result = parser.extract_tool_calls(self._build("maybe_int", "9"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"maybe_int": 9}

    def test_literal_null_always_becomes_none(self, parser, typed_request):
        """``"null"`` coerces to ``None`` regardless of declared type."""
        result = parser.extract_tool_calls(self._build("i", "null"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"i": None}

    def test_string_type_preserves_raw_value(self, parser, typed_request):
        result = parser.extract_tool_calls(self._build("s", "42"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"s": "42"}

    def test_conversion_failure_falls_back_to_string(self, parser, typed_request):
        """Malformed numeric input returns the raw string, does not raise.

        vLLM's other typed parsers (qwen3xml, qwen3coder, step3p5, seed_oss)
        all degrade to the raw string on conversion failure instead of
        raising, so a single malformed tool call does not break the whole
        OpenAI-compatible response. This test pins that convention for
        ``solar_open2`` as well.
        """
        result = parser.extract_tool_calls(
            self._build("i", "not-a-number"), typed_request
        )
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "i": "not-a-number"
        }

    def test_unknown_function_falls_back_to_string(self, parser, typed_request):
        """Tool call for a function not in ``tools`` → string fallback.

        Models occasionally hallucinate function names that were not in the
        request's ``tools`` list. Rather than raise, the parser returns the
        raw string so the client can decide how to handle the malformed
        call — consistent with qwen3xml / qwen3coder behavior.
        """
        model_output = (
            "<|tool_call:start|>unknown_fn\n"
            "<|tool_arg:start|>x<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"x": "5"}

    def test_unknown_param_falls_back_to_string(self, parser, typed_request):
        """Param not declared in the function's schema → string fallback.

        Similar to the unknown-function case above, but one level deeper:
        the function exists but the model emitted a parameter name that is
        not in ``properties``. Again, string fallback rather than raising,
        matching the other typed parsers.
        """
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>undeclared<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "undeclared": "5"
        }


def _run_stream(parser, model_output, request, *, chunk_size=1):
    """Feed ``model_output`` into the streaming parser in fixed-size chunks
    and return ``(content, tool_calls)`` where ``tool_calls`` is a dict of
    ``{index: {"name": str | None, "arguments": str, "id": str | None}}``
    reconstructed from the deltas.
    """
    assembled_content = ""
    tool_calls: dict[int, dict] = {}
    previous_text = ""
    previous_token_ids: list[int] = []
    for i in range(0, len(model_output), chunk_size):
        delta_text = model_output[i : i + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=previous_token_ids,
            delta_token_ids=[],
            request=request,
        )
        previous_text = current_text
        if delta is None:
            continue
        if delta.content:
            assembled_content += delta.content
        for tc in delta.tool_calls or []:
            slot = tool_calls.setdefault(
                tc.index, {"name": None, "arguments": "", "id": None}
            )
            if tc.id is not None:
                slot["id"] = tc.id
            fn = tc.function
            if fn is not None:
                if fn.name is not None:
                    slot["name"] = fn.name
                if fn.arguments:
                    slot["arguments"] += fn.arguments
    return assembled_content, tool_calls


class TestStreaming:
    """Incremental streaming extraction for solar_open2 tool calls."""

    def test_single_chunk_matches_non_stream(self, parser, typed_request):
        """Feeding the entire output as one chunk reproduces the non-stream
        parsing result exactly, including type coercion.
        """
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>vLLM<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(
            parser, model_output, typed_request, chunk_size=len(model_output)
        )
        assert content == ""
        assert list(calls.keys()) == [0]
        assert calls[0]["name"] == "search"
        assert calls[0]["id"] is not None and calls[0]["id"].startswith("call_")
        assert json.loads(calls[0]["arguments"]) == {
            "query": "vLLM",
            "max_results": 5,
        }

    def test_char_by_char_reassembles_to_valid_json(self, parser, typed_request):
        """Feeding one character at a time — including through every sentinel
        boundary — must still reassemble into identical, coerced JSON."""
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            "<|tool_arg:start|>i<|tool_arg:value|>42<|tool_arg:end|>\n"
            "<|tool_arg:start|>n<|tool_arg:value|>3.14<|tool_arg:end|>\n"
            "<|tool_arg:start|>b<|tool_arg:value|>true<|tool_arg:end|>\n"
            "<|tool_arg:start|>arr<|tool_arg:value|>[1, 2, 3]<|tool_arg:end|>\n"
            '<|tool_arg:start|>obj<|tool_arg:value|>{"k": "v"}<|tool_arg:end|>\n'
            "<|tool_arg:start|>s<|tool_arg:value|>hello<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=1)
        assert calls[0]["name"] == "do_all_types"
        assert json.loads(calls[0]["arguments"]) == {
            "i": 42,
            "n": 3.14,
            "b": True,
            "arr": [1, 2, 3],
            "obj": {"k": "v"},
            "s": "hello",
        }

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 13, 31])
    def test_various_chunk_sizes(self, parser, typed_request, chunk_size):
        """Any chunk size must produce the same final args JSON, including
        splits in the middle of any sentinel or value."""
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            "<|tool_arg:start|>i<|tool_arg:value|>7<|tool_arg:end|>\n"
            "<|tool_arg:start|>b<|tool_arg:value|>false<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(
            parser, model_output, typed_request, chunk_size=chunk_size
        )
        assert json.loads(calls[0]["arguments"]) == {"i": 7, "b": False}

    def test_content_before_tool_call_is_emitted(self, parser, typed_request):
        """Free-form text that precedes the first tool call must come through
        as streamed content, not be swallowed."""
        model_output = (
            "I'll check the weather. "
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>3<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=4)
        assert content == "I'll check the weather. "
        assert json.loads(calls[0]["arguments"]) == {
            "query": "Seoul",
            "max_results": 3,
        }

    def test_no_tool_calls_passthrough(self, parser, typed_request):
        """Pure text output (no sentinels) must stream as content only."""
        model_output = "Hello! I cannot help with that."
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert content == model_output
        assert calls == {}

    def test_multiple_tool_calls(self, parser, typed_request):
        """Two consecutive tool calls get distinct indices and independently
        valid argument JSONs."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>2<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Tokyo<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>3<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=5)
        assert set(calls.keys()) == {0, 1}
        assert json.loads(calls[0]["arguments"]) == {
            "query": "Seoul",
            "max_results": 2,
        }
        assert json.loads(calls[1]["arguments"]) == {
            "query": "Tokyo",
            "max_results": 3,
        }

    def test_empty_args_call(self, parser, typed_request):
        """Tool call with zero args streams a literal ``{}`` as arguments."""
        model_output = "<|tool_call:start|>get_time\n<|tool_call:end|>"
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert calls[0]["name"] == "get_time"
        assert calls[0]["arguments"] == "{}"
        assert json.loads(calls[0]["arguments"]) == {}

    def test_malformed_value_falls_back_to_string(self, parser, typed_request):
        """If a value fails to coerce (e.g. non-numeric for an ``integer``
        param), the streaming parser must degrade to the raw string rather
        than raising — same fallback policy as the non-stream path."""
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            "<|tool_arg:start|>i<|tool_arg:value|>not-a-number<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=4)
        assert json.loads(calls[0]["arguments"]) == {"i": "not-a-number"}

    def test_literal_null_becomes_none(self, parser, typed_request):
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            "<|tool_arg:start|>maybe_int<|tool_arg:value|>null<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=1)
        assert json.loads(calls[0]["arguments"]) == {"maybe_int": None}

    def test_streamed_state_reset_between_streams(self, parser, typed_request):
        """A second stream (previous_text="") must not leak tool_call_id or
        buffer from the first stream."""
        first = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        second = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>2<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls1 = _run_stream(parser, first, typed_request, chunk_size=3)
        _, calls2 = _run_stream(parser, second, typed_request, chunk_size=3)
        # Each stream must report the tool call at index 0 — the second
        # stream must not carry the index forward to 1.
        assert list(calls1.keys()) == [0]
        assert list(calls2.keys()) == [0]
        assert json.loads(calls2[0]["arguments"]) == {
            "query": "B",
            "max_results": 2,
        }

    def test_value_containing_left_angle_bracket(self, parser, typed_request):
        """A ``<`` byte inside a string value is fine (it is never a full
        sentinel prefix that matters because we consume up to
        ``<|tool_arg:end|>``)."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>a<b<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert json.loads(calls[0]["arguments"]) == {
            "query": "a<b",
            "max_results": 1,
        }

    def test_streaming_matches_non_stream_on_full_fixture(self, parser, typed_request):
        """The reconstructed streaming result must equal the non-stream
        result for the same input. Pins the invariant that streaming and
        non-streaming parse paths agree end-to-end."""
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            "<|tool_arg:start|>i<|tool_arg:value|>42<|tool_arg:end|>\n"
            "<|tool_arg:start|>n<|tool_arg:value|>7.0<|tool_arg:end|>\n"
            "<|tool_arg:start|>b<|tool_arg:value|>true<|tool_arg:end|>\n"
            "<|tool_arg:start|>arr<|tool_arg:value|>[1, 2]<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        # Non-stream: use a fresh parser so stream state does not interfere.
        non_stream_parser = SolarOpen2ToolParser(parser.model_tokenizer)
        ns_result = non_stream_parser.extract_tool_calls(model_output, typed_request)
        ns_args = json.loads(ns_result.tool_calls[0].function.arguments)

        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert json.loads(calls[0]["arguments"]) == ns_args

    def test_reasoning_tags_before_tool_call(self, parser, typed_request):
        """Reasoning + plain content before a tool call must both flow through
        as streamed content — identical to the non-stream convention where
        ``content`` is everything before the first ``<|tool_call:start|>``.

        This pins that the ``<|`` prefix on ``<|think:start|>`` doesn't get
        stuck in the sentinel hold-back buffer: the parser eventually sees
        enough bytes to rule it out and flush.
        """
        model_output = (
            "<|think:start|>I should use a tool<|think:end|>"
            "Sure, let me help."
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>3<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        # Char-by-char is the strictest chunking: every ``<|`` prefix hits the
        # hold-back path and the parser must correctly resolve each one.
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=1)
        assert content == (
            "<|think:start|>I should use a tool<|think:end|>Sure, let me help."
        )
        assert json.loads(calls[0]["arguments"]) == {
            "query": "Seoul",
            "max_results": 3,
        }

    def test_content_between_tool_calls(self, parser, typed_request):
        """Free-form text between two consecutive tool calls must stream as
        content, and both tool calls must still produce valid argument JSON
        at their correct indices."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "Also checking..."
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>2<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert content == "Also checking..."
        assert set(calls.keys()) == {0, 1}
        assert json.loads(calls[0]["arguments"]) == {"query": "A", "max_results": 1}
        assert json.loads(calls[1]["arguments"]) == {"query": "B", "max_results": 2}

    def test_empty_string_value(self, parser, typed_request):
        """A zero-length string value (``<|tool_arg:value|><|tool_arg:end|>``)
        must round-trip as ``""``. This is the boundary between the
        ``READING_ARG_NAME`` and ``READING_ARG_VALUE`` sentinels landing
        back-to-back with nothing between them."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|><|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert json.loads(calls[0]["arguments"]) == {"query": "", "max_results": 1}

    def test_false_positive_sentinel_prefix_in_content(self, parser, typed_request):
        """Content that begins with ``<|`` but is NOT a tool sentinel must
        eventually flush. Regression guard for the hold-back logic: the
        parser must not hoard bytes forever just because they share the
        ``<|`` prefix with the sentinels it's waiting for."""
        model_output = (
            "<|thinking about it|> here is the plan. "
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=1)
        assert content == "<|thinking about it|> here is the plan. "
        assert json.loads(calls[0]["arguments"]) == {"query": "x", "max_results": 1}

    def test_unicode_value(self, parser, typed_request):
        """Multibyte characters inside an argument value must survive the
        streaming path unchanged. Exercises a non-ASCII value paired with a
        typed (integer) arg so both string-preserving and coercion paths
        see unicode in the same stream."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>서울 날씨<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>3<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert json.loads(calls[0]["arguments"]) == {
            "query": "서울 날씨",
            "max_results": 3,
        }

    def test_truncated_stream_before_tool_call_end(self, parser, typed_request):
        """If the stream ends mid tool call (no ``<|tool_call:end|>`` arrives),
        the parser must not raise. Whatever args already finalized should be
        reachable via ``prev_tool_call_arr``; the final ``}`` is simply
        never emitted. Upstream serving_chat handles the dangling call."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            # Deliberately no <|tool_call:end|>.
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        # Name was emitted and both args were finalized — only the closing
        # ``}`` is missing because the sentinel never arrived.
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments"] == '{"query": "x", "max_results": 5'

    def test_delta_protocol_shape(self, parser, typed_request):
        """The first delta of each tool call must carry ``type="function"``,
        an ``id``, a ``name``, and an ``arguments=""`` seed. Subsequent deltas
        for the same tool must carry only ``arguments`` (no name/id/type
        repeated). The ``index`` must stay identical across all deltas for
        one tool. OpenAI clients depend on this shape to reassemble.
        """
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        # Collect every raw DeltaToolCall emitted, keeping emission order.
        raw_deltas: list = []
        previous_text = ""
        for i in range(len(model_output)):
            delta_text = model_output[i]
            current_text = previous_text + delta_text
            d = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=typed_request,
            )
            previous_text = current_text
            if d is not None:
                raw_deltas.extend(d.tool_calls)

        # First delta has the full identity set.
        first = raw_deltas[0]
        assert first.index == 0
        assert first.type == "function"
        assert first.id is not None and first.id.startswith("call_")
        assert first.function is not None
        assert first.function.name == "search"
        assert first.function.arguments == ""

        # Every subsequent delta for this tool must carry ONLY arguments —
        # repeating name/id/type would cause OpenAI clients to treat them as
        # a new tool call.
        for d in raw_deltas[1:]:
            assert d.index == 0
            assert d.type is None
            assert d.id is None
            assert d.function is not None
            assert d.function.name is None
            assert d.function.arguments is not None and d.function.arguments != ""

    def test_base_class_state_after_stream(self, parser, typed_request):
        """After a completed stream, ``prev_tool_call_arr`` and
        ``streamed_args_for_tool`` must equal the fully-assembled JSON for
        each tool. ``serving_chat`` reads these to decide whether there are
        unstreamed argument tokens; if they drift, it emits a duplicate
        flush at the end of the stream.
        """
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>2<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        # prev_tool_call_arr must mirror both tool calls and their full JSONs.
        assert len(parser.prev_tool_call_arr) == 2
        assert parser.prev_tool_call_arr[0]["name"] == "search"
        assert json.loads(parser.prev_tool_call_arr[0]["arguments"]) == {
            "query": "A",
            "max_results": 1,
        }
        assert parser.prev_tool_call_arr[1]["name"] == "search"
        assert json.loads(parser.prev_tool_call_arr[1]["arguments"]) == {
            "query": "B",
            "max_results": 2,
        }
        # streamed_args_for_tool must match what the client received.
        for i in (0, 1):
            assert parser.streamed_args_for_tool[i] == calls[i]["arguments"], (
                f"tool {i}: stream tracking diverged from emitted deltas"
            )
        assert parser.current_tool_id == 1  # zero-indexed, two calls emitted

    def test_streaming_without_tools_schema(self, parser):
        """Without ``request.tools`` the parser must still stream valid JSON —
        values simply stay strings because there is no schema to coerce
        against. Matches the non-stream ``test_multiple_arguments_without_schema``.
        """
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>vLLM<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, None, chunk_size=4)
        assert json.loads(calls[0]["arguments"]) == {
            "query": "vLLM",
            "max_results": "5",
        }

    def test_back_to_back_tool_calls_no_separator(self, parser, typed_request):
        """``<|tool_call:end|><|tool_call:start|>`` with nothing (not even a
        newline) between them must still produce two distinct tool calls.
        This is the tightest possible boundary between calls — the state
        machine must transition cleanly without emitting phantom content.
        """
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=4)
        assert content == ""
        assert set(calls.keys()) == {0, 1}
        assert json.loads(calls[0]["arguments"]) == {"query": "A"}
        assert json.loads(calls[1]["arguments"]) == {"query": "B"}

    def test_zero_args_call_followed_by_args_call(self, parser, typed_request):
        """Empty-args (``{}``) followed by non-empty args: the ``_first_arg_in_call``
        flag must be correctly reset between calls so the second tool opens
        with ``{`` and not ``, `` — a state-leak regression would produce
        ``{, "query": ...`` and break JSON parsing.
        """
        model_output = (
            "<|tool_call:start|>get_time\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert calls[0]["arguments"] == "{}"
        # Most important: the second call must open with ``{``, not ``, ``.
        assert calls[1]["arguments"].startswith("{")
        assert json.loads(calls[1]["arguments"]) == {"query": "x"}

    def test_unique_ids_across_tool_calls(self, parser, typed_request):
        """Each tool call in the same stream must get a distinct ``id``.
        UUIDs collide with vanishing probability but an off-by-one or reuse
        bug in ID assignment would surface here."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>C<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        ids = [calls[i]["id"] for i in sorted(calls.keys())]
        assert len(ids) == 3
        assert len(set(ids)) == 3, f"duplicate ids: {ids}"
        assert all(cid and cid.startswith("call_") for cid in ids)

    def test_empty_delta_text_no_raise(self, parser, typed_request):
        """vLLM occasionally invokes the parser with ``delta_text=""`` but
        non-empty ``delta_token_ids`` (e.g. a stop token with no visible
        glyph). The parser must not raise and must not fabricate output.
        """
        previous_text = "<|tool_call:start|>search\n"
        # Feed an empty-text delta that represents a stop/special token.
        d = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=previous_text,
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[0],
            delta_token_ids=[0],
            request=typed_request,
        )
        # No tool call has been emitted yet (no trailing \n after name),
        # so the delta must be either None or an empty-payload DeltaMessage.
        assert d is None or (not d.content and not d.tool_calls)

    def test_content_and_tool_name_in_same_delta(self, parser, typed_request):
        """When a single chunk spans ``<text>...<|tool_call:start|>NAME\\n``,
        the returned ``DeltaMessage`` must carry BOTH ``content`` and
        ``tool_calls`` simultaneously. The OpenAI spec allows this, and
        batching them reduces SSE round-trips."""
        model_output = (
            "prefix "
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        # Single-chunk ensures one DeltaMessage holds everything that was
        # parseable in that chunk.
        d = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=model_output,
            delta_text=model_output,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=typed_request,
        )
        assert d is not None
        assert d.content == "prefix "
        assert len(d.tool_calls) >= 1
        # First tool_calls entry is the name-seed delta.
        first_tc = d.tool_calls[0]
        assert first_tc.function is not None
        assert first_tc.function.name == "search"

    @pytest.mark.parametrize(
        "arg_name,raw_value,expected_py",
        [
            ("i", "42", 42),
            ("i", "null", None),
            ("i", "not-a-number", "not-a-number"),  # conversion failure fallback
            ("n", "3.14", 3.14),
            ("n", "7.0", 7),  # number → int downcast
            ("b", "true", True),
            ("b", "False", False),
            ("arr", "[1, 2, 3]", [1, 2, 3]),
            ("obj", '{"k": "v"}', {"k": "v"}),
            ("maybe_int", "9", 9),  # [integer, null] union prefers non-null
            ("s", "42", "42"),  # explicit string preserves raw
        ],
    )
    def test_parity_with_non_stream_across_all_coercions(
        self, parser, typed_request, arg_name, raw_value, expected_py
    ):
        """For every coercion path, the streamed arguments must parse to the
        same Python value as the non-stream parser would produce. This pins
        the invariant that streaming never diverges from batch parsing on
        the same input."""
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            f"<|tool_arg:start|>{arg_name}<|tool_arg:value|>{raw_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        ns_parser = SolarOpen2ToolParser(parser.model_tokenizer)
        ns_result = ns_parser.extract_tool_calls(model_output, typed_request)
        ns_args = json.loads(ns_result.tool_calls[0].function.arguments)
        assert ns_args == {arg_name: expected_py}

        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        stream_args = json.loads(calls[0]["arguments"])
        assert stream_args == ns_args


class TestStreamingWhitespaceNormalization:
    """Streamed content must match the non-streaming whitespace semantics:
    content is everything before the FIRST tool call, and whitespace-only
    content around tool calls is suppressed (the serving layer renders it as
    no content). Regression tests for the tau2 multi-turn drift where
    streamed assistant tool-call messages carried content="\\n"."""

    TOOL = (
        "<|tool_call:start|>get_weather\n"
        "<|tool_arg:start|>location<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    )

    def test_ws_only_before_tool_call_is_dropped(self, parser, typed_request):
        for chunk_size in (1, 3, 7, 1000):
            parser._reset_stream_state()
            content, tool_calls = _run_stream(
                parser, "\n" + self.TOOL, typed_request, chunk_size=chunk_size
            )
            assert content == "", f"chunk_size={chunk_size}: {content!r}"
            assert len(tool_calls) == 1

    def test_ws_between_tool_calls_is_dropped(self, parser, typed_request):
        out = self.TOOL + "\n\n" + self.TOOL
        for chunk_size in (1, 4, 1000):
            parser._reset_stream_state()
            content, tool_calls = _run_stream(
                parser, out, typed_request, chunk_size=chunk_size
            )
            assert content == "", f"chunk_size={chunk_size}: {content!r}"
            assert len(tool_calls) == 2

    def test_trailing_ws_after_text_before_first_call_is_kept(
        self, parser, typed_request
    ):
        # Non-streaming keeps "Let me check.\n" (content = prefix before the
        # first call, verbatim).
        out = "Let me check.\n" + self.TOOL
        for chunk_size in (1, 5, 1000):
            parser._reset_stream_state()
            content, tool_calls = _run_stream(
                parser, out, typed_request, chunk_size=chunk_size
            )
            assert content == "Let me check.\n", f"chunk_size={chunk_size}: {content!r}"
            assert len(tool_calls) == 1
