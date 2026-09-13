# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from types import SimpleNamespace

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning.solar_open2_reasoning_parser import (
    SolarOpen2ReasoningParser,
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


class _CharacterTokenizer:
    """Small offline tokenizer for parser-only tests."""

    def __init__(self):
        self._vocab = {chr(i): i for i in range(128)}

    def get_vocab(self):
        return self._vocab

    def tokenize(self, text: str):
        return list(text)

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.fixture(scope="module")
def tokenizer():
    if os.path.isdir(MODEL):
        return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)
    return _CharacterTokenizer()


@pytest.fixture
def parser(tokenizer):
    return SolarOpen2ReasoningParser(tokenizer)


# Under the current chat template, the generation prompt always prefills
# ``<|think:start|>`` (medium/high effort) or ``<|think:start|><|think:end|>``
# (low effort). The model output therefore never contains
# ``<|think:start|>`` — only ``<|think:end|>`` (or nothing, for low effort).


class TestExtractReasoning:
    """Non-streaming reasoning extraction tests."""

    def test_standard_reasoning_and_content(self, parser):
        model_output = "I need to analyze<|think:end|>The answer is 42"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "I need to analyze"
        assert content == "The answer is 42"

    def test_reasoning_only_no_end_tag(self, parser):
        model_output = "Still thinking..."
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        # No end tag means regex doesn't match -> treat as no reasoning
        assert reasoning is None
        assert content == model_output

    def test_reasoning_with_empty_content(self, parser):
        model_output = "My reasoning<|think:end|>"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "My reasoning"
        assert content is None

    def test_pure_content_no_tags(self, parser):
        # Low-effort path: prompt already prefilled <|think:start|><|think:end|>,
        # so the generation itself is content only.
        model_output = "Hello, world!"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning is None
        assert content == "Hello, world!"

    def test_empty_reasoning(self, parser):
        # Medium/high effort but the model emits an empty think block.
        model_output = "<|think:end|>Direct answer"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning is None
        assert content == "Direct answer"

    def test_multiline_reasoning(self, parser):
        model_output = "Step 1: analyze\nStep 2: solve\n<|think:end|>Final answer"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "Step 1: analyze\nStep 2: solve\n"
        assert content == "Final answer"


class TestEmbeddedToolCallPromotion:
    """Non-streaming recovery of tool calls emitted *inside* the think
    block (mirrors the upstream Qwen3 fix, vllm-project/vllm#39055).

    The chat-completion server extracts reasoning first and the tool
    parser only inspects the content channel, so a tool call emitted
    before ``<|think:end|>`` would otherwise vanish from the response.
    ``extract_reasoning`` promotes complete tool-call blocks out of
    reasoning into content — appended *after* any existing content,
    because ``SolarOpen2ToolParser`` keeps only the text preceding the
    first ``<|tool_call:start|>`` as content.
    """

    TOOL_BLOCK = (
        "<|tool_call:start|>get_weather\n"
        "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    )
    TOOL_BLOCK_2 = (
        "<|tool_call:start|>get_time\n"
        "<|tool_arg:start|>tz<|tool_arg:value|>KST<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    )

    def test_embedded_tool_call_promoted_to_content(self, parser):
        """Embedded case: tool call inside think, nothing after the end
        tag. The block must surface in content; the surrounding thinking
        text stays in reasoning."""
        model_output = f"I should check the weather.\n{self.TOOL_BLOCK}\n<|think:end|>"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "I should check the weather."
        assert content == self.TOOL_BLOCK

    def test_promoted_block_appended_after_existing_content(self, parser):
        """Ordering pin: real content after <|think:end|> must come first
        so the tool parser (content = text before the first tool call)
        does not drop it."""
        model_output = f"thinking\n{self.TOOL_BLOCK}\n<|think:end|>The answer is 42"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "thinking"
        assert content == f"The answer is 42\n{self.TOOL_BLOCK}"

    def test_embedded_plus_tool_call_after_think_end(self, parser):
        """A tool call inside think *and* a regular one after the end tag:
        both must end up in content."""
        model_output = (
            f"plan calls\n{self.TOOL_BLOCK}\n<|think:end|>{self.TOOL_BLOCK_2}"
        )
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "plan calls"
        assert content == f"{self.TOOL_BLOCK_2}\n{self.TOOL_BLOCK}"

    def test_multiple_embedded_blocks(self, parser):
        model_output = (
            f"first\n{self.TOOL_BLOCK}\nthen\n{self.TOOL_BLOCK_2}\n<|think:end|>"
        )
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "first\n\nthen"
        assert content == f"{self.TOOL_BLOCK}\n{self.TOOL_BLOCK_2}"

    def test_block_without_args_promoted(self, parser):
        block = "<|tool_call:start|>refresh\n<|tool_call:end|>"
        model_output = f"no args needed\n{block}\n<|think:end|>done"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "no args needed"
        assert content == f"done\n{block}"

    def test_mere_sentinel_mention_not_promoted(self, parser):
        """Pure-reasoning no-op guard: the model merely *talking about*
        a sentinel in its thinking must not trigger promotion, and the
        reasoning text must round-trip byte-identical (no strip)."""
        thinking = "Should I emit <|tool_call:start|> here? No.\n"
        model_output = f"{thinking}<|think:end|>Answer"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == thinking
        assert content == "Answer"

    def test_abandoned_call_not_promoted(self, parser):
        """Malformed case: the model starts a tool call in think but never
        closes it before <|think:end|>. The fragment fails the tool-parser
        grammar, so it stays in reasoning untouched and content is
        unaffected."""
        thinking = (
            "let me call\n<|tool_call:start|>foo\n"
            "<|tool_arg:start|>a<|tool_arg:value|>1\n"
        )
        model_output = f"{thinking}<|think:end|>after"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == thinking
        assert content == "after"

    def test_truncated_mid_think_regression(self, parser):
        """Truncated case: generation cut off inside the think block (no
        <|think:end|> at all). The existing no-match branch routes the
        whole output to content, where the tool parser can already see
        the block — promotion must not interfere with this path."""
        model_output = f"checking weather\n{self.TOOL_BLOCK}"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning is None
        assert content == model_output

    def test_promoted_blocks_parse_via_tool_parser(self, parser, tokenizer):
        """End-to-end contract with SolarOpen2ToolParser: feeding the
        promoted content channel to the tool parser must yield the tool
        call *and* keep the real content."""
        model_output = (
            f"Need the weather.\n{self.TOOL_BLOCK}\n<|think:end|>Checking now."
        )
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "Need the weather."

        tool_parser = SolarOpen2ToolParser(tokenizer)
        info = tool_parser.extract_tool_calls(
            content, request=SimpleNamespace(tools=None)
        )
        assert info.tools_called
        assert [tc.function.name for tc in info.tool_calls] == ["get_weather"]
        assert json.loads(info.tool_calls[0].function.arguments) == {"city": "Seoul"}
        assert info.content == "Checking now.\n"


class TestIsReasoningEnd:
    """Test is_reasoning_end with token IDs."""

    def test_reasoning_ended(self, parser, tokenizer):
        text = "some reasoning<|think:end|>content"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is True

    def test_reasoning_not_ended(self, parser, tokenizer):
        text = "still reasoning"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is False

    def test_empty_ids(self, parser):
        assert parser.is_reasoning_end([]) is False

    def test_multi_turn_prompt_high_effort_not_ended(self, parser, tokenizer):
        """Regression: a multi-turn prompt contains a *prior* assistant
        turn rendered as ``<|think:start|><|think:end|>`` followed by a
        fresh, still-open ``<|think:start|>`` for the current generation
        (medium/high effort). ``is_reasoning_end`` must return False so the
        chat-completion server keeps routing the new generation through the
        reasoning parser. A plain ``THINK_END in text`` check used to fire
        on the previous turn's end tag, making the server dump the whole
        new generation (thinking text + literal ``<|think:end|>``) into
        ``content`` — the streaming multi-turn reasoning-leak bug."""
        prompt = (
            "<|im:start|>user<|im:content|>q1<|im:end|>\n"
            "<|im:start|>assistant<|im:content|>"
            "<|think:start|><|think:end|>a1<|im:end|>\n"
            "<|im:start|>user<|im:content|>q2<|im:end|>\n"
            "<|im:start|>assistant<|im:content|><|think:start|>"
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is False

    def test_multi_turn_prompt_low_effort_ended(self, parser, tokenizer):
        """Low effort: the generation prompt prefills the empty think pair
        ``<|think:start|><|think:end|>`` as the *last* think block, so the
        new generation is content-only and reasoning has ended."""
        prompt = (
            "<|im:start|>user<|im:content|>q1<|im:end|>\n"
            "<|im:start|>assistant<|im:content|>"
            "<|think:start|><|think:end|>a1<|im:end|>\n"
            "<|im:start|>user<|im:content|>q2<|im:end|>\n"
            "<|im:start|>assistant<|im:content|><|think:start|><|think:end|>"
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is True

    def test_single_turn_prompt_high_effort_not_ended(self, parser, tokenizer):
        """Single-turn high-effort prompt ends with an open
        ``<|think:start|>`` and no end tag anywhere — not ended."""
        prompt = (
            "<|im:start|>user<|im:content|>q1<|im:end|>\n"
            "<|im:start|>assistant<|im:content|><|think:start|>"
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is False

    def test_literal_start_tag_in_prior_text_anchors_on_last(self, parser, tokenizer):
        """Adversarial: a prior assistant turn whose reasoning text itself
        contains a literal ``<|think:start|>`` must not confuse the anchor.
        ``is_reasoning_end`` keys off the *last* ``<|think:start|>`` (always
        the generation prompt's), so a still-open final block stays
        not-ended regardless of earlier stray start tags."""
        prompt = (
            "<|im:start|>assistant<|im:content|>"
            "<|think:start|>I'll mention <|think:start|> here<|think:end|>"
            "ok<|im:end|>\n"
            "<|im:start|>user<|im:content|>again?<|im:end|>\n"
            "<|im:start|>assistant<|im:content|><|think:start|>"
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is False


def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _run_reasoning_stream(parser, deltas):
    """Local streaming driver that accepts DeltaMessages carrying both
    ``reasoning`` and ``content`` simultaneously — the solar_open2 parser
    emits both when the reasoning end tag and trailing content arrive in
    one chunk, which is valid OpenAI streaming but stricter than
    ``tests.reasoning.utils.StreamingReasoningReconstructor``.
    """
    reasoning: str | None = None
    content: str | None = None
    previous_text = ""
    for delta in deltas:
        current_text = previous_text + delta
        msg = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        previous_text = current_text
        if msg is None:
            continue
        if msg.reasoning is not None:
            reasoning = (reasoning or "") + msg.reasoning
        if msg.content is not None:
            content = (content or "") + msg.content
    return reasoning, content


class TestExtractReasoningStreaming:
    """Streaming reasoning extraction — feeds the output in chunks and
    checks the reconstructed (reasoning, content) tuple matches the
    non-stream parser on the same input."""

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 100])
    def test_standard_reasoning_and_content(self, parser, chunk_size):
        model_output = "I need to analyze<|think:end|>The answer is 42"
        deltas = _split_into_chunks(model_output, chunk_size)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        assert reasoning == "I need to analyze"
        assert content == "The answer is 42"

    def test_multiline_reasoning(self, parser):
        model_output = "Step 1: analyze\nStep 2: solve\n<|think:end|>Final answer"
        deltas = _split_into_chunks(model_output, 3)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        assert reasoning == "Step 1: analyze\nStep 2: solve\n"
        assert content == "Final answer"

    def test_empty_reasoning(self, parser):
        """Medium/high effort but model emits empty think block: output
        starts with ``<|think:end|>`` directly. Reasoning is None, only
        content is emitted."""
        model_output = "<|think:end|>Direct answer"
        deltas = _split_into_chunks(model_output, 2)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        # Empty reasoning emits as either None (never appended) or "" —
        # what matters is the content round-trips exactly.
        assert reasoning in (None, "")
        assert content == "Direct answer"

    def test_reasoning_with_empty_content(self, parser):
        """Reasoning block followed by nothing — content stays None."""
        model_output = "My reasoning<|think:end|>"
        deltas = _split_into_chunks(model_output, 4)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        assert reasoning == "My reasoning"
        assert content is None

    # Low-effort streaming (prompt prefilled ``<|think:start|><|think:end|>``,
    # so the generation has no end tag at all) is inherently ambiguous in
    # the streaming protocol: the parser cannot distinguish "low effort,
    # all content" from "high effort, still thinking" without seeing the
    # prompt, and the streaming API does not expose the request. The
    # current bias is reasoning-first, mirroring every other ``<think>``-
    # style parser in this directory; clients that want low-effort
    # behaviour should drive the streaming path through the non-stream
    # extractor or omit the reasoning parser entirely. The non-streaming
    # ``test_pure_content_no_tags`` above pins the batch path.

    def test_sentinel_split_across_chunks(self, parser):
        """The ``<|think:end|>`` sentinel is split exactly mid-token across
        a chunk boundary. The buffer must hold the partial prefix and
        recognise it once the tail arrives — otherwise the ``<|think:`` on
        the reasoning side would be emitted as reasoning text and break
        the transition."""
        prefix_end = "<|think:"
        tail_end = "end|>"
        chunks = [
            "reasoning body ",
            prefix_end,
            tail_end + "after",
        ]
        reasoning, content = _run_reasoning_stream(parser, chunks)
        assert reasoning == "reasoning body "
        assert content == "after"

    def test_is_reasoning_end_defers_during_active_stream(self, parser, tokenizer):
        """``is_reasoning_end`` must defer the "reasoning ended" signal
        until the parser has *textually* observed ``<|think:end|>``.

        The chat-completion server bypasses the reasoning parser for all
        subsequent deltas once ``is_reasoning_end`` returns True. If the
        flip happened while the parser was still holding partial end-tag
        bytes (because the single-token ``<|think:end|>`` was only
        partially decoded by vLLM's incremental decoder), the suffix
        bytes would later leak through to the tool parser as a spurious
        content delta. This test pins the deferred-flip behaviour.
        """
        end_text = "<|think:end|>"

        # During an active stream where the parser is mid-holdback,
        # is_reasoning_end stays False even if the token list already
        # decodes to text containing the end tag.
        parser.extract_reasoning_streaming(
            previous_text="",
            current_text="reasoning <|th",
            delta_text="reasoning <|th",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        end_token_ids = tokenizer.encode(end_text, add_special_tokens=False)
        assert end_text in tokenizer.decode(end_token_ids)
        assert parser.is_reasoning_end(end_token_ids) is False

        # Once the rest of the end tag arrives textually, the parser
        # transitions and is_reasoning_end flips to True.
        parser.extract_reasoning_streaming(
            previous_text="reasoning <|th",
            current_text="reasoning <|think:end|>content",
            delta_text="ink:end|>content",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        assert parser.is_reasoning_end(end_token_ids) is True

    def test_stream_state_reset_between_streams(self, parser):
        """Running two independent streams on the same parser instance
        must not leak buffered state from the first into the second."""
        first = "A<|think:end|>alpha"
        second = "B<|think:end|>beta"
        r1, c1 = _run_reasoning_stream(parser, _split_into_chunks(first, 3))
        r2, c2 = _run_reasoning_stream(parser, _split_into_chunks(second, 3))
        assert (r1, c1) == ("A", "alpha")
        assert (r2, c2) == ("B", "beta")

    @pytest.mark.parametrize(
        "model_output",
        [
            "I need to analyze<|think:end|>The answer is 42",
            "My reasoning<|think:end|>",
            "<|think:end|>Direct answer",
            "Step 1: analyze\nStep 2: solve\n<|think:end|>Final answer",
        ],
    )
    def test_streaming_parity_with_non_stream(self, parser, model_output):
        """Char-by-char streaming output must match the non-stream parser
        for every canonical shape — pins the invariant that streaming
        never diverges from batch on well-formed input."""
        ns_r, ns_c = run_reasoning_extraction(parser, [model_output], streaming=False)
        s_r, s_c = _run_reasoning_stream(parser, _split_into_chunks(model_output, 1))
        # Normalize empty reasoning: non-stream returns None, streaming may
        # surface an empty string from the flush. Collapse both.
        assert (s_r or None) == ns_r
        assert s_c == ns_c
