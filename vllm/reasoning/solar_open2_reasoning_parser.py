# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class SolarOpen2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Solar Open2 model.

    Solar Open2 uses ``<|think:start|>`` and ``<|think:end|>`` as reasoning
    delimiters. Both are *single* special tokens in the tokenizer, but the
    chat-completion streaming endpoint feeds the parser the
    incrementally-decoded text — and vLLM's incremental decoder can split
    the bytes of a single special token across delta boundaries (e.g.
    ``"<|th"`` arrives in one delta and ``"ink:end|>"`` only in the next).

    The chat template always prefills ``<|think:start|>`` into the
    generation prompt:

    - ``reasoning_effort in {medium, high}`` → prompt ends with
      ``<|im:content|><|think:start|>``; the model emits
      ``<reasoning><|think:end|><content>`` (or ``<|think:end|><content>``
      for empty reasoning).
    - ``reasoning_effort == low`` → prompt ends with
      ``<|im:content|><|think:start|><|think:end|>``; the model emits
      ``<content>`` only.

    Therefore the model output itself never contains ``<|think:start|>``.
    The parser only needs to find ``<|think:end|>`` and split around it.

    Streaming uses a two-state buffer (REASONING -> CONTENT) with a
    suffix-only hold-back: emit everything except trailing bytes that
    could be the start of ``<|think:end|>``. Anchor on the last ``<``
    rather than substring-containment (Olmo3-style) so single-character
    deltas like ``"r"`` do not stall on the ``r`` in ``<|think:sta`r`t|>``.

    ``is_reasoning_end`` is overridden to defer the "reasoning ended"
    signal until the parser has *textually* observed ``<|think:end|>`` in
    the stream — even if the end token id is already in the decoded
    token list. The chat-completion server bypasses this parser for all
    subsequent deltas once ``is_reasoning_end`` flips, so flipping while
    the parser is still holding partial end-tag bytes would cause the
    remaining suffix bytes (e.g. ``"ink:end|>"``) to leak through to the
    tool parser as a spurious content delta.

    The non-streaming (prompt-side) branch of ``is_reasoning_end`` must
    answer "has the reasoning for the *current* generation already ended
    in the prompt?" — the chat-completion server feeds it the whole
    ``prompt_token_ids`` to decide whether to bypass reasoning extraction
    (``prompt_is_reasoning_end_arr`` in serving_chat, originally meant for
    the ``enable_thinking=False`` case). A naive "``<|think:end|>`` appears
    anywhere in the prompt" check is wrong for Solar Open2 because the chat
    template renders *every prior assistant turn* as
    ``<|think:start|>...<|think:end|>`` — so any multi-turn prompt contains
    an end tag from a previous turn. That falsely flips the server into
    "reasoning already ended" mode and routes the entire new generation
    (thinking text + the literal ``<|think:end|>`` token) into ``content``.
    Only the *current* (last) think block matters: reasoning has ended iff
    ``<|think:end|>`` appears *after the last* ``<|think:start|>`` — true
    for low effort (prompt ends ``<|think:start|><|think:end|>``) and false
    for medium/high effort (prompt ends with an open ``<|think:start|>``).

    ``extract_reasoning`` additionally promotes complete tool-call blocks
    that the model emitted *inside* the think block into the content
    channel so the tool parser can recover them — see
    ``_promote_embedded_tool_calls``. Non-streaming only.
    """

    THINK_START = "<|think:start|>"
    THINK_END = "<|think:end|>"

    # Tool-call sentinels — must stay in sync with ``SolarOpen2ToolParser``.
    # Used only to *recognize* complete embedded tool-call blocks inside
    # reasoning; actually parsing them remains the tool parser's job.
    TOOL_CALL_START = "<|tool_call:start|>"
    TOOL_CALL_END = "<|tool_call:end|>"
    TOOL_ARG_START = "<|tool_arg:start|>"
    TOOL_ARG_END = "<|tool_arg:end|>"

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        end_re = re.escape(self.THINK_END)
        self.reasoning_regex = re.compile(
            rf"^(?P<reasoning>.*?){end_re}(?P<content>.*)$",
            re.DOTALL,
        )

        tc_start = re.escape(self.TOOL_CALL_START)
        tc_end = re.escape(self.TOOL_CALL_END)
        ta_start = re.escape(self.TOOL_ARG_START)
        ta_end = re.escape(self.TOOL_ARG_END)
        # Mirrors ``SolarOpen2ToolParser.tool_call_pattern`` (sans capture
        # groups): only blocks the tool parser can actually parse are
        # promoted out of reasoning.
        self.embedded_tool_call_regex = re.compile(
            rf"{tc_start}.+?\n"
            rf"(?:{ta_start}.*?{ta_end}\n?)*"
            rf"{tc_end}",
            re.DOTALL,
        )

        self._reset_stream()

    def _reset_stream(self) -> None:
        """Reset streaming-only state. Called on init and at the start of
        every new stream (detected via empty ``previous_text``)."""
        self._stream_buffer: str = ""
        self._stream_in_content: bool = False
        # ``True`` once we've handled at least one streaming delta in this
        # request — used to gate the streaming-aware ``is_reasoning_end``
        # behavior so non-streaming and prompt-side callers fall back to
        # the canonical token-decode check.
        self._stream_active: bool = False

    @staticmethod
    def _holdback_suffix(buf: str, sentinels: tuple[str, ...]) -> int:
        """Return the number of trailing bytes of ``buf`` that are a proper,
        non-empty prefix of any sentinel in ``sentinels`` — these bytes
        might complete into the sentinel once more data arrives and must
        stay in the buffer.

        Anchor on the last ``<`` (all Solar Open2 sentinels start with
        ``<|``) and only consider genuine *tail* prefixes; a letter that
        happens to appear inside a sentinel is NOT held back.
        """
        last_lt = buf.rfind("<")
        if last_lt == -1:
            return 0
        tail = buf[last_lt:]
        for s in sentinels:
            if len(tail) < len(s) and s.startswith(tail):
                return len(tail)
        return 0

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Streaming-aware path: defer the flip until our text-level
        # detection has fired. Prevents the chat-completion server from
        # bypassing this parser while partial bytes of ``<|think:end|>``
        # are still in our buffer (or still pending in vLLM's incremental
        # decoder), which would otherwise leak the remaining suffix
        # through to the tool parser as a spurious content delta.
        if self._stream_active:
            return self._stream_in_content
        text = self.model_tokenizer.decode(input_ids)
        # Only the current (last) think block matters. The chat template
        # renders prior assistant turns as
        # ``<|think:start|>...<|think:end|>``, so a plain containment check
        # would fire on a previous turn's end tag in any multi-turn prompt
        # and make the server treat the whole new generation as content.
        last_start = text.rfind(self.THINK_START)
        if last_start == -1:
            # No ``<|think:start|>`` (e.g. raw model output during a
            # non-streaming check, or a non-chat prompt): fall back to the
            # canonical containment check.
            return self.THINK_END in text
        return self.THINK_END in text[last_start + len(self.THINK_START) :]

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return []

    def _promote_embedded_tool_calls(
        self,
        reasoning: str | None,
        content: str | None,
    ) -> tuple[str | None, str | None]:
        """Move tool-call blocks embedded in reasoning into the content
        channel.

        The model occasionally emits complete tool calls *before*
        ``<|think:end|>``. The chat-completion server extracts reasoning
        first and the tool parser only inspects the content channel, so
        such calls would otherwise be silently dropped from the response.
        Mirrors the upstream Qwen3 fix (vllm-project/vllm#39055), with one
        deliberate difference: promoted blocks are appended *after* the
        existing content rather than before it, because
        ``SolarOpen2ToolParser`` keeps only the text preceding the first
        ``<|tool_call:start|>`` as content — blocks placed in front would
        silently drop the real content.

        Only blocks matching the tool parser's full grammar are promoted;
        a sentinel merely *mentioned* in the thinking text (or an
        abandoned/malformed call) stays in reasoning. Non-streaming only —
        the streaming path has already emitted reasoning deltas by the
        time the think block closes, so recovery there needs serving-layer
        support (cf. upstream #40783).
        """
        if not reasoning or self.TOOL_CALL_START not in reasoning:
            return reasoning, content

        blocks: list[str] = []

        def _collect(match) -> str:
            blocks.append(match.group(0))
            return ""

        remaining = self.embedded_tool_call_regex.sub(_collect, reasoning)
        if not blocks:
            return reasoning, content

        parts = [p for p in (content, "\n".join(blocks)) if p]
        return remaining.strip() or None, "\n".join(parts)

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        match = self.reasoning_regex.match(model_output)
        if match:
            reasoning = match.group("reasoning") or None
            content = match.group("content") or None
            return self._promote_embedded_tool_calls(reasoning, content)

        # No <|think:end|> in output (e.g. low-effort prompt where the
        # template already prefilled the empty think pair, or generation
        # truncated mid-think) — treat the whole output as content. Any
        # tool-call blocks are already visible to the tool parser there,
        # so no promotion is needed on this path.
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Fresh stream — ReasoningParser instances are reused across
        # requests so streaming state must be re-initialized.
        if not previous_text:
            self._reset_stream()
        self._stream_active = True

        if delta_text:
            self._stream_buffer += delta_text

        # After the reasoning block closes, the stream is all content.
        if self._stream_in_content:
            if not self._stream_buffer:
                return None
            out = self._stream_buffer
            self._stream_buffer = ""
            return DeltaMessage(content=out)

        # REASONING state. Emit reasoning up to — but not including — any
        # occurrence of ``<|think:end|>``. If the end tag is not yet in
        # the buffer, emit everything except trailing bytes that could be
        # the start of the end tag.
        end_idx = self._stream_buffer.find(self.THINK_END)
        if end_idx >= 0:
            reasoning_out = self._stream_buffer[:end_idx]
            self._stream_buffer = self._stream_buffer[end_idx + len(self.THINK_END) :]
            self._stream_in_content = True
            # If the same chunk carried both the end tag and some content
            # that follows it, emit both on one DeltaMessage — the OpenAI
            # streaming schema allows ``reasoning`` and ``content`` to be
            # set on the same delta, and otherwise the trailing content
            # would be stuck in the buffer if no further deltas arrive
            # (single-chunk / large-chunk synthetic inputs, final token
            # that happens to close reasoning + emit content together).
            content_out = self._stream_buffer
            self._stream_buffer = ""
            if not reasoning_out and not content_out:
                return None
            return DeltaMessage(
                reasoning=reasoning_out or None,
                content=content_out or None,
            )

        hb = self._holdback_suffix(self._stream_buffer, (self.THINK_END,))
        if hb == len(self._stream_buffer):
            # Every byte currently in the buffer could still become the
            # end-tag prefix. Wait.
            return None
        reasoning_out = self._stream_buffer[: len(self._stream_buffer) - hb]
        self._stream_buffer = self._stream_buffer[len(self._stream_buffer) - hb :]
        if not reasoning_out:
            return None
        return DeltaMessage(reasoning=reasoning_out)
