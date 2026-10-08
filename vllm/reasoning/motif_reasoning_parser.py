# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.motif import MotifParser


class MotifReasoningParser(ParserEngineReasoningAdapter):
    _parser_engine_cls = MotifParser
