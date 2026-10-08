# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.adapters import ParserEngineToolAdapter
from vllm.parser.motif import MotifParser


class MotifToolParser(ParserEngineToolAdapter):
    _parser_engine_cls = MotifParser
    structural_tag_model = None
