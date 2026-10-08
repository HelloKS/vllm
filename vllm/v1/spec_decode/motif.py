# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer


class MotifMTPProposer(Step3p5MTPProposer):
    """Select Motif's MTP head per draft step and share the target LM head."""

    def _maybe_share_lm_head(self, target_language_model: torch.nn.Module) -> None:
        SpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)
