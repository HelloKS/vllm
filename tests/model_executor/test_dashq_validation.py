# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the independent reference and acceptance evidence."""

import json

import torch

from examples.quantization.dashq.make_tiny_checkpoint import make_checkpoint
from examples.quantization.dashq.reference import streamed_model
from examples.quantization.dashq.smoke import graph_sizes


@torch.inference_mode()
def test_streamed_reference_matches_resident_and_reloads(tmp_path):
    folder = tmp_path / "tiny"
    make_checkpoint(folder)
    streamed, reader = streamed_model(folder, "cpu")
    resident, resident_reader = streamed_model(folder, "cpu")
    try:
        # Load each group once, then remove streaming hooks. All resident
        # weights must reproduce the repeatedly loaded reference exactly.
        for module in resident.modules():
            for hook in module._forward_pre_hooks.values():
                hook(module, ())
            module._forward_pre_hooks.clear()
            module._forward_hooks.clear()
        assert not any(p.is_meta for p in resident.parameters())
        gate = resident.model.layers[2].mixer.gate
        assert gate.weight.dtype == torch.float32
        assert gate.e_score_correction_bias.dtype == torch.float32
        sequences = json.loads((folder / "prompts.json").read_text())
        for ids in [*sequences, sequences[0]]:
            tokens = torch.tensor([ids])
            expected = resident(tokens, use_cache=False, logits_to_keep=0).logits
            actual = streamed(tokens, use_cache=False, logits_to_keep=0).logits
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert all(p.is_meta for p in streamed.parameters())
    finally:
        reader.close()
        resident_reader.close()


def test_graph_evidence_excludes_capture_configuration_and_padded_rows():
    assert not graph_sizes('cudagraph_capture_sizes=[1, 2], mode="FULL"')
    assert not graph_sizes("| 1 | 2 | 1 | CUDAGraphMode.FULL | 12 |")
    assert not graph_sizes("| 1 | 1 | 0 | CUDAGraphMode.FULL | 0 |")
    assert not graph_sizes("| 1 | 1 | 0 | CUDAGraphMode.PIECEWISE | 12 |")
    assert graph_sizes(
        "| 1 | 1 | 0 | CUDAGraphMode.FULL | 12 |\n"
        "| 2 | 2 | 0 | CUDAGraphMode.FULL | 11 |"
    ) == {1, 2}
