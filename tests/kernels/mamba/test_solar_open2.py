# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.mamba import mamba_utils
from vllm.model_executor.layers.mamba.gdn import solar_open2_linear_attn as solar
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Solar KDA kernels require CUDA"
)

# Solar's head dimension, with fewer local heads to keep the test small.
H, D, WIDTH, SPEC = 2, 128, 4, 7
PREFIX = "model.layers.1.self_attn"


def _tensor(data):
    return torch.tensor(data, dtype=torch.int32, device="cuda")


def _metadata(**kwargs):
    counts: dict[str, Any] = dict(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=0,
    )
    counts.update(kwargs)
    return GDNAttentionMetadata(**counts)


def _run(layer, metadata, projections, g, beta):
    out = torch.zeros(1, g.shape[1], H, D, dtype=torch.bfloat16, device="cuda")
    ctx = SimpleNamespace(attn_metadata={PREFIX: metadata})
    with patch.object(solar, "get_forward_context", return_value=ctx):
        solar.SolarOpen2KimiDeltaAttention._forward(
            layer,
            q_proj_states=projections[0].clone(),
            k_proj_states=projections[1].clone(),
            v_proj_states=projections[2].clone(),
            g1=g,
            beta=beta,
            core_attn_out=out,
        )
    return out


def _decode(layer, projections, g, beta, block=1):
    outputs = []
    for t in range(g.shape[1]):
        metadata = _metadata(
            num_decodes=1,
            num_decode_tokens=1,
            num_actual_tokens=1,
            non_spec_state_indices_tensor=_tensor([block]),
            non_spec_query_start_loc=_tensor([0, 1]),
        )
        outputs.append(
            _run(
                layer,
                metadata,
                [x[t : t + 1] for x in projections],
                g[:, t : t + 1],
                beta[:, t : t + 1],
            )
        )
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize("accepted", [1, 4, 8])
@pytest.mark.parametrize("mixed_prefill", [False, True])
def test_solar_kda_rejection_matches_sequential_decode(
    monkeypatch, layout, accepted, mixed_prefill
):
    """Verify zero/partial/full acceptance and mixed-prefill output placement.

    The accepted count includes the verifier's anchor token. Compare the next
    verification against ordinary decoding of only the previous accepted prefix.
    """
    monkeypatch.setattr(mamba_utils, "get_conv_state_layout", lambda: layout)
    torch.manual_seed(0)
    blocks = SPEC + 3
    conv = torch.zeros(
        blocks, 3 * H * D, WIDTH - 1 + SPEC, dtype=torch.bfloat16, device="cuda"
    )
    if not mamba_utils.is_conv_state_dim_first():
        conv = conv.transpose(-1, -2).contiguous()
    state = torch.zeros(blocks, H, D, D, device="cuda")
    weights = [
        SimpleNamespace(
            weight=torch.randn(H * D, 1, WIDTH, device="cuda") * 0.1, bias=None
        )
        for _ in range(3)
    ]
    layer = SimpleNamespace(
        prefix=PREFIX,
        local_num_heads=H,
        head_dim=D,
        q_conv1d=weights[0],
        k_conv1d=weights[1],
        v_conv1d=weights[2],
        kv_cache=(conv, state),
    )
    reference = SimpleNamespace(**vars(layer))
    reference.kv_cache = (conv.clone(), state.clone())
    total = 2 * (SPEC + 1)
    projections = [
        torch.randn(total, H * D, dtype=torch.bfloat16, device="cuda") for _ in range(3)
    ]
    g = -torch.rand(1, total, H, D, device="cuda")
    # Exercise Solar's beta > 1 (negative eigenvalues), unlike standard GDN.
    beta = 2 * torch.rand(1, total, H, device="cuda")
    spec_metadata = _metadata(
        num_spec_decodes=1,
        num_spec_decode_tokens=SPEC + 1,
        num_actual_tokens=SPEC + 1,
        spec_sequence_masks=torch.tensor([True], device="cuda"),
        spec_query_start_loc=_tensor([0, SPEC + 1]),
        spec_state_indices_tensor=_tensor([list(range(1, SPEC + 2))]),
        num_accepted_tokens=_tensor([1]),
    )
    _run(
        layer,
        spec_metadata,
        [x[: SPEC + 1] for x in projections],
        g[:, : SPEC + 1],
        beta[:, : SPEC + 1],
    )
    _decode(
        reference,
        [x[:accepted] for x in projections],
        g[:, :accepted],
        beta[:, :accepted],
    )
    torch.testing.assert_close(state[accepted], reference.kv_cache[1][1])

    projections = [x[SPEC + 1 :] for x in projections]
    g, beta = g[:, SPEC + 1 :], beta[:, SPEC + 1 :]
    expected = _decode(reference, projections, g, beta)
    spec_metadata.num_accepted_tokens = _tensor([accepted])
    spec_positions = list(range(SPEC + 1))
    if mixed_prefill:
        # Insert a fresh two-token prefill inside the spec tokens.
        ns_positions = [1, 4]
        spec_positions = [i for i in range(SPEC + 3) if i not in ns_positions]
        ns_projections = [x[:2].clone() for x in projections]
        ns_g, ns_beta = g[:, :2].clone(), beta[:, :2].clone()
        expected_ns = _decode(reference, ns_projections, ns_g, ns_beta, block=SPEC + 2)
        for i, x in enumerate(projections):
            merged = x.new_empty(SPEC + 3, H * D)
            merged[spec_positions], merged[ns_positions] = x, ns_projections[i]
            projections[i] = merged
        merged_g = g.new_empty(1, SPEC + 3, H, D)
        merged_beta = beta.new_empty(1, SPEC + 3, H)
        merged_g[:, spec_positions], merged_g[:, ns_positions] = g, ns_g
        merged_beta[:, spec_positions], merged_beta[:, ns_positions] = beta, ns_beta
        g, beta = merged_g, merged_beta
        spec_metadata.num_prefills = 1
        spec_metadata.num_prefill_tokens = 2
        spec_metadata.num_actual_tokens += 2
        spec_metadata.spec_token_indx = _tensor(spec_positions)
        spec_metadata.non_spec_token_indx = _tensor(ns_positions)
        spec_metadata.non_spec_query_start_loc = _tensor([0, 2])
        spec_metadata.non_spec_state_indices_tensor = _tensor([SPEC + 2])
        spec_metadata.has_initial_state = torch.tensor([False], device="cuda")
        (
            spec_metadata.nums_dict,
            spec_metadata.batch_ptr,
            spec_metadata.token_chunk_offset_ptr,
        ) = compute_causal_conv1d_metadata(
            torch.tensor([0, 2], dtype=torch.int32), device=torch.device("cuda")
        )
    else:
        # Full CUDA graphs can include token rows with no corresponding query.
        projections = [torch.cat([x, x[:3]]) for x in projections]
        g = torch.cat([g, g[:, :3]], dim=1)
        beta = torch.cat([beta, beta[:, :3]], dim=1)
        spec_metadata.num_actual_tokens += 3
    actual = _run(layer, spec_metadata, projections, g, beta)
    torch.testing.assert_close(
        actual[:, spec_positions], expected, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        state[SPEC + 1], reference.kv_cache[1][1], atol=2e-2, rtol=2e-2
    )
    if mixed_prefill:
        torch.testing.assert_close(
            actual[:, ns_positions], expected_ns, atol=2e-2, rtol=2e-2
        )
    else:
        torch.testing.assert_close(
            actual[:, SPEC + 1 :], torch.zeros_like(actual[:, SPEC + 1 :])
        )
