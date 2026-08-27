# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.models.glm5next.nvidia import kda as glm_kda


def test_kda_missing_warmup_metadata_is_noop(monkeypatch) -> None:
    layer = object.__new__(glm_kda.Glm5NextLinearAttention)
    object.__setattr__(
        layer,
        "prefix",
        "language_model.model.layers.0.self_attn",
    )
    monkeypatch.setattr(
        glm_kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={}),
    )

    # The method returns before touching tensor arguments when a CUDA graph
    # dummy run intentionally omits mamba-family metadata.
    glm_kda.Glm5NextLinearAttention._forward(
        layer,
        qkv_proj_states=None,  # type: ignore[arg-type]
        g1=None,  # type: ignore[arg-type]
        beta=None,  # type: ignore[arg-type]
        core_attn_out=None,  # type: ignore[arg-type]
    )
