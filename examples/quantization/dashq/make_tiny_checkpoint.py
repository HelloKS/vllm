# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make a small Mamba + attention + latent MoE checkpoint for TP regression.

Synthetic quantization exercises the storage contract, not DASH-Q's calibration
algorithm. No production weights or tokenizer downloads are needed.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import NemotronHConfig, NemotronHForCausalLM


def make_checkpoint(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(735)
    config = NemotronHConfig(
        vocab_size=256,
        hidden_size=128,
        layers_block_type=["linear_attention", "full_attention", "moe"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        mamba_num_heads=8,
        mamba_head_dim=32,
        ssm_state_size=16,
        n_groups=2,
        chunk_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_latent_size=64,
        moe_shared_expert_overlap=False,
        max_position_embeddings=2048,
        num_nextn_predict_layers=0,
    )
    model = NemotronHForCausalLM(config).bfloat16().eval()
    state = {
        name: value.detach().contiguous() for name, value in model.state_dict().items()
    }
    metadata = {}

    def quantize(name, weight):
        n, k = weight.shape
        groups = weight.float().reshape(n, k // 32, 32)
        lo, hi = groups.amin(-1), groups.amax(-1)
        scale = ((hi - lo) / 3).clamp_min(1e-5).half()
        zero = (-lo / scale.float()).half()
        codes = groups / scale.float()[..., None] + zero.float()[..., None]
        codes = codes.round().clamp(0, 3).to(torch.int64).reshape(n, k // 16, 16)
        packed = (codes << (torch.arange(16) * 2)).sum(-1).to(torch.int32)
        state[f"{name}.W_q_packed"] = packed.flatten()
        state[f"{name}.scale"] = scale.reshape(-1, 1)
        state[f"{name}.zero"] = zero.reshape(-1, 1)
        metadata[name] = dict(
            nbits=2,
            group_size=32,
            packing="int2_packed_u32",
            scale_zero_dtype="float16",
            linear_dtype="bfloat16",
            in_features=k,
            out_features=n,
            quant_in_features=k,
            num_groups=n * k // 32,
        )

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name != "lm_head":
            quantize(name, state.pop(name + ".weight"))
    for name in list(state):
        if name.endswith((".experts.up_proj", ".experts.down_proj")):
            weights = state.pop(name)
            for expert in range(weights.shape[0]):
                quantize(f"{name}_list.{expert}", weights[expert])
    config.architectures = ["DashQNemotronHForCausalLM"]
    config.dtype = torch.bfloat16
    config.dashq = dict(
        format="dashq-packed-linear",
        format_version=1,
        method="dashq",
        n_quantized_modules=len(metadata),
    )
    config.to_json_file(folder / "config.json")
    (folder / "dashq_config.json").write_text(
        json.dumps(
            {
                "format": "dashq-packed-linear",
                "format_version": 1,
                "quantized_modules": metadata,
            },
            indent=2,
        )
    )
    save_file(state, str(folder / "model.safetensors"))
    (folder / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        t.numel() * t.element_size() for t in state.values()
                    )
                },
                "weight_map": {name: "model.safetensors" for name in state},
            }
        )
    )
    (folder / "prompts.json").write_text(
        json.dumps(
            [[1, *torch.randint(3, 256, (length,)).tolist()] for length in (7, 31, 63)]
        )
    )
    print(f"Wrote {len(metadata)} quantized modules to {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    make_checkpoint(args.output)


if __name__ == "__main__":
    main()
