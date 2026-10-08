# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent, slow PyTorch reference with one resident Transformers layer.

This intentionally does not import the vLLM DASH-Q kernels. Run on an idle
node, separately from the serving cluster. Packed experts remain compressed;
only the matrix being multiplied is dequantized.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, NemotronHConfig, NemotronHForCausalLM

from vllm.transformers_utils.dashq import (
    DashQTensorReader,
    checkpoint_fingerprint,
    validate_metadata,
)

PROMPTS = [
    "Explain why the sky is blue using three complete sentences.",
    "Describe the difference between a stack and a queue with examples.",
    "서울에서 부산까지 여행할 때 고려할 교통수단을 설명해 주세요.",
    "Write a Python function that returns the sum of even numbers in a list.",
    "If a train travels 120 kilometers in two hours, explain its average speed.",
    "Compare renewable and nonrenewable energy in a short paragraph.",
    "Explain tensor parallelism and how two GPUs cooperate during inference.",
    "Give a step by step explanation of how to make a cup of tea.",
]


class ReferenceLinear(nn.Module):
    def __init__(self, meta):
        super().__init__()
        self.n, self.k = meta["out_features"], meta["in_features"]
        self.register_buffer(
            "W_q_packed",
            torch.empty(self.n * self.k // 16, dtype=torch.int32, device="meta"),
        )
        for name in ("scale", "zero"):
            self.register_buffer(
                name,
                torch.empty(
                    self.n * self.k // 32, 1, dtype=torch.bfloat16, device="meta"
                ),
            )

    def forward(self, x):
        shifts = torch.arange(16, device=x.device, dtype=torch.int32) * 2
        q = ((self.W_q_packed[:, None] >> shifts) & 3).reshape(self.n, self.k)
        s = self.scale.reshape(self.n, -1).float().repeat_interleave(32, -1)
        z = self.zero.reshape(self.n, -1).float().repeat_interleave(32, -1)
        w = (q.float() - z) * s
        if x.numel() // self.k == 1:
            return F.linear(x.float(), w).to(x.dtype)
        return F.linear(x, w.to(x.dtype))


class ReferenceExperts(nn.Module):
    def __init__(self, prefix, count, metadata):
        super().__init__()
        self.up_proj_list = nn.ModuleList(
            [
                ReferenceLinear(metadata[f"{prefix}.up_proj_list.{e}"])
                for e in range(count)
            ]
        )
        self.down_proj_list = nn.ModuleList(
            [
                ReferenceLinear(metadata[f"{prefix}.down_proj_list.{e}"])
                for e in range(count)
            ]
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states, dtype=torch.float32)
        for e, (up, down) in enumerate(zip(self.up_proj_list, self.down_proj_list)):
            rows, slots = torch.where(top_k_index == e)
            if rows.numel():
                y = down(up(hidden_states[rows]).relu().square())
                y = y.float() * top_k_weights[rows, slots, None]
                result.index_add_(0, rows, y)
        return result.to(hidden_states.dtype)


def streamed_model(folder, device):
    folder = Path(folder)
    with (folder / "dashq_config.json").open() as f:
        metadata = validate_metadata(json.load(f))
    config = NemotronHConfig.from_pretrained(folder)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = NemotronHForCausalLM(config).to(dtype=torch.bfloat16).eval()
    # Match the native GB10 router and Mamba A storage. In particular, rounding
    # correction bias to BF16 can change the selected experts near a tie.
    for name, parameter in model.named_parameters():
        if name.endswith((".gate.weight", ".A_log")):
            parameter.data = parameter.data.float()
    for name, buffer in list(model.named_buffers()):
        if name.endswith(".e_score_correction_bias"):
            parent, leaf = name.rsplit(".", 1)
            model.get_submodule(parent)._buffers[leaf] = buffer.float()
    expert_prefixes = {
        name.split(".up_proj_list.")[0] for name in metadata if ".up_proj_list." in name
    }
    for prefix in expert_prefixes:
        parent, child = prefix.rsplit(".", 1)
        setattr(
            model.get_submodule(parent),
            child,
            ReferenceExperts(prefix, config.n_routed_experts, metadata),
        )
    for name, meta in metadata.items():
        if ".experts." not in name:
            parent, child = name.rsplit(".", 1)
            old = getattr(model.get_submodule(parent), child)
            if getattr(old, "bias", None) is not None:
                raise ValueError(f"Reference does not support quantized bias: {name}")
            setattr(model.get_submodule(parent), child, ReferenceLinear(meta))

    reader = DashQTensorReader(folder)

    def prehook(prefix):
        def load(module, _args):
            # Snapshot names: replacing parameters must not mutate the iterator.
            params = dict(module.named_parameters())
            buffers = dict(module.named_buffers())
            for name, target in {**params, **buffers}.items():
                full_name = f"{prefix}.{name}"
                if full_name not in reader.index:
                    if not target.is_meta:
                        continue
                    raise ValueError(f"Reference tensor missing: {full_name}")
                tensor = reader.tensor(full_name).to(device=device, dtype=target.dtype)
                parent, _, leaf = name.rpartition(".")
                owner = module.get_submodule(parent) if parent else module
                if name in params:
                    owner._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
                else:
                    owner._buffers[leaf] = tensor

        return load

    def unload(module, _args, output):
        module.to_empty(device="meta")
        return output

    groups = [("model.embeddings", model.model.embeddings)]
    groups += [
        (f"model.layers.{i}", layer) for i, layer in enumerate(model.model.layers)
    ]
    groups += [("model.norm_f", model.model.norm_f), ("lm_head", model.lm_head)]
    for prefix, layer in groups:
        layer.register_forward_pre_hook(prehook(prefix))
        layer.register_forward_hook(unload)
    return model, reader


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local pinned HF snapshot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prompts", help="JSON list of input token ID lists (tiny fixture)"
    )
    args = parser.parse_args()
    if args.prompts:
        sequences = json.loads(Path(args.prompts).read_text())
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        sequences = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in PROMPTS
        ]
    model, reader = streamed_model(args.model, args.device)
    result = {
        "model": str(Path(args.model).resolve()),
        "reference": "streamed-pytorch",
        "torch": torch.__version__,
        "fingerprint": checkpoint_fingerprint(args.model),
        "expected_cases": len(sequences),
        "complete": False,
        "cases": [],
    }
    try:
        for i, ids in enumerate(sequences):
            tokens = torch.tensor([ids], device=args.device)
            logits = model(tokens, use_cache=False, logits_to_keep=0).logits[0]
            logp = logits[:-1].float().log_softmax(-1)
            target = tokens[0, 1:]
            values = logp.gather(1, target[:, None]).squeeze(1)
            if not torch.isfinite(values).all():
                raise RuntimeError("Nonfinite reference logits")
            result["cases"].append(
                {
                    "input_ids": ids,
                    "logprobs": values.cpu().tolist(),
                    "next_token": logits[-1].argmax().item(),
                }
            )
            Path(args.output).write_text(json.dumps(result, indent=2))
            print(f"Reference {i + 1}/{len(sequences)} complete", flush=True)
    finally:
        reader.close()
    result["complete"] = True
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
