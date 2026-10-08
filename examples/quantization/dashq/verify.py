# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real vLLM TP logits to the independent streamed reference."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2], default=2)
    parser.add_argument("--backend", choices=["mp", "ray"], default="ray")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--baseline", help="Eager report for decode replay comparison")
    parser.add_argument("--memory-utilization", type=float, default=0.88)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.dashq import checkpoint_fingerprint

    reference = json.loads(Path(args.reference).read_text())
    if (
        not reference.get("complete")
        or len(reference["cases"]) != reference["expected_cases"]
    ):
        raise ValueError("Reference run is incomplete")
    if reference["fingerprint"] != checkpoint_fingerprint(args.model):
        raise ValueError("Reference checkpoint does not match the tested checkpoint")
    engine = LLM(
        model=args.model,
        quantization="dashq",
        load_format="dashq",
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        distributed_executor_backend=args.backend,
        max_model_len=2048,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        enforce_eager=args.eager,
        gpu_memory_utilization=args.memory_utilization,
        skip_tokenizer_init=True,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    )
    prompts = [{"prompt_token_ids": c["input_ids"]} for c in reference["cases"]]
    outputs = engine.generate(
        prompts,
        SamplingParams(
            temperature=0,
            max_tokens=8,
            prompt_logprobs=1,
            ignore_eos=True,
        ),
    )
    if len(outputs) != len(prompts):
        raise RuntimeError("Not all requests completed")
    errors, expected, observed = [], [], []
    cases = []
    for case, output in zip(reference["cases"], outputs):
        logprobs = output.prompt_logprobs
        if logprobs is None or len(logprobs) != len(case["input_ids"]):
            raise RuntimeError("Incomplete vLLM prompt logprobs")
        actual = [
            logprobs[i][token].logprob for i, token in enumerate(case["input_ids"]) if i
        ]
        if len(actual) != len(case["logprobs"]) or not all(map(math.isfinite, actual)):
            raise RuntimeError("Nonfinite/incomplete vLLM results")
        errors.extend(abs(a - b) for a, b in zip(actual, case["logprobs"]))
        observed.extend(actual)
        expected.extend(case["logprobs"])
        cases.append(
            {
                "logprobs": actual,
                "generated_ids": output.outputs[0].token_ids,
                "reference_next_token": case["next_token"],
            }
        )
    if not errors:
        raise RuntimeError("No comparison tokens")
    mean_error = sum(errors) / len(errors)
    nll_relative = abs(sum(observed) - sum(expected)) / max(abs(sum(expected)), 1e-12)
    passed = mean_error <= 0.05 and nll_relative <= 0.01
    decode_match = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        if (
            not baseline.get("passed")
            or not baseline.get("eager")
            or baseline.get("fingerprint") != reference["fingerprint"]
        ):
            raise ValueError("Baseline must be a passing eager run of this checkpoint")
        decode_match = [c["generated_ids"] for c in cases] == [
            c["generated_ids"] for c in baseline["cases"]
        ]
        passed = passed and decode_match
    report = dict(
        passed=passed,
        tp=args.tp,
        eager=args.eager,
        mean_absolute_logprob_error=mean_error,
        relative_nll_error=nll_relative,
        fingerprint=reference["fingerprint"],
        decode_matches_eager=decode_match,
        cases=cases,
    )
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
