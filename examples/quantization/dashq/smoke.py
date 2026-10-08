# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""User-run API acceptance checks; graph success requires runtime log evidence."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

import regex as re


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=1800)


def chat(base, model, stream=False, cancel=False):
    body = dict(
        model=model,
        temperature=0,
        max_tokens=128,
        ignore_eos=True,
        stream=stream,
        messages=[
            {
                "role": "user",
                "content": "Explain how tensor parallelism works, with an example.",
            }
        ],
    )
    started = time.monotonic()
    first = None
    text = ""
    if not stream:
        with request(base, "/v1/chat/completions", body) as response:
            result = json.load(response)
        choice = result["choices"][0]
        message = choice["message"]
        text = (message.get("content") or "") + (message.get("reasoning") or "")
        text += message.get("reasoning_content") or ""
        if result["usage"]["completion_tokens"] != 128:
            raise RuntimeError("Expected 128 generated tokens")
    else:
        done = False
        with request(base, "/v1/chat/completions", body) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                for choice in chunk.get("choices", []):
                    delta = choice["delta"]
                    token = (
                        delta.get("content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or ""
                    )
                    if token:
                        first = first or time.monotonic()
                        text += token
                        if cancel:
                            return {"cancelled_after_first_token": True}
        if not done:
            raise RuntimeError("Streaming response ended without [DONE]")
    if not text.strip():
        raise RuntimeError("Empty model output")
    return dict(
        stream=stream,
        seconds=time.monotonic() - started,
        ttft_seconds=None if first is None else first - started,
        text=text,
    )


def graph_sizes(log):
    # Parse observed dispatch rows, not the requested capture configuration.
    pattern = (
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|"
        r"\s*CUDAGraphMode\.FULL\s*\|\s*(\d+)\s*\|"
    )
    return {
        int(padded)
        for unpadded, padded, count in re.findall(pattern, log)
        if int(count) > 0 and int(unpadded) == int(padded)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="dashq-nemotron")
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-log", help="Fresh head-node server log from this run")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if not args.eager and not args.server_log:
        parser.error("Graph validation requires --server-log")
    # Ignore previous runs in an append-only log.
    log_start = Path(args.server_log).stat().st_size if args.server_log else 0
    report = {"api_passed": False, "graph_passed": False, "cases": []}
    try:
        with request(args.base_url, "/health") as response:
            assert response.status == 200
        report["cases"].append(chat(args.base_url, args.model))
        report["cases"].append(chat(args.base_url, args.model, stream=True))
        with request(
            args.base_url,
            "/tokenize",
            {
                "model": args.model,
                "prompt": "Explain parallel inference. " * 1024,
            },
        ) as response:
            ids = json.load(response)["tokens"][:1024]
        if len(ids) != 1024:
            raise RuntimeError("Could not prepare a 1024-token prompt")
        with request(
            args.base_url,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": ids,
                "max_tokens": 128,
                "temperature": 0,
                "ignore_eos": True,
            },
        ) as response:
            long_result = json.load(response)
        if (
            long_result["usage"]["prompt_tokens"] != 1024
            or long_result["usage"]["completion_tokens"] != 128
        ):
            raise RuntimeError("Long-prompt generation did not complete")
        report["long_prompt_usage"] = long_result["usage"]
        for _ in range(3):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(chat, args.base_url, args.model, True) for _ in range(2)
                ]
                report["cases"].extend(f.result() for f in futures)
        report["cases"].append(
            chat(args.base_url, args.model, stream=True, cancel=True)
        )
        report["cases"].append(chat(args.base_url, args.model))
        report["api_passed"] = True
        if args.server_log:
            time.sleep(12)  # Allow the server's aggregate metrics logger to flush.
            with Path(args.server_log).open("rb") as f:
                f.seek(log_start)
                observed = graph_sizes(f.read().decode(errors="replace"))
            report["observed_full_graph_sizes"] = sorted(observed)
            report["graph_passed"] = {1, 2}.issubset(observed)
            if not args.eager and not report["graph_passed"]:
                raise RuntimeError(
                    "No runtime FULL graph evidence for both batches 1 and 2"
                )
        with request(args.base_url, "/metrics") as response:
            report["server_metrics"] = response.read().decode()
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("cases", "server_metrics")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
