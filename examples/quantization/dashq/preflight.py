# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""User-run, two-node NCCL/graph preflight. Launch using torchrun on both nodes."""

import argparse
import json
import os
import platform
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError(
            "Launch on both nodes with torchrun --nnodes=2 --nproc-per-node=1"
        )
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    try:
        tensor = torch.full((4096,), float(rank + 1), device="cuda")
        for _ in range(3):
            tensor.fill_(rank + 1)
            dist.all_reduce(tensor)
        torch.accelerator.synchronize()
        if not (tensor == 3).all():
            raise RuntimeError("Incorrect NCCL all-reduce result")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.all_reduce(tensor)
        for _ in range(10):
            tensor.fill_(rank + 1)
            graph.replay()
            torch.accelerator.synchronize()
            if not (tensor == 3).all():
                raise RuntimeError("Incorrect NCCL graph replay")
        free, total = torch.accelerator.memory.get_memory_info()
        report = dict(
            rank=rank,
            hostname=platform.node(),
            machine=platform.machine(),
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            nccl=torch.cuda.nccl.version(),
            free_bytes=free,
            total_bytes=total,
            nccl_graph_passed=True,
        )
        report["meminfo"] = Path("/proc/meminfo").read_text()
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"preflight-rank-{rank}.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
