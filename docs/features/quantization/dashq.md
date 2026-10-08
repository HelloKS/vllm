# DASH-Q INT2/g32 on two GB10 nodes

This implementation loads the original
`jkim96/Nemotron-3-Ultra-550B-A55B-DASHQ-INT2-g32` checkpoint into native vLLM
NemotronH layers. It preserves packed INT2 weights for Linear and routed experts,
uses Triton for prefill and decode, and supports TP=1/2 with BF16 activations.
It is experimental: a successful CPU test is not evidence of GB10 serving or
CUDA Graph correctness. Run the gates below before treating it as operational.

Supported profile: two Linux ARM64 GB10 128GB nodes, one rank per node, TP=2,
PP=DP=1, no EP/SP/EPLB, LoRA, or speculative decoding. The initial serving
profile has a 2,048-token context, two sequences, 128 prefill tokens per batch,
and full decode graphs at batch sizes 1 and 2. Prefill runs eagerly.

Local validation on 2026-10-08: 21 CPU tests passed on Windows with PyTorch
2.14.1+cpu and Transformers 5.19.0, including exact streamed-versus-resident
reference comparison. The three-prompt tiny reference run, Ruff, repository
import/API/header checks, Python syntax, Bash syntax, and Markdown lint passed.
The CUDA kernel cases, native Linux vLLM loading, full-model accuracy, TP=2,
memory peaks, and API/CUDA Graph serving have **not** been verified on GB10 yet.

## TorchInductor launch argument fix (2026-10-09)

A production-checkpoint run reported a Triton compilation failure at
`pm * BM + tl.arange(0, BM)` with `NoneType ... type` through TorchInductor.
The original launches omitted tile arguments and relied on JIT signature
defaults. All five kernels now require explicit tile constexprs, and every
launch supplies them. Six GPU regression cases cover `torch.compile` for Linear
and MoE at batch sizes 1, 2 and 33. The fix still requires GB10 runtime validation.

For the first retry use the small serving profile below with `EAGER=1`, then
validate graphs separately. Large contexts, concurrency and prefill batches
should be increased only after the accuracy and memory gates pass. A compiled
run and an eager run exercise different compiler paths; successful eager
inference alone does not verify the Inductor regression.

## Dynamic token dimension fix (2026-10-09)

A subsequent default-compiler run specialized the dynamic token dimension to
32768 during AOT guard generation. The serving path now uses opaque
`vllm::dashq_linear` and `vllm::dashq_moe` custom ops with symbolic fake output
shapes. Batch-size dispatch and Triton constexprs stay inside the runtime op;
vLLM's dynamic-shape constraints are retained. Raw Triton entry points are still
tested separately for the earlier explicit-constexpr fix.

Two CPU strict-export regression tests pass for a 32768-token example and retain
variable output shapes at 1, 2, 31, 32, 33, 128 and 32768 tokens. These validate
operator registration and symbolic shapes, not CUDA execution. GPU tests also
cover strict dynamic Inductor execution and CUDA Graph replay through the new
ops. Full GB10 TP=2 validation remains pending.

## Build and pin the environment

Use the same source checkout and absolute environment path on both nodes, CUDA
13.0 development tools, a compatible driver, and the repository's ARM64 build
prerequisites. Do not overlay these sources onto an unrelated installed vLLM
wheel. The native extension must match this checkout.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=4 NVCC_THREADS=2 \
  uv pip install --no-build-isolation -e . --torch-backend=cu130
uv pip install -r requirements/test/cuda.in -r requirements/lint.txt
uv pip install 'ray[default]'
pre-commit install
mkdir -p dashq-reports
uv pip freeze > dashq-reports/environment.lock.txt
git rev-parse HEAD > dashq-reports/base-commit.txt
git diff > dashq-reports/implementation.patch
```

The repository pins PyTorch and native dependencies. Copy the resolved lock to
the second node and use `uv pip sync` against that lock (including the editable
checkout at the identical absolute path). Record the final diff and added files
when distributing an uncommitted implementation; `git diff` alone omits untracked
files. Build before loading the 210GB model. GB10's CPU/GPU memory is shared.

## Gate 1: small tests before downloading production weights

```bash
uv run --no-project .venv/bin/python -m pytest \
  tests/model_executor/test_dashq_format.py \
  tests/model_executor/test_dashq_validation.py \
  tests/model_executor/test_dashq_compile.py \
  tests/kernels/quantization/test_dashq.py -v
uv run --no-project .venv/bin/python -m examples.quantization.dashq.make_tiny_checkpoint \
  /data/dashq-tiny
uv run --no-project .venv/bin/python -m examples.quantization.dashq.reference \
  /data/dashq-tiny --prompts /data/dashq-tiny/prompts.json \
  --output dashq-reports/tiny-reference.json
uv run --no-project .venv/bin/python -m examples.quantization.dashq.verify \
  /data/dashq-tiny --reference dashq-reports/tiny-reference.json \
  --tp 1 --backend mp --eager --output dashq-reports/tiny-tp1.json
```

The synthetic checkpoint contains Mamba, attention, latent projections, shared
experts and routed experts in the production storage format. It does not claim
to reproduce DASH-Q calibration quality. The reference streams each Transformers
layer independently and uses PyTorch quantized-weight reconstruction, not the
new serving kernels. It also understands checkpoints whose experts are stored
as `up_proj_list`/`down_proj_list`, even when Transformers uses fused 3D experts.

Copy the tiny checkpoint and reference JSON to the second node before TP=2.
Run `verify` with `--tp 2 --backend ray` after the next gate, first with `--eager`,
then without it and with `--baseline` pointing to the TP=2 eager report.
Compare the generated IDs and logprobs in TP=1/2 reports.
The CUDA tests separately change routes during 100 graph replays.

## Gate 2: transport and Ray

Use the cluster IPs and RDMA interface names already configured for the direct
ConnectX link. SSH/control-plane reachability alone does not prove NCCL uses
that link. Set `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and, if necessary,
`NCCL_IB_HCA` to your actual interfaces on **both** nodes. Use `NCCL_DEBUG=INFO`
for preflight and inspect transport selection; do not silently accept a socket
fallback as validation of the RDMA setup.

On both nodes, with `NODE_RANK=0` on the head and `NODE_RANK=1` on the worker:

```bash
uv run --no-project .venv/bin/python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$HEAD_IP" --master-port=29511 \
  -m examples.quantization.dashq.preflight --output-dir dashq-reports
```

This checks actual two-rank all-reduce and ten NCCL CUDA Graph replays and writes
rank-specific device, version and available-memory reports.

Start Ray under each node's identical virtual environment:

```bash
# Head, with VLLM_HOST_IP set to HEAD_IP:
.venv/bin/ray start --head --node-ip-address="$VLLM_HOST_IP" --port=6379
# Worker, with VLLM_HOST_IP set to the worker cluster IP:
.venv/bin/ray start --address="$HEAD_IP:6379" --node-ip-address="$VLLM_HOST_IP"
# Head:
.venv/bin/ray status
```

Both nodes must advertise one GPU. Keep the same environment variables and
checkout available to the Ray workers. No SSH configuration is modified by the
provided scripts.

## Gate 3: pinned full checkpoint and accuracy

Download on each node to the same local path:

```bash
.venv/bin/hf download jkim96/Nemotron-3-Ultra-550B-A55B-DASHQ-INT2-g32 \
  --revision 6305602dcc8c33cda5cde046d8043a3e7bc78eac \
  --local-dir /data/dashq-nemotron
export MODEL_DIR=/data/dashq-nemotron
```

The loader reads the safetensors index lazily, transposes packed words without
unpacking an integer matrix, and stages at most 64MiB of source data per chunk.
The transpose can temporarily require another chunk. QKV and Mamba projections
are split by logical segment; expert intermediate dimensions are TP-sharded;
latent projections are replicated. All quantized modules and remaining model
parameters must be accounted for. Missing or incompatible tensors fail loading.

Loading displays a tensor-count progress bar with elapsed time, ETA and loading
rate, using the same text format and rank-zero policy as the safetensors loader.
It covers packed weights, scale/zero tensors and the remaining unquantized
weights, and respects `LoadConfig.use_tqdm_on_load`. Progress counts completed
tensors rather than bytes, so differently sized tensors take different times.

Run the slow full reference on an idle node, **before** serving. Only one layer
is resident; do not try to load the whole 550B Transformers model on one GB10.

```bash
uv run --no-project .venv/bin/python -m examples.quantization.dashq.reference \
  "$MODEL_DIR" --output dashq-reports/full-reference.json
uv run --no-project .venv/bin/python -m examples.quantization.dashq.verify \
  "$MODEL_DIR" --reference dashq-reports/full-reference.json \
  --eager --output dashq-reports/full-tp2-eager.json
uv run --no-project .venv/bin/python -m examples.quantization.dashq.verify \
  "$MODEL_DIR" --reference dashq-reports/full-reference.json \
  --baseline dashq-reports/full-tp2-eager.json \
  --output dashq-reports/full-tp2-graph.json
```

Run these sequentially, releasing engine workers between runs. Reference files
are marked complete only after every prompt succeeds; metadata fingerprints must
match. Acceptance is mean absolute target-token logprob error <=0.05 nat and
relative mean NLL error <=1%. Failed accuracy checks require diagnosis, not
automatic threshold relaxation. The kernel tests require relative L2 error <1%.

## Gate 4: user-run serving smoke test

On the head, after the previous engine has exited:

```bash
# First run with EAGER=1, then restart without it for graph validation.
bash examples/quantization/dashq/serve.sh
```

In a second head-node terminal:

```bash
uv run --no-project .venv/bin/python -m examples.quantization.dashq.smoke \
  --server-log dashq-reports/server.log --output dashq-reports/smoke-graph.json
```

For the eager server use `--eager --output dashq-reports/smoke-eager.json`.
The smoke test covers streaming/non-streaming chat, a 1,024-token prompt,
128 generated tokens, three rounds of concurrent requests, cancellation, and
recovery. It fails graph validation unless **observed runtime** FULL graph
dispatch rows for unpadded batches 1 and 2 appear in the new server log output.
Requested capture sizes or successful server startup do not count as proof.

Monitor both nodes' `nvidia-smi`, `/proc/meminfo`, swap activity, and server logs
during load, graph capture and repeated requests. Record peaks and check that
available memory stabilizes. Do not add CPU offload to resolve unified-memory
pressure. The loader reports rank-local weight bytes; actual state/workspace/
graph/OS usage determines whether the full profile fits. `GPU_MEMORY_UTILIZATION`
is an explicit launch override, default 0.88; raising it without system headroom
is not a validated fix for OOM.

Keep both rank preflight reports, environment lock, eager/graph accuracy reports,
smoke reports, server logs and memory observations. Only after all gates pass is
the status **GB10 TP=2 verified**. Until then, report the last completed gate and
the exact failure. TTFT and request durations are recorded; there is no promised
tokens/second target for this initial implementation.
