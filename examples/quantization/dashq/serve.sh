#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the identical pinned snapshot path on both nodes}"
: "${VLLM_HOST_IP:?Set VLLM_HOST_IP to the head node cluster interface IP}"
mkdir -p "${REPORT_DIR:-dashq-reports}"
args=(
  serve "$MODEL_DIR"
  --served-model-name dashq-nemotron
  --quantization dashq --load-format dashq --dtype bfloat16
  --tensor-parallel-size 2 --pipeline-parallel-size 1
  --distributed-executor-backend ray --disable-custom-all-reduce
  --max-model-len 2048 --max-num-seqs 2 --max-num-batched-tokens 128
  --enable-chunked-prefill --no-enable-prefix-caching
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}"
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'
  --cudagraph-metrics --host 127.0.0.1 --port "${PORT:-8000}"
)
if [[ "${EAGER:-0}" == 1 ]]; then
  args+=(--enforce-eager)
fi
uv run --no-project .venv/bin/python -m vllm.entrypoints.cli.main "${args[@]}" \
  2>&1 | tee "${REPORT_DIR:-dashq-reports}/server.log"
