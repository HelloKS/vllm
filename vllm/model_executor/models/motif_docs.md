# Motif Inference Architecture

## Integration into this checkout

Ported from [MotifTechnologies/vllm-motif3-v0.26.0](https://github.com/MotifTechnologies/vllm-motif3-v0.26.0/commit/9ad41a8ebcc52c8fbc940947a0022b0b974b664d)
onto vLLM revision `1995c0fcd0`. The architecture and training-parity notes below
come from that fork; they are not new numerical validation of this port.

The port includes GDLA/sliding-window attention, PolyNorm experts, Motif mHC,
MTP, reasoning/tool parsers, repetition truncation, thinking budgets, and the
BF16, MXFP8, block-FP8 and NVFP4 expert paths. The MoE methods and MTP proposer
use this checkout's current APIs. The parsers use the unified parser engine.

Build this checkout on a supported Linux CUDA host before serving: the new
PolyNorm and quantization operators require rebuilding the native extensions.
Use the repository's [incremental build instructions](../../../docs/contributing/incremental_build.md).
A BF16 serving example for a host with sufficient aggregate GPU memory is:

```bash
vllm serve Motif-Technologies/Motif-3 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --reasoning-parser motif \
  --tool-call-parser motif \
  --enable-auto-tool-choice
```

Adjust the parallelism and context length to the available memory. Optional
flags are `--quantization modelopt_mxfp8`, `--quantization modelopt_blockfp8`,
or `--quantization modelopt_nvfp4`. NVFP4 requires compatible Blackwell hardware.
The block-FP8 expert path uses Triton; MXFP8 experts use Marlin. To enable MTP, add
`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.

### Port validation

CPU unit tests cover reasoning (including delayed special-token text), tool-call
JSON repair, repetition guards/thinking budgets, and local checkpoint config/MTP
conversion. On the Windows development host these run with `--noconftest` and a
process-local `uvloop.run = asyncio.run` shim; no server event loop is exercised.
Python syntax, Ruff check/format and clang-format are also checked. The mHC
PyTorch reference was checked for clipping, weighted residuals, Sinkhorn sums,
and the Motif storage transpose.

CUDA compilation, the TileLang comparison tests, quantized kernel correctness,
and full-model generation/evaluation have **not** been run on this host. Before
production use, run the following on the Linux GPU build and evaluate the model:

```bash
.venv/bin/python -m pytest tests/reasoning/test_motif_reasoning_parser.py \
  tests/tool_parsers/test_motif_tool_parser.py \
  tests/v1/sample/test_repetition_guard.py \
  tests/transformers_utils/test_config.py::test_motif_config_loads_without_remote_code
.venv/bin/python -m pytest tests/kernels/moe/test_motif_nvfp4.py
.venv/bin/python -m pytest tests/model_executor/test_motif_weight_loading.py \
  tests/kernels/attention/test_motif_attention.py \
  tests/kernels/moe/test_motif_quantized_experts.py
.venv/bin/python -m pytest tests/kernels/mhc/test_mhc_kernels.py -k motif
```

### GB10 implementation and validation plan

The target is two 128 GB GB10 systems, TP=2, with a 262144-token context.
The baseline is [Motif-3-NVFP4](https://huggingface.co/Motif-Technologies/Motif-3-NVFP4).
This target has **not been validated on GB10**. The changes below implement the
revised plan; GPU compilation, correctness, memory measurements and model
evaluation remain acceptance requirements.

Local validation of this revision: 144 CPU tests passed and 17 CUDA tests were
skipped. Ruff, Python AST checks, clang-format, Markdown lint, SPDX/import/API
checks and the test-tethering check passed. The full pre-commit runner could not
provision an unrelated hook environment under the Windows sandbox, so these
checks were run individually. No CUDA build or GB10 model run is implied.

The previously reported failure used the NVFP4 checkpoint with kernel capability
guards removed. That is not the execution path here. Its exact failing kernel
cannot be identified without the original traceback. FlashInfer TRT-LLM attention
and TRT-LLM MoE are separate paths, and passing a capability guard does not make
their tile sizes or shared-memory requirements valid on GB10.

| Component | Implemented path | Verification required on GB10 |
| --- | --- | --- |
| NVFP4 expert storage | Allocate packed weights and scales directly before checkpoint loading | Peak CUDA allocation, RSS and system memory availability |
| NVFP4 GEMMs | Existing native CUTLASS SM120-family grouped GEMM; check compiled shared storage against the device opt-in limit before initialization | Build for the target device, then both expert GEMMs |
| NVFP4 activation preparation | Unfused SM12 path; retain the SM100 restriction on Motif's fused gather/PolyNorm quantization kernels | Compare against NVFP4 reference, including expert routing |
| GDLA sliding-window attention | Triton DiffKV, QK=192/V=128, 16x32 tiles, one stage | Window boundary, paged cache, prefix cache, graph replay |
| Full MLA prefill | Triton DiffKV with ragged sequences and context-chunk LSE | Chunk merge, empty context, long-context generation |
| Full MLA decode | Existing Triton MLA with 16-token tiles and one stage on SM12 | Decode and MTP multi-token queries |
| Block FP8 experts | E4M3 weights, 128x128 FP32 scales, Triton W8A8 with PolyNorm | Quantized outputs and model evaluation |
| MXFP8 experts | E4M3/E8M0 checkpoint storage, per-expert Marlin repack, W8A16 with PolyNorm | Quantized outputs and model evaluation |
| INT4 experts | AWQ or symmetric GPTQ, Marlin W4A16 with PolyNorm | Compatible serialized checkpoint, conversion, outputs and model evaluation |

All these expert paths select the Motif method before allocating weights. The
old NVFP4 direct-load path first allocated BF16 experts and then replaced them
with packed weights. It did not need to dequantize the entire NVFP4 checkpoint;
the temporary BF16 allocation and scale-processing copies caused avoidable peaks.
For 192 local experts, H=4096 and I=1280, the old BF16 allocation was 5.625 GiB
per MoE layer. Packed NVFP4 weights plus scales occupy about 1.582 GiB instead.
These are allocation calculations, not measured process peaks.

Scale validation/swizzling now operates one expert at a time. Dynamic NVFP4
loading stages only rank-local experts, with a 128 MiB per-expert staging limit;
dynamic block FP8 and MXFP8 quantize 128 rows at a time. The source checkpoint
tensor can still be memory-mapped in host memory. INT4 conversion uses the
upstream layer-level Marlin repacker and can still allocate packed temporary
copies. MLA KV projections are copied directly into the local query-head layout,
avoiding expansion to all global query heads first. MTP uses the same expert
loader as the main model.

Set `VLLM_MOTIF_LOAD_PROFILE=1` to log allocation/ready stages with process RSS,
system available memory, CUDA allocated/reserved memory and cumulative CUDA peak
allocation. On GB10, RSS and CUDA accounting may overlap in unified memory;
do not add those counters together. Sample system available memory throughout
loading as well: the per-layer log alone cannot capture every transient peak.

Supported alternative checkpoint contracts:

- Dynamic FP8/NVFP4 accepts the original fused BF16/FP16 Motif expert tensors.
  It never constructs a full BF16 expert parameter set before quantization.
- Serialized block FP8 uses `quant_method=fp8`, `weight_block_size=[128,128]`,
  E4M3 expert weights and FP32 block scales (`weight_scale` or
  `weight_scale_inv`). Per-tensor FP8 is not supported by this Motif path.
- Serialized MXFP8 uses the ModelOpt MXFP8 configuration and E4M3 weights with
  E8M0 scales stored as uint8. Dynamic MXFP8 remains available through
  `--quantization modelopt_mxfp8`.
- INT4 accepts AWQ with zero points or symmetric GPTQ with `desc_act=False`,
  4-bit weights and group sizes 32/64/128. Per-layer dynamic overrides are
  rejected. Expert keys may use `experts.<id>.{gate_proj,up_proj,down_proj}`
  or fused `experts.{gate_up_proj,down_proj}` followed by the quantized parameter
  suffix. The exporter must preserve Motif's unquantized PolyNorm parameters.
- Quantization of dense/shared/attention projections follows the respective
  existing linear method. These expert additions do not provide a checkpoint
  exporter, calibration workflow or CPU offloading. Full FP8-model testing
  requires hardware with sufficient memory beyond the two-GB10 target.

After building the native extensions on the Linux GPU hosts and configuring a
two-node Ray cluster, start the NVFP4 acceptance run with:

```bash
VLLM_MOTIF_LOAD_PROFILE=1 vllm serve Motif-Technologies/Motif-3-NVFP4 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 262144 --max-num-seqs 1 \
  --max-num-batched-tokens 512 --enable-chunked-prefill \
  --block-size 128 --kv-cache-dtype bfloat16 --enforce-eager
```

Pin and record the checkpoint revision, vLLM revision, CUDA/PyTorch/FlashInfer
versions, device compute capability and shared-memory limits. Run the kernel
tests above first. Then test 8K, 32K, 128K and 256K total context, ending with a
261120-token prompt plus 1024 generated tokens. Record peak memory, minimum
system memory available, time to first token, decode throughput and finite logits.
Target at least 8 GiB system memory headroom per node, but determine feasibility
from measurements rather than treating `max-model-len` as a memory guarantee.
Repeat with CUDA graphs, prefix caching and one-token MTP enabled individually,
then together. Check mHC and PolyNorm against references and evaluate output
quality against the baseline. FP8/INT4 kernel tests are only a prerequisite for
full-model accuracy evaluation, not a replacement for it.

## Overview

Motif is a decoder-only language model combining three key architectural innovations:

1. **GDLA** – Grouped Differential Latent Attention
2. **MHC** – Manifold-constrained Hyper-Connections (optional residual path)
3. **Hybrid MoE** – Dense-first layers followed by interleaved MoE layers

```text
input_ids [T]
    │
    ▼
VocabParallelEmbedding → hidden_states [T, H]
    │
    │  (if mhc_enabled) expand → [T, E, H]   (E = mhc_expansion_rate)
    ▼
MotifDecoderLayer × num_hidden_layers
    │
    │  (if mhc_enabled) reduce mean → [T, H]
    ▼
RMSNorm → ParallelLMHead → logits [T, vocab_size]
```

---

## Layer Assignment

Each decoder layer is assigned one of two feed-forward types based on its index:

```text
layer_idx < n_dense_first_layers          → always MotifMLP (dense)
layer_idx >= n_dense_first_layers
    AND (layer_idx + 1) % interleave_moe_layer_step == 0  → MotifMoE
    otherwise                                              → MotifMLP (dense)
```

---

## MotifDecoderLayer

### Standard Path (mhc_enabled=False)

```text
hidden_states [T, H]
    │
    ├─ input_layernorm (RMSNorm, fused residual)
    │
    ▼
MotifGDLAttention → attn_out [T, H]
    │
    ├─ post_attention_layernorm (RMSNorm, fused residual)
    │
    ▼
MotifMLP or MotifMoE → [T, H]
```

### MHC Path (mhc_enabled=True)

The tensor lives as `[T, E, H]` throughout all layers. Each sublayer (attn, ffn) follows the same pattern:

```text
x [1, T, E, H]
    │
    ▼  MotifMHCLayer
    ├── h_pre  [1, T, E]     (sigmoid gate, selects which expansion dim to read)
    ├── h_post [1, T, E]     (2*sigmoid gate, scales how to write back)
    └── h_res  [1, T, E, E]  (doubly-stochastic residual mixing matrix via Sinkhorn)
    │
    ▼
apply_h_pre(x, h_pre)  →  x_reduced [T, H]    (weighted sum over E dim)
    │
    ▼
LayerNorm → sublayer (attn or ffn) → out [T, H]
    │
    ▼
apply_h_post(out, h_post) →  out_expanded [1, T, E, H]
apply_h_res(x, h_res)     →  x_mixed      [1, T, E, H]
    │
    ▼
x_next = x_mixed + out_expanded  [1, T, E, H]
```

---

## MotifGDLAttention (GDLA)

Grouped Differential Latent Attention combines:

- **Low-rank latent compression** for Q and KV (like MLA)
- **Differential noise cancellation** (like DIFF Transformer)
- **RoPE** applied only over a subset of head dims (`qk_rope_head_dim`)

### Weight Naming: Checkpoint → vLLM

| Checkpoint key | vLLM name | Type |
| --- | --- | --- |
| `wq_a` | `q_a_proj` | ReplicatedLinear |
| `q_norm` | `q_a_layernorm` | RMSNorm |
| `wq_b` | `q_b_proj` | ColumnParallelLinear |
| `wq_b_gate` | `q_b_gate` | ColumnParallelLinear |
| `wkv_a` | `kv_a_proj_with_mqa` | ReplicatedLinear |
| `kv_norm` | `kv_a_layernorm` | RMSNorm |
| `wkv_b` | `kv_b_proj` | ColumnParallelLinear |
| `lambda_proj` | `lambda_proj` | ColumnParallelLinear |
| `wo` | `o_proj` | RowParallelLinear |

### Forward Flow

```text
hidden_states [T, H]
    │
    ├─── Q path ───────────────────────────────────────────────────────┐
    │   q_a_proj (Replicated) → [T, q_lora_rank]                       │
    │   q_a_layernorm                                                  │
    │   q_b_proj (ColumnParallel) → [T, num_local_heads * head_dim]    │
    │   reshape → [T, num_local_heads, head_dim]                       │
    │   split → q_nope [T, H_local, nope_dim]                          │
    │         + q_pe   [T, H_local, rope_dim]                          │
    │                                                                  │
    ├─── KV path ──────────────────────────────────────────────────────┤
    │   kv_a_proj (Replicated) → [T, kv_lora_rank + rope_dim]          │
    │   split → kv_latent [T, kv_lora_rank]                            │
    │         + k_pe      [T, rope_dim]                                │
    │   kv_a_layernorm                                                 │
    │   kv_b_proj (ColumnParallel) → [T, kv_heads_local*(nope+v_dim)]  │
    │   split → k_nope [T, kv_H_local, nope_dim]                       │
    │         + v      [T, kv_H_local, v_head_dim]                     │
    │                                                                  │
    ├─── RoPE ─────────────────────────────────────────────────────────┤
    │   rotary_emb(positions, q_pe, k_pe) → rotated q_pe, k_pe         │
    │   k_pe expanded to all kv heads                                  │
    │   q_full = cat(q_nope, q_pe) [T, H_local, head_dim]              │
    │   k_full = cat(k_nope, k_pe) [T, kv_H_local, head_dim]           │
    │                                                                  │
    ├─── Attention ────────────────────────────────────────────────────┤
    │   v padded to head_dim (flash-attn constraint)                   │
    │   attn_out [T, num_local_heads, head_dim] → slice to v_head_dim  │
    │                                                                  │
    └─── Differential combination ─────────────────────────────────────┤
        reshape → [T, noise_heads_local, grouped_ratio+1, v_head_dim]  │
        attn1 = first grouped_ratio heads  (signal)                    │
        attn2 = last 1 head, repeat_interleave  (noise)                │
        lambda_vals = lambda_proj(hidden) → [T, signal_heads_local]    │
        result = attn1 - sigmoid(lambda) * attn2                       │
        (optional) * sigmoid(q_b_gate)    (elementwise gate)           │
        o_proj (RowParallel) → [T, H]                                  │
```

**Key shapes:**

- `num_heads = n_signal_heads + num_noise_heads`
- `n_signal_heads = grouped_ratio * num_noise_heads`
- `grouped_ratio = (num_heads - num_noise_heads) // num_noise_heads`

---

## MotifMLP (Dense Feed-Forward)

Used for the first `n_dense_first_layers` layers and as `shared_experts` inside MoE layers.

```text
hidden_states [T, H]
    │
    ├─ gate_proj (ColumnParallel) → [T, I // tp_size]
    ├─ up_proj   (ColumnParallel) → [T, I // tp_size]
    │
    ▼
act_fn(gate) * up   [T, I // tp_size]
    │
    ▼
down_proj (RowParallel, includes all-reduce) → [T, H]
```

### PolyNorm in MotifMLP — TP Correction

When `hidden_act = "poly_norm"`, each TP rank only sees `I // tp_size` of the intermediate dimension. The RMS normalization inside PolyNorm:

```text
_norm(x) = x / sqrt(mean(x², dim=-1) + eps)
```

computes the mean over a **shard**, giving wrong statistics. Fix: `PolyNorm(tp_size=tp_size)` performs an all-reduce over the partial sum-of-squares before computing the norm.

---

## MotifMoE (Mixture of Experts)

### Routing

```text
hidden_states [T, H]
    │
    ▼
gate (Replicated Linear) → scores [T, num_experts]
    │
    ▼ sigmoid or softmax
    │
    │  (+ e_score_correction_bias for load balancing, routing only)
    ▼
topk(scores, k=experts_top_k) → top_idx [T, K], top_scores [T, K]
    │
    ▼ (optional) route_norm: normalize top_scores to sum=1
    │
    ▼ * route_scale
    │
    ├─ MotifMoEExperts(hidden_states, top_idx, top_scores) → expert_out [T, H]
    └─ shared_experts(hidden_states)  [T, H]  (MotifMLP, if num_shared_experts > 0)
    │
    ▼ sum
out [T, H]
```

### MotifMoEExperts — Expert Parallelism via TP

**Sharding strategy: expert parallelism, not hidden-dim parallelism.**

Each TP rank holds `num_experts // tp_size` complete experts (full `H` and `I` dimensions):

```text
gate_up_proj: [local_E, 2*I, H]   ← full H, full I
down_proj:    [local_E, H,   I]   ← full H, full I
```

Hidden states are **replicated** — every rank receives the full `[T, H]` tensor.

```text
hidden_states [T, H]  ← replicated on ALL ranks
    │
    ▼  gate+up GEMM (one big batched matmul)
    [T, H] × [H, local_E * 2I]  →  [T, local_E, 2I]
    split → gate [T, local_E, I] + up [T, local_E, I]
    │
    ▼  activation (GroupedPolyNorm, per-expert coefficients)
    gate_act [T, local_E, I]
    │
    ▼  down GEMM (bmm over expert dim)
    [local_E, T, I] × [local_E, I, H]  →  [local_E, T, H]
    permute → [T, local_E, H]
    │
    ▼  apply routing weights (effective_w = 0 for non-local experts)
    (out.float() * effective_w.unsqueeze(-1)).sum(dim=1)  →  [T, H]
    │
    ▼  tensor_model_parallel_all_reduce
out [T, H]
```

The routing mask `effective_w` handles the gating: if a token was not routed to any expert on this rank, all entries are zero, so that rank contributes nothing before the all-reduce.

### GroupedPolyNorm in MotifMoEExperts — No TP Fix Needed

Each expert operates on the **full** intermediate dimension `I` (not split by TP). Therefore, the mean in `_norm` is computed over the full `I` locally — no all-reduce needed.

### Comparison: PolyNorm TP Treatment

| Layer | Intermediate sharding | PolyNorm needs all-reduce? |
| --- | --- | --- |
| `MotifMLP` (dense, first N layers) | `I // tp_size` (ColumnParallel) | **Yes** → `PolyNorm(tp_size)` |
| `MotifMoEExperts` | full `I` per expert | **No** → `GroupedPolyNorm` unchanged |
| `MotifMoE.shared_experts` (MotifMLP) | `I // tp_size` (ColumnParallel) | **Yes** → via `MotifMLP` ✓ |

---

## Tensor Parallelism Summary

| Component | What is sharded | All-reduce location |
| --- | --- | --- |
| Attention Q/KV/O | heads (`num_heads // tp`) | inside `RowParallelLinear` (o_proj) |
| MotifMLP gate/up | intermediate dim (`I // tp`) | inside `RowParallelLinear` (down_proj) |
| MotifMoEExperts | experts (`E // tp`) | explicit `tensor_model_parallel_all_reduce` at end of forward |
| Router gate | replicated | none |
| Embeddings | vocab dim | `VocabParallelEmbedding` |

---

## Sliding Window Attention

Configured per layer based on `sliding_window_pattern`:

- `"all"`: every layer uses `sliding_window` context
- `"interleave"`: layers where `(layer_idx + 1) % sliding_window_period != 0` use sliding window; the rest use full context

### Window size: the `+1` convention

Training passes flash-attn `window_size=(sliding_window, 0)` directly, so a query
at position *i* attends to itself **plus** `sliding_window` past keys —
`sliding_window + 1` tokens in total. vLLM's FlashAttention backend instead
converts a per-layer window `W` to `(W - 1, 0)`. To reproduce the trained window
exactly, `MotifGDLAttention` sets `per_layer_sliding_window = config.sliding_window + 1`
(matching the HF reference's `config.sliding_window + 1`). Without the `+1`, every
SWA layer would attend to one fewer key than training — a small but systematic
shift that compounds across the (majority) SWA layers.

---

## Weight Loading

The checkpoint uses different key names than vLLM's internal naming. `load_weights` applies a rename map at load time:

```text
moe.router.gate   →  moe.gate
moe.expert_bias   →  moe.e_score_correction_bias
self_attn.wq_a    →  self_attn.q_a_proj
self_attn.wq_b    →  self_attn.q_b_proj
... (see attn_rename dict in MotifForCausalLM.load_weights)
```

Expert weights (`gate_up_proj`, `down_proj`, `act_fn.weight/bias`) are stored as the full expert stack `[num_experts, ...]` in the checkpoint. During loading, each TP rank slices `[start_expert : start_expert + local_num_experts]` from dim 0.

---

## Training-Parity Notes (numerical consistency audit)

These notes capture an audit of the vLLM port against the `llm-training` source of
truth (`llm_training/motif/`). The training code is authoritative; the HF reference
(`huggingface/motif/modeling_motif.py`) is a simplified/occasionally-stale fallback
and must **not** be used to validate numerics where it disagrees with training.

### Fixed discrepancies

1. **Routed-expert PolyNorm bias clamp.** Training clamps the routed-expert
   PolyNorm bias to `[-polynorm_bias_clamp, +polynorm_bias_clamp]` inside
   `GroupedExpertsPolyNorm.forward` (every step). The bias is frozen at inference,
   so vLLM clamps the loaded `moe.experts.act_fn_bias` parameter once at load time
   (`MotifForCausalLM._load_fused_moe_expert_weight`), which is numerically
   identical and covers all expert backends (Triton / MXFP8 / DeepGEMM — they read
   the same parameter). The value comes from `config.polynorm_bias_clamp`
   (`MotifConfig`; `None` = no clamp). **Scope:** routed experts only. The dense and
   shared-expert MLPs (`FeedForward`) leave the bias unclamped in training, so
   `MotifMLP` correctly does not clamp it.

2. **Sliding-window `+1`.** See [Window size: the `+1` convention](#window-size-the-1-convention).
   `per_layer_sliding_window = config.sliding_window + 1`.

3. **tilelang MHC pre-sigmoid clamp.** The eager / Triton MHC path and training
   clamp the pre-sigmoid argument of `h_pre` / `h_post` to `[-10, 10]`. The tilelang
   fused kernel (`layers/mhc.py`) now applies the same `T.max(T.min(x, 10), -10)`
   clamp for bit-closer parity (impact is limited to the deep-saturation tail since
   sigmoid is already saturated past `±10`).

4. **PolyNorm `* up` multiply in fp32 (MotifMLP).** Training's `FusedMulPolyNorm`
   computes `poly * up` in fp32 and downcasts only the final result
   (`grouped_polynorm.py`: `m = mul.float()`, `result = poly * m`, `.to(orig_dtype)`
   last). vLLM downcast `poly` to bf16 *before* the multiply, losing ~1e-3 per
   element. All three MLP paths now multiply in fp32: the fused TP=1 path
   (`_poly_norm_mlp_compute`), the eager non-TP path, and the TP&gt;1 path
   (`_forward_tp`) — the last is the common case in production (TP&gt;1 routes
   dense and shared-expert MLPs through it). `polynorm_output_scale` is applied
   after the downcast, matching the training `FeedForward`. Scope: dense +
   shared-expert MLPs; the routed experts already multiply in fp32 inside the
   CUDA `grouped_poly_norm_forward` kernel.

### Verified matching (do not "fix")

- **MHC `h_post` coefficient is `1.0`.** Training uses `(1 + mhc_h_post_alpha_end) * sigmoid(...)`
  with `mhc_h_post_alpha_end = 0.0` for motif3, i.e. coefficient `1.0`. vLLM matches.
  The HF reference's hardcoded `2 * sigmoid(...)` is **stale** — do not copy it.
- **RoPE / YaRN.** vLLM follows training (YaRN inv-freq interpolation applied;
  `apply_yarn_scaling=false` only suppresses the cos/sin mscale, not the
  frequencies). The HF reference resets `inv_freq` to plain RoPE in `_init_weights`
  and disagrees with training; vLLM is correct.
- **RoPE weight layout.** The training→HF conversion permutes RoPE rows from
  consecutive-pair to half-split, so vLLM's `is_neox_style=True` is correct.
- **`polynorm_output_scale` / `hidden_clamp`.** Read from `config.json`
  (`0.5` / `1e6` for motif3). These are **not** declared as defaults that match
  training, so the served `config.json` must carry them — confirm they are present
  (the deployed motif3 config sets both).

### Config fields the deployed `config.json` must set

`polynorm_output_scale`, `hidden_clamp`, `polynorm_bias_clamp`, `polynorm_sigmoid_weight`,
`swa_rope_theta`. Several of these are read via `getattr(config, ...)` with defaults
that differ from the trained values, so a missing field silently degrades accuracy.
