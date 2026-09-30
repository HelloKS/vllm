# Solar Open 2

`upstage/Solar-Open2-250B` uses the `SolarOpen2ForCausalLM` architecture.
It combines gated full attention, Kimi Delta Attention, and shared and routed
experts. The configuration and model implementation are built into vLLM;
`--trust-remote-code` is not required.

For chat serving, select the Solar reasoning and tool parsers. The optional
template logits processor enforces the model's control-token envelope:

```bash
vllm serve upstage/Solar-Open2-250B \
    --tensor-parallel-size 8 \
    --reasoning-parser solar_open2 \
    --enable-auto-tool-choice \
    --tool-call-parser solar_open2 \
    --logits-processors vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor
```

Choose the parallelism and maximum context length for the available GPU memory.
The model's configured maximum context length is 1,048,576 tokens.

The template logits processor defaults to a reasoning budget of 131,072 tokens.
The same processor path supports both the V1 and V2 model runners. On V2 it
reads committed token history back to the CPU and evaluates draft prefixes
separately; this adds device synchronization overhead.
Set `SOLAR_REASONING_BUDGET` to change that default; zero disables the budget.
Requests can override it with `solar_open2_reasoning_budget` in `vllm_xargs`,
or disable the processor with `disable_solar_open2_logits_processor: 1`.

## DSpark speculative decoding

Solar Open 2 implements DSpark target-state verification and auxiliary hidden
states for
[`sionic-ai/Solar-Open2-250B-ultrafast-Draft-LM-head`](https://huggingface.co/sionic-ai/Solar-Open2-250B-ultrafast-Draft-LM-head).
Use this standalone draft repository; the similarly named repository without
`-LM-head` contains the 250B target at its root and the draft in a subdirectory.

The draft uses the existing `Qwen3DSparkModel` implementation. Its config selects
target layers `[3, 15, 27, 39, 47]` (zero-based, after each decoder layer and before
the final RMSNorm), mask token 26, and a seven-token block. The five residual
streams are concatenated into 20,480 features for the draft's projection.
The checkpoint includes its embedding, LM head, Markov head, and confidence head.

Example for the Nota NVFP4 target, with the draft kept in BF16:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nota-ai/Solar-Open2-250B-Nota-NVFP4 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --moe-backend cutlass \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --reasoning-parser solar_open2 \
    --enable-auto-tool-choice \
    --tool-call-parser solar_open2 \
    --speculative-config '{"method":"dspark","model":"sionic-ai/Solar-Open2-250B-ultrafast-Draft-LM-head","revision":"b22337d89e7f09aa69d9703a871b12327a478777","num_speculative_tokens":7}'
```

Adjust tensor parallelism and memory limits for your hardware. DSpark requires
the V2 GPU model runner. Pipeline parallelism is not supported for Solar DSpark.
The example omits the template logits processor. Add
`--logits-processors vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor`
to enforce its control-token constraints and separate reasoning budget with
V2 speculative decoding. Without it, the reasoning and tool parsers interpret
generated output but do not enforce those constraints. The processor's CPU
synchronization overhead and end-to-end DSpark performance remain unverified.

The published config omits `sample_from_anchor`; the existing DSpark default
(`true`) is used, giving seven draft query positions and eight target verification
positions. Attention is causal with a 128-token sliding window on every draft
layer, as specified by its `layer_types`. No draft quantization override is needed.

This integration has been checked statically against the pinned checkpoint's
config and all 64 tensor headers. CUDA execution, rejection recovery, acceptance
rate, and speed on SM121 remain unverified. The publisher's B300/FP8 serving
results do not establish performance for an NVFP4 target on SM121.
