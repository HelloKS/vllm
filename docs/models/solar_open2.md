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
Set `SOLAR_REASONING_BUDGET` to change that default; zero disables the budget.
Requests can override it with `solar_open2_reasoning_budget` in `vllm_xargs`,
or disable the processor with `disable_solar_open2_logits_processor: 1`.

Speculative decoding is not supported by this implementation.
