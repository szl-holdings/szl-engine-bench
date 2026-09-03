# szl-engine-bench

Honest inference-engine benchmark harness for the SZL estate. Measures any
OpenAI-compatible streaming endpoint — vLLM, SGLang, llama.cpp, MLX, TGI,
Transformers serve — and reports TTFT p50/p95/p99, total-latency percentiles,
and tokens/second. Stdlib only; no downloads, no API keys for local servers.

Doctrine: an engine without a configured endpoint returns `BLOCKED` with a
reason. A crashed engine returns `FAILED` with the real exception. It never
fabricates a latency number. Run states: `MEASURED | BLOCKED | INVALID | FAILED`.

## Install and verify

```
pip install -e . pytest
python -m pytest tests/ -q      # 8 tests, including a real local streaming server
python engine_bench.py          # demo: every unconfigured engine honestly BLOCKED
```

## Measure real engines

```
export VLLM_ENDPOINT=http://127.0.0.1:8000        # vllm serve
export SGLANG_ENDPOINT=http://127.0.0.1:30000     # python -m sglang.launch_server
export LLAMACPP_ENDPOINT=http://127.0.0.1:8080    # llama-server
export MLX_ENDPOINT=http://127.0.0.1:8081         # mlx_lm.server
python engine_bench.py --model YOUR_MODEL --runs 5 --max-tokens 64
```

## Fairness gates

- Engines are compared only when the model string and run count are identical;
  otherwise the comparison is `INVALID`, not a leaderboard.
- Fewer than two `MEASURED` engines -> comparison `BLOCKED`, never an invented winner.
- Every run emits a SHA-256 hash-chained `UNSIGNED_HONEST` receipt
  (integrity + order, not identity); `chain_valid` is printed with every report.

Pre-push verification for v0.1.0: a real streaming request over a real socket
returned MEASURED (TTFT p50 0.69 ms, 97.32 tok/s against a local mock server);
five unconfigured engines returned BLOCKED; a model-mismatched comparison
returned INVALID; the receipt chain verified.

Apache-2.0 · Doctrine v11 · SZL Holdings
