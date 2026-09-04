# szl-engine-bench

An honest, standard-library-only benchmark client for OpenAI-compatible
streaming endpoints exposed by vLLM, SGLang, llama.cpp, MLX, TGI, and
Transformers Serve.

Version 0.2.0 reports separate distributions for time to first token/chunk
(TTFT), inter-chunk latency (ITL), end-to-end latency, and the fraction of
runs meeting a declared TTFT service-level objective. It does not collapse an
engine comparison into one winner.

The distinction is deliberate. SGLang's serving guide reports TTFT, TPOT,
ITL, and throughput as separate online-serving metrics, while vLLM Bench
separately exposes TTFT/ITL percentiles and SLO goodput. Cache shape, request
rate, concurrency, model, and hardware can all change the result; a benchmark
from a different workload is not evidence for this one.

- [SGLang benchmark and profiling guide (source snapshot)](https://github.com/sgl-project/sglang/blob/50ed4c011ffb37e1c16248b99deab7d4dbdde79b/docs/developer_guide/benchmark_and_profiling.md)
- [vLLM Bench metrics (source snapshot)](https://github.com/vllm-project/vllm-bench/blob/f3131823f167aa1e6f31a48be1ae15a18667a534/README.md#metrics)

## States and measurement semantics

- `MEASURED`: every requested run produced at least two non-empty streaming
  response chunks and valid monotonic arrival times.
- `BLOCKED`: the engine endpoint was not configured, or fewer than two engines
  produced measurements for comparison.
- `INVALID`: configuration, stream shape, or fairness inputs were unsuitable
  for the requested metric.
- `FAILED`: the configured endpoint or streaming protocol actually failed.

ITL is measured between non-empty SSE response chunks. An SSE chunk is not
assumed to equal one model token. `chunks_per_s` is always labeled as such;
`tok_per_s` is `UNAVAILABLE` unless the endpoint itself supplies an integer
`usage.completion_tokens` value. This avoids turning packet counts into fake
token throughput.

The comparison keeps the v0.1.0 fairness gates: measured engines must use the
same model string and run count. `goodput_at_slo` is the auditable fraction of
runs whose TTFT is less than or equal to `--slo-ttft-ms`; the boundary is
inclusive. It is not claimed to be a maximum sustainable request rate.

## Install and verify

```bash
python -m pip install -e . pytest
python -m pytest tests -q
python -m compileall -q engine_bench.py tests
python engine_bench.py --engines vllm --runs 1
```

The last command is a fail-closed smoke test: without `VLLM_ENDPOINT`, it emits
a `BLOCKED` result and a valid receipt rather than latency values.

The deterministic burst fixture from issue #2 can be reproduced without an
endpoint:

```bash
python -c "import engine_bench as e; print(e.run_stats([.05,.06,.07,.4,.45,.5,.55,.6], 8))"
```

Its TTFT is 50 ms. The inter-chunk gaps are 10, 10, 330, 50, 50, 50, and
50 ms, so linear interpolation yields ITL p50/p95/p99 of 50, 246, and
313.2 ms. This is a unit fixture, not a hardware benchmark.

## Measure configured engines

PowerShell:

```powershell
$env:VLLM_ENDPOINT = "http://127.0.0.1:8000"
$env:SGLANG_ENDPOINT = "http://127.0.0.1:30000"
python engine_bench.py --engines vllm sglang --model YOUR_MODEL --runs 10 --max-tokens 64 --slo-ttft-ms 200
```

POSIX shells:

```bash
export VLLM_ENDPOINT=http://127.0.0.1:8000
export SGLANG_ENDPOINT=http://127.0.0.1:30000
python engine_bench.py --engines vllm sglang --model YOUR_MODEL --runs 10 --max-tokens 64 --slo-ttft-ms 200
```

Supported variables are `VLLM_ENDPOINT`, `SGLANG_ENDPOINT`,
`LLAMACPP_ENDPOINT`, `MLX_ENDPOINT`, `TGI_ENDPOINT`, and
`TRANSFORMERS_ENDPOINT`. Each must be an absolute HTTP(S) base URL without
embedded credentials. The client posts to `/v1/completions`.

## Receipts

Every CLI invocation emits one canonical-JSON SHA-256 chain receipt using the
repository's `prev_hash` / `self_hash` schema and the all-zero genesis hash.
The receipt anchors:

- a canonical SHA-256 of every complete engine result;
- a canonical SHA-256 of each measured engine's nested ITL gap arrays;
- the benchmark configuration and a prompt hash; and
- the comparison verdict and every engine state.

Each run sample also includes its raw `itl_gaps_ms` array and matching
`itl_gaps_sha256`, so the hash can be independently recomputed.
`UNSIGNED_HONEST` proves integrity and order only; it does not claim signer
identity or external attestation.

Apache-2.0 · Doctrine v11 · SZL Holdings
