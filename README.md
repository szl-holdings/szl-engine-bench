# szl-engine-bench

An honest, standard-library-only benchmark client for OpenAI-compatible
streaming endpoints exposed by vLLM, SGLang, llama.cpp, MLX, TGI, and
Transformers Serve.

Version 0.3.0 reports separate distributions for time to first token/chunk
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
the request asks the server for streamed usage with
`stream_options.include_usage=true`, and `tok_per_s` is `UNAVAILABLE` unless
the endpoint actually supplies an integer `usage.completion_tokens` value.
This avoids turning packet counts into fake token throughput.

The comparison keeps the v0.1.0 fairness gates: measured engines must use the
same model string and run count for low-level numerical diagnostics.
Version 0.3.0 additionally requires the declared identity contract below for
CLI comparisons; the old model-name check alone is explicitly unqualified. `goodput_at_slo` is the auditable fraction of
runs whose TTFT is less than or equal to `--slo-ttft-ms`; the boundary is
inclusive. It is not claimed to be a maximum sustainable request rate.

## Install and verify

```bash
python -m pip install -e . pytest
python -m pytest tests -q
python -m compileall -q engine_bench.py benchmark_manifest.py tests
python engine_bench.py --engines vllm --runs 1
```

The last command is a fail-closed smoke test: without a declaration it emits
a `BLOCKED` result and a valid receipt, even when an endpoint is configured.
It issues no network request. No model or GPU is needed for the test suite.

The deterministic burst fixture from issue #2 can be reproduced without an
endpoint:

```bash
python -c "import engine_bench as e; print(e.run_stats([.05,.06,.07,.4,.45,.5,.55,.6], 8))"
```

Its TTFT is 50 ms. The inter-chunk gaps are 10, 10, 330, 50, 50, 50, and
50 ms, so linear interpolation yields ITL p50/p95/p99 of 50, 246, and
313.2 ms. This is a unit fixture, not a hardware benchmark.

## Exact experiment declarations (v0.3.0)

The default CLI requires `--manifest comparison.json` before contacting any
configured engine. The complete selected cohort is checked first: a missing or
changed endpoint, extra engine, mismatched request configuration, invalid digest,
or different hardware declaration cannot become a successful partial comparison.

The manifest uses the closed `szl.engine-comparison/v1` contract. Its template is
[`examples/comparison.synthetic.json`](examples/comparison.synthetic.json).
**Every identity in that file is a synthetic test fixture, not a qualified model,
hardware profile, or live endpoint.** Populate real operator-reviewed declarations
before measuring a deployment. The loader never downloads or executes a manifest
reference and rejects duplicate JSON keys, non-finite values, oversized inputs,
unknown fields, control characters, mutable revision names, and newline-suffixed
hashes. SHA checks use full-string matching, not permissive end anchors.

| Declaration | Meaning |
|---|---|
| Model/tokenizer revisions | Exact 40-character lowercase Git subjects; not branch names |
| Weights/tokenizer SHA-256 | Exact single artifact bytes, or SHA-256 of a separately retained canonical multi-file digest manifest |
| Template/adapter SHA-256 | Deployed template/adapter artifact identity; null adapter means explicitly none |
| Quantization/precision | Operator-declared configuration; no implicit conversion between engines |
| Suite/prompt SHA-256 | Retained suite artifact plus SHA-256 of the exact UTF-8 prompt sent by the CLI |
| Runs/token limit/timeout | Must match the requested invocation; timeout is the client's socket timeout, not a promised total wall-clock deadline |
| Sampling | Explicit temperature, top-p, and integer seed, included in every actual HTTP request |
| Engine build/configuration | Separate SHA-256 identities for each engine; these may differ in an engine comparison |
| Hardware SHA-256 | Same retained hardware-profile digest across the declared cohort |
| Endpoint SHA-256 | SHA-256 of the exact configured base URL after removing trailing slashes; no credentials allowed |

`benchmark_manifest.endpoint_digest()` computes the endpoint digest. File digests
refer to exact bytes; they are not a cryptographic demonstration that a remote
server loaded those bytes. Engine names must match the supported registry. The
client runs serially, performs no warmup/reset, and requires `concurrency: 1` and
`cache_state: UNCONTROLLED`; comparing cold and warm caches is not silently claimed.
It freezes the preflight endpoints and refuses redirects for bound requests.

Every sample commits the exact serialized request bytes. A run binding includes
the experiment, model/artifact subject, workload, per-engine build/configuration,
hardware, endpoint, and request hashes. `compare_bound()` rejects altered bindings,
missing samples, different wire requests, and omitted/failed cohort members.
No failed engine is discarded to obtain a better-looking comparison.

**Identity evidence remains `OPERATOR_DECLARED_NOT_RUNTIME_ATTESTED`.** A matching
manifest is not server identity attestation. Comparison reports keep
`runtime_identity: NOT_ATTESTED`, `quality_evaluation: NOT_PERFORMED`,
`production_admission: NOT_EVALUATED`, and uncontrolled cache state. They measure
client-observed latency only and never decide model promotion or an engine winner.
Different hardware should be evaluated as a separately designed system experiment,
not misrepresented as an engine-only improvement.

An incomplete or invalid explicit `--manifest` invocation exits 2 after emitting
its available receipt. The default offline smoke command retains its historical
zero exit. Library `run_engine()`, `compare()`, and `compare_engines()` remain
available for numerical diagnostics, labeled `UNBOUND_METRICS_ONLY` in comparisons.
The CLI requires `--unbound-diagnostics` to explicitly opt into that legacy path;
it cannot be combined with `--manifest` and is not release admission evidence.

## Measure configured engines

PowerShell:

```powershell
$env:VLLM_ENDPOINT = "http://127.0.0.1:8000"
$env:SGLANG_ENDPOINT = "http://127.0.0.1:30000"
python engine_bench.py --engines vllm sglang --model YOUR_MODEL --runs 10 --max-tokens 64 --slo-ttft-ms 200 --manifest comparison.json
```

POSIX shells:

```bash
export VLLM_ENDPOINT=http://127.0.0.1:8000
export SGLANG_ENDPOINT=http://127.0.0.1:30000
python engine_bench.py --engines vllm sglang --model YOUR_MODEL --runs 10 --max-tokens 64 --slo-ttft-ms 200 --manifest comparison.json
```

Supported variables are `VLLM_ENDPOINT`, `SGLANG_ENDPOINT`,
`LLAMACPP_ENDPOINT`, `MLX_ENDPOINT`, `TGI_ENDPOINT`, and
`TRANSFORMERS_ENDPOINT`. Each must be an absolute HTTP(S) base URL without
embedded credentials, query parameters, fragments, or an invalid TCP port.
Duplicate names in `--engines` are rejected as `INVALID` before any request,
so receipt maps cannot silently overwrite a result. The client posts to
`/v1/completions`.

## Receipts

Every CLI invocation emits one canonical-JSON SHA-256 chain receipt using the
repository's `prev_hash` / `self_hash` schema and the all-zero genesis hash.
The receipt anchors:

- a canonical SHA-256 of every complete engine result;
- a canonical SHA-256 of each measured engine's nested ITL gap arrays;
- the ordered engine selection, benchmark configuration, and a prompt hash; and
- the comparison verdict and every engine state; and
- the canonical manifest digest and each exact wire-request/run binding (schema version 3).

Each run sample also includes its raw `itl_gaps_ms` array and matching
`itl_gaps_sha256`, so the hash can be independently recomputed.
`UNSIGNED_HONEST` proves integrity and order only; it does not claim signer
identity or external attestation.

## Verification scope

The test suite covers strict declarations, complete-cohort comparisons, actual
loopback HTTP request bytes/sampling fields, redirect refusal, preflight zero-
request behavior, endpoint freezing, bad JSON/digests, failure retention, and
receipt commitments, in addition to the original streaming and numerical tests.
Loopback test responses are fixtures: no trained model, remote production system,
GPU performance, energy measurement, or quality result is represented by CI.

Apache-2.0 · Doctrine v11 · SZL Holdings
