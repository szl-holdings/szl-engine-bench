# szl-engine-bench domain leader study r1 (propose-only)

Date: 2026-11-25. Method: web-verified leader landscape + owner-direct execution under standing authority GREENLIGHT-RULING-2026-11-24-K1 rank5 OUROBOROS_NEXT_TRANCHE. The Kimi governing seat remains stood down per explicit owner directive (disclosed); owner-direct nonce T19-OWNER-DIRECT-20261125-K1. Study pinned to HEAD e09ad25a9b32dc8404bd4f0c72f45a6e55065f3a, fetched 2026-11-25. This card is documentation only; it changes no code, CI, config, or existing PR status. All absorb actions are FUTURE PROPOSALS (non-authorizing); implementation and merges remain separately reviewed and gated. L Conjecture 1: OPEN (advisory, never a closed theorem). No speedup, energy, or proven-trust claims; upstream performance claims are labeled as upstream claims only. No unverified signing, attestation, or provenance claims; this study verifies nothing about this repo's artifacts.

## Domain: honest stdlib-only benchmark client for OpenAI-compatible streaming endpoints (separate TTFT/ITL/E2E distributions, SLO goodput, hash-chained receipts, fail-closed manifest declarations).

## Leaders studied (web-verified 2026-11-25)
1. vLLM — https://github.com/vllm-project/vllm — the serving engine whose vllm-bench metric conventions (separate TTFT/ITL percentiles, SLO goodput) this repo already cites as its honest-metrics reference; moat is the discipline of never collapsing a comparison into one number.
2. SGLang — https://github.com/sgl-project/sglang — the serving stack whose benchmark guide separates TTFT, TPOT, ITL and throughput as distinct online-serving metrics; moat is the metric-semantics hygiene this repo mirrors.
3. llama.cpp server — https://github.com/ggerganov/llama.cpp — the widely deployed minimal endpoint whose OpenAI-compatible streaming surface is one of the supported engines; moat is the minimal-compat surface that keeps a stdlib client testable.

## Absorb actions (FUTURE PROPOSALS, non-authorizing)
1. (vLLM) Propose a documentation-only review of vllm-bench's goodput percentile conventions against this repo's inclusive-boundary goodput_at_slo; no change to the auditable fraction semantics is proposed.
2. (SGLang) Propose a doc note mapping each declared state (MEASURED/BLOCKED/INVALID/FAILED/DECLARED) to SGLang's metric taxonomy so no SSE chunk is ever conflated with a model token.
3. (llama.cpp) Propose a documentation-only checklist for adding or re-verifying an engine registration, preserving the exact SHA-256 identity declarations and the OPERATOR_DECLARED_NOT_RUNTIME_ATTESTED boundary.
4. (baseline) Preserve the fail-closed baseline: no warmup/reset, concurrency 1, cache_state UNCONTROLLED, no failed engine discarded, UNSIGNED_HONEST receipts only; identity evidence stays operator-declared, never attested.


## Do not do (per governing seat)

Do not merge, bypass the signature wall, or claim implementation beyond this card.
