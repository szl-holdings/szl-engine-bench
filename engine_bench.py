#!/usr/bin/env python3
"""szl-engine-bench — honest inference-engine benchmark harness.

Measures any OpenAI-compatible streaming endpoint (vLLM, SGLang, llama.cpp,
MLX, TGI, Transformers serve). Stdlib only. No downloads, no API keys needed
for local servers.

Doctrine: an engine without a configured endpoint returns BLOCKED with a
reason. It never fabricates a latency number. Run states:
MEASURED | BLOCKED | INVALID | FAILED.

Usage:
    export VLLM_ENDPOINT=http://127.0.0.1:8000      # any subset of engines
    export SGLANG_ENDPOINT=http://127.0.0.1:30000
    python engine_bench.py --model my-model --prompt "Hello" --runs 3
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request

ENGINES = {
    "vllm": "VLLM_ENDPOINT",
    "sglang": "SGLANG_ENDPOINT",
    "llamacpp": "LLAMACPP_ENDPOINT",
    "mlx": "MLX_ENDPOINT",
    "tgi": "TGI_ENDPOINT",
    "transformers": "TRANSFORMERS_ENDPOINT",
}
STATES = ("MEASURED", "BLOCKED", "INVALID", "FAILED")
GENESIS = "0" * 64


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ReceiptChain:
    """Hash-chained run receipts. UNSIGNED_HONEST: integrity + order, not identity."""

    def __init__(self):
        self.chain = []

    def emit(self, record):
        prev = self.chain[-1]["self_hash"] if self.chain else GENESIS
        body = {"prev_hash": prev, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "signature": "UNSIGNED_HONEST", "run": record}
        body["self_hash"] = hashlib.sha256(_canonical(body).encode()).hexdigest()
        self.chain.append(body)
        return body

    def verify(self):
        prev = GENESIS
        for r in self.chain:
            if r["prev_hash"] != prev:
                return False
            check = dict(r)
            h = check.pop("self_hash")
            if hashlib.sha256(_canonical(check).encode()).hexdigest() != h:
                return False
            prev = h
        return True


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return round(sorted_vals[i], 2)


def measure_once(endpoint, model, prompt, max_tokens, timeout=120):
    """One real streaming request. Returns TTFT, per-run latency, token count."""
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "stream": True,
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft_ms = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            text = (chunk.get("choices") or [{}])[0].get("text", "")
            if text:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                tokens += 1
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {"ttft_ms": round(ttft_ms or total_ms, 2), "total_ms": round(total_ms, 2),
            "tokens": tokens,
            "tok_per_s": round(tokens / (total_ms / 1000.0), 2) if total_ms > 0 else 0.0}


def run_engine(name, model, prompt, runs, max_tokens):
    var = ENGINES[name]
    endpoint = os.environ.get(var)
    if not endpoint:
        return {"state": "BLOCKED", "engine": name,
                "reason": f"{var} not set; refusing to fabricate results"}
    samples = []
    try:
        for _ in range(runs):
            samples.append(measure_once(endpoint, model, prompt, max_tokens))
    except Exception as exc:  # real failure, reported honestly
        return {"state": "FAILED", "engine": name, "reason": f"{type(exc).__name__}: {exc}"}
    ttfts = sorted(s["ttft_ms"] for s in samples)
    totals = sorted(s["total_ms"] for s in samples)
    tps = sorted(s["tok_per_s"] for s in samples)
    return {"state": "MEASURED", "engine": name, "endpoint": endpoint, "model": model,
            "runs": runs,
            "ttft_ms": {"p50": percentile(ttfts, 50), "p95": percentile(ttfts, 95),
                        "p99": percentile(ttfts, 99)},
            "total_ms": {"p50": percentile(totals, 50), "p95": percentile(totals, 95),
                         "p99": percentile(totals, 99)},
            "tok_per_s": {"p50": percentile(tps, 50)},
            "tokens_per_run": [s["tokens"] for s in samples]}


def compare(runs):
    """Fairness gate: identical model AND identical run count, or INVALID."""
    measured = [r for r in runs if r.get("state") == "MEASURED"]
    if len(measured) < 2:
        return {"state": "BLOCKED", "reason": "fewer than two MEASURED engines to compare"}
    models = {r["model"] for r in measured}
    nruns = {r["runs"] for r in measured}
    if len(models) != 1 or len(nruns) != 1:
        return {"state": "INVALID",
                "reason": "model or run-count differs across engines; comparison is unfair"}
    board = [{"engine": r["engine"], "ttft_p50": r["ttft_ms"]["p50"],
              "tok_per_s_p50": r["tok_per_s"]["p50"]} for r in measured]
    board.sort(key=lambda r: -(r["tok_per_s_p50"] or 0))
    return {"state": "MEASURED", "leaderboard": board, "winner": board[0]["engine"]}


def main():
    ap = argparse.ArgumentParser(description="Honest inference-engine benchmark harness")
    ap.add_argument("--model", default="local-model")
    ap.add_argument("--prompt", default="Explain hash-chained receipts in one sentence.")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--engines", nargs="*", default=list(ENGINES))
    args = ap.parse_args()
    chain = ReceiptChain()
    results = [run_engine(e, args.model, args.prompt, args.runs, args.max_tokens)
               for e in args.engines]
    verdict = compare(results)
    receipt = chain.emit({"type": "engine_bench", "verdict": verdict,
                          "states": {r["engine"]: r["state"] for r in results}})
    print(json.dumps({"results": results, "comparison": verdict,
                      "receipt": receipt, "chain_valid": chain.verify()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
