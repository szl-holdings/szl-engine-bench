#!/usr/bin/env python3
"""szl-engine-bench — honest inference-engine benchmark harness.

Measures any OpenAI-compatible streaming endpoint (vLLM, SGLang, llama.cpp,
MLX, TGI, Transformers serve). Stdlib only. No downloads or API keys are
needed for local servers.

An engine without a configured endpoint returns BLOCKED with a reason. Bad
configuration or incomparable input returns INVALID. Runtime and protocol
errors return FAILED. Measurements are never synthesized from those states.
"""

import argparse
import hashlib
import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping

__version__ = "0.2.0"

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
    """Return the single JSON representation used by every repository hash."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_canonical(obj):
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class ReceiptChain:
    """Hash-chained run receipts. UNSIGNED_HONEST proves order, not identity."""

    def __init__(self):
        self.chain = []

    def emit(self, record):
        prev = self.chain[-1]["self_hash"] if self.chain else GENESIS
        body = {
            "prev_hash": prev,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "signature": "UNSIGNED_HONEST",
            "run": record,
        }
        body["self_hash"] = _sha256_canonical(body)
        self.chain.append(body)
        return body

    def verify(self):
        prev = GENESIS
        try:
            for receipt in self.chain:
                if not isinstance(receipt, Mapping) or receipt["prev_hash"] != prev:
                    return False
                check = dict(receipt)
                claimed_hash = check.pop("self_hash")
                if not isinstance(claimed_hash, str) or len(claimed_hash) != 64:
                    return False
                if _sha256_canonical(check) != claimed_hash:
                    return False
                prev = claimed_hash
        except (KeyError, TypeError, ValueError):
            return False
        return True


def percentile(values, p):
    """Linearly interpolated percentile (the same rule as the issue fixture)."""
    if not _finite_number(p) or not 0 <= float(p) <= 100:
        raise ValueError("p must be a finite number from 0 through 100")
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError("values must be an iterable of finite numbers")
    try:
        ordered = list(values)
    except TypeError as exc:
        raise ValueError("values must be an iterable of finite numbers") from exc
    if not ordered:
        return None
    if any(not _finite_number(value) for value in ordered):
        raise ValueError("values must contain only finite numbers")
    ordered = sorted(float(value) for value in ordered)
    position = (len(ordered) - 1) * (float(p) / 100.0)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_stats(chunk_times, token_count=None):
    """Measure one stream from non-empty chunk arrivals relative to request start.

    ``chunk_times`` is expressed in seconds. ``token_count`` must come from an
    endpoint-provided usage record; it may be ``None`` when the endpoint does
    not report usage. Stream chunks are not assumed to be individual tokens.
    """
    if chunk_times is None or isinstance(chunk_times, (str, bytes)):
        return {"state": "INVALID", "detail": "chunk_times must be a sequence"}
    try:
        times = list(chunk_times)
    except TypeError:
        return {"state": "INVALID", "detail": "chunk_times must be a sequence"}
    if len(times) < 2:
        return {"state": "INVALID", "detail": "need >= 2 chunk arrivals"}
    if any(not _finite_number(value) or float(value) < 0 for value in times):
        return {
            "state": "INVALID",
            "detail": "chunk arrivals must be finite non-negative seconds",
        }
    times = [float(value) for value in times]
    if any(current < previous for previous, current in zip(times, times[1:])):
        return {
            "state": "INVALID",
            "detail": "chunk arrivals must be monotonic non-decreasing",
        }
    if token_count is not None and (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count < 0
    ):
        return {
            "state": "INVALID",
            "detail": "token_count must be a non-negative integer or null",
        }

    gaps_ms = [
        round((current - previous) * 1000.0, 6)
        for previous, current in zip(times, times[1:])
    ]
    last_arrival = times[-1]
    ttft_ms = round(times[0] * 1000.0, 3)
    itl_p50_ms = round(percentile(gaps_ms, 50), 3)
    itl_p95_ms = round(percentile(gaps_ms, 95), 3)
    itl_p99_ms = round(percentile(gaps_ms, 99), 3)
    return {
        "state": "MEASURED",
        "ttft_ms": ttft_ms,
        "itl_p50_ms": itl_p50_ms,
        "itl_p95_ms": itl_p95_ms,
        "itl_p99_ms": itl_p99_ms,
        "itl_gaps_ms": gaps_ms,
        "itl_gaps_sha256": _sha256_canonical(gaps_ms),
        "last_chunk_ms": round(last_arrival * 1000.0, 3),
        "chunks": len(times),
        "completion_tokens": token_count,
        "tok_per_s": (
            round(token_count / last_arrival, 2)
            if token_count is not None and last_arrival > 0
            else None
        ),
        "chunks_per_s": (
            round(len(times) / last_arrival, 2) if last_arrival > 0 else None
        ),
    }


def measure_once(endpoint, model, prompt, max_tokens, timeout=120):
    """Execute one streaming request and retain real non-empty chunk arrivals."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=body,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
    )
    started = time.perf_counter()
    chunk_times = []
    completion_tokens = None
    saw_done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type != "text/event-stream":
            raise ValueError(
                f"expected text/event-stream response, received {content_type}"
            )
        for raw in response:
            arrival = time.perf_counter() - started
            line = raw.decode("utf-8", "strict").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise ValueError("stream contained a non-SSE data line")
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError("stream contained invalid JSON data") from exc
            if not isinstance(chunk, Mapping):
                raise ValueError("stream JSON data must be an object")
            usage = chunk.get("usage")
            if isinstance(usage, Mapping) and "completion_tokens" in usage:
                completion_tokens = usage["completion_tokens"]
            choices = chunk.get("choices") or []
            if choices and not isinstance(choices, list):
                raise ValueError("stream choices must be an array")
            text = (choices[0] if choices else {}).get("text", "")
            if text:
                chunk_times.append(arrival)
    if not saw_done:
        raise ValueError("stream ended before the [DONE] event")
    total_ms = round((time.perf_counter() - started) * 1000.0, 3)
    stats = run_stats(chunk_times, completion_tokens)
    stats["total_ms"] = total_ms
    return stats


def _validate_endpoint(endpoint):
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return "endpoint must be a valid HTTP(S) URL"
    if parsed.scheme not in ("http", "https") or not hostname:
        return "endpoint must be an absolute HTTP(S) URL"
    if port == 0:
        return "endpoint port must be from 1 through 65535"
    if parsed.username is not None or parsed.password is not None:
        return "endpoint URL must not embed credentials"
    if parsed.query or parsed.fragment:
        return "endpoint base URL must not contain a query or fragment"
    return None


def run_engine(name, model, prompt, runs, max_tokens, timeout=120):
    if name not in ENGINES:
        return {"state": "INVALID", "engine": name, "reason": "unknown engine name"}
    if not isinstance(model, str) or not model.strip():
        return {
            "state": "INVALID",
            "engine": name,
            "reason": "model must be a non-empty string",
        }
    if not isinstance(prompt, str) or not prompt:
        return {
            "state": "INVALID",
            "engine": name,
            "reason": "prompt must be a non-empty string",
        }
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
        return {
            "state": "INVALID",
            "engine": name,
            "reason": "runs must be an integer >= 1",
        }
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 1
    ):
        return {
            "state": "INVALID",
            "engine": name,
            "reason": "max_tokens must be an integer >= 1",
        }
    if not _finite_number(timeout) or float(timeout) <= 0:
        return {
            "state": "INVALID",
            "engine": name,
            "reason": "timeout must be a finite number > 0",
        }

    variable = ENGINES[name]
    endpoint = os.environ.get(variable)
    if not endpoint:
        return {
            "state": "BLOCKED",
            "engine": name,
            "reason": f"{variable} not set; refusing to fabricate results",
        }
    endpoint_error = _validate_endpoint(endpoint)
    if endpoint_error:
        return {"state": "INVALID", "engine": name, "reason": endpoint_error}

    samples = []
    try:
        for run_index in range(runs):
            sample = measure_once(
                endpoint, model, prompt, max_tokens, timeout=float(timeout)
            )
            if sample.get("state") != "MEASURED":
                return {
                    "state": "INVALID",
                    "engine": name,
                    "reason": sample.get("detail", "run did not produce valid metrics"),
                    "run_index": run_index,
                }
            samples.append(sample)
    except Exception as exc:  # the actual network/protocol failure is retained
        return {
            "state": "FAILED",
            "engine": name,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    ttfts = [sample["ttft_ms"] for sample in samples]
    totals = [sample["total_ms"] for sample in samples]
    all_gaps = [gap for sample in samples for gap in sample["itl_gaps_ms"]]
    chunk_rates = [
        sample["chunks_per_s"]
        for sample in samples
        if sample["chunks_per_s"] is not None
    ]
    token_rates = [
        sample["tok_per_s"] for sample in samples if sample["tok_per_s"] is not None
    ]
    nested_gaps = [sample["itl_gaps_ms"] for sample in samples]

    return {
        "state": "MEASURED",
        "engine": name,
        "endpoint": endpoint,
        "model": model,
        "runs": runs,
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 2),
            "p95": round(percentile(ttfts, 95), 2),
            "p99": round(percentile(ttfts, 99), 2),
        },
        "itl_ms": {
            "p50": round(percentile(all_gaps, 50), 3),
            "p95": round(percentile(all_gaps, 95), 3),
            "p99": round(percentile(all_gaps, 99), 3),
        },
        "total_ms": {
            "p50": round(percentile(totals, 50), 2),
            "p95": round(percentile(totals, 95), 2),
            "p99": round(percentile(totals, 99), 2),
        },
        "tok_per_s": {
            "state": "MEASURED" if len(token_rates) == runs else "UNAVAILABLE",
            "p50": (
                round(percentile(token_rates, 50), 2)
                if len(token_rates) == runs
                else None
            ),
        },
        "chunks_per_s": {
            "p50": round(percentile(chunk_rates, 50), 2),
        },
        "ttft_samples_ms": ttfts,
        "completion_tokens_per_run": [
            sample["completion_tokens"] for sample in samples
        ],
        "chunks_per_run": [sample["chunks"] for sample in samples],
        "run_samples": samples,
        "itl_gaps_sha256": _sha256_canonical(nested_gaps),
    }


def _extract_ttft_samples(runs, label):
    if runs is None or isinstance(runs, (str, bytes)):
        return None, f"{label} runs must be a non-empty sequence"
    try:
        items = list(runs)
    except TypeError:
        return None, f"{label} runs must be a non-empty sequence"
    if not items:
        return None, f"{label} runs must be a non-empty sequence"

    values = []
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            if item.get("state", "MEASURED") != "MEASURED":
                return None, f"{label} run {index} is not MEASURED"
            value = item.get("ttft_ms")
        elif isinstance(item, (list, tuple)) and item:
            # Public fixture compatibility: chunk arrival sequences are seconds.
            value = float(item[0]) * 1000.0 if _finite_number(item[0]) else item[0]
        else:
            return None, f"{label} run {index} has no TTFT"
        if not _finite_number(value) or float(value) < 0:
            return None, f"{label} run {index} TTFT must be finite and non-negative"
        values.append(float(value))
    return values, None


def _goodput_summary(ttfts_ms, slo_ttft_ms):
    good_runs = sum(value <= slo_ttft_ms for value in ttfts_ms)
    return {
        "ttft_p50_ms": round(percentile(ttfts_ms, 50), 2),
        "ttft_p95_ms": round(percentile(ttfts_ms, 95), 2),
        "goodput_at_slo": round(good_runs / len(ttfts_ms), 4),
        "good_runs": good_runs,
        "runs": len(ttfts_ms),
    }


def compare_engines(a_runs, b_runs, slo_ttft_ms=200):
    """Compare two equally sized run sets without reducing them to one winner."""
    if not _finite_number(slo_ttft_ms) or float(slo_ttft_ms) <= 0:
        return {"state": "INVALID", "reason": "slo_ttft_ms must be a finite number > 0"}
    a_ttfts, error = _extract_ttft_samples(a_runs, "a")
    if error:
        return {"state": "INVALID", "reason": error}
    b_ttfts, error = _extract_ttft_samples(b_runs, "b")
    if error:
        return {"state": "INVALID", "reason": error}
    if len(a_ttfts) != len(b_ttfts):
        return {
            "state": "INVALID",
            "reason": "run-count differs across engines; comparison is unfair",
        }
    slo = float(slo_ttft_ms)
    return {
        "state": "MEASURED",
        "slo_ttft_ms": round(slo, 3),
        "a": _goodput_summary(a_ttfts, slo),
        "b": _goodput_summary(b_ttfts, slo),
        "label": "TTFT and goodput; inspect ITL separately, never one winner",
    }


def compare(runs, slo_ttft_ms=200):
    """Preserve model/run fairness, then report distributions for every engine."""
    if not _finite_number(slo_ttft_ms) or float(slo_ttft_ms) <= 0:
        return {"state": "INVALID", "reason": "slo_ttft_ms must be a finite number > 0"}
    if runs is None or isinstance(runs, (str, bytes)):
        return {"state": "INVALID", "reason": "runs must be a sequence of results"}
    try:
        results = list(runs)
    except TypeError:
        return {"state": "INVALID", "reason": "runs must be a sequence of results"}
    if any(not isinstance(result, Mapping) for result in results):
        return {"state": "INVALID", "reason": "every result must be an object"}
    measured = [result for result in results if result.get("state") == "MEASURED"]
    if len(measured) < 2:
        return {
            "state": "BLOCKED",
            "reason": "fewer than two MEASURED engines to compare",
        }
    for result in measured:
        if not isinstance(result.get("engine"), str) or not result["engine"]:
            return {
                "state": "INVALID",
                "reason": "every MEASURED result needs an engine name",
            }
        if not isinstance(result.get("model"), str) or not result["model"]:
            return {
                "state": "INVALID",
                "reason": f"{result['engine']} has an invalid model",
            }
        if (
            not isinstance(result.get("runs"), int)
            or isinstance(result["runs"], bool)
            or result["runs"] < 1
        ):
            return {
                "state": "INVALID",
                "reason": f"{result['engine']} has an invalid run count",
            }
        itl = result.get("itl_ms")
        if not isinstance(itl, Mapping) or any(
            not _finite_number(itl.get(key)) for key in ("p50", "p95", "p99")
        ):
            return {
                "state": "INVALID",
                "reason": f"{result['engine']} has invalid ITL metrics",
            }
    models = {result["model"] for result in measured}
    run_counts = {result["runs"] for result in measured}
    if len(models) != 1 or len(run_counts) != 1:
        return {
            "state": "INVALID",
            "reason": "model or run-count differs across engines; comparison is unfair",
        }
    names = [result["engine"] for result in measured]
    if len(set(names)) != len(names):
        return {
            "state": "INVALID",
            "reason": "duplicate engine names make comparison ambiguous",
        }

    slo = float(slo_ttft_ms)
    metrics = {}
    for result in measured:
        ttfts, error = _extract_ttft_samples(
            result.get("run_samples"), result["engine"]
        )
        if error or len(ttfts) != result["runs"]:
            return {
                "state": "INVALID",
                "reason": error or f"{result['engine']} run samples are incomplete",
            }
        summary = _goodput_summary(ttfts, slo)
        summary.update(
            {
                "itl_p50_ms": result["itl_ms"]["p50"],
                "itl_p95_ms": result["itl_ms"]["p95"],
                "itl_p99_ms": result["itl_ms"]["p99"],
            }
        )
        metrics[result["engine"]] = summary

    return {
        "state": "MEASURED",
        "model": measured[0]["model"],
        "runs": measured[0]["runs"],
        "slo_ttft_ms": round(slo, 3),
        "engines": metrics,
        "label": "TTFT, ITL, and goodput; never a single-number winner",
    }


def _result_receipt_record(results, verdict, args):
    return {
        "type": "engine_bench",
        "schema_version": 2,
        "benchmark_version": __version__,
        "config": {
            "model": args.model,
            "engines": args.engines,
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "slo_ttft_ms": args.slo_ttft_ms,
            "timeout_seconds": args.timeout_seconds,
            "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        },
        "verdict": verdict,
        "states": {result["engine"]: result["state"] for result in results},
        "result_sha256": {
            result["engine"]: _sha256_canonical(result) for result in results
        },
        "itl_gaps_sha256": {
            result["engine"]: result["itl_gaps_sha256"]
            for result in results
            if result["state"] == "MEASURED"
        },
        "itl_run_gaps_sha256": {
            result["engine"]: [
                sample["itl_gaps_sha256"] for sample in result["run_samples"]
            ]
            for result in results
            if result["state"] == "MEASURED"
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Honest inference-engine benchmark harness"
    )
    parser.add_argument("--model", default="local-model")
    parser.add_argument(
        "--prompt", default="Explain hash-chained receipts in one sentence."
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--slo-ttft-ms", type=float, default=200.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--engines", nargs="*", choices=sorted(ENGINES), default=list(ENGINES)
    )
    args = parser.parse_args()

    chain = ReceiptChain()
    if len(args.engines) != len(set(args.engines)):
        reason = "duplicate engine selections are invalid"
        results = [{"state": "INVALID", "engine": "selection", "reason": reason}]
        verdict = {"state": "INVALID", "reason": reason}
    else:
        results = [
            run_engine(
                engine,
                args.model,
                args.prompt,
                args.runs,
                args.max_tokens,
                timeout=args.timeout_seconds,
            )
            for engine in args.engines
        ]
        verdict = compare(results, args.slo_ttft_ms)
    receipt = chain.emit(_result_receipt_record(results, verdict, args))
    print(
        json.dumps(
            {
                "benchmark_version": __version__,
                "results": results,
                "comparison": verdict,
                "receipt": receipt,
                "chain_valid": chain.verify(),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
