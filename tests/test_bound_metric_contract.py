# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for bound comparison metric shape."""
from __future__ import annotations

import copy
import hashlib

import benchmark_manifest as bm
import engine_bench as eb


def _manifest(endpoint: str = "http://127.0.0.1:9876") -> dict:
    return {
        "schema": bm.SCHEMA,
        "claim_boundary": bm.BOUNDARY,
        "subject": {
            "model_id": "fixture-model",
            "model_revision": "a" * 40,
            "weights_sha256": "1" * 64,
            "tokenizer_revision": "b" * 40,
            "tokenizer_sha256": "2" * 64,
            "template_sha256": "3" * 64,
            "adapter_sha256": None,
            "quantization": "none",
            "precision": "float16",
        },
        "workload": {
            "suite_sha256": "4" * 64,
            "prompt_sha256": hashlib.sha256(b"hi").hexdigest(),
            "runs": 1,
            "max_tokens": 6,
            "timeout_seconds": 2,
            "sampling": {"temperature": 0, "top_p": 1, "seed": 42},
            "concurrency": 1,
            "cache_state": "UNCONTROLLED",
        },
        "engines": {
            name: {
                "build_sha256": build * 64,
                "configuration_sha256": "6" * 64,
                "hardware_sha256": "7" * 64,
                "endpoint_sha256": bm.endpoint_digest(endpoint),
            }
            for name, build in (("vllm", "8"), ("sglang", "9"))
        },
    }


def _results(manifest: dict) -> list[dict]:
    request_hash = bm.digest(
        bm.request_payload(
            "fixture-model", "hi", 6, manifest["workload"]["sampling"]
        )
    )
    return [
        {
            "engine": name,
            "state": "MEASURED",
            "model": "fixture-model",
            "runs": 1,
            "endpoint": "http://127.0.0.1:9876",
            "binding": bm.run_binding(manifest, name, request_hash),
            "run_samples": [
                {
                    "state": "MEASURED",
                    "ttft_ms": 20,
                    "request_sha256": request_hash,
                }
            ],
            "itl_ms": {"p50": 1, "p95": 2, "p99": 3},
        }
        for name in ("vllm", "sglang")
    ]


def test_bound_comparison_accepts_physical_itl_percentiles():
    manifest = _manifest()
    verdict = eb.compare_bound(_results(manifest), manifest)
    assert verdict["state"] == "MEASURED"
    assert verdict["comparison_scope"] == "DECLARED_SAME_WORKLOAD_AND_HARDWARE"


def test_bound_comparison_rejects_negative_itl_summary():
    manifest = _manifest()
    results = _results(manifest)
    results[0]["itl_ms"] = {"p50": -1, "p95": 2, "p99": 3}

    verdict = eb.compare_bound(results, manifest)
    assert verdict["state"] == "INVALID"
    assert "non-negative" in verdict["reason"]


def test_bound_comparison_rejects_nonmonotonic_itl_summary():
    manifest = _manifest()
    results = _results(manifest)
    results[1]["itl_ms"] = {"p50": 3, "p95": 2, "p99": 4}

    verdict = eb.compare_bound(results, manifest)
    assert verdict["state"] == "INVALID"
    assert "monotonic" in verdict["reason"]


def test_bound_comparison_rejects_extra_itl_summary_fields():
    manifest = _manifest()
    results = _results(manifest)
    results[1]["itl_ms"]["source"] = "unbound"

    verdict = eb.compare_bound(copy.deepcopy(results), manifest)
    assert verdict["state"] == "INVALID"
    assert "missing or unrecognized" in verdict["reason"]
