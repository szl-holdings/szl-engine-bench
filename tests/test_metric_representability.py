"""Physical metric and finite-output regressions, using only synthetic inputs."""
import copy
import json
import math
from pathlib import Path

import pytest

from szl_engine_bench import benchmark_manifest as bm
from szl_engine_bench import engine_bench as eb


@pytest.mark.parametrize("arrivals,tokens", [
    ([10**400, 10**400], 2),
    ([1e308, 1e308], 2),
    ([0, 1e308], 2),
    ([1e-310, 2e-310], 2),
    ([0.1, 0.2], 10**400),
])
def test_unrepresentable_stream_metrics_are_invalid(arrivals, tokens):
    result = eb.run_stats(arrivals, tokens)
    assert result["state"] == "INVALID"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("arrivals,tokens", [
    ([0.01, 0.02, 0.03], 12),
    ([1e305, 1e305], 2),
    ([0, 0], None),
    ([1e-100, 2e-100], 0),
])
def test_representable_stream_metrics_remain_measured(arrivals, tokens):
    result = eb.run_stats(arrivals, tokens)
    assert result["state"] == "MEASURED"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("p,expected", [(0, -1e308), (25, -5e307), (50, 0), (75, 5e307), (100, 1e308)])
def test_percentile_interpolation_does_not_overflow(p, expected):
    actual = eb.percentile([-1e308, 1e308], p)
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected)


def test_percentile_rejects_unrepresentable_integer_as_value_error():
    with pytest.raises(ValueError, match="finite"):
        eb.percentile([0, 10**400], 50)


def _results():
    return [
        {"engine": name, "state": "MEASURED", "model": "fixture-model", "runs": 1,
         "run_samples": [{"state": "MEASURED", "ttft_ms": 20}],
         "itl_ms": {"p50": 1, "p95": 2, "p99": 3}}
        for name in ("vllm", "sglang")
    ]


@pytest.mark.parametrize("itl", [
    {"p50": -1, "p95": 2, "p99": 3},
    {"p50": 3, "p95": 2, "p99": 4},
    {"p50": 1, "p95": 4, "p99": 3},
    {"p50": 1, "p95": 2, "p99": 10**400},
])
def test_unbound_comparison_rejects_impossible_itl(itl):
    results = _results()
    results[0]["itl_ms"] = copy.deepcopy(itl)
    assert eb.compare(results)["state"] == "INVALID"


def test_unbound_comparison_retains_finite_metrics_and_scope():
    results = _results()
    results[0]["itl_ms"]["note"] = "legacy diagnostic metadata"
    result = eb.compare(results)
    assert result["state"] == "MEASURED"
    assert result["comparison_scope"] == "UNBOUND_METRICS_ONLY"
    assert result["engines"]["vllm"]["itl_p99_ms"] == 3


def test_bound_comparison_rejects_integer_overflow_without_crashing():
    manifest = json.loads((Path(__file__).parents[1] / "examples/comparison.synthetic.json").read_text())
    endpoint = "http://127.0.0.1:9876"
    request_hash = bm.digest(bm.request_payload(
        "fixture-model", "hi", 6, manifest["workload"]["sampling"]))
    results = _results()
    for result in results:
        name = result["engine"]
        manifest["engines"][name]["endpoint_sha256"] = bm.endpoint_digest(endpoint)
    for result in results:
        result["endpoint"] = endpoint
        result["binding"] = bm.run_binding(manifest, result["engine"], request_hash)
        result["run_samples"][0]["request_sha256"] = request_hash
    assert eb.compare_bound(results, manifest)["state"] == "MEASURED"
    results[0]["itl_ms"]["p99"] = 10**400
    assert eb.compare_bound(results, manifest)["state"] == "INVALID"


@pytest.mark.parametrize("call", [
    lambda: eb.compare(_results(), 10**400),
    lambda: eb.compare_engines([[1]], [[1]], 10**400),
    lambda: eb.compare_engines([[10**400]], [[1]]),
])
def test_comparison_overflow_is_invalid_not_exception(call):
    assert call()["state"] == "INVALID"
