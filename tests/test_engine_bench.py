"""Network-path, fixture, validation, comparison, and receipt tests."""

import argparse
import json
import math
import sys
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import engine_bench as eb


class MockEngine(BaseHTTPRequestHandler):
    tokens = ["Hash", "-chained", " receipts", " prove", " order", "."]
    include_usage = True
    content_type = "text/event-stream"
    send_done = True

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", self.content_type)
        self.end_headers()
        if self.content_type != "text/event-stream":
            return
        for token in self.tokens:
            event = {"choices": [{"text": token}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.005)
        if self.include_usage:
            event = {"choices": [], "usage": {"completion_tokens": len(self.tokens)}}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        if self.send_done:
            self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args):
        pass


class SingleChunkEngine(MockEngine):
    tokens = ["one response chunk"]


class NoUsageEngine(MockEngine):
    include_usage = False


class WrongContentTypeEngine(MockEngine):
    content_type = "application/json"

    def do_POST(self):
        body = b'{"not":"a stream"}'
        self.send_response(200)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


class MissingDoneEngine(MockEngine):
    send_done = False


@pytest.fixture
def server_factory():
    servers = []

    def start(handler=MockEngine):
        server = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _fake_engine_result(name, model, ttfts, itl=(5.0, 7.0, 9.0)):
    return {
        "state": "MEASURED",
        "engine": name,
        "model": model,
        "runs": len(ttfts),
        "run_samples": [{"state": "MEASURED", "ttft_ms": value} for value in ttfts],
        "itl_ms": {"p50": itl[0], "p95": itl[1], "p99": itl[2]},
    }


def test_measured_over_real_socket_includes_itl_and_gap_hash(
    server_factory, monkeypatch
):
    monkeypatch.setenv("VLLM_ENDPOINT", server_factory())
    result = eb.run_engine("vllm", "mock-1", "hi", 2, 6)

    assert result["state"] == "MEASURED"
    assert result["ttft_ms"]["p50"] > 0
    assert result["itl_ms"]["p50"] > 0
    assert result["itl_ms"]["p99"] >= result["itl_ms"]["p50"]
    assert result["completion_tokens_per_run"] == [6, 6]
    assert result["tok_per_s"]["state"] == "MEASURED"
    assert result["tok_per_s"]["p50"] > 0
    nested_gaps = [sample["itl_gaps_ms"] for sample in result["run_samples"]]
    assert result["itl_gaps_sha256"] == eb._sha256_canonical(nested_gaps)


def test_two_real_socket_results_produce_multi_metric_comparison(
    server_factory, monkeypatch
):
    endpoint = server_factory()
    monkeypatch.setenv("VLLM_ENDPOINT", endpoint)
    monkeypatch.setenv("SGLANG_ENDPOINT", endpoint)
    results = [
        eb.run_engine("vllm", "mock-1", "hi", 1, 6),
        eb.run_engine("sglang", "mock-1", "hi", 1, 6),
    ]

    comparison = eb.compare(results, slo_ttft_ms=200)
    assert comparison["state"] == "MEASURED"
    assert comparison["engines"]["vllm"]["goodput_at_slo"] == 1.0
    assert comparison["engines"]["vllm"]["itl_p99_ms"] > 0
    assert "winner" not in comparison


def test_missing_usage_does_not_fabricate_token_throughput(server_factory, monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", server_factory(NoUsageEngine))
    result = eb.run_engine("vllm", "mock-1", "hi", 1, 6)

    assert result["state"] == "MEASURED"
    assert result["tok_per_s"] == {"state": "UNAVAILABLE", "p50": None}
    assert result["chunks_per_s"]["p50"] > 0


def test_single_chunk_stream_is_invalid(server_factory, monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", server_factory(SingleChunkEngine))
    result = eb.run_engine("vllm", "mock-1", "hi", 1, 6)

    assert result["state"] == "INVALID"
    assert result["reason"] == "need >= 2 chunk arrivals"


def test_wrong_content_type_is_failed(server_factory, monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", server_factory(WrongContentTypeEngine))
    result = eb.run_engine("vllm", "mock-1", "hi", 1, 6)

    assert result["state"] == "FAILED"
    assert result["reason"]


def test_measure_once_rejects_wrong_content_type_deterministically(monkeypatch):
    class JsonHeaders:
        @staticmethod
        def get_content_type():
            return "application/json"

    class JsonResponse:
        headers = JsonHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        eb.urllib.request, "urlopen", lambda request, timeout: JsonResponse()
    )
    with pytest.raises(ValueError, match="expected text/event-stream"):
        eb.measure_once("https://example.invalid", "m", "p", 1)


def test_chunk_arrival_is_timestamped_before_json_parsing(monkeypatch):
    clock = {"value": 0.0}
    original_loads = eb.json.loads

    class EventHeaders:
        @staticmethod
        def get_content_type():
            return "text/event-stream"

    class EventResponse:
        headers = EventHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            events = [
                (0.1, b'data: {"choices":[{"text":"a"}]}\n'),
                (1.2, b'data: {"choices":[{"text":"b"}]}\n'),
                (2.3, b"data: [DONE]\n"),
            ]
            for timestamp, event in events:
                clock["value"] = timestamp
                yield event

    def slow_loads(payload):
        result = original_loads(payload)
        clock["value"] += 1.0
        return result

    monkeypatch.setattr(eb.time, "perf_counter", lambda: clock["value"])
    monkeypatch.setattr(eb.json, "loads", slow_loads)
    monkeypatch.setattr(
        eb.urllib.request, "urlopen", lambda request, timeout: EventResponse()
    )

    result = eb.measure_once("https://example.invalid", "m", "p", 2)
    assert result["state"] == "MEASURED"
    assert result["ttft_ms"] == 100.0


def test_stream_without_done_event_is_failed(server_factory, monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", server_factory(MissingDoneEngine))
    result = eb.run_engine("vllm", "mock-1", "hi", 1, 6)

    assert result["state"] == "FAILED"
    assert "before the [DONE]" in result["reason"]


def test_blocked_without_endpoint(monkeypatch):
    monkeypatch.delenv("SGLANG_ENDPOINT", raising=False)
    result = eb.run_engine("sglang", "m", "p", 1, 4)
    assert result["state"] == "BLOCKED"
    assert "refusing" in result["reason"]


def test_failed_on_connection_refused(monkeypatch):
    monkeypatch.setenv("MLX_ENDPOINT", "http://127.0.0.1:1")
    result = eb.run_engine("mlx", "m", "p", 1, 4, timeout=0.5)
    assert result["state"] == "FAILED"
    assert result["reason"]


@pytest.mark.parametrize(
    ("name", "model", "prompt", "runs", "max_tokens", "reason"),
    [
        ("unknown", "m", "p", 1, 4, "unknown engine"),
        ("vllm", "", "p", 1, 4, "model"),
        ("vllm", "m", "", 1, 4, "prompt"),
        ("vllm", "m", "p", 0, 4, "runs"),
        ("vllm", "m", "p", True, 4, "runs"),
        ("vllm", "m", "p", 1, 0, "max_tokens"),
    ],
)
def test_run_engine_rejects_invalid_configuration(
    name, model, prompt, runs, max_tokens, reason
):
    result = eb.run_engine(name, model, prompt, runs, max_tokens)
    assert result["state"] == "INVALID"
    assert reason in result["reason"]


def test_endpoint_with_embedded_credentials_is_invalid(monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", "https://user:secret@example.invalid")
    result = eb.run_engine("vllm", "m", "p", 1, 4)
    assert result["state"] == "INVALID"
    assert "credentials" in result["reason"]


@pytest.mark.parametrize(
    "endpoint",
    ["http://host:notaport", "http://host:99999", "http://host:0"],
)
def test_endpoint_with_invalid_port_is_invalid(monkeypatch, endpoint):
    monkeypatch.setenv("VLLM_ENDPOINT", endpoint)
    result = eb.run_engine("vllm", "m", "p", 1, 4)
    assert result["state"] == "INVALID"
    assert "endpoint" in result["reason"]


def test_percentile_uses_linear_interpolation_and_sorts_values():
    assert eb.percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert eb.percentile([10, 10, 330, 50, 50, 50, 50], 95) == pytest.approx(246)
    assert eb.percentile([], 50) is None


@pytest.mark.parametrize(
    ("values", "p"),
    [([1.0, math.nan], 50), ([1.0], True), ([1.0], 101), ("123", 50)],
)
def test_percentile_rejects_noncanonical_input(values, p):
    with pytest.raises(ValueError):
        eb.percentile(values, p)


def test_bursty_fixture_reproduces_issue_values_independently():
    arrivals_seconds = [0.05, 0.06, 0.07, 0.4, 0.45, 0.5, 0.55, 0.6]
    result = eb.run_stats(arrivals_seconds, token_count=8)

    assert result["state"] == "MEASURED"
    assert result["ttft_ms"] == 50.0
    assert result["itl_p50_ms"] == 50.0
    assert result["itl_p95_ms"] == 246.0
    assert result["itl_p99_ms"] == 313.2
    assert result["itl_gaps_sha256"] == eb._sha256_canonical(result["itl_gaps_ms"])


@pytest.mark.parametrize(
    ("chunk_times", "token_count", "detail"),
    [
        ([0.1], 1, "need >= 2"),
        ([0.2, 0.1], 2, "monotonic"),
        ([0.1, math.inf], 2, "finite"),
        ([0.1, 0.2], True, "token_count"),
        ("0.1,0.2", 2, "sequence"),
    ],
)
def test_run_stats_fails_closed(chunk_times, token_count, detail):
    result = eb.run_stats(chunk_times, token_count)
    assert result["state"] == "INVALID"
    assert detail in result["detail"]


def test_compare_engines_reproduces_goodput_and_includes_slo_boundary():
    a_runs = [[0.100, 0.110]] * 9 + [[0.201, 0.211]]
    b_runs = [[0.200, 0.210]] * 10
    result = eb.compare_engines(a_runs, b_runs, slo_ttft_ms=200)

    assert result["state"] == "MEASURED"
    assert result["a"]["goodput_at_slo"] == 0.9
    assert result["b"]["goodput_at_slo"] == 1.0
    assert result["b"]["good_runs"] == 10


def test_compare_engines_rejects_bad_slo_and_run_count():
    assert eb.compare_engines([[0.1]], [[0.1]], 0)["state"] == "INVALID"
    result = eb.compare_engines([[0.1]], [[0.1], [0.2]], 200)
    assert result["state"] == "INVALID"
    assert "run-count" in result["reason"]


def test_compare_blocked_with_one_measured():
    result = eb.compare([_fake_engine_result("a", "m", [10.0])])
    assert result["state"] == "BLOCKED"


def test_compare_preserves_model_and_run_count_fairness_gates():
    mismatch = eb.compare(
        [
            _fake_engine_result("a", "m1", [10.0, 20.0]),
            _fake_engine_result("b", "m2", [10.0, 20.0]),
        ]
    )
    assert mismatch["state"] == "INVALID"

    mismatch = eb.compare(
        [
            _fake_engine_result("a", "m", [10.0]),
            _fake_engine_result("b", "m", [10.0, 20.0]),
        ]
    )
    assert mismatch["state"] == "INVALID"


def test_compare_rejects_malformed_result_objects():
    assert eb.compare(None)["state"] == "INVALID"
    assert eb.compare(["bad", "data"])["state"] == "INVALID"
    result = eb.compare(
        [
            {"state": "MEASURED", "engine": "a"},
            {"state": "MEASURED", "engine": "b"},
        ]
    )
    assert result["state"] == "INVALID"


def test_compare_reports_distributions_and_no_single_winner():
    result = eb.compare(
        [
            _fake_engine_result("a", "m", [100.0, 150.0, 250.0]),
            _fake_engine_result("b", "m", [100.0, 120.0, 130.0]),
        ],
        slo_ttft_ms=200,
    )

    assert result["state"] == "MEASURED"
    assert result["engines"]["a"]["ttft_p50_ms"] == 150.0
    assert result["engines"]["a"]["goodput_at_slo"] == pytest.approx(0.6667)
    assert result["engines"]["a"]["itl_p99_ms"] == 9.0
    assert "winner" not in result


def test_receipt_chain_verifies_and_detects_tamper_or_malformed_data():
    chain = eb.ReceiptChain()
    chain.emit({"a": 1})
    chain.emit({"b": 2})
    assert chain.verify()
    chain.chain[0]["run"]["a"] = 999
    assert not chain.verify()

    malformed = eb.ReceiptChain()
    malformed.chain.append({"prev_hash": eb.GENESIS})
    assert not malformed.verify()


def test_receipt_record_anchors_results_and_gap_hashes():
    result = {
        "state": "MEASURED",
        "engine": "a",
        "itl_gaps_sha256": "a" * 64,
        "run_samples": [{"itl_gaps_sha256": "b" * 64}],
    }
    args = argparse.Namespace(
        model="m",
        engines=["a"],
        runs=1,
        max_tokens=4,
        slo_ttft_ms=200.0,
        timeout_seconds=120.0,
        prompt="p",
    )
    record = eb._result_receipt_record([result], {"state": "BLOCKED"}, args)

    assert record["benchmark_version"] == "0.2.0"
    assert record["itl_gaps_sha256"] == {"a": "a" * 64}
    assert record["itl_run_gaps_sha256"] == {"a": ["b" * 64]}
    assert record["result_sha256"]["a"] == eb._sha256_canonical(result)


def test_cli_without_endpoints_emits_blocked_receipted_json(monkeypatch, capsys):
    monkeypatch.delenv("VLLM_ENDPOINT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["engine_bench.py", "--engines", "vllm", "--runs", "1"],
    )
    eb.main()
    output = json.loads(capsys.readouterr().out)

    assert output["benchmark_version"] == "0.2.0"
    assert output["results"][0]["state"] == "BLOCKED"
    assert output["receipt"]["run"]["itl_gaps_sha256"] == {}
    assert output["receipt"]["run"]["itl_run_gaps_sha256"] == {}
    assert output["chain_valid"] is True


def test_cli_rejects_duplicate_engines_before_measurement(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(
        sys,
        "argv",
        ["engine_bench.py", "--engines", "vllm", "vllm", "--runs", "1"],
    )
    eb.main()
    output = json.loads(capsys.readouterr().out)

    assert output["results"] == [
        {
            "state": "INVALID",
            "engine": "selection",
            "reason": "duplicate engine selections are invalid",
        }
    ]
    assert output["comparison"]["state"] == "INVALID"
    assert output["receipt"]["run"]["config"]["engines"] == ["vllm", "vllm"]
    assert output["chain_valid"] is True


def test_module_and_package_versions_match():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert eb.__version__ == project["project"]["version"] == "0.2.0"
