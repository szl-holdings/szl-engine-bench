"""Tests run against a real local SSE streaming server — no fabricated fixtures."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import engine_bench as eb


class MockEngine(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for tok in ["Hash", "-chained", " receipts", " prove", " order", "."]:
            self.wfile.write(f"data: {json.dumps({'choices': [{'text': tok}]})}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.005)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), MockEngine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_measured_over_real_socket(server, monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", server)
    r = eb.run_engine("vllm", "mock-1", "hi", 2, 6)
    assert r["state"] == "MEASURED"
    assert r["ttft_ms"]["p50"] > 0 and r["tokens_per_run"] == [6, 6]
    assert r["tok_per_s"]["p50"] > 0


def test_blocked_without_endpoint(monkeypatch):
    monkeypatch.delenv("SGLANG_ENDPOINT", raising=False)
    r = eb.run_engine("sglang", "m", "p", 1, 4)
    assert r["state"] == "BLOCKED" and "refusing" in r["reason"]


def test_failed_on_connection_refused(monkeypatch):
    monkeypatch.setenv("MLX_ENDPOINT", "http://127.0.0.1:1")
    r = eb.run_engine("mlx", "m", "p", 1, 4)
    assert r["state"] == "FAILED" and r["reason"]


def test_percentile():
    assert eb.percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert eb.percentile([], 50) is None


def test_receipt_chain_verifies_and_detects_tamper():
    c = eb.ReceiptChain()
    c.emit({"a": 1})
    c.emit({"b": 2})
    assert c.verify()
    c.chain[0]["run"]["a"] = 999
    assert not c.verify()


def test_compare_blocked_with_one_measured():
    assert eb.compare([{"state": "MEASURED", "engine": "a"}])["state"] == "BLOCKED"


def test_compare_invalid_on_model_mismatch():
    mk = lambda e, m: {"state": "MEASURED", "engine": e, "model": m, "runs": 3,
                       "ttft_ms": {"p50": 1.0}, "tok_per_s": {"p50": 10.0}}
    assert eb.compare([mk("a", "m1"), mk("b", "m2")])["state"] == "INVALID"


def test_compare_fair_winner():
    mk = lambda e, tps: {"state": "MEASURED", "engine": e, "model": "m1", "runs": 3,
                         "ttft_ms": {"p50": 1.0}, "tok_per_s": {"p50": tps}}
    v = eb.compare([mk("slow", 10.0), mk("fast", 20.0)])
    assert v["state"] == "MEASURED" and v["winner"] == "fast"
