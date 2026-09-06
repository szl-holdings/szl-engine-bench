# SPDX-License-Identifier: Apache-2.0
"""Offline declarations and a loopback SSE server; no model or GPU is invoked."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler

import pytest

import benchmark_manifest as bm
import engine_bench as eb
from test_engine_bench import MockEngine, server_factory


def declaration(endpoint="http://127.0.0.1:9876"):
    # Synthetic identities exercise shape and comparison, not artifact validity.
    return {
        "schema": bm.SCHEMA,
        "claim_boundary": bm.BOUNDARY,
        "subject": {
            "model_id": "fixture-model", "model_revision": "a" * 40,
            "weights_sha256": "1" * 64, "tokenizer_revision": "b" * 40,
            "tokenizer_sha256": "2" * 64, "template_sha256": "3" * 64,
            "adapter_sha256": None, "quantization": "none", "precision": "float16",
        },
        "workload": {
            "suite_sha256": "4" * 64,
            "prompt_sha256": hashlib.sha256(b"hi").hexdigest(),
            "runs": 1, "max_tokens": 6, "timeout_seconds": 2,
            "sampling": {"temperature": 0, "top_p": 1, "seed": 42},
            "concurrency": 1, "cache_state": "UNCONTROLLED",
        },
        "engines": {
            name: {"build_sha256": char * 64, "configuration_sha256": "6" * 64,
                   "hardware_sha256": "7" * 64, "endpoint_sha256": bm.endpoint_digest(endpoint)}
            for name, char in (("vllm", "8"), ("sglang", "9"))
        },
    }


def arguments(**updates):
    return argparse.Namespace(**{
        "engines": ["vllm", "sglang"], "model": "fixture-model", "prompt": "hi",
        "runs": 1, "max_tokens": 6, "timeout_seconds": 2, "slo_ttft_ms": 200,
        **updates,
    })


def measured_results(manifest):
    request_hash = bm.digest(bm.request_payload("fixture-model", "hi", 6, manifest["workload"]["sampling"]))
    return [
        {"engine": name, "state": "MEASURED", "model": "fixture-model", "runs": 1,
         "endpoint": "http://127.0.0.1:9876",
         "binding": bm.run_binding(manifest, name, request_hash),
         "run_samples": [{"state": "MEASURED", "ttft_ms": 20,
                          "request_sha256": request_hash}],
         "itl_ms": {"p50": 1, "p95": 2, "p99": 3}}
        for name in ("vllm", "sglang")
    ]


def test_canonical_detached_manifest_and_explicit_claim_boundary():
    original = declaration()
    result = bm.validate_manifest(original)
    assert result == original and result is not original
    original["subject"]["model_revision"] = "c" * 40
    assert result["subject"]["model_revision"] == "a" * 40
    assert bm.digest(result) == bm.digest(json.loads(json.dumps(result, sort_keys=True)))
    assert result["claim_boundary"] == "OPERATOR_DECLARED_NOT_RUNTIME_ATTESTED"


@pytest.mark.parametrize("bad", ["a" * 40 + "\n", "a" * 40 + "\r\n", "A" * 40, "main", "a" * 39, "a" * 41, 1, None])
def test_exact_revision_rejects_whitespace_mutable_refs_and_wrong_type(bad):
    data = declaration()
    data["subject"]["model_revision"] = bad
    with pytest.raises(bm.ManifestError):
        bm.validate_manifest(data)


@pytest.mark.parametrize("path,bad", [
    (("subject", "weights_sha256"), "1" * 64 + "\n"),
    (("subject", "adapter_sha256"), "latest"),
    (("subject", "model_id"), "m\n"),
    (("workload", "runs"), True),
    (("workload", "runs"), 1001),
    (("workload", "max_tokens"), 0),
    (("workload", "timeout_seconds"), float("nan")),
    (("workload", "timeout_seconds"), float("inf")),
    (("workload", "timeout_seconds"), 10**1000),
    (("workload", "concurrency"), 2),
    (("workload", "concurrency"), True),
    (("workload", "cache_state"), "COLD"),
])
def test_declaration_budgets_and_no_invented_cache_or_concurrency(path, bad):
    data = declaration()
    data[path[0]][path[1]] = bad
    with pytest.raises(bm.ManifestError):
        bm.validate_manifest(data)


@pytest.mark.parametrize("field,bad", [("temperature", True), ("temperature", 2.1), ("top_p", 0), ("top_p", -1), ("seed", -1), ("seed", 2**31), ("seed", True)])
def test_sampling_has_one_bounded_contract(field, bad):
    data = declaration()
    data["workload"]["sampling"][field] = bad
    with pytest.raises(bm.ManifestError):
        bm.validate_manifest(data)


@pytest.mark.parametrize("path", [(), ("subject",), ("workload",), ("engines", "vllm"), ("workload", "sampling")])
def test_unknown_fields_are_rejected_at_each_contract_boundary(path):
    data = declaration()
    target = data
    for key in path:
        target = target[key]
    target["untracked"] = "not-admitted"
    with pytest.raises(bm.ManifestError):
        bm.validate_manifest(data)


def test_foreign_engine_and_different_hardware_are_refused():
    data = declaration()
    data["engines"]["other"] = data["engines"].pop("sglang")
    with pytest.raises(bm.ManifestError):
        bm.validate_manifest(data)
    data = declaration()
    data["engines"]["sglang"]["hardware_sha256"] = "c" * 64
    with pytest.raises(bm.ManifestError, match="hardware"):
        bm.validate_manifest(data)


@pytest.mark.parametrize("payload", [b'{"schema":1,"schema":2}', b'{"nested":{"a":1,"a":2}}', b'{"x":NaN}', b'\xff', b'{' * 2000])
def test_loader_rejects_ambiguous_json_without_echoing_input(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_bytes(payload)
    with pytest.raises(bm.ManifestError) as caught:
        bm.load_manifest(path)
    assert str(path) not in str(caught.value)


def test_loader_byte_budget_missing_file_and_valid_fixture(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b"x" * (bm.MAX_MANIFEST_BYTES + 1))
    with pytest.raises(bm.ManifestError, match="budget"):
        bm.load_manifest(path)
    path.unlink()
    with pytest.raises(bm.ManifestError):
        bm.load_manifest(path)
    path.write_text(json.dumps(declaration()))
    assert bm.load_manifest(path) == declaration()


@pytest.mark.parametrize("change", [{"model": "other"}, {"prompt": "other"}, {"runs": 2}, {"max_tokens": 8}, {"timeout_seconds": 3}, {"engines": ["vllm"]}])
def test_preflight_configuration_drift_makes_zero_requests(monkeypatch, change):
    for variable in ("VLLM_ENDPOINT", "SGLANG_ENDPOINT"):
        monkeypatch.setenv(variable, "http://127.0.0.1:9876")
    monkeypatch.setattr(eb, "measure_once", lambda *a, **k: pytest.fail("preflight must not issue a request"))
    results, verdict = eb.run_declared(arguments(**change), declaration())
    assert results == [] and verdict["state"] == "INVALID"


def test_missing_or_changed_endpoint_blocks_before_first_engine(monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", "http://127.0.0.1:9876")
    monkeypatch.delenv("SGLANG_ENDPOINT", raising=False)
    monkeypatch.setattr(eb, "measure_once", lambda *a, **k: pytest.fail("no request permitted"))
    assert eb.run_declared(arguments(), declaration())[1]["state"] == "BLOCKED"
    monkeypatch.setenv("SGLANG_ENDPOINT", "http://127.0.0.1:9877")
    assert eb.run_declared(arguments(), declaration())[1]["state"] == "INVALID"


def test_bound_comparison_records_declared_identity_without_promotion():
    data = declaration()
    verdict = eb.compare_bound(measured_results(data), data)
    assert verdict["state"] == "MEASURED"
    assert verdict["production_admission"] == "NOT_EVALUATED"
    assert verdict["runtime_identity"] == "NOT_ATTESTED"
    assert verdict["quality_evaluation"] == "NOT_PERFORMED"
    assert verdict["cache_state"] == "UNCONTROLLED"
    assert "winner" not in verdict


@pytest.mark.parametrize("field", ["weights_sha256", "tokenizer_revision", "tokenizer_sha256", "template_sha256", "adapter_sha256", "model_revision", "quantization", "precision"])
def test_same_model_name_cannot_hide_changed_subject(field):
    data = declaration()
    results = measured_results(data)
    other = copy.deepcopy(data)
    replacement = "c" * (40 if field.endswith("revision") else 64)
    other["subject"][field] = replacement
    assert eb.compare_bound(results, other)["state"] == "INVALID"


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "unmeasured", "sample_request", "run_count", "missing_binding", "engine_build", "endpoint", "extra_binding", "cohort_request"])
def test_bound_results_cannot_silently_discard_or_relabel_evidence(mutation):
    data = declaration()
    results = measured_results(data)
    if mutation == "omit":
        results.pop()
    elif mutation == "duplicate":
        results.append(copy.deepcopy(results[0]))
    elif mutation == "unmeasured":
        results[1]["state"] = "FAILED"
    elif mutation == "sample_request":
        results[1]["run_samples"][0]["request_sha256"] = "f" * 64
    elif mutation == "run_count":
        results[1]["runs"] = True
    elif mutation == "missing_binding":
        results[1].pop("binding")
    elif mutation == "engine_build":
        results[1]["binding"]["build_sha256"] = "f" * 64
    elif mutation == "endpoint":
        results[1]["endpoint"] = "http://127.0.0.1:1234"
    elif mutation == "extra_binding":
        results[1]["binding"]["is_attested"] = True
    else:
        results[1]["binding"] = bm.run_binding(data, "sglang", "f" * 64)
        results[1]["run_samples"][0]["request_sha256"] = "f" * 64
    assert eb.compare_bound(results, data)["state"] == "INVALID"


def test_loopback_wire_sampling_and_digest_match_manifest(server_factory, monkeypatch):
    observed = []

    class Capture(MockEngine):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            observed.append(raw)
            payload = json.loads(raw)
            assert {key: payload[key] for key in ("temperature", "top_p", "seed")} == {"temperature": 0, "top_p": 1, "seed": 42}
            assert self.path == "/v1/completions"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"text":"a"}]}\n\ndata: {"choices":[{"text":"b"}]}\n\ndata: [DONE]\n\n')

    endpoint = server_factory(Capture)
    monkeypatch.setenv("VLLM_ENDPOINT", endpoint)
    monkeypatch.setenv("SGLANG_ENDPOINT", endpoint)
    data = declaration(endpoint)
    results, verdict = eb.run_declared(arguments(), data)
    assert verdict["state"] == "MEASURED" and len(observed) == 2
    assert observed[0] == observed[1]
    for result in results:
        assert result["binding"]["request_sha256"] == hashlib.sha256(observed[0]).hexdigest()
        assert result["tok_per_s"]["state"] == "UNAVAILABLE"
        assert result["run_samples"][0]["request_sha256"] == result["binding"]["request_sha256"]


def test_bound_redirect_never_reaches_alternate_target(server_factory, monkeypatch):
    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(307)
            self.send_header("Location", "http://127.0.0.1:1/other")
            self.end_headers()
        def log_message(self, *args):
            pass
    endpoint = server_factory(Redirect)
    monkeypatch.setenv("VLLM_ENDPOINT", endpoint)
    result = eb.run_engine("vllm", "fixture-model", "hi", 1, 6, timeout=1,
                           sampling=declaration()["workload"]["sampling"])
    assert result["state"] == "FAILED"
    assert "redirect" in result["reason"]


def test_cli_without_declaration_makes_no_request_even_with_endpoints(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(eb, "run_engine", lambda *a, **k: pytest.fail("no undeclared request"))
    monkeypatch.setattr(sys, "argv", ["engine_bench.py", "--engines", "vllm"])
    assert eb.main() == 0
    assert json.loads(capsys.readouterr().out)["comparison"]["state"] == "BLOCKED"


def test_cli_bad_manifest_returns_failure_and_receipt_without_echo(tmp_path, monkeypatch, capsys):
    path = tmp_path / "input.json"
    path.write_text('{"do_not_echo":"private-value"}')
    monkeypatch.setattr(sys, "argv", ["engine_bench.py", "--manifest", str(path)])
    assert eb.main() == 2
    output = capsys.readouterr().out
    assert "private-value" not in output and str(path) not in output
    assert json.loads(output)["chain_valid"]


def test_cli_bound_loopback_emits_manifest_result_and_receipt_commitments(tmp_path, server_factory, monkeypatch, capsys):
    endpoint = server_factory()
    data = declaration(endpoint)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("VLLM_ENDPOINT", endpoint)
    monkeypatch.setenv("SGLANG_ENDPOINT", endpoint)
    monkeypatch.setattr(sys, "argv", ["engine_bench.py", "--engines", "vllm", "sglang",
                                     "--model", "fixture-model", "--prompt", "hi", "--runs", "1",
                                     "--max-tokens", "6", "--timeout-seconds", "2", "--manifest", str(path)])
    assert eb.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["comparison"]["state"] == "MEASURED"
    assert output["receipt"]["run"]["manifest_sha256"] == bm.digest(data)
    for result in output["results"]:
        assert output["receipt"]["run"]["result_sha256"][result["engine"]] == bm.digest(result)
    assert output["chain_valid"]


def test_declared_failures_keep_every_engine_and_block_comparison(monkeypatch):
    for variable in ("VLLM_ENDPOINT", "SGLANG_ENDPOINT"):
        monkeypatch.setenv(variable, "http://127.0.0.1:9876")
    called = []
    def fail(name, *args, **kwargs):
        called.append((name, kwargs["endpoint_override"]))
        return {"state": "FAILED", "engine": name, "reason": "fixture protocol failure"}
    monkeypatch.setattr(eb, "run_engine", fail)
    results, verdict = eb.run_declared(arguments(), declaration())
    assert called == [("vllm", "http://127.0.0.1:9876"), ("sglang", "http://127.0.0.1:9876")]
    assert [r["engine"] for r in results] == ["vllm", "sglang"]
    assert all(r["state"] == "FAILED" and "binding" in r for r in results)
    assert verdict["state"] == "BLOCKED"


def test_preflight_endpoint_is_frozen_for_entire_cohort(monkeypatch):
    for variable in ("VLLM_ENDPOINT", "SGLANG_ENDPOINT"):
        monkeypatch.setenv(variable, "http://127.0.0.1:9876")
    observed = []
    def fail(name, *args, **kwargs):
        observed.append(kwargs["endpoint_override"])
        monkeypatch.setenv("SGLANG_ENDPOINT", "http://127.0.0.1:1")
        return {"state": "FAILED", "engine": name, "reason": "fixture"}
    monkeypatch.setattr(eb, "run_engine", fail)
    eb.run_declared(arguments(), declaration())
    assert observed == ["http://127.0.0.1:9876"] * 2
