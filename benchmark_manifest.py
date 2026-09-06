# SPDX-License-Identifier: Apache-2.0
"""Bounded, non-executing benchmark declarations and exact request commitments.

Declarations are supplied by an operator. Validating and hashing them does NOT
attest what a remote server loaded, measure model quality, or admit a release.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA = "szl.engine-comparison/v1"
BOUNDARY = "OPERATOR_DECLARED_NOT_RUNTIME_ATTESTED"
MAX_MANIFEST_BYTES = 65536
ENGINE_NAMES = frozenset({"vllm", "sglang", "llamacpp", "mlx", "tgi", "transformers"})


class ManifestError(ValueError):
    """A declaration is incomplete, ambiguous, or incomparable."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _keys(value: Any, expected: set[str], field: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ManifestError(f"{field}: missing or unrecognized fields")
    return value


def _hex(value: Any, size: int, field: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % size, value) is None:
        raise ManifestError(f"{field}: expected exact lowercase hexadecimal digest")


def _identifier(value: Any, field: str) -> None:
    if (not isinstance(value, str) or not 1 <= len(value) <= 200
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*", value) is None):
        raise ManifestError(f"{field}: expected bounded ASCII identifier")


def _integer(value: Any, minimum: int, maximum: int, field: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ManifestError(f"{field}: integer outside allowed range")


def _number(value: Any, minimum: float, maximum: float, field: str) -> None:
    if (type(value) not in (int, float) or not minimum <= value <= maximum
            or not math.isfinite(value)):
        raise ManifestError(f"{field}: finite number outside allowed range")


def sampling_parameters(value: Any) -> dict:
    value = _keys(value, {"temperature", "top_p", "seed"}, "sampling")
    _number(value["temperature"], 0, 2, "temperature")
    _number(value["top_p"], 0, 1, "top_p")
    if value["top_p"] == 0:
        raise ManifestError("top_p: must be positive")
    _integer(value["seed"], 0, 2147483647, "seed")
    return dict(value)


def validate_manifest(value: Any) -> dict:
    """Validate a closed declaration; return a detached canonical JSON snapshot."""
    value = _keys(value, {"schema", "claim_boundary", "subject", "workload", "engines"}, "manifest")
    if value["schema"] != SCHEMA or value["claim_boundary"] != BOUNDARY:
        raise ManifestError("manifest: unsupported schema or claim boundary")
    subject = _keys(value["subject"], {
        "model_id", "model_revision", "weights_sha256", "tokenizer_revision",
        "tokenizer_sha256", "template_sha256", "adapter_sha256", "quantization", "precision",
    }, "subject")
    for field in ("model_id", "quantization", "precision"):
        _identifier(subject[field], field)
    for field in ("model_revision", "tokenizer_revision"):
        _hex(subject[field], 40, field)
    for field in ("weights_sha256", "tokenizer_sha256", "template_sha256"):
        _hex(subject[field], 64, field)
    if subject["adapter_sha256"] is not None:
        _hex(subject["adapter_sha256"], 64, "adapter_sha256")
    workload = _keys(value["workload"], {
        "suite_sha256", "prompt_sha256", "runs", "max_tokens", "timeout_seconds",
        "sampling", "concurrency", "cache_state",
    }, "workload")
    for field in ("suite_sha256", "prompt_sha256"):
        _hex(workload[field], 64, field)
    _integer(workload["runs"], 1, 1000, "runs")
    _integer(workload["max_tokens"], 1, 65536, "max_tokens")
    _number(workload["timeout_seconds"], 0.01, 600, "timeout_seconds")
    sampling_parameters(workload["sampling"])
    if type(workload["concurrency"]) is not int or workload["concurrency"] != 1:
        raise ManifestError("concurrency: this client measures serial requests only")
    if workload["cache_state"] != "UNCONTROLLED":
        raise ManifestError("cache_state: this client does not establish server cache state")
    engines = value["engines"]
    if (not isinstance(engines, dict) or not 2 <= len(engines) <= len(ENGINE_NAMES)
            or not set(engines).issubset(ENGINE_NAMES)):
        raise ManifestError("engines: expected two or more distinct supported engines")
    for engine in engines.values():
        _keys(engine, {"build_sha256", "configuration_sha256", "hardware_sha256", "endpoint_sha256"}, "engine")
        for field, field_value in engine.items():
            _hex(field_value, 64, field)
    if len({engine["hardware_sha256"] for engine in engines.values()}) != 1:
        raise ManifestError("hardware_sha256: different hardware is not an engine-only comparison")
    return json.loads(canonical(value))


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError("manifest JSON contains duplicate object fields")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ManifestError("manifest JSON contains a non-finite constant")


def load_manifest(path: str | Path) -> dict:
    try:
        with Path(path).open("rb") as source:
            payload = source.read(MAX_MANIFEST_BYTES + 1)
        if len(payload) > MAX_MANIFEST_BYTES:
            raise ManifestError("manifest exceeds byte budget")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
        return validate_manifest(value)
    except ManifestError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError, OverflowError):
        # Never echo an operator-controlled filename, JSON value, or exception.
        raise ManifestError("manifest is unreadable or invalid JSON") from None


def request_payload(model: str, prompt: str, max_tokens: int, sampling: dict | None = None) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "stream": True, "stream_options": {"include_usage": True}}
    if sampling is not None:
        payload.update(sampling_parameters(sampling))
    return payload


def endpoint_digest(endpoint: str) -> str:
    # The caller validates the URL. Only trailing path separators are normalized.
    return hashlib.sha256(endpoint.rstrip("/").encode("utf-8")).hexdigest()


def preflight(value: dict, *, model: str, prompt: str, runs: int, max_tokens: int,
              timeout: float, endpoints: dict[str, str]) -> dict:
    manifest = validate_manifest(value)
    workload = manifest["workload"]
    if set(endpoints) != set(manifest["engines"]):
        raise ManifestError("selected engines must exactly match the manifest")
    if (model != manifest["subject"]["model_id"] or runs != workload["runs"]
            or max_tokens != workload["max_tokens"] or timeout != workload["timeout_seconds"]
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != workload["prompt_sha256"]):
        raise ManifestError("requested model or workload differs from its declaration")
    for name, endpoint in endpoints.items():
        if endpoint_digest(endpoint) != manifest["engines"][name]["endpoint_sha256"]:
            raise ManifestError("configured endpoint differs from its declaration")
    return manifest


def run_binding(manifest: dict, name: str, request_sha256: str) -> dict:
    """Commit the declaration and wire request, not an assertion about remote weights."""
    _hex(request_sha256, 64, "request_sha256")
    engine = manifest["engines"][name]
    return {
        "schema": "szl.engine-run-binding/v1", "claim_boundary": BOUNDARY,
        "engine": name, "manifest_sha256": digest(manifest),
        "subject_sha256": digest(manifest["subject"]),
        "workload_sha256": digest(manifest["workload"]),
        "request_sha256": request_sha256, **engine,
    }


def verify_result_bindings(manifest: dict, results: list[dict]) -> None:
    """Refuse partial cohorts, changed declarations, or changed request bodies."""
    names = [result.get("engine") for result in results]
    if (any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or set(names) != set(manifest["engines"])):
        raise ManifestError("comparison cohort is incomplete, duplicated, or unexpected")
    requests = set()
    for result in results:
        name = result["engine"]
        if result.get("state") != "MEASURED":
            raise ManifestError("every declared engine must produce measurements")
        binding = result.get("binding")
        if not isinstance(binding, dict):
            raise ManifestError("measured result lacks an exact request binding")
        request_hash = binding.get("request_sha256")
        _hex(request_hash, 64, "request_sha256")
        if binding != run_binding(manifest, name, request_hash):
            raise ManifestError("result binding differs from the declared experiment")
        if result.get("model") != manifest["subject"]["model_id"]:
            raise ManifestError("result model differs from the declaration")
        if type(result.get("runs")) is not int or result["runs"] != manifest["workload"]["runs"]:
            raise ManifestError("result run count differs from the declaration")
        endpoint = result.get("endpoint")
        if not isinstance(endpoint, str) or endpoint_digest(endpoint) != binding["endpoint_sha256"]:
            raise ManifestError("result endpoint differs from its binding")
        samples = result.get("run_samples")
        if not isinstance(samples, list) or len(samples) != result["runs"]:
            raise ManifestError("bound run samples are incomplete")
        if any(not isinstance(sample, dict) or sample.get("request_sha256") != request_hash
               for sample in samples):
            raise ManifestError("a wire request differs from the bound request")
        requests.add(request_hash)
    if len(requests) != 1:
        raise ManifestError("wire requests differ across engines")
