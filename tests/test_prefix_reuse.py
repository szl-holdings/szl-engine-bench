"""Offline checks for a declared prefix-reuse receipt. No engine and no network."""
from __future__ import annotations

from szl_engine_bench.prefix_reuse import prefix_reuse_receipt

PREFIX = "ab" * 32


def declared(**overrides):
    payload = {
        "prefix_sha256": PREFIX,
        "reused_tokens": 4,
        "computed_tokens": 3,
        "total_tokens": 7,
        "fuel_cap_tokens": 16,
        "description_cap_bytes": 64,
        "prompt": "prefix",
    }
    payload.update(overrides)
    return prefix_reuse_receipt(**payload)


def test_consistent_declaration_is_not_a_measurement():
    body = declared()
    assert body["state"] == "DECLARED"
    assert body["measurement"] == "DECLARED_SOURCE"
    assert body["runtime_attestation"] == "NOT_ATTESTED"
    assert body["receipt_minted"] is False
    assert body["production_authorization"] is False
    assert body["signer"] == "ABSENT"
    assert body["prefix_bound_to_prompt"] is False
    assert body["formula_correspondence"] == "UNVERIFIED"
    assert body["lambda"] == "Conjecture 1"
    assert body["winner"] is None
    assert body["reused_tokens"] + body["computed_tokens"] == body["total_tokens"]
    assert body["description_bytes"] == len("prefix".encode("utf-8"))
    assert body["chain_valid"] is True
    assert body["receipt"]["signature"] == "UNSIGNED_HONEST"
    assert body["state"] != "MEASURED"


def test_token_conservation_is_required():
    body = declared(total_tokens=8)
    assert body["state"] == "INVALID"
    assert "total_tokens" in body["reason"]


def test_fuel_cap_rejects_the_declared_total():
    body = declared(fuel_cap_tokens=6)
    assert body["state"] == "INVALID"
    assert "fuel cap" in body["reason"]


def test_utf8_description_cap_rejects_a_longer_prompt():
    body = declared(prompt="piñata", description_cap_bytes=6)
    assert len("piñata".encode("utf-8")) == 7
    assert body["state"] == "INVALID"
    assert "description cap" in body["reason"]


def test_prefix_must_be_lowercase_hex():
    assert declared(prefix_sha256=PREFIX.upper())["state"] == "INVALID"
    assert declared(prefix_sha256="g" * 64)["state"] == "INVALID"
    assert declared(prefix_sha256=PREFIX[:-1])["state"] == "INVALID"


def test_counts_reject_bools_and_floats():
    assert declared(reused_tokens=True)["state"] == "INVALID"
    assert declared(computed_tokens=1.5)["state"] == "INVALID"
    assert declared(fuel_cap_tokens=0)["state"] == "INVALID"


def test_samples_stay_a_list_and_do_not_select_a_winner():
    body = declared(samples=[
        {"reused_tokens": 1, "computed_tokens": 1, "total_tokens": 2},
        {"reused_tokens": 5, "computed_tokens": 1, "total_tokens": 6},
    ])
    assert body["state"] == "DECLARED"
    assert body["samples"] == [
        {"reused_tokens": 1, "computed_tokens": 1, "total_tokens": 2},
        {"reused_tokens": 5, "computed_tokens": 1, "total_tokens": 6},
    ]
    assert "best" not in body
    assert body["winner"] is None


def test_a_sample_that_names_a_winner_is_invalid():
    body = declared(samples=[{
        "reused_tokens": 1,
        "computed_tokens": 1,
        "total_tokens": 2,
        "winner": "sglang",
    }])
    assert body["state"] == "INVALID"
