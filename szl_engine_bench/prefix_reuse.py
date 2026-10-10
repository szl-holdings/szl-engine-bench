"""Declared prefix-reuse receipt.

A caller states the hash of a reused prefix, how many tokens were reused,
how many were computed, and the predeclared caps. This module checks that
declaration. It does not contact vLLM, SGLang, or any other engine, and a
well-formed result is not evidence that a KV page or radix node was reused.
"""
from __future__ import annotations

from collections.abc import Mapping

from .engine_bench import ReceiptChain, _sha256_canonical

SCHEMA = "szl.prefix-reuse-receipt/v1"


def _whole(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _invalid(reason: str) -> dict:
    return {
        "schema": SCHEMA,
        "state": "INVALID",
        "reason": reason,
        "measurement": "DECLARED_SOURCE",
        "runtime_attestation": "NOT_ATTESTED",
        "receipt_minted": False,
        "production_authorization": False,
        "signer": "ABSENT",
        "formula_correspondence": "UNVERIFIED",
        "lambda": "Conjecture 1",
    }


def _sample(item, fuel_cap: int) -> dict | None:
    if not isinstance(item, Mapping):
        return None
    if "winner" in item:
        return None
    reused = _whole(item.get("reused_tokens"))
    computed = _whole(item.get("computed_tokens"))
    total = _whole(item.get("total_tokens"))
    if None in (reused, computed, total):
        return None
    if reused + computed != total or total > fuel_cap:
        return None
    return {
        "reused_tokens": reused,
        "computed_tokens": computed,
        "total_tokens": total,
    }


def prefix_reuse_receipt(
    *,
    prefix_sha256: str,
    reused_tokens: int,
    computed_tokens: int,
    total_tokens: int,
    fuel_cap_tokens: int,
    description_cap_bytes: int,
    prompt: str,
    samples: list | None = None,
) -> dict:
    """Check one declared prefix-reuse record and keep extra samples unranked.

    ``description_cap_bytes`` is a UTF-8 byte cap. It is not a Kolmogorov
    complexity. ``fuel_cap_tokens`` is a declared token cap. It is not a
    proof of the halting problem. Formula ids are registry labels.
    """
    if (
        not isinstance(prefix_sha256, str)
        or len(prefix_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in prefix_sha256)
    ):
        return _invalid("prefix_sha256 must be 64 lowercase hex characters")

    reused = _whole(reused_tokens)
    computed = _whole(computed_tokens)
    total = _whole(total_tokens)
    fuel = _whole(fuel_cap_tokens)
    cap = _whole(description_cap_bytes)
    if None in (reused, computed, total, fuel, cap) or fuel == 0 or cap == 0:
        return _invalid("token counts and caps must be non-negative integers, and caps must be positive")
    if not isinstance(prompt, str):
        return _invalid("prompt must be a string")
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > cap:
        return _invalid("prompt exceeds the declared UTF-8 description cap")
    if reused + computed != total:
        return _invalid("reused_tokens + computed_tokens must equal total_tokens")
    if total > fuel:
        return _invalid("total_tokens exceeds the declared fuel cap")

    recorded = []
    if samples is not None:
        if not isinstance(samples, list):
            return _invalid("samples must be a list")
        for item in samples:
            checked = _sample(item, fuel)
            if checked is None:
                return _invalid("every sample must conserve its own token counts and stay within the fuel cap")
            recorded.append(checked)

    declaration = {
        "schema": SCHEMA,
        "state": "DECLARED",
        "measurement": "DECLARED_SOURCE",
        "runtime_attestation": "NOT_ATTESTED",
        "receipt_minted": False,
        "production_authorization": False,
        "signer": "ABSENT",
        "formula_ids": ["F18", "F19", "F4"],
        "formula_correspondence": "UNVERIFIED",
        "lambda": "Conjecture 1",
        "description_unit": "utf8_bytes",
        "description_bytes": prompt_bytes,
        "description_cap_bytes": cap,
        "fuel_cap_tokens": fuel,
        "prefix_sha256": prefix_sha256,
        "prefix_bound_to_prompt": False,
        "reused_tokens": reused,
        "computed_tokens": computed,
        "total_tokens": total,
        "samples": recorded,
        "winner": None,
        "note": (
            "UTF-8 byte cap and token cap are declared bounds. "
            "They are not F18 or F19 proofs. Samples stay a list."
        ),
    }
    declaration["declaration_sha256"] = _sha256_canonical(declaration)
    chain = ReceiptChain()
    receipt = chain.emit({"type": "prefix_reuse", "declaration_sha256": declaration["declaration_sha256"]})
    declaration["receipt"] = receipt
    declaration["chain_valid"] = chain.verify()
    return declaration
