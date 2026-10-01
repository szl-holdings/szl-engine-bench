"""szl-engine-bench — honest inference-engine benchmark harness.

Public modules:
  szl_engine_bench.engine_bench       — harness, stats, fairness gates, receipts
  szl_engine_bench.benchmark_manifest — manifest and endpoint/file digests

Namespaced in 0.3.2: the former top-level modules ``engine_bench`` and
``benchmark_manifest`` now live under this package so the wheel no longer
claims generic import names. Run the CLI with ``szl-engine-bench`` or
``python -m szl_engine_bench``.
"""
from . import benchmark_manifest, engine_bench  # noqa: F401

__all__ = ["benchmark_manifest", "engine_bench"]
