"""Regression test: cxl_emulation_mode must only change injected *timing*
cost, never *placement* decisions.

This is the contract the CXL sensitivity sweep (planned scenario S3 —
sweeping cxl_bandwidth_gbps and cxl_emulation_mode to check whether the
architectural conclusion survives a wide swing in CXL assumptions) leans
on. If emulation_mode leaked into control flow — e.g. a "disabled" run
skipping an eviction it should have made, or landing an expert in a
different tier — then a sweep across emulation modes would actually be
measuring a confound, not fidelity sensitivity, and the whole point of
running the sweep (rather than a cycle-accurate simulator) would be
undermined.

Reuses the CXL-reachability workload from test_memory_pressure.py
(same skewed routing trace, same DRAM-pressure config) so this exercises
a realistic multi-eviction, multi-tier scenario rather than a toy case.
"""
from __future__ import annotations

import numpy as np
import pytest

from memtier_moe.core.config import MemTierConfig
from memtier_moe.runtime.engine import InferenceEngine


def _make_skewed_routing(num_layers=6, num_experts=20, top_k=2, num_tokens=80, seed=1):
    rng = np.random.RandomState(seed)
    p = np.array([1.0 / (i + 1) for i in range(num_experts)])
    p /= p.sum()
    return [
        [rng.choice(num_experts, size=top_k, replace=False, p=p).tolist() for _ in range(num_layers)]
        for _ in range(num_tokens)
    ]


def _run_with_mode(mode: str):
    num_layers, num_experts = 6, 20
    expert_sizes = {(l, e): 1000 for l in range(num_layers) for e in range(num_experts)}
    total_bytes = len(expert_sizes) * 1000

    config = MemTierConfig(
        hbm_cache_budget_bytes=8_000,
        host_dram_bytes=total_bytes + 4_000,
        cxl_memory_bytes=1_000_000,
        placement_warm_threshold=1.0,
        dram_reserve_fraction=0.10,
        cxl_emulation_mode=mode,
    )
    engine = InferenceEngine(config)
    engine.setup_from_profile(expert_sizes, {k: 0.1 for k in expert_sizes})

    routing = _make_skewed_routing(num_layers=num_layers, num_experts=num_experts)
    for t, token_layers in enumerate(routing):
        engine.run_token(t, token_layers)

    return engine


@pytest.mark.parametrize("mode", ["latency_only", "disabled"])
def test_emulation_mode_does_not_change_placement_or_hit_rate(mode):
    baseline = _run_with_mode("full")
    variant = _run_with_mode(mode)

    base_report = baseline.metrics.report()
    var_report = variant.metrics.report()

    assert var_report["hit_rate"] == base_report["hit_rate"]
    assert var_report["cache_hits"] == base_report["cache_hits"]
    assert var_report["cache_misses"] == base_report["cache_misses"]
    assert var_report["evictions"] == base_report["evictions"]
    assert var_report["cxl_hits"] == base_report["cxl_hits"]

    base_summary = baseline.tier_manager.summary()
    var_summary = variant.tier_manager.summary()
    for tier in ("hbm", "dram", "cxl"):
        assert var_summary[tier]["expert_count"] == base_summary[tier]["expert_count"], (
            f"cxl_emulation_mode={mode!r} changed {tier} tier occupancy — "
            f"emulation fidelity must never affect placement."
        )

    # Sanity: the CXL tier was actually exercised, or this test would pass
    # vacuously (identical zero placement in every mode).
    assert base_summary["cxl"]["expert_count"] > 0


def test_disabled_mode_is_not_slower_than_full():
    """Weak timing sanity check: 'disabled' must not inject MORE cost than
    'full'. Not a strict ordering assertion (CI timing is noisy) — just
    guards against the modes being wired backwards."""
    import time

    t0 = time.perf_counter()
    _run_with_mode("full")
    full_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    _run_with_mode("disabled")
    disabled_elapsed = time.perf_counter() - t0

    # Generous slack: this only needs to catch a wiring inversion, not
    # assert a precise speedup.
    assert disabled_elapsed <= full_elapsed * 1.5
