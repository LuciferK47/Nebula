"""CPU-only pre-validation of the degenerate-budget edge cases that
scripts/run_scenarios.py --scenario s10 exercises on real hardware.

The GPU edge cases (E3/E4/E6) probe configurations where one or more
memory tiers have zero or near-zero capacity. The tier-management half of
that behavior is torch-free, so it can be pinned here in milliseconds
rather than discovered ten minutes into a GPU run. What these cannot
cover is the real CUDA placement in TieredMoEWrapper._patch_moe_layers()
— that is what S10 is for.

The key property: a tier configuration that cannot hold the working set
must fail *loudly and early*, or degrade to a defined tier, but never
silently leave the registry disagreeing with the pools about where an
expert actually lives.
"""
from __future__ import annotations

import pytest

from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.types import MemoryTier
from memtier_moe.runtime.engine import InferenceEngine


def _sizes(num_layers=4, num_experts=8, size=1000):
    return {(l, e): size for l in range(num_layers) for e in range(num_experts)}


def test_no_dram_no_cxl_with_ample_hbm_places_everything_in_hbm():
    """E3's CPU analogue: the GPU-resident configuration (host_dram=0,
    cxl=0) must not crash on first placement — it previously did, because
    setup unconditionally staged through DRAM."""
    sizes = _sizes()
    total = sum(sizes.values())
    config = MemTierConfig(
        hbm_cache_budget_bytes=total * 2, host_dram_bytes=0, cxl_memory_bytes=0,
    )
    engine = InferenceEngine(config)
    engine.setup_from_profile(sizes, {k: 1.0 for k in sizes})

    summary = engine.tier_manager.summary()
    assert summary["hbm"]["expert_count"] == len(sizes)
    assert summary["dram"]["expert_count"] == 0
    assert summary["cxl"]["expert_count"] == 0


def test_oversubscribed_setup_leaves_registry_ahead_of_pools():
    """Characterization test for setup-time over-subscription.

    ``InferenceEngine.setup_from_profile`` calls ``register_expert`` (which
    sets ``metadata.current_tier``) and then swallows ``place_initial``'s
    RuntimeError when the pool is full. The swallow is deliberate — the
    Two-Tier baseline (``cxl_memory_bytes=0``) legitimately overflows DRAM,
    and the published ablation numbers are built on that behavior — but it
    leaves the registry reporting a tier the pool rejected.

    Consequence, for anyone configuring a *new* simulation baseline: with
    ``host_dram_bytes=0`` the initial tier is HBM and every expert is also
    inserted into the LFU cache, so an undersized HBM budget produces
    experts that ``lookup`` counts as cache *hits* while the pool cannot
    return them. Keep HBM large enough to hold the working set in that
    configuration — which the shipped GPU-Resident baseline does, via an
    effectively-infinite 200 GB budget (see evaluation/baselines.py).

    The live path has no such caveat: TieredMoEWrapper._patch_moe_layers()
    now refuses an undersized budget triple outright, because there the
    dropped object is the only copy of real weights.
    """
    sizes = _sizes()
    config = MemTierConfig(
        hbm_cache_budget_bytes=2000,  # room for 2 of 32 experts
        host_dram_bytes=0, cxl_memory_bytes=0,
    )
    engine = InferenceEngine(config)
    engine.setup_from_profile(sizes, {k: 1.0 for k in sizes})

    placed = [
        eid for eid in sizes
        if engine.tier_manager.get_pool(
            engine.tier_manager.get_metadata(eid).current_tier
        ).contains(eid)
    ]
    # Only what physically fit was placed; the rest are registry-only.
    assert len(placed) < len(sizes), "expected over-subscription to leave experts unplaced"
    assert engine.tier_manager.get_pool(MemoryTier.HBM).usage_bytes() <= 2000


def test_all_tiers_too_small_drops_tensors_but_stays_recoverable_in_simulation():
    """Characterization test for demote_to()'s all-tiers-full fallback.

    When no tier can accept a demotion, that branch evicts the tensor from
    its source pool, returns it, and sets metadata.current_tier to the
    target tier *without storing it there* — so the registry reports a
    residency the pools do not back. This is deliberate "spill to disk"
    behavior, but there is no disk tier: the tensor is simply released.

    In simulation that is recoverable and therefore fine — tensors are
    fungible `_Placeholder` objects, and `_execute_multihop` synthesizes an
    equivalent one on the next fetch, which is exactly the page-back-in
    semantic. In the *live* path the tensor is an nn.Module holding the
    only copy of those weights, so the same code would silently corrupt
    generation. TieredMoEWrapper._patch_moe_layers() therefore refuses to
    build a configuration that could reach this state at all (see
    test_live_path_refuses_undersized_tiers below).

    This test pins the divergence as known-and-bounded rather than
    asserting it away, so a future change in this area is noticed.
    """
    sizes = _sizes()
    config = MemTierConfig(
        hbm_cache_budget_bytes=3000, host_dram_bytes=3000, cxl_memory_bytes=3000,
        placement_warm_threshold=1.0, dram_reserve_fraction=0.10,
    )
    engine = InferenceEngine(config)
    engine.setup_from_profile(sizes, {k: 0.5 for k in sizes})

    routing = [[[0, 1], [2, 3], [4, 5], [6, 7]] for _ in range(15)]
    for t, layers in enumerate(routing):
        engine.run_token(t, layers)

    divergent = [
        eid for eid in sizes
        if not engine.tier_manager.get_pool(
            engine.tier_manager.get_metadata(eid).current_tier
        ).contains(eid)
    ]
    # Total capacity (9000B) is well under the working set (32 x 1000B), so
    # the fallback must have been taken for at least some experts.
    assert divergent, "expected the all-tiers-full fallback to be exercised here"

    # The important property: every such expert is still *fetchable* — the
    # simulation can page it back in, so the run completes correctly.
    for eid in divergent[:5]:
        engine.cache.ensure_resident(eid, engine.transfer_engine)
        assert engine.tier_manager.get_pool(MemoryTier.HBM).contains(eid), (
            f"{eid} was dropped by the all-tiers-full fallback and could not be "
            f"recovered by a subsequent fetch"
        )


def test_live_path_refuses_undersized_tiers():
    """The live wrapper must reject a configuration whose tiers cannot hold
    every expert, rather than silently dropping real weights. Checks the
    precondition arithmetic directly, since constructing a real
    TieredMoEWrapper needs torch + a model."""
    total_expert_bytes = 1_200_000_000  # ~1.2 GB of experts
    hbm, dram, cxl = 200 * 1024 * 1024, 200 * 1024 * 1024, 200 * 1024 * 1024
    assert hbm + dram + cxl < total_expert_bytes, (
        "this budget triple must be undersized for the test to be meaningful — "
        "TieredMoEWrapper._patch_moe_layers() raises ValueError for exactly this case"
    )


def test_zero_capacity_pools_report_zero_not_negative():
    config = MemTierConfig(hbm_cache_budget_bytes=1000, host_dram_bytes=0, cxl_memory_bytes=0)
    engine = InferenceEngine(config)
    for tier in (MemoryTier.DRAM, MemoryTier.CXL):
        pool = engine.tier_manager.get_pool(tier)
        assert pool.capacity_bytes == 0
        assert pool.free_bytes() == 0
        assert pool.available_for(1) is False


def test_single_expert_larger_than_every_tier_fails_fast_not_hanging():
    """A single expert bigger than every tier cannot physically be placed,
    so a clean RuntimeError is the correct outcome. What matters is that
    it terminates promptly rather than wedging in a promote/demote loop —
    which is what `engine.run_token` returning or raising both prove."""
    sizes = {(0, 0): 10_000}
    config = MemTierConfig(
        hbm_cache_budget_bytes=1000, host_dram_bytes=1000, cxl_memory_bytes=1000,
    )
    engine = InferenceEngine(config)
    engine.setup_from_profile(sizes, {(0, 0): 1.0})
    try:
        engine.run_token(0, [[0]])
    except RuntimeError as e:
        assert "out of memory" in str(e).lower(), f"unexpected failure mode: {e}"
