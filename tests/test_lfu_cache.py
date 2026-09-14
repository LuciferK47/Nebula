"""Tests for LFU Cache."""
from __future__ import annotations
import pytest

from memtier_moe.core.types import ExpertId, MemoryTier, TransferState
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.memory.expert_metadata import ExpertMetadata
from memtier_moe.cache.eviction import LFUEviction


# ── Unit tests for decay math and eviction (no mocks needed) ──────────


def test_decay_math_correctness():
    """Verify the exponential decay formula produces expected values."""
    meta = ExpertMetadata(expert_id=(0, 0), size_bytes=100)
    meta.decayed_frequency = 1.0
    meta.last_access_token = 0

    # After 1 half-life: 1.0 * 0.5^1 + 1.0 = 1.5
    meta.update_frequency(100, half_life=100)
    assert abs(meta.decayed_frequency - 1.5) < 1e-5

    # After another half-life: 1.5 * 0.5^1 + 1.0 = 1.75
    meta.update_frequency(200, half_life=100)
    assert abs(meta.decayed_frequency - 1.75) < 1e-5


def test_decay_frequency_pure():
    """decay_frequency() should not mutate state."""
    meta = ExpertMetadata(expert_id=(0, 0), size_bytes=100)
    meta.decayed_frequency = 4.0
    meta.last_access_token = 0

    result = meta.decay_frequency(current_token=100, half_life=100)
    assert abs(result - 2.0) < 1e-5          # 4.0 * 0.5^1
    assert meta.decayed_frequency == 4.0      # unchanged
    assert meta.last_access_token == 0        # unchanged


def test_eviction_ordering():
    """LFUEviction should evict lowest-frequency experts first."""
    eviction = LFUEviction(half_life=100)
    candidates = []
    for i in range(5):
        meta = ExpertMetadata(expert_id=(0, i), size_bytes=100)
        meta.decayed_frequency = float(i)
        meta.last_access_token = 0
        candidates.append(meta)

    victims = eviction.select_victims(candidates, needed_bytes=250, current_token=0)
    assert victims == [(0, 0), (0, 1), (0, 2)]


def test_eviction_frees_enough_space():
    """Eviction should stop once needed_bytes is met."""
    eviction = LFUEviction(half_life=100)
    sizes = [300, 400, 500]
    candidates = []
    for i, size in enumerate(sizes):
        meta = ExpertMetadata(expert_id=(0, i), size_bytes=size)
        meta.decayed_frequency = float(i)
        candidates.append(meta)

    # Need 1000, all three total 1200 — must evict all three
    victims = eviction.select_victims(candidates, needed_bytes=1000, current_token=0)
    assert victims == [(0, 0), (0, 1), (0, 2)]

    # Need 600 — first two (300+400=700) suffice
    victims2 = eviction.select_victims(candidates, needed_bytes=600, current_token=0)
    assert victims2 == [(0, 0), (0, 1)]


def test_lookup_does_not_double_count_with_caller_tier_recording():
    """lookup() must not record hit/miss metrics itself.

    Both real callers (runtime/engine.py and the weight_transfer branch of
    runtime/tiered_model.py) always follow lookup() with a per-expert
    tier-specific record_*() call, because a miss doesn't know which tier
    it will resolve to until the caller checks. lookup() used to *also*
    call record_hit()/record_miss() directly, silently doubling
    cache_hits/cache_misses relative to the real number of lookups (2494
    vs 1247 actual in one committed S2 scenario run) and, because only the
    HBM side happened to be doubled symmetrically, understating
    dram_hit_rate/cxl_hit_rate by exactly 2x without touching hit_rate at
    all — the kind of bug that hides in a metric nobody happens to chart.
    """
    from memtier_moe.cache.lfu_cache import LFUExpertCache, CachedExpert

    class MinimalTM:
        def __init__(self):
            self.metadata = {}
        def get_metadata(self, eid):
            return self.metadata[eid]
        def get_pool(self, tier):
            raise NotImplementedError

    tm = MinimalTM()
    tm.metadata[(0, 0)] = ExpertMetadata((0, 0), size_bytes=100)  # will be a "hit"
    tm.metadata[(0, 1)] = ExpertMetadata((0, 1), size_bytes=100)  # will be a "miss"

    config = MemTierConfig(hbm_cache_budget_bytes=10000)
    metrics = MetricsTracker()
    cache = LFUExpertCache(tm, config, metrics)
    # Make (0, 0) resident without the full transfer machinery insert() needs.
    cache._cache[(0, 0)] = CachedExpert(expert_id=(0, 0), metadata=tm.metadata[(0, 0)], last_used_token=0)

    hits, misses = cache.lookup([(0, 0), (0, 1)])
    assert hits == [(0, 0)] and misses == [(0, 1)]

    # lookup() alone must be metrics-neutral.
    assert metrics.counters["cache_hits"] == 0
    assert metrics.counters["cache_misses"] == 0

    # Now perform exactly what both real call sites do: one tier-specific
    # record per hit and per miss.
    for eid in hits:
        metrics.record_hbm_hit()
    for eid in misses:
        metrics.record_dram_hit()

    assert metrics.counters["cache_hits"] + metrics.counters["cache_misses"] == 2
    assert metrics.counters["hbm_hits"] == 1
    assert metrics.counters["dram_hits"] == 1


def test_warm_start_preserves_frequencies():
    """warm_start should set decayed_frequency on metadata."""
    from memtier_moe.cache.lfu_cache import LFUExpertCache

    config = MemTierConfig(hbm_cache_budget_bytes=10000, frequency_decay_half_life=100)
    metrics = MetricsTracker()

    # Minimal mock tier manager — warm_start only calls get_metadata()
    class MinimalTM:
        def __init__(self):
            self.metadata = {}
        def get_metadata(self, eid):
            return self.metadata[eid]
        def get_pool(self, tier):
            raise NotImplementedError

    tm = MinimalTM()
    tm.metadata[(0, 0)] = ExpertMetadata((0, 0), size_bytes=100)
    tm.metadata[(0, 1)] = ExpertMetadata((0, 1), size_bytes=100)

    cache = LFUExpertCache(tm, config, metrics)
    cache.warm_start({(0, 0): 5.0, (0, 1): 10.0})

    assert tm.metadata[(0, 0)].decayed_frequency == 5.0
    assert tm.metadata[(0, 1)].decayed_frequency == 10.0
