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
