"""Tests for M2 features: CXL pool, 3-tier placement, multi-hop transfers."""
from __future__ import annotations
import time
import pytest

from memtier_moe.core.types import MemoryTier, TransferState
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.memory.pool import CXLPool
from memtier_moe.memory.tier_manager import TierManager, _next_higher_tier, _next_lower_tier
from memtier_moe.memory.transfer_engine import TransferEngine, DoubleBuffer
from memtier_moe.memory.expert_metadata import ExpertMetadata
from memtier_moe.cache.placement import PlacementPolicy
from memtier_moe.cache.eviction import SizeAwareLFUEviction


# ── Reusable fake tensor ──────────────────────────────────────────────


class FakeTensor:
    def __init__(self, size_bytes: int):
        self._size = size_bytes

    def element_size(self):
        return 1

    def numel(self):
        return self._size

    def cpu(self):
        return self

    def cuda(self):
        return self

    def pin_memory(self):
        return self


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return MemTierConfig(
        hbm_cache_budget_bytes=5000,
        host_dram_bytes=50000,
        cxl_memory_bytes=200000,
        cxl_latency_ns=50,       # low for fast tests
        cxl_bandwidth_gbps=100.0,  # high so rate-limiter doesn't block
        max_inflight_transfers=4,
    )


@pytest.fixture
def metrics():
    return MetricsTracker()


@pytest.fixture
def tm(config, metrics):
    return TierManager(config, metrics)


def _place(tm, eid, size, tier):
    """Register + place a fake tensor in the given tier."""
    tm.register_expert(eid, size, tier)
    tm.place_initial(eid, FakeTensor(size), tier)


# ── CXL Pool ──────────────────────────────────────────────────────────


class TestCXLPool:
    def test_store_retrieve(self):
        pool = CXLPool(capacity_bytes=10000, latency_ns=50, bandwidth_gbps=100.0)
        ft = FakeTensor(200)
        pool.store((0, 0), ft)

        assert pool.contains((0, 0))
        assert pool.usage_bytes() == 200
        assert pool.free_bytes() == 9800

        retrieved = pool.retrieve((0, 0))
        assert retrieved is ft

    def test_capacity_enforcement(self):
        pool = CXLPool(capacity_bytes=100, latency_ns=50, bandwidth_gbps=100.0)
        with pytest.raises(RuntimeError):
            pool.store((0, 0), FakeTensor(200))

    def test_evict(self):
        pool = CXLPool(capacity_bytes=10000, latency_ns=50, bandwidth_gbps=100.0)
        pool.store((0, 0), FakeTensor(200))
        evicted = pool.evict((0, 0))

        assert evicted is not None
        assert not pool.contains((0, 0))
        assert pool.usage_bytes() == 0

    def test_latency_injected(self):
        """CXL retrieve should be measurably slower than a plain dict lookup."""
        pool = CXLPool(capacity_bytes=10000, latency_ns=5000, bandwidth_gbps=100.0)
        pool.store((0, 0), FakeTensor(100))

        start = time.perf_counter_ns()
        pool.retrieve((0, 0))
        elapsed_ns = time.perf_counter_ns() - start

        # Should be >= 5000 ns (we injected 5 µs).  Allow some slack.
        assert elapsed_ns >= 3000, f"CXL latency too low: {elapsed_ns} ns"


# ── Tier ordering helpers ─────────────────────────────────────────────


class TestTierOrdering:
    def test_next_higher(self):
        assert _next_higher_tier(MemoryTier.CXL) == MemoryTier.DRAM
        assert _next_higher_tier(MemoryTier.DRAM) == MemoryTier.HBM
        assert _next_higher_tier(MemoryTier.HBM) is None

    def test_next_lower(self):
        assert _next_lower_tier(MemoryTier.HBM) == MemoryTier.DRAM
        assert _next_lower_tier(MemoryTier.DRAM) == MemoryTier.CXL
        assert _next_lower_tier(MemoryTier.CXL) is None


# ── 3-tier TierManager ────────────────────────────────────────────────


class TestThreeTierManager:
    def test_three_pools_initialized(self, tm):
        for tier in MemoryTier:
            assert tm.get_pool(tier) is not None

    def test_promote_cxl_to_dram(self, tm):
        _place(tm, (0, 0), 100, MemoryTier.CXL)
        assert tm.promote((0, 0)) is True
        assert tm.get_tier((0, 0)) == MemoryTier.DRAM

    def test_promote_cxl_to_hbm_multihop(self, tm):
        _place(tm, (0, 0), 100, MemoryTier.CXL)
        assert tm.promote_to_hbm((0, 0)) is True
        assert tm.get_tier((0, 0)) == MemoryTier.HBM

    def test_demote_hbm_to_dram(self, tm):
        _place(tm, (0, 0), 100, MemoryTier.HBM)
        tm.demote((0, 0))
        assert tm.get_tier((0, 0)) == MemoryTier.DRAM

    def test_demote_dram_to_cxl(self, tm):
        _place(tm, (0, 0), 100, MemoryTier.DRAM)
        tm.demote((0, 0))
        assert tm.get_tier((0, 0)) == MemoryTier.CXL

    def test_demote_cxl_returns_none(self, tm):
        _place(tm, (0, 0), 100, MemoryTier.CXL)
        result = tm.demote((0, 0))
        assert result is None  # can't go lower

    def test_summary_counts_all_tiers(self, tm):
        _place(tm, (0, 0), 100, MemoryTier.HBM)
        _place(tm, (0, 1), 200, MemoryTier.DRAM)
        _place(tm, (0, 2), 300, MemoryTier.CXL)

        s = tm.summary()
        assert s["hbm"]["expert_count"] == 1
        assert s["dram"]["expert_count"] == 1
        assert s["cxl"]["expert_count"] == 1


# ── Multi-hop Transfer Engine ─────────────────────────────────────────


class TestMultiHopTransfer:
    def test_demand_fetch_from_cxl(self, tm, config, metrics):
        _place(tm, (0, 0), 100, MemoryTier.CXL)
        engine = TransferEngine(tm, config, metrics)

        engine.demand_fetch((0, 0))
        assert tm.get_tier((0, 0)) == MemoryTier.HBM

    def test_demand_fetch_from_dram(self, tm, config, metrics):
        _place(tm, (0, 0), 100, MemoryTier.DRAM)
        engine = TransferEngine(tm, config, metrics)

        engine.demand_fetch((0, 0))
        assert tm.get_tier((0, 0)) == MemoryTier.HBM

    def test_already_in_hbm_noop(self, tm, config, metrics):
        _place(tm, (0, 0), 100, MemoryTier.HBM)
        engine = TransferEngine(tm, config, metrics)

        result = engine.demand_fetch((0, 0))
        assert result is not None
        assert tm.get_tier((0, 0)) == MemoryTier.HBM


# ── Double Buffer ─────────────────────────────────────────────────────


class TestDoubleBuffer:
    def test_swap_alternates(self):
        db = DoubleBuffer(buffer_size_bytes=1024)
        assert db.compute_idx == 0
        assert db.transfer_idx == 1

        db.swap()
        assert db.compute_idx == 1
        assert db.transfer_idx == 0

    def test_set_get_clear(self):
        db = DoubleBuffer(buffer_size_bytes=1024)
        db.set_tensor(0, (0, 0), "tensor_a")
        db.set_tensor(1, (0, 1), "tensor_b")

        assert db.get_tensor(0) == "tensor_a"
        assert db.get_expert(1) == (0, 1)

        db.clear(0)
        assert db.get_tensor(0) is None
        assert db.get_expert(0) is None


# ── Placement Policy ──────────────────────────────────────────────────


class TestPlacementPolicy:
    def test_warm_expert_goes_to_dram(self):
        policy = PlacementPolicy(warm_threshold=2.0)
        expert = ExpertMetadata(expert_id=(0, 0), size_bytes=100)
        expert.decayed_frequency = 5.0

        dest = policy.select_eviction_destination(
            expert, dram_free_bytes=10000, dram_capacity_bytes=50000, cxl_free_bytes=100000
        )
        assert dest == MemoryTier.DRAM

    def test_cold_expert_goes_to_cxl(self):
        policy = PlacementPolicy(warm_threshold=2.0)
        expert = ExpertMetadata(expert_id=(0, 0), size_bytes=100)
        expert.decayed_frequency = 0.5

        dest = policy.select_eviction_destination(
            expert, dram_free_bytes=0, dram_capacity_bytes=50000, cxl_free_bytes=100000
        )
        assert dest == MemoryTier.CXL

    def test_from_frequency_percentile(self):
        freqs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        policy = PlacementPolicy.from_frequency_percentile(freqs, percentile=50.0)
        # p50 of [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] → index 3 → 2.0
        assert policy.warm_threshold == 2.0


# ── Size-Aware Eviction ───────────────────────────────────────────────


class TestSizeAwareLFUEviction:
    def test_prefers_large_cold_experts(self):
        """Given equal frequency, larger experts should be evicted first."""
        eviction = SizeAwareLFUEviction(half_life=100)
        candidates = [
            ExpertMetadata(expert_id=(0, 0), size_bytes=100),
            ExpertMetadata(expert_id=(0, 1), size_bytes=1000),
        ]
        # Same frequency → score = freq / size → smaller score evicted first
        # (0,0): 0.0/100 = 0.0,  (0,1): 0.0/1000 = 0.0  → tied, but
        # let's set distinct frequencies to verify ordering
        candidates[0].decayed_frequency = 1.0  # 1.0/100 = 0.01
        candidates[1].decayed_frequency = 1.0  # 1.0/1000 = 0.001

        victims = eviction.select_victims(candidates, needed_bytes=500, current_token=0)
        # (0,1) has lower score (0.001) → evicted first
        assert victims[0] == (0, 1)
