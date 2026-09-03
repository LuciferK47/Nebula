"""Tests for Tier Manager."""
from __future__ import annotations
import pytest

from memtier_moe.core.types import MemoryTier, ExpertId
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.memory.tier_manager import TierManager


class FakeTensor:
    """Lightweight mock tensor for testing without torch."""
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


@pytest.fixture
def config():
    return MemTierConfig(hbm_cache_budget_bytes=1000, host_dram_bytes=10000)


@pytest.fixture
def metrics():
    return MetricsTracker()


def test_register_expert(config, metrics):
    tm = TierManager(config, metrics)
    meta = tm.register_expert((0, 0), 500, MemoryTier.DRAM)

    assert meta.expert_id == (0, 0)
    assert meta.size_bytes == 500
    assert meta.current_tier == MemoryTier.DRAM

    retrieved = tm.get_metadata((0, 0))
    assert retrieved == meta


def test_promote_demote(config, metrics):
    tm = TierManager(config, metrics)
    tm.register_expert((0, 0), 500, MemoryTier.DRAM)

    # Must place a tensor in DRAM first so promote() can retrieve it
    tm.place_initial((0, 0), FakeTensor(500), MemoryTier.DRAM)

    success = tm.promote((0, 0))
    assert success is True
    assert tm.get_tier((0, 0)) == MemoryTier.HBM

    tm.demote((0, 0))
    assert tm.get_tier((0, 0)) == MemoryTier.DRAM


def test_promote_fails_when_full(config, metrics):
    tm = TierManager(config, metrics)
    tm.register_expert((0, 0), 500, MemoryTier.DRAM)
    tm.register_expert((0, 1), 600, MemoryTier.DRAM)

    tm.place_initial((0, 0), FakeTensor(500), MemoryTier.DRAM)
    tm.place_initial((0, 1), FakeTensor(600), MemoryTier.DRAM)

    assert tm.promote((0, 0)) is True
    assert tm.promote((0, 1)) is False


def test_experts_by_frequency_ordering(config, metrics):
    tm = TierManager(config, metrics)
    for i in range(3):
        meta = tm.register_expert((0, i), 100, MemoryTier.DRAM)
        meta.decayed_frequency = float(i)

    experts = tm.experts_by_frequency(tier=MemoryTier.DRAM, ascending=True)
    assert [e.expert_id for e in experts] == [(0, 0), (0, 1), (0, 2)]

    experts_desc = tm.experts_by_frequency(tier=MemoryTier.DRAM, ascending=False)
    assert [e.expert_id for e in experts_desc] == [(0, 2), (0, 1), (0, 0)]


def test_bulk_register(config, metrics):
    tm = TierManager(config, metrics)
    for i in range(10):
        tm.register_expert((0, i), 100, MemoryTier.DRAM)

    # experts_in_tier uses pool.experts for pool-backed tiers,
    # but register_expert doesn't store tensors, so use registry
    all_meta = tm.experts_by_frequency(tier=MemoryTier.DRAM)
    assert len(all_meta) == 10
