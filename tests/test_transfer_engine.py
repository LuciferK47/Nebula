"""Tests for Transfer Engine."""
from __future__ import annotations
import pytest

from memtier_moe.core.types import MemoryTier, TransferState
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.memory.tier_manager import TierManager
from memtier_moe.memory.transfer_engine import TransferEngine


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
def base_setup():
    config = MemTierConfig(
        max_inflight_transfers=2,
        hbm_cache_budget_bytes=10000,
        host_dram_bytes=100000,
    )
    metrics = MetricsTracker()
    tm = TierManager(config, metrics)
    engine = TransferEngine(tm, config, metrics)
    return tm, engine


def _register_and_place(tm, expert_id, size_bytes):
    """Helper: register expert and place a fake tensor in DRAM."""
    tm.register_expert(expert_id, size_bytes, MemoryTier.DRAM)
    tm.place_initial(expert_id, FakeTensor(size_bytes), MemoryTier.DRAM)


def test_demand_fetch_moves_tier(base_setup):
    tm, engine = base_setup
    _register_and_place(tm, (0, 0), 100)

    engine.demand_fetch((0, 0))

    assert tm.get_tier((0, 0)) == MemoryTier.HBM


def test_duplicate_fetch_prevention(base_setup):
    tm, engine = base_setup
    _register_and_place(tm, (0, 0), 100)

    handle1 = engine.async_fetch((0, 0))
    handle2 = engine.async_fetch((0, 0))

    assert handle1 is handle2


def test_inflight_tracking(base_setup):
    tm, engine = base_setup
    _register_and_place(tm, (0, 0), 100)

    engine.async_fetch((0, 0))
    assert engine.is_inflight((0, 0)) is True


def test_max_inflight_limit(base_setup):
    tm, engine = base_setup
    for i in range(3):
        _register_and_place(tm, (0, i), 100)

    assert engine.can_accept_transfer() is True
    engine.async_fetch((0, 0))
    engine.async_fetch((0, 1))

    assert engine.can_accept_transfer() is False
