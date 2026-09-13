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


def test_multihop_promotion_cxl_to_dram_to_hbm(config, metrics):
    """Test explicit step-by-step promotion CXL -> DRAM -> HBM."""
    tm = TierManager(config, metrics)
    tm.place_initial((0, 5), FakeTensor(200), MemoryTier.CXL)
    assert tm.get_tier((0, 5)) == MemoryTier.CXL

    # Hop 1: CXL -> DRAM
    assert tm.promote((0, 5)) is True
    assert tm.get_tier((0, 5)) == MemoryTier.DRAM

    # Hop 2: DRAM -> HBM
    assert tm.promote((0, 5)) is True
    assert tm.get_tier((0, 5)) == MemoryTier.HBM

    # Repeated promotion when already at highest tier returns True safely
    assert tm.promote((0, 5)) is True
    assert tm.get_tier((0, 5)) == MemoryTier.HBM


def test_multihop_promote_to_hbm_convenience(config, metrics):
    """Test promote_to_hbm which traverses multi-hop CXL -> DRAM -> HBM."""
    tm = TierManager(config, metrics)
    tm.place_initial((1, 0), FakeTensor(300), MemoryTier.CXL)
    assert tm.get_tier((1, 0)) == MemoryTier.CXL

    success = tm.promote_to_hbm((1, 0))
    assert success is True
    assert tm.get_tier((1, 0)) == MemoryTier.HBM


def test_promote_missing_expert(config, metrics):
    """Attempting to promote an expert without stored tensor returns False cleanly."""
    tm = TierManager(config, metrics)
    tm.register_expert((2, 0), 200, MemoryTier.DRAM)
    assert tm.promote((2, 0)) is False


def test_multihop_promotion_blocked_by_capacity(metrics):
    """When intermediate DRAM tier is full, promoting from CXL to DRAM fails gracefully."""
    cfg = MemTierConfig(hbm_cache_budget_bytes=1000, host_dram_bytes=300, cxl_memory_bytes=10000)
    tm = TierManager(cfg, metrics)

    # Fill DRAM (300 bytes)
    tm.place_initial((0, 1), FakeTensor(300), MemoryTier.DRAM)

    # Place expert in CXL (200 bytes)
    tm.place_initial((0, 2), FakeTensor(200), MemoryTier.CXL)

    # Promoting (0, 2) from CXL requires DRAM capacity, which is full
    assert tm.promote((0, 2)) is False
    assert tm.get_tier((0, 2)) == MemoryTier.CXL


def test_multihop_demotion_hbm_to_dram_to_cxl(config, metrics):
    """Test demotion cascade from HBM -> DRAM -> CXL."""
    tm = TierManager(config, metrics)
    tm.place_initial((3, 0), FakeTensor(200), MemoryTier.HBM)
    assert tm.get_tier((3, 0)) == MemoryTier.HBM

    # Demote 1: HBM -> DRAM
    demoted_tensor = tm.demote((3, 0))
    assert demoted_tensor is not None
    assert tm.get_tier((3, 0)) == MemoryTier.DRAM

    # Demote 2: DRAM -> CXL
    demoted_tensor2 = tm.demote((3, 0))
    assert demoted_tensor2 is not None
    assert tm.get_tier((3, 0)) == MemoryTier.CXL

    # Already at lowest tier: demote returns None
    demoted_tensor3 = tm.demote((3, 0))
    assert demoted_tensor3 is None
    assert tm.get_tier((3, 0)) == MemoryTier.CXL
