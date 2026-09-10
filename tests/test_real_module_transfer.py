"""Unit tests for transferring real PyTorch nn.Module experts across memory tiers."""
import pytest
import torch
import torch.nn as nn

from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.memory.tier_manager import TierManager
from memtier_moe.memory.transfer_engine import TransferEngine
from memtier_moe.memory.pool import _tensor_size_bytes


class DummyExpert(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_tensor_size_bytes_module():
    expert = DummyExpert(64, 64)
    expected_bytes = sum(p.numel() * p.element_size() for p in expert.parameters())
    assert _tensor_size_bytes(expert) == expected_bytes
    assert expected_bytes > 0


def test_real_module_demand_fetch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MemTierConfig(
        gpu_vram_bytes=100 * 1024 * 1024,
        hbm_cache_budget_bytes=50 * 1024 * 1024,
        host_dram_bytes=200 * 1024 * 1024,
        cxl_memory_bytes=200 * 1024 * 1024,
    )
    metrics = MetricsTracker()
    tm = TierManager(config, metrics)
    engine = TransferEngine(tm, config, metrics)

    eid: ExpertId = (0, 0)
    expert = DummyExpert(64, 64)
    size = _tensor_size_bytes(expert)

    # Reference output
    x = torch.randn(2, 64)
    with torch.no_grad():
        ref_out = expert(x)

    # Register in DRAM initially
    tm.register_expert(eid, size, MemoryTier.DRAM)
    tm.place_initial(eid, expert, MemoryTier.DRAM)

    assert tm.get_metadata(eid).current_tier == MemoryTier.DRAM

    # Demand fetch to HBM
    fetched_expert = engine.demand_fetch(eid)
    assert tm.get_metadata(eid).current_tier == MemoryTier.HBM
    assert tm.get_pool(MemoryTier.HBM).contains(eid)

    # Run forward on retrieved expert
    x_dev = x.to(device)
    with torch.no_grad():
        out = fetched_expert(x_dev)

    assert torch.allclose(ref_out, out.cpu(), atol=1e-5)


def test_real_module_cxl_to_hbm_multihop():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MemTierConfig(
        gpu_vram_bytes=100 * 1024 * 1024,
        hbm_cache_budget_bytes=50 * 1024 * 1024,
        host_dram_bytes=200 * 1024 * 1024,
        cxl_memory_bytes=200 * 1024 * 1024,
    )
    metrics = MetricsTracker()
    tm = TierManager(config, metrics)
    engine = TransferEngine(tm, config, metrics)

    eid: ExpertId = (1, 2)
    expert = DummyExpert(64, 64)
    size = _tensor_size_bytes(expert)

    x = torch.randn(2, 64)
    with torch.no_grad():
        ref_out = expert(x)

    # Register in CXL initially
    tm.register_expert(eid, size, MemoryTier.CXL)
    tm.place_initial(eid, expert, MemoryTier.CXL)

    # Demand fetch from CXL (triggers CXL -> DRAM -> HBM)
    fetched_expert = engine.demand_fetch(eid)
    assert tm.get_metadata(eid).current_tier == MemoryTier.HBM

    x_dev = x.to(device)
    with torch.no_grad():
        out = fetched_expert(x_dev)

    assert torch.allclose(ref_out, out.cpu(), atol=1e-5)


def test_real_module_async_fetch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MemTierConfig(
        gpu_vram_bytes=100 * 1024 * 1024,
        hbm_cache_budget_bytes=50 * 1024 * 1024,
        host_dram_bytes=200 * 1024 * 1024,
        cxl_memory_bytes=200 * 1024 * 1024,
    )
    metrics = MetricsTracker()
    tm = TierManager(config, metrics)
    engine = TransferEngine(tm, config, metrics)

    eid: ExpertId = (2, 1)
    expert = DummyExpert(64, 64)
    size = _tensor_size_bytes(expert)

    x = torch.randn(2, 64)
    with torch.no_grad():
        ref_out = expert(x)

    tm.register_expert(eid, size, MemoryTier.DRAM)
    tm.place_initial(eid, expert, MemoryTier.DRAM)

    handle = engine.async_fetch(eid)
    assert handle.request.expert_id == eid
    assert engine.is_inflight(eid)

    fetched_expert = engine.wait_for(eid)
    assert fetched_expert is not None
    assert not engine.is_inflight(eid)
    assert tm.get_metadata(eid).current_tier == MemoryTier.HBM

    x_dev = x.to(device)
    with torch.no_grad():
        out = fetched_expert(x_dev)

    assert torch.allclose(ref_out, out.cpu(), atol=1e-5)
