"""Regression tests for the DRAM->HBM double-charge fix.

Context: DRAMPool.retrieve() injects a synthetic latency + token-bucket
bandwidth cost documented as modeling "a real host-DRAM-to-HBM transfer
over PCIe". In the live TieredMoEWrapper path, a real torch.Tensor/
nn.Module retrieved from DRAM is then handed to a *real* CUDA copy
(`.to('cuda', non_blocking=True)`) — charging the synthetic cost AND
paying the real copy double-counts the same physical hop, inflating
weight-transfer's measured cost by roughly 2x relative to what the
hardware actually did.

The fix makes the charge conditional (`retrieve(expert_id, charge=...)`)
rather than removing it outright, because the pure-simulation path
(InferenceEngine / benchmark_runner.py, driven by `_Placeholder` objects
whose `.cuda()`/`.cpu()` are no-ops) has no real hardware transfer to
fall back on — there, the injected cost is the *only* signal, and must
stay. These tests pin both halves of that contract.
"""
from __future__ import annotations
import time

import pytest

from memtier_moe.memory.pool import DRAMPool
from memtier_moe.memory.transfer_engine import TransferEngine


class FakeTensor:
    """Stand-in for a torch.Tensor/nn.Module: NOT an instance of either,
    so it always exercises the "no real CUDA transfer follows" branch —
    matching how the pure-simulation _Placeholder behaves in this
    torch-less test environment.
    """

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


def test_charge_true_is_the_default_and_injects_cost():
    """Default behavior (charge=True) is unchanged: simulation callers
    that never pass `charge` keep paying the synthetic cost."""
    pool = DRAMPool(capacity_bytes=10_000, latency_ns=5000, bandwidth_gbps=100.0)
    pool.store((0, 0), FakeTensor(100))

    start = time.perf_counter_ns()
    pool.retrieve((0, 0))
    elapsed_ns = time.perf_counter_ns() - start

    assert elapsed_ns >= 3000, f"DRAM latency too low with charge=True: {elapsed_ns} ns"


def test_charge_false_skips_injected_cost():
    """charge=False (the live-GPU-transfer path) must not busy-wait —
    the real CUDA copy that follows is where the cost is actually paid."""
    pool = DRAMPool(capacity_bytes=10_000, latency_ns=5_000_000, bandwidth_gbps=0.001)
    pool.store((0, 0), FakeTensor(100))

    start = time.perf_counter_ns()
    pool.retrieve((0, 0), charge=False)
    elapsed_ns = time.perf_counter_ns() - start

    # With charge=True this latency/bandwidth combo would take milliseconds;
    # charge=False should return in well under 1ms.
    assert elapsed_ns < 1_000_000, f"charge=False still injected cost: {elapsed_ns} ns"


def test_charge_false_still_returns_the_stored_tensor():
    pool = DRAMPool(capacity_bytes=10_000, latency_ns=50, bandwidth_gbps=100.0)
    ft = FakeTensor(200)
    pool.store((0, 0), ft)
    assert pool.retrieve((0, 0), charge=False) is ft


def test_charge_false_missing_expert_still_raises():
    pool = DRAMPool(capacity_bytes=10_000, latency_ns=50, bandwidth_gbps=100.0)
    with pytest.raises(KeyError):
        pool.retrieve((0, 0), charge=False)


def test_dram_hop_is_real_false_without_torch_types():
    """A non-torch object (mirrors _Placeholder in this torch-less
    environment) must never be treated as a real CUDA hop — this is the
    exact guard that keeps the pure-simulation path's fidelity intact."""
    assert TransferEngine._dram_hop_is_real(FakeTensor(100)) is False
    assert TransferEngine._dram_hop_is_real(None) is False


def test_dram_pool_starts_with_empty_bucket_like_cxl():
    """DRAM's rate limiter should not start with an unrealistic multi-GB
    head start relative to CXL's — both start empty (see pool.py)."""
    pool = DRAMPool(capacity_bytes=10_000, latency_ns=50, bandwidth_gbps=16.0)
    assert pool._rate_limiter.tokens == 0.0
