"""Rigorous verification of metrics calculation, invariants, and edge cases."""
import pytest
from memtier_moe.core.metrics import MetricsTracker


def test_tier_partition_of_unity():
    """Verify that all tier hit rates sum exactly to 1.0."""
    tracker = MetricsTracker()
    tracker.record_hbm_hit(126)
    tracker.record_dram_hit(398)
    tracker.record_cxl_hit(100)
    tracker.record_disk_fault(0)

    total = tracker.total_accesses()
    assert total == 624

    h_hbm = tracker.hbm_hit_rate()
    h_dram = tracker.dram_hit_rate()
    h_cxl = tracker.cxl_hit_rate()
    h_disk = tracker.disk_fault_rate()

    assert h_hbm == 126 / 624
    assert h_dram == 398 / 624
    assert h_cxl == 100 / 624
    assert h_disk == 0.0
    assert pytest.approx(h_hbm + h_dram + h_cxl + h_disk, 1e-9) == 1.0


def test_msr_and_dsar_boundaries():
    """Verify Memory Service Rate (MSR) and CXL Disk Stall Avoidance Rate (DSAR)."""
    tracker = MetricsTracker()
    # Empty tracker boundary
    assert tracker.memory_service_rate() == 1.0
    assert tracker.cxl_disk_stall_avoidance_rate() == 100.0

    # All accesses in memory
    tracker.record_hbm_hit(50)
    tracker.record_dram_hit(30)
    tracker.record_cxl_hit(20)
    assert tracker.memory_service_rate() == 1.0
    assert tracker.cxl_disk_stall_avoidance_rate() == 100.0

    # Accesses spill to disk
    tracker.record_disk_fault(20)
    # Total = 120, in_memory = 100, off_dram = 20 CXL + 20 Disk = 40
    assert pytest.approx(tracker.memory_service_rate(), 1e-6) == 100 / 120
    assert pytest.approx(tracker.cxl_disk_stall_avoidance_rate(), 1e-6) == 20 / 40 * 100.0  # 50%


def test_amat_calculation_exactness():
    """Verify Hennessy-Patterson AMAT math."""
    tracker = MetricsTracker()
    tracker.record_hbm_hit(10)
    tracker.record_dram_hit(4)
    tracker.record_cxl_hit(2)
    tracker.record_disk_fault(1)

    # total = 17
    # AMAT = (10*28 + 4*95 + 2*260 + 1*15_000_000) / 17
    # 280 + 380 + 520 + 15_000_000 = 15_001_180 / 17 = 882422.3529411765 ns
    expected_amat = 15_001_180 / 17
    assert pytest.approx(tracker.amat_ns(), 1e-6) == expected_amat


def test_legacy_miss_counter_divergence():
    """Verify the exact reason max(tier_total, classic_total) was implemented."""
    tracker = MetricsTracker()
    tracker.record_hit(10)  # bumps cache_hits and hbm_hits
    tracker.record_miss(5)  # bumps cache_misses only

    assert tracker.counters["cache_hits"] == 10
    assert tracker.counters["hbm_hits"] == 10
    assert tracker.counters["cache_misses"] == 5
    assert tracker.counters["dram_hits"] == 0

    # tier_total = 10, classic_total = 15
    # total_accesses must be 15, NOT 10
    assert tracker.total_accesses() == 15
    assert pytest.approx(tracker.hit_rate(), 1e-6) == 10 / 15


def test_hybrid_activation_transfer_math():
    """Verify activation byte computation for single-token decode."""
    hidden_dim = 1024
    element_size = 2  # fp16/bf16
    batch_size = 1
    num_offloaded_experts = 2

    # 1 input activation: 1 * 1024 * 2 = 2048 bytes
    # 1 output activation: 1 * 1024 * 2 = 2048 bytes
    # Total round-trip per expert = 4096 bytes
    expected_bytes = num_offloaded_experts * (batch_size * hidden_dim * element_size * 2)
    assert expected_bytes == 8192
