"""Tests for MetricsTracker.snapshot() / windowed_rates() — used by the
long-horizon scenario (S5) to report hit-rate in rolling windows instead
of only a single cumulative number.

The core property under test: a windowed rate computed by diffing two
snapshots must match what a *fresh* tracker fed only that window's events
would report, not the cumulative-since-start rate report() would give.
"""
from __future__ import annotations

from memtier_moe.core.metrics import MetricsTracker


def test_windowed_rate_matches_a_fresh_tracker_over_the_same_window():
    m = MetricsTracker()

    # "Before" window: mostly misses.
    for _ in range(8):
        m.record_dram_hit()
    for _ in range(2):
        m.record_hbm_hit()
    snap_before = m.snapshot()

    # "Window": mostly hits — should read very differently in isolation
    # than lumped into the cumulative total.
    for _ in range(9):
        m.record_hbm_hit()
    m.record_dram_hit()
    snap_after = m.snapshot()

    windowed = MetricsTracker.windowed_rates(snap_before, snap_after)
    assert windowed["hbm_hit_rate"] == 0.9
    assert windowed["dram_hit_rate"] == 0.1
    assert windowed["hit_rate"] == 0.9

    # Cross-check against a fresh tracker fed only the window's events.
    fresh = MetricsTracker()
    for _ in range(9):
        fresh.record_hbm_hit()
    fresh.record_dram_hit()
    fresh_report = fresh.report()
    assert windowed["hbm_hit_rate"] == fresh_report["hbm_hit_rate"]
    assert windowed["hit_rate"] == fresh_report["hit_rate"]

    # And this must differ from the naive (wrong) approach of subtracting
    # report()'s cumulative hit_rate at the two endpoints.
    cumulative_before = 2 / 10
    cumulative_after = 11 / 20
    assert windowed["hit_rate"] != (cumulative_after - cumulative_before)


def test_snapshot_excludes_computed_ratio_keys():
    m = MetricsTracker()
    m.record_hbm_hit()
    snap = m.snapshot()
    for ratio_key in ("hit_rate", "amat_ns", "dsar_pct", "prefetch_precision"):
        assert ratio_key not in snap


def test_empty_window_returns_zero_rates_not_error():
    m = MetricsTracker()
    snap = m.snapshot()
    windowed = MetricsTracker.windowed_rates(snap, snap)
    assert windowed["hit_rate"] == 0.0
    assert windowed["tokens_processed"] == 0
