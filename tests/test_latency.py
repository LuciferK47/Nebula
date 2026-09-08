"""Tests for core/latency.py: latency injection and the bandwidth rate limiter.

Two real bugs are covered here:
  1. TokenBucketRateLimiter treated `bandwidth_gbps` as gigabits/sec
     (dividing by 8) even though every caller and the config field name
     mean gigabytes/sec — an 8x understatement of every tier's bandwidth.
  2. acquire() spun on `while self.tokens < nbytes: pass` with no bound,
     so a single request larger than the bucket's burst capacity (e.g. a
     multi-hundred-MB expert against a burst sized for ~1.5s of traffic)
     would never accumulate enough tokens and hang forever.
"""
from __future__ import annotations
import time

import pytest

from memtier_moe.core.latency import TokenBucketRateLimiter, inject_latency_ns


def test_bandwidth_is_gigabytes_not_gigabits():
    rl = TokenBucketRateLimiter(bandwidth_gbps=8.0)
    assert rl.bytes_per_second == pytest.approx(8e9)


def test_transfer_time_matches_configured_bandwidth():
    rl = TokenBucketRateLimiter(bandwidth_gbps=16.0)
    # 16 GB/s -> 1 GB should take ~1/16 s
    assert rl.transfer_time_s(1_000_000_000) == pytest.approx(1 / 16, rel=1e-6)


def test_acquire_oversized_request_does_not_hang():
    """A request larger than the burst bucket must still complete."""
    rl = TokenBucketRateLimiter(bandwidth_gbps=1000.0, burst_factor=1.0)  # 1 TB/s burst
    oversized = int(rl.burst_bytes * 3)

    start = time.perf_counter()
    rl.acquire(oversized)
    elapsed = time.perf_counter() - start

    # Should take roughly (oversized - burst) / rate seconds, and in
    # particular must terminate at all (the old code spun forever here).
    assert elapsed < 5.0


def test_acquire_within_burst_is_fast():
    rl = TokenBucketRateLimiter(bandwidth_gbps=1000.0, burst_factor=2.0)
    start = time.perf_counter()
    rl.acquire(1000)  # tiny relative to a 1000 GB/s burst bucket
    elapsed = time.perf_counter() - start
    assert elapsed < 0.05


def test_inject_latency_ns_roughly_respects_target():
    # Not nanosecond-precise (documented approximation), but should not be
    # off by orders of magnitude the way time.sleep() would be for ~1us.
    target_ns = 50_000  # 50 us
    actual_ns = inject_latency_ns(target_ns)
    assert actual_ns >= target_ns
    assert actual_ns < target_ns * 20  # generous slack for CI jitter


def test_inject_latency_ns_zero_is_free():
    start = time.perf_counter_ns()
    inject_latency_ns(0)
    assert time.perf_counter_ns() - start < 1_000_000  # under 1ms
