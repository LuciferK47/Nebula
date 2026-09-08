"""Latency injection and bandwidth rate limiting utilities."""
from __future__ import annotations
import time

# Measured floor of the spin loop itself (one perf_counter_ns call plus loop
# overhead).  Targets below this cannot be resolved by busy-waiting in Python.
SPIN_RESOLUTION_NS = 100


def inject_latency_ns(target_ns: int) -> int:
    """Busy-wait for approximately *target_ns* nanoseconds.

    ``time.sleep`` has ~1 ms granularity on CPython, so a 350 ns target would
    become 1-15 ms (four orders of magnitude too large).  Spinning against
    ``perf_counter_ns`` gets within roughly 1-3x of the target.

    The absolute value is an approximation; what the evaluation relies on is
    that the *relative* ordering HBM < DRAM < CXL is preserved and measurable.

    Returns the actual elapsed nanoseconds so callers can report calibration
    error instead of assuming the target was hit exactly.
    """
    if target_ns <= 0:
        return 0
    start = time.perf_counter_ns()
    while time.perf_counter_ns() - start < target_ns:
        pass
    return time.perf_counter_ns() - start


def measure_spin_overhead_ns(samples: int = 1000) -> float:
    """Measure this machine's floor for :func:`inject_latency_ns`.

    Useful for reporting the calibration error of the CXL latency injection
    rather than claiming nanosecond precision.
    """
    start = time.perf_counter_ns()
    for _ in range(samples):
        time.perf_counter_ns()
    return (time.perf_counter_ns() - start) / samples


class TokenBucketRateLimiter:
    """Token-bucket rate limiter used to emulate a tier's bandwidth ceiling.

    One token is one byte.  ``bandwidth_gbps`` is interpreted as **gigabytes
    per second** (1 GB/s = 1e9 bytes/s), matching the tier bandwidths in
    :class:`~memtier_moe.core.config.MemTierConfig` (e.g. PCIe 4.0 x8 is
    ~16 GB/s).

    Rather than spinning until enough tokens accumulate, :meth:`acquire`
    computes the exact deficit wait time up front.  This makes the limiter
    both cheaper and immune to the deadlock that occurs when a single request
    is larger than the bucket's burst capacity.
    """

    def __init__(self, bandwidth_gbps: float, burst_factor: float = 1.5) -> None:
        """
        Args:
            bandwidth_gbps: Sustained bandwidth in gigabytes per second.
            burst_factor: Bucket capacity as a multiple of one second of
                bandwidth.  Requests larger than this are still served (they
                simply wait proportionally longer).
        """
        if bandwidth_gbps <= 0:
            raise ValueError(f"bandwidth_gbps must be positive, got {bandwidth_gbps}")

        self.bandwidth_gbps = bandwidth_gbps
        # Bytes per second.  NOTE: gigaBYTES, not gigabits — do not divide by 8.
        self.bytes_per_second = bandwidth_gbps * 1e9
        self.burst_bytes = self.bytes_per_second * burst_factor

        self.tokens: float = self.burst_bytes
        self.last_update = time.perf_counter()

    # Backwards-compatible alias: older code referred to the bucket rate as
    # ``max_tokens``.
    @property
    def max_tokens(self) -> float:
        """Sustained refill rate in bytes per second."""
        return self.bytes_per_second

    def _refill(self) -> None:
        """Add tokens proportional to elapsed wall time, capped at the burst size."""
        now = time.perf_counter()
        self.tokens = min(
            self.tokens + (now - self.last_update) * self.bytes_per_second,
            self.burst_bytes,
        )
        self.last_update = now

    def transfer_time_s(self, nbytes: int) -> float:
        """Time this many bytes would take at the sustained rate (no waiting)."""
        return nbytes / self.bytes_per_second

    def acquire(self, nbytes: int) -> float:
        """Consume *nbytes* tokens, busy-waiting for any deficit.

        Unlike a spin-until-available loop, the wait is computed analytically,
        so a request larger than the bucket capacity completes instead of
        hanging forever.

        Returns:
            Wait time in seconds.
        """
        if nbytes <= 0:
            return 0.0

        start = time.perf_counter()
        self._refill()

        if self.tokens < nbytes:
            deficit_s = (nbytes - self.tokens) / self.bytes_per_second
            deadline = time.perf_counter() + deficit_s
            while time.perf_counter() < deadline:
                pass
            self._refill()

        self.tokens -= nbytes
        return time.perf_counter() - start
