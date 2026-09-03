"""Latency injection and rate limiting utilities."""
from __future__ import annotations
import time

def inject_latency_ns(target_ns: int) -> None:
    """
    Inject latency by busy-waiting for the target nanoseconds.
    
    Using time.sleep() is unreliable for sub-millisecond precision because
    OS scheduling overhead typically dominates sleep calls, making them wait
    far longer than requested. Busy-waiting provides nanosecond precision.
    """
    start = time.perf_counter_ns()
    while time.perf_counter_ns() - start < target_ns:
        pass

class TokenBucketRateLimiter:
    """Rate limiter for memory bandwidth using a token bucket algorithm."""
    
    def __init__(self, bandwidth_gbps: float, burst_factor: float = 1.5) -> None:
        """
        Initialize the rate limiter.
        
        Args:
            bandwidth_gbps: Bandwidth in gigabytes per second
            burst_factor: Multiplier for maximum burst capacity
        """
        self.max_tokens = int(bandwidth_gbps * 1e9 / 8) # bytes per second
        self.burst_bytes = int(self.max_tokens * burst_factor)
        self.tokens = float(self.max_tokens)
        self.last_update = time.perf_counter()
        
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.perf_counter()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.max_tokens
        self.tokens = min(self.tokens + new_tokens, float(self.burst_bytes))
        self.last_update = now
        
    def acquire(self, nbytes: int) -> float:
        """
        Consume tokens, busy-waiting if insufficient.
        
        Returns:
            Wait time in seconds.
        """
        start = time.perf_counter()
        self._refill()
        
        while self.tokens < nbytes:
            # Busy wait
            pass
            self._refill()
            
        self.tokens -= nbytes
        return time.perf_counter() - start
