"""Metrics tracking for cache and memory transfers."""
from __future__ import annotations
from typing import Dict, Any

class MetricsTracker:
    """Tracker for cache hits, misses, and transfer metrics."""
    def __init__(self) -> None:
        """Initialize all counters to zero."""
        self.counters: Dict[str, Any] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "promotions": 0,
            "demotions": 0,
            "prefetch_issued": 0,
            "prefetch_useful": 0,
            "prefetch_wasted": 0,
            "tokens_processed": 0,
            "total_transfer_bytes": 0,
            "total_transfer_time_ms": 0.0,
        }
        
    def record_hit(self, n: int = 1) -> None:
        """Record cache hits."""
        self.counters["cache_hits"] += n
        
    def record_miss(self, n: int = 1) -> None:
        """Record cache misses."""
        self.counters["cache_misses"] += n
        
    def record_eviction(self, n: int = 1) -> None:
        """Record expert evictions."""
        self.counters["evictions"] += n
        
    def record_promotion(self, n: int = 1) -> None:
        """Record expert promotions to higher tier."""
        self.counters["promotions"] += n
        
    def record_demotion(self, n: int = 1) -> None:
        """Record expert demotions to lower tier."""
        self.counters["demotions"] += n
        
    def record_prefetch(self, useful: bool) -> None:
        """Record a prefetch and whether it was useful."""
        self.counters["prefetch_issued"] += 1
        if useful:
            self.counters["prefetch_useful"] += 1
        else:
            self.counters["prefetch_wasted"] += 1
            
    def record_token(self) -> None:
        """Record a processed token."""
        self.counters["tokens_processed"] += 1
        
    def record_transfer(self, size_bytes: int, time_ms: float) -> None:
        """Record a memory transfer."""
        self.counters["total_transfer_bytes"] += size_bytes
        self.counters["total_transfer_time_ms"] += time_ms
        
    def hit_rate(self) -> float:
        """Calculate the cache hit rate."""
        total = self.counters["cache_hits"] + self.counters["cache_misses"]
        return self.counters["cache_hits"] / total if total > 0 else 0.0
        
    def prefetch_precision(self) -> float:
        """Calculate prefetch precision."""
        total = self.counters["prefetch_useful"] + self.counters["prefetch_wasted"]
        return self.counters["prefetch_useful"] / total if total > 0 else 0.0
        
    def report(self) -> Dict[str, Any]:
        """Return a copy of all counters and computed rates."""
        stats = self.counters.copy()
        stats["hit_rate"] = self.hit_rate()
        stats["prefetch_precision"] = self.prefetch_precision()
        return stats
        
    def reset(self) -> None:
        """Reset all counters to zero."""
        for key in self.counters:
            if isinstance(self.counters[key], int):
                self.counters[key] = 0
            else:
                self.counters[key] = 0.0
