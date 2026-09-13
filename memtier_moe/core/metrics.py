"""Metrics tracking for cache, CXL memory expansion, and multi-tier transfers."""
from __future__ import annotations
from typing import Dict, Any

class MetricsTracker:
    """Tracker for hierarchical memory tier hits, CXL expansion, and transfers."""

    def __init__(self) -> None:
        """Initialize all counters to zero."""
        self.counters: Dict[str, Any] = {
            # Classical cache counters (backwards compatible)
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

            # First-class hierarchical memory tier counters
            "hbm_hits": 0,
            "dram_hits": 0,
            "cxl_hits": 0,
            "disk_faults": 0,
            "cxl_transfer_bytes": 0,
            "pcie_activation_bytes": 0,
            "emulate_access_time_ms": 0.0,
            "cpu_expert_exec_time_ms": 0.0,
        }

    def record_hit(self, n: int = 1) -> None:
        """Record general cache hits (defaulting to HBM)."""
        self.counters["cache_hits"] += n
        self.counters["hbm_hits"] += n

    def record_miss(self, n: int = 1) -> None:
        """Record general cache misses."""
        self.counters["cache_misses"] += n

    def record_hbm_hit(self, n: int = 1) -> None:
        """Record expert access served directly by GPU HBM (Tier 0)."""
        self.counters["cache_hits"] += n
        self.counters["hbm_hits"] += n

    def record_dram_hit(self, n: int = 1) -> None:
        """Record expert access served by Host Local DRAM (Tier 1)."""
        self.counters["cache_misses"] += n
        self.counters["dram_hits"] += n

    def record_cxl_hit(self, n: int = 1) -> None:
        """Record expert access served by CXL Type 3 Memory Expansion (Tier 2)."""
        self.counters["cache_misses"] += n
        self.counters["cxl_hits"] += n

    def record_disk_fault(self, n: int = 1) -> None:
        """Record expert access that missed all memory and incurred a swap/disk page fault."""
        self.counters["cache_misses"] += n
        self.counters["disk_faults"] += n

    def record_cxl_transfer(self, size_bytes: int) -> None:
        """Record memory volume moved across the CXL link."""
        self.counters["cxl_transfer_bytes"] += size_bytes

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

    def total_accesses(self) -> int:
        """Total memory access requests across all tiers."""
        tier_total = (
            self.counters["hbm_hits"]
            + self.counters["dram_hits"]
            + self.counters["cxl_hits"]
            + self.counters["disk_faults"]
        )
        classic_total = self.counters["cache_hits"] + self.counters["cache_misses"]
        return max(tier_total, classic_total)

    def hit_rate(self) -> float:
        """Calculate the HBM cache hit rate (backwards compatible)."""
        total = self.total_accesses()
        hits = max(self.counters["cache_hits"], self.counters["hbm_hits"])
        return hits / total if total > 0 else 0.0

    def hbm_hit_rate(self) -> float:
        """Percentage of expert requests served by GPU HBM ($H_{\\text{HBM}}$)."""
        total = self.total_accesses()
        return self.counters["hbm_hits"] / total if total > 0 else 0.0

    def dram_hit_rate(self) -> float:
        """Percentage of expert requests served by Host DRAM ($H_{\\text{DRAM}}$)."""
        total = self.total_accesses()
        return self.counters["dram_hits"] / total if total > 0 else 0.0

    def cxl_hit_rate(self) -> float:
        """Percentage of expert requests served by CXL Memory ($H_{\\text{CXL}}$)."""
        total = self.total_accesses()
        return self.counters["cxl_hits"] / total if total > 0 else 0.0

    def disk_fault_rate(self) -> float:
        """Percentage of expert requests that fell through to disk swap ($H_{\\text{Disk}}$)."""
        total = self.total_accesses()
        return self.counters["disk_faults"] / total if total > 0 else 0.0

    def memory_service_rate(self) -> float:
        """Memory Service Rate (MSR): fraction of accesses served by coherent semiconductor memory."""
        total = self.total_accesses()
        in_memory = (
            self.counters["hbm_hits"]
            + self.counters["dram_hits"]
            + self.counters["cxl_hits"]
        )
        return in_memory / total if total > 0 else 1.0

    def cxl_disk_stall_avoidance_rate(self) -> float:
        """CXL Disk-Stall Avoidance Rate (DSAR): % of off-DRAM requests intercepted by CXL rather than disk."""
        off_dram = self.counters["cxl_hits"] + self.counters["disk_faults"]
        if off_dram == 0:
            return 100.0
        return (self.counters["cxl_hits"] / off_dram) * 100.0

    def amat_ns(
        self,
        t_hbm: float = 28.0,
        t_dram: float = 95.0,
        t_cxl: float = 260.0,
        t_disk: float = 15_000_000.0,
    ) -> float:
        """Hennessy-Patterson Average Memory Access Time (AMAT) in nanoseconds."""
        total = self.total_accesses()
        if total == 0:
            return t_hbm
        h_hbm = self.counters["hbm_hits"] / total
        h_dram = self.counters["dram_hits"] / total
        h_cxl = self.counters["cxl_hits"] / total
        h_disk = self.counters["disk_faults"] / total

        # Fallback if specific tier hits were not granularly recorded
        if h_hbm == 0 and h_dram == 0 and h_cxl == 0 and h_disk == 0:
            h_hbm = self.hit_rate()
            h_dram = 1.0 - h_hbm

        return (h_hbm * t_hbm) + (h_dram * t_dram) + (h_cxl * t_cxl) + (h_disk * t_disk)

    def prefetch_precision(self) -> float:
        """Calculate prefetch precision."""
        total = self.counters["prefetch_useful"] + self.counters["prefetch_wasted"]
        return self.counters["prefetch_useful"] / total if total > 0 else 0.0

    def snapshot(self) -> Dict[str, Any]:
        """Point-in-time copy of the raw additive counters.

        Deliberately excludes computed ratios (hit_rate, amat_ns, dsar_pct,
        ...) that report() adds — those are cumulative-since-start values,
        and subtracting two of them between two points in time does NOT
        give "the rate during that window" (e.g. hit_rate at token 250
        minus hit_rate at token 200 is not the hit rate of tokens 200-250).
        Pair with windowed_rates() to compute a correct windowed rate from
        two snapshots — used by long-horizon runs that report hit-rate /
        throughput in rolling windows rather than only a final cumulative
        number.
        """
        return dict(self.counters)

    @staticmethod
    def windowed_rates(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, float]:
        """Compute hit-rate-style ratios for the window between two
        snapshot() calls (curr taken strictly after prev), instead of the
        cumulative-since-start values report() would give at either
        endpoint.
        """
        delta = {k: curr.get(k, 0) - prev.get(k, 0) for k in curr}
        hbm = delta.get("hbm_hits", 0)
        dram = delta.get("dram_hits", 0)
        cxl = delta.get("cxl_hits", 0)
        disk = delta.get("disk_faults", 0)
        tier_total = hbm + dram + cxl + disk
        classic_total = delta.get("cache_hits", 0) + delta.get("cache_misses", 0)
        total = max(tier_total, classic_total)
        hits = max(delta.get("cache_hits", 0), hbm)

        return {
            "tokens_processed": delta.get("tokens_processed", 0),
            "hit_rate": hits / total if total > 0 else 0.0,
            "hbm_hit_rate": hbm / total if total > 0 else 0.0,
            "dram_hit_rate": dram / total if total > 0 else 0.0,
            "cxl_hit_rate": cxl / total if total > 0 else 0.0,
            "disk_fault_rate": disk / total if total > 0 else 0.0,
            "evictions": delta.get("evictions", 0),
            "total_transfer_bytes": delta.get("total_transfer_bytes", 0),
        }

    def report(self) -> Dict[str, Any]:
        """Return a copy of all counters and computed rates."""
        stats = self.counters.copy()
        stats["hit_rate"] = self.hit_rate()
        stats["hbm_hit_rate"] = self.hbm_hit_rate()
        stats["dram_hit_rate"] = self.dram_hit_rate()
        stats["cxl_hit_rate"] = self.cxl_hit_rate()
        stats["disk_fault_rate"] = self.disk_fault_rate()
        stats["memory_service_rate"] = self.memory_service_rate()
        stats["dsar_pct"] = self.cxl_disk_stall_avoidance_rate()
        stats["amat_ns"] = self.amat_ns()
        stats["prefetch_precision"] = self.prefetch_precision()
        stats["cxl_transfer_mb"] = self.counters["cxl_transfer_bytes"] / 1e6
        return stats

    def reset(self) -> None:
        """Reset all counters to zero."""
        for key in self.counters:
            if isinstance(self.counters[key], int):
                self.counters[key] = 0
            else:
                self.counters[key] = 0.0

