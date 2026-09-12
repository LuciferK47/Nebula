"""Simulator-Compatible Memory Trace Exporter for MemTier-MoE.

Captures runtime memory transactions across HBM/VRAM, DRAM, and CXL tiers,
and exports them into standard formats for external cycle-accurate memory
simulators (DRAMSim3, gem5) and Pandas dataframe analytics.

Note: Timestamps are derived from runtime wall-clock events (perf_counter_ns) and
addresses are synthetic tier-segmented base offsets. The module formats traces
for external simulator consumption rather than executing an internal cycle-accurate simulator.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from memtier_moe.core.types import ExpertId, MemoryTier


@dataclass(frozen=True)
class MemoryTransaction:
    """A discrete memory access transaction across physical tiers."""

    timestamp_ns: int
    tier: MemoryTier
    access_type: str  # "READ" or "WRITE"
    address: int
    size_bytes: int
    expert_id: Optional[Tuple[int, int]]  # (layer_idx, expert_idx)
    tag: str  # e.g., "activation_offload", "weight_swap", "prefetch"


class TraceExporter:
    """Transaction recorder exporting traces compatible with DRAMSim3 and gem5 simulators."""

    def __init__(self, max_records: int = 500_000, enabled: bool = False) -> None:
        self.max_records = max_records
        self.enabled = enabled
        self._records: List[MemoryTransaction] = []
        self._start_time_ns = time.perf_counter_ns()
        # Synthetic base address space per tier for simulator memory map
        self._tier_base_addresses = {
            MemoryTier.HBM: 0x1000_0000_0000,   # 16 TB base
            MemoryTier.DRAM: 0x2000_0000_0000,  # 32 TB base
            MemoryTier.CXL: 0x3000_0000_0000,   # 48 TB base
        }

    def start_tracing(self) -> None:
        """Enable live transaction tracing."""
        self.enabled = True
        self._start_time_ns = time.perf_counter_ns()

    def enable(self, capacity: Optional[int] = None) -> None:
        """Enable live transaction tracing, optionally setting max_records capacity."""
        if capacity is not None:
            self.max_records = capacity
        self.start_tracing()

    def stop_tracing(self) -> None:
        """Disable live transaction tracing."""
        self.enabled = False

    def clear(self) -> None:
        """Clear recorded transaction history."""
        self._records.clear()
        self._start_time_ns = time.perf_counter_ns()

    def record(
        self,
        tier: MemoryTier,
        size_bytes: int,
        access_type: str = "READ",
        address: Optional[int] = None,
        expert_id: Optional[ExpertId] = None,
        tag: str = "",
    ) -> None:
        """Record a physical memory transaction with minimal critical-path overhead."""
        if not self.enabled:
            return

        if len(self._records) >= self.max_records:
            return  # bounded memory protection

        now_ns = time.perf_counter_ns() - self._start_time_ns

        # Synthesize realistic deterministic memory map address if not explicitly passed
        if address is None:
            base = self._tier_base_addresses.get(tier, 0x1000_0000_0000)
            if expert_id is not None:
                offset = (expert_id[0] * 64 + expert_id[1]) * 32 * 1024 * 1024
            else:
                offset = len(self._records) * 4096
            address = base + offset

        tx = MemoryTransaction(
            timestamp_ns=now_ns,
            tier=tier,
            access_type=access_type.upper(),
            address=address,
            size_bytes=size_bytes,
            expert_id=expert_id,
            tag=tag,
        )
        self._records.append(tx)

    def export_dramsim3(self, filepath: str, clock_rate_ghz: float = 2.0) -> int:
        """Export trace in standard DRAMSim3 trace format: `0x<hex_address> <READ|WRITE> <cycle_count>`.

        Args:
            filepath: Destination file path for the .dramsim3 trace.
            clock_rate_ghz: Memory controller clock frequency in GHz (default: 2.0 GHz = 0.5ns/cycle).

        Returns:
            Number of transactions written.
        """
        written = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for tx in self._records:
                # Convert nanoseconds to memory controller clock cycles
                cycle = int(tx.timestamp_ns * clock_rate_ghz)
                f.write(f"0x{tx.address:012x} {tx.access_type} {cycle}\n")
                written += 1
        return written

    def export_gem5(self, filepath: str) -> int:
        """Export trace in standard gem5 ASCII memory packet format: `<tick_ps> <type> <addr_hex> <size> <flags>`.

        Returns:
            Number of transactions written.
        """
        written = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for tx in self._records:
                # gem5 uses picosecond ticks (1 ns = 1000 ps)
                tick_ps = tx.timestamp_ns * 1000
                f.write(f"{tick_ps} {tx.access_type} 0x{tx.address:x} {tx.size_bytes} 0\n")
                written += 1
        return written

    def export_csv(self, filepath: str) -> int:
        """Export structured tabular memory transaction log for Pandas analytics."""
        written = 0
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_ns",
                "tier_name",
                "tier_id",
                "access_type",
                "address_hex",
                "size_bytes",
                "layer_idx",
                "expert_idx",
                "tag",
            ])
            for tx in self._records:
                layer_idx = tx.expert_id[0] if tx.expert_id else -1
                exp_idx = tx.expert_id[1] if tx.expert_id else -1
                tier_id_map = {"hbm": 0, "dram": 1, "cxl": 2}
                tier_id = tier_id_map.get(tx.tier.value.lower(), 0) if hasattr(tx.tier, "value") else 0
                writer.writerow([
                    tx.timestamp_ns,
                    tx.tier.name if hasattr(tx.tier, "name") else str(tx.tier),
                    tier_id,
                    tx.access_type,
                    f"0x{tx.address:012x}",
                    tx.size_bytes,
                    layer_idx,
                    exp_idx,
                    tx.tag,
                ])
                written += 1
        return written

    def to_pandas(self) -> Any:
        """Convert recorded transactions directly to a Pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas must be installed to use to_pandas().")

        rows = []
        for tx in self._records:
            rows.append({
                "timestamp_ns": tx.timestamp_ns,
                "tier": tx.tier.name,
                "access_type": tx.access_type,
                "address": tx.address,
                "size_bytes": tx.size_bytes,
                "layer_idx": tx.expert_id[0] if tx.expert_id else None,
                "expert_idx": tx.expert_id[1] if tx.expert_id else None,
                "tag": tx.tag,
            })
        return pd.DataFrame(rows)

    @property
    def records(self) -> List[MemoryTransaction]:
        """List of recorded transactions."""
        return list(self._records)

    @property
    def transaction_count(self) -> int:
        """Total number of recorded transactions."""
        return len(self._records)


# Global trace exporter instance accessible across modules
GLOBAL_TRACE_EXPORTER = TraceExporter()
