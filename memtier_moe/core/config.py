"""Configuration settings for MemTier-MoE."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class MemTierConfig:
    """Configuration for MemTier-MoE memory and routing policies."""
    gpu_vram_bytes: int = 6_442_450_944
    hbm_cache_budget_bytes: int = 4_000_000_000
    host_dram_bytes: int = 16_000_000_000
    cxl_memory_bytes: int = 32_000_000_000
    pcie_bandwidth_gbps: float = 16.0
    cxl_bandwidth_gbps: float = 8.0
    cxl_latency_ns: int = 350
    dram_latency_ns: int = 100
    frequency_decay_half_life: int = 500
    eviction_policy: str = "lfu"
    max_inflight_transfers: int = 4
    max_prefetches_per_decision: int = 2
    prefetch_confidence_threshold: float = 0.3
    prefetch_lookahead_layers: int = 2
    top_k_experts: int = 4
    num_layers: int = 24
    num_experts_per_layer: int = 60
    num_shared_experts: int = 4

    # Eviction destination policy (memory/placement.py): when an expert is
    # evicted from HBM, experts at/above this decayed frequency are kept in
    # DRAM (likely to be reused soon); colder ones go to CXL. Reserve a
    # fraction of DRAM so CXL->DRAM promotions always have headroom.
    placement_warm_threshold: float = 1.0
    dram_reserve_fraction: float = 0.10

    # Adaptive prefetch gating: avoid speculative thrashing under memory pressure.
    # When cache headroom is low, do not evict resident experts whose decayed frequency
    # exceeds max_speculative_eviction_freq.
    adaptive_prefetch_gating: bool = True
    max_speculative_eviction_freq: float = 1.5

RTX4050_PRESET = MemTierConfig()
L40S_PRESET = MemTierConfig(
    gpu_vram_bytes=48_000_000_000,
    hbm_cache_budget_bytes=40_000_000_000,
    host_dram_bytes=128_000_000_000,
    pcie_bandwidth_gbps=32.0
)
