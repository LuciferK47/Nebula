"""Configuration settings for MemTier-MoE."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

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

    # CXL emulation fidelity mode, forwarded to CXLPool (memory/pool.py):
    #   "full"         — inject both latency and token-bucket bandwidth cost (default)
    #   "latency_only" — inject only the fixed access latency, skip bandwidth limiting
    #   "disabled"     — no injected cost at all
    # Used for the CXL sensitivity sweep (S3): if the architectural
    # conclusion (which baseline wins) is unchanged across these modes and
    # across a wide cxl_bandwidth_gbps range, the conclusion does not
    # depend on the CXL emulation's calibration accuracy. Must never
    # affect *placement* decisions (which tier an expert lives in, hit
    # rate) — only the injected timing cost. See test_dram_pool_charging.py
    # and test_emulation_mode_placement_parity.py for the pinned contract.
    cxl_emulation_mode: str = "full"
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

    # Optional formal specification profile name
    hardware_profile: Optional[str] = None

    @classmethod
    def from_profile(cls, profile_name: str, **overrides) -> MemTierConfig:
        """Instantiate a MemTierConfig calibrated to a formal JEDEC/CXL hardware specification."""
        from memtier_moe.core.hardware_profiles import get_hardware_profile
        prof = get_hardware_profile(profile_name)
        cfg = prof.create_config(**overrides)
        # return with hardware_profile tag attached
        import dataclasses
        return dataclasses.replace(cfg, hardware_profile=prof.name)

    @classmethod
    def auto_detect(
        cls,
        hbm_cache_budget_bytes: Optional[int] = None,
        host_dram_bytes: Optional[int] = None,
        cxl_memory_bytes: Optional[int] = None,
        **overrides,
    ) -> MemTierConfig:
        """Dynamically detect host GPU VRAM and host DRAM to adapt automatically to any reviewer machine."""
        import torch

        detected_vram = 6 * 1024 * 1024 * 1024
        if torch.cuda.is_available():
            try:
                detected_vram = int(torch.cuda.get_device_properties(0).total_memory)
            except Exception:
                pass

        detected_dram = 16 * 1024 * 1024 * 1024
        try:
            import psutil
            detected_dram = int(psutil.virtual_memory().total)
        except Exception:
            pass

        budget = hbm_cache_budget_bytes if hbm_cache_budget_bytes is not None else int(detected_vram * 0.65)
        dram_budget = host_dram_bytes if host_dram_bytes is not None else int(detected_dram * 0.5)
        cxl_budget = cxl_memory_bytes if cxl_memory_bytes is not None else max(32 * 1024 * 1024 * 1024, dram_budget * 2)

        return cls(
            gpu_vram_bytes=detected_vram,
            hbm_cache_budget_bytes=budget,
            host_dram_bytes=dram_budget,
            cxl_memory_bytes=cxl_budget,
            **overrides,
        )

RTX4050_PRESET = MemTierConfig()
L40S_PRESET = MemTierConfig(
    gpu_vram_bytes=48_000_000_000,
    hbm_cache_budget_bytes=40_000_000_000,
    host_dram_bytes=128_000_000_000,
    pcie_bandwidth_gbps=32.0
)

