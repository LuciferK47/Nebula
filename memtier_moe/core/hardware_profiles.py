"""Formal Hardware Specification Profiles for MemTier-MoE.

Calibrated against official JEDEC standards (HBM3/HBM3e, DDR5) and CXL Consortium
specifications (CXL 2.0/3.0 Type 3 Memory Expanders over PCIe 5.0/6.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from memtier_moe.core.config import MemTierConfig


@dataclass(frozen=True)
class HardwareProfile:
    """Rigorous physical specification profile for memory hierarchy modeling."""

    name: str
    description: str
    reference_spec: str

    # GPU / Accelerator Memory (Tier 0)
    gpu_memory_type: str
    gpu_vram_bytes: int
    hbm_bandwidth_gbps: float
    hbm_latency_ns: int

    # Host Interconnect
    pcie_generation: str
    pcie_bandwidth_gbps: float
    pcie_latency_ns: int

    # Host DRAM (Tier 1)
    dram_type: str
    host_dram_bytes: int
    dram_bandwidth_gbps: float
    dram_latency_ns: int

    # Disaggregated Expansion Memory (Tier 2)
    cxl_standard: str
    cxl_memory_bytes: int
    cxl_bandwidth_gbps: float
    cxl_latency_ns: int
    cxl_flit_overhead_pct: float = 3.125  # 256B Flit framing overhead in CXL 2.0/3.0

    def create_config(
        self,
        hbm_cache_budget_bytes: Optional[int] = None,
        **overrides,
    ) -> MemTierConfig:
        """Create a MemTierConfig calibrated directly from this hardware specification."""
        budget = hbm_cache_budget_bytes if hbm_cache_budget_bytes is not None else int(self.gpu_vram_bytes * 0.65)
        params = {
            "gpu_vram_bytes": self.gpu_vram_bytes,
            "hbm_cache_budget_bytes": budget,
            "host_dram_bytes": self.host_dram_bytes,
            "cxl_memory_bytes": self.cxl_memory_bytes,
            "pcie_bandwidth_gbps": self.pcie_bandwidth_gbps,
            "cxl_bandwidth_gbps": self.cxl_bandwidth_gbps,
            "cxl_latency_ns": self.cxl_latency_ns,
            "dram_latency_ns": self.dram_latency_ns,
        }
        params.update(overrides)
        return MemTierConfig(**params)


# --- Standard Pre-Calibrated Specification Profiles ---

JEDEC_HBM3_CXL2 = HardwareProfile(
    name="jedec-hbm3-cxl2",
    description="Enterprise Accelerator with JEDEC HBM3 and CXL 2.0 Type 3 DDR5 Expander",
    reference_spec="JEDEC JESD238 (HBM3) & CXL Consortium Specification 2.0 (Type 3 Pool)",
    gpu_memory_type="HBM3 (16-channel)",
    gpu_vram_bytes=48 * 1024 * 1024 * 1024,      # 48 GB
    hbm_bandwidth_gbps=819.2,                    # 819.2 GB/s per stack
    hbm_latency_ns=28,
    pcie_generation="PCIe Gen5 x16",
    pcie_bandwidth_gbps=31.5,                    # Practical unidirectional throughput
    pcie_latency_ns=120,
    dram_type="DDR5-5600 (8-channel ECC)",
    host_dram_bytes=128 * 1024 * 1024 * 1024,    # 128 GB
    dram_bandwidth_gbps=89.6,
    dram_latency_ns=85,
    cxl_standard="CXL 2.0 Type 3 (PCIe 5.0 x8)",
    cxl_memory_bytes=256 * 1024 * 1024 * 1024,   # 256 GB CXL pool
    cxl_bandwidth_gbps=32.0,
    cxl_latency_ns=230,                          # 85ns DRAM + 120ns PCIe link + 25ns CXL.mem unpack
    cxl_flit_overhead_pct=3.125,
)

JEDEC_HBM3E_CXL3 = HardwareProfile(
    name="jedec-hbm3e-cxl3",
    description="Next-Gen Hyperscale Node with JEDEC HBM3e and CXL 3.0 Fabric-Attached Pool",
    reference_spec="JEDEC JESD238A (HBM3e) & CXL Consortium Specification 3.0 (Fabric Direct)",
    gpu_memory_type="HBM3e (24GB/stack)",
    gpu_vram_bytes=96 * 1024 * 1024 * 1024,      # 96 GB
    hbm_bandwidth_gbps=1150.0,                   # 1.15 TB/s per stack
    hbm_latency_ns=24,
    pcie_generation="PCIe Gen6 x16 (PAM4)",
    pcie_bandwidth_gbps=63.0,
    pcie_latency_ns=90,
    dram_type="DDR5-6400 (8-channel ECC)",
    host_dram_bytes=256 * 1024 * 1024 * 1024,    # 256 GB
    dram_bandwidth_gbps=102.4,
    dram_latency_ns=75,
    cxl_standard="CXL 3.0 Fabric (PCIe 6.0 x8 PAM4)",
    cxl_memory_bytes=512 * 1024 * 1024 * 1024,   # 512 GB CXL Fabric
    cxl_bandwidth_gbps=64.0,
    cxl_latency_ns=180,                          # Low-latency flit switching + DDR5
    cxl_flit_overhead_pct=3.125,
)

WORKSTATION_RTX4050 = HardwareProfile(
    name="workstation-rtx4050",
    description="Edge Workstation with NVIDIA GeForce RTX 4050 Laptop GPU & Emulated CXL",
    reference_spec="NVIDIA Ada Lovelace Architecture Spec & JEDEC DDR5 Spec",
    gpu_memory_type="GDDR6 (96-bit)",
    gpu_vram_bytes=6 * 1024 * 1024 * 1024,       # 6 GB
    hbm_bandwidth_gbps=192.0,                    # 192 GB/s GDDR6 bus
    hbm_latency_ns=120,
    pcie_generation="PCIe Gen4 x16 (Consumer Laptop/Desktop Root Complex)",
    pcie_bandwidth_gbps=16.0,                    # Practical measured single-direction DMA under consumer desktop/Windows OS overhead
    pcie_latency_ns=180,
    dram_type="DDR5-4800 (Dual-channel)",
    host_dram_bytes=16 * 1024 * 1024 * 1024,     # 16 GB
    dram_bandwidth_gbps=76.8,
    dram_latency_ns=95,
    cxl_standard="CXL 2.0 Emulated",
    cxl_memory_bytes=32 * 1024 * 1024 * 1024,    # 32 GB
    cxl_bandwidth_gbps=8.0,
    cxl_latency_ns=350,
    cxl_flit_overhead_pct=3.125,
)

DATACENTER_A100 = HardwareProfile(
    name="datacenter-a100",
    description="Datacenter Node with NVIDIA A100 80GB SXM4 and CXL 2.0 Host Expansion",
    reference_spec="NVIDIA Ampere Architecture Whitepaper & PCIe 4.0 Spec",
    gpu_memory_type="HBM2e (5 stacks)",
    gpu_vram_bytes=80 * 1024 * 1024 * 1024,      # 80 GB
    hbm_bandwidth_gbps=2039.0,                   # 2.0 TB/s HBM2e
    hbm_latency_ns=30,
    pcie_generation="PCIe Gen4 x16 (Server Direct Root Complex)",
    pcie_bandwidth_gbps=25.0,                    # Measured enterprise server DMA throughput utilizing Linux hugepages & NUMA affinity
    pcie_latency_ns=150,
    dram_type="DDR4-3200 (8-channel)",
    host_dram_bytes=512 * 1024 * 1024 * 1024,    # 512 GB
    dram_bandwidth_gbps=204.8,
    dram_latency_ns=110,
    cxl_standard="CXL 2.0 Type 3",
    cxl_memory_bytes=1024 * 1024 * 1024 * 1024,  # 1 TB
    cxl_bandwidth_gbps=32.0,
    cxl_latency_ns=260,
    cxl_flit_overhead_pct=3.125,
)

PROFILES: Dict[str, HardwareProfile] = {
    "jedec-hbm3-cxl2": JEDEC_HBM3_CXL2,
    "jedec-hbm3e-cxl3": JEDEC_HBM3E_CXL3,
    "workstation-rtx4050": WORKSTATION_RTX4050,
    "datacenter-a100": DATACENTER_A100,
}


def get_hardware_profile(name: str) -> HardwareProfile:
    """Retrieve a formal hardware profile by case-insensitive name."""
    clean = name.strip().lower()
    if clean in PROFILES:
        return PROFILES[clean]
    raise KeyError(f"Unknown hardware profile '{name}'. Available: {list_hardware_profiles()}")


def list_hardware_profiles() -> List[str]:
    """Return all registered hardware profile identifiers."""
    return list(PROFILES.keys())
