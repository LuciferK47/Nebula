"""Four-baseline ablation configurations.

Each baseline uses the same InferenceEngine but with different tier and
prefetch settings, enabling controlled comparison.

| Baseline     | Tiers         | Cache | Prefetch |
|------------- |---------------|-------|----------|
| GPU-Resident | HBM only      | N/A   | No       |
| Two-Tier     | HBM + DRAM    | LFU   | No       |
| CXL-Only     | HBM+DRAM+CXL | LFU   | No       |
| MemTier-MoE  | HBM+DRAM+CXL | LFU   | Yes      |
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from memtier_moe.core.config import MemTierConfig

logger = logging.getLogger(__name__)


class BaselineType(Enum):
    GPU_RESIDENT = "gpu_resident"
    TWO_TIER = "two_tier"
    CXL_ONLY = "cxl_only"
    MEMTIER_MOE = "memtier_moe"


@dataclass
class BaselineConfig:
    """Configuration for a single baseline experiment."""
    name: str
    baseline_type: BaselineType
    config: MemTierConfig
    enable_prefetch: bool = False
    description: str = ""


def make_baseline_configs(
    hbm_budget_bytes: int = 4_000_000_000,
    host_dram_bytes: int = 16_000_000_000,
    cxl_memory_bytes: int = 32_000_000_000,
) -> Dict[BaselineType, BaselineConfig]:
    """Create the four baseline configurations.

    Parameters
    ----------
    hbm_budget_bytes : int
        HBM cache budget (same across all baselines for fair comparison).
    host_dram_bytes : int
        Available host DRAM.
    cxl_memory_bytes : int
        Available CXL memory (ignored for GPU-Resident and Two-Tier).

    Returns
    -------
    Dict mapping BaselineType → BaselineConfig.
    """
    base = MemTierConfig(
        hbm_cache_budget_bytes=hbm_budget_bytes,
        host_dram_bytes=host_dram_bytes,
        cxl_memory_bytes=cxl_memory_bytes,
    )

    configs = {}

    # 1. GPU-Resident: everything in HBM (huge budget, no offload)
    configs[BaselineType.GPU_RESIDENT] = BaselineConfig(
        name="GPU-Resident",
        baseline_type=BaselineType.GPU_RESIDENT,
        config=MemTierConfig(
            hbm_cache_budget_bytes=200_000_000_000,  # effectively infinite
            host_dram_bytes=0,
            cxl_memory_bytes=0,
        ),
        enable_prefetch=False,
        description="All experts in HBM. No offloading.",
    )

    # 2. Two-Tier: HBM + DRAM, no CXL
    configs[BaselineType.TWO_TIER] = BaselineConfig(
        name="Two-Tier (HBM+DRAM)",
        baseline_type=BaselineType.TWO_TIER,
        config=MemTierConfig(
            hbm_cache_budget_bytes=hbm_budget_bytes,
            host_dram_bytes=host_dram_bytes,
            cxl_memory_bytes=0,
        ),
        enable_prefetch=False,
        description="Classic two-tier offloading without CXL.",
    )

    # 3. CXL-Only: 3-tier but no predictive prefetching
    configs[BaselineType.CXL_ONLY] = BaselineConfig(
        name="CXL-Only (no prefetch)",
        baseline_type=BaselineType.CXL_ONLY,
        config=base,
        enable_prefetch=False,
        description="Three-tier placement with LFU cache, no prefetching.",
    )

    # 4. MemTier-MoE: full system
    configs[BaselineType.MEMTIER_MOE] = BaselineConfig(
        name="MemTier-MoE (full)",
        baseline_type=BaselineType.MEMTIER_MOE,
        config=base,
        enable_prefetch=True,
        description="Three-tier + LFU cache + router-predictive prefetching.",
    )

    return configs


def describe_baselines(configs: Dict[BaselineType, BaselineConfig]) -> str:
    """Pretty-print the baseline matrix."""
    lines = ["Baseline Configurations:", "=" * 60]
    for bt, bc in configs.items():
        lines.append(f"\n  [{bt.value}] {bc.name}")
        lines.append(f"    {bc.description}")
        lines.append(f"    HBM: {bc.config.hbm_cache_budget_bytes / 1e9:.1f} GB")
        lines.append(f"    DRAM: {bc.config.host_dram_bytes / 1e9:.1f} GB")
        lines.append(f"    CXL: {bc.config.cxl_memory_bytes / 1e9:.1f} GB")
        lines.append(f"    Prefetch: {'Yes' if bc.enable_prefetch else 'No'}")
    return "\n".join(lines)
