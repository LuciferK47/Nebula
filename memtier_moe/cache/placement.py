"""Eviction destination logic for the three-tier hierarchy.

When an expert is evicted from HBM the placement policy decides whether
it should land in DRAM (warm — likely needed again soon) or CXL (cold —
infrequently accessed).  The decision is based on the expert's decayed
frequency relative to the population and the available capacity in each
lower tier.
"""
from __future__ import annotations

import logging
from typing import Optional

from memtier_moe.core.types import MemoryTier
from memtier_moe.memory.expert_metadata import ExpertMetadata

logger = logging.getLogger(__name__)


class PlacementPolicy:
    """Decides which lower tier an evicted expert should be demoted to.

    Parameters
    ----------
    warm_threshold : float
        Experts with ``decayed_frequency >= warm_threshold`` are placed
        in DRAM (they are likely to be recalled soon).  Below the
        threshold they go to CXL.
    dram_reserve_fraction : float
        Keep at least this fraction of DRAM free so that CXL→DRAM
        promotions always have headroom.  Default 0.1 (10 %).
    """

    def __init__(
        self,
        warm_threshold: float = 1.0,
        dram_reserve_fraction: float = 0.10,
    ) -> None:
        self.warm_threshold = warm_threshold
        self.dram_reserve_fraction = dram_reserve_fraction

    def select_eviction_destination(
        self,
        expert: ExpertMetadata,
        dram_free_bytes: int,
        dram_capacity_bytes: int,
        cxl_free_bytes: int,
    ) -> MemoryTier:
        """Choose DRAM or CXL for an expert being evicted from HBM.

        Decision tree:
        1. If frequency ≥ warm_threshold  →  DRAM  (likely recalled)
        2. If DRAM free – expert_size > reserve  →  DRAM  (space exists)
        3. Otherwise  →  CXL  (truly cold)
        """
        dram_reserve = int(dram_capacity_bytes * self.dram_reserve_fraction)
        expert_fits_dram = (dram_free_bytes - expert.size_bytes) >= dram_reserve

        if expert.decayed_frequency >= self.warm_threshold and expert_fits_dram:
            logger.debug(
                f"Placement {expert.expert_id}: DRAM (freq {expert.decayed_frequency:.2f} "
                f">= threshold {self.warm_threshold:.2f})"
            )
            return MemoryTier.DRAM

        if expert_fits_dram:
            logger.debug(
                f"Placement {expert.expert_id}: DRAM (space available, "
                f"free={dram_free_bytes}B)"
            )
            return MemoryTier.DRAM

        if cxl_free_bytes >= expert.size_bytes:
            logger.debug(
                f"Placement {expert.expert_id}: CXL (cold, "
                f"freq={expert.decayed_frequency:.2f})"
            )
            return MemoryTier.CXL

        # Absolute fallback — try DRAM even without reserve headroom
        logger.warning(
            f"Placement {expert.expert_id}: DRAM (forced — CXL also full)"
        )
        return MemoryTier.DRAM

    @staticmethod
    def from_frequency_percentile(
        frequencies: list[float],
        percentile: float = 50.0,
        dram_reserve_fraction: float = 0.10,
    ) -> "PlacementPolicy":
        """Create a policy with warm_threshold derived from frequency data.

        Parameters
        ----------
        frequencies : list[float]
            Decayed frequencies from the expert population.
        percentile : float
            Percentile of the frequency distribution to use as the warm
            threshold (default: p50 = median).
        """
        if not frequencies:
            return PlacementPolicy(warm_threshold=1.0, dram_reserve_fraction=dram_reserve_fraction)

        sorted_freqs = sorted(frequencies)
        idx = int(len(sorted_freqs) * percentile / 100.0)
        idx = min(idx, len(sorted_freqs) - 1)
        threshold = sorted_freqs[idx]

        logger.info(
            f"PlacementPolicy: p{percentile:.0f} threshold = {threshold:.3f} "
            f"(from {len(frequencies)} experts)"
        )
        return PlacementPolicy(
            warm_threshold=threshold,
            dram_reserve_fraction=dram_reserve_fraction,
        )
