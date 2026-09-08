"""Manager for tracking expert placement across different memory tiers."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.memory.expert_metadata import ExpertMetadata
from memtier_moe.memory.pool import MemoryPool, HBMPool, DRAMPool, CXLPool

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

# Tier ordering for promotion / demotion traversal
_TIER_ORDER = [MemoryTier.CXL, MemoryTier.DRAM, MemoryTier.HBM]


def _next_higher_tier(tier: MemoryTier) -> Optional[MemoryTier]:
    """Return the next tier *above* the given one, or None if already at top."""
    idx = _TIER_ORDER.index(tier)
    return _TIER_ORDER[idx + 1] if idx + 1 < len(_TIER_ORDER) else None


def _next_lower_tier(tier: MemoryTier) -> Optional[MemoryTier]:
    """Return the next tier *below* the given one, or None if already at bottom."""
    idx = _TIER_ORDER.index(tier)
    return _TIER_ORDER[idx - 1] if idx > 0 else None


class TierManager:
    """Manages placement and promotion/demotion of experts across memory tiers.

    M2 upgrade:  Three-tier support (HBM / DRAM / CXL) with helper utilities
    for tier traversal and multi-hop promotion.
    """

    def __init__(self, config: MemTierConfig, metrics: MetricsTracker):
        self.config = config
        self.metrics = metrics

        self._registry: Dict[ExpertId, ExpertMetadata] = {}

        # Initialize all three pools
        self._pools: Dict[MemoryTier, MemoryPool] = {
            MemoryTier.HBM: HBMPool(config.hbm_cache_budget_bytes),
            MemoryTier.DRAM: DRAMPool(
                capacity_bytes=config.host_dram_bytes,
                latency_ns=config.dram_latency_ns,
                bandwidth_gbps=config.pcie_bandwidth_gbps,
            ),
            MemoryTier.CXL: CXLPool(
                capacity_bytes=config.cxl_memory_bytes,
                latency_ns=config.cxl_latency_ns,
                bandwidth_gbps=config.cxl_bandwidth_gbps,
            ),
        }

    # ── Registration ──────────────────────────────────────────────────

    def register_expert(
        self,
        expert_id: ExpertId,
        size_bytes: int,
        initial_tier: MemoryTier = MemoryTier.DRAM,
    ) -> ExpertMetadata:
        """Register a new expert's metadata without storing its tensor."""
        metadata = ExpertMetadata(
            expert_id=expert_id, size_bytes=size_bytes, current_tier=initial_tier
        )
        self._registry[expert_id] = metadata
        logger.debug(f"Registered expert {expert_id} (size {size_bytes}B) to {initial_tier}")
        return metadata

    def register_experts_from_profile(self, profile_data: Dict[ExpertId, int]) -> None:
        """Bulk register experts from profiling data mapping ID → size in bytes."""
        for expert_id, size_bytes in profile_data.items():
            self.register_expert(expert_id, size_bytes)

    # ── Tensor placement ──────────────────────────────────────────────

    def place_initial(self, expert_id: ExpertId, tensor: Any, tier: MemoryTier) -> None:
        """Store the tensor in the specified pool and update metadata."""
        if expert_id not in self._registry:
            size_bytes = (
                tensor.element_size() * tensor.numel()
                if HAS_TORCH and isinstance(tensor, torch.Tensor)
                else (tensor.element_size() * tensor.numel()
                      if hasattr(tensor, 'element_size') else 1024)
            )
            self.register_expert(expert_id, size_bytes, tier)

        pool = self.get_pool(tier)
        pool.store(expert_id, tensor)
        self._registry[expert_id].current_tier = tier
        logger.debug(f"Placed initial tensor for {expert_id} in {tier}")

    # ── Promotion / demotion ──────────────────────────────────────────

    def promote(self, expert_id: ExpertId, tensor: Optional[Any] = None) -> bool:
        """Promote an expert one tier upward (CXL→DRAM or DRAM→HBM).

        Returns True on success, False if the target pool has no room.
        """
        metadata = self.get_metadata(expert_id)
        current_tier = metadata.current_tier

        target_tier = _next_higher_tier(current_tier)
        if target_tier is None:
            logger.debug(f"{expert_id} already at highest tier")
            return True  # already at top

        target_pool = self.get_pool(target_tier)

        if not target_pool.available_for(metadata.size_bytes):
            logger.debug(f"Cannot promote {expert_id} to {target_tier}: not enough space.")
            return False

        if tensor is None:
            current_pool = self.get_pool(current_tier)
            tensor = current_pool.retrieve(expert_id)
            current_pool.evict(expert_id)

        target_pool.store(expert_id, tensor)
        metadata.current_tier = target_tier
        self.metrics.record_promotion()
        logger.info(f"Promoted {expert_id} from {current_tier} to {target_tier}")
        return True

    def demote(self, expert_id: ExpertId) -> Optional[Any]:
        """Demote an expert one tier downward (HBM→DRAM or DRAM→CXL).

        Returns the tensor after it has been moved.

        Raises RuntimeError if the target tier has no room, rather than
        silently dropping the tensor on the floor (it has already been
        evicted from the source pool at that point, so swallowing the
        error would leave the expert nowhere — corrupting tier_manager
        state without a trace).
        """
        metadata = self.get_metadata(expert_id)
        current_tier = metadata.current_tier

        target_tier = _next_lower_tier(current_tier)
        if target_tier is None:
            logger.warning(f"Cannot demote {expert_id} — already at lowest tier (CXL)")
            return None

        return self.demote_to(expert_id, target_tier)

    def demote_to(self, expert_id: ExpertId, target_tier: MemoryTier) -> Optional[Any]:
        """Move an expert directly to *target_tier*, which need not be
        adjacent to its current tier (e.g. HBM straight to CXL).

        Used by the eviction placement policy (cache/placement.py), which
        decides per-expert whether a cold eviction from HBM should land in
        DRAM or skip straight to CXL, and by :meth:`demote` for the
        single-hop case.
        """
        metadata = self.get_metadata(expert_id)
        current_tier = metadata.current_tier
        if current_tier == target_tier:
            return None

        current_pool = self.get_pool(current_tier)
        target_pool = self.get_pool(target_tier)

        if not target_pool.available_for(metadata.size_bytes):
            raise RuntimeError(
                f"Cannot demote {expert_id} from {current_tier} to {target_tier}: "
                f"target tier full ({target_pool.usage_bytes()}/{target_pool.capacity_bytes} bytes used, "
                f"need {metadata.size_bytes})."
            )

        tensor = current_pool.evict(expert_id)
        if tensor is None:
            raise RuntimeError(f"Expert {expert_id} not found in source pool {current_tier}")

        target_pool.store(expert_id, tensor)
        metadata.current_tier = target_tier
        self.metrics.record_demotion()
        logger.info(f"Demoted {expert_id} from {current_tier} to {target_tier}")
        return tensor

    def promote_to_hbm(self, expert_id: ExpertId) -> bool:
        """Multi-hop promote: bring an expert all the way to HBM.

        Used by the transfer engine for CXL→DRAM→HBM two-hop transfers.
        """
        metadata = self.get_metadata(expert_id)
        while metadata.current_tier != MemoryTier.HBM:
            if not self.promote(expert_id):
                return False
        return True

    # ── Lookup helpers ────────────────────────────────────────────────

    def get_metadata(self, expert_id: ExpertId) -> ExpertMetadata:
        """Lookup metadata for an expert."""
        if expert_id not in self._registry:
            raise KeyError(f"Expert {expert_id} not found in registry")
        return self._registry[expert_id]

    def get_tier(self, expert_id: ExpertId) -> MemoryTier:
        """Return the current tier of an expert."""
        return self.get_metadata(expert_id).current_tier

    def get_pool(self, tier: MemoryTier) -> MemoryPool:
        """Return the pool instance for the given tier."""
        if tier not in self._pools:
            raise ValueError(f"Pool for tier {tier} not initialized")
        return self._pools[tier]

    def experts_in_tier(self, tier: MemoryTier) -> List[ExpertId]:
        """Return all expert IDs currently residing in the specified tier."""
        if tier in self._pools:
            return self._pools[tier].experts
        return [eid for eid, meta in self._registry.items() if meta.current_tier == tier]

    def experts_by_frequency(
        self, tier: Optional[MemoryTier] = None, ascending: bool = True
    ) -> List[ExpertMetadata]:
        """Return metadata sorted by decayed_frequency, optionally filtered by tier."""
        experts = list(self._registry.values())
        if tier is not None:
            experts = [meta for meta in experts if meta.current_tier == tier]

        experts.sort(key=lambda m: m.decayed_frequency, reverse=not ascending)
        return experts

    def hbm_usage(self) -> Tuple[int, int]:
        """Return a tuple of (used_bytes, total_bytes) for the HBM pool."""
        pool = self.get_pool(MemoryTier.HBM)
        return pool.usage_bytes(), pool.capacity_bytes

    def summary(self) -> Dict[str, Any]:
        """Return a summary of memory usage and expert distribution."""
        result = {}
        for tier in MemoryTier:
            count = len([m for m in self._registry.values() if m.current_tier == tier])
            usage = self._pools[tier].usage_bytes() if tier in self._pools else 0
            capacity = self._pools[tier].capacity_bytes if tier in self._pools else 0
            result[tier.value] = {
                "expert_count": count,
                "usage_bytes": usage,
                "capacity_bytes": capacity,
                "utilization_pct": (usage / capacity * 100) if capacity > 0 else 0.0,
            }
        return result
