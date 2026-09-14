"""LFU Expert Cache implementation."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.memory.tier_manager import TierManager
from memtier_moe.memory.expert_metadata import ExpertMetadata
from memtier_moe.cache.eviction import EvictionPolicy, LFUEviction, SizeAwareLFUEviction
from memtier_moe.cache.placement import PlacementPolicy

logger = logging.getLogger(__name__)


def _build_eviction_policy(config: MemTierConfig) -> EvictionPolicy:
    """`config.eviction_policy` used to be declared but never read — the
    cache always hardcoded plain LFU, so `SizeAwareLFUEviction` (evicts by
    frequency-per-byte rather than frequency alone, favoring keeping many
    small hot experts over one large one) was unreachable outside its own
    unit test."""
    if config.eviction_policy == "size_aware_lfu":
        return SizeAwareLFUEviction(half_life=config.frequency_decay_half_life)
    return LFUEviction(half_life=config.frequency_decay_half_life)


@dataclass
class CachedExpert:
    """Represents an expert currently in the cache."""
    expert_id: ExpertId
    metadata: ExpertMetadata
    last_used_token: int


class LFUExpertCache:
    """Tracks which experts are resident in HBM and evicts under pressure.

    HBM occupancy has a single source of truth: the ``HBMPool`` inside
    ``tier_manager``. This class used to keep a parallel ``_hbm_used_bytes``
    counter that drifted out of sync with the pool whenever an expert left
    HBM via ``tier_manager.demote()`` directly (the counter wasn't told),
    which made ``needs_eviction()`` under-report usage and let callers
    attempt to store into an already-full pool. Querying the pool directly
    removes that failure mode by construction.
    """

    def __init__(
        self,
        tier_manager: TierManager,
        config: MemTierConfig,
        metrics: MetricsTracker,
        placement_policy: Optional[PlacementPolicy] = None,
    ):
        """Initialize the LFU Cache.

        Args:
            tier_manager: The tier manager handling memory locations.
            config: Configuration for memory tiering.
            metrics: Tracker for cache events.
            placement_policy: Decides whether an expert evicted from HBM
                lands in DRAM or CXL. Defaults to a policy built from
                ``config.placement_warm_threshold`` / ``dram_reserve_fraction``.
        """
        self.tier_manager = tier_manager
        self.config = config
        self.metrics = metrics

        self._cache: Dict[ExpertId, CachedExpert] = {}
        self._current_token: int = 0
        self.eviction_policy = _build_eviction_policy(config)
        self.placement_policy = placement_policy or PlacementPolicy(
            warm_threshold=config.placement_warm_threshold,
            dram_reserve_fraction=config.dram_reserve_fraction,
        )

    @property
    def _hbm_pool(self):
        return self.tier_manager.get_pool(MemoryTier.HBM)

    def hbm_used_bytes(self) -> int:
        """Current HBM occupancy, read directly from the pool."""
        return self._hbm_pool.usage_bytes()

    def lookup(self, expert_ids: List[ExpertId]) -> Tuple[List[ExpertId], List[ExpertId]]:
        """Check which experts are in the cache.

        Deliberately does NOT record hit/miss metrics itself. Both callers
        (runtime/engine.py and runtime/tiered_model.py's weight_transfer
        branch) always follow this with a per-expert tier-specific call —
        record_hbm_hit() for each hit, record_dram_hit()/record_cxl_hit()/
        record_disk_fault() for each miss — which is where the tier
        breakdown actually comes from (a miss here doesn't yet know which
        tier it will resolve to). This method used to *also* call
        record_hit()/record_miss() directly, which double-counted every
        lookup: cache_hits/cache_misses ended up 2x the real total (2494
        vs the 1247 actual lookups for one S2 run), and because only the
        HBM side was doubled symmetrically, hbm_hit_rate stayed correct
        by accident while dram_hit_rate/cxl_hit_rate came out understated
        by exactly 2x. See tests/test_lfu_cache.py::test_lookup_does_not_
        double_count_with_caller_tier_recording.

        Args:
            expert_ids: List of requested expert IDs.

        Returns:
            Tuple of (hits, misses).
        """
        hits = []
        misses = []
        for eid in expert_ids:
            if eid in self._cache:
                hits.append(eid)
            else:
                misses.append(eid)
        return hits, misses

    def get_weights(self, expert_id: ExpertId) -> Any:
        """Retrieve expert weights from the HBM pool."""
        pool = self.tier_manager.get_pool(MemoryTier.HBM)
        return pool.retrieve(expert_id)

    def make_room(self, incoming_size: int, pinned: Optional[Set[ExpertId]] = None) -> List[ExpertId]:
        """Evict cached experts until HBM has room for `incoming_size` bytes."""
        if self._hbm_pool.available_for(incoming_size):
            return []

        shortfall = incoming_size - self._hbm_pool.free_bytes()
        pinned_set = pinned or set()
        candidates = [ce.metadata for ce in self._cache.values() if ce.expert_id not in pinned_set]
        victims = self.eviction_policy.select_victims(
            candidates, shortfall, self._current_token
        )

        dram_pool = self.tier_manager.get_pool(MemoryTier.DRAM)
        cxl_pool = self.tier_manager.get_pool(MemoryTier.CXL)

        evicted: List[ExpertId] = []
        for victim_id in victims:
            meta = self._cache[victim_id].metadata
            destination = self.placement_policy.select_eviction_destination(
                meta,
                dram_free_bytes=dram_pool.free_bytes(),
                dram_capacity_bytes=dram_pool.capacity_bytes,
                cxl_free_bytes=cxl_pool.free_bytes(),
            )
            self.tier_manager.demote_to(victim_id, destination)
            del self._cache[victim_id]
            self.metrics.record_eviction()
            evicted.append(victim_id)

            logger.debug(f"Evicted expert {victim_id} of size {meta.size_bytes} -> {destination}")

            if self._hbm_pool.available_for(incoming_size):
                break

        if not self._hbm_pool.available_for(incoming_size):
            # Every non-pinned resident was evicted and there is still not
            # enough room. This happens when a caller's pinned set (e.g. a
            # layer's whole top-k active set, which must all be resident
            # simultaneously to compute that layer) is larger than the HBM
            # budget. Silently returning here — the old behavior — lets the
            # subsequent store() raise a generic "out of memory" error two
            # frames away with no indication of *why*; raise here instead,
            # with the numbers that actually explain it.
            pinned_bytes = sum(
                self.tier_manager.get_metadata(eid).size_bytes for eid in pinned_set
            )
            raise RuntimeError(
                f"Cannot make room for {incoming_size / 1e6:.1f} MB in HBM: "
                f"{len(pinned_set)} pinned expert(s) already need {pinned_bytes / 1e6:.1f} MB "
                f"against an HBM budget of {self._hbm_pool.capacity_bytes / 1e6:.1f} MB, "
                f"leaving no non-pinned resident to evict. A pinned set (e.g. one layer's "
                f"full top-k active set) must fit entirely in HBM at once — raise "
                f"hbm_cache_budget_bytes, or use execution_mode='hybrid' so cold experts "
                f"execute in place instead of being pinned."
            )

        return evicted

    def ensure_resident(
        self,
        expert_id: ExpertId,
        transfer_engine: Any,
        pinned: Optional[Set[ExpertId]] = None,
    ) -> bool:
        """Guarantee `expert_id` is in HBM and registered as cached, fetching
        it if necessary.
        """
        if expert_id in self._cache:
            return True

        meta = self.tier_manager.get_metadata(expert_id)
        pinned_set = set(pinned) if pinned else set()
        pinned_set.add(expert_id)
        self.make_room(meta.size_bytes, pinned=pinned_set)

        if transfer_engine.is_inflight(expert_id):
            # Pass the same pinned set used above: if this in-flight
            # transfer's completion needs to evict to make room, it must
            # not evict a same-layer sibling this caller is relying on
            # staying resident.
            transfer_engine.wait_for(expert_id, pinned=pinned_set)
        else:
            transfer_engine.demand_fetch(expert_id)

        self.insert(expert_id)
        return False

    def insert(self, expert_id: ExpertId) -> None:
        """Register an expert as cached.

        The tensor must already be physically resident in the HBM pool
        (i.e. ``transfer_engine.demand_fetch``/``wait_for`` has run) —
        ``make_room`` must be called beforehand to guarantee space exists.
        This method only updates the cache index.
        """
        if expert_id in self._cache:
            return

        meta = self.tier_manager.get_metadata(expert_id)
        self._cache[expert_id] = CachedExpert(
            expert_id=expert_id,
            metadata=meta,
            last_used_token=self._current_token,
        )

    def set_token(self, token_idx: int) -> None:
        """Set the logical token counter used for decay calculations.

        Decay is defined per *token*, not per layer-visit — with 24+
        MoE layers a naive per-call increment would make the configured
        half-life run out ~24x faster than intended. Callers should call
        this once per token (``InferenceEngine`` does so automatically).
        """
        self._current_token = token_idx

    def record_accesses(self, expert_ids: List[ExpertId]) -> None:
        """Update access frequencies for the given experts at the current token.

        Args:
            expert_ids: List of accessed experts.
        """
        for eid in expert_ids:
            if eid in self._cache:
                expert = self._cache[eid]
                expert.metadata.update_frequency(self._current_token, self.config.frequency_decay_half_life)
                expert.last_used_token = self._current_token

    def warm_start(self, frequency_data: Dict[ExpertId, float]) -> None:
        """Pre-load frequency counters from profiling data.

        Args:
            frequency_data: Mapping from expert_id to initial frequency.
        """
        for eid, freq in frequency_data.items():
            meta = self.tier_manager.get_metadata(eid)
            meta.decayed_frequency = freq
            meta.raw_count = int(freq)
            # If the expert was already placed in HBM during setup, track it
            if meta.current_tier == MemoryTier.HBM and eid not in self._cache:
                self._cache[eid] = CachedExpert(
                    expert_id=eid,
                    metadata=meta,
                    last_used_token=0
                )

    def size(self) -> int:
        """Return the number of experts in HBM."""
        return len(self._cache)

    def hit_rate(self) -> float:
        """Return the current hit rate."""
        return self.metrics.hit_rate()
