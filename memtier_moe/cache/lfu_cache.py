"""LFU Expert Cache implementation."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.memory.tier_manager import TierManager
from memtier_moe.memory.expert_metadata import ExpertMetadata
from memtier_moe.cache.eviction import LFUEviction

logger = logging.getLogger(__name__)


@dataclass
class CachedExpert:
    """Represents an expert currently in the cache."""
    expert_id: ExpertId
    metadata: ExpertMetadata
    last_used_token: int


class LFUExpertCache:
    """Manages the HBM cache of experts."""

    def __init__(self, tier_manager: TierManager, config: MemTierConfig, metrics: MetricsTracker):
        """Initialize the LFU Cache.
        
        Args:
            tier_manager: The tier manager handling memory locations.
            config: Configuration for memory tiering.
            metrics: Tracker for cache events.
        """
        self.tier_manager = tier_manager
        self.config = config
        self.metrics = metrics
        
        self._cache: Dict[ExpertId, CachedExpert] = {}
        self._current_token: int = 0
        self.eviction_policy = LFUEviction(half_life=config.frequency_decay_half_life)
        
        # Keep track of current HBM usage within the cache
        self._hbm_used_bytes: int = 0

    def lookup(self, expert_ids: List[ExpertId]) -> Tuple[List[ExpertId], List[ExpertId]]:
        """Check which experts are in the cache.
        
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
                self.metrics.record_hit()
            else:
                misses.append(eid)
                self.metrics.record_miss()
        return hits, misses

    def get_weights(self, expert_id: ExpertId) -> Any:
        """Retrieve expert weights from the HBM pool."""
        pool = self.tier_manager.get_pool(MemoryTier.HBM)
        return pool.retrieve(expert_id)

    def insert(self, expert_id: ExpertId) -> List[ExpertId]:
        """Insert an expert into the cache, evicting if necessary.
        
        Args:
            expert_id: The expert to insert.
            
        Returns:
            List of evicted expert IDs.
        """
        if expert_id in self._cache:
            return []
            
        meta = self.tier_manager.get_metadata(expert_id)
        incoming_size = meta.size_bytes
        
        evicted = []
        if self.needs_eviction(incoming_size):
            evicted = self.evict_for(incoming_size)
            
        self._cache[expert_id] = CachedExpert(
            expert_id=expert_id,
            metadata=meta,
            last_used_token=self._current_token
        )
        self._hbm_used_bytes += incoming_size
        return evicted

    def record_accesses(self, expert_ids: List[ExpertId]) -> None:
        """Update access frequencies for experts and advance token.
        
        Args:
            expert_ids: List of accessed experts.
        """
        self._current_token += 1
        for eid in expert_ids:
            if eid in self._cache:
                expert = self._cache[eid]
                expert.metadata.update_frequency(self._current_token, self.config.frequency_decay_half_life)
                expert.last_used_token = self._current_token

    def needs_eviction(self, incoming_size: int) -> bool:
        """Check if eviction is needed for the incoming size."""
        return (self._hbm_used_bytes + incoming_size) > self.config.hbm_cache_budget_bytes

    def evict_for(self, needed_bytes: int) -> List[ExpertId]:
        """Evict experts until enough space is available.
        
        Args:
            needed_bytes: Space needed in bytes.
            
        Returns:
            List of evicted expert IDs.
        """
        available = self.config.hbm_cache_budget_bytes - self._hbm_used_bytes
        shortfall = needed_bytes - available
        
        if shortfall <= 0:
            return []
            
        candidates = [ce.metadata for ce in self._cache.values()]
        victims = self.eviction_policy.select_victims(
            candidates, shortfall, self._current_token
        )
        
        for victim_id in victims:
            victim_size = self._cache[victim_id].metadata.size_bytes
            del self._cache[victim_id]
            self._hbm_used_bytes -= victim_size
            self.metrics.record_eviction()
            
            logger.debug(f"Evicted expert {victim_id} of size {victim_size}")
            
        return victims

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
                self._hbm_used_bytes += meta.size_bytes

    def size(self) -> int:
        """Return the number of experts in HBM."""
        return len(self._cache)

    def hit_rate(self) -> float:
        """Return the current hit rate."""
        return self.metrics.hit_rate()
