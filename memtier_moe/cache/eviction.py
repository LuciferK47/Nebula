"""Eviction policies for the MoE cache."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from memtier_moe.core.types import ExpertId
from memtier_moe.memory.expert_metadata import ExpertMetadata


class EvictionPolicy(ABC):
    """Abstract base class for eviction policies."""

    @abstractmethod
    def select_victims(
        self, candidates: List[ExpertMetadata], needed_bytes: int, current_token: int
    ) -> List[ExpertId]:
        """Select experts to evict to free up `needed_bytes`.
        
        Args:
            candidates: List of experts currently in HBM that can be evicted.
            needed_bytes: Number of bytes that need to be freed.
            current_token: The current token index, used for decay calculations.
            
        Returns:
            List of ExpertIds to evict.
        """
        pass


class LFUEviction(EvictionPolicy):
    """Least Frequently Used eviction policy with time decay."""

    def __init__(self, half_life: int = 500):
        """Initialize LFU eviction.
        
        Args:
            half_life: The half-life parameter for frequency decay.
        """
        self.half_life = half_life

    def select_victims(
        self, candidates: List[ExpertMetadata], needed_bytes: int, current_token: int
    ) -> List[ExpertId]:
        """Select victims based on lowest decayed frequency."""
        sorted_candidates = sorted(
            candidates, 
            key=lambda meta: meta.decay_frequency(current_token, self.half_life)
        )

        freed_bytes = 0
        victims = []
        for meta in sorted_candidates:
            if freed_bytes >= needed_bytes:
                break
            victims.append(meta.expert_id)
            freed_bytes += meta.size_bytes

        return victims


class SizeAwareLFUEviction(EvictionPolicy):
    """Size-aware LFU eviction policy."""

    def __init__(self, half_life: int = 500):
        self.half_life = half_life

    def select_victims(
        self, candidates: List[ExpertMetadata], needed_bytes: int, current_token: int
    ) -> List[ExpertId]:
        """Select victims based on score = decayed_frequency / size_bytes.
        
        TODO(M2): Further tune the scoring function.
        """
        sorted_candidates = sorted(
            candidates, 
            key=lambda meta: (meta.decay_frequency(current_token, self.half_life) / meta.size_bytes)
        )

        freed_bytes = 0
        victims = []
        for meta in sorted_candidates:
            if freed_bytes >= needed_bytes:
                break
            victims.append(meta.expert_id)
            freed_bytes += meta.size_bytes

        return victims
