"""Expert metadata tracking, including access frequencies and tier placement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from memtier_moe.core.types import ExpertId, MemoryTier, TransferState


@dataclass
class ExpertMetadata:
    """Metadata tracking for a single MoE expert."""
    expert_id: ExpertId
    size_bytes: int
    current_tier: MemoryTier = MemoryTier.DRAM
    raw_count: int = 0
    decayed_frequency: float = 0.0
    last_access_token: int = 0
    transfer_state: TransferState = TransferState.IDLE
    co_occurrence_partners: List[ExpertId] = field(default_factory=list)

    def update_frequency(self, current_token: int, half_life: int) -> None:
        """Update the decayed frequency based on current access.
        
        Applies exponential decay formula:
        decayed_frequency = decayed_frequency * (0.5 ** ((current_token - last_access_token) / half_life)) + 1.0
        """
        self.decayed_frequency = self.decay_frequency(current_token, half_life) + 1.0
        self.raw_count += 1
        self.last_access_token = current_token

    def decay_frequency(self, current_token: int, half_life: int) -> float:
        """Calculate what the decayed frequency WOULD be without updating (for comparison).
        
        Pure computation, no side effects.
        """
        if half_life <= 0:
            return 0.0
        decay_factor = 0.5 ** ((current_token - self.last_access_token) / half_life)
        return self.decayed_frequency * decay_factor

    def is_cold(self, threshold: float) -> bool:
        """Check if the expert is considered cold based on a threshold."""
        return self.decayed_frequency < threshold

    def is_transferring(self) -> bool:
        """Check if the expert is currently being transferred."""
        return self.transfer_state == TransferState.IN_FLIGHT
