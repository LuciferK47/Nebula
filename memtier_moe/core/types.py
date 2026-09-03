"""Core types for MemTier-MoE."""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import time

class MemoryTier(Enum):
    """Memory tier locations."""
    HBM = "hbm"
    DRAM = "dram"
    CXL = "cxl"

class TransferState(Enum):
    """State of a memory transfer."""
    IDLE = "idle"
    IN_FLIGHT = "in_flight"
    COMPLETE = "complete"
    FAILED = "failed"

ExpertId = Tuple[int, int]

@dataclass
class TransferRequest:
    """Request to transfer an expert between memory tiers."""
    expert_id: ExpertId
    source_tier: MemoryTier
    dest_tier: MemoryTier
    size_bytes: int
    priority: float = 0.0
    state: TransferState = TransferState.IDLE
    created_at: float = -1.0
    completed_at: Optional[float] = None
    
    def __post_init__(self) -> None:
        if self.created_at == -1.0:
            self.created_at = time.time()

@dataclass
class ExpertLocation:
    """Location information for an expert."""
    expert_id: ExpertId
    tier: MemoryTier
    offset: int = 0
