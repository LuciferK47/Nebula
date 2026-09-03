"""Inference Engine for tiered MoE execution."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.memory.tier_manager import TierManager
from memtier_moe.memory.transfer_engine import TransferEngine
from memtier_moe.cache.lfu_cache import LFUExpertCache

logger = logging.getLogger(__name__)


class _Placeholder:
    """Lightweight stand-in for a tensor in simulation mode.
    
    Satisfies the duck-typed element_size()/numel() interface used by
    memory pools for size calculations.
    """
    def __init__(self, size_bytes: int):
        self._size = size_bytes
    def element_size(self): return 1
    def numel(self): return self._size
    def cpu(self): return self
    def cuda(self): return self
    def pin_memory(self): return self


@dataclass
class LayerResult:
    layer_idx: int
    hits: int
    misses: int
    transfer_time_ms: float


class InferenceEngine:
    """Coordinates caching, transferring, and execution of MoE layers."""

    def __init__(self, config: MemTierConfig):
        """Initialize the Inference Engine."""
        self.config = config
        self.metrics = MetricsTracker()
        self.tier_manager = TierManager(config, self.metrics)
        self.transfer_engine = TransferEngine(self.tier_manager, config, self.metrics)
        self.cache = LFUExpertCache(self.tier_manager, config, self.metrics)

    def setup_from_profile(
        self, expert_sizes: Dict[ExpertId, int], initial_frequencies: Optional[Dict[ExpertId, float]] = None
    ) -> None:
        """Setup experts based on profile.
        
        Args:
            expert_sizes: Map from expert ID to size in bytes.
            initial_frequencies: Map from expert ID to frequency score.
        """
        for eid, size in expert_sizes.items():
            self.tier_manager.register_expert(eid, size, MemoryTier.DRAM)
            # Store a placeholder in the pool so retrieve() works
            placeholder = _Placeholder(size)
            self.tier_manager.place_initial(eid, placeholder, MemoryTier.DRAM)
            
        if initial_frequencies:
            # Promote top-k experts to HBM
            sorted_experts = sorted(
                initial_frequencies.items(), key=lambda item: item[1], reverse=True
            )
            hbm_used = 0
            for eid, freq in sorted_experts:
                size = expert_sizes[eid]
                if hbm_used + size <= self.config.hbm_cache_budget_bytes:
                    self.tier_manager.promote(eid)
                    hbm_used += size
                else:
                    break
                    
            self.cache.warm_start(initial_frequencies)

    def forward_moe_layer(self, layer_idx: int, hidden_states: Any, router_output: List[int]) -> LayerResult:
        """Execute a single MoE layer.
        
        Args:
            layer_idx: The index of the layer.
            hidden_states: Input to the layer.
            router_output: List of expert indices selected by the router.
            
        Returns:
            LayerResult with execution statistics.
        """
        expert_ids = [(layer_idx, exp_idx) for exp_idx in router_output]
        hits, misses = self.cache.lookup(expert_ids)
        
        transfer_time_ms = 0.0
        
        # Hits are executed immediately
        for eid in hits:
            weights = self.cache.get_weights(eid)
            # Computation would happen here
            
        # Misses are fetched on demand
        for eid in misses:
            # For simulation, we assume demand_fetch blocks and returns the tensor
            tensor = self.transfer_engine.demand_fetch(eid)
            
            # Mock transfer time: 1ms per fetch
            transfer_time_ms += 1.0
            
            evicted = self.cache.insert(eid)
            for ev_id in evicted:
                self.tier_manager.demote(ev_id)
                
            # Computation would happen here
            
        self.cache.record_accesses(expert_ids)
        
        return LayerResult(
            layer_idx=layer_idx,
            hits=len(hits),
            misses=len(misses),
            transfer_time_ms=transfer_time_ms
        )

    def run_token(self, token_idx: int, layer_outputs: List[Any]) -> Dict[str, Any]:
        """Run all layers for a single token."""
        self.metrics.record_token()
        layer_stats = []
        
        # layer_outputs would contain routing decisions in a real system
        for layer_idx, router_out in enumerate(layer_outputs):
            res = self.forward_moe_layer(layer_idx, None, router_out)
            layer_stats.append(res)
            
        return {"token_idx": token_idx, "layer_stats": layer_stats}

    def run_batch(self, num_tokens: int, router_decisions: List[List[List[int]]]) -> Dict[str, Any]:
        """Run multiple tokens with pre-recorded routing decisions."""
        for token_idx in range(num_tokens):
            self.run_token(token_idx, router_decisions[token_idx])
            
        return self.report()

    def report(self) -> Dict[str, Any]:
        """Return engine metrics."""
        return {
            "metrics": self.metrics.report(),
            "hbm_usage": self.tier_manager.hbm_usage(),
            "cache_size": self.cache.size()
        }

    def reset(self) -> None:
        """Reset engine state."""
        self.metrics.reset()
