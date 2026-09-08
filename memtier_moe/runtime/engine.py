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
        self.cache = LFUExpertCache(self.tier_manager, config, self.metrics)
        # ensure_room_fn: a completed async transfer (wait_for, driven by
        # the background poll loop) needs a fresh room check right before
        # it stores into HBM — space reserved at prefetch issue time can
        # be claimed by an intervening demand-fetch before completion.
        self.transfer_engine = TransferEngine(
            self.tier_manager, config, self.metrics, ensure_room_fn=self.cache.make_room
        )
        # Tracks logical token index for cache decay. A MoE forward pass
        # visits many layers per token, so we tick this once per token
        # (detected via layer_idx == 0) rather than once per layer — the
        # eviction half-life is defined in tokens, not layer-visits.
        self._current_token_idx: int = -1

    def setup_from_profile(
        self, expert_sizes: Dict[ExpertId, int], initial_frequencies: Optional[Dict[ExpertId, float]] = None
    ) -> None:
        """Setup experts based on profile.

        Args:
            expert_sizes: Map from expert ID to size in bytes.
            initial_frequencies: Map from expert ID to frequency score.
        """
        # DRAM is the default staging tier, but a config with no DRAM
        # capacity at all (host_dram_bytes == 0) means "no offload tier —
        # everything lives in HBM", as the GPU-Resident baseline expresses
        # it. Staging into DRAM there isn't just wrong, it's impossible:
        # the very first placement fails against a zero-capacity pool.
        initial_tier = MemoryTier.DRAM if self.config.host_dram_bytes > 0 else MemoryTier.HBM

        for eid, size in expert_sizes.items():
            self.tier_manager.register_expert(eid, size, initial_tier)
            # Store a placeholder in the pool so retrieve() works
            placeholder = _Placeholder(size)
            self.tier_manager.place_initial(eid, placeholder, initial_tier)

        if initial_tier == MemoryTier.HBM:
            # Every expert just landed directly in HBM. Register all of
            # them as cached — not only the ones covered by
            # initial_frequencies — or lookup() would report a miss for
            # something that's already sitting in the pool.
            for eid in expert_sizes:
                self.cache.insert(eid)
            if initial_frequencies:
                self.cache.warm_start(initial_frequencies)
            return

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
        # A MoE forward pass restarts at layer 0 for every new token, so use
        # that as the token boundary signal (see __init__ for why this
        # can't just be an increment-per-call in the cache itself).
        if layer_idx == 0:
            self._current_token_idx += 1
            self.cache.set_token(self._current_token_idx)

        expert_ids = [(layer_idx, exp_idx) for exp_idx in router_output]
        hits, misses = self.cache.lookup(expert_ids)

        transfer_time_ms = 0.0

        # Hits are executed immediately
        for eid in hits:
            weights = self.cache.get_weights(eid)
            # Computation would happen here

        # Misses are fetched on demand — or, if a prefetch already has this
        # expert in flight, waited on instead of racing a second fetch.
        # ensure_resident() also frees HBM room before storing, which the
        # transfer engine itself does not do.
        for eid in misses:
            before_ms = self.metrics.counters["total_transfer_time_ms"]
            self.cache.ensure_resident(eid, self.transfer_engine)
            transfer_time_ms += self.metrics.counters["total_transfer_time_ms"] - before_ms

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
