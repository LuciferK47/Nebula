"""Prefetch scheduler: queues predictions and submits transfers.

Sits between the predictor and the transfer engine.  Responsibilities:
- Receive predictions sorted by confidence
- Filter: skip HBM-resident, skip in-flight, respect max-inflight limit
- Submit top-N to the transfer engine as async_fetch calls
- Record outcomes (useful / wasted) for precision tracking
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.cache.lfu_cache import LFUExpertCache
from memtier_moe.memory.transfer_engine import TransferEngine
from memtier_moe.memory.tier_manager import TierManager
from memtier_moe.prefetch.predictor import ExpertPredictor, Prediction

logger = logging.getLogger(__name__)


class PrefetchScheduler:
    """Orchestrates prefetch transfers based on predictions.

    Parameters
    ----------
    predictor : ExpertPredictor
        The predictor that generates confidence-scored expert predictions.
    transfer_engine : TransferEngine
        The engine that actually moves data between tiers.
    tier_manager : TierManager
        For checking current expert locations.
    cache : LFUExpertCache
        The HBM cache. Without a reference here, a completed prefetch had
        no way to register itself as cached — ``cache.lookup()`` kept
        reporting a miss for an expert that had *already* been transferred
        into HBM in the background, defeating the entire point of
        prefetching, and the demand-fetch path would then race a second,
        redundant transfer against the still-in-flight one.
    config : MemTierConfig
        System configuration (max inflight, etc.).
    metrics : MetricsTracker
        For recording prefetch stats.
    """

    def __init__(
        self,
        predictor: ExpertPredictor,
        transfer_engine: TransferEngine,
        tier_manager: TierManager,
        cache: LFUExpertCache,
        config: MemTierConfig,
        metrics: MetricsTracker,
    ) -> None:
        self.predictor = predictor
        self.transfer_engine = transfer_engine
        self.tier_manager = tier_manager
        self.cache = cache
        self.config = config
        self.metrics = metrics

        # Track which experts were prefetched (to check usefulness later)
        self._prefetched: Set[ExpertId] = set()
        self._prefetch_count: int = 0
        self._useful_count: int = 0
        self._wasted_count: int = 0

    def on_routing_decision(
        self,
        layer_idx: int,
        selected_experts: List[int],
        pinned: Optional[Set[ExpertId]] = None,
    ) -> List[ExpertId]:
        """Called after each layer's routing decision.

        Generates predictions, submits prefetch transfers, and returns
        the list of expert IDs that were prefetched.

        Parameters
        ----------
        layer_idx : int
            The layer that just made its routing decision.
        selected_experts : list[int]
            Expert indices chosen by the router at this layer.
        pinned : set of ExpertId, optional
            Experts currently active and protected from eviction.

        Returns
        -------
        List of ExpertIds for which prefetch was initiated.
        """
        # Protect active experts of the current layer from eviction
        pinned_set: Set[ExpertId] = set(pinned) if pinned is not None else set()
        for exp_idx in selected_experts:
            pinned_set.add((layer_idx, exp_idx))

        # Determine what's already in HBM
        hbm_experts = set(self.tier_manager.experts_in_tier(MemoryTier.HBM))
        inflight = {
            eid for eid in self.transfer_engine._inflight
        }

        # Calculate space required for current layer's un-resident demand experts
        needed_by_current = sum(
            self.tier_manager.get_metadata(eid).size_bytes
            for eid in pinned_set
            if eid in self.tier_manager._registry
            and self.tier_manager.get_metadata(eid).current_tier != MemoryTier.HBM
        )

        hbm_pool = self.tier_manager.get_pool(MemoryTier.HBM)
        max_prefetches = getattr(self.config, "max_prefetches_per_decision", 2)

        # Get predictions
        predictions = self.predictor.predict(
            current_layer=layer_idx,
            current_experts=selected_experts,
            hbm_resident=hbm_experts,
            inflight=inflight,
        )

        # Submit prefetches (bounded by available transfer slots, headroom, and limits)
        prefetched = self._issue_prefetches(predictions, pinned_set, needed_by_current)

        # Flush stale predictions from past layers
        self.predictor.flush_stale(layer_idx)

        return prefetched

    def on_lookahead_decision(
        self,
        current_layer: int,
        target_layer: int,
        predicted_experts: List[int],
        confidences: Optional[List[float]] = None,
        pinned: Optional[Set[ExpertId]] = None,
    ) -> List[ExpertId]:
        """Schedule prefetching for upcoming layer based on lookahead pre-gating."""
        pinned_set: Set[ExpertId] = set(pinned) if pinned is not None else set()
        hbm_experts = set(self.tier_manager.experts_in_tier(MemoryTier.HBM))
        inflight = set(self.transfer_engine._inflight)

        needed_by_current = sum(
            self.tier_manager.get_metadata(eid).size_bytes
            for eid in pinned_set
            if eid in self.tier_manager._registry
            and self.tier_manager.get_metadata(eid).current_tier != MemoryTier.HBM
        )

        predictions = []
        for i, exp_idx in enumerate(predicted_experts):
            if (target_layer, exp_idx) in hbm_experts or (target_layer, exp_idx) in inflight:
                continue
            conf = confidences[i] if confidences is not None and i < len(confidences) else 0.85
            predictions.append(
                Prediction(
                    expert_id=(target_layer, exp_idx),
                    confidence=conf,
                    source_layer=current_layer,
                    target_layer=target_layer,
                )
            )
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        prefetched = self._issue_prefetches(predictions, pinned_set, needed_by_current)
        self.predictor.flush_stale(current_layer)
        return prefetched

    def _issue_prefetches(
        self,
        predictions: List[Prediction],
        pinned_set: Set[ExpertId],
        needed_by_current: int,
    ) -> List[ExpertId]:
        """Internal helper to validate headroom and submit async prefetch requests."""
        hbm_pool = self.tier_manager.get_pool(MemoryTier.HBM)
        max_prefetches = getattr(self.config, "max_prefetches_per_decision", 2)
        prefetched: List[ExpertId] = []

        for pred in predictions:
            if len(prefetched) >= max_prefetches:
                break
            if not self.transfer_engine.can_accept_transfer():
                break

            eid = pred.expert_id
            try:
                meta = self.tier_manager.get_metadata(eid)
            except KeyError:
                continue

            if meta.current_tier == MemoryTier.HBM:
                continue

            free_after_current = hbm_pool.free_bytes() - needed_by_current
            if free_after_current < meta.size_bytes:
                evictable_bytes = sum(
                    ce.metadata.size_bytes
                    for ce in self.cache._cache.values()
                    if ce.expert_id not in pinned_set
                )
                if evictable_bytes + free_after_current < meta.size_bytes:
                    continue
                if pred.confidence < max(0.20, self.config.prefetch_confidence_threshold):
                    continue

                # Adaptive Prefetch Gating:
                # Under memory pressure where free headroom is deficient, avoid speculative thrashing!
                # Guard experts that were accessed in the current token or have above-median frequency.
                if getattr(self.config, "adaptive_prefetch_gating", True):
                    candidates = [ce.metadata for ce in self.cache._cache.values() if ce.expert_id not in pinned_set]
                    if not candidates:
                        continue
                    current_tok = getattr(self.cache, "_current_token", 0)
                    half_life = getattr(getattr(self.cache, "eviction_policy", None), "half_life", 500)
                    candidates_sorted = sorted(candidates, key=lambda m: m.decay_frequency(current_tok, half_life))
                    prime_victim = candidates_sorted[0]
                    victim_freq = prime_victim.decay_frequency(current_tok, half_life)
                    median_freq = candidates_sorted[len(candidates_sorted) // 2].decay_frequency(current_tok, half_life)

                    if (prime_victim.last_access_token == current_tok and prime_victim.raw_count > 0) or (victim_freq > 2.0 and victim_freq >= median_freq):
                        logger.debug(
                            f"Adaptive prefetch throttled: victim {prime_victim.expert_id} is active/hot "
                            f"(freq={victim_freq:.2f}, median={median_freq:.2f}). Preserving cache."
                        )
                        continue

            self.cache.make_room(meta.size_bytes, pinned=pinned_set)
            self.transfer_engine.async_fetch(eid)
            self._prefetched.add(eid)
            self._prefetch_count += 1
            self.metrics.counters["prefetch_issued"] += 1
            prefetched.append(eid)

            logger.debug(
                f"Prefetch scheduled: {eid} (conf={pred.confidence:.3f}, "
                f"target_layer={pred.target_layer})"
            )

        return prefetched

    def on_expert_accessed(self, expert_id: ExpertId) -> None:
        """Called when an expert is actually used — marks prefetch as useful."""
        if expert_id in self._prefetched:
            self._prefetched.discard(expert_id)
            self._useful_count += 1
            self.predictor.record_access(expert_id)

    def poll_and_complete(self) -> List[ExpertId]:
        """Poll the transfer engine for completed prefetches, register each
        as cached (this is the step that used to be missing entirely — a
        completed prefetch never became a cache hit), and resync
        MetricsTracker's useful/wasted counters with this scheduler's own
        tally.

        The useful/wasted split isn't known at issue time — a prefetch is
        only "wasted" once we know it was never subsequently accessed,
        which ``self.prefetch_precision`` already derives correctly from
        ``_prefetch_count`` and ``_useful_count``. This mirrors that same
        derivation into ``MetricsTracker`` so ``engine.report()`` agrees
        with ``scheduler.stats()`` instead of the counters living their
        own, permanently-zero life.
        """
        done = self.transfer_engine.poll_completed()
        for eid in done:
            self.cache.insert(eid)
        self.metrics.counters["prefetch_useful"] = self._useful_count
        self.metrics.counters["prefetch_wasted"] = self._wasted_remaining
        return done

    @property
    def prefetch_precision(self) -> float:
        """Fraction of prefetches that were actually used."""
        total = self._useful_count + self._wasted_remaining
        return self._useful_count / total if total > 0 else 0.0

    @property
    def _wasted_remaining(self) -> int:
        return self._prefetch_count - self._useful_count

    def stats(self) -> Dict[str, float]:
        return {
            "prefetch_total": self._prefetch_count,
            "prefetch_useful": self._useful_count,
            "prefetch_pending": len(self._prefetched),
            "prefetch_precision": self.prefetch_precision,
            "predictor": self.predictor.stats(),
        }
