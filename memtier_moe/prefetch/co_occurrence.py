"""Cross-layer expert co-occurrence model.

Stores a sparse conditional probability table:
    P(dst_expert @ dst_layer | src_expert @ src_layer)

Built from real routing traces via ``introspect.analysis``.  The model
is queried at inference time by the predictor to identify experts
likely to be needed in upcoming layers.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from memtier_moe.core.types import ExpertId
from memtier_moe.introspect.analysis import (
    CoOccurrenceStats,
    build_conditional_probabilities,
)

logger = logging.getLogger(__name__)


class CoOccurrenceModel:
    """Sparse cross-layer expert co-occurrence model.

    Internally stores ``cond_probs[(src_layer, src_expert, dst_layer,
    dst_expert)] → float`` and provides efficient lookups for a given
    source-layer routing decision.

    Parameters
    ----------
    lookahead : int
        Maximum number of layers ahead to predict.
    min_probability : float
        Discard pairs below this threshold to save memory.
    """

    def __init__(self, lookahead: int = 2, min_probability: float = 0.05) -> None:
        self.lookahead = lookahead
        self.min_probability = min_probability

        # Main table
        self._cond_probs: Dict[Tuple[int, int, int, int], float] = {}

        # Inverted index for fast lookup:
        #   _by_source[(src_layer, src_expert)] → [(dst_layer, dst_expert, prob), ...]
        self._by_source: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = defaultdict(list)

    # ── Construction ──────────────────────────────────────────────────

    def build_from_stats(self, stats: CoOccurrenceStats) -> None:
        """Build the model from pre-computed co-occurrence statistics."""
        self._cond_probs = build_conditional_probabilities(
            stats, min_threshold=self.min_probability
        )
        self._rebuild_index()
        logger.info(
            f"CoOccurrenceModel built: {len(self._cond_probs)} entries, "
            f"lookahead={self.lookahead}"
        )

    def build_from_probabilities(
        self, probs: Dict[Tuple[int, int, int, int], float]
    ) -> None:
        """Load a pre-computed probability table directly."""
        self._cond_probs = {
            k: v for k, v in probs.items() if v >= self.min_probability
        }
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the inverted index for fast source-keyed lookups."""
        self._by_source.clear()
        for (sl, se, dl, de), prob in self._cond_probs.items():
            self._by_source[(sl, se)].append((dl, de, prob))

        # Sort each entry list by probability descending for early exit
        for key in self._by_source:
            self._by_source[key].sort(key=lambda x: x[2], reverse=True)

    # ── Query ─────────────────────────────────────────────────────────

    def predict(
        self,
        src_layer: int,
        src_experts: List[int],
        confidence_threshold: float = 0.0,
    ) -> List[Tuple[ExpertId, float]]:
        """Predict experts for upcoming layers given current routing.

        Parameters
        ----------
        src_layer : int
            The current layer index.
        src_experts : list[int]
            Expert indices selected at the current layer.
        confidence_threshold : float
            Minimum probability to include in results.

        Returns
        -------
        List of ((dst_layer, dst_expert), probability) sorted by
        probability descending.  Duplicates across source experts are
        merged by taking the max probability.
        """
        predictions: Dict[ExpertId, float] = {}

        for src_exp in src_experts:
            key = (src_layer, src_exp)
            if key not in self._by_source:
                continue

            for dst_layer, dst_exp, prob in self._by_source[key]:
                if prob < confidence_threshold:
                    break  # list is sorted descending, so we can stop
                eid = (dst_layer, dst_exp)
                # Merge: keep highest probability across source experts
                if eid not in predictions or prob > predictions[eid]:
                    predictions[eid] = prob

        # Sort by probability descending
        result = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return result

    def get_probability(
        self, src_layer: int, src_expert: int, dst_layer: int, dst_expert: int
    ) -> float:
        """Look up a specific conditional probability."""
        return self._cond_probs.get((src_layer, src_expert, dst_layer, dst_expert), 0.0)

    # ── Stats ─────────────────────────────────────────────────────────

    @property
    def num_entries(self) -> int:
        return len(self._cond_probs)

    @property
    def num_source_keys(self) -> int:
        return len(self._by_source)

    def summary(self) -> str:
        if not self._cond_probs:
            return "CoOccurrenceModel: empty (not built)"
        probs = list(self._cond_probs.values())
        return (
            f"CoOccurrenceModel: {self.num_entries} entries, "
            f"{self.num_source_keys} source keys, "
            f"prob range [{min(probs):.3f}, {max(probs):.3f}], "
            f"lookahead={self.lookahead}"
        )
