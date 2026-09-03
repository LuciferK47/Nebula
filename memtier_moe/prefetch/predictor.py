"""Confidence-scored expert predictor.

Wraps the co-occurrence model and adds:
- Rolling precision tracking (was the prefetched expert actually used?)
- Adaptive confidence threshold (stretch goal, fixed for M3 demo)
- Filtering of already-resident experts
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single prefetch prediction."""
    expert_id: ExpertId
    confidence: float
    source_layer: int
    target_layer: int


@dataclass
class PredictionOutcome:
    """Tracks whether a prediction was useful (the expert was actually accessed)."""
    prediction: Prediction
    was_useful: bool = False


class ExpertPredictor:
    """Predicts which experts will be needed in upcoming layers.

    Uses the co-occurrence model to generate predictions, filters them
    against currently resident experts, and tracks precision over a
    rolling window for monitoring.

    Parameters
    ----------
    model : CoOccurrenceModel
        The trained co-occurrence model.
    confidence_threshold : float
        Minimum confidence to issue a prediction (default 0.3).
    max_predictions_per_layer : int
        Cap on predictions per source-layer routing event.
    precision_window : int
        Rolling window size for precision tracking.
    """

    def __init__(
        self,
        model: CoOccurrenceModel,
        confidence_threshold: float = 0.3,
        max_predictions_per_layer: int = 8,
        precision_window: int = 200,
    ) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_predictions_per_layer = max_predictions_per_layer

        # Rolling precision tracking
        self._outcomes: Deque[PredictionOutcome] = deque(maxlen=precision_window)
        self._pending: Dict[ExpertId, PredictionOutcome] = {}

        # Counters
        self.total_predictions: int = 0
        self.total_useful: int = 0
        self.total_wasted: int = 0

    def predict(
        self,
        current_layer: int,
        current_experts: List[int],
        hbm_resident: Set[ExpertId],
        inflight: Set[ExpertId],
    ) -> List[Prediction]:
        """Generate predictions for upcoming layers.

        Parameters
        ----------
        current_layer : int
            Layer that just finished routing.
        current_experts : list[int]
            Expert indices selected at the current layer.
        hbm_resident : set[ExpertId]
            Experts already in HBM (no need to prefetch).
        inflight : set[ExpertId]
            Experts already being transferred (skip duplicates).

        Returns
        -------
        List of Prediction objects, sorted by confidence descending.
        """
        raw = self.model.predict(
            src_layer=current_layer,
            src_experts=current_experts,
            confidence_threshold=self.confidence_threshold,
        )

        predictions: List[Prediction] = []
        for (dst_layer, dst_expert), prob in raw:
            eid: ExpertId = (dst_layer, dst_expert)

            # Skip if already in HBM or in-flight
            if eid in hbm_resident or eid in inflight:
                continue

            pred = Prediction(
                expert_id=eid,
                confidence=prob,
                source_layer=current_layer,
                target_layer=dst_layer,
            )
            predictions.append(pred)

            if len(predictions) >= self.max_predictions_per_layer:
                break

        # Track pending predictions
        for pred in predictions:
            outcome = PredictionOutcome(prediction=pred)
            self._pending[pred.expert_id] = outcome
            self.total_predictions += 1

        if predictions:
            logger.debug(
                f"Predictor @ layer {current_layer}: "
                f"{len(predictions)} predictions "
                f"(top conf={predictions[0].confidence:.3f})"
            )

        return predictions

    def record_access(self, expert_id: ExpertId) -> None:
        """Mark a prediction as useful if the expert was actually accessed."""
        if expert_id in self._pending:
            outcome = self._pending.pop(expert_id)
            outcome.was_useful = True
            self._outcomes.append(outcome)
            self.total_useful += 1

    def flush_stale(self, current_layer: int) -> None:
        """Flush predictions for layers that have already passed (wasted)."""
        stale_keys = [
            eid for eid, outcome in self._pending.items()
            if outcome.prediction.target_layer <= current_layer
        ]
        for eid in stale_keys:
            outcome = self._pending.pop(eid)
            outcome.was_useful = False
            self._outcomes.append(outcome)
            self.total_wasted += 1

    def rolling_precision(self) -> float:
        """Precision over the rolling window."""
        if not self._outcomes:
            return 0.0
        useful = sum(1 for o in self._outcomes if o.was_useful)
        return useful / len(self._outcomes)

    def lifetime_precision(self) -> float:
        """Precision over all predictions ever made."""
        total = self.total_useful + self.total_wasted
        return self.total_useful / total if total > 0 else 0.0

    def stats(self) -> Dict[str, float]:
        return {
            "total_predictions": self.total_predictions,
            "total_useful": self.total_useful,
            "total_wasted": self.total_wasted,
            "pending": len(self._pending),
            "rolling_precision": self.rolling_precision(),
            "lifetime_precision": self.lifetime_precision(),
        }
