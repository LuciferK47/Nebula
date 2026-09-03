"""Co-occurrence analysis extracted from routing traces.

Builds cross-layer expert co-occurrence statistics: given that expert E
was selected at layer L, what is the conditional probability that expert
E' is selected at layer L+k?  These statistics power the predictive
prefetcher.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from memtier_moe.core.types import ExpertId

logger = logging.getLogger(__name__)


@dataclass
class CoOccurrenceStats:
    """Aggregated co-occurrence statistics for expert pairs across layers."""

    # joint_counts[(src_layer, src_expert, dst_layer, dst_expert)] = count
    joint_counts: Dict[Tuple[int, int, int, int], int] = field(default_factory=lambda: defaultdict(int))

    # marginal_counts[(layer, expert)] = count  (how often each expert is selected)
    marginal_counts: Dict[Tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))

    num_tokens: int = 0
    num_layers: int = 0
    lookahead: int = 2


def extract_co_occurrences(
    token_indices: np.ndarray,
    layer_indices: np.ndarray,
    expert_ids: np.ndarray,
    num_layers: int,
    lookahead: int = 2,
) -> CoOccurrenceStats:
    """Extract co-occurrence counts from raw routing arrays.

    Parameters
    ----------
    token_indices : (N,) array of token indices
    layer_indices : (N,) array of layer indices
    expert_ids    : (N, top_k) array of selected expert indices
    num_layers    : total number of layers in the model
    lookahead     : how many layers ahead to track co-occurrence

    Returns
    -------
    CoOccurrenceStats with populated joint and marginal counts.
    """
    stats = CoOccurrenceStats(num_layers=num_layers, lookahead=lookahead)

    # Group decisions by token
    token_decisions: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
    for i in range(len(token_indices)):
        tok = int(token_indices[i])
        lay = int(layer_indices[i])
        experts = expert_ids[i].tolist() if hasattr(expert_ids[i], 'tolist') else list(expert_ids[i])
        token_decisions[tok][lay] = experts

    stats.num_tokens = len(token_decisions)

    for tok, layer_map in token_decisions.items():
        for src_layer, src_experts in layer_map.items():
            for src_exp in src_experts:
                stats.marginal_counts[(src_layer, src_exp)] += 1

                # Look ahead
                for offset in range(1, lookahead + 1):
                    dst_layer = src_layer + offset
                    if dst_layer in layer_map:
                        for dst_exp in layer_map[dst_layer]:
                            stats.joint_counts[
                                (src_layer, src_exp, dst_layer, dst_exp)
                            ] += 1

    logger.info(
        f"Extracted co-occurrences: {len(stats.joint_counts)} pairs "
        f"from {stats.num_tokens} tokens, lookahead={lookahead}"
    )
    return stats


def build_conditional_probabilities(
    stats: CoOccurrenceStats,
    min_threshold: float = 0.05,
) -> Dict[Tuple[int, int, int, int], float]:
    """Convert joint counts to conditional probabilities.

    P(dst_expert @ dst_layer | src_expert @ src_layer) =
        joint_count / marginal_count(src_layer, src_expert)

    Only pairs above *min_threshold* are retained (sparsity filter).

    Returns
    -------
    Dict mapping (src_layer, src_expert, dst_layer, dst_expert) → probability.
    """
    cond_probs: Dict[Tuple[int, int, int, int], float] = {}

    for key, joint in stats.joint_counts.items():
        src_layer, src_exp, dst_layer, dst_exp = key
        marginal = stats.marginal_counts.get((src_layer, src_exp), 0)
        if marginal == 0:
            continue

        prob = joint / marginal
        if prob >= min_threshold:
            cond_probs[key] = prob

    logger.info(
        f"Conditional probabilities: {len(cond_probs)} pairs "
        f"above threshold {min_threshold}"
    )
    return cond_probs
