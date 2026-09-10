"""Shared utilities for interpreting MoE gate module outputs.

The gate module in real MoE checkpoints (Mixtral's ``block_sparse_moe.gate``,
Qwen1.5/2-MoE's ``mlp.gate``) is a plain ``nn.Linear(hidden_dim, num_experts,
bias=False)``. Its forward output is a single tensor of *raw router logits*,
shape ``(N, num_experts)`` — softmax and top-k happen afterwards, inside the
parent MoE block's own forward, not inside the gate. A forward hook attached
to the gate therefore never sees a ``(weights, indices)`` tuple; it only ever
sees the logits.

``routing_tracer.py`` and ``router_interceptor.py`` both used to assume the
tuple shape, so a hook on a real model's gate fired on every token but never
matched the ``isinstance(outputs, tuple)`` check — routing traces came back
empty with no error. This module recomputes top-k here (softmax + topk),
which is what both Mixtral and Qwen1.5/2-MoE do internally for standard
top-k routing, so a hooked gate produces the same selection the model itself
would have made.
"""
from __future__ import annotations

from typing import Any, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def extract_topk_routing(outputs: Any, top_k: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Extract ``(gating_weights, expert_indices)`` from a gate module's output.

    Handles two shapes:
      1. Raw router logits — a single ``Tensor`` of shape ``(N, num_experts)``,
         the standard ``nn.Linear`` gate output. Softmax + top-k is applied
         here.
      2. An already-computed ``(weights, indices)`` tuple, for hooks attached
         to a higher-level dispatch call (or test mocks) that already did
         the top-k selection.

    Returns:
        ``(weights, indices)``, each of shape ``(N, top_k)``.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required to interpret gate outputs")

    if isinstance(outputs, tuple):
        if len(outputs) >= 3 and outputs[2].dtype in (torch.int32, torch.int64):
            # Mixtral style: (router_logits, router_scores, router_indices)
            weights, indices = outputs[1], outputs[2]
        elif len(outputs) >= 2 and outputs[1].dtype in (torch.int32, torch.int64):
            # Standard: (weights, indices)
            weights, indices = outputs[0], outputs[1]
        elif len(outputs) >= 1 and isinstance(outputs[0], torch.Tensor) and outputs[0].is_floating_point():
            logits = outputs[0]
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
            weights, indices = torch.topk(probs, top_k, dim=-1)
        else:
            weights, indices = outputs[0], outputs[1]

        if hasattr(weights, "ndim") and weights.ndim > 2:
            weights = weights.reshape(-1, top_k)
            indices = indices.reshape(-1, top_k)
        elif hasattr(weights, "ndim") and weights.ndim == 1:
            weights = weights.unsqueeze(0)
            indices = indices.unsqueeze(0)
        return weights, indices

    logits = outputs
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Unrecognized gate output type: {type(logits)!r}")

    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    weights, indices = torch.topk(probs, top_k, dim=-1)

    if weights.ndim > 2:
        weights = weights.reshape(-1, top_k)
        indices = indices.reshape(-1, top_k)
    elif weights.ndim == 1:
        weights = weights.unsqueeze(0)
        indices = indices.unsqueeze(0)

    return weights, indices
