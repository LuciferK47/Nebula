"""Live router interceptor for feeding routing decisions to the prefetcher.

During inference, this module hooks into the MoE router to capture
real-time routing decisions and immediately feed them to the prefetch
scheduler.  It also records decisions for post-hoc analysis.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from memtier_moe.core.types import ExpertId
from memtier_moe.introspect.gate_utils import extract_topk_routing
from memtier_moe.prefetch.prefetch_scheduler import PrefetchScheduler

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RouterInterceptor:
    """Intercepts MoE routing decisions and drives the prefetch pipeline.

    Can operate in two modes:
    1. **Live mode** — hooks into a real model's gate modules
    2. **Replay mode** — replays pre-recorded routing decisions

    Parameters
    ----------
    scheduler : PrefetchScheduler
        The prefetch scheduler to feed predictions into.
    num_layers : int
        Number of MoE layers in the model.
    top_k : int
        Number of experts selected per token per layer.
    """

    def __init__(
        self,
        scheduler: PrefetchScheduler,
        num_layers: int = 24,
        top_k: int = 4,
    ) -> None:
        self.scheduler = scheduler
        self.num_layers = num_layers
        self.top_k = top_k
        self._hooks: List[Any] = []
        self._current_token: int = 0

    # ── Live mode ─────────────────────────────────────────────────────

    def attach_to_model(self, model: Any) -> int:
        """Register forward hooks on the model's gate modules.

        Returns the number of hooks registered.
        """
        if not HAS_TORCH:
            logger.warning("torch not available — cannot attach hooks")
            return 0

        count = 0
        layer_idx = 0
        for name, module in model.named_modules():
            if "block_sparse_moe.gate" in name or "mlp.gate" in name:
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._hooks.append(hook)
                layer_idx += 1
                count += 1

        logger.info(f"RouterInterceptor: attached {count} hooks")
        return count

    def detach(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("RouterInterceptor: all hooks removed")

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that feeds routing to the scheduler."""
        def hook_fn(module: Any, inputs: Any, outputs: Any) -> None:
            try:
                _, indices = extract_topk_routing(outputs, self.top_k)
                if HAS_TORCH and isinstance(indices, torch.Tensor):
                    # indices shape: (batch_size, top_k)
                    for row in indices:
                        experts = row.detach().cpu().tolist()
                        self.scheduler.on_routing_decision(
                            layer_idx, experts
                        )
            except Exception as e:
                logger.warning(f"Hook error at layer {layer_idx}: {e}")

        return hook_fn

    # ── Replay mode ───────────────────────────────────────────────────

    def replay_decisions(
        self,
        routing_decisions: List[List[List[int]]],
    ) -> Dict[str, Any]:
        """Replay pre-recorded routing decisions through the prefetcher.

        Parameters
        ----------
        routing_decisions : list[list[list[int]]]
            Shape: [num_tokens][num_layers][top_k_experts]
            Each entry is the list of expert indices for that token/layer.

        Returns
        -------
        Dict with replay statistics.
        """
        num_tokens = len(routing_decisions)
        total_prefetched = 0

        for token_idx, token_layers in enumerate(routing_decisions):
            self._current_token = token_idx
            for layer_idx, experts in enumerate(token_layers):
                # Feed to scheduler
                prefetched = self.scheduler.on_routing_decision(
                    layer_idx, experts
                )
                total_prefetched += len(prefetched)

                # Mark accessed experts as "used" for precision tracking
                for exp_idx in experts:
                    eid: ExpertId = (layer_idx, exp_idx)
                    self.scheduler.on_expert_accessed(eid)

            # Poll completed transfers between tokens
            self.scheduler.poll_and_complete()

        return {
            "tokens_replayed": num_tokens,
            "total_prefetched": total_prefetched,
            "scheduler_stats": self.scheduler.stats(),
        }

    def advance_token(self) -> None:
        """Signal that we are moving to the next token."""
        self._current_token += 1
