"""Live model integration: wraps HuggingFace MoE models with tiered memory execution.

Replaces static expert residency with dynamic tiering (GPU HBM, host DRAM, CXL),
LFU caching, demand-fetching, and asynchronous predictive prefetching during
live autoregressive token generation.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.runtime.engine import InferenceEngine
from memtier_moe.prefetch.prefetch_scheduler import PrefetchScheduler
from memtier_moe.memory.pool import _tensor_size_bytes

logger = logging.getLogger(__name__)


class SingleMixtralExpert(nn.Module):
    """Encapsulates a single expert extracted from batched MixtralExperts tensor weights."""

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, act_fn: Any):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_proj.clone().detach())
        self.down_proj = nn.Parameter(down_proj.clone().detach())
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up_proj.device != x.device:
            self.to(x.device)
        gate, up = F.linear(x, self.gate_up_proj).chunk(2, dim=-1)
        return F.linear(self.act_fn(gate) * up, self.down_proj)


class TieredMoEBlock(nn.Module):
    """Wraps an MoE layer block to dynamically fetch/prefetch experts via MemTier-MoE."""

    def __init__(
        self,
        original_block: nn.Module,
        layer_idx: int,
        engine: InferenceEngine,
        num_experts: int,
        top_k: int = 2,
        scheduler: Optional[PrefetchScheduler] = None,
        next_gate: Optional[nn.Module] = None,
        execution_mode: str = "weight_transfer",
        enable_lookahead_gating: bool = False,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.layer_idx = layer_idx
        self.engine = engine
        self.num_experts = num_experts
        self.top_k = top_k
        self.scheduler = scheduler
        self.next_gate = next_gate
        self.execution_mode = execution_mode
        self.enable_lookahead_gating = enable_lookahead_gating

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Any:
        """Forward pass with tiered expert fetching."""
        # Drain transfers completed since previous layer
        if self.scheduler is not None:
            self.scheduler.poll_and_complete()

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        # 1. Compute Gate Routing
        if hasattr(self.original_block, "gate"):
            gate_out = self.original_block.gate(flat_hidden)
            if isinstance(gate_out, tuple) and len(gate_out) == 3:
                # Mixtral format: (_, top_k_weights, top_k_index)
                _, top_k_weights, selected_experts = gate_out
            else:
                # Qwen/Standard format: router_logits
                router_logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out
                probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
                top_k_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
                top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Block at layer {self.layer_idx} does not have a gate attribute")

        # 2. Extract active expert mask
        with torch.no_grad():
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        active_expert_indices = [int(idx[0]) for idx in expert_hit if int(idx[0]) < self.num_experts]
        expert_ids: List[ExpertId] = [(self.layer_idx, exp_idx) for exp_idx in active_expert_indices]
        expert_ids_set = set(expert_ids)

        # 3. Prefetching: Decoupled Lookahead Pre-Gating (SOTA) or Markovian transition
        if self.enable_lookahead_gating and self.next_gate is not None and self.scheduler is not None:
            with torch.no_grad():
                nxt_gate_out = self.next_gate(flat_hidden)
                from memtier_moe.introspect.gate_utils import extract_topk_routing
                _, nxt_indices = extract_topk_routing(nxt_gate_out, self.top_k)
                lookahead_experts = [int(idx) for idx in nxt_indices[0]]
            self.scheduler.on_lookahead_decision(
                current_layer=self.layer_idx,
                target_layer=self.layer_idx + 1,
                predicted_experts=lookahead_experts,
                pinned=expert_ids_set,
            )
            self.scheduler.poll_and_complete()
        elif self.scheduler is not None and active_expert_indices:
            self.scheduler.on_routing_decision(self.layer_idx, active_expert_indices, pinned=expert_ids_set)
            self.scheduler.poll_and_complete()

        # 4 & 5. Computation dispatch based on execution_mode
        final_hidden_states = torch.zeros_like(flat_hidden)

        if self.execution_mode == "hybrid":
            # SOTA Hybrid Compute (Activation Offload / Fiddler Mode):
            # Hot experts in HBM execute on GPU tensor cores.
            # Cold experts in Host DRAM execute on CPU via activation migration (8 KB vs 13.3 MB).
            for exp_idx in active_expert_indices:
                eid = (self.layer_idx, exp_idx)
                meta = self.engine.tier_manager.get_metadata(eid)

                top_k_pos, token_idx = torch.where(expert_mask[exp_idx])
                if token_idx.numel() == 0:
                    continue

                current_state = flat_hidden[token_idx]

                if meta.current_tier == MemoryTier.HBM:
                    # GPU Tensor Core Execution
                    self.engine.metrics.record_hit()
                    expert_layer = self.engine.cache.get_weights(eid)
                    current_hidden = expert_layer(current_state)
                else:
                    # Host CPU Activation Offload
                    self.engine.metrics.record_miss()
                    pool = self.engine.tier_manager.get_pool(meta.current_tier)
                    expert_layer = pool._store.get(eid)
                    if expert_layer is None:
                        expert_layer = self.engine.cache.get_weights(eid)

                    state_cpu = current_state.to("cpu")
                    out_cpu = expert_layer(state_cpu)
                    current_hidden = out_cpu.to(current_state.device, non_blocking=True)

                    # Track small activation payload
                    act_bytes = state_cpu.numel() * state_cpu.element_size() * 2
                    self.engine.metrics.counters["total_transfer_bytes"] += act_bytes

                current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

            self.engine.cache.record_accesses(expert_ids)
            if self.scheduler is not None:
                for eid in expert_ids:
                    self.scheduler.on_expert_accessed(eid)

        else:
            # Standard Weight Transfer Mode
            hits, misses = self.engine.cache.lookup(expert_ids)
            expert_modules: Dict[int, nn.Module] = {}

            for exp_idx in active_expert_indices:
                eid = (self.layer_idx, exp_idx)
                self.engine.cache.ensure_resident(eid, self.engine.transfer_engine, pinned=expert_ids_set)
                mod = self.engine.cache.get_weights(eid)
                expert_modules[exp_idx] = mod

            self.engine.cache.record_accesses(expert_ids)
            if self.scheduler is not None:
                for eid in expert_ids:
                    self.scheduler.on_expert_accessed(eid)

            for exp_idx in active_expert_indices:
                top_k_pos, token_idx = torch.where(expert_mask[exp_idx])
                if token_idx.numel() == 0:
                    continue

                current_state = flat_hidden[token_idx]
                expert_layer = expert_modules[exp_idx]
                current_hidden = expert_layer(current_state)
                current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

        # 6. Shared expert for Qwen architectures if present
        if hasattr(self.original_block, "shared_expert"):
            shared_out = self.original_block.shared_expert(flat_hidden)
            if hasattr(self.original_block, "shared_expert_gate"):
                shared_gate = torch.sigmoid(self.original_block.shared_expert_gate(flat_hidden))
                shared_out = shared_out * shared_gate
            final_hidden_states = final_hidden_states + shared_out

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class TieredMoEWrapper:
    """Wraps a HuggingFace MoE model for live tiered inference.

    Partitions the model:
      - Non-expert layers (embeddings, attention, norms, LM head) -> GPU VRAM
      - Expert layers -> Managed by TierManager (HBM, host DRAM, CXL)
    """

    def __init__(
        self,
        model: nn.Module,
        config: MemTierConfig,
        enable_prefetch: bool = False,
        co_occurrence_model: Optional[Any] = None,
        initial_hbm_budget_bytes: Optional[int] = None,
        execution_mode: str = "weight_transfer",
        enable_lookahead_gating: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.metrics = MetricsTracker()
        self.engine = InferenceEngine(config)
        self.co_occurrence_model = co_occurrence_model
        self.execution_mode = execution_mode
        self.enable_lookahead_gating = enable_lookahead_gating

        # Scheduler
        self.scheduler: Optional[PrefetchScheduler] = None
        if (enable_prefetch and co_occurrence_model is not None) or enable_lookahead_gating:
            from memtier_moe.prefetch.predictor import ExpertPredictor
            from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel
            predictor = ExpertPredictor(
                model=co_occurrence_model if co_occurrence_model is not None else CoOccurrenceModel(),
                confidence_threshold=config.prefetch_confidence_threshold,
            )
            self.scheduler = PrefetchScheduler(
                predictor=predictor,
                transfer_engine=self.engine.transfer_engine,
                tier_manager=self.engine.tier_manager,
                cache=self.engine.cache,
                config=config,
                metrics=self.metrics,
            )

        self._patch_moe_layers(initial_hbm_budget_bytes)

    def _patch_moe_layers(self, hbm_budget: Optional[int] = None) -> None:
        """Scan model, extract expert modules, and replace with TieredMoEBlock."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Place entire base model (non-experts) on GPU
        self.model.eval()
        self.model.to(device)

        hbm_budget = hbm_budget or self.config.hbm_cache_budget_bytes
        hbm_used = 0
        dram_budget = self.config.host_dram_bytes
        dram_used = 0

        layer_count = 0
        total_experts_registered = 0

        # Scan layers
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            layers = self.model.transformer.layers

        if layers is None:
            raise ValueError(f"Could not locate transformer layers in {type(self.model)}")

        # Collect next_gates across layers
        next_gates = {}
        for l_idx, layer in enumerate(layers):
            if l_idx + 1 < len(layers):
                nxt_moe = getattr(layers[l_idx + 1], "block_sparse_moe", None) or getattr(layers[l_idx + 1], "mlp", None)
                if nxt_moe is not None and hasattr(nxt_moe, "gate"):
                    next_gates[l_idx] = nxt_moe.gate

        for l_idx, layer in enumerate(layers):
            moe_block = getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
            attr_name = "block_sparse_moe" if hasattr(layer, "block_sparse_moe") else "mlp"

            if moe_block is None:
                continue

            expert_modules_list = []
            if hasattr(moe_block, "experts"):
                if isinstance(moe_block.experts, (nn.ModuleList, list)):
                    expert_modules_list = list(moe_block.experts)
                elif hasattr(moe_block.experts, "gate_up_proj"):
                    # Modern batched MixtralExperts tensor format
                    num_exp = getattr(moe_block.experts, "num_experts", moe_block.experts.gate_up_proj.shape[0])
                    act_fn = getattr(moe_block.experts, "act_fn", F.silu)
                    for e in range(num_exp):
                        exp_mod = SingleMixtralExpert(
                            moe_block.experts.gate_up_proj[e],
                            moe_block.experts.down_proj[e],
                            act_fn,
                        )
                        expert_modules_list.append(exp_mod)

            if not expert_modules_list:
                continue

            num_experts = len(expert_modules_list)
            top_k = getattr(moe_block, "num_experts_per_tok", getattr(moe_block, "top_k", 2))

            for exp_idx in range(num_experts):
                eid: ExpertId = (l_idx, exp_idx)
                expert_module = expert_modules_list[exp_idx]
                expert_size = _tensor_size_bytes(expert_module)

                # Initial placement: fill HBM up to budget, then DRAM, then CXL
                if hbm_used + expert_size <= hbm_budget:
                    initial_tier = MemoryTier.HBM
                    hbm_used += expert_size
                elif dram_used + expert_size <= dram_budget:
                    initial_tier = MemoryTier.DRAM
                    dram_used += expert_size
                else:
                    initial_tier = MemoryTier.CXL

                self.engine.tier_manager.register_expert(eid, expert_size, initial_tier)
                self.engine.tier_manager.place_initial(eid, expert_module, initial_tier)

                if initial_tier == MemoryTier.HBM:
                    self.engine.cache.insert(eid)

                total_experts_registered += 1

            # Replace moe_block with TieredMoEBlock
            tiered_block = TieredMoEBlock(
                original_block=moe_block,
                layer_idx=l_idx,
                engine=self.engine,
                num_experts=num_experts,
                top_k=top_k,
                scheduler=self.scheduler,
                next_gate=next_gates.get(l_idx),
                execution_mode=self.execution_mode,
                enable_lookahead_gating=self.enable_lookahead_gating,
            )
            setattr(layer, attr_name, tiered_block)
            layer_count += 1

        logger.info(
            f"TieredMoEWrapper: patched {layer_count} layers, "
            f"{total_experts_registered} experts registered. "
            f"HBM initial allocation: {hbm_used / 1e6:.1f} MB"
        )

    def generate(self, *args, **kwargs) -> Any:
        """Autoregressive generation pass-through."""
        with torch.no_grad():
            return self.model.generate(*args, **kwargs)

    def unpatch(self) -> None:
        """Restore original MoE blocks back to the base model."""
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            layers = self.model.transformer.layers

        if layers is not None:
            for layer in layers:
                attr_name = "block_sparse_moe" if hasattr(layer, "block_sparse_moe") else "mlp"
                curr = getattr(layer, attr_name, None)
                if isinstance(curr, TieredMoEBlock):
                    setattr(layer, attr_name, curr.original_block)
        logger.info("TieredMoEWrapper: unpatched all layers, original blocks restored.")

    def report(self) -> Dict[str, Any]:
        """Return runtime memory and cache metrics."""
        rep = self.engine.report()
        if self.scheduler is not None:
            rep["scheduler_stats"] = self.scheduler.stats()
        return rep
