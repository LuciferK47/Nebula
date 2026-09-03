"""Tracer for MoE routing decisions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging
import numpy as np
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    """A single routing decision for a token."""
    token_idx: int
    layer_idx: int
    top_k_expert_ids: List[int]
    gating_weights: List[float]

@dataclass
class RoutingTrace:
    """Trace of routing decisions across layers and tokens."""
    model_name: str
    num_tokens: int
    num_layers: int
    decisions: List[RoutingDecision] = field(default_factory=list)
    
    def expert_frequency_histogram(self, layer_idx: Optional[int] = None) -> Dict[int, int]:
        """Count activations per expert, optionally filtered by layer."""
        hist: Dict[int, int] = defaultdict(int)
        for dec in self.decisions:
            if layer_idx is None or dec.layer_idx == layer_idx:
                for exp_id in dec.top_k_expert_ids:
                    hist[exp_id] += 1
        return dict(hist)
        
    def top_k_experts_by_frequency(self, k: int, layer_idx: Optional[int] = None) -> List[Tuple[int, int]]:
        """Return top-k experts by activation frequency."""
        hist = self.expert_frequency_histogram(layer_idx)
        sorted_experts = sorted(hist.items(), key=lambda x: x[1], reverse=True)
        return sorted_experts[:k]
        
    def save(self, path: str) -> None:
        """Save the trace to a .npz file."""
        token_indices = np.array([d.token_idx for d in self.decisions])
        layer_indices = np.array([d.layer_idx for d in self.decisions])
        expert_ids = np.array([d.top_k_expert_ids for d in self.decisions])
        weights = np.array([d.gating_weights for d in self.decisions])
        
        np.savez_compressed(
            path,
            model_name=self.model_name,
            num_tokens=self.num_tokens,
            num_layers=self.num_layers,
            token_idx=token_indices,
            layer_idx=layer_indices,
            expert_ids=expert_ids,
            gating_weights=weights
        )
        
    @classmethod
    def load(cls, path: str) -> RoutingTrace:
        """Load a trace from a .npz file."""
        data = np.load(path)
        
        token_indices = data['token_idx']
        layer_indices = data['layer_idx']
        expert_ids = data['expert_ids']
        gating_weights = data['gating_weights']
        
        decisions = []
        for i in range(len(token_indices)):
            decisions.append(RoutingDecision(
                token_idx=int(token_indices[i]),
                layer_idx=int(layer_indices[i]),
                top_k_expert_ids=expert_ids[i].tolist(),
                gating_weights=gating_weights[i].tolist()
            ))
            
        return cls(
            model_name=str(data['model_name']),
            num_tokens=int(data['num_tokens']),
            num_layers=int(data['num_layers']),
            decisions=decisions
        )

class RoutingTracer:
    """Tracer to capture MoE routing decisions during inference."""
    
    def __init__(self, model: torch.nn.Module, num_tokens: int = 1000) -> None:
        """Initialize the tracer with a model."""
        self.model = model
        self.num_tokens = num_tokens
        self.gate_modules = self._find_gate_modules(model)
        self.hooks: List[Any] = []
        self.decisions: List[RoutingDecision] = []
        self.token_offset = 0
        
    def _find_gate_modules(self, model: torch.nn.Module) -> Dict[int, torch.nn.Module]:
        """Find gating modules in the model."""
        gates = {}
        layer_idx = 0
        for name, module in model.named_modules():
            if 'block_sparse_moe.gate' in name or 'mlp.gate' in name:
                gates[layer_idx] = module
                layer_idx += 1
        return gates
        
    def _hook_fn(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""
        def hook(module: torch.nn.Module, inputs: Tuple, outputs: Tuple) -> None:
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                weights = outputs[0]
                indices = outputs[1]
                
                weights_np = weights.detach().cpu().numpy()
                indices_np = indices.detach().cpu().numpy()
                
                for i in range(weights_np.shape[0]):
                    self.decisions.append(RoutingDecision(
                        token_idx=self.token_offset + i,
                        layer_idx=layer_idx,
                        top_k_expert_ids=indices_np[i].tolist(),
                        gating_weights=weights_np[i].tolist()
                    ))
        return hook
        
    def _register_hooks(self) -> None:
        """Register forward hooks on gate modules."""
        for layer_idx, module in self.gate_modules.items():
            self.hooks.append(module.register_forward_hook(self._hook_fn(layer_idx)))
            
    def trace(self, input_ids: torch.Tensor) -> RoutingTrace:
        """Run a forward pass and trace routing decisions."""
        self.decisions = []
        self.token_offset = 0
        self._register_hooks()
        
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            
        trace = RoutingTrace(
            model_name=self.model.__class__.__name__,
            num_tokens=input_ids.numel(),
            num_layers=len(self.gate_modules),
            decisions=self.decisions
        )
        return trace
        
    def trace_dataset(self, dataset_name: str = 'wikitext', split: str = 'test', max_tokens: int = 1000) -> RoutingTrace:
        """Convenience method to trace on a dataset."""
        logger.info(f"Tracing dataset {dataset_name} split {split} max_tokens {max_tokens}")
        return RoutingTrace(self.model.__class__.__name__, max_tokens, len(self.gate_modules))
