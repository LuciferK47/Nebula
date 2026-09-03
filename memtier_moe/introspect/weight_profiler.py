"""Profiler for MoE expert weights."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import re
import json
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class ExpertWeightInfo:
    """Information about a specific expert parameter."""
    layer_idx: int
    expert_idx: int
    param_name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int

@dataclass
class LayerExpertSummary:
    """Summary of all parameters for a specific expert."""
    layer_idx: int
    expert_idx: int
    total_size_fp16: int
    total_size_int4: int
    param_count: int
    param_names: List[str]

@dataclass
class ModelWeightProfile:
    """Complete weight profile of a model's experts."""
    model_name: str
    num_layers: int
    num_experts_per_layer: int
    num_shared_experts: int
    experts: List[LayerExpertSummary]
    total_expert_bytes_fp16: int
    total_expert_bytes_int4: int
    total_shared_bytes: int
    total_non_expert_bytes: int

class WeightProfiler:
    """Profiles model weights to extract expert size and structure."""
    
    def __init__(self, model_name_or_path: str) -> None:
        """Initialize with model name or path."""
        self.model_name_or_path = model_name_or_path
        
    def profile_from_safetensors(self, path: str) -> ModelWeightProfile:
        """Parse safetensors index JSON to profile weights."""
        with open(path, 'r') as f:
            index = json.load(f)
            
        weight_map = index.get("weight_map", {})
        
        return ModelWeightProfile(
            model_name=os.path.basename(self.model_name_or_path),
            num_layers=0,
            num_experts_per_layer=0,
            num_shared_experts=0,
            experts=[],
            total_expert_bytes_fp16=0,
            total_expert_bytes_int4=0,
            total_shared_bytes=0,
            total_non_expert_bytes=0
        )
        
    def profile_from_model(self, model: Any) -> ModelWeightProfile:
        """Walk a loaded model's parameters to profile weights."""
        experts_dict: Dict[Tuple[int, int], LayerExpertSummary] = {}
        total_expert_fp16 = 0
        total_expert_int4 = 0
        total_shared = 0
        total_non_expert = 0
        
        for name, param in model.named_parameters():
            parsed = self._parse_expert_key(name)
            size_bytes = param.numel() * param.element_size()
            
            if parsed:
                layer_idx, expert_idx, param_name = parsed
                key = (layer_idx, expert_idx)
                
                if key not in experts_dict:
                    experts_dict[key] = LayerExpertSummary(
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        total_size_fp16=0,
                        total_size_int4=0,
                        param_count=0,
                        param_names=[]
                    )
                
                summary = experts_dict[key]
                summary.total_size_fp16 += size_bytes
                summary.total_size_int4 += self._compute_int4_size(size_bytes)
                summary.param_count += 1
                summary.param_names.append(param_name)
                
                total_expert_fp16 += size_bytes
                total_expert_int4 += self._compute_int4_size(size_bytes)
            else:
                total_non_expert += size_bytes
                
        experts = list(experts_dict.values())
        
        return ModelWeightProfile(
            model_name=str(model.__class__.__name__),
            num_layers=max([s.layer_idx for s in experts], default=-1) + 1 if experts else 0,
            num_experts_per_layer=max([s.expert_idx for s in experts], default=-1) + 1 if experts else 0,
            num_shared_experts=0,
            experts=experts,
            total_expert_bytes_fp16=total_expert_fp16,
            total_expert_bytes_int4=total_expert_int4,
            total_shared_bytes=total_shared,
            total_non_expert_bytes=total_non_expert
        )
        
    def _parse_expert_key(self, key: str) -> Optional[Tuple[int, int, str]]:
        """Parse expert parameter key to extract indices and name."""
        pattern = r"model\.layers\.(\d+).*experts\.(\d+)\.(\w+)\.weight"
        match = re.search(pattern, key)
        if match:
            return (int(match.group(1)), int(match.group(2)), match.group(3))
        return None
        
    def _compute_int4_size(self, fp16_size: int) -> int:
        """Compute INT4 size given FP16 size."""
        return fp16_size // 4
        
    def summarize(self) -> str:
        """Pretty-print the profile."""
        return f"WeightProfiler({self.model_name_or_path})"
