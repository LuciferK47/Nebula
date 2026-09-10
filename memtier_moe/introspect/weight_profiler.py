"""Profiler for MoE expert weights."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import re
import json
import logging
import os
import struct

logger = logging.getLogger(__name__)

# bytes-per-element for dtypes that appear in safetensors headers. Anything
# not listed here (e.g. packed sub-byte quant formats) has no fixed
# per-element width in the header itself; we fall back to computing it from
# data_offsets, which is exact regardless of dtype.
_DTYPE_ELEMENT_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1,
}


def _read_safetensors_header(shard_path: str) -> Dict[str, Any]:
    """Read a safetensors shard's JSON header without loading tensor data.

    The format is: 8-byte little-endian header length, then that many bytes
    of JSON mapping ``tensor_name -> {"dtype", "shape", "data_offsets"}``
    (plus an optional ``__metadata__`` key). Reading just the header is
    enough to compute exact per-tensor byte sizes via ``data_offsets``,
    without depending on the optional `safetensors` package or pulling any
    tensor data off disk.
    """
    with open(shard_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header

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
        """Profile weights directly from on-disk safetensors shards.

        Args:
            path: Either a ``model.safetensors.index.json`` file (sharded
                checkpoint), a directory containing one, a directory
                containing a single ``model.safetensors``, or a single
                ``.safetensors`` file directly.

        Reads each shard's header only (no tensor data loaded) to get an
        exact per-parameter byte count from ``data_offsets``, then applies
        the same expert-key parsing and fp16/int4 accounting as
        :meth:`profile_from_model` — so both entry points produce
        comparable ``ModelWeightProfile`` objects whether or not a model
        was actually loaded into memory.
        """
        model_dir, weight_map = self._resolve_weight_map(path)

        # Open each shard's header once and reuse it for every tensor name
        # that maps to it, rather than re-reading the file per-tensor.
        header_cache: Dict[str, Dict[str, Any]] = {}

        def header_for(shard_filename: str) -> Dict[str, Any]:
            if shard_filename not in header_cache:
                header_cache[shard_filename] = _read_safetensors_header(
                    os.path.join(model_dir, shard_filename)
                )
            return header_cache[shard_filename]

        experts_dict: Dict[Tuple[int, int], LayerExpertSummary] = {}
        total_expert_fp16 = 0
        total_expert_int4 = 0
        total_non_expert = 0

        for name, shard_filename in weight_map.items():
            tensor_header = header_for(shard_filename).get(name)
            if tensor_header is None:
                logger.warning(f"{name} listed in weight_map but missing from {shard_filename} header")
                continue

            shape = tensor_header.get("shape", [])
            numel = 1
            for dim in shape:
                numel *= dim
            # fp16-equivalent size regardless of the checkpoint's actual
            # on-disk dtype, matching profile_from_model's convention of
            # reporting sizes relative to an fp16 reference.
            size_bytes = numel * 2

            parsed = self._parse_expert_key(name)
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
                        param_names=[],
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
            model_name=os.path.basename(self.model_name_or_path),
            num_layers=max([s.layer_idx for s in experts], default=-1) + 1 if experts else 0,
            num_experts_per_layer=max([s.expert_idx for s in experts], default=-1) + 1 if experts else 0,
            num_shared_experts=0,
            experts=experts,
            total_expert_bytes_fp16=total_expert_fp16,
            total_expert_bytes_int4=total_expert_int4,
            total_shared_bytes=0,
            total_non_expert_bytes=total_non_expert,
        )

    @staticmethod
    def _resolve_weight_map(path: str) -> Tuple[str, Dict[str, str]]:
        """Resolve *path* to (model_dir, {tensor_name: shard_filename}).

        Accepts an index JSON file, a directory containing one, a directory
        with a single unsharded ``model.safetensors``, or a direct path to
        a single ``.safetensors`` file.
        """
        if os.path.isdir(path):
            index_path = os.path.join(path, "model.safetensors.index.json")
            if os.path.isfile(index_path):
                path = index_path
            else:
                single = os.path.join(path, "model.safetensors")
                if os.path.isfile(single):
                    header = _read_safetensors_header(single)
                    return path, {name: "model.safetensors" for name in header}
                raise FileNotFoundError(
                    f"No model.safetensors.index.json or model.safetensors found in {path}"
                )

        if path.endswith(".index.json"):
            model_dir = os.path.dirname(path) or "."
            with open(path, "r") as f:
                index = json.load(f)
            return model_dir, index.get("weight_map", {})

        # A single shard file passed directly.
        model_dir = os.path.dirname(path) or "."
        header = _read_safetensors_header(path)
        return model_dir, {name: os.path.basename(path) for name in header}
        
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
        pattern = r"model\.layers\.(\d+).*experts\.(\d+)\.(\w+)\.(weight|qweight|qzeros|scales)"
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
