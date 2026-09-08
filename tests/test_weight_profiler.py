"""Tests for the real WeightProfiler (previously this file tested a
locally copy-pasted regex that had drifted from the module — a bug in
its own right, since it could pass while the real parser was broken)."""
from __future__ import annotations
import json
import os
import struct

import pytest

from memtier_moe.introspect.weight_profiler import WeightProfiler, _read_safetensors_header


def _write_fake_shard(path: str, tensors: dict) -> None:
    """tensors: {name: (dtype, shape)} -> writes a minimal valid safetensors file."""
    dtype_bytes = {"F16": 2, "F32": 4, "I8": 1}
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * dtype_bytes[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)


def test_parse_expert_key():
    wp = WeightProfiler("dummy")
    assert wp._parse_expert_key("model.layers.0.block_sparse_moe.experts.5.w1.weight") == (0, 5, "w1")
    assert wp._parse_expert_key("model.layers.23.block_sparse_moe.experts.59.w3.weight") == (23, 59, "w3")
    assert wp._parse_expert_key("model.embed_tokens.weight") is None


def test_int4_size_calculation():
    wp = WeightProfiler("dummy")
    assert wp._compute_int4_size(1024) == 256


def test_safetensors_header_roundtrip(tmp_path):
    """The header parser should recover exact shapes and byte offsets
    without needing the `safetensors` package or loading tensor data."""
    shard = tmp_path / "model.safetensors"
    _write_fake_shard(str(shard), {"a": ("F16", (2, 3)), "b": ("F32", (4,))})

    header = _read_safetensors_header(str(shard))
    assert header["a"]["shape"] == [2, 3]
    assert header["a"]["data_offsets"] == [0, 12]  # 2*3*2 bytes
    assert header["b"]["data_offsets"] == [12, 28]  # 12 + 4*4 bytes
    assert "__metadata__" not in header


def test_profile_from_safetensors_groups_by_expert(tmp_path):
    """End-to-end: on-disk shard -> per-expert size aggregation, matching
    what profile_from_model computes from a live model's named_parameters()."""
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()
    _write_fake_shard(str(model_dir / "model.safetensors"), {
        "model.embed_tokens.weight": ("F16", (100, 16)),
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": ("F16", (8, 16)),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": ("F16", (16, 8)),
        "model.layers.0.block_sparse_moe.experts.1.w1.weight": ("F16", (8, 16)),
    })

    wp = WeightProfiler(str(model_dir))
    profile = wp.profile_from_safetensors(str(model_dir))

    assert profile.num_layers == 1
    assert profile.num_experts_per_layer == 2

    by_id = {(e.layer_idx, e.expert_idx): e for e in profile.experts}
    assert by_id[(0, 0)].param_count == 2
    assert by_id[(0, 0)].total_size_fp16 == 8 * 16 * 2 + 16 * 8 * 2  # 512
    assert by_id[(0, 0)].total_size_int4 == 512 // 4
    assert by_id[(0, 1)].total_size_fp16 == 8 * 16 * 2  # 256

    assert profile.total_expert_bytes_fp16 == 512 + 256
    assert profile.total_non_expert_bytes == 100 * 16 * 2  # embed_tokens


def test_profile_from_safetensors_with_index_json(tmp_path):
    """Sharded checkpoints ship a model.safetensors.index.json weight_map
    instead of a single file; the profiler should follow it across shards."""
    model_dir = tmp_path / "sharded_model"
    model_dir.mkdir()
    _write_fake_shard(str(model_dir / "shard-0.safetensors"), {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": ("F16", (4, 4)),
    })
    _write_fake_shard(str(model_dir / "shard-1.safetensors"), {
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": ("F16", (4, 4)),
    })
    index = {
        "weight_map": {
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": "shard-0.safetensors",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight": "shard-1.safetensors",
        }
    }
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    wp = WeightProfiler(str(model_dir))
    profile = wp.profile_from_safetensors(str(model_dir))

    assert profile.experts[0].param_count == 2
    assert profile.experts[0].total_size_fp16 == 4 * 4 * 2 * 2
