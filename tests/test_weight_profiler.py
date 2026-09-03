"""Tests for weight profiler."""
from __future__ import annotations
import pytest
import re

def parse_expert_key(key: str):
    pattern = r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w\d)\.weight"
    match = re.match(pattern, key)
    if match:
        return (int(match.group(1)), int(match.group(2)), match.group(3))
    return None

def test_parse_expert_key():
    assert parse_expert_key("model.layers.0.block_sparse_moe.experts.5.w1.weight") == (0, 5, "w1")
    assert parse_expert_key("model.layers.23.block_sparse_moe.experts.59.w3.weight") == (23, 59, "w3")
    assert parse_expert_key("model.embed_tokens.weight") is None

def test_int4_size_calculation():
    fp16_size = 1024
    int4_size = fp16_size // 4
    assert int4_size == 256

def test_grouping_by_expert():
    keys_sizes = [
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", 100),
        ("model.layers.0.block_sparse_moe.experts.0.w2.weight", 150),
        ("model.layers.0.block_sparse_moe.experts.1.w1.weight", 100),
    ]
    
    grouped = {}
    for key, size in keys_sizes:
        parsed = parse_expert_key(key)
        if parsed:
            layer, expert, w_type = parsed
            expert_id = (layer, expert)
            if expert_id not in grouped:
                grouped[expert_id] = 0
            grouped[expert_id] += size
            
    assert grouped[(0, 0)] == 250
    assert grouped[(0, 1)] == 100
