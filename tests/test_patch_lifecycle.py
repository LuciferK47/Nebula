"""Tests for TieredMoEWrapper's patch/unpatch lifecycle.

No test exercised this before, which is exactly why the S10 cascade shipped
(see GPU_RUNBOOK.md and results/scenarios/s10.json): a run that crashed
mid-generation left every transformer layer wearing a dead TieredMoEBlock
tied to the crashed wrapper's now-discarded engine and pools, and the next
wrapper silently patched nothing, running generation on that stale state
for the rest of the process.

Needs torch (TieredMoEWrapper imports it unconditionally) — run on the GPU
box alongside test_real_module_transfer.py, both skipped in CPU-only dev
environments.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from memtier_moe.core.config import MemTierConfig
from memtier_moe.runtime.tiered_model import TieredMoEWrapper, TieredMoEBlock


class FakeExpert(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FakeMoEBlock(nn.Module):
    """Minimal stand-in for a Mixtral/Qwen2MoE block: an `experts` ModuleList
    plus a `gate` and `num_experts_per_tok`, which is all `_patch_moe_layers`
    reads."""

    def __init__(self, num_experts: int = 4, d: int = 8, top_k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([FakeExpert(d) for _ in range(num_experts)])
        self.gate = nn.Linear(d, num_experts)
        self.num_experts_per_tok = top_k


class FakeLayer(nn.Module):
    def __init__(self, num_experts: int = 4, d: int = 8):
        super().__init__()
        self.mlp = FakeMoEBlock(num_experts, d)


class FakeDenseLayer(nn.Module):
    """A layer with no MoE block at all — `mlp` has no `.experts`."""

    def __init__(self, d: int = 8):
        super().__init__()
        self.mlp = nn.Linear(d, d)


class FakeInner(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class FakeMoEModel(nn.Module):
    """Minimal stand-in for a HF CausalLM: `.model.layers[i].mlp`."""

    def __init__(self, layers):
        super().__init__()
        self.model = FakeInner(layers)


def _wrapper_config(num_experts: int, expert_bytes_each: int = 4096) -> MemTierConfig:
    return MemTierConfig(
        hbm_cache_budget_bytes=num_experts * expert_bytes_each * 4,
        host_dram_bytes=num_experts * expert_bytes_each * 4,
        cxl_memory_bytes=num_experts * expert_bytes_each * 4,
    )


def test_patch_then_unpatch_restores_original_blocks():
    model = FakeMoEModel([FakeLayer(num_experts=4)])
    original = model.model.layers[0].mlp  # capture the instance before patching

    wrapper = TieredMoEWrapper(model=model, config=_wrapper_config(4))
    assert isinstance(model.model.layers[0].mlp, TieredMoEBlock)

    wrapper.unpatch()
    assert model.model.layers[0].mlp is original


def test_unpatch_is_idempotent():
    model = FakeMoEModel([FakeLayer(num_experts=4)])
    wrapper = TieredMoEWrapper(model=model, config=_wrapper_config(4))

    wrapper.unpatch()
    original_restored = model.model.layers[0].mlp
    assert not isinstance(original_restored, TieredMoEBlock)

    # A second call must be a no-op, not an error — unpatch() only touches
    # layers that are still TieredMoEBlock instances.
    wrapper.unpatch()
    assert model.model.layers[0].mlp is original_restored


def test_patching_an_already_patched_model_self_heals():
    """Simulates exactly what run_scenarios.py's missing try/finally used to
    produce: a wrapper that never got to call unpatch() (e.g. it crashed
    mid-generate()), leaving the model mid-patched when a second wrapper is
    constructed on it.

    Before the fix, _patch_moe_layers() did getattr(layer, "mlp") and found
    the stale TieredMoEBlock, which has no `.experts` attribute, so every
    layer was silently skipped — the new wrapper patched zero layers and
    controlled nothing, while generation kept running on the *first*
    wrapper's now-orphaned engine and pools.
    """
    model = FakeMoEModel([FakeLayer(num_experts=4)])
    original = model.model.layers[0].mlp

    wrapper1 = TieredMoEWrapper(model=model, config=_wrapper_config(4))
    assert isinstance(model.model.layers[0].mlp, TieredMoEBlock)
    # Deliberately do NOT call wrapper1.unpatch() — this is the crash path.

    wrapper2 = TieredMoEWrapper(model=model, config=_wrapper_config(4))

    # wrapper2 must have found and re-wrapped the *original* block, not
    # wrapper1's stale TieredMoEBlock, and must actually control the layer:
    # get_metadata() raises KeyError for an expert that was never
    # registered, so this fails if wrapper2 silently patched zero layers.
    assert isinstance(model.model.layers[0].mlp, TieredMoEBlock)
    assert model.model.layers[0].mlp.original_block is original
    wrapper2.engine.tier_manager.get_metadata((0, 0))

    wrapper2.unpatch()
    assert model.model.layers[0].mlp is original


def test_empty_layer_meta_raises():
    """A model with no MoE layers the scanner recognizes must raise loudly
    at construction, not silently hand back a wrapper that controls
    nothing — that's the same failure mode as the self-heal case above,
    just from a different cause."""
    model = FakeMoEModel([FakeDenseLayer()])
    with pytest.raises(ValueError, match="no MoE layers"):
        TieredMoEWrapper(model=model, config=_wrapper_config(1))
