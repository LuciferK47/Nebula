"""Regression tests for running the system under real memory pressure.

The existing M1-M3 test suites all used HBM/DRAM budgets large enough
that no baseline ever actually had to evict anything under load — so the
whole point of a memory-tiering system (the crash-free path under
capacity pressure) was untested. All four baselines crashed as soon as
HBM was smaller than the working set:
  - Two independent HBM-occupancy counters (the pool's own and the
    cache's private copy) drifted apart.
  - demand_fetch() ran before eviction instead of after.
  - The GPU-Resident baseline (host_dram_bytes=0) crashed on its very
    first placement, since setup unconditionally staged into DRAM.
  - The prefetch scheduler and the cache were never wired together, so
    a completed prefetch never became a cache hit, and a demand-fetch
    could race a second transfer against one already in flight.
  - The 2-slot double-buffer was used to stage up to
    `max_inflight_transfers` (4) concurrent async transfers, silently
    corrupting an earlier transfer's tensor once more than 2 were
    in flight at once.

These tests exercise the system the way the evaluation harness actually
does: HBM sized well below the routing trace's working set, across all
four baselines including prefetch.
"""
from __future__ import annotations

import numpy as np
import pytest

from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.types import MemoryTier
from memtier_moe.evaluation.baselines import make_baseline_configs, BaselineType
from memtier_moe.evaluation.benchmark_runner import BenchmarkRunner
from memtier_moe.introspect.analysis import extract_co_occurrences
from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel
from memtier_moe.runtime.engine import InferenceEngine


def _make_skewed_routing(num_layers=6, num_experts=12, top_k=2, num_tokens=40, seed=0):
    """Zipf-ish routing so caching has something real to do (a uniform
    distribution over few experts makes every policy look equally good)."""
    rng = np.random.RandomState(seed)
    p = np.array([1.0 / (i + 1) for i in range(num_experts)])
    p /= p.sum()

    token_idx, layer_idx, expert_ids, routing = [], [], [], []
    for t in range(num_tokens):
        token_layers = []
        for l in range(num_layers):
            experts = rng.choice(num_experts, size=top_k, replace=False, p=p).tolist()
            token_idx.append(t)
            layer_idx.append(l)
            expert_ids.append(experts)
            token_layers.append(experts)
        routing.append(token_layers)

    return np.array(token_idx), np.array(layer_idx), np.array(expert_ids), routing, num_layers, num_experts


@pytest.fixture
def skewed_routing():
    return _make_skewed_routing()


@pytest.fixture
def co_occurrence_model(skewed_routing):
    tok_idx, lay_idx, exp_ids, _, num_layers, _ = skewed_routing
    stats = extract_co_occurrences(tok_idx, lay_idx, exp_ids, num_layers=num_layers)
    model = CoOccurrenceModel(lookahead=2, min_probability=0.0)
    model.build_from_stats(stats)
    return model


class TestAllBaselinesSurviveMemoryPressure:
    """The core regression: HBM sized well below the working set, for
    every baseline in the headline ablation table."""

    def test_all_four_baselines_complete(self, skewed_routing, co_occurrence_model):
        _, _, _, routing, num_layers, num_experts = skewed_routing
        expert_sizes = {(l, e): 1000 for l in range(num_layers) for e in range(num_experts)}

        # Deliberately small: num_layers*num_experts*1000 = 72,000 bytes
        # total, HBM budget of 20,000 forces real eviction pressure.
        configs = make_baseline_configs(
            hbm_budget_bytes=20_000, host_dram_bytes=1_000_000, cxl_memory_bytes=1_000_000
        )
        runner = BenchmarkRunner(expert_sizes=expert_sizes, co_occurrence_model=co_occurrence_model)

        results = {}
        for bt in BaselineType:
            results[bt] = runner.run_single(configs[bt], routing, domain="pressure_test")

        # All four ran to completion (this used to raise RuntimeError for
        # every baseline except the trivially-sized ones).
        for bt, r in results.items():
            assert r.num_tokens == len(routing)
            assert 0.0 <= r.cache_hit_rate <= 1.0

        # GPU-Resident has (effectively) infinite HBM: nothing is ever a miss.
        assert results[BaselineType.GPU_RESIDENT].cache_hit_rate == 1.0
        assert results[BaselineType.GPU_RESIDENT].evictions == 0

        # The offloading baselines must have actually evicted something —
        # otherwise the "pressure" in this test isn't real pressure.
        assert results[BaselineType.TWO_TIER].evictions > 0
        assert results[BaselineType.CXL_ONLY].evictions > 0
        assert results[BaselineType.MEMTIER_MOE].evictions > 0

    def test_prefetch_baseline_reports_nonzero_precision(self, skewed_routing, co_occurrence_model):
        """The prefetch scheduler used to always report metrics precision
        of 0.0 regardless of outcomes (issue-time write of useful=False,
        never corrected). A stable, skewed routing distribution should
        produce a real, non-degenerate precision value."""
        _, _, _, routing, num_layers, num_experts = skewed_routing
        expert_sizes = {(l, e): 1000 for l in range(num_layers) for e in range(num_experts)}
        configs = make_baseline_configs(
            hbm_budget_bytes=20_000, host_dram_bytes=1_000_000, cxl_memory_bytes=1_000_000
        )
        runner = BenchmarkRunner(expert_sizes=expert_sizes, co_occurrence_model=co_occurrence_model)

        result = runner.run_single(configs[BaselineType.MEMTIER_MOE], routing, domain="pressure_test")
        assert result.prefetch_total > 0
        assert result.prefetch_precision > 0.0


class TestCXLTierIsReachable:
    """PlacementPolicy previously had zero callers outside its own test
    file — every eviction went HBM->DRAM only, so the CXL tier (the
    system's stated differentiator) was never actually populated."""

    def test_cold_experts_spill_into_cxl_under_dram_pressure(self):
        num_layers, num_experts = 6, 20
        expert_sizes = {(l, e): 1000 for l in range(num_layers) for e in range(num_experts)}
        total_bytes = len(expert_sizes) * 1000  # 120,000

        rng = np.random.RandomState(1)
        p = np.array([1.0 / (i + 1) for i in range(num_experts)])
        p /= p.sum()
        routing = [
            [rng.choice(num_experts, size=2, replace=False, p=p).tolist() for _ in range(num_layers)]
            for _ in range(80)
        ]

        # DRAM sized just above the full working set, HBM small — forces
        # both HBM->DRAM eviction pressure and, once DRAM's free margin
        # drops below its reserve, HBM->CXL direct placement.
        config = MemTierConfig(
            hbm_cache_budget_bytes=8_000,
            host_dram_bytes=total_bytes + 4_000,
            cxl_memory_bytes=1_000_000,
            placement_warm_threshold=1.0,
            dram_reserve_fraction=0.10,
        )
        engine = InferenceEngine(config)
        engine.setup_from_profile(expert_sizes, {k: 0.1 for k in expert_sizes})

        for t, token_layers in enumerate(routing):
            engine.run_token(t, token_layers)

        summary = engine.tier_manager.summary()
        assert summary["cxl"]["expert_count"] > 0, (
            "No expert ever reached the CXL tier under DRAM pressure — "
            "PlacementPolicy is not being consulted on eviction."
        )


class TestDemoteFailsLoudNotSilent:
    """demote() used to silently drop the tensor (data loss with no
    error) if the target tier had no room. It should now raise instead,
    and leave the expert exactly where it was."""

    def test_demote_raises_when_target_full(self):
        class FakeTensor:
            def __init__(self, n):
                self._n = n

            def element_size(self):
                return 1

            def numel(self):
                return self._n

            def cpu(self):
                return self

            def cuda(self):
                return self

            def pin_memory(self):
                return self

        from memtier_moe.core.metrics import MetricsTracker
        from memtier_moe.memory.tier_manager import TierManager

        config = MemTierConfig(hbm_cache_budget_bytes=100, host_dram_bytes=50, cxl_memory_bytes=50)
        tm = TierManager(config, MetricsTracker())
        tm.register_expert((0, 0), 100, MemoryTier.HBM)
        tm.place_initial((0, 0), FakeTensor(100), MemoryTier.HBM)

        with pytest.raises(RuntimeError):
            tm.demote((0, 0))  # DRAM capacity (50) < expert size (100)

        # Must not have been lost in the process.
        assert tm.get_pool(MemoryTier.HBM).contains((0, 0))
        assert tm.get_tier((0, 0)) == MemoryTier.HBM


class TestDecayIsPerToken:
    """The cache's decay counter used to increment once per
    forward_moe_layer call — i.e. once per layer, not once per token —
    making the configured half-life run out ~num_layers times faster
    than intended."""

    def test_half_life_measured_in_tokens_not_layers(self):
        config = MemTierConfig(
            hbm_cache_budget_bytes=10_000, host_dram_bytes=100_000, frequency_decay_half_life=500
        )
        engine = InferenceEngine(config)
        num_layers = 24
        sizes = {(l, 0): 100 for l in range(num_layers)}
        engine.setup_from_profile(sizes, {(0, 0): 1.0})

        engine.run_token(0, [[0] if l == 0 else [] for l in range(num_layers)])
        meta = engine.tier_manager.get_metadata((0, 0))
        freq_before = meta.decayed_frequency

        for t in range(1, 501):
            engine.run_token(t, [[] for _ in range(num_layers)])

        # 501 tokens * 24 layers = 12,024 forward_moe_layer calls, but the
        # logical token counter should read ~500, not ~12,000.
        assert engine.cache._current_token == 500

        freq_after = meta.decay_frequency(engine.cache._current_token, config.frequency_decay_half_life)
        assert freq_after == pytest.approx(freq_before * 0.5, rel=0.01)
