"""Tests for benchmark methodology honesty and metric separation."""
from __future__ import annotations
import pytest

from memtier_moe.core.config import MemTierConfig
from memtier_moe.evaluation.baselines import BaselineConfig, BaselineType
from memtier_moe.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkResult, BenchmarkMatrix


def test_simulation_metrics_explicitly_labeled():
    """Verify that BenchmarkResult created by BenchmarkRunner has explicit simulation/modeled metadata."""
    expert_sizes = {(0, 0): 1000, (0, 1): 1000, (1, 0): 1000}
    initial_freqs = {(0, 0): 2.0, (0, 1): 1.0, (1, 0): 1.0}
    runner = BenchmarkRunner(expert_sizes=expert_sizes, initial_frequencies=initial_freqs)

    config = MemTierConfig(hbm_cache_budget_bytes=2000, host_dram_bytes=4000, cxl_memory_bytes=8000)
    baseline = BaselineConfig(name="MemTier-MoE", baseline_type=BaselineType.MEMTIER_MOE, config=config)

    # 2 tokens, 2 layers, 1 expert per layer
    routing_decisions = [
        [[0], [0]],
        [[1], [0]],
    ]

    result = runner.run_single(baseline, routing_decisions, domain="test_domain")

    # Verify that metadata explicitly marks this as simulation / modeled
    assert result.benchmark_mode == "simulation"
    assert result.computation_type == "modeled"
    assert result.transfer_latency_type == "modeled"
    assert result.cxl_type == "emulated"
    assert "RTX 4050" in result.hardware_device

    # Verify simulation wallclock metric exists
    assert hasattr(result, "simulation_wallclock_tokens_per_second")
    assert result.simulation_wallclock_tokens_per_second > 0

    # Verify table representation clearly distinguishes simulation metrics
    matrix = BenchmarkMatrix(results=[result])
    table = matrix.to_table()
    assert "SimWallTok/s" in table
    assert "ModPipeTok/s" in table

    # Verify dictionaries contain all transparent methodology fields
    dicts = matrix.to_dicts()
    assert len(dicts) == 1
    d = dicts[0]
    assert d["benchmark_mode"] == "simulation"
    assert d["computation_type"] == "modeled"
    assert d["cxl_type"] == "emulated"
    assert "simulation_wallclock_tokens_per_second" in d


def test_capacity_sweep_std_degrees_of_freedom():
    """Verify that capacity sweep std devs mathematically match population and sample formulas."""
    import json
    import math
    import numpy as np

    with open("results/cxl_capacity_sweep_results.json") as f:
        data = json.load(f)

    for item in data:
        throughputs = item["trial_throughputs"]
        assert len(throughputs) == 5
        pop_std = float(np.std(throughputs, ddof=0))
        sample_std = float(np.std(throughputs, ddof=1))

        # Check exact Bessel correction ratio sqrt(5/4) = 1.1180339887...
        expected_ratio = math.sqrt(5.0 / 4.0)
        actual_ratio = sample_std / pop_std
        assert abs(actual_ratio - expected_ratio) < 1e-6

        # Check that stored fields match within rounding
        assert abs(round(pop_std, 2) - item["std_tokens_per_second"]) <= 0.015
        assert abs(sample_std - item["std_tokens_per_second_sample"]) < 1e-4
        assert abs(pop_std - item["std_tokens_per_second_pop"]) < 1e-4


def test_live_benchmark_results_integrity():
    """Verify that live_benchmark_results.json matches canonical RTX 4050 physical measurements."""
    import json

    with open("results/live_benchmark_results.json") as f:
        data = json.load(f)

    by_scenario = {item["scenario"]: item for item in data}

    # GPU-Resident baseline
    assert "GPU-Resident (Full VRAM)" in by_scenario
    assert by_scenario["GPU-Resident (Full VRAM)"]["tokens_per_second"] == 8.73

    # Two-Tier 600MB baseline
    assert "Two-Tier (Weight Transfer 600M)" in by_scenario
    assert by_scenario["Two-Tier (Weight Transfer 600M)"]["tokens_per_second"] == 5.71
    assert by_scenario["Two-Tier (Weight Transfer 600M)"]["evictions"] == 776

    # Lookahead Pre-Gating (post-adaptive-gating)
    assert "Lookahead Pre-Gating (600MB)" in by_scenario
    lookahead = by_scenario["Lookahead Pre-Gating (600MB)"]
    assert lookahead["tokens_per_second"] == 5.26
    assert lookahead["evictions"] == 778
    assert round(lookahead["transfer_mb"], 2) == 13460.57
    assert lookahead["prefetch_issued"] == 11
    assert lookahead["prefetch_useful"] == 11

