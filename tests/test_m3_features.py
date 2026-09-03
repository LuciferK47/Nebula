"""Tests for M3 features: co-occurrence, predictor, scheduler, baselines, benchmarks."""
from __future__ import annotations
import pytest
import numpy as np

from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.memory.expert_metadata import ExpertMetadata
from memtier_moe.introspect.analysis import (
    extract_co_occurrences,
    build_conditional_probabilities,
)
from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel
from memtier_moe.prefetch.predictor import ExpertPredictor
from memtier_moe.evaluation.baselines import (
    BaselineType,
    make_baseline_configs,
    describe_baselines,
)


# ── Synthetic routing data for testing ────────────────────────────────


def _make_routing_data(
    num_tokens: int = 10,
    num_layers: int = 4,
    top_k: int = 2,
    num_experts: int = 8,
    seed: int = 42,
):
    """Generate synthetic routing decisions.

    Returns (token_indices, layer_indices, expert_ids) as numpy arrays,
    and routing_decisions as the list-of-lists format.
    """
    rng = np.random.RandomState(seed)
    token_indices = []
    layer_indices = []
    expert_ids_flat = []
    routing_decisions = []

    for tok in range(num_tokens):
        token_layers = []
        for lay in range(num_layers):
            experts = rng.choice(num_experts, size=top_k, replace=False).tolist()
            token_indices.append(tok)
            layer_indices.append(lay)
            expert_ids_flat.append(experts)
            token_layers.append(experts)
        routing_decisions.append(token_layers)

    return (
        np.array(token_indices),
        np.array(layer_indices),
        np.array(expert_ids_flat),
        routing_decisions,
    )


# ── Co-occurrence Analysis ────────────────────────────────────────────


class TestCoOccurrenceAnalysis:
    def test_extract_counts(self):
        tok_idx, lay_idx, exp_ids, _ = _make_routing_data()
        stats = extract_co_occurrences(tok_idx, lay_idx, exp_ids, num_layers=4)

        assert stats.num_tokens == 10
        assert stats.num_layers == 4
        assert len(stats.marginal_counts) > 0
        assert len(stats.joint_counts) > 0

    def test_conditional_probabilities_bounded(self):
        tok_idx, lay_idx, exp_ids, _ = _make_routing_data()
        stats = extract_co_occurrences(tok_idx, lay_idx, exp_ids, num_layers=4)
        probs = build_conditional_probabilities(stats, min_threshold=0.0)

        for prob in probs.values():
            assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_threshold_filters(self):
        tok_idx, lay_idx, exp_ids, _ = _make_routing_data()
        stats = extract_co_occurrences(tok_idx, lay_idx, exp_ids, num_layers=4)

        probs_low = build_conditional_probabilities(stats, min_threshold=0.0)
        probs_high = build_conditional_probabilities(stats, min_threshold=0.5)

        assert len(probs_high) <= len(probs_low)


# ── Co-occurrence Model ───────────────────────────────────────────────


class TestCoOccurrenceModel:
    def test_build_and_query(self):
        tok_idx, lay_idx, exp_ids, _ = _make_routing_data()
        stats = extract_co_occurrences(tok_idx, lay_idx, exp_ids, num_layers=4)

        model = CoOccurrenceModel(lookahead=2, min_probability=0.0)
        model.build_from_stats(stats)

        assert model.num_entries > 0
        assert model.num_source_keys > 0

    def test_predict_returns_sorted(self):
        tok_idx, lay_idx, exp_ids, _ = _make_routing_data()
        stats = extract_co_occurrences(tok_idx, lay_idx, exp_ids, num_layers=4)

        model = CoOccurrenceModel(lookahead=2, min_probability=0.0)
        model.build_from_stats(stats)

        predictions = model.predict(src_layer=0, src_experts=[0, 1])
        # Should be sorted by probability descending
        for i in range(len(predictions) - 1):
            assert predictions[i][1] >= predictions[i + 1][1]

    def test_empty_model_returns_empty(self):
        model = CoOccurrenceModel()
        predictions = model.predict(src_layer=0, src_experts=[0])
        assert predictions == []


# ── Expert Predictor ──────────────────────────────────────────────────


class TestExpertPredictor:
    def _make_predictor(self):
        tok_idx, lay_idx, exp_ids, _ = _make_routing_data()
        stats = extract_co_occurrences(tok_idx, lay_idx, exp_ids, num_layers=4)
        model = CoOccurrenceModel(lookahead=2, min_probability=0.0)
        model.build_from_stats(stats)
        return ExpertPredictor(model=model, confidence_threshold=0.0)

    def test_predict_filters_hbm_resident(self):
        predictor = self._make_predictor()
        all_preds = predictor.predict(
            current_layer=0,
            current_experts=[0, 1],
            hbm_resident=set(),
            inflight=set(),
        )
        # Now with all predicted experts in HBM — none should reappear
        hbm = {p.expert_id for p in all_preds}
        filtered = predictor.predict(
            current_layer=0,
            current_experts=[0, 1],
            hbm_resident=hbm,
            inflight=set(),
        )
        # No prediction should overlap with what's in HBM
        filtered_ids = {p.expert_id for p in filtered}
        assert filtered_ids.isdisjoint(hbm), (
            f"Predictions overlap with HBM: {filtered_ids & hbm}"
        )

    def test_precision_tracking(self):
        predictor = self._make_predictor()
        preds = predictor.predict(
            current_layer=0,
            current_experts=[0],
            hbm_resident=set(),
            inflight=set(),
        )
        if preds:
            # Mark first as useful
            predictor.record_access(preds[0].expert_id)
            # Flush rest as stale
            predictor.flush_stale(current_layer=10)

            assert predictor.total_useful >= 1
            stats = predictor.stats()
            assert stats["rolling_precision"] >= 0.0

    def test_max_predictions_cap(self):
        predictor = self._make_predictor()
        predictor.max_predictions_per_layer = 2
        preds = predictor.predict(
            current_layer=0,
            current_experts=[0, 1, 2, 3],
            hbm_resident=set(),
            inflight=set(),
        )
        assert len(preds) <= 2


# ── Baselines ─────────────────────────────────────────────────────────


class TestBaselines:
    def test_four_baselines_created(self):
        configs = make_baseline_configs()
        assert len(configs) == 4
        assert BaselineType.GPU_RESIDENT in configs
        assert BaselineType.TWO_TIER in configs
        assert BaselineType.CXL_ONLY in configs
        assert BaselineType.MEMTIER_MOE in configs

    def test_only_memtier_has_prefetch(self):
        configs = make_baseline_configs()
        for bt, bc in configs.items():
            if bt == BaselineType.MEMTIER_MOE:
                assert bc.enable_prefetch is True
            else:
                assert bc.enable_prefetch is False

    def test_describe_output(self):
        configs = make_baseline_configs()
        text = describe_baselines(configs)
        assert "GPU-Resident" in text
        assert "MemTier-MoE" in text


# ── Benchmark Runner (lightweight) ────────────────────────────────────


class TestBenchmarkRunner:
    def test_single_run(self):
        from memtier_moe.evaluation.benchmark_runner import BenchmarkRunner

        # Small synthetic data
        num_layers = 4
        num_experts = 8
        expert_sizes = {(l, e): 100 for l in range(num_layers) for e in range(num_experts)}

        _, _, _, routing = _make_routing_data(num_tokens=5, num_layers=num_layers)

        configs = make_baseline_configs(hbm_budget_bytes=100000, host_dram_bytes=1000000)
        runner = BenchmarkRunner(expert_sizes=expert_sizes)

        result = runner.run_single(
            baseline=configs[BaselineType.TWO_TIER],
            routing_decisions=routing,
            domain="test",
        )

        assert result.num_tokens == 5
        assert result.wall_time_seconds > 0
        assert result.cache_hit_rate >= 0.0

    def test_matrix_run(self):
        from memtier_moe.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkMatrix

        num_layers = 4
        num_experts = 8
        expert_sizes = {(l, e): 100 for l in range(num_layers) for e in range(num_experts)}
        _, _, _, routing = _make_routing_data(num_tokens=3, num_layers=num_layers)

        # Use only 2 baselines for speed
        all_configs = make_baseline_configs(hbm_budget_bytes=100000, host_dram_bytes=1000000)
        subset = {
            BaselineType.TWO_TIER: all_configs[BaselineType.TWO_TIER],
            BaselineType.CXL_ONLY: all_configs[BaselineType.CXL_ONLY],
        }

        runner = BenchmarkRunner(expert_sizes=expert_sizes)
        matrix = runner.run_matrix(subset, {"test_domain": routing})

        assert len(matrix.results) == 2
        table = matrix.to_table()
        assert "Two-Tier" in table

    def test_result_to_dicts(self):
        from memtier_moe.evaluation.benchmark_runner import BenchmarkRunner

        num_layers = 4
        num_experts = 8
        expert_sizes = {(l, e): 100 for l in range(num_layers) for e in range(num_experts)}
        _, _, _, routing = _make_routing_data(num_tokens=3, num_layers=num_layers)

        configs = make_baseline_configs(hbm_budget_bytes=100000, host_dram_bytes=1000000)
        runner = BenchmarkRunner(expert_sizes=expert_sizes)

        result = runner.run_single(configs[BaselineType.CXL_ONLY], routing, "test")
        matrix_obj = __import__(
            "memtier_moe.evaluation.benchmark_runner", fromlist=["BenchmarkMatrix"]
        ).BenchmarkMatrix(results=[result])
        dicts = matrix_obj.to_dicts()

        assert len(dicts) == 1
        assert "hit_rate" in dicts[0]
