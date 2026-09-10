#!/usr/bin/env python3
"""Run the complete 16-run MemTier-MoE ablation benchmark matrix.

Matrix structure:
  4 Baselines:
    1. GPU-Resident (HBM only, infinite budget)
    2. Two-Tier (HBM + DRAM, LFU, no CXL, no prefetch)
    3. CXL-Only (HBM + DRAM + CXL, LFU, no prefetch)
    4. MemTier-MoE (HBM + DRAM + CXL, LFU + predictive prefetching)
  x 2 HBM Budgets:
    - 3.0 GB (high memory pressure / offloading stress)
    - 4.0 GB (realistic RTX 4050 usable cache budget)
  x 2 Domains:
    - WikiText (prose)
    - Code (Python algorithms)
  = 16 runs total
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
from typing import Dict, List, Tuple
import numpy as np

from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.types import ExpertId
from memtier_moe.introspect.routing_tracer import RoutingTrace
from memtier_moe.introspect.analysis import extract_co_occurrences
from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel
from memtier_moe.evaluation.baselines import make_baseline_configs, BaselineType
from memtier_moe.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkMatrix
from memtier_moe.evaluation.visualization import (
    plot_hit_rate_comparison,
    plot_throughput_comparison,
    plot_expert_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_routing_traces(trace_dir: str = "traces") -> Tuple[Dict[str, List[List[List[int]]]], Dict[str, RoutingTrace]]:
    """Load pre-generated .npz routing traces for all domains."""
    routing_data = {}
    raw_traces = {}

    for domain in ["wikitext", "code"]:
        path = os.path.join(trace_dir, f"routing_trace_{domain}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Trace file not found: {path}. Run scripts/generate_traces.py first.")

        trace = RoutingTrace.load(path)
        raw_traces[domain] = trace

        # Group decisions by [token_idx][layer_idx] -> list of expert IDs
        tokens_map: Dict[int, Dict[int, List[int]]] = {}
        for d in trace.decisions:
            if d.token_idx not in tokens_map:
                tokens_map[d.token_idx] = {}
            tokens_map[d.token_idx][d.layer_idx] = d.top_k_expert_ids

        # Convert to nested list [token_idx][layer_idx][expert_idx]
        sorted_tokens = sorted(tokens_map.keys())
        decisions_nested = []
        for t in sorted_tokens:
            layers = [tokens_map[t][l] for l in range(trace.num_layers)]
            decisions_nested.append(layers)

        routing_data[domain] = decisions_nested
        logger.info(f"Loaded {domain} trace: {len(decisions_nested)} tokens, {trace.num_layers} layers")

    return routing_data, raw_traces


def main():
    print("=" * 70)
    print("MemTier-MoE: 16-Run Full Evaluation Matrix Benchmark")
    print("=" * 70)

    # 1. Load traces
    routing_data, raw_traces = load_routing_traces("traces")

    # 2. Build CoOccurrenceModel from real WikiText trace
    wiki_trace = raw_traces["wikitext"]
    token_indices = np.array([d.token_idx for d in wiki_trace.decisions])
    layer_indices = np.array([d.layer_idx for d in wiki_trace.decisions])
    expert_ids = np.array([d.top_k_expert_ids for d in wiki_trace.decisions])

    stats = extract_co_occurrences(token_indices, layer_indices, expert_ids, num_layers=wiki_trace.num_layers)
    co_occurr_model = CoOccurrenceModel(lookahead=2, min_probability=0.05)
    co_occurr_model.build_from_stats(stats)
    print(f"\n{co_occurr_model.summary()}")

    # 3. Setup expert sizes based on detected trace architecture
    max_expert_id = max(
        max(e for d in trace.decisions for e in d.top_k_expert_ids)
        for trace in raw_traces.values()
    )
    num_experts = max_expert_id + 1
    num_layers = wiki_trace.num_layers

    if num_experts <= 8:
        # Real Qwen1.5-4x0.5B-MoE checkpoint: 13.3 MB per expert block
        expert_size = 13_300_000
        budgets_gb = [0.6, 0.9]  # 600 MB (47% capacity) and 900 MB (71% capacity)
        total_model_bytes = num_layers * num_experts * expert_size
        print(f"\nReal MoE Checkpoint Configuration: {num_layers} layers x {num_experts} experts (13.3 MB each)")
        print(f"Total Active MoE Expert Weights: {total_model_bytes / 1e6:.1f} MB ({num_layers * num_experts} expert blocks)")
    else:
        # Full Qwen1.5-MoE-A2.7B INT4 architecture: 4.33 MB per expert block
        expert_size = 4_325_376
        budgets_gb = [3.0, 4.0]
        total_model_bytes = num_layers * num_experts * expert_size
        print(f"\nTotal MoE Model Size (INT4): {total_model_bytes / 1e9:.2f} GB ({num_layers * num_experts} expert blocks)")

    expert_sizes: Dict[ExpertId, int] = {
        (l, e): expert_size
        for l in range(num_layers)
        for e in range(num_experts)
    }

    # Compute initial frequencies from trace
    hist = wiki_trace.expert_frequency_histogram()
    initial_frequencies = {(l, e): float(hist.get(e, 0)) for l in range(num_layers) for e in range(num_experts)}

    # 4. Run benchmark matrix for 2 HBM budgets
    all_results = BenchmarkMatrix()

    runner = BenchmarkRunner(
        expert_sizes=expert_sizes,
        initial_frequencies=initial_frequencies,
        co_occurrence_model=co_occurr_model,
    )

    for budget in budgets_gb:
        budget_bytes = int(budget * 1e9)
        print(f"\n>>> Running Evaluation for HBM Cache Budget: {budget:.1f} GB <<<")
        baselines = make_baseline_configs(
            hbm_budget_bytes=budget_bytes,
            host_dram_bytes=16_000_000_000,
            cxl_memory_bytes=32_000_000_000,
        )

        for bt, b_cfg in baselines.items():
            for domain, decisions in routing_data.items():
                res = runner.run_single(b_cfg, decisions, domain=domain)
                all_results.results.append(res)
                print(
                    f"  [{res.baseline_name:<25}] {res.domain:<9} "
                    f"HitRate: {res.cache_hit_rate:.1%} | "
                    f"Prefetch Prec: {res.prefetch_precision:.1%} | "
                    f"Tok/s: {res.tokens_per_second:5.1f} | "
                    f"Wall: {res.wall_time_seconds:.2f}s"
                )

    # 5. Print comprehensive results table
    print("\n" + "=" * 90)
    print("FINAL ABLATION BENCHMARK RESULTS TABLE")
    print("=" * 90)
    print(all_results.to_table())

    # 6. Generate Publication Figures
    os.makedirs("results", exist_ok=True)
    dicts = all_results.to_dicts()

    primary_budget = max(budgets_gb)
    res_primary = [d for d in dicts if abs(d["hbm_budget_gb"] - primary_budget) < 0.1]

    plot_hit_rate_comparison(
        res_primary,
        output_path="results/hit_rate_comparison.png",
        title=f"Cache Hit Rate Comparison ({primary_budget:.1f} GB HBM Budget)",
    )

    plot_throughput_comparison(
        res_primary,
        output_path="results/simulated_throughput_comparison.png",
        title="Simulated Pipeline Throughput (tokens/sec)",
    )

    # Expert Heatmaps for WikiText & Code
    plot_expert_heatmap(
        wiki_trace,
        output_path="results/expert_heatmap_wikitext.png",
        title="Qwen1.5-MoE Routing Heatmap (WikiText)",
    )
    code_trace = raw_traces["code"]
    plot_expert_heatmap(
        code_trace,
        output_path="results/expert_heatmap_code.png",
        title="Qwen1.5-MoE Routing Heatmap (Code)",
    )

    # Save raw JSON results
    with open("results/benchmark_results.json", "w") as f:
        json.dump(dicts, f, indent=2)

    print("\nAll benchmark artifacts & plots generated in ./results/")


if __name__ == "__main__":
    main()
