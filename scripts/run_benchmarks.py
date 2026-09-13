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
logging.getLogger("memtier_moe").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


import argparse

def load_routing_traces(trace_dir: str = "traces", trace_prefix: str = "routing_trace_") -> Tuple[Dict[str, List[List[List[int]]]], Dict[str, RoutingTrace]]:
    """Load pre-generated .npz routing traces for all domains."""
    routing_data = {}
    raw_traces = {}

    for domain in ["wikitext", "code"]:
        path = os.path.join(trace_dir, f"{trace_prefix}{domain}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Trace file not found: {path}. Run scripts/generate_traces.py or generate_qwen14b_traces.py first.")

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
    parser = argparse.ArgumentParser(description="MemTier-MoE Full Evaluation Matrix Benchmark")
    parser.add_argument("--trace-prefix", type=str, default="routing_trace_", help="Prefix for trace files (e.g. routing_trace_qwen14b_)")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 expert weights (17.3MB) instead of INT4 (4.33MB)")
    parser.add_argument("--budgets", type=str, default=None, help="Comma-separated HBM budgets in GB (e.g. '1.5,3.0')")
    parser.add_argument("--dram-gb", type=float, default=2.0, help="Host DRAM pool budget in GB (default: 2.0 GB)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save plots and results")
    args = parser.parse_args()

    print("=" * 70)
    print(f"MemTier-MoE: 16-Run Full Evaluation Matrix Benchmark ({args.trace_prefix})")
    print(f"DRAM Working Set Budget: {args.dram_gb:.1f} GB | CXL Pool: 32.0 GB")
    print("=" * 70)

    # 1. Load traces
    routing_data, raw_traces = load_routing_traces("traces", trace_prefix=args.trace_prefix)

    # 2. Build CoOccurrenceModel from real WikiText trace
    wiki_trace = raw_traces["wikitext"]
    token_indices = np.array([d.token_idx for d in wiki_trace.decisions])
    layer_indices = np.array([d.layer_idx for d in wiki_trace.decisions])
    expert_ids = np.array([d.top_k_expert_ids for d in wiki_trace.decisions])

    stats = extract_co_occurrences(token_indices, layer_indices, expert_ids, num_layers=wiki_trace.num_layers)
    co_occurr_model = CoOccurrenceModel(lookahead=2, min_probability=0.03)
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
        budgets_gb = [0.6, 0.9] if args.budgets is None else [float(b) for b in args.budgets.split(",")]
        total_model_bytes = num_layers * num_experts * expert_size
        print(f"\nReal MoE Checkpoint Configuration: {num_layers} layers x {num_experts} experts (13.3 MB each)")
        print(f"Total Active MoE Expert Weights: {total_model_bytes / 1e6:.1f} MB ({num_layers * num_experts} expert blocks)")
    else:
        # Full Qwen1.5-MoE-A2.7B architecture
        if args.fp16:
            expert_size = 17_301_504  # 8,650,752 params * 2 bytes (FP16)
            budgets_gb = [1.5, 3.0] if args.budgets is None else [float(b) for b in args.budgets.split(",")]
            total_model_bytes = num_layers * num_experts * expert_size
            print(f"\nTotal MoE Model Size (FP16): {total_model_bytes / 1e9:.2f} GB ({num_layers * num_experts} expert blocks @ 17.3 MB each)")
        else:
            expert_size = 4_325_376  # 8,650,752 params * 0.5 bytes (INT4)
            budgets_gb = [1.5, 3.0] if args.budgets is None else [float(b) for b in args.budgets.split(",")]
            total_model_bytes = num_layers * num_experts * expert_size
            print(f"\nTotal MoE Model Size (INT4): {total_model_bytes / 1e9:.2f} GB ({num_layers * num_experts} expert blocks @ 4.33 MB each)")

    expert_sizes = {
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
            host_dram_bytes=int(args.dram_gb * 1e9),
            cxl_memory_bytes=32_000_000_000,
        )

        for bt, b_cfg in baselines.items():
            for domain, decisions in routing_data.items():
                res = runner.run_single(b_cfg, decisions, domain=domain)
                all_results.results.append(res)
                print(
                    f"  [{res.baseline_name:<24}] {res.domain:<8} "
                    f"H_HBM: {res.hbm_hit_rate:.1%} | "
                    f"H_DRAM: {res.dram_hit_rate:.1%} | "
                    f"H_CXL: {res.cxl_hit_rate:.1%} | "
                    f"AMAT: {res.amat_ns:.0f}ns | "
                    f"Tok/s: {res.tokens_per_second:5.1f} | "
                    f"Wall: {res.wall_time_seconds:.2f}s"
                )

    # 5. Print comprehensive results table
    print("\n" + "=" * 90)
    print("FINAL ABLATION BENCHMARK RESULTS TABLE")
    print("=" * 90)
    print(all_results.to_table())

    # 6. Generate Publication Figures
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    dicts = all_results.to_dicts()

    prefix = "qwen14b_" if "qwen14b" in args.trace_prefix else ""
    primary_budget = max(budgets_gb)
    res_primary = [d for d in dicts if abs(d["hbm_budget_gb"] - primary_budget) < 0.1]

    plot_hit_rate_comparison(
        res_primary,
        output_path=os.path.join(out_dir, f"{prefix}hit_rate_comparison.png"),
        title=f"Cache Hit Rate Comparison ({primary_budget:.1f} GB HBM Budget)",
    )

    plot_throughput_comparison(
        res_primary,
        output_path=os.path.join(out_dir, f"{prefix}simulated_throughput_comparison.png"),
        title="Simulated Pipeline Throughput (tokens/sec)",
    )

    from memtier_moe.evaluation.visualization import plot_cxl_tier_breakdown
    plot_cxl_tier_breakdown(
        res_primary,
        output_path=os.path.join(out_dir, f"{prefix}cxl_tier_breakdown.png"),
        title=f"CXL Tier-Decomposed Memory Distribution & AMAT ({primary_budget:.1f} GB HBM Budget)",
    )

    # Expert Heatmaps for WikiText & Code
    plot_expert_heatmap(
        wiki_trace,
        output_path=os.path.join(out_dir, f"{prefix}expert_heatmap_wikitext.png"),
        title="Qwen1.5-MoE-A2.7B Routing Heatmap (WikiText)",
    )
    code_trace = raw_traces["code"]
    plot_expert_heatmap(
        code_trace,
        output_path=os.path.join(out_dir, f"{prefix}expert_heatmap_code.png"),
        title="Qwen1.5-MoE-A2.7B Routing Heatmap (Code)",
    )

    # Save raw JSON results
    json_path = os.path.join(out_dir, f"{prefix}benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(dicts, f, indent=2)

    print(f"\nAll benchmark artifacts & plots generated in ./{out_dir}/ ({prefix}*)")


if __name__ == "__main__":
    main()
