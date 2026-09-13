#!/usr/bin/env python3
"""Run real hardware live inference benchmarks on GPU across tiering baselines.

Measures actual wall-clock tokens/second, cache hit rate, eviction counts,
and transfer volumes using real model weights (Qwen1.5-4x0.5B-Chat-MoE / Qwen1.5-MoE-A2.7B)
running autoregressive generation on NVIDIA GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memtier_moe.core.config import MemTierConfig
from memtier_moe.introspect.routing_tracer import RoutingTrace
from memtier_moe.introspect.analysis import extract_co_occurrences
from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel
from memtier_moe.runtime.tiered_model import TieredMoEWrapper


BENCHMARK_PROMPT = (
    "Mixture of Experts architecture enables efficient scaling of neural parameters "
    "by selectively routing inputs through specialized feed-forward layers."
)


def load_or_calibrate_co_occurrence(
    trace_path: str = "traces/routing_trace_wikitext.npz",
    model: AutoModelForCausalLM = None,
    tokenizer: AutoTokenizer = None,
) -> CoOccurrenceModel:
    """Load co-occurrence model from pre-generated real trace or calibrate."""
    if os.path.exists(trace_path):
        print(f"Loading real routing trace from {trace_path}...")
        trace = RoutingTrace.load(trace_path)
        token_indices = np.array([d.token_idx for d in trace.decisions])
        layer_indices = np.array([d.layer_idx for d in trace.decisions])
        expert_ids = np.array([d.top_k_expert_ids for d in trace.decisions])
        stats = extract_co_occurrences(token_indices, layer_indices, expert_ids, num_layers=trace.num_layers, lookahead=2)
        co_model = CoOccurrenceModel(lookahead=2, min_probability=0.05)
        co_model.build_from_stats(stats)
        print(f"Loaded CoOccurrenceModel: {len(co_model._cond_probs)} learned transitions.")
        return co_model

    print("Trace not found, calibrating directly on prompt...")
    from scripts.run_live_inference import calibrate_routing_model
    return calibrate_routing_model(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")


def run_benchmark_scenario(
    name: str,
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    hbm_budget_mb: int,
    dram_budget_mb: int,
    cxl_budget_mb: int,
    enable_prefetch: bool,
    co_model: CoOccurrenceModel,
    max_new_tokens: int = 30,
    execution_mode: str = "weight_transfer",
    enable_lookahead_gating: bool = False,
) -> Dict[str, Any]:
    """Execute a single live inference benchmark scenario."""
    print(f"\n--- Running Scenario: {name} ---")
    print(f"  HBM: {hbm_budget_mb} MB | DRAM: {dram_budget_mb} MB | Mode: {execution_mode} | Lookahead: {enable_lookahead_gating}")

    config = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=hbm_budget_mb * 1024 * 1024,
        host_dram_bytes=dram_budget_mb * 1024 * 1024,
        cxl_memory_bytes=cxl_budget_mb * 1024 * 1024,
        max_prefetches_per_decision=2,
    )

    tiered_model = TieredMoEWrapper(
        model=base_model,
        config=config,
        enable_prefetch=enable_prefetch,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=hbm_budget_mb * 1024 * 1024,
        execution_mode=execution_mode,
        enable_lookahead_gating=enable_lookahead_gating,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Warmup pass (3 tokens) to stabilize CUDA cache and runtime
    with torch.no_grad():
        tiered_model.generate(**inputs, max_new_tokens=3, do_sample=False)

    # Timed generation pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = tiered_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0

    generated_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = generated_tokens / wall_time if wall_time > 0 else 0.0

    report = tiered_model.report()
    m = report["metrics"]
    sched = report.get("scheduler_stats", {})

    result = {
        "scenario": name,
        "hbm_budget_mb": hbm_budget_mb,
        "dram_budget_mb": dram_budget_mb,
        "execution_mode": execution_mode,
        "enable_prefetch": enable_prefetch,
        "enable_lookahead": enable_lookahead_gating,
        "generated_tokens": generated_tokens,
        "wall_time_seconds": round(wall_time, 3),
        "tokens_per_second": round(tok_per_sec, 2),
        "hit_rate": round(m.get("hit_rate", 0.0), 4),
        "cache_hits": m.get("cache_hits", 0),
        "cache_misses": m.get("cache_misses", 0),
        "evictions": m.get("evictions", 0),
        "transfer_bytes": m.get("total_transfer_bytes", 0),
        "transfer_mb": round(m.get("total_transfer_bytes", 0) / 1e6, 2),
        "prefetch_precision": round(sched.get("prefetch_precision", 0.0), 4) if enable_prefetch else 0.0,
        "prefetch_issued": sched.get("prefetch_total", 0) if enable_prefetch else 0,
        "prefetch_useful": sched.get("prefetch_useful", 0) if enable_prefetch else 0,
    }

    print(f"  Result: {tok_per_sec:.2f} tok/s | HitRate: {result['hit_rate']:.1%} | Evictions: {result['evictions']} | Transfer: {result['transfer_mb']:.1f}MB")

    # Cleanly restore base model layers for next scenario
    tiered_model.unpatch()
    return result


def plot_live_throughput(results: List[Dict[str, Any]], output_path: str = "results/throughput_comparison.png"):
    """Plot publication-quality throughput and memory traffic comparison from live hardware measurements."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    names = [r["scenario"] for r in results]
    throughputs = [r["tokens_per_second"] for r in results]
    hit_rates = [r["hit_rate"] for r in results]
    transfers = [r["transfer_mb"] for r in results]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

    # Color palette
    colors = [
        "#1f77b4",  # GPU Resident: blue
        "#d62728",  # Two-Tier 600MB: red
        "#ff7f0e",  # Lookahead 600MB: orange
        "#2ca02c",  # Hybrid 600MB: green
        "#9467bd",  # Two-Tier 900MB: purple
        "#17becf",  # Hybrid 900MB: cyan
        "#2e7d32",  # Hybrid 1500MB (Near Full VRAM): dark green
    ]
    bar_colors = colors[:len(names)]

    x = np.arange(len(names))
    width = 0.55

    # Panel 1: Throughput and Cache Hit Rate
    bars = ax1.bar(x, throughputs, width, color=bar_colors, edgecolor="#222", linewidth=1.2, zorder=3)
    ax1.set_ylabel("Inference Throughput (tokens / sec)", fontsize=12, fontweight="bold", color="#111")
    ax1.set_title("Live Inference Throughput & Cache Hit Rate\n(NVIDIA GeForce RTX 4050 Laptop GPU)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, tp in zip(bars, throughputs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{tp:.1f} tok/s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    ax2.plot(x, [hr * 100 for hr in hit_rates], color="#b23a22", marker="s", linewidth=2.2, markersize=7, label="Hit Rate (%)", zorder=4)
    ax2.set_ylabel("Cache Hit Rate (%)", fontsize=11, fontweight="bold", color="#b23a22")
    ax2.set_ylim(0, 110)

    for i, hr in enumerate(hit_rates):
        ax2.annotate(
            f"{hr:.0%}",
            (x[i], hr * 100),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#b23a22",
        )

    ax1.set_ylim(0, max(throughputs) * 1.25)

    # Panel 2: PCIe Transfer Volume
    bars2 = ax3.bar(x, transfers, width, color=bar_colors, edgecolor="#222", linewidth=1.2, zorder=3)
    ax3.set_ylabel("PCIe Transfer Volume (MB)", fontsize=12, fontweight="bold", color="#111")
    ax3.set_title("PCIe Memory Bus Traffic (MB Transferred)\n(Activation Offloading vs Weight Migration)", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, tr in zip(bars2, transfers):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(transfers) * 0.015 if max(transfers) > 0 else 0.1),
            f"{tr:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax3.set_ylim(0, max(transfers) * 1.2 if max(transfers) > 0 else 10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved live throughput & traffic chart to: {output_path}")


def main():
    local_chat_moe = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen1.5-4x0.5B-Chat-MoE")
    default_model = local_chat_moe if os.path.exists(local_chat_moe) else "Qwen/Qwen1.5-MoE-A2.7B"
    parser = argparse.ArgumentParser(description="Live Hardware MoE Benchmark")
    parser.add_argument("--model-id", type=str, default=default_model)
    parser.add_argument("--tokens", type=int, default=25, help="Tokens to generate per run")
    parser.add_argument("--output-json", type=str, default="results/live_benchmark_results.json")
    parser.add_argument("--output-plot", type=str, default="results/throughput_comparison.png")
    args = parser.parse_args()

    print("=" * 80)
    print("MemTier-MoE: Genuine Live Hardware Inference Benchmark Suite")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Compute Device: {device.upper()} ({device_name})")
    print(f"Target Model:   {args.model_id}")
    print(f"Tokens/Run:     {args.tokens}")

    # 1. Load tokenizer and base model
    print("\n[1/3] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    # 2. Setup CoOccurrenceModel
    print("\n[2/3] Setting up router co-occurrence model...")
    co_model = load_or_calibrate_co_occurrence(model=base_model, tokenizer=tokenizer)

    # 3. Define Scenarios: Systematic Evaluation & Scaling toward Full VRAM
    scenarios = [
        {
            "name": "GPU-Resident (Full VRAM)",
            "hbm_mb": 2500,
            "dram_mb": 500,
            "cxl_mb": 1000,
            "prefetch": False,
            "execution_mode": "weight_transfer",
            "lookahead": False,
        },
        {
            "name": "Two-Tier (Weight Transfer 600M)",
            "hbm_mb": 600,
            "dram_mb": 1500,
            "cxl_mb": 1000,
            "prefetch": False,
            "execution_mode": "weight_transfer",
            "lookahead": False,
        },
        {
            "name": "Lookahead Pre-Gating (600MB)",
            "hbm_mb": 600,
            "dram_mb": 1500,
            "cxl_mb": 1000,
            "prefetch": True,
            "execution_mode": "weight_transfer",
            "lookahead": True,
        },
        {
            "name": "Hybrid SOTA (Fiddler 600MB)",
            "hbm_mb": 600,
            "dram_mb": 1500,
            "cxl_mb": 1000,
            "prefetch": False,
            "execution_mode": "hybrid",
            "lookahead": False,
        },
        {
            "name": "Two-Tier (Weight Transfer 900M)",
            "hbm_mb": 900,
            "dram_mb": 1500,
            "cxl_mb": 1000,
            "prefetch": False,
            "execution_mode": "weight_transfer",
            "lookahead": False,
        },
        {
            "name": "Hybrid SOTA (Fiddler 900MB)",
            "hbm_mb": 900,
            "dram_mb": 1500,
            "cxl_mb": 1000,
            "prefetch": False,
            "execution_mode": "hybrid",
            "lookahead": False,
        },
        {
            "name": "Hybrid SOTA (Near Full 1500MB)",
            "hbm_mb": 1500,
            "dram_mb": 1500,
            "cxl_mb": 1000,
            "prefetch": False,
            "execution_mode": "hybrid",
            "lookahead": False,
        },
    ]

    print(f"\n[3/3] Executing {len(scenarios)} live inference benchmarks...")
    results = []
    for sc in scenarios:
        res = run_benchmark_scenario(
            name=sc["name"],
            base_model=base_model,
            tokenizer=tokenizer,
            prompt=BENCHMARK_PROMPT,
            hbm_budget_mb=sc["hbm_mb"],
            dram_budget_mb=sc["dram_mb"],
            cxl_budget_mb=sc["cxl_mb"],
            enable_prefetch=sc.get("prefetch", False),
            co_model=co_model,
            max_new_tokens=args.tokens,
            execution_mode=sc.get("execution_mode", "weight_transfer"),
            enable_lookahead_gating=sc.get("lookahead", False),
        )
        results.append(res)

    # Print Formatted Table
    print("\n" + "=" * 105)
    print("LIVE HARDWARE INFERENCE BENCHMARK RESULTS (NVIDIA GPU)")
    print("=" * 105)
    header = f"{'Configuration':<32} {'HBM(MB)':<8} {'HitRate':<8} {'Hits':<6} {'Miss':<6} {'Evict':<6} {'tok/s':<8} {'Xfer(MB)':<9} {'Wall(s)':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['scenario']:<32} "
            f"{r['hbm_budget_mb']:<8} "
            f"{r['hit_rate']:<8.1%} "
            f"{r['cache_hits']:<6} "
            f"{r['cache_misses']:<6} "
            f"{r['evictions']:<6} "
            f"{r['tokens_per_second']:<8.2f} "
            f"{r['transfer_mb']:<9.1f} "
            f"{r['wall_time_seconds']:<8.2f}"
        )
    print("=" * 105)

    # Save results JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved live benchmark results to: {args.output_json}")

    # Generate Chart
    plot_live_throughput(results, output_path=args.output_plot)


if __name__ == "__main__":
    main()
