#!/usr/bin/env python3
"""Parametric CXL Memory Capacity & Spill-Over Sweep on Physical GPU.

Quantifies how inference throughput and memory tier access distributions evolve
as Host DRAM capacity is progressively constrained below the model's footprint,
forcing working set experts into the software-emulated CXL tier.

Evaluated on:
  - Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6GB GDDR6)
  - Model: Qwen1.5-4x0.5B-Chat-MoE (Qwen-MoE architecture)
  - Fixed GPU VRAM: 400 MB (~24 experts)
  - Variable Host DRAM: [1200, 800, 600, 400, 200] MB
  - Emulated CXL Pool: Sized to accommodate remaining overflow experts
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
from memtier_moe.core.types import MemoryTier
from memtier_moe.runtime.tiered_model import TieredMoEWrapper

BENCHMARK_PROMPT = (
    "Mixture of Experts architecture enables efficient scaling of neural parameters "
    "by selectively routing inputs through specialized feed-forward layers."
)

LOCAL_CHAT_MOE = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen1.5-4x0.5B-Chat-MoE")
DEFAULT_CXL_MODEL = LOCAL_CHAT_MOE if os.path.exists(LOCAL_CHAT_MOE) else "Qwen/Qwen1.5-MoE-A2.7B"


def run_capacity_sweep(
    model_id: str = DEFAULT_CXL_MODEL,
    tokens_per_run: int = 10,
    num_trials: int = 5,
    output_json: str = "results/cxl_capacity_sweep_results.json",
    output_plot: str = "results/cxl_capacity_curve.png",
) -> List[Dict[str, Any]]:
    print("=" * 85)
    print("MEMTIER-MOE: PARAMETRIC CXL CAPACITY & PROGRESSIVE SPILL-OVER SWEEP")
    print("=" * 85)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device:       {device.upper()} ({device_name})")
    print(f"Model:        {model_id}")
    print(f"Tokens/Run:   {tokens_per_run}")
    print(f"Trials/Conf:  {num_trials} repeated trials (reporting Mean ± Std)")

    # 1. Load tokenizer and base model
    print("\n[1/3] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    inputs = tokenizer(BENCHMARK_PROMPT, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # 2. Sweep Configurations (Fixed 400 MB GPU VRAM budget, variable Host DRAM, CXL absorbs overflow)
    # Total active expert weights: 96 * 13.3 MB = 1,276.8 MB
    sweep_configs = [
        {"name": "No CXL Spill (Full DRAM)", "hbm_mb": 400, "dram_mb": 1200, "cxl_mb": 400},
        {"name": "Mild CXL Spill",           "hbm_mb": 400, "dram_mb": 800,  "cxl_mb": 600},
        {"name": "Moderate CXL Spill",       "hbm_mb": 400, "dram_mb": 600,  "cxl_mb": 600},
        {"name": "Heavy CXL Spill",          "hbm_mb": 400, "dram_mb": 400,  "cxl_mb": 800},
        {"name": "Extreme CXL Spill",        "hbm_mb": 400, "dram_mb": 200,  "cxl_mb": 1000},
    ]

    print(f"\n[2/3] Executing {len(sweep_configs)} memory capacity scenarios across {num_trials} trials each...")
    results = []

    for cfg in sweep_configs:
        name = cfg["name"]
        hbm_mb = cfg["hbm_mb"]
        dram_mb = cfg["dram_mb"]
        cxl_mb = cfg["cxl_mb"]
        total_fast_mem = hbm_mb + dram_mb

        print(f"\n--- Scenario: {name} ---")
        print(f"  GPU Budget: {hbm_mb} MB | Host DRAM: {dram_mb} MB | Fast Mem: {total_fast_mem} MB | CXL Pool: {cxl_mb} MB")

        config = MemTierConfig(
            gpu_vram_bytes=6 * 1024 * 1024 * 1024,
            hbm_cache_budget_bytes=hbm_mb * 1024 * 1024,
            host_dram_bytes=dram_mb * 1024 * 1024,
            cxl_memory_bytes=cxl_mb * 1024 * 1024,
        )

        tiered_model = TieredMoEWrapper(
            model=base_model,
            config=config,
            enable_prefetch=False,
            initial_hbm_budget_bytes=hbm_mb * 1024 * 1024,
            execution_mode="hybrid",
        )

        # Count initial placement
        tier_counts = {MemoryTier.HBM: 0, MemoryTier.DRAM: 0, MemoryTier.CXL: 0}
        for eid, meta in tiered_model.engine.tier_manager._registry.items():
            tier_counts[meta.current_tier] += 1

        print(f"  Placement: HBM={tier_counts[MemoryTier.HBM]} exp | DRAM={tier_counts[MemoryTier.DRAM]} exp | CXL={tier_counts[MemoryTier.CXL]} exp")

        # Warmup pass (2 tokens)
        with torch.no_grad():
            tiered_model.generate(**inputs, max_new_tokens=2, do_sample=False)

        # Repeated measurement trials
        trial_throughputs: List[float] = []
        trial_wall_times: List[float] = []
        last_rep = None

        for t in range(num_trials):
            # Reset counters for the timed measurement trial
            tiered_model.engine.metrics = tiered_model.engine.metrics.__class__()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                output_ids = tiered_model.generate(
                    **inputs,
                    max_new_tokens=tokens_per_run,
                    do_sample=False,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_time = time.perf_counter() - t0

            gen_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
            tok_s = gen_tokens / wall_time if wall_time > 0 else 0.0
            trial_throughputs.append(tok_s)
            trial_wall_times.append(wall_time)
            last_rep = tiered_model.report()
            print(f"    Trial {t + 1}/{num_trials}: {tok_s:.2f} tok/s ({wall_time:.3f}s)")

        mean_tok_s = float(np.mean(trial_throughputs))
        std_tok_s = float(np.std(trial_throughputs))
        mean_wall_time = float(np.mean(trial_wall_times))

        m = last_rep["metrics"]
        h_hbm = m.get("hbm_hits", 0)
        h_dram = m.get("dram_hits", 0)
        h_cxl = m.get("cxl_hits", 0)
        total_hits = h_hbm + h_dram + h_cxl
        rate_hbm = (h_hbm / total_hits) if total_hits > 0 else 0.0
        rate_dram = (h_dram / total_hits) if total_hits > 0 else 0.0
        rate_cxl = (h_cxl / total_hits) if total_hits > 0 else 0.0

        cxl_vol_mb = round(h_cxl * 17.301504, 2)

        res = {
            "scenario": name,
            "hbm_budget_mb": hbm_mb,
            "dram_budget_mb": dram_mb,
            "fast_memory_mb": total_fast_mem,
            "cxl_budget_mb": cxl_mb,
            "cxl_experts_placed": tier_counts[MemoryTier.CXL],
            "tokens_generated": tokens_per_run,
            "num_trials": num_trials,
            "mean_wall_time_seconds": round(mean_wall_time, 3),
            "mean_tokens_per_second": round(mean_tok_s, 2),
            "std_tokens_per_second": round(std_tok_s, 2),
            "trial_throughputs": [round(x, 2) for x in trial_throughputs],
            "total_expert_accesses": total_hits,
            "hbm_hits": h_hbm,
            "dram_hits": h_dram,
            "cxl_hits": h_cxl,
            "hbm_hit_rate": round(rate_hbm, 4),
            "dram_hit_rate": round(rate_dram, 4),
            "cxl_hit_rate": round(rate_cxl, 4),
            "calculated_cxl_read_volume_mb": cxl_vol_mb,
            "disk_page_faults": m.get("disk_faults", 0),
        }
        results.append(res)
        print(f"  Summary: {mean_tok_s:.2f} ± {std_tok_s:.2f} tok/s | Hits: HBM={rate_hbm:.1%} DRAM={rate_dram:.1%} CXL={rate_cxl:.1%} ({h_cxl} hits, {cxl_vol_mb:.1f} MB)")

        tiered_model.unpatch()

    # 3. Print Markdown Table
    print("\n" + "=" * 115)
    print("PARAMETRIC CXL CAPACITY & PROGRESSIVE SPILL-OVER RESULTS (RTX 4050 GPU, N=5 TRIALS)")
    print("=" * 115)
    header = f"{'Scenario':<26} {'GPU(MB)':<8} {'DRAM(MB)':<9} {'CXL(MB)':<8} {'CXL Exp':<8} {'CXL Hits':<9} {'CXL Rate':<9} {'CXL Vol(MB)':<12} {'Mean tok/s ± Std':<18}"
    print(header)
    print("-" * len(header))
    for r in results:
        t_str = f"{r['mean_tokens_per_second']:.2f} ± {r['std_tokens_per_second']:.2f}"
        print(
            f"{r['scenario']:<26} "
            f"{r['hbm_budget_mb']:<8} "
            f"{r['dram_budget_mb']:<9} "
            f"{r['cxl_budget_mb']:<8} "
            f"{r['cxl_experts_placed']:<8} "
            f"{r['cxl_hits']:<9} "
            f"{r['cxl_hit_rate']:<9.1%} "
            f"{r['calculated_cxl_read_volume_mb']:<12.1f} "
            f"{t_str:<18}"
        )
    print("=" * 115)

    # 4. Save JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved capacity sweep results to: {output_json}")

    # 5. Generate Publication Plot
    plot_capacity_curve(results, output_plot)
    return results


def plot_capacity_curve(results: List[Dict[str, Any]], output_path: str):
    """Plot the memory capacity vs performance and CXL spill curve."""
    dram_caps = [r["dram_budget_mb"] for r in results]
    mean_throughputs = [r["mean_tokens_per_second"] for r in results]
    std_throughputs = [r["std_tokens_per_second"] for r in results]
    hbm_rates = [r["hbm_hit_rate"] * 100 for r in results]
    dram_rates = [r["dram_hit_rate"] * 100 for r in results]
    cxl_rates = [r["cxl_hit_rate"] * 100 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Throughput Curve with Error Bars as DRAM Capacity Drops
    ax1.errorbar(
        dram_caps,
        mean_throughputs,
        yerr=std_throughputs,
        fmt="-o",
        color="#1f77b4",
        ecolor="#d62728",
        elinewidth=1.8,
        capsize=5,
        capthick=1.8,
        linewidth=2.2,
        markersize=7,
        label="Throughput (Mean ± Std)",
    )
    ax1.set_xlabel("Host DRAM Budget (MB)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Throughput (tokens / second)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.set_title("Inference Throughput vs. Host DRAM Budget (Mean ± Std, N=5)\n(Fixed 400 MB GPU VRAM Budget, Qwen1.5-4x0.5B on RTX 4050)", fontsize=11, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.invert_xaxis()  # Shows progression from abundant DRAM (1200MB) to constrained DRAM (200MB)

    for x, mean_y, std_y in zip(dram_caps, mean_throughputs, std_throughputs):
        ax1.annotate(
            f"{mean_y:.2f} ± {std_y:.2f}",
            (x, mean_y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel 2: Stacked Tier Access Distribution (%)
    x_indices = np.arange(len(dram_caps))
    bar_width = 0.55

    ax2.bar(x_indices, hbm_rates, bar_width, label="GPU VRAM (HBM)", color="#2ca02c", edgecolor="#222")
    ax2.bar(x_indices, dram_rates, bar_width, bottom=hbm_rates, label="Host DRAM", color="#ff7f0e", edgecolor="#222")
    bottom_cxl = np.array(hbm_rates) + np.array(dram_rates)
    ax2.bar(x_indices, cxl_rates, bar_width, bottom=bottom_cxl, label="Software-Emulated CXL (350ns, 8GB/s)", color="#d62728", edgecolor="#222")

    ax2.set_xlabel("Scenario (Host DRAM Budget)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Tier Access Distribution (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Progressive Tier Access Migration\n(Fast Memory Saturation & CXL Working-Set Absorption)", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"{d} MB" for d in dram_caps], fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9)

    for i, (cxl_pct, cxl_hit) in enumerate(zip(cxl_rates, [r["cxl_hits"] for r in results])):
        if cxl_pct > 0:
            ax2.text(
                x_indices[i],
                bottom_cxl[i] + (cxl_pct / 2) - 3,
                f"{cxl_pct:.1f}%\n({cxl_hit})",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved capacity curve plot to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CXL Memory Capacity Curve Benchmark")
    parser.add_argument("--tokens", type=int, default=10, help="Tokens to generate per trial")
    parser.add_argument("--trials", type=int, default=5, help="Repeated trials per configuration")
    parser.add_argument("--output-json", type=str, default="results/cxl_capacity_sweep_results.json")
    parser.add_argument("--output-plot", type=str, default="results/cxl_capacity_curve.png")
    args = parser.parse_args()

    run_capacity_sweep(
        tokens_per_run=args.tokens,
        num_trials=args.trials,
        output_json=args.output_json,
        output_plot=args.output_plot,
    )
