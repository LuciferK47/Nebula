#!/usr/bin/env python3
"""GPU scenario suite for MemTier-MoE (Nebula) fidelity + capability probes.

Companion to the "Emulation Fidelity Verdict + GPU Scenario Suite" plan.
Each scenario is a self-contained probe answering one specific question
about the system; they share one loaded model to fit the whole core
suite (S1-S7, S9) in well under 30 minutes on an RTX 4050. S8 (scale
validation) reuses the same functions with different --model-id/--budget
flags on a larger GPU (A100/L40S).

Every result dict carries explicit "computation_type" / "transfer_latency_type"
badges (measured vs. modeled) so the frontend never renders a modeled
number next to a measured one without saying so — see benchmark_runner.py's
BenchmarkResult for the same convention on the pure-simulation ablation
path.

Scenarios:
  s1  Capacity cliff        — HBM budget sweep, hybrid mode, headline chart
  s2  Execution-mode        — weight_transfer / +prefetch / +lookahead / hybrid crossover
  s3  CXL sensitivity       — cxl_bandwidth_gbps x cxl_emulation_mode sweep (the paper defense)
  s4  Domain shift / OOD    — WikiText-calibrated placement run cross-domain
  s5  Long-horizon          — 500 tokens, windowed hit-rate/throughput time series
  s6  Correctness           — vs. full-VRAM reference: prefix match, divergence, logit delta
  s7  Batch scaling         — hybrid's edge vs. weight_transfer as batch size grows
  s9  Overlap proof         — re-run verify_overlap.py, save CUDA-event + kernel-trace verdict
  s10 Edge cases            — degenerate budgets, missing tiers, VRAM over-commit, leak checks (runs last)
  s8  Scale validation      — same functions, run with --model-id/--hbm-budgets for a bigger model

Usage:
    python scripts/run_scenarios.py --scenario s1
    python scripts/run_scenarios.py --scenario all
    python scripts/run_scenarios.py --scenario s1 --model-id <bigger-model> --hbm-budgets 3000,6000,...  # S8 on A100/L40S
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memtier_moe.core.config import MemTierConfig
from memtier_moe.runtime.tiered_model import TieredMoEWrapper
from scripts.run_live_benchmark import load_or_calibrate_co_occurrence

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "scenarios")
LOCAL_CHAT_MOE = os.path.join(REPO_ROOT, "models", "Qwen1.5-4x0.5B-Chat-MoE")
DEFAULT_MODEL_ID = LOCAL_CHAT_MOE if os.path.exists(LOCAL_CHAT_MOE) else "Qwen/Qwen1.5-MoE-A2.7B"
DEFAULT_TRACE = os.path.join(REPO_ROOT, "traces", "routing_trace_wikitext.npz")
DEFAULT_PROMPT = "Explain the fundamental principles of hierarchical memory tiering in high-performance computing."

GB = 1024 * 1024 * 1024
MB = 1024 * 1024


# ── Shared setup ─────────────────────────────────────────────────────────


def load_resources(model_id: str, trace_path: str):
    """Load tokenizer, base model, and the WikiText-calibrated co-occurrence
    profile once, for reuse across every scenario in a run."""
    print(f"[Scenarios] Loading tokenizer/model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True,
    )
    base_model.eval()

    print(f"[Scenarios] Loading/calibrating co-occurrence profile from {trace_path}")
    co_model = load_or_calibrate_co_occurrence(trace_path, model=base_model, tokenizer=tokenizer)
    freq_map = getattr(getattr(co_model, "stats", None), "marginal_counts", None)

    return base_model, tokenizer, co_model, freq_map


def run_generation(
    base_model,
    tokenizer,
    co_model,
    freq_map,
    prompt,
    hbm_mb: int,
    dram_mb: int = 1500,
    cxl_mb: int = 1500,
    execution_mode: str = "hybrid",
    enable_prefetch: bool = False,
    enable_lookahead: bool = False,
    max_new_tokens: int = 25,
    cxl_bandwidth_gbps: Optional[float] = None,
    cxl_emulation_mode: Optional[str] = None,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """One tagged generation run. Every field a caller might need for a
    scenario JSON is produced here so scenario functions stay thin
    wrappers around a sweep of parameters."""
    config_kwargs: Dict[str, Any] = dict(
        gpu_vram_bytes=6 * GB,
        hbm_cache_budget_bytes=hbm_mb * MB,
        host_dram_bytes=dram_mb * MB,
        cxl_memory_bytes=cxl_mb * MB,
        max_prefetches_per_decision=2,
    )
    if cxl_bandwidth_gbps is not None:
        config_kwargs["cxl_bandwidth_gbps"] = cxl_bandwidth_gbps
    if cxl_emulation_mode is not None:
        config_kwargs["cxl_emulation_mode"] = cxl_emulation_mode
    config = MemTierConfig(**config_kwargs)

    wrapper = TieredMoEWrapper(
        model=base_model,
        config=config,
        enable_prefetch=enable_prefetch,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=hbm_mb * MB,
        execution_mode=execution_mode,
        enable_lookahead_gating=enable_lookahead,
        expert_frequency=freq_map,
    )

    if isinstance(prompt, list):
        inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = wrapper.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=1.15,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0

    batch_size = inputs["input_ids"].shape[0]
    gen_tokens_per_seq = output_ids.shape[1] - inputs["input_ids"].shape[1]
    total_gen_tokens = gen_tokens_per_seq * batch_size
    tok_per_sec = total_gen_tokens / wall_time if wall_time > 0 else 0.0

    report = wrapper.report()
    m = report["metrics"]
    sched = report.get("scheduler_stats", {})
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    result = {
        "hbm_budget_mb": hbm_mb,
        "dram_budget_mb": dram_mb,
        "cxl_budget_mb": cxl_mb,
        "execution_mode": execution_mode,
        "enable_prefetch": enable_prefetch,
        "enable_lookahead": enable_lookahead,
        "cxl_bandwidth_gbps": cxl_bandwidth_gbps,
        "cxl_emulation_mode": cxl_emulation_mode,
        "batch_size": batch_size,
        "generated_tokens_per_seq": int(gen_tokens_per_seq),
        "total_generated_tokens": int(total_gen_tokens),
        "wall_time_seconds": round(wall_time, 4),
        "tokens_per_second": round(tok_per_sec, 2),
        "hit_rate": round(m.get("hit_rate", 0.0), 4),
        "hbm_hit_rate": round(m.get("hbm_hit_rate", 0.0), 4),
        "dram_hit_rate": round(m.get("dram_hit_rate", 0.0), 4),
        "cxl_hit_rate": round(m.get("cxl_hit_rate", 0.0), 4),
        "cache_hits": m.get("cache_hits", 0),
        "cache_misses": m.get("cache_misses", 0),
        "evictions": m.get("evictions", 0),
        "transfer_mb": round(m.get("total_transfer_bytes", 0) / 1e6, 3),
        "prefetch_precision": round(sched.get("prefetch_precision", 0.0), 4) if enable_prefetch or enable_lookahead else 0.0,
        "peak_vram_mb": round(peak_vram_mb, 1),
        # Fidelity badges: this whole path is a real HF model on real
        # hardware — compute and DRAM<->HBM transfers are measured;
        # only the CXL tier's timing (never its placement) is modeled.
        "benchmark_mode": "live",
        "computation_type": "measured",
        "transfer_latency_type": "measured+modeled_cxl",
    }

    wrapper.unpatch()
    return result


def _save(scenario_id: str, payload: Dict[str, Any], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scenario_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Scenarios] {scenario_id}: wrote {path}")
    return path


# ── S1: Capacity cliff ───────────────────────────────────────────────────


def scenario_s1(ctx, out_dir: str) -> Dict[str, Any]:
    budgets = [200, 300, 450, 600, 750, 900, 1200, 1500, 2000, 2500]
    runs = []
    for b in budgets:
        print(f"  [S1] HBM budget {b} MB...")
        runs.append(run_generation(
            *ctx.resources, prompt=ctx.prompt, hbm_mb=b, dram_mb=1500, cxl_mb=1500,
            execution_mode="hybrid", max_new_tokens=ctx.tokens,
        ))
    return {"scenario": "s1_capacity_cliff", "description": "HBM budget sweep, hybrid mode: where does tiering stop being free?", "runs": runs}


# ── S2: Execution-mode crossover ────────────────────────────────────────


def scenario_s2(ctx, out_dir: str) -> Dict[str, Any]:
    modes = [
        dict(label="weight_transfer", execution_mode="weight_transfer", enable_prefetch=False, enable_lookahead=False),
        dict(label="weight_transfer+prefetch", execution_mode="weight_transfer", enable_prefetch=True, enable_lookahead=False),
        dict(label="weight_transfer+lookahead", execution_mode="weight_transfer", enable_prefetch=True, enable_lookahead=True),
        dict(label="hybrid", execution_mode="hybrid", enable_prefetch=False, enable_lookahead=False),
        dict(label="hybrid+lookahead", execution_mode="hybrid", enable_prefetch=False, enable_lookahead=True),
    ]
    runs = []
    for hbm_mb in (600, 900):
        for m in modes:
            print(f"  [S2] {hbm_mb}MB / {m['label']}...")
            r = run_generation(
                *ctx.resources, prompt=ctx.prompt, hbm_mb=hbm_mb, dram_mb=1500, cxl_mb=1500,
                execution_mode=m["execution_mode"], enable_prefetch=m["enable_prefetch"],
                enable_lookahead=m["enable_lookahead"], max_new_tokens=ctx.tokens,
            )
            r["mode_label"] = m["label"]
            runs.append(r)
    return {"scenario": "s2_execution_mode_crossover", "description": "Ablation of prefetch/lookahead/hybrid at fixed HBM budgets.", "runs": runs}


# ── S3: CXL sensitivity (the paper-defense scenario) ────────────────────


def scenario_s3(ctx, out_dir: str) -> Dict[str, Any]:
    """Sweep CXL bandwidth assumption and emulation fidelity level. If the
    ranking of baselines is stable across this sweep, the architectural
    conclusion does not depend on the CXL emulation's calibration
    accuracy — a stronger claim than validating against a cycle-accurate
    simulator, since it bounds sensitivity to the *entire* plausible
    range rather than to one simulator's specific assumptions.

    Also exports the sweep's raw memory transactions in DRAMSim3/gem5
    trace format (memtier_moe/memory/trace_exporter.py) — "ship the
    trace, don't run the simulator": anyone who wants to cross-check the
    bandwidth model against a cycle-accurate simulator can do so offline
    from this file, without making either a runtime dependency.
    """
    from memtier_moe.memory.trace_exporter import GLOBAL_TRACE_EXPORTER

    GLOBAL_TRACE_EXPORTER.clear()
    GLOBAL_TRACE_EXPORTER.enable()

    runs = []
    default_bw = 8.0
    for bw in (4.0, 8.0, 16.0, 32.0, 64.0):
        print(f"  [S3] cxl_bandwidth_gbps={bw}, mode=full...")
        runs.append(run_generation(
            *ctx.resources, prompt=ctx.prompt, hbm_mb=600, dram_mb=400, cxl_mb=4000,
            execution_mode="hybrid", max_new_tokens=ctx.tokens,
            cxl_bandwidth_gbps=bw, cxl_emulation_mode="full",
        ))
    for mode in ("disabled", "latency_only"):
        print(f"  [S3] cxl_bandwidth_gbps={default_bw}, mode={mode}...")
        runs.append(run_generation(
            *ctx.resources, prompt=ctx.prompt, hbm_mb=600, dram_mb=400, cxl_mb=4000,
            execution_mode="hybrid", max_new_tokens=ctx.tokens,
            cxl_bandwidth_gbps=default_bw, cxl_emulation_mode=mode,
        ))

    GLOBAL_TRACE_EXPORTER.stop_tracing()
    dramsim3_path = os.path.join(out_dir, "s3_trace.dramsim3")
    gem5_path = os.path.join(out_dir, "s3_trace_gem5.txt")
    csv_path = os.path.join(out_dir, "s3_trace.csv")
    os.makedirs(out_dir, exist_ok=True)
    n_dramsim3 = GLOBAL_TRACE_EXPORTER.export_dramsim3(dramsim3_path)
    n_gem5 = GLOBAL_TRACE_EXPORTER.export_gem5(gem5_path)
    n_csv = GLOBAL_TRACE_EXPORTER.export_csv(csv_path)

    hit_rates = {r["cxl_bandwidth_gbps"]: r["hit_rate"] for r in runs if r["cxl_emulation_mode"] == "full"}
    placement_stable = len(set(round(v, 6) for v in hit_rates.values())) == 1
    return {
        "scenario": "s3_cxl_sensitivity",
        "description": "cxl_bandwidth_gbps x cxl_emulation_mode sweep. hit_rate must stay constant across all "
                        "runs (emulation affects timing only, never placement) — see test_emulation_mode_parity.py "
                        "for the pinned CPU-only version of this same contract.",
        "placement_stable_across_bandwidth_sweep": placement_stable,
        "runs": runs,
        "exported_traces": {
            "dramsim3": {"path": os.path.relpath(dramsim3_path, REPO_ROOT), "transactions": n_dramsim3},
            "gem5": {"path": os.path.relpath(gem5_path, REPO_ROOT), "transactions": n_gem5},
            "csv": {"path": os.path.relpath(csv_path, REPO_ROOT), "transactions": n_csv},
        },
    }


# ── S4: Domain shift / OOD ───────────────────────────────────────────────

_DOMAIN_PROMPTS = [
    ("prose", "The history of the printing press reshaped how knowledge spread across medieval Europe."),
    ("python_code", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n"),
    ("mathematics", "The Riemann Hypothesis posits that all non-trivial zeros of the zeta function possess a real coordinate of 1/2."),
    ("multilingual", "La inteligencia artificial está transformando la manera en que las empresas procesan datos."),
    ("dialogue", "\"I don't think we should trust him,\" she said, glancing back at the empty hallway."),
]


def scenario_s4(ctx, out_dir: str) -> Dict[str, Any]:
    runs = []
    for domain, text in _DOMAIN_PROMPTS:
        print(f"  [S4] domain={domain}...")
        r = run_generation(
            *ctx.resources, prompt=text, hbm_mb=600, dram_mb=1500, cxl_mb=1500,
            execution_mode="hybrid", max_new_tokens=ctx.tokens,
        )
        r["domain"] = domain
        runs.append(r)
    return {
        "scenario": "s4_domain_shift",
        "description": "Profile-guided placement was calibrated on WikiText; how much does hit rate degrade OOD?",
        "wikitext_calibrated_hit_rate": next((r["hit_rate"] for r in runs if r["domain"] == "prose"), None),
        "runs": runs,
    }


# ── S5: Long-horizon stability ───────────────────────────────────────────


def scenario_s5(ctx, out_dir: str, num_tokens: int = 500, window: int = 50) -> Dict[str, Any]:
    """500-token generation at a constrained budget, with hit-rate and
    throughput reported in rolling windows via MetricsTracker.snapshot()/
    windowed_rates() rather than only a single cumulative number — shows
    whether the LFU decay converges or thrashes over a long run."""
    from memtier_moe.core.metrics import MetricsTracker

    base_model, tokenizer, co_model, freq_map = ctx.resources
    config = MemTierConfig(
        gpu_vram_bytes=6 * GB, hbm_cache_budget_bytes=600 * MB,
        host_dram_bytes=1500 * MB, cxl_memory_bytes=1500 * MB,
        max_prefetches_per_decision=2,
    )
    wrapper = TieredMoEWrapper(
        model=base_model, config=config, co_occurrence_model=co_model,
        initial_hbm_budget_bytes=600 * MB, execution_mode="hybrid", expert_frequency=freq_map,
    )

    inputs = tokenizer(ctx.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    windows: List[Dict[str, Any]] = []
    prev_snapshot = wrapper.metrics.snapshot()
    prev_wall = time.perf_counter()

    remaining = num_tokens
    current_ids = inputs
    generated_total = 0
    # Generate in `window`-token chunks so we can snapshot metrics between
    # chunks. Each chunk continues from the previous chunk's full sequence.
    full_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    while remaining > 0:
        step = min(window, remaining)
        gen_kwargs = dict(input_ids=full_ids, max_new_tokens=step, do_sample=False, repetition_penalty=1.15)
        if attn is not None:
            gen_kwargs["attention_mask"] = attn
        with torch.no_grad():
            out = wrapper.generate(**gen_kwargs)
        full_ids = out
        if attn is not None:
            pad = torch.ones((attn.shape[0], out.shape[1] - attn.shape[1]), dtype=attn.dtype, device=attn.device)
            attn = torch.cat([attn, pad], dim=1)

        now_wall = time.perf_counter()
        curr_snapshot = wrapper.metrics.snapshot()
        rates = MetricsTracker.windowed_rates(prev_snapshot, curr_snapshot)
        window_wall_s = now_wall - prev_wall
        windows.append({
            "token_range": [generated_total, generated_total + step],
            "hit_rate": round(rates["hit_rate"], 4),
            "hbm_hit_rate": round(rates["hbm_hit_rate"], 4),
            "dram_hit_rate": round(rates["dram_hit_rate"], 4),
            "cxl_hit_rate": round(rates["cxl_hit_rate"], 4),
            "evictions_in_window": rates["evictions"],
            "tokens_per_second": round(step / window_wall_s, 2) if window_wall_s > 0 else 0.0,
        })
        prev_snapshot = curr_snapshot
        prev_wall = now_wall
        generated_total += step
        remaining -= step

    report = wrapper.report()
    wrapper.unpatch()
    return {
        "scenario": "s5_long_horizon_stability",
        "description": f"{num_tokens} tokens at 600MB HBM budget, hybrid mode, {window}-token rolling windows.",
        "num_tokens": num_tokens,
        "window_size": window,
        "cumulative_hit_rate": round(report["metrics"].get("hit_rate", 0.0), 4),
        "cumulative_evictions": report["metrics"].get("evictions", 0),
        "windows": windows,
        "benchmark_mode": "live", "computation_type": "measured",
    }


# ── S6: Correctness vs full VRAM ─────────────────────────────────────────

_S6_PROMPTS = [
    "The Mixture of Experts architecture routes tokens through specialized feed-forward layers",
    "In distributed systems, consensus protocols ensure agreement despite node failures",
    "Photosynthesis converts light energy into chemical energy stored in glucose",
    "The French Revolution fundamentally altered the political landscape of Europe",
    "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n",
]


def scenario_s6(ctx, out_dir: str) -> Dict[str, Any]:
    base_model, tokenizer, co_model, freq_map = ctx.resources
    per_prompt = []

    for prompt in _S6_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            ref_out = base_model.generate(
                **inputs, max_new_tokens=ctx.tokens, do_sample=False, repetition_penalty=1.1,
                output_scores=True, return_dict_in_generate=True,
            )

        config = MemTierConfig(
            gpu_vram_bytes=6 * GB, hbm_cache_budget_bytes=900 * MB,
            host_dram_bytes=1500 * MB, cxl_memory_bytes=1500 * MB,
        )
        wrapper = TieredMoEWrapper(
            model=base_model, config=config, co_occurrence_model=co_model,
            initial_hbm_budget_bytes=900 * MB, execution_mode="hybrid", expert_frequency=freq_map,
        )
        with torch.no_grad():
            test_out = wrapper.generate(
                **inputs, max_new_tokens=ctx.tokens, do_sample=False, repetition_penalty=1.1,
                output_scores=True, return_dict_in_generate=True,
            )
        wrapper.unpatch()

        ref_ids = ref_out.sequences[0].tolist()
        test_ids = test_out.sequences[0].tolist()

        first_divergence = None
        for i, (a, b) in enumerate(zip(ref_ids, test_ids)):
            if a != b:
                first_divergence = i
                break
        prefix_match_len = first_divergence if first_divergence is not None else min(len(ref_ids), len(test_ids))

        max_abs_logit_delta = None
        if ref_out.scores and test_out.scores:
            # Only compare steps up to and including the first divergence:
            # scores[i] is the distribution used to pick token i, and both
            # runs share an identical prefix through that point. Past
            # divergence the two runs are conditioned on different
            # histories, so their scores stop being a same-context
            # comparison and comparing them would measure "different
            # input" rather than "different computation on the same
            # input" — not the numerical-drift bound this is meant to be.
            n_common = min(len(ref_out.scores), len(test_out.scores))
            if first_divergence is not None:
                n_common = min(n_common, first_divergence + 1)
            deltas = [
                (ref_out.scores[i].float() - test_out.scores[i].float()).abs().max().item()
                for i in range(n_common)
            ]
            max_abs_logit_delta = max(deltas) if deltas else None

        per_prompt.append({
            "prompt": prompt[:60],
            "total_tokens": len(ref_ids),
            "exact_prefix_match_length": prefix_match_len,
            "first_divergence_index": first_divergence,
            "max_abs_logit_delta": max_abs_logit_delta,
            "has_nan": bool(torch.isnan(test_out.sequences.float()).any().item()),
            "has_inf": bool(torch.isinf(test_out.sequences.float()).any().item()),
        })

    return {
        "scenario": "s6_correctness_vs_full_vram",
        "description": "Stricter than an elementwise token-id comparison: exact-prefix-match length and "
                        "first-divergence index catch a single early wrong token instead of averaging it "
                        "away across a long generation; max_abs_logit_delta bounds numerical drift directly.",
        "hbm_budget_mb": 900,
        "runs": per_prompt,
        "benchmark_mode": "live", "computation_type": "measured",
    }


# ── S7: Batch scaling / sparsity erosion ─────────────────────────────────


def scenario_s7(ctx, out_dir: str) -> Dict[str, Any]:
    all_prompts = [
        "Artificial intelligence revolutionizes data science.",
        "Quantum computing fundamentally accelerates cryptography.",
        "Distributed database systems provide fault tolerance.",
        "Compiler optimization maximizes pipeline parallelism.",
        "Neural networks approximate arbitrary continuous functions.",
        "Cryptographic hashing ensures data integrity at scale.",
        "Operating systems schedule processes across CPU cores.",
        "Version control systems track collaborative code changes.",
    ]
    runs = []
    for bsz in (1, 2, 4, 8):
        for mode in ("hybrid", "weight_transfer"):
            print(f"  [S7] batch={bsz}, mode={mode}...")
            r = run_generation(
                *ctx.resources, prompt=all_prompts[:bsz], hbm_mb=600, dram_mb=1500, cxl_mb=1500,
                execution_mode=mode, max_new_tokens=15,
            )
            runs.append(r)
    return {
        "scenario": "s7_batch_scaling",
        "description": "Hybrid's per-token edge over weight_transfer should erode as batching activates more "
                        "distinct experts per layer, reducing the sparsity hybrid exploits. An honest negative "
                        "result here strengthens the paper's scope claims rather than weakening them.",
        "runs": runs,
    }


# ── S9: Overlap proof ─────────────────────────────────────────────────────


def scenario_s9(ctx, out_dir: str) -> Dict[str, Any]:
    from scripts.verify_overlap import verify_overlap, analyse_chrome_trace

    if not torch.cuda.is_available():
        return {"scenario": "s9_overlap_proof", "status": "SKIPPED", "reason": "CUDA not available"}

    result = verify_overlap(matrix_size=2048, expert_size_mb=32)
    if result is False:
        return {"scenario": "s9_overlap_proof", "status": "FAIL", "reason": "verify_overlap returned False"}

    prof, _, _, timing = result
    trace_path = os.path.join(out_dir, "s9_overlap_trace.json")
    prof.export_chrome_trace(trace_path)
    stats = analyse_chrome_trace(trace_path)

    timing_overlap = timing["hidden_latency_ms"] > 0.5 * min(timing["compute_alone_ms"], timing["transfer_alone_ms"])
    is_success = stats["overlap_detected"] or timing_overlap

    return {
        "scenario": "s9_overlap_proof",
        "status": "PASS" if is_success else "FAIL",
        "timing_ms": timing,
        "kernel_trace_stats": stats,
        "chrome_trace_path": os.path.relpath(trace_path, REPO_ROOT),
        "benchmark_mode": "live", "computation_type": "measured",
    }


# ── S10: Edge cases / robustness ─────────────────────────────────────────
#
# Unlike S1-S9 (which measure performance on well-behaved inputs), these
# deliberately try to break the system. Each case reports PASS /
# EXPECTED_FAIL / FAIL rather than throughput, and every case is isolated
# in try/except so one crash does not abort the rest.
#
# Why this runs LAST: TieredMoEWrapper._patch_moe_layers() moves expert
# weights between devices *before* it patches any layer (tiered_model.py
# step 3 vs step 4). If place_initial() raises partway through — which
# the degenerate-budget cases below are designed to trigger — the base
# model is left with some experts on CPU and some on GPU, and unpatch()
# is a no-op because no block was ever replaced. That damaged model would
# silently poison every later scenario sharing the same process, so each
# case that is expected to fail is followed by an integrity check against
# the unmodified base model.


def _integrity_check(base_model, tokenizer) -> Dict[str, Any]:
    """Confirm the shared base model still generates after a deliberate
    failure. If this trips, later scenarios in the same process cannot be
    trusted and the suite must be re-run in a fresh process."""
    try:
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            base_model.generate(**inputs, max_new_tokens=3, do_sample=False)
        return {"model_intact": True}
    except Exception as e:
        return {"model_intact": False, "integrity_error": f"{type(e).__name__}: {e}"}


def scenario_s10(ctx, out_dir: str) -> Dict[str, Any]:
    base_model, tokenizer, co_model, freq_map = ctx.resources
    cases: List[Dict[str, Any]] = []
    contaminated = False

    def record(name, description, expect_fail=False, integrity_check=False, **run_kwargs):
        """Run one edge case, classifying an exception as EXPECTED_FAIL when
        the case exists precisely to confirm we fail cleanly."""
        nonlocal contaminated
        print(f"  [S10] {name}...")
        entry: Dict[str, Any] = {"case": name, "description": description, "expected_to_fail": expect_fail}
        try:
            r = run_generation(base_model, tokenizer, co_model, freq_map, **run_kwargs)
            entry["status"] = "UNEXPECTED_PASS" if expect_fail else "PASS"
            entry["result"] = {
                k: r[k] for k in
                ("hbm_budget_mb", "dram_budget_mb", "cxl_budget_mb", "execution_mode",
                 "tokens_per_second", "hit_rate", "cxl_hit_rate", "evictions", "peak_vram_mb")
            }
        except Exception as e:
            entry["status"] = "EXPECTED_FAIL" if expect_fail else "FAIL"
            entry["error"] = f"{type(e).__name__}: {e}"
        if integrity_check:
            check = _integrity_check(base_model, tokenizer)
            entry.update(check)
            if not check["model_intact"]:
                contaminated = True
        cases.append(entry)

    # E1 — HBM budget far below a single layer's top-k working set. Hybrid
    # mode should survive this (it computes cold experts on CPU and never
    # needs their weights in HBM at all), just with a very low hit rate.
    record("e1_tiny_budget_hybrid",
           "50MB HBM budget — smaller than one layer's top-k working set.",
           prompt=ctx.prompt, hbm_mb=50, dram_mb=1500, cxl_mb=1500,
           execution_mode="hybrid", max_new_tokens=10)

    # E2 — Same tiny budget in weight_transfer mode, which *must* land
    # top-k experts in HBM to compute at all. This is the stress test for
    # the "Pinned Active Set" guard against intra-layer mutual eviction.
    record("e2_tiny_budget_weight_transfer",
           "50MB HBM budget in weight_transfer mode — probes top-k eviction pinning.",
           prompt=ctx.prompt, hbm_mb=50, dram_mb=1500, cxl_mb=1500,
           execution_mode="weight_transfer", max_new_tokens=10)

    # E3 — True GPU-resident: no DRAM, no CXL, ample HBM. Historically the
    # very first placement crashed here because setup unconditionally
    # staged through DRAM (see tests/test_memory_pressure.py docstring).
    record("e3_gpu_resident_no_dram_no_cxl",
           "host_dram=0, cxl=0, ample HBM — the documented historical crash case.",
           prompt=ctx.prompt, hbm_mb=2500, dram_mb=0, cxl_mb=0,
           execution_mode="weight_transfer", max_new_tokens=10,
           integrity_check=True)

    # E4 — No tier anywhere can hold the model. Must fail *cleanly* with a
    # RuntimeError from the pool, not hang or half-patch the model.
    record("e4_no_tier_has_room",
           "host_dram=0, cxl=0, HBM too small — must raise cleanly, not hang or corrupt.",
           expect_fail=True, integrity_check=True,
           prompt=ctx.prompt, hbm_mb=100, dram_mb=0, cxl_mb=0,
           execution_mode="weight_transfer", max_new_tokens=5)

    # E5 — HBM budget nominally larger than the card's physical VRAM. The
    # budget is a cap rather than an allocation, and this model's expert set
    # (~1.2GB) fits comfortably, so the expected outcome is a PASS with
    # every expert in HBM: it confirms the budget is not validated against
    # physical VRAM and that an over-large cap degrades to "hold
    # everything" instead of erroring. A model whose experts genuinely
    # exceeded 6GB would OOM here instead — that case belongs to S8 on the
    # larger GPU, not to this suite.
    record("e5_budget_nominally_exceeds_physical_vram",
           "8000MB HBM cap on a 6GB card with a ~1.2GB expert set — cap should simply "
           "mean 'everything in HBM', not an error.",
           integrity_check=True,
           prompt=ctx.prompt, hbm_mb=8000, dram_mb=1500, cxl_mb=1500,
           execution_mode="hybrid", max_new_tokens=5)

    # E6 — All three tiers together sized below the model's expert weights.
    # Without a disk tier, demote_to()'s all-tiers-full fallback releases
    # the tensor while marking the expert resident, which in the live path
    # would silently drop the only copy of real weights and produce wrong
    # output. _patch_moe_layers() now rejects this up front, so the correct
    # outcome is a clean ValueError naming the shortfall — never a run that
    # "succeeds" with corrupted numerics.
    record("e6_total_capacity_below_model_size",
           "HBM+DRAM+CXL = 600MB total for a ~1.2GB expert set — must be refused at "
           "construction rather than silently dropping weights.",
           expect_fail=True, integrity_check=True,
           prompt=ctx.prompt, hbm_mb=200, dram_mb=200, cxl_mb=200,
           execution_mode="hybrid", max_new_tokens=10)

    # E7 — Prefill-dominant: a long prompt with few generated tokens forces
    # the multi-token general dispatch path instead of the optimized
    # single-token decode fast path (which requires flat_hidden.shape[0]==1).
    long_prompt = ("Memory hierarchies matter because capacity and latency trade off against each other. " * 40)
    record("e7_long_prompt_prefill_dominant",
           "~800-token prompt, 5 new tokens — exercises multi-token prefill, not the decode fast path.",
           prompt=long_prompt, hbm_mb=600, dram_mb=1500, cxl_mb=1500,
           execution_mode="hybrid", max_new_tokens=5)

    # E8 — Degenerate single-step generation; nothing downstream should
    # divide by a zero token count.
    record("e8_single_token_generation",
           "max_new_tokens=1 — degenerate single-step decode.",
           prompt=ctx.prompt, hbm_mb=600, dram_mb=1500, cxl_mb=1500,
           execution_mode="hybrid", max_new_tokens=1)

    # E9 — Prefetch abuse: threshold 0.0 means "prefetch everything", the
    # maximal-thrash setting, at a budget tight enough to guarantee
    # eviction races. Must stay bounded by max_inflight_transfers rather
    # than deadlocking or unbounded-queueing.
    print("  [S10] e9_prefetch_threshold_zero...")
    e9: Dict[str, Any] = {
        "case": "e9_prefetch_threshold_zero",
        "description": "prefetch_confidence_threshold=0.0 at a 400MB budget — maximal speculative thrash.",
        "expected_to_fail": False,
    }
    try:
        cfg = MemTierConfig(
            gpu_vram_bytes=6 * GB, hbm_cache_budget_bytes=400 * MB,
            host_dram_bytes=1500 * MB, cxl_memory_bytes=1500 * MB,
            prefetch_confidence_threshold=0.0, max_prefetches_per_decision=8,
        )
        w = TieredMoEWrapper(
            model=base_model, config=cfg, enable_prefetch=True, co_occurrence_model=co_model,
            initial_hbm_budget_bytes=400 * MB, execution_mode="weight_transfer",
            enable_lookahead_gating=False, expert_frequency=freq_map,
        )
        try:
            inputs = tokenizer(ctx.prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            t0 = time.perf_counter()
            with torch.no_grad():
                w.generate(**inputs, max_new_tokens=10, do_sample=False)
            rep = w.report()
            sched = rep.get("scheduler_stats", {})
            e9["status"] = "PASS"
            e9["result"] = {
                "wall_time_seconds": round(time.perf_counter() - t0, 3),
                "evictions": rep["metrics"].get("evictions", 0),
                "prefetch_issued": sched.get("prefetch_total", 0),
                "prefetch_precision": round(sched.get("prefetch_precision", 0.0), 4),
                "inflight_bound_respected": w.engine.transfer_engine.num_inflight() <= cfg.max_inflight_transfers,
            }
        finally:
            w.unpatch()
    except Exception as e:
        e9["status"] = "FAIL"
        e9["error"] = f"{type(e).__name__}: {e}"
    cases.append(e9)

    # E10 — Repeated patch/unpatch must not leak VRAM. The full suite
    # builds 40+ wrappers in one process, so a per-construction leak would
    # accumulate silently and eventually OOM mid-run.
    print("  [S10] e10_patch_unpatch_leak...")
    e10: Dict[str, Any] = {
        "case": "e10_patch_unpatch_leak",
        "description": "5x construct+unpatch — VRAM must return near baseline (the full suite builds 40+ wrappers).",
        "expected_to_fail": False,
    }
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        baseline_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        for _ in range(5):
            cfg = MemTierConfig(
                gpu_vram_bytes=6 * GB, hbm_cache_budget_bytes=600 * MB,
                host_dram_bytes=1500 * MB, cxl_memory_bytes=1500 * MB,
            )
            w = TieredMoEWrapper(
                model=base_model, config=cfg, co_occurrence_model=co_model,
                initial_hbm_budget_bytes=600 * MB, execution_mode="hybrid",
                expert_frequency=freq_map,
            )
            w.unpatch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        final_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        growth_mb = final_mb - baseline_mb
        e10["status"] = "PASS"
        e10["result"] = {
            "baseline_allocated_mb": round(baseline_mb, 1),
            "final_allocated_mb": round(final_mb, 1),
            "growth_mb": round(growth_mb, 1),
            # A whole expert is ~13MB, so >100MB of growth over 5 cycles
            # means real leakage rather than allocator noise.
            "leak_suspected": bool(growth_mb > 100.0),
        }
    except Exception as e:
        e10["status"] = "FAIL"
        e10["error"] = f"{type(e).__name__}: {e}"
    cases.append(e10)

    statuses = [c["status"] for c in cases]
    return {
        "scenario": "s10_edge_cases",
        "description": "Deliberate robustness probes: degenerate budgets, missing tiers, VRAM over-commit, "
                        "the disk-swap fallback, prefill-dominant input, prefetch abuse, and leak checks. "
                        "Runs last because the failure cases can leave the shared base model half-migrated.",
        "summary": {
            "total": len(cases),
            "passed": statuses.count("PASS"),
            "expected_fail": statuses.count("EXPECTED_FAIL"),
            "failed": statuses.count("FAIL"),
            "unexpected_pass": statuses.count("UNEXPECTED_PASS"),
        },
        "model_contaminated": contaminated,
        "contamination_note": (
            "A deliberate failure left the shared base model unusable — re-run any scenario after this one "
            "in a fresh process." if contaminated else None
        ),
        "cases": cases,
        "benchmark_mode": "live", "computation_type": "measured",
    }


# Insertion order is run order for --scenario all (dicts preserve it).
# s10 is last: its deliberate-failure cases can leave the shared base model
# half-migrated between devices, which would poison any scenario after it.
SCENARIOS = {
    "s1": scenario_s1,
    "s2": scenario_s2,
    "s3": scenario_s3,
    "s4": scenario_s4,
    "s5": scenario_s5,
    "s6": scenario_s6,
    "s7": scenario_s7,
    "s9": scenario_s9,
    "s10": scenario_s10,
}


class _Ctx:
    def __init__(self, resources, prompt, tokens):
        self.resources = resources  # (base_model, tokenizer, co_model, freq_map)
        self.prompt = prompt
        self.tokens = tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MemTier-MoE GPU scenario suite")
    parser.add_argument("--scenario", default="all", choices=list(SCENARIOS.keys()) + ["all"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Override for S8 scale validation on a bigger GPU")
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--tokens", type=int, default=25, help="max_new_tokens for single-shot scenarios (S1-S4, S7)")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    to_run = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    resources = load_resources(args.model_id, args.trace)
    ctx = _Ctx(resources, DEFAULT_PROMPT, args.tokens)

    suite_start = time.perf_counter()
    for scenario_id in to_run:
        print(f"\n{'=' * 90}\nRunning {scenario_id}\n{'=' * 90}")
        t0 = time.perf_counter()
        fn = SCENARIOS[scenario_id]
        payload = fn(ctx, args.out_dir)
        payload["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
        payload["model_id"] = args.model_id
        _save(scenario_id, payload, args.out_dir)
        print(f"  -> {scenario_id} done in {payload['elapsed_seconds']:.1f}s")

    print(f"\nTotal suite time: {time.perf_counter() - suite_start:.1f}s")


if __name__ == "__main__":
    main()
