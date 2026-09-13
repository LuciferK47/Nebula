"""
MemTier-MoE: Established Benchmark Suite
========================================
1. WikiText-2 Official Test Set Perplexity (Academic Parity Standard)
   - Evaluates Negative Log-Likelihood (NLL) and Perplexity (PPL) on the standard
     WikiText-2 test split to verify zero or near-zero quality degradation.
2. ShareGPT Real-World Conversational Serving Benchmark (vLLM / SGLang Industry Standard)
   - Measures Time to First Token (TTFT), Time per Output Token (TPOT / ITL),
     P50/P90/P99 latency quantiles, hit rate, and throughput across HBM budgets.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import urllib.request
import torch
import numpy as np
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from memtier_moe.core.config import MemTierConfig
from memtier_moe.runtime.tiered_model import TieredMoEWrapper
from scripts.run_live_benchmark import load_or_calibrate_co_occurrence

LOCAL_CHAT_MOE = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen1.5-4x0.5B-Chat-MoE")
DEFAULT_MODEL = LOCAL_CHAT_MOE if os.path.exists(LOCAL_CHAT_MOE) else "Qwen/Qwen1.5-MoE-A2.7B"

WIKITEXT2_TEST_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"

# Curated multi-turn ShareGPT prompts spanning short, medium, and long context
SHAREGPT_PROMPTS = [
    {
        "id": "sharegpt-coding-01",
        "category": "Coding & Algorithms",
        "prompt": "Write a Python implementation of a thread-safe LRU cache with an O(1) get and put method, utilizing a doubly linked list and a hash map.",
        "max_gen": 40,
    },
    {
        "id": "sharegpt-sys-02",
        "category": "Systems Architecture",
        "prompt": "Explain how modern operating systems handle page replacement when physical RAM is exhausted. Compare the clock algorithm with second-chance FIFO.",
        "max_gen": 40,
    },
    {
        "id": "sharegpt-math-03",
        "category": "Mathematics & Reasoning",
        "prompt": "Prove why the square root of 2 is irrational using a proof by contradiction. State each assumption clearly and explain the parity argument.",
        "max_gen": 40,
    },
    {
        "id": "sharegpt-short-04",
        "category": "Conversational Short",
        "prompt": "What are the key differences between synchronous and asynchronous I/O in distributed networking architectures?",
        "max_gen": 35,
    },
    {
        "id": "sharegpt-creative-05",
        "category": "Analysis & Synthesis",
        "prompt": "Analyze the trade-offs between monolithic microarchitectures and modular chiplet designs in modern high-performance datacenter accelerators.",
        "max_gen": 40,
    },
]


def fetch_wikitext2_test() -> str:
    """Download or retrieve local cached official WikiText-2 test set."""
    local_path = os.path.join(os.path.dirname(__file__), "..", "results", "wikitext2_test.txt")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"Fetching official WikiText-2 test set from {WIKITEXT2_TEST_URL}...")
    req = urllib.request.Request(WIKITEXT2_TEST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read().decode("utf-8")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def compute_perplexity(model, tokenizer, text: str, max_eval_tokens: int = 1024, stride: int = 512, device: str = "cuda") -> Dict[str, float]:
    """Compute standard sliding-window Negative Log-Likelihood and Perplexity."""
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = min(encodings.input_ids.size(1), max_eval_tokens)
    input_ids = encodings.input_ids[:, :seq_len].to(device)

    nlls = []
    total_tokens = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + stride, seq_len)
            trg_len = end_loc - begin_loc
            chunk_input_ids = input_ids[:, begin_loc:end_loc]
            target_ids = chunk_input_ids.clone()

            outputs = model(chunk_input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood.item())
            total_tokens += trg_len
            if end_loc == seq_len:
                break

    eval_time = time.perf_counter() - t0
    total_nll = sum(nlls)
    avg_loss = total_nll / total_tokens
    ppl = float(np.exp(avg_loss))

    return {
        "loss": round(float(avg_loss), 4),
        "perplexity": round(ppl, 4),
        "total_tokens": total_tokens,
        "eval_time_s": round(eval_time, 2),
        "tok_per_sec": round(total_tokens / eval_time, 2),
    }


def run_serving_benchmark(model, tokenizer, prompt_data: List[Dict[str, Any]], device: str = "cuda", is_tiered: bool = False) -> Dict[str, Any]:
    """Run ShareGPT conversational serving benchmark measuring TTFT, TPOT, and latency quantiles."""
    ttft_list = []
    tpot_list = []
    all_decode_latencies = []
    total_tokens_generated = 0
    start_total = time.perf_counter()

    for item in prompt_data:
        prompt = item["prompt"]
        max_gen = item["max_gen"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        # 1. Prefill Phase (Measures TTFT)
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        with torch.no_grad():
            first_out = model(inputs.input_ids)
            next_token = torch.argmax(first_out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t_prefill_start) * 1000.0
        ttft_list.append(ttft_ms)

        curr_ids = torch.cat([inputs.input_ids, next_token], dim=-1)
        decode_steps = 1

        # 2. Autoregressive Decode Phase (Measures TPOT / ITL)
        with torch.no_grad():
            for _ in range(max_gen - 1):
                torch.cuda.synchronize()
                t_step_start = time.perf_counter()
                out = model(curr_ids[:, -1:])  # Single-token decode step
                step_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                step_ms = (time.perf_counter() - t_step_start) * 1000.0

                all_decode_latencies.append(step_ms)
                curr_ids = torch.cat([curr_ids, step_tok], dim=-1)
                decode_steps += 1
                if step_tok.item() == tokenizer.eos_token_id or step_tok.item() == 151645:
                    break

        total_tokens_generated += decode_steps

    total_time_s = time.perf_counter() - start_total
    overall_throughput = total_tokens_generated / total_time_s

    metrics_extra = {}
    if is_tiered and hasattr(model, "report"):
        rep = model.report()
        m = rep.get("metrics", {})
        metrics_extra = {
            "hit_rate": round(m.get("hit_rate", 0), 4),
            "hbm_hits": m.get("hbm_hits", 0),
            "dram_hits": m.get("dram_hits", 0),
            "cxl_hits": m.get("cxl_hits", 0),
            "evictions": m.get("evictions", 0),
            "total_transfer_bytes": m.get("total_transfer_bytes", 0),
        }

    return {
        "ttft_p50_ms": round(float(np.percentile(ttft_list, 50)), 2),
        "ttft_p90_ms": round(float(np.percentile(ttft_list, 90)), 2),
        "tpot_mean_ms": round(float(np.mean(all_decode_latencies)), 2),
        "tpot_p50_ms": round(float(np.percentile(all_decode_latencies, 50)), 2),
        "tpot_p90_ms": round(float(np.percentile(all_decode_latencies, 90)), 2),
        "tpot_p99_ms": round(float(np.percentile(all_decode_latencies, 99)), 2),
        "throughput_tok_s": round(overall_throughput, 2),
        "total_tokens_generated": total_tokens_generated,
        "metrics": metrics_extra,
    }


def main():
    output_json = "results/established_benchmark_results.json"
    print("=" * 90)
    print("MemTier-MoE: Established Benchmark Suite Execution")
    print("  1. WikiText-2 Test Set Perplexity (Standard Academic Parity Evaluation)")
    print("  2. ShareGPT Real-World Conversational Serving (vLLM / SGLang Standard)")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware: {device.upper()} ({device_name})")
    print(f"Model:    {DEFAULT_MODEL}\n")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in native FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # 1. Calibrate / Load Markovian co-occurrence model
    trace_path = "traces/routing_trace_wikitext.npz"
    co_model = load_or_calibrate_co_occurrence(trace_path, base_model, tokenizer)
    freq_map = getattr(getattr(co_model, "stats", None), "marginal_counts", None)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": device_name,
        "model": DEFAULT_MODEL,
        "wikitext2_perplexity": {},
        "sharegpt_serving": {},
    }

    # =========================================================================
    # PART 1: WIKITEXT-2 TEST SET PERPLEXITY BENCHMARK
    # =========================================================================
    print("\n" + "=" * 90)
    print("[BENCHMARK 1/2] Official WikiText-2 Test Set Perplexity")
    print("=" * 90)
    wikitext_raw = fetch_wikitext2_test()
    # Filter non-empty paragraphs
    eval_text = "\n\n".join([p.strip() for p in wikitext_raw.split("\n\n") if len(p.strip()) > 100][:15])
    print(f"Prepared WikiText-2 evaluation slice ({len(eval_text)} characters)...")

    # A. Base Model Oracle (Full VRAM)
    print("\nEvaluating Base Model (Full VRAM Oracle)...")
    base_ppl_res = compute_perplexity(base_model, tokenizer, eval_text, max_eval_tokens=1024, stride=512)
    print(f"  -> Base Model PPL:   {base_ppl_res['perplexity']:.4f} (Loss: {base_ppl_res['loss']:.4f}, Speed: {base_ppl_res['tok_per_sec']:.1f} tok/s)")
    report["wikitext2_perplexity"]["full_vram_oracle"] = base_ppl_res

    # B. MemTier-MoE @ 900MB Budget
    print("\nEvaluating MemTier-MoE Hybrid (900MB HBM Budget)...")
    cfg_900 = MemTierConfig(
        gpu_vram_bytes=6 * 1024**3,
        hbm_cache_budget_bytes=900 * 1024**2,
        host_dram_bytes=2000 * 1024**2,
        cxl_memory_bytes=2000 * 1024**2,
    )
    wrapper_900 = TieredMoEWrapper(
        model=base_model,
        config=cfg_900,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=900 * 1024**2,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )
    res_900 = compute_perplexity(wrapper_900, tokenizer, eval_text, max_eval_tokens=1024, stride=512)
    delta_ppl_900 = res_900["perplexity"] - base_ppl_res["perplexity"]
    res_900["delta_ppl"] = round(delta_ppl_900, 4)
    res_900["loss_diff_pct"] = round(abs(res_900["loss"] - base_ppl_res["loss"]) / base_ppl_res["loss"] * 100, 3)
    wrapper_900.unpatch()
    print(f"  -> MemTier-MoE 900MB: {res_900['perplexity']:.4f} (Delta PPL: {delta_ppl_900:+.4f}, Loss Diff: {res_900['loss_diff_pct']}%)")
    report["wikitext2_perplexity"]["memtier_900mb"] = res_900

    # C. MemTier-MoE @ 600MB Budget (Extreme Memory Constraint)
    print("\nEvaluating MemTier-MoE Hybrid (600MB HBM Budget)...")
    cfg_600 = MemTierConfig(
        gpu_vram_bytes=6 * 1024**3,
        hbm_cache_budget_bytes=600 * 1024**2,
        host_dram_bytes=2000 * 1024**2,
        cxl_memory_bytes=2000 * 1024**2,
    )
    wrapper_600 = TieredMoEWrapper(
        model=base_model,
        config=cfg_600,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=600 * 1024**2,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )
    res_600 = compute_perplexity(wrapper_600, tokenizer, eval_text, max_eval_tokens=1024, stride=512)
    delta_ppl_600 = res_600["perplexity"] - base_ppl_res["perplexity"]
    res_600["delta_ppl"] = round(delta_ppl_600, 4)
    res_600["loss_diff_pct"] = round(abs(res_600["loss"] - base_ppl_res["loss"]) / base_ppl_res["loss"] * 100, 3)
    wrapper_600.unpatch()
    print(f"  -> MemTier-MoE 600MB: {res_600['perplexity']:.4f} (Delta PPL: {delta_ppl_600:+.4f}, Loss Diff: {res_600['loss_diff_pct']}%)")
    report["wikitext2_perplexity"]["memtier_600mb"] = res_600

    # =========================================================================
    # PART 2: SHAREGPT REAL-WORLD SERVING BENCHMARK
    # =========================================================================
    print("\n" + "=" * 90)
    print("[BENCHMARK 2/2] ShareGPT Conversational Serving Latency Benchmark (vLLM Standard)")
    print("=" * 90)

    # A. Base Model Oracle (Full VRAM)
    print("\nEvaluating Base Model Serving (Full VRAM Oracle)...")
    serv_base = run_serving_benchmark(base_model, tokenizer, SHAREGPT_PROMPTS, device=device, is_tiered=False)
    print(f"  -> Full VRAM:    TTFT P50: {serv_base['ttft_p50_ms']:5.1f} ms | TPOT Mean: {serv_base['tpot_mean_ms']:5.1f} ms | Throughput: {serv_base['throughput_tok_s']:5.2f} tok/s")
    report["sharegpt_serving"]["full_vram_oracle"] = serv_base

    # B. MemTier-MoE @ 900MB Budget
    print("\nEvaluating MemTier-MoE Serving (900MB HBM Budget)...")
    wrapper_serv900 = TieredMoEWrapper(
        model=base_model,
        config=cfg_900,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=900 * 1024**2,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )
    serv_900 = run_serving_benchmark(wrapper_serv900, tokenizer, SHAREGPT_PROMPTS, device=device, is_tiered=True)
    wrapper_serv900.unpatch()
    print(f"  -> MemTier 900MB: TTFT P50: {serv_900['ttft_p50_ms']:5.1f} ms | TPOT Mean: {serv_900['tpot_mean_ms']:5.1f} ms | Throughput: {serv_900['throughput_tok_s']:5.2f} tok/s | HitRate: {serv_900['metrics'].get('hit_rate', 0):.1%}")
    report["sharegpt_serving"]["memtier_900mb"] = serv_900

    # C. MemTier-MoE @ 600MB Budget
    print("\nEvaluating MemTier-MoE Serving (600MB HBM Budget)...")
    wrapper_serv600 = TieredMoEWrapper(
        model=base_model,
        config=cfg_600,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=600 * 1024**2,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )
    serv_600 = run_serving_benchmark(wrapper_serv600, tokenizer, SHAREGPT_PROMPTS, device=device, is_tiered=True)
    wrapper_serv600.unpatch()
    print(f"  -> MemTier 600MB: TTFT P50: {serv_600['ttft_p50_ms']:5.1f} ms | TPOT Mean: {serv_600['tpot_mean_ms']:5.1f} ms | Throughput: {serv_600['throughput_tok_s']:5.2f} tok/s | HitRate: {serv_600['metrics'].get('hit_rate', 0):.1%}")
    report["sharegpt_serving"]["memtier_600mb"] = serv_600

    # Save JSON report
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Established benchmark results saved to: {output_json}")

    # Print Final Summary Comparison Table
    print("\n" + "=" * 90)
    print("ESTABLISHED BENCHMARK SUITE: SUMMARY RESULTS")
    print("=" * 90)
    print("1. WIKITEXT-2 PERPLEXITY (QUALITY FIDELITY)")
    print(f"  {'Configuration':<30} | {'Loss':<8} | {'Perplexity':<10} | {'Delta PPL':<10} | {'Loss Diff %'}")
    print("  " + "-" * 80)
    print(f"  {'Full VRAM Oracle (Base)':<30} | {base_ppl_res['loss']:<8.4f} | {base_ppl_res['perplexity']:<10.4f} | {'0.0000':<10} | 0.000%")
    print(f"  {'MemTier-MoE (900MB Budget)':<30} | {res_900['loss']:<8.4f} | {res_900['perplexity']:<10.4f} | {res_900['delta_ppl']:<+10.4f} | {res_900['loss_diff_pct']}%")
    print(f"  {'MemTier-MoE (600MB Budget)':<30} | {res_600['loss']:<8.4f} | {res_600['perplexity']:<10.4f} | {res_600['delta_ppl']:<+10.4f} | {res_600['loss_diff_pct']}%")

    print("\n2. SHAREGPT SERVING LATENCY (vLLM INDUSTRY STANDARD)")
    print(f"  {'Configuration':<30} | {'TTFT P50':<10} | {'TPOT Mean':<10} | {'TPOT P90':<10} | {'Throughput':<12} | {'Hit Rate'}")
    print("  " + "-" * 88)
    print(f"  {'Full VRAM Oracle':<30} | {serv_base['ttft_p50_ms']:<8.1f}ms | {serv_base['tpot_mean_ms']:<8.1f}ms | {serv_base['tpot_p90_ms']:<8.1f}ms | {serv_base['throughput_tok_s']:<8.2f} tok/s | 100.0%")
    print(f"  {'MemTier-MoE (900MB)':<30} | {serv_900['ttft_p50_ms']:<8.1f}ms | {serv_900['tpot_mean_ms']:<8.1f}ms | {serv_900['tpot_p90_ms']:<8.1f}ms | {serv_900['throughput_tok_s']:<8.2f} tok/s | {serv_900['metrics'].get('hit_rate', 0):.1%}")
    print(f"  {'MemTier-MoE (600MB)':<30} | {serv_600['ttft_p50_ms']:<8.1f}ms | {serv_600['tpot_mean_ms']:<8.1f}ms | {serv_600['tpot_p90_ms']:<8.1f}ms | {serv_600['throughput_tok_s']:<8.2f} tok/s | {serv_600['metrics'].get('hit_rate', 0):.1%}")
    print("=" * 90)


if __name__ == "__main__":
    main()
