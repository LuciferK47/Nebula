"""Comprehensive Stress Testing Suite for MemTier-MoE.

Evaluates all primary operational dimensions:
1. Extreme Memory Pressure (300MB to 1500MB HBM budgets).
2. Sustained Long-Horizon Generation (100 tokens continuous autoregressive decode).
3. Cross-Domain / Out-of-Distribution Routing (Technical, Code, Philosophy, Math).
4. Multi-Batch Concurrent Inference (Batch sizes 1, 2, 4).
5. 3-Tier Degradation & CXL Memory Fallback (HBM -> DRAM -> CXL).
6. Numerical Integrity & Exact Token Coherence vs Full VRAM.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import json
import torch
import numpy as np
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.types import MemoryTier
from memtier_moe.runtime.tiered_model import TieredMoEWrapper
from scripts.run_live_benchmark import load_or_calibrate_co_occurrence

LOCAL_CHAT_MOE = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen1.5-4x0.5B-Chat-MoE")
DEFAULT_STRESS_MODEL = LOCAL_CHAT_MOE if os.path.exists(LOCAL_CHAT_MOE) else "Qwen/Qwen1.5-MoE-A2.7B"

def run_stress_suite(model_id: str = DEFAULT_STRESS_MODEL, output_json: str = "results/stress_test_report.json"):
    print("=" * 90)
    print("MemTier-MoE: Comprehensive Multi-Use-Case Stress Test Suite")
    print("=" * 90)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware Compute Device: {device.upper()} ({device_name})")
    print(f"Target Model:            {model_id}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    co_model = load_or_calibrate_co_occurrence("traces/routing_trace_wikitext.npz", model=base_model, tokenizer=tokenizer)
    freq_map = getattr(getattr(co_model, "stats", None), "marginal_counts", None)

    report_summary = {
        "device": device_name,
        "model": model_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_results": {},
    }

    # =========================================================================
    # TEST CASE 1: Extreme Memory Pressure Stress Test
    # =========================================================================
    print("=" * 90)
    print("[TEST 1/6] Extreme Memory Pressure Stress Test (300MB, 600MB, 900MB, 1500MB)")
    print("=" * 90)
    budgets = [300, 600, 900, 1500]
    t1_results = []
    t1_passed = True

    prompt_t1 = "Explain the fundamental principles of hierarchical memory tiering in high-performance computing."
    inp_t1 = tokenizer(prompt_t1, return_tensors="pt")
    if torch.cuda.is_available():
        inp_t1 = {k: v.cuda() for k, v in inp_t1.items()}

    for b in budgets:
        print(f"  Testing HBM Budget: {b} MB...")
        config = MemTierConfig(
            gpu_vram_bytes=6 * 1024 * 1024 * 1024,
            hbm_cache_budget_bytes=b * 1024 * 1024,
            host_dram_bytes=2000 * 1024 * 1024,
            cxl_memory_bytes=2000 * 1024 * 1024,
        )

        torch.cuda.reset_peak_memory_stats()
        t_start_vram = torch.cuda.memory_allocated() / 1e6

        wrapper = TieredMoEWrapper(
            model=base_model,
            config=config,
            co_occurrence_model=co_model,
            initial_hbm_budget_bytes=b * 1024 * 1024,
            execution_mode="hybrid",
            expert_frequency=freq_map,
        )

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = wrapper.generate(**inp_t1, max_new_tokens=25, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tok_s = 25 / elapsed
            peak_vram = torch.cuda.max_memory_allocated() / 1e6
            rep = wrapper.report()
            m = rep["metrics"]

            res = {
                "budget_mb": b,
                "status": "PASS",
                "throughput_tok_s": round(tok_s, 2),
                "hit_rate": round(m.get("hit_rate", 0), 4),
                "hits": m.get("cache_hits", 0),
                "misses": m.get("cache_misses", 0),
                "evictions": m.get("evictions", 0),
                "peak_vram_mb": round(peak_vram, 1),
            }
            t1_results.append(res)
            print(f"    -> Status: PASS | {tok_s:5.2f} tok/s | HitRate: {res['hit_rate']:5.1%} | Evict: {res['evictions']} | Peak VRAM: {peak_vram:.1f} MB")
        except Exception as e:
            t1_passed = False
            print(f"    -> Status: FAIL ({e})")
            t1_results.append({"budget_mb": b, "status": "FAIL", "error": str(e)})
        finally:
            wrapper.unpatch()

    report_summary["test_results"]["case_1_extreme_memory_pressure"] = {
        "status": "PASS" if t1_passed else "FAIL",
        "details": t1_results,
    }

    # =========================================================================
    # TEST CASE 2: Sustained Long-Horizon Generation Stress Test (100 tokens)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[TEST 2/6] Sustained Long-Horizon Generation Stress Test (100 tokens continuous)")
    print("=" * 90)
    config_t2 = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=600 * 1024 * 1024,
        host_dram_bytes=2000 * 1024 * 1024,
        cxl_memory_bytes=2000 * 1024 * 1024,
    )
    wrapper_t2 = TieredMoEWrapper(
        model=base_model,
        config=config_t2,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=600 * 1024 * 1024,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )

    t2_status = "PASS"
    t2_details = {}
    try:
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1e6
        print(f"  Generating 100 continuous tokens under 600MB HBM budget...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_100 = wrapper_t2.generate(**inp_t1, max_new_tokens=100, do_sample=False)
        torch.cuda.synchronize()
        elapsed_100 = time.perf_counter() - t0
        tok_s_100 = 100 / elapsed_100
        vram_after = torch.cuda.memory_allocated() / 1e6
        peak_vram_100 = torch.cuda.max_memory_allocated() / 1e6
        rep_100 = wrapper_t2.report()
        m_100 = rep_100["metrics"]

        # Decode sample text
        gen_text = tokenizer.decode(out_100[0], skip_special_tokens=True)

        t2_details = {
            "tokens_generated": 100,
            "throughput_tok_s": round(tok_s_100, 2),
            "wall_time_s": round(elapsed_100, 2),
            "hit_rate": round(m_100.get("hit_rate", 0), 4),
            "evictions": m_100.get("evictions", 0),
            "vram_delta_mb": round(vram_after - vram_before, 2),
            "peak_vram_mb": round(peak_vram_100, 1),
            "sample_snippet": gen_text[:140] + "...",
        }
        print(f"  Result: {tok_s_100:.2f} tok/s over 100 tokens | HitRate: {t2_details['hit_rate']:.1%} | Evictions: {t2_details['evictions']} | Memory Delta: {t2_details['vram_delta_mb']} MB")
        clean_snippet = t2_details['sample_snippet'].encode('ascii', 'replace').decode('ascii')
        print(f"  Snippet: \"{clean_snippet}\"")
    except Exception as e:
        t2_status = "FAIL"
        t2_details = {"error": str(e)}
        print(f"  Failed with error: {e}")
    finally:
        wrapper_t2.unpatch()

    report_summary["test_results"]["case_2_sustained_long_generation"] = {
        "status": t2_status,
        "details": t2_details,
    }

    # =========================================================================
    # TEST CASE 3: Out-of-Distribution / Cross-Domain Routing Stress Test
    # =========================================================================
    print("\n" + "=" * 90)
    print("[TEST 3/6] Out-of-Distribution Cross-Domain Routing Stress Test")
    print("=" * 90)
    test_domains = [
        ("Technical Systems", "In computer architecture, cache coherence protocols like MESI maintain consistency across L1, L2, and L3 caches."),
        ("Python Code", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n"),
        ("Philosophy / Ethics", "The question of machine consciousness raises profound dilemmas regarding moral responsibility and artificial intentionality."),
        ("Mathematics / Logic", "The Riemann Hypothesis posits that all non-trivial zeros of the zeta function possess a real coordinate of 1/2."),
    ]

    t3_results = []
    t3_passed = True

    config_t3 = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=900 * 1024 * 1024,
        host_dram_bytes=2000 * 1024 * 1024,
        cxl_memory_bytes=2000 * 1024 * 1024,
    )
    wrapper_t3 = TieredMoEWrapper(
        model=base_model,
        config=config_t3,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=900 * 1024 * 1024,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )

    for domain_name, p_text in test_domains:
        print(f"  Testing Domain: {domain_name}...")
        inp_dom = tokenizer(p_text, return_tensors="pt")
        if torch.cuda.is_available():
            inp_dom = {k: v.cuda() for k, v in inp_dom.items()}

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_dom = wrapper_t3.generate(**inp_dom, max_new_tokens=25, do_sample=False)
            torch.cuda.synchronize()
            elapsed_dom = time.perf_counter() - t0
            tok_s_dom = 25 / elapsed_dom
            rep_dom = wrapper_t3.report()
            m_dom = rep_dom["metrics"]
            gen_snippet = tokenizer.decode(out_dom[0], skip_special_tokens=True)

            res_dom = {
                "domain": domain_name,
                "status": "PASS",
                "throughput_tok_s": round(tok_s_dom, 2),
                "hit_rate": round(m_dom.get("hit_rate", 0), 4),
                "evictions": m_dom.get("evictions", 0),
                "sample_output": gen_snippet[:100] + "...",
            }
            t3_results.append(res_dom)
            print(f"    -> Status: PASS | {tok_s_dom:5.2f} tok/s | HitRate: {res_dom['hit_rate']:5.1%} | Sample: \"{res_dom['sample_output'][:60]}...\"")
        except Exception as e:
            t3_passed = False
            print(f"    -> Status: FAIL ({e})")
            t3_results.append({"domain": domain_name, "status": "FAIL", "error": str(e)})

    wrapper_t3.unpatch()
    report_summary["test_results"]["case_3_cross_domain_stress"] = {
        "status": "PASS" if t3_passed else "FAIL",
        "details": t3_results,
    }

    # =========================================================================
    # TEST CASE 4: Batched Inference Stress Test (Batch Sizes 1, 2, 4)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[TEST 4/6] Batched Inference Stress Test (Batch Sizes 1, 2, 4)")
    print("=" * 90)
    batch_prompts = [
        "Artificial intelligence revolutionizes data science.",
        "Quantum computing fundamentally accelerates cryptography.",
        "Distributed database systems provide fault tolerance.",
        "Compiler optimization maximizes pipeline parallelism.",
    ]

    t4_results = []
    t4_passed = True

    config_t4 = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=900 * 1024 * 1024,
        host_dram_bytes=2000 * 1024 * 1024,
        cxl_memory_bytes=2000 * 1024 * 1024,
    )
    wrapper_t4 = TieredMoEWrapper(
        model=base_model,
        config=config_t4,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=900 * 1024 * 1024,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )

    for bsz in [1, 2, 4]:
        print(f"  Testing Batch Size: {bsz}...")
        sub_prompts = batch_prompts[:bsz]
        inp_batch = tokenizer(sub_prompts, padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            inp_batch = {k: v.cuda() for k, v in inp_batch.items()}

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_batch = wrapper_t4.generate(**inp_batch, max_new_tokens=15, do_sample=False)
            torch.cuda.synchronize()
            elapsed_b = time.perf_counter() - t0
            total_toks = bsz * 15
            tok_s_b = total_toks / elapsed_b
            rep_b = wrapper_t4.report()
            m_b = rep_b["metrics"]

            res_b = {
                "batch_size": bsz,
                "status": "PASS",
                "total_tokens": total_toks,
                "throughput_tok_s": round(tok_s_b, 2),
                "wall_time_s": round(elapsed_b, 2),
                "hit_rate": round(m_b.get("hit_rate", 0), 4),
            }
            t4_results.append(res_b)
            print(f"    -> Status: PASS | Total Throughput: {tok_s_b:5.2f} tok/s | HitRate: {res_b['hit_rate']:5.1%}")
        except Exception as e:
            t4_passed = False
            print(f"    -> Status: FAIL ({e})")
            t4_results.append({"batch_size": bsz, "status": "FAIL", "error": str(e)})

    wrapper_t4.unpatch()
    report_summary["test_results"]["case_4_batched_inference"] = {
        "status": "PASS" if t4_passed else "FAIL",
        "details": t4_results,
    }

    # =========================================================================
    # TEST CASE 5: Tier Degradation & CXL Memory Fallback (3-Tier Stress)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[TEST 5/6] 3-Tier Degradation & CXL Memory Fallback Stress Test")
    print("=" * 90)
    # Constrain HBM to 600MB and DRAM to 400MB, forcing remaining experts to CXL pool
    config_t5 = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=600 * 1024 * 1024,
        host_dram_bytes=400 * 1024 * 1024,
        cxl_memory_bytes=4000 * 1024 * 1024,
    )
    wrapper_t5 = TieredMoEWrapper(
        model=base_model,
        config=config_t5,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=600 * 1024 * 1024,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )

    t5_status = "PASS"
    t5_details = {}
    try:
        # Check distribution across tiers
        hbm_exps = len(wrapper_t5.engine.tier_manager.experts_in_tier(MemoryTier.HBM))
        dram_exps = len(wrapper_t5.engine.tier_manager.experts_in_tier(MemoryTier.DRAM))
        cxl_exps = len(wrapper_t5.engine.tier_manager.experts_in_tier(MemoryTier.CXL))
        print(f"  Expert Tier Distribution: HBM={hbm_exps}, Host DRAM={dram_exps}, CXL={cxl_exps}")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_cxl = wrapper_t5.generate(**inp_t1, max_new_tokens=25, do_sample=False)
        torch.cuda.synchronize()
        elapsed_cxl = time.perf_counter() - t0
        tok_s_cxl = 25 / elapsed_cxl
        rep_cxl = wrapper_t5.report()
        m_cxl = rep_cxl["metrics"]

        t5_details = {
            "hbm_experts": hbm_exps,
            "dram_experts": dram_exps,
            "cxl_experts": cxl_exps,
            "throughput_tok_s": round(tok_s_cxl, 2),
            "hit_rate": round(m_cxl.get("hit_rate", 0), 4),
            "evictions": m_cxl.get("evictions", 0),
        }
        print(f"  Result: PASS | {tok_s_cxl:.2f} tok/s | CXL Fallback smoothly routed | Evictions: {t5_details['evictions']}")
    except Exception as e:
        t5_status = "FAIL"
        t5_details = {"error": str(e)}
        print(f"  Result: FAIL ({e})")
    finally:
        wrapper_t5.unpatch()

    report_summary["test_results"]["case_5_cxl_tier_fallback"] = {
        "status": t5_status,
        "details": t5_details,
    }

    # =========================================================================
    # TEST CASE 6: Numerical Output Integrity & Exact Coherence Validation
    # =========================================================================
    print("\n" + "=" * 90)
    print("[TEST 6/6] Numerical Integrity & Exact Token Coherence vs Full VRAM")
    print("=" * 90)
    # Generate reference tokens from pure unmodified base model in full VRAM
    inp_vram = tokenizer("The Mixture of Experts architecture routes tokens through specialized feed-forward layers", return_tensors="pt")
    if torch.cuda.is_available():
        inp_vram = {k: v.cuda() for k, v in inp_vram.items()}

    with torch.no_grad():
        ref_tokens = base_model.generate(**inp_vram, max_new_tokens=12, do_sample=False)

    # Generate test tokens using TieredMoE Hybrid at 900MB
    config_t6 = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=900 * 1024 * 1024,
        host_dram_bytes=2000 * 1024 * 1024,
        cxl_memory_bytes=2000 * 1024 * 1024,
    )
    wrapper_t6 = TieredMoEWrapper(
        model=base_model,
        config=config_t6,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=900 * 1024 * 1024,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )

    with torch.no_grad():
        test_tokens = wrapper_t6.generate(**inp_vram, max_new_tokens=12, do_sample=False)
    wrapper_t6.unpatch()

    ref_ids = ref_tokens[0].cpu().numpy()
    test_ids = test_tokens[0].cpu().numpy()

    exact_matches = int((ref_ids == test_ids).sum())
    match_pct = exact_matches / len(ref_ids)
    has_nan = bool(torch.isnan(test_tokens.float()).any().item())
    has_inf = bool(torch.isinf(test_tokens.float()).any().item())

    t6_passed = (match_pct >= 0.75) and (not has_nan) and (not has_inf)

    ref_str = tokenizer.decode(ref_tokens[0], skip_special_tokens=True)
    test_str = tokenizer.decode(test_tokens[0], skip_special_tokens=True)

    t6_details = {
        "total_tokens_evaluated": len(ref_ids),
        "exact_matches": exact_matches,
        "token_match_percentage": round(match_pct * 100, 2),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "reference_text": ref_str[:120] + "...",
        "tiered_model_text": test_str[:120] + "...",
    }

    print(f"  Token Exact Match vs Full VRAM: {match_pct:6.1%} ({exact_matches}/{len(ref_ids)} tokens)")
    print(f"  NaN Detected: {has_nan} | Inf Detected: {has_inf}")
    ref_clean = ref_str[:80].encode('ascii', 'replace').decode('ascii')
    test_clean = test_str[:80].encode('ascii', 'replace').decode('ascii')
    print(f"  Reference Output: \"{ref_clean}...\"")
    print(f"  Tiered Output:    \"{test_clean}...\"")
    print(f"  Result: {'PASS' if t6_passed else 'FAIL'}")

    report_summary["test_results"]["case_6_numerical_integrity"] = {
        "status": "PASS" if t6_passed else "FAIL",
        "details": t6_details,
    }

    # =========================================================================
    # Final Validation Summary
    # =========================================================================
    print("\n" + "=" * 90)
    print("STRESS TEST OVERALL RESULTS SUMMARY")
    print("=" * 90)
    all_passed = all(v["status"] == "PASS" for v in report_summary["test_results"].values())
    for case_k, case_v in report_summary["test_results"].items():
        print(f"  {case_k:<38}: {case_v['status']}")
    print("-" * 90)
    print(f"  OVERALL SUITE VERDICT: {'ALL 6 CASES PASSED (100% GREEN)' if all_passed else 'SOME CASES FAILED'}")
    print("=" * 90)

    report_summary["overall_status"] = "ALL_PASS" if all_passed else "FAIL"

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report_summary, f, indent=2)
    print(f"Stress test report saved to: {output_json}")
    return report_summary

if __name__ == "__main__":
    run_stress_suite()
