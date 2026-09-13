import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from memtier_moe.core.config import MemTierConfig
from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel
from memtier_moe.runtime.tiered_model import TieredMoEWrapper
from scripts.run_live_benchmark import BENCHMARK_PROMPT, load_or_calibrate_co_occurrence

def test_budget(base_model, tokenizer, co_model, freq_map, hbm_mb, tokens=25):
    inputs = tokenizer(BENCHMARK_PROMPT, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    config = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=hbm_mb * 1024 * 1024,
        host_dram_bytes=2000 * 1024 * 1024,
        cxl_memory_bytes=2000 * 1024 * 1024,
    )

    wrapper = TieredMoEWrapper(
        model=base_model,
        config=config,
        co_occurrence_model=co_model,
        initial_hbm_budget_bytes=hbm_mb * 1024 * 1024,
        execution_mode="hybrid",
        expert_frequency=freq_map,
    )

    # Warmup
    with torch.no_grad():
        wrapper.generate(**inputs, max_new_tokens=3, do_sample=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = wrapper.generate(**inputs, max_new_tokens=tokens, do_sample=False, repetition_penalty=1.1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gen_toks = out.shape[1] - inputs["input_ids"].shape[1]
    tok_s = gen_toks / elapsed

    rep = wrapper.report()
    m = rep["metrics"]
    wrapper.unpatch()
    return {
        "hbm_mb": hbm_mb,
        "tok_s": round(tok_s, 2),
        "hit_rate": round(m.get("hit_rate", 0), 4),
        "hits": m.get("cache_hits", 0),
        "misses": m.get("cache_misses", 0),
        "evictions": m.get("evictions", 0),
        "elapsed": round(elapsed, 2),
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    local_chat_moe = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen1.5-4x0.5B-Chat-MoE")
    model_id = local_chat_moe if os.path.exists(local_chat_moe) else "Qwen/Qwen1.5-MoE-A2.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
    base_model.eval()

    co_model = load_or_calibrate_co_occurrence(trace_path="traces/routing_trace_wikitext.npz", model=base_model, tokenizer=tokenizer)
    freq_map = getattr(getattr(co_model, "stats", None), "marginal_counts", None)

    # Measure Full VRAM baseline
    print("\n[Baseline] Measuring GPU-Resident Full VRAM...")
    inputs = tokenizer(BENCHMARK_PROMPT, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        base_model.generate(**inputs, max_new_tokens=3, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=25, do_sample=False, repetition_penalty=1.1)
    torch.cuda.synchronize()
    vram_tok_s = 25 / (time.perf_counter() - t0)
    print(f"Full VRAM (2500MB): {vram_tok_s:.2f} tok/s (100% of peak)")

    print("\n--- Sweeping Profile-Guided Hybrid Compute across HBM Budgets ---")
    budgets = [600, 900, 1200, 1500]
    for b in budgets:
        res = test_budget(base_model, tokenizer, co_model, freq_map, b, tokens=25)
        pct = (res["tok_s"] / vram_tok_s) * 100
        print(f"HBM {res['hbm_mb']}MB: {res['tok_s']:5.2f} tok/s ({pct:5.1f}% of Full VRAM) | HitRate: {res['hit_rate']:6.1%} | Hits: {res['hits']:4d} | Misses: {res['misses']:3d} | Evict: {res['evictions']}")

if __name__ == "__main__":
    main()
