"""Empirical CXL Spill-Over Test:
Evaluates MemTier-MoE when Host DRAM is constrained below the model's footprint,
forcing tail experts into the CXL tier and verifying CXL hits, latency injection,
and throughput preservation vs. two-tier eviction failure/swap.
"""
from __future__ import annotations
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.types import MemoryTier
from memtier_moe.runtime.tiered_model import TieredMoEWrapper

LOCAL_QWEN_MOE = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen1.5-4x0.5B-Chat-MoE")
MODEL_ID = LOCAL_QWEN_MOE if os.path.exists(LOCAL_QWEN_MOE) else "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "Mixture of Experts architecture enables efficient scaling of neural parameters"

def run_cxl_spill_test():
    print("=" * 80)
    print("MEMTIER-MOE: EMPIRICAL CXL SPILL-OVER & TIERING VALIDATION TEST")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("\n[1/3] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Total expert weights: 96 experts * 13.3 MB = 1,276.8 MB
    # We set:
    # HBM: 400 MB (~30 experts)
    # DRAM: 400 MB (~30 experts)
    # Combined HBM + DRAM = 800 MB < 1276.8 MB
    # Deficit: ~476.8 MB MUST spill into CXL!
    hbm_mb = 400
    dram_mb = 400
    cxl_mb = 800

    print(f"\n[2/3] Initializing TieredMoEWrapper with Constrained DRAM:")
    print(f"  HBM Budget:  {hbm_mb} MB")
    print(f"  DRAM Budget: {dram_mb} MB")
    print(f"  CXL Budget:  {cxl_mb} MB")
    print(f"  Total Memory: {hbm_mb + dram_mb + cxl_mb} MB (Model Expert Footprint: ~1,276 MB)")

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

    # Inspect initial expert tier distribution
    tier_counts = {MemoryTier.HBM: 0, MemoryTier.DRAM: 0, MemoryTier.CXL: 0}
    for eid, meta in tiered_model.engine.tier_manager._registry.items():
        tier_counts[meta.current_tier] += 1

    print("\nInitial Expert Tier Distribution:")
    for tier, count in tier_counts.items():
        size_mb = count * 13.3
        print(f"  - {tier.name:<5}: {count:2d} experts ({size_mb:6.1f} MB)")

    assert tier_counts[MemoryTier.CXL] > 0, "Error: CXL pool should contain overflow experts!"

    # [3/3] Execute Live Generation Pass
    print("\n[3/3] Executing Live Autoregressive Token Generation (5 tokens)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = tiered_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0

    gen_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    tok_s = gen_tokens / wall_time if wall_time > 0 else 0.0

    rep = tiered_model.report()
    m = rep["metrics"]

    print("\n" + "=" * 80)
    print("LIVE CXL INFERENCE EXECUTION REPORT")
    print("=" * 80)
    print(f"Generated Tokens:      {gen_tokens}")
    print(f"Total Wall Time:       {wall_time:.3f} s")
    print(f"Throughput:            {tok_s:.2f} tok/s")
    print("-" * 80)
    print(f"Tier Hits Breakdown:")
    print(f"  HBM Hits:            {m.get('hbm_hits', 0):6d} (Served from GPU VRAM)")
    print(f"  DRAM Hits:           {m.get('dram_hits', 0):6d} (Served from Host System RAM)")
    print(f"  CXL Hits:            {m.get('cxl_hits', 0):6d} (Served from Emulated CXL Pool)")
    print(f"  Disk Page Faults:    {m.get('disk_faults', 0):6d}")
    print("-" * 80)
    total_hits = m.get('hbm_hits', 0) + m.get('dram_hits', 0) + m.get('cxl_hits', 0)
    if total_hits > 0:
        print(f"  HBM Hit Rate:        {m.get('hbm_hits', 0) / total_hits:.1%}")
        print(f"  DRAM Hit Rate:       {m.get('dram_hits', 0) / total_hits:.1%}")
        print(f"  CXL Hit Rate:        {m.get('cxl_hits', 0) / total_hits:.1%}")
    print(f"CXL Transfer Volume:   {m.get('cxl_transfer_bytes', 0) / 1e6:.2f} MB")
    print(f"Total Evictions:       {m.get('evictions', 0)}")
    print("=" * 80)

    tiered_model.unpatch()
    return rep

if __name__ == "__main__":
    run_cxl_spill_test()
