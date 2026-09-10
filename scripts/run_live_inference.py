#!/usr/bin/env python3
"""Live MoE Model Inference with MemTier-MoE Dynamic Tiering.

Demonstrates real model-in-the-loop inference:
  - Real MoE model weights loaded into memory
  - Non-expert weights (embeddings, attention, norms, LM head) resident on GPU
  - Expert weights partitioned across GPU HBM, host DRAM, and emulated CXL
  - Autoregressive generation with real gate routing, demand fetches, and LFU caching
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from memtier_moe.core.config import MemTierConfig
from memtier_moe.runtime.tiered_model import TieredMoEWrapper


def calibrate_routing_model(model: torch.nn.Module, tokenizer: AutoTokenizer, device: str = "cuda"):
    """Run a fast calibration pass on real weights to build a cross-layer expert co-occurrence model."""
    from memtier_moe.introspect.analysis import extract_co_occurrences
    from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel
    import numpy as np

    print("  Calibrating router co-occurrence on sample text...")
    calibration_text = (
        "Mixture-of-Experts architectures partition deep neural network layers into specialized "
        "feed-forward subnetworks. Routing mechanisms dynamically dispatch tokens to expert pairs, "
        "exhibiting high cross-layer correlation and temporal locality in modern language models."
    )
    inputs = tokenizer(calibration_text, return_tensors="pt")
    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    layers = getattr(model, "model", model).layers
    num_layers = len(layers)
    layer_topk = {}
    hooks = []

    for l_idx, layer in enumerate(layers):
        moe_block = getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
        if moe_block is not None and hasattr(moe_block, "gate"):
            top_k = getattr(moe_block, "num_experts_per_tok", getattr(moe_block, "top_k", 2))
            def _hook(l_i, k_val):
                def fn(module, inp, out):
                    logits = out if isinstance(out, torch.Tensor) else out[0]
                    if logits.dim() == 3:
                        logits = logits.squeeze(0)
                    indices = torch.topk(logits.detach(), k_val, dim=-1).indices.cpu().numpy()
                    layer_topk[l_i] = indices
                return fn
            hooks.append(moe_block.gate.register_forward_hook(_hook(l_idx, top_k)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    seq_len = inputs["input_ids"].shape[1]
    token_indices = []
    layer_indices = []
    expert_ids = []
    for t in range(seq_len):
        for l in range(num_layers):
            if l in layer_topk and t < len(layer_topk[l]):
                token_indices.append(t)
                layer_indices.append(l)
                expert_ids.append(layer_topk[l][t])

    stats = extract_co_occurrences(
        np.array(token_indices),
        np.array(layer_indices),
        np.array(expert_ids),
        num_layers=num_layers,
        lookahead=2,
    )
    co_model = CoOccurrenceModel(lookahead=2, min_probability=0.01)
    co_model.build_from_stats(stats)
    print(f"  Router predictor calibrated ({len(co_model._cond_probs)} cross-layer pairs learned)")
    return co_model


def main():
    parser = argparse.ArgumentParser(description="Live MoE Inference with MemTier-MoE")
    parser.add_argument(
        "--model-id",
        type=str,
        default="nopainkiller/Qwen1.5-4x0.5B-MoE",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Mixture-of-Experts architecture improves language model efficiency by",
        help="Input text prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=25,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--hbm-budget-mb",
        type=int,
        default=800,
        help="GPU HBM cache budget for experts in MB (forces offloading)",
    )
    parser.add_argument(
        "--dram-budget-mb",
        type=int,
        default=500,
        help="Host DRAM cache budget for experts in MB (forces spill to CXL)",
    )
    parser.add_argument(
        "--enable-prefetch",
        action="store_true",
        default=False,
        help="Enable router-predictive prefetcher",
    )
    args = parser.parse_args()

    # Ensure UTF-8 output on Windows terminal
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 75)
    print("MemTier-MoE: Real Model Live Inference Engine")
    print("=" * 75)
    device_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    print(f"Target Model: {args.model_id}")
    print(f"HBM Expert Cache Budget:  {args.hbm_budget_mb} MB")
    print(f"DRAM Expert Cache Budget: {args.dram_budget_mb} MB")
    print(f"Predictive Prefetch:      {'ENABLED' if args.enable_prefetch else 'DISABLED'}")

    # 1. Load Tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model
    print(f"\n[2/4] Loading real model weights ({args.model_id})...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    load_start = time.time()
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = base_model.to(device)
    print(f"Model loaded from disk in {time.time() - load_start:.2f}s")

    # Optional Prefetch Calibration
    co_occurr_model = None
    if args.enable_prefetch:
        co_occurr_model = calibrate_routing_model(base_model, tokenizer, device=device)

    # 3. Partition Model & Wrap with MemTier-MoE
    print("\n[3/4] Partitioning layers into 3-Tier Hierarchy (HBM / DRAM / CXL)...")
    config = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=args.hbm_budget_mb * 1024 * 1024,
        host_dram_bytes=args.dram_budget_mb * 1024 * 1024,
        cxl_memory_bytes=8 * 1024 * 1024 * 1024,
    )

    tiered_model = TieredMoEWrapper(
        model=base_model,
        config=config,
        enable_prefetch=args.enable_prefetch,
        co_occurrence_model=co_occurr_model,
        initial_hbm_budget_bytes=args.hbm_budget_mb * 1024 * 1024,
    )

    # Initial distribution
    tm = tiered_model.engine.tier_manager
    from memtier_moe.core.types import MemoryTier
    hbm_count = len(tm.get_pool(MemoryTier.HBM).experts)
    dram_count = len(tm.get_pool(MemoryTier.DRAM).experts)
    cxl_count = len(tm.get_pool(MemoryTier.CXL).experts)
    used_hbm, _ = tm.hbm_usage()
    print(f"Tier Manager Initial Distribution:")
    print(f"  Total Registered Experts: {len(tm._registry)}")
    print(f"  Hot Tier (HBM):           {hbm_count} experts ({used_hbm / 1e6:.1f} MB / {args.hbm_budget_mb} MB limit)")
    print(f"  Warm Tier (Host DRAM):    {dram_count} experts")
    print(f"  Cold Tier (Emulated CXL): {cxl_count} experts")

    # 4. Run Autoregressive Generation
    print(f"\n[4/4] Executing Live Generation for prompt:")
    print(f"  >>> \"{args.prompt}\" <<<\n")
    print("-" * 75)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = tiered_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=False,
            repetition_penalty=1.2,
        )
    total_time = time.perf_counter() - t0

    num_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]
    print("\n" + "-" * 75)
    print("\n[Generation Complete]")
    print(f"Generated Tokens: {num_generated}")
    print(f"Inference Wall Time: {total_time:.2f}s")
    print(f"Throughput: {num_generated / total_time:.2f} tok/s")

    # Metrics Summary
    stats = tiered_model.report()
    m = stats["metrics"]
    print("\n" + "=" * 75)
    print("MemTier-MoE Runtime Statistics:")
    print("=" * 75)
    print(f"  Cache Hit Rate:     {m.get('hit_rate', 0.0):.1%}")
    print(f"  Cache Hits:         {m.get('cache_hits', 0):,}")
    print(f"  Cache Misses:       {m.get('cache_misses', 0):,}")
    print(f"  LFU Evictions:      {m.get('evictions', 0):,}")
    print(f"  Total Transferred:  {m.get('total_transfer_bytes', 0) / 1e6:.2f} MB")
    print(f"  Transfer Time:      {m.get('total_transfer_time_ms', 0.0):.2f} ms")
    if args.enable_prefetch and "scheduler_stats" in stats:
        sched = stats["scheduler_stats"]
        print(f"  Prefetch Issued:    {sched.get('prefetch_total', 0):,}")
        print(f"  Prefetch Useful:    {sched.get('prefetch_useful', 0):,}")
        print(f"  Prefetch Pending:   {sched.get('prefetch_pending', 0):,}")
        print(f"  Prefetch Precision: {sched.get('prefetch_precision', 0.0):.1%}")
    print("=" * 75)


if __name__ == "__main__":
    main()
