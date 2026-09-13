import json
import os
from pathlib import Path
import psutil
import torch
from transformers import AutoConfig, AutoTokenizer

def run_check():
    target = Path("models/Qwen1.5-MoE-A2.7B")
    print("============================================================")
    print(" 1. Directory & File Inventory")
    print("============================================================")
    files = list(target.iterdir())
    total_bytes = 0
    for f in sorted(files, key=lambda x: x.name):
        if f.is_file():
            size_mb = f.stat().st_size / (1024**2)
            total_bytes += f.stat().st_size
            print(f"  {f.name:<38} {size_mb:>9.2f} MB")
        elif f.is_dir():
            subdir_size = sum(sf.stat().st_size for sf in f.rglob("*") if sf.is_file())
            print(f"  [DIR] {f.name:<32} {subdir_size / (1024**2):>9.2f} MB")
    print(f"  --> Total Top-Level Size: {total_bytes / (1024**3):.2f} GB")

    print("\n============================================================")
    print(" 2. Shard & Weight Map Integrity")
    print("============================================================")
    index_file = target / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as fp:
            index_data = json.load(fp)
        weight_map = index_data.get("weight_map", {})
        shards_in_map = sorted(set(weight_map.values()))
        print(f"Total tensors mapped in index: {len(weight_map)}")
        print(f"Referenced shards in index: {len(shards_in_map)}")
        all_ok = True
        for s in shards_in_map:
            s_path = target / s
            exists = s_path.exists()
            if not exists:
                all_ok = False
            status = "EXISTS (OK)" if exists else "MISSING (ERROR)"
            s_size = f"{s_path.stat().st_size / (1024**3):.2f} GB" if exists else "N/A"
            print(f"  {s:<38} {status:<15} {s_size}")
        print(f"All Shards Present: {'YES [PASS]' if all_ok else 'NO [FAIL]'}")
    else:
        print("index file not found!")

    print("\n============================================================")
    print(" 3. Cache & Temp Files Cleanliness")
    print("============================================================")
    cache_dir = target / ".cache"
    if cache_dir.exists():
        incompletes = list(cache_dir.rglob("*.incomplete"))
        locks = list(cache_dir.rglob("*.lock"))
        print(f"Incomplete download shards: {len(incompletes)}")
        for inc in incompletes:
            print(f"  - {inc.name}")
        print(f"Active lock files: {len(locks)}")
        for lk in locks:
            print(f"  - {lk.name}")
        if not incompletes and not locks:
            print("  --> Cache clean, all transfers fully committed to disk.")
    else:
        print("No .cache directory.")

    print("\n============================================================")
    print(" 4. Tokenizer & Config Load Test")
    print("============================================================")
    try:
        cfg = AutoConfig.from_pretrained(str(target))
        tok = AutoTokenizer.from_pretrained(str(target))
        print("AutoConfig:")
        print(f"  model_type: {cfg.model_type}")
        print(f"  num_hidden_layers: {cfg.num_hidden_layers}")
        print(f"  num_experts: {getattr(cfg, 'num_experts', 'N/A')}")
        print(f"  num_experts_per_tok: {getattr(cfg, 'num_experts_per_tok', 'N/A')}")
        print(f"  hidden_size: {cfg.hidden_size}")
        print("AutoTokenizer:")
        print(f"  vocab_size: {tok.vocab_size}")
        test_enc = tok.encode("Testing Qwen1.5-MoE-A2.7B")
        print(f"  Tokenization Test: {test_enc} -> '{tok.decode(test_enc)}'")
        print("  Tokenizer & Config Status: [PASS]")
    except Exception as e:
        print(f"Config/Tokenizer test failed: {e}")

    print("\n============================================================")
    print(" 5. System Memory Snapshot")
    print("============================================================")
    vm = psutil.virtual_memory()
    print(f"System RAM: {vm.total / (1024**3):.1f} GB total, {vm.available / (1024**3):.1f} GB available ({vm.percent}% used)")
    if torch.cuda.is_available():
        d = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(d)
        alloc = torch.cuda.memory_allocated(d) / (1024**3)
        res = torch.cuda.memory_reserved(d) / (1024**3)
        print(f"GPU: {props.name} ({props.total_memory / (1024**3):.2f} GB total)")
        print(f"  CUDA Allocated: {alloc:.2f} GB, Reserved: {res:.2f} GB")

if __name__ == "__main__":
    run_check()
