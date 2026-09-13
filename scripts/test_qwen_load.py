"""Test loading Qwen1.5-MoE-A2.7B safely with memory monitoring."""
import gc
import sys
import time
import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def monitor_mem(stage: str):
    vm = psutil.virtual_memory()
    print(f"[{stage}] System RAM: {vm.available / (1024**3):.2f} GB free / {vm.total / (1024**3):.2f} GB total ({vm.percent}% used)")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        res = torch.cuda.memory_reserved() / (1024**3)
        print(f"[{stage}] CUDA VRAM: {alloc:.2f} GB alloc, {res:.2f} GB res")

def main():
    model_path = "models/Qwen1.5-MoE-A2.7B"
    monitor_mem("Initial")

    print(f"\n1. Loading Config and Tokenizer from {model_path}...")
    cfg = AutoConfig.from_pretrained(model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    print(f"Config loaded: {cfg.model_type}, {cfg.num_hidden_layers} layers, {getattr(cfg, 'num_experts', 60)} experts")

    print("\n2. Attempting to load model weights with low_cpu_mem_usage=True on CPU...")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        print(f"SUCCESS: Model loaded on CPU in {time.time() - t0:.2f}s!")
        monitor_mem("After Model Load")
        return model, tok
    except Exception as e:
        print(f"FAILED to load model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()
