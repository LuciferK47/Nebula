"""Test loading Qwen1.5-MoE-A2.7B using 4-bit quantization to fit safely within host RAM and GPU VRAM."""
import gc
import sys
import time
import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def monitor(stage: str):
    vm = psutil.virtual_memory()
    print(f"[{stage}] System RAM: {vm.available / (1024**3):.2f} GB free / {vm.total / (1024**3):.2f} GB total ({vm.percent}% used)")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        res = torch.cuda.memory_reserved() / (1024**3)
        print(f"[{stage}] CUDA VRAM: {alloc:.2f} GB alloc, {res:.2f} GB res")

def main():
    model_path = "models/Qwen1.5-MoE-A2.7B"
    monitor("Initial")

    print("\n1. Initializing Tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_path)

    print("\n2. Configuring 4-bit quantization (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n3. Loading model with device_map='auto' (max_memory: 4.5GiB GPU, 10GiB CPU)...")
    t0 = time.time()
    max_mem = {0: "4.5GiB", "cpu": "10GiB"}
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_mem,
            low_cpu_mem_usage=True,
        )
        print(f"\nSUCCESS: Qwen1.5-MoE-A2.7B loaded in {time.time() - t0:.2f}s!")
        monitor("After Model Load")
        
        print("\nDevice distribution of modules:")
        devices = {}
        for name, param in model.named_parameters():
            dev = str(param.device)
            devices[dev] = devices.get(dev, 0) + param.numel()
        for dev, count in devices.items():
            print(f"  {dev}: {count:,} parameters ({count * 0.5 / 1e6:.1f} MB in 4-bit)")

        # Quick test generation
        prompt = "Mixture of Experts overcomes the memory wall by"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        print(f"\n4. Generating 10 tokens from prompt: '{prompt}'...")
        t_gen = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        gen_time = time.time() - t_gen
        decoded = tok.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {decoded}")
        print(f"Generation time: {gen_time:.2f}s ({10 / gen_time:.2f} tok/s)")
        
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
