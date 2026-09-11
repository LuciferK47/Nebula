#!/usr/bin/env python3
"""Interactive Presentation Server for MemTier-MoE.

Provides a responsive Web API and static frontend for live prompt execution,
baseline comparative benchmarking, and real-time visualization of throughput,
cache hit rate, evictions, and PCIe memory bus traffic.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

MODEL_LOCK = threading.Lock()
CACHED_MODEL = None
CACHED_TOKENIZER = None
CACHED_CO_MODEL = None
CACHED_FREQ_MAP = None
DEFAULT_MODEL_ID = "nopainkiller/Qwen1.5-4x0.5B-MoE"

BASELINE_SPECS = [
    {
        "id": "gpu_resident",
        "name": "GPU-Resident (Full VRAM)",
        "badge": "Upper Bound",
        "color": "#6366f1",
        "hbm_budget_mb": 2500,
        "dram_budget_mb": 500,
        "cxl_budget_mb": 0,
        "execution_mode": "weight_transfer",
        "enable_prefetch": False,
        "enable_lookahead": False,
        "description": "Entire 2.5GB model resident in GPU VRAM. Zero offloading, maximum memory cost.",
    },
    {
        "id": "hybrid_sota_600",
        "name": "Hybrid SOTA (Fiddler 600MB)",
        "badge": "Recommended / SOTA",
        "color": "#10b981",
        "hbm_budget_mb": 600,
        "dram_budget_mb": 1500,
        "cxl_budget_mb": 1500,
        "execution_mode": "hybrid",
        "enable_prefetch": False,
        "enable_lookahead": False,
        "description": "Transfers activations instead of weights for cache misses. 0 evictions, minimal PCIe traffic.",
    },
    {
        "id": "lookahead_600",
        "name": "Lookahead Pre-Gating (600MB)",
        "badge": "Prefetch Gated",
        "color": "#38bdf8",
        "hbm_budget_mb": 600,
        "dram_budget_mb": 1500,
        "cxl_budget_mb": 1500,
        "execution_mode": "weight_transfer",
        "enable_prefetch": True,
        "enable_lookahead": True,
        "description": "Predictive expert prefetching with threshold gating (tau=0.15) to curb cache thrashing.",
    },
    {
        "id": "two_tier_600",
        "name": "Two-Tier (Weight Transfer 600MB)",
        "badge": "Naive Baseline",
        "color": "#f43f5e",
        "hbm_budget_mb": 600,
        "dram_budget_mb": 1500,
        "cxl_budget_mb": 0,
        "execution_mode": "weight_transfer",
        "enable_prefetch": False,
        "enable_lookahead": False,
        "description": "Conventional LRU expert weight-swapping over PCIe. High cache thrashing and bus saturation.",
    },
    {
        "id": "hybrid_sota_900",
        "name": "Hybrid SOTA (Fiddler 900MB)",
        "badge": "High Hit Rate",
        "color": "#a855f7",
        "hbm_budget_mb": 900,
        "dram_budget_mb": 1500,
        "cxl_budget_mb": 1500,
        "execution_mode": "hybrid",
        "enable_prefetch": False,
        "enable_lookahead": False,
        "description": "900MB HBM budget with hybrid execution, surpassing GPU-resident throughput.",
    },
]


def get_system_info() -> Dict[str, Any]:
    """Inspect physical execution environment."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    vram_total_gb = 0.0
    vram_used_gb = 0.0
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = round(props.total_memory / (1024**3), 2)
        vram_used_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 2)

    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_total_gb": vram_total_gb,
        "vram_used_gb": vram_used_gb,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "default_model": DEFAULT_MODEL_ID,
        "active_tiers": ["HBM (GPU VRAM)", "Host DRAM (Pinned)", "CXL Memory Pool (Emulated Tier-3)"],
    }


def load_model_resources():
    """Lazily load model and tokenizer onto GPU once."""
    global CACHED_MODEL, CACHED_TOKENIZER, CACHED_CO_MODEL, CACHED_FREQ_MAP
    with MODEL_LOCK:
        if CACHED_MODEL is not None:
            return CACHED_MODEL, CACHED_TOKENIZER, CACHED_CO_MODEL, CACHED_FREQ_MAP

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.run_live_benchmark import load_or_calibrate_co_occurrence

        print(f"[Server] Loading tokenizer for {DEFAULT_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        print(f"[Server] Loading {DEFAULT_MODEL_ID} on {device.upper()} (dtype: {dtype})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_ID,
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        base_model.eval()

        print("[Server] Loading co-occurrence routing profile...")
        co_model = load_or_calibrate_co_occurrence("traces/routing_trace_wikitext.npz", model=base_model, tokenizer=tokenizer)
        freq_map = getattr(getattr(co_model, "stats", None), "marginal_counts", None)

        CACHED_MODEL = base_model
        CACHED_TOKENIZER = tokenizer
        CACHED_CO_MODEL = co_model
        CACHED_FREQ_MAP = freq_map
        print("[Server] Model resources initialized successfully.")
        return CACHED_MODEL, CACHED_TOKENIZER, CACHED_CO_MODEL, CACHED_FREQ_MAP


def load_historical_results() -> Dict[str, Any]:
    """Read pre-computed live hardware benchmark and stress test files."""
    live_path = os.path.join(REPO_ROOT, "results", "live_benchmark_results.json")
    stress_path = os.path.join(REPO_ROOT, "results", "stress_test_report.json")
    benchmark_path = os.path.join(REPO_ROOT, "results", "benchmark_results.json")

    results: Dict[str, Any] = {
        "live_benchmarks": [],
        "stress_test": {},
        "pipeline_benchmarks": {},
    }

    if os.path.exists(live_path):
        try:
            with open(live_path, "r", encoding="utf-8") as f:
                results["live_benchmarks"] = json.load(f)
        except Exception as e:
            print(f"[Server] Warning reading {live_path}: {e}")

    if os.path.exists(stress_path):
        try:
            with open(stress_path, "r", encoding="utf-8") as f:
                results["stress_test"] = json.load(f)
        except Exception as e:
            print(f"[Server] Warning reading {stress_path}: {e}")

    if os.path.exists(benchmark_path):
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                results["pipeline_benchmarks"] = json.load(f)
        except Exception as e:
            print(f"[Server] Warning reading {benchmark_path}: {e}")

    return results


def run_single_inference(
    prompt: str,
    baseline_id: str,
    max_new_tokens: int = 25,
) -> Dict[str, Any]:
    """Execute live prompt generation on GPU using selected baseline."""
    spec = next((s for s in BASELINE_SPECS if s["id"] == baseline_id), BASELINE_SPECS[1])

    base_model, tokenizer, co_model, freq_map = load_model_resources()

    from memtier_moe.core.config import MemTierConfig
    from memtier_moe.runtime.tiered_model import TieredMoEWrapper

    config = MemTierConfig(
        gpu_vram_bytes=6 * 1024 * 1024 * 1024,
        hbm_cache_budget_bytes=spec["hbm_budget_mb"] * 1024 * 1024,
        host_dram_bytes=spec["dram_budget_mb"] * 1024 * 1024,
        cxl_memory_bytes=spec["cxl_budget_mb"] * 1024 * 1024,
        max_prefetches_per_decision=2,
    )

    with MODEL_LOCK:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        wrapper = TieredMoEWrapper(
            model=base_model,
            config=config,
            enable_prefetch=spec["enable_prefetch"],
            co_occurrence_model=co_model,
            initial_hbm_budget_bytes=spec["hbm_budget_mb"] * 1024 * 1024,
            execution_mode=spec["execution_mode"],
            enable_lookahead_gating=spec["enable_lookahead"],
            expert_frequency=freq_map,
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Timed execution
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output_ids = wrapper.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - t0

        input_len = inputs["input_ids"].shape[1]
        generated_token_count = output_ids.shape[1] - input_len
        new_token_ids = output_ids[0][input_len:]
        new_tokens_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        report = wrapper.report()
        m = report["metrics"]
        sched = report.get("scheduler_stats", {})

        wrapper.unpatch()

        tok_per_sec = generated_token_count / wall_time if wall_time > 0 else 0.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

        return {
            "baseline": spec,
            "prompt": prompt,
            "generated_text": new_tokens_text,
            "full_text": full_text,
            "token_ids": [int(t) for t in new_token_ids[:30]],
            "generated_tokens": generated_token_count,
            "wall_time_seconds": round(wall_time, 3),
            "tokens_per_second": round(tok_per_sec, 2),
            "hit_rate": round(m.get("hit_rate", 0.0), 4),
            "cache_hits": m.get("cache_hits", 0),
            "cache_misses": m.get("cache_misses", 0),
            "evictions": m.get("evictions", 0),
            "transfer_bytes": m.get("total_transfer_bytes", 0),
            "transfer_mb": round(m.get("total_transfer_bytes", 0) / 1e6, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "prefetch_precision": round(sched.get("prefetch_precision", 0.0), 4) if spec["enable_prefetch"] else 0.0,
            "timestamp": time.strftime("%H:%M:%S"),
        }


class MemTierRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Handler serving API endpoints and static frontend assets."""

    def __init__(self, *args, **kwargs):
        presentation_dir = os.path.join(REPO_ROOT, "presentation")
        super().__init__(*args, directory=presentation_dir, **kwargs)

    def end_headers(self):
        # Allow cross-origin requests and disable aggressive browser caching for dev
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def send_json_response(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/system_info":
            try:
                info = get_system_info()
                self.send_json_response(info)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=500)
            return

        if path == "/api/baselines":
            self.send_json_response(BASELINE_SPECS)
            return

        if path == "/api/historical_results":
            try:
                data = load_historical_results()
                self.send_json_response(data)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=500)
            return

        # Serve presentation frontend
        if path == "/" or path == "":
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(post_body.decode("utf-8")) if post_body else {}
        except json.JSONDecodeError:
            self.send_json_response({"error": "Malformed JSON payload"}, status=400)
            return

        if path == "/api/run":
            prompt = payload.get("prompt", "Mixture of Experts architecture enables efficient scaling.")
            baseline_id = payload.get("baseline_id", "hybrid_sota_600")
            max_tokens = int(payload.get("max_tokens", 25))

            try:
                res = run_single_inference(prompt=prompt, baseline_id=baseline_id, max_new_tokens=max_tokens)
                self.send_json_response(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({"error": str(e), "traceback": traceback.format_exc()}, status=500)
            return

        if path == "/api/compare":
            prompt = payload.get("prompt", "Mixture of Experts architecture enables efficient scaling.")
            max_tokens = int(payload.get("max_tokens", 25))
            selected_ids = payload.get("baselines", [b["id"] for b in BASELINE_SPECS])

            results = []
            try:
                for b_id in selected_ids:
                    print(f"[Server] Running comparison baseline: {b_id}")
                    r = run_single_inference(prompt=prompt, baseline_id=b_id, max_new_tokens=max_tokens)
                    results.append(r)
                self.send_json_response({"prompt": prompt, "results": results})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({"error": str(e), "traceback": traceback.format_exc()}, status=500)
            return

        self.send_json_response({"error": f"Path not found: {path}"}, status=404)


def main():
    parser = argparse.ArgumentParser(description="Start MemTier-MoE Presentation Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, MemTierRequestHandler)
    print("=" * 80)
    print(f"[MemTier-MoE] Presentation Frontend Server running at:")
    print(f"   Local:   http://localhost:{args.port}/")
    print(f"   Network: http://{args.host}:{args.port}/")
    print("=" * 80)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down cleanly...")
        httpd.server_close()


if __name__ == "__main__":
    main()
