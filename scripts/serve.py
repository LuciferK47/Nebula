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
LOCAL_CHAT_MOE = os.path.join(REPO_ROOT, "models", "Qwen1.5-4x0.5B-Chat-MoE")
# No fallback to a hub model here: Qwen/Qwen1.5-MoE-A2.7B is 26.68 GB in
# fp16, which does not fit a 6 GB card and would silently try to download
# ~28 GB on the first /api/run call. Same reasoning as
# scripts/run_scenarios.py's --model-id handling.
DEFAULT_MODEL_ID = LOCAL_CHAT_MOE if os.path.exists(LOCAL_CHAT_MOE) else None

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
        "name": "Hybrid SOTA (Activation Offload)",
        "badge": "Recommended / SOTA",
        "color": "#10b981",
        "hbm_budget_mb": 600,
        "dram_budget_mb": 1500,
        "cxl_budget_mb": 1500,
        "execution_mode": "hybrid",
        "enable_prefetch": False,
        "enable_lookahead": False,
        "description": "Transfers activations (KB) instead of weights (MB) for cache misses. Zero weight evictions, minimal PCIe traffic.",
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
        "name": "Hybrid SOTA (Dynamic Headroom)",
        "badge": "Dynamic Headroom",
        "color": "#a855f7",
        "hbm_budget_mb": 900,
        "dram_budget_mb": 1500,
        "cxl_budget_mb": 1500,
        "execution_mode": "hybrid",
        "enable_prefetch": False,
        "enable_lookahead": False,
        "description": "Dynamically expands HBM headroom (1.5x constraint) for hot expert retention, boosting hit rate and throughput.",
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
        "default_model": (
            "Qwen1.5-4x0.5B-Chat-MoE (Instruction-Tuned)" if DEFAULT_MODEL_ID
            else "not built — run scripts/build_chat_moe.py"
        ),
        "active_tiers": ["HBM (GPU VRAM)", "Host DRAM (Pinned)", "CXL Memory Pool (Emulated Tier-3)"],
    }


def load_model_resources():
    """Lazily load model and tokenizer onto GPU once."""
    global CACHED_MODEL, CACHED_TOKENIZER, CACHED_CO_MODEL, CACHED_FREQ_MAP
    with MODEL_LOCK:
        if CACHED_MODEL is not None:
            return CACHED_MODEL, CACHED_TOKENIZER, CACHED_CO_MODEL, CACHED_FREQ_MAP

        if DEFAULT_MODEL_ID is None:
            raise RuntimeError(
                f"No model found at {LOCAL_CHAT_MOE}. Build it first with:\n"
                f"    python scripts/build_chat_moe.py\n"
                f"There is deliberately no fallback to a HF hub model here — the natural "
                f"fallback, Qwen/Qwen1.5-MoE-A2.7B, is 26.68 GB in fp16 and does not fit a "
                f"6 GB card."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.run_live_benchmark import load_or_calibrate_co_occurrence

        print(f"[Server] Loading tokenizer for {DEFAULT_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
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


# (result key, filename, default when absent) — table-driven so adding a
# new results/*.json file the frontend should see is a one-line change
# instead of another copy-pasted try/except block. Keys beyond the
# original three (live_benchmarks, stress_test, pipeline_benchmarks) are
# additive: the real-MoE data (qwen14b_*) was previously produced but had
# no endpoint at all, making it unreachable from the frontend.
_HISTORICAL_FILES: List[tuple] = [
    ("live_benchmarks", "live_benchmark_results.json", []),
    ("stress_test", "stress_test_report.json", {}),
    ("pipeline_benchmarks", "benchmark_results.json", {}),
    ("qwen14b_live_metrics", "qwen14b_live_metrics.json", {}),
    ("qwen14b_benchmark_results", "qwen14b_benchmark_results.json", []),
    ("established_benchmark_results", "established_benchmark_results.json", {}),
    ("cxl_capacity_sweep", "cxl_capacity_sweep_results.json", []),
    ("cxl_emulation_ablation", "cxl_emulation_ablation_results.json", {}),
]


def load_historical_results() -> Dict[str, Any]:
    """Read pre-computed live hardware benchmark and stress test files."""
    results: Dict[str, Any] = {}
    for key, filename, default in _HISTORICAL_FILES:
        results[key] = default
        path = os.path.join(REPO_ROOT, "results", filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results[key] = json.load(f)
            except Exception as e:
                print(f"[Server] Warning reading {path}: {e}")

    # scripts/run_scenarios.py writes one JSON file per scenario (s1, s2, ...)
    # into results/scenarios/ — surface all of them keyed by scenario id so
    # the frontend can render whichever ones have been run without the
    # server needing to know the scenario list in advance. Only *.json:
    # the same directory also holds non-JSON trace exports (s3_trace.csv,
    # s3_trace.dramsim3, s3_trace_gem5.txt) that aren't meant for this API.
    scenarios_dir = os.path.join(REPO_ROOT, "results", "scenarios")
    scenarios: Dict[str, Any] = {}
    if os.path.isdir(scenarios_dir):
        for fname in sorted(os.listdir(scenarios_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(scenarios_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    scenarios[fname[:-len(".json")]] = json.load(f)
            except Exception as e:
                print(f"[Server] Warning reading {fpath}: {e}")
    results["scenarios"] = scenarios

    return results


def run_single_inference(
    prompt: str,
    baseline_id: str,
    max_new_tokens: int = 25,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    memory_constraint_mb: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute live prompt generation on GPU using selected baseline."""
    matched_spec = next(
        (s for s in BASELINE_SPECS if s["id"] == baseline_id or (s["id"] == "hybrid_sota_600" and baseline_id in ("hybrid_sota", "hybrid_sota_unconstrained"))),
        BASELINE_SPECS[1]
    )
    spec = dict(matched_spec)

    # Dynamic memory constraint override from frontend
    if memory_constraint_mb is not None:
        try:
            val = int(memory_constraint_mb)
            if val <= 0 or val >= 2500:
                spec["hbm_budget_mb"] = 2500
            elif spec["id"] == "gpu_resident":
                spec["hbm_budget_mb"] = 2500
            elif spec["id"] == "hybrid_sota_900":
                spec["hbm_budget_mb"] = min(2500, int(val * 1.5))
            else:
                spec["hbm_budget_mb"] = val
        except (ValueError, TypeError):
            pass

    base_model, tokenizer, co_model, freq_map = load_model_resources()

    # Unconstrained tokens: if max_new_tokens <= 0 or None, generate naturally up to 2048 or EOS
    effective_max_tokens = 2048 if (max_new_tokens is None or int(max_new_tokens) <= 0) else int(max_new_tokens)

    from memtier_moe.core.config import MemTierConfig
    from memtier_moe.runtime.tiered_model import TieredMoEWrapper

    # Dynamically detect hardware memory so reviewer can test on any GPU or CPU without tinkering
    vram_bytes = 6 * 1024 * 1024 * 1024
    if torch.cuda.is_available():
        try:
            vram_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            pass

    config = MemTierConfig(
        gpu_vram_bytes=vram_bytes,
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

        # Format prompt with chat template if available for coherent instruction following
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(formatted_prompt, return_tensors="pt")
            except Exception:
                inputs = tokenizer(prompt, return_tensors="pt")
        else:
            inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        eos_ids = [151645, 151643]
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(tokenizer.eos_token_id)
        eos_ids = list(set([t for t in eos_ids if t is not None]))

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": effective_max_tokens,
            "repetition_penalty": 1.15,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id or 151643,
            "eos_token_id": eos_ids,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(float(temperature), 0.1)
            gen_kwargs["top_p"] = min(max(float(top_p), 0.1), 1.0)
        else:
            gen_kwargs["do_sample"] = False

        # Timed execution
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output_ids = wrapper.generate(
                **inputs,
                **gen_kwargs,
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
            "memory_constraint_mb": spec["hbm_budget_mb"],
            "timestamp": time.strftime("%H:%M:%S"),
        }


FRONTEND_DIST = os.path.join(REPO_ROOT, "frontend", "dist")


class MemTierRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Handler serving API endpoints and static frontend assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIST, **kwargs)

    def end_headers(self):
        # Allow cross-origin requests.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # Vite content-hashes filenames under /assets/ (e.g. index-Ch6tu0ye.js),
        # so those can be cached indefinitely — a given hash never changes
        # content, and a rebuild produces a new hash. Blanket no-store here
        # meant the whole 3D bundle re-downloaded on every reload. Everything
        # else (index.html, API responses) keeps the old no-cache behavior
        # since it can legitimately change between requests.
        if self.path.startswith("/assets/"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
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
            raw_tokens = payload.get("max_tokens")
            if raw_tokens is None:
                raw_tokens = payload.get("max_new_tokens")
            if raw_tokens is None:
                raw_tokens = 128

            if str(raw_tokens).strip().lower() in ("unlimited", "none", "auto", "0", "complete", "eos"):
                max_tokens = 0
            else:
                try:
                    max_tokens = int(raw_tokens)
                except (ValueError, TypeError):
                    max_tokens = 0

            raw_mem = payload.get("memory_constraint_mb") or payload.get("hbm_budget_mb")
            if str(raw_mem).lower() in ("unconstrained", "full", "max"):
                memory_constraint_mb = 2500
            elif raw_mem is None or str(raw_mem).lower() in ("none", "auto"):
                memory_constraint_mb = None
            else:
                try:
                    val = int(raw_mem)
                    memory_constraint_mb = 2500 if val <= 0 else val
                except (ValueError, TypeError):
                    memory_constraint_mb = None

            do_sample = bool(payload.get("do_sample", False))
            temperature = float(payload.get("temperature", 0.7))
            top_p = float(payload.get("top_p", 0.9))

            try:
                res = run_single_inference(
                    prompt=prompt,
                    baseline_id=baseline_id,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    memory_constraint_mb=memory_constraint_mb,
                )
                self.send_json_response(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({"error": str(e), "traceback": traceback.format_exc()}, status=500)
            return

        if path == "/api/compare":
            prompt = payload.get("prompt", "Mixture of Experts architecture enables efficient scaling.")
            raw_tokens = payload.get("max_tokens")
            if raw_tokens is None:
                raw_tokens = payload.get("max_new_tokens")
            if raw_tokens is None:
                raw_tokens = 128

            if str(raw_tokens).strip().lower() in ("unlimited", "none", "auto", "0", "complete", "eos"):
                max_tokens = 0
            else:
                try:
                    max_tokens = int(raw_tokens)
                except (ValueError, TypeError):
                    max_tokens = 0

            raw_mem = payload.get("memory_constraint_mb") or payload.get("hbm_budget_mb")
            if str(raw_mem).lower() in ("unconstrained", "full", "max"):
                memory_constraint_mb = 2500
            elif raw_mem is None or str(raw_mem).lower() in ("none", "auto"):
                memory_constraint_mb = None
            else:
                try:
                    val = int(raw_mem)
                    memory_constraint_mb = 2500 if val <= 0 else val
                except (ValueError, TypeError):
                    memory_constraint_mb = None

            selected_ids = payload.get("baselines", [b["id"] for b in BASELINE_SPECS])

            results = []
            try:
                for b_id in selected_ids:
                    print(f"[Server] Running comparison baseline: {b_id} (mem constraint: {memory_constraint_mb} MB)")
                    r = run_single_inference(
                        prompt=prompt,
                        baseline_id=b_id,
                        max_new_tokens=max_tokens,
                        memory_constraint_mb=memory_constraint_mb,
                    )
                    results.append(r)
                self.send_json_response({"prompt": prompt, "results": results, "memory_constraint_mb": memory_constraint_mb})
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

    if not os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        print("=" * 80)
        print(f"[Server] ERROR: no build found at {FRONTEND_DIST}")
        print("[Server] Run this first:")
        print("[Server]     cd frontend && npm install && npm run build")
        print("=" * 80)
        sys.exit(1)

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
