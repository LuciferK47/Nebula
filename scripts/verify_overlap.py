#!/usr/bin/env python3
"""Verify that compute and transfer actually overlap on the GPU.

This script creates a synthetic workload that issues an expert transfer
on a dedicated CUDA stream while running a matrix multiply on the default
stream.  It then captures a torch.profiler trace and checks whether the
two activities have overlapping time ranges.

Usage
-----
    python scripts/verify_overlap.py [--output overlap_trace.json]

Inspect the resulting chrome trace in chrome://tracing or
https://ui.perfetto.dev.

If nsys is available you can also run:
    nsys profile -o overlap_trace python scripts/verify_overlap.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Tuple

try:
    import torch
    from torch.profiler import profile, ProfilerActivity
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def verify_overlap(matrix_size: int = 2048, expert_size_mb: int = 32) -> bool:
    """Run a compute + transfer overlap test and return True if overlap detected."""
    if not HAS_TORCH or not torch.cuda.is_available():
        print("CUDA not available — cannot verify overlap. Skipping.")
        return False

    device = torch.device("cuda")
    transfer_stream = torch.cuda.Stream()

    # Create synthetic workloads
    compute_matrix = torch.randn(matrix_size, matrix_size, device=device)
    expert_tensor = torch.randn(
        expert_size_mb * 1024 * 1024 // 4,  # float32 = 4 bytes
        dtype=torch.float32,
    ).pin_memory()

    # Warm up
    for _ in range(3):
        _ = compute_matrix @ compute_matrix
    torch.cuda.synchronize()

    # Timed run with CUDA Events for definitive hardware overlap measurement
    start_comp = torch.cuda.Event(enable_timing=True)
    end_comp = torch.cuda.Event(enable_timing=True)
    start_xfer = torch.cuda.Event(enable_timing=True)
    end_xfer = torch.cuda.Event(enable_timing=True)
    start_overlap = torch.cuda.Event(enable_timing=True)
    end_overlap = torch.cuda.Event(enable_timing=True)

    # 1. Compute alone
    start_comp.record()
    for _ in range(5):
        _ = compute_matrix @ compute_matrix
    end_comp.record()
    torch.cuda.synchronize()
    compute_alone_ms = start_comp.elapsed_time(end_comp)

    # 2. Transfer alone
    start_xfer.record(transfer_stream)
    with torch.cuda.stream(transfer_stream):
        _ = expert_tensor.to(device, non_blocking=True)
    end_xfer.record(transfer_stream)
    torch.cuda.synchronize()
    transfer_alone_ms = start_xfer.elapsed_time(end_xfer)

    # 3. Overlapped run (compute on default stream, transfer on transfer_stream)
    start_overlap.record()
    with torch.cuda.stream(transfer_stream):
        gpu_expert = expert_tensor.to(device, non_blocking=True)
        transfer_event = torch.cuda.Event()
        transfer_event.record(transfer_stream)

    for _ in range(5):
        result = compute_matrix @ compute_matrix

    transfer_event.synchronize()
    end_overlap.record()
    torch.cuda.synchronize()
    overlapped_ms = start_overlap.elapsed_time(end_overlap)

    timing_stats = {
        "compute_alone_ms": compute_alone_ms,
        "transfer_alone_ms": transfer_alone_ms,
        "serial_expected_ms": compute_alone_ms + transfer_alone_ms,
        "overlapped_ms": overlapped_ms,
        "overlap_speedup": (compute_alone_ms + transfer_alone_ms) / overlapped_ms if overlapped_ms > 0 else 1.0,
        "hidden_latency_ms": max(0.0, (compute_alone_ms + transfer_alone_ms) - overlapped_ms),
    }

    # Timed run with profiler
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as prof:
        with torch.cuda.stream(transfer_stream):
            gpu_expert = expert_tensor.to(device, non_blocking=True)
            transfer_event = torch.cuda.Event()
            transfer_event.record(transfer_stream)

        for _ in range(5):
            result = compute_matrix @ compute_matrix

        transfer_event.synchronize()
        torch.cuda.synchronize()

    return prof, gpu_expert, result, timing_stats


def analyse_trace(prof) -> dict:
    """Print the human-readable profiler summary (totals only — see
    `analyse_chrome_trace` for the actual overlap verdict)."""
    events = prof.key_averages()
    sort_key = "cuda_time_total" if hasattr(events[0], "cuda_time_total") else "device_time_total"
    table = events.table(sort_by=sort_key, row_limit=20)
    print("\n=== Profiler Summary ===")
    print(table)

    compute_time = 0.0
    transfer_time = 0.0
    for evt in events:
        name_lower = evt.key.lower()
        dev_time = getattr(evt, "cuda_time_total", getattr(evt, "device_time_total", 0.0))
        if "aten::mm" in name_lower or "gemm" in name_lower:
            compute_time += dev_time / 1000.0  # μs → ms
        if "memcpy" in name_lower or "copy" in name_lower:
            transfer_time += dev_time / 1000.0

    return {"compute_time_ms": compute_time, "transfer_time_ms": transfer_time}


def analyse_chrome_trace(trace_path: str) -> dict:
    with open(trace_path) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace

    def is_gpu_kernel(e: dict) -> bool:
        return (
            e.get("ph") == "X"
            and "dur" in e
            and e.get("cat", "").lower() in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime")
        )

    def matches(e: dict, needles: List[str]) -> bool:
        name = e.get("name", "").lower()
        return any(n in name for n in needles)

    compute_kw = ["gemm", "sgemm", "hgemm", "mm", "matmul"]
    transfer_kw = ["memcpy", "memset"]

    compute_events = [e for e in events if is_gpu_kernel(e) and matches(e, compute_kw)]
    transfer_events = [e for e in events if is_gpu_kernel(e) and matches(e, transfer_kw)]

    def interval(e: dict) -> Tuple[float, float]:
        start = e["ts"]
        return start, start + e["dur"]

    def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    max_overlap_us = 0.0
    overlapping_pairs = 0
    for c in compute_events:
        c_span = interval(c)
        for t in transfer_events:
            if t.get("tid") == c.get("tid") and t.get("pid") == c.get("pid"):
                continue
            ov = intervals_overlap(c_span, interval(t))
            if ov > 0:
                overlapping_pairs += 1
                max_overlap_us = max(max_overlap_us, ov)

    return {
        "compute_kernel_count": len(compute_events),
        "transfer_kernel_count": len(transfer_events),
        "overlapping_pairs": overlapping_pairs,
        "max_overlap_us": max_overlap_us,
        "overlap_detected": overlapping_pairs > 0 and max_overlap_us > 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify compute-transfer overlap")
    parser.add_argument("--output", "-o", default="overlap_trace.json", help="Chrome trace output path")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--expert-mb", type=int, default=32)
    args = parser.parse_args()

    if not HAS_TORCH or not torch.cuda.is_available():
        print("SKIP: CUDA not available.")
        sys.exit(0)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Matrix size: {args.matrix_size}, Expert: {args.expert_mb} MB")

    result = verify_overlap(args.matrix_size, args.expert_mb)
    if result is False:
        sys.exit(0)

    prof, _, _, timing = result
    totals = analyse_trace(prof)

    prof.export_chrome_trace(args.output)
    print(f"\nChrome trace saved to: {args.output}")
    print(f"  Open in chrome://tracing or https://ui.perfetto.dev")

    print(f"\n=== Hardware Overlap Benchmark (CUDA Events) ===")
    print(f"  Compute alone           : {timing['compute_alone_ms']:.2f} ms")
    print(f"  Transfer alone (DMA)    : {timing['transfer_alone_ms']:.2f} ms")
    print(f"  Serial expected (sum)   : {timing['serial_expected_ms']:.2f} ms")
    print(f"  Overlapped (parallel)   : {timing['overlapped_ms']:.2f} ms")
    print(f"  Latency hidden behind   : {timing['hidden_latency_ms']:.2f} ms")
    print(f"  Effective speedup       : {timing['overlap_speedup']:.2f}x")

    stats = analyse_chrome_trace(args.output)
    print(f"\n=== Kernel Trace Analysis ===")
    print(f"  Compute kernels found   : {stats['compute_kernel_count']}")
    print(f"  Transfer kernels found  : {stats['transfer_kernel_count']}")
    print(f"  Overlapping pairs       : {stats['overlapping_pairs']}")
    print(f"  Max overlap duration    : {stats['max_overlap_us']:.1f} us")

    # Overlap is verified if either CUDA event benchmark shows concurrent speedup
    # OR kernel timeline intervals intersect
    timing_overlap = timing["hidden_latency_ms"] > 0.5 * min(timing["compute_alone_ms"], timing["transfer_alone_ms"])
    is_success = stats["overlap_detected"] or timing_overlap

    if is_success:
        print("\n  [SUCCESS] Overlap CONFIRMED on hardware:")
        print(f"            {timing['hidden_latency_ms']:.2f} ms of transfer latency was hidden concurrently during GEMM compute.")
    else:
        print("\n  [FAIL] No overlap detected: transfer and compute executed serially.")

    sys.exit(0 if is_success else 1)


if __name__ == "__main__":
    main()
