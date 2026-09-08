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

    # Timed run with profiler
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as prof:
        # Start async transfer
        with torch.cuda.stream(transfer_stream):
            gpu_expert = expert_tensor.to(device, non_blocking=True)
            transfer_event = torch.cuda.Event()
            transfer_event.record(transfer_stream)

        # Compute on default stream while transfer is in flight
        for _ in range(5):
            result = compute_matrix @ compute_matrix

        # Wait for transfer
        transfer_event.synchronize()

    return prof, gpu_expert, result


def analyse_trace(prof) -> dict:
    """Print the human-readable profiler summary (totals only — see
    `analyse_chrome_trace` for the actual overlap verdict)."""
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    print("\n=== Profiler Summary ===")
    print(table)

    events = prof.key_averages()
    compute_time = 0.0
    transfer_time = 0.0
    for evt in events:
        name_lower = evt.key.lower()
        if "aten::mm" in name_lower or "gemm" in name_lower:
            compute_time += evt.cuda_time_total / 1000.0  # μs → ms
        if "memcpy" in name_lower or "copy" in name_lower:
            transfer_time += evt.cuda_time_total / 1000.0

    return {"compute_time_ms": compute_time, "transfer_time_ms": transfer_time}


def analyse_chrome_trace(trace_path: str) -> dict:
    """Determine whether compute and transfer kernels actually ran
    concurrently on the GPU, from the exported Chrome trace.

    `key_averages()` totals (what `analyse_trace` prints) only tell you
    that both kinds of work happened *somewhere* in the profiled window —
    a compute-time total > 0 and a transfer-time total > 0 is consistent
    with the two running back-to-back on the same stream, which is exactly
    the non-overlapping case this check exists to catch. Real overlap
    means a compute kernel's [start, end) interval and a transfer kernel's
    interval intersect, on two different GPU streams (same-stream "overlap"
    is impossible — a stream serializes its own kernels). We check that
    directly, using the actual per-kernel timestamps Chrome/Perfetto trace
    format records (`ts`, `dur`, `tid` = stream id, `pid` = device).
    """
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
        """Return overlap duration in the same units as ts/dur (µs), 0 if none."""
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    max_overlap_us = 0.0
    overlapping_pairs = 0
    for c in compute_events:
        c_span = interval(c)
        # Different streams only — same-stream kernels never truly overlap.
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
        # A genuine, non-trivial overlap, not a rounding artifact.
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

    prof, _, _ = result
    totals = analyse_trace(prof)

    # Export chrome trace — this is also the source of truth for the
    # overlap verdict below, not just a debugging artifact.
    prof.export_chrome_trace(args.output)
    print(f"\nChrome trace saved to: {args.output}")
    print(f"  Open in chrome://tracing or https://ui.perfetto.dev")

    print(f"\n=== Overlap Analysis ===")
    print(f"  Compute CUDA time (total) : {totals['compute_time_ms']:.2f} ms")
    print(f"  Transfer CUDA time (total): {totals['transfer_time_ms']:.2f} ms")
    print(
        "  Note: nonzero totals for both only means both kinds of work "
        "happened somewhere in the window — including back-to-back on the "
        "same stream. The verdict below checks actual concurrent execution."
    )

    stats = analyse_chrome_trace(args.output)
    print(f"\n  Compute kernels found : {stats['compute_kernel_count']}")
    print(f"  Transfer kernels found: {stats['transfer_kernel_count']}")
    print(f"  Overlapping kernel pairs: {stats['overlapping_pairs']}")
    print(f"  Max overlap duration: {stats['max_overlap_us']:.1f} us")

    if stats["compute_kernel_count"] == 0 or stats["transfer_kernel_count"] == 0:
        print("  ⚠️  Could not find both kernel kinds in the trace — check the "
              "name/category matching in analyse_chrome_trace() against this "
              "torch/CUDA version's kernel naming.")
    elif stats["overlap_detected"]:
        print("  ✅ Overlap DETECTED — compute and transfer kernels ran concurrently on the GPU")
    else:
        print("  ❌ No overlap detected — transfer and compute ran serially. Check stream configuration.")

    sys.exit(0 if stats["overlap_detected"] else 1)


if __name__ == "__main__":
    main()
