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
import sys
import time

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
    """Extract timing info from the profiler trace."""
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    print("\n=== Profiler Summary ===")
    print(table)

    # Look for overlap indicators
    events = prof.key_averages()
    compute_time = 0.0
    transfer_time = 0.0

    for evt in events:
        name_lower = evt.key.lower()
        if "aten::mm" in name_lower or "gemm" in name_lower:
            compute_time += evt.cuda_time_total / 1000.0  # μs → ms
        if "memcpy" in name_lower or "copy" in name_lower:
            transfer_time += evt.cuda_time_total / 1000.0

    return {
        "compute_time_ms": compute_time,
        "transfer_time_ms": transfer_time,
        "overlap_likely": compute_time > 0 and transfer_time > 0,
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
    stats = analyse_trace(prof)

    # Export chrome trace
    prof.export_chrome_trace(args.output)
    print(f"\nChrome trace saved to: {args.output}")
    print(f"  Open in chrome://tracing or https://ui.perfetto.dev")

    print(f"\n=== Overlap Analysis ===")
    print(f"  Compute CUDA time : {stats['compute_time_ms']:.2f} ms")
    print(f"  Transfer CUDA time: {stats['transfer_time_ms']:.2f} ms")

    if stats["overlap_likely"]:
        print("  ✅ Overlap DETECTED — compute and transfer ran concurrently")
    else:
        print("  ❌ No overlap detected — check stream configuration")

    sys.exit(0 if stats["overlap_likely"] else 1)


if __name__ == "__main__":
    main()
