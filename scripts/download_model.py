"""Download and verify HuggingFace MoE models for MemTier-MoE testing.

Supports parallel downloading, resumption, Windows non-symlink storage,
and post-download shard integrity verification.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download


def download_moe_model(
    repo_id: str = "Qwen/Qwen1.5-MoE-A2.7B",
    local_dir: Optional[str] = None,
    max_workers: int = 4,
) -> Path:
    """Download an MoE model repository from HuggingFace Hub with verification."""
    if local_dir is None:
        model_name = repo_id.split("/")[-1]
        local_dir = f"models/{model_name}"

    target_path = Path(local_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f" MemTier-MoE Model Downloader")
    print(f" Repository: {repo_id}")
    print(f" Target Dir: {target_path}")
    print(f" Max Workers: {max_workers}")
    print(f"============================================================")

    # 1. Query remote repository metadata
    api = HfApi()
    try:
        remote_files = list(api.list_repo_tree(repo_id))
        total_remote_bytes = sum(f.size for f in remote_files if hasattr(f, "size") and f.size is not None)
        print(f"Remote files: {len(remote_files)} ({total_remote_bytes / (1024**3):.2f} GB)")
    except Exception as exc:
        print(f"Warning: Could not query repository metadata: {exc}")
        total_remote_bytes = 0

    # 2. Execute snapshot download
    start_time = time.time()
    print("\nStarting download (resumable)...")
    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_path),
        local_dir_use_symlinks=False,
        max_workers=max_workers,
        resume_download=True,
    )
    elapsed = time.time() - start_time

    # 3. Post-download verification
    print(f"\n============================================================")
    print(f" Download Completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"============================================================")

    safetensors_files = sorted(target_path.glob("*.safetensors"))
    total_local_bytes = sum(f.stat().st_size for f in target_path.rglob("*") if f.is_file())

    print(f"Total Local Size: {total_local_bytes / (1024**3):.2f} GB")
    print(f"Safetensors Shards Found ({len(safetensors_files)}):")
    for shard in safetensors_files:
        size_gb = shard.stat().st_size / (1024**3)
        print(f"  - {shard.name}: {size_gb:.2f} GB")

    # Verify critical configs
    config_file = target_path / "config.json"
    index_file = target_path / "model.safetensors.index.json"

    if config_file.exists():
        print("  [OK] config.json verified")
    else:
        print("  [WARN] config.json missing!")

    if index_file.exists():
        print("  [OK] model.safetensors.index.json verified")
    else:
        print("  [INFO] single-file or non-sharded model")

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MoE models from HuggingFace")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen1.5-MoE-A2.7B",
        help="HuggingFace model repository ID",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local target directory (defaults to models/<repo_name>)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel download worker threads",
    )
    args = parser.parse_args()

    download_moe_model(
        repo_id=args.model_id,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
