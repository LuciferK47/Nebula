"""Visualization utilities for benchmark results.

Generates the 4 key plots from the implementation plan:
1. Cache hit rate comparison (bar chart across baselines)
2. Prefetch precision over time (line chart)
3. Throughput comparison (bar chart)
4. Expert activation heatmap (per-layer frequency)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_matplotlib() -> None:
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


# ── 1. Hit-rate comparison bar chart ──────────────────────────────────


def plot_hit_rate_comparison(
    results: List[Dict[str, Any]],
    output_path: str = "hit_rate_comparison.png",
    title: str = "Cache Hit Rate by Baseline",
) -> str:
    """Bar chart comparing cache hit rates across baselines.

    Parameters
    ----------
    results : list of dicts
        Each dict should have 'baseline', 'hit_rate', 'domain'.
    output_path : str
        Where to save the figure.

    Returns
    -------
    The output path.
    """
    _check_matplotlib()

    domains = sorted(set(r["domain"] for r in results))
    baselines = []
    seen = set()
    for r in results:
        if r["baseline"] not in seen:
            baselines.append(r["baseline"])
            seen.add(r["baseline"])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(baselines))
    width = 0.8 / max(len(domains), 1)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(domains)))

    for i, domain in enumerate(domains):
        hit_rates = []
        for bl in baselines:
            matching = [r for r in results if r["baseline"] == bl and r["domain"] == domain]
            hit_rates.append(matching[0]["hit_rate"] if matching else 0.0)

        bars = ax.bar(x + i * width, hit_rates, width, label=domain, color=colors[i])
        for bar, hr in zip(bars, hit_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{hr:.1%}", ha="center", va="bottom", fontsize=9
            )

    ax.set_xlabel("Baseline")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(domains) - 1) / 2)
    ax.set_xticklabels(baselines, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Domain")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved hit rate plot: {output_path}")
    return output_path


# ── 2. Prefetch precision over time ───────────────────────────────────


def plot_prefetch_precision(
    precision_history: List[float],
    output_path: str = "prefetch_precision.png",
    window: int = 50,
    title: str = "Prefetch Precision Over Time",
) -> str:
    """Line chart of rolling prefetch precision.

    Parameters
    ----------
    precision_history : list[float]
        Per-step precision values (0 or 1 for each prediction).
    window : int
        Rolling average window size.
    """
    _check_matplotlib()

    if not precision_history:
        logger.warning("No precision data to plot")
        return output_path

    arr = np.array(precision_history, dtype=float)
    # Rolling average
    if len(arr) >= window:
        kernel = np.ones(window) / window
        rolling = np.convolve(arr, kernel, mode="valid")
    else:
        rolling = arr

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rolling, color="#2196F3", linewidth=1.5, alpha=0.9)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% target")
    ax.set_xlabel("Prediction Index")
    ax.set_ylabel("Rolling Precision")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved prefetch precision plot: {output_path}")
    return output_path


# ── 3. Throughput comparison bar chart ────────────────────────────────


def plot_throughput_comparison(
    results: List[Dict[str, Any]],
    output_path: str = "throughput_comparison.png",
    title: str = "Throughput by Baseline (tokens/sec)",
) -> str:
    """Bar chart comparing throughput across baselines."""
    _check_matplotlib()

    baselines = []
    seen = set()
    for r in results:
        if r["baseline"] not in seen:
            baselines.append(r["baseline"])
            seen.add(r["baseline"])

    # Average throughput across domains
    throughputs = []
    for bl in baselines:
        matching = [r["tokens_per_second"] for r in results if r["baseline"] == bl]
        throughputs.append(np.mean(matching) if matching else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    bars = ax.bar(baselines, throughputs, color=colors[: len(baselines)])

    for bar, tp in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{tp:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_xlabel("Baseline")
    ax.set_ylabel("Tokens / Second")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved throughput plot: {output_path}")
    return output_path


# ── 4. Expert activation heatmap ──────────────────────────────────────


def plot_expert_heatmap(
    activation_matrix: Any,
    output_path: str = "expert_heatmap.png",
    title: str = "Expert Activation Frequency",
    num_experts_shown: int = 30,
) -> str:
    """Heatmap of expert activation frequency per layer.

    Parameters
    ----------
    activation_matrix : np.ndarray or RoutingTrace
        Shape: (num_layers, num_experts) or RoutingTrace object.
    num_experts_shown : int
        Only show the top-N most active experts (for readability).
    """
    _check_matplotlib()

    if hasattr(activation_matrix, "decisions"):
        trace = activation_matrix
        max_exp = 0
        for d in trace.decisions:
            if d.top_k_expert_ids:
                max_exp = max(max_exp, max(d.top_k_expert_ids))
        actual_num_experts = max_exp + 1
        mat = np.zeros((trace.num_layers, actual_num_experts), dtype=int)
        for d in trace.decisions:
            for eid in d.top_k_expert_ids:
                mat[d.layer_idx, eid] += 1
        activation_matrix = mat
        num_experts_shown = min(num_experts_shown, actual_num_experts)
    else:
        num_experts_shown = min(num_experts_shown, activation_matrix.shape[1])

    # Select top experts by total activation
    total_per_expert = activation_matrix.sum(axis=0)
    top_indices = np.argsort(total_per_expert)[-num_experts_shown:][::-1]
    subset = activation_matrix[:, top_indices]

    fig_height = 5 if num_experts_shown <= 8 else 8
    fig, ax = plt.subplots(figsize=(12, fig_height))
    im = ax.imshow(
        subset.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_xlabel("Layer Index", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Expert Index (top {num_experts_shown})", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([str(i) for i in top_indices], fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Activation Count", shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved expert heatmap: {output_path}")
    return output_path


# ── Combined report ───────────────────────────────────────────────────


def generate_all_plots(
    results: List[Dict[str, Any]],
    precision_history: Optional[List[float]] = None,
    activation_matrix: Optional[np.ndarray] = None,
    output_dir: str = ".",
) -> List[str]:
    """Generate all visualization plots.

    Returns list of output file paths.
    """
    _check_matplotlib()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(plot_hit_rate_comparison(
        results, str(out / "hit_rate_comparison.png")
    ))
    paths.append(plot_throughput_comparison(
        results, str(out / "throughput_comparison.png")
    ))

    if precision_history:
        paths.append(plot_prefetch_precision(
            precision_history, str(out / "prefetch_precision.png")
        ))

    if activation_matrix is not None:
        paths.append(plot_expert_heatmap(
            activation_matrix, str(out / "expert_heatmap.png")
        ))

    return paths
