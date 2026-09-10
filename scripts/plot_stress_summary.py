import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    report_path = "results/stress_test_report.json"
    output_png = "results/stress_test_summary.png"
    if not os.path.exists(report_path):
        print(f"Report {report_path} not found.")
        return

    with open(report_path, "r") as f:
        data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Extreme Memory Pressure
    p1 = data["test_results"]["case_1_extreme_memory_pressure"]["details"]
    budgets = [str(x["budget_mb"]) + " MB" for x in p1]
    tps1 = [x["throughput_tok_s"] for x in p1]
    hrs1 = [x["hit_rate"] * 100 for x in p1]

    x1 = np.arange(len(budgets))
    bars1 = ax1.bar(x1, tps1, width=0.5, color="#2b5c8f", edgecolor="#111", linewidth=1.2, zorder=3)
    ax1.set_ylabel("Throughput (tokens / sec)", fontweight="bold", fontsize=11)
    ax1.set_title("Extreme Memory Budget Stress\n(300MB to 1500MB HBM)", fontweight="bold", fontsize=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(budgets, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for b, tp in zip(bars1, tps1):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{tp:.1f} tok/s", ha="center", fontweight="bold", fontsize=9)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(x1, hrs1, color="#e76f51", marker="o", linewidth=2, markersize=6)
    ax1_twin.set_ylabel("Hit Rate (%)", color="#e76f51", fontweight="bold")
    ax1_twin.set_ylim(0, 115)
    for i, hr in enumerate(hrs1):
        ax1_twin.annotate(f"{hr:.0f}%", (x1[i], hr + 4), ha="center", color="#b23a22", fontweight="bold", fontsize=8)

    # Panel 2: Batched Serving
    p4 = data["test_results"]["case_4_batched_inference"]["details"]
    bsz = [f"Batch {x['batch_size']}" for x in p4]
    tps4 = [x["throughput_tok_s"] for x in p4]
    x4 = np.arange(len(bsz))
    bars4 = ax2.bar(x4, tps4, width=0.5, color="#2a9d8f", edgecolor="#111", linewidth=1.2, zorder=3)
    ax2.set_ylabel("Total Serving Throughput (tok/s)", fontweight="bold", fontsize=11)
    ax2.set_title("Multi-Batch Throughput Scaling\n(Batch 1, 2, 4 Concurrent)", fontweight="bold", fontsize=12)
    ax2.set_xticks(x4)
    ax2.set_xticklabels(bsz, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for b, tp in zip(bars4, tps4):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{tp:.1f} tok/s", ha="center", fontweight="bold", fontsize=9)

    # Panel 3: Domain Routing
    p3 = data["test_results"]["case_3_cross_domain_stress"]["details"]
    domains = ["Tech Systems", "Python Code", "Philosophy", "Math / Logic"]
    hrs3 = [x["hit_rate"] * 100 for x in p3]
    x3 = np.arange(len(domains))
    bars3 = ax3.bar(x3, hrs3, width=0.5, color="#e07a5f", edgecolor="#111", linewidth=1.2, zorder=3)
    ax3.set_ylabel("Cache Hit Rate (%)", fontweight="bold", fontsize=11)
    ax3.set_title("Cross-Domain Routing Resilience\n(Out-of-Distribution Prompts)", fontweight="bold", fontsize=12)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(domains, rotation=15, ha="right", fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.set_ylim(0, 100)

    for b, hr in zip(bars3, hrs3):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{hr:.1f}%", ha="center", fontweight="bold", fontsize=9)

    fig.suptitle("MemTier-MoE Full Operational Stress Testing Suite (RTX 4050 GPU)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    print(f"Saved {output_png}")

if __name__ == "__main__":
    main()
