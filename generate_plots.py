"""
Generate matplotlib visualisation plots from experiment results.
Saves PNG images to output/ directory for the web frontend.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path="output/results.json"):
    with open(path, "r") as f:
        return json.load(f)


def setup_style():
    """Apply a clean, modern style to all plots."""
    plt.rcParams.update({
        "figure.facecolor": "#0f0f23",
        "axes.facecolor": "#1a1a3e",
        "axes.edgecolor": "#3d3d7a",
        "axes.labelcolor": "#e0e0ff",
        "text.color": "#e0e0ff",
        "xtick.color": "#b0b0dd",
        "ytick.color": "#b0b0dd",
        "grid.color": "#2a2a5a",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 12,
    })


ALGO_NAMES = ["Greedy\n(Popularity)", "Similarity\n(CF)", "Hybrid\n(DPP)"]
ALGO_KEYS = ["greedy", "similarity", "hybrid"]
BAR_COLORS = ["#ff6b6b", "#4ecdc4", "#a855f7"]


def plot_diversity(results, output_dir="output"):
    """Bar chart – ILD Diversity Comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [results[k]["ild"] for k in ALGO_KEYS]
    bars = ax.bar(ALGO_NAMES, values, color=BAR_COLORS, width=0.5, edgecolor="#ffffff22", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#e0e0ff")

    ax.set_ylabel("Intra-List Diversity (ILD)", fontsize=13)
    ax.set_title("Diversity Comparison", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diversity_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ diversity_comparison.png")


def plot_fairness(results, output_dir="output"):
    """Bar chart – Gini Index (Fairness) Comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [results[k]["gini"] for k in ALGO_KEYS]
    bars = ax.bar(ALGO_NAMES, values, color=BAR_COLORS, width=0.5, edgecolor="#ffffff22", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#e0e0ff")

    ax.set_ylabel("Gini Index", fontsize=13)
    ax.set_title("Fairness Comparison (Lower = Better)", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, max(values) * 1.45)
    ax.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ fairness_comparison.png")


def plot_popularity(results, output_dir="output"):
    """Bar chart – Average Popularity."""
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [results[k]["avg_popularity"] for k in ALGO_KEYS]
    bars = ax.bar(ALGO_NAMES, values, color=BAR_COLORS, width=0.5, edgecolor="#ffffff22", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#e0e0ff")

    ax.set_ylabel("Average Popularity Score", fontsize=13)
    ax.set_title("Average Popularity of Recommendations", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "popularity_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ popularity_comparison.png")


def plot_tradeoff(results, output_dir="output"):
    """Line graph – ILD vs Popularity trade-off."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(ALGO_NAMES))
    ild_vals = [results[k]["ild"] for k in ALGO_KEYS]
    pop_vals = [results[k]["avg_popularity"] for k in ALGO_KEYS]

    # ILD line
    line1 = ax1.plot(x, ild_vals, "o-", color="#a855f7", linewidth=2.5, markersize=10,
                     label="ILD (Diversity)", zorder=5)
    ax1.set_ylabel("Intra-List Diversity (ILD)", fontsize=13, color="#a855f7")
    ax1.tick_params(axis="y", labelcolor="#a855f7")
    ax1.set_ylim(0, max(ild_vals) * 1.4)

    # Popularity line (secondary axis)
    ax2 = ax1.twinx()
    line2 = ax2.plot(x, pop_vals, "s--", color="#ff6b6b", linewidth=2.5, markersize=10,
                     label="Avg Popularity", zorder=5)
    ax2.set_ylabel("Average Popularity", fontsize=13, color="#ff6b6b")
    ax2.tick_params(axis="y", labelcolor="#ff6b6b")
    ax2.set_ylim(0, 100)

    ax1.set_xticks(x)
    ax1.set_xticklabels(ALGO_NAMES)
    ax1.set_title("Diversity vs Popularity Trade-off", fontsize=16, fontweight="bold", pad=15)
    ax1.grid(axis="both", linestyle="--")

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", framealpha=0.3, facecolor="#1a1a3e", edgecolor="#3d3d7a")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tradeoff_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ tradeoff_chart.png")


def plot_niche_percentage(results, output_dir="output"):
    """Bar chart – Percentage of Niche Songs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [results[k]["niche_pct"] for k in ALGO_KEYS]
    bars = ax.bar(ALGO_NAMES, values, color=BAR_COLORS, width=0.5, edgecolor="#ffffff22", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#e0e0ff")

    ax.set_ylabel("Niche Songs (popularity < 40) %", fontsize=13)
    ax.set_title("Niche Song Representation", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, max(max(values) * 1.3, 10))
    ax.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "niche_percentage.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ niche_percentage.png")


def main():
    setup_style()
    results = load_results()
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating plots...")
    plot_diversity(results, output_dir)
    plot_fairness(results, output_dir)
    plot_popularity(results, output_dir)
    plot_tradeoff(results, output_dir)
    plot_niche_percentage(results, output_dir)
    print("\n✅ All plots saved to output/")


if __name__ == "__main__":
    main()
