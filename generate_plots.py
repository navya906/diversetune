"""
Generate matplotlib plots from output/results.json (written by run_experiment).

Five series are plotted in every comparison chart:
  Greedy, Content Filtering, MMR (lambda=0.7), MMR + niche floor, Graph DPP Rerank.
Bars show the mean over the 30 runs, error bars one standard deviation, and the
white diamond the median (ILD and niche% are bimodal, so mean and median differ).
The other two MMR lambdas (0.5, 0.9) appear only in mmr_lambda_curve.png to keep
the main charts readable.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (results.json key, axis label, colour). Keys match recommendation_engine.STRATEGIES.
SERIES = [
    ("greedy",            "Greedy\n(Popularity)",  "#ff6b6b"),
    ("content_filtering", "Content\nFiltering",    "#4ecdc4"),
    ("mmr_0.7",           "MMR\n(λ=0.7)",          "#f59e0b"),
    ("mmr_floor",         "MMR + niche\nfloor",     "#22c55e"),
    ("graph_dpp_rerank",  "Graph DPP\nRerank",     "#a855f7"),
]
ALGO_KEYS   = [k for k, _, _ in SERIES]
ALGO_NAMES  = [n for _, n, _ in SERIES]
BAR_COLORS  = [c for _, _, c in SERIES]
TEXT = "#e0e0ff"


def load_results(path="output/results.json"):
    with open(path, "r") as f:
        return json.load(f)


def setup_style():
    """Apply a clean, modern style to all plots."""
    plt.rcParams.update({
        "figure.facecolor": "#0f0f23",
        "axes.facecolor": "#1a1a3e",
        "axes.edgecolor": "#3d3d7a",
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": "#b0b0dd",
        "ytick.color": "#b0b0dd",
        "grid.color": "#2a2a5a",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def _bar_chart(results, metric, ylabel, title, filename, fmt, output_dir, ylim=None):
    """Mean bars with ±1 std error bars and a median marker."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    means = np.array([results[k][metric] for k in ALGO_KEYS])
    stds = np.array([results[k].get(metric + "_std", 0.0) for k in ALGO_KEYS])
    meds = np.array([results[k].get(metric + "_median", results[k][metric]) for k in ALGO_KEYS])
    x = np.arange(len(ALGO_KEYS))

    ax.bar(x, means, color=BAR_COLORS, width=0.6, edgecolor="#ffffff22", linewidth=1.2, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#ffffffcc", capsize=5, linewidth=1.4, zorder=3)
    ax.scatter(x, meds, marker="D", s=55, color="white", edgecolor="#0f0f23", zorder=4, label="median")

    top = (means + stds).max()
    for xi, m in zip(x, means):
        ax.text(xi, (m + stds[xi]) + top * 0.02, fmt.format(m), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT)

    n = results[ALGO_KEYS[0]].get("n_runs", "?")
    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_NAMES)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}\n(mean ± 1 std, n={n} runs)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(*(ylim if ylim else (0, top * 1.18)))
    ax.grid(axis="y", linestyle="--", zorder=0)
    ax.legend(loc="upper right", framealpha=0.3, facecolor="#1a1a3e", edgecolor="#3d3d7a")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_diversity(results, output_dir="output"):
    _bar_chart(results, "ild", "Intra-List Diversity (ILD)", "Diversity Comparison",
               "diversity_comparison.png", "{:.3f}", output_dir)


def plot_fairness(results, output_dir="output"):
    _bar_chart(results, "gini", "Gini Index of list popularity",
               "Popularity Concentration (Lower = More Equal)",
               "fairness_comparison.png", "{:.3f}", output_dir)


def plot_popularity(results, output_dir="output"):
    _bar_chart(results, "avg_popularity", "Average Popularity Score",
               "Average Popularity of Recommendations",
               "popularity_comparison.png", "{:.1f}", output_dir, ylim=(0, 110))


def plot_niche_percentage(results, output_dir="output"):
    _bar_chart(results, "niche_pct", "Niche songs (popularity < 40), %",
               "Niche Song Representation", "niche_percentage.png", "{:.1f}%", output_dir,
               ylim=(0, 75))


def plot_tradeoff(results, output_dir="output"):
    """ILD (median) vs average popularity per method, one point per series."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for key, name, color in SERIES:
        r = results[key]
        ax.scatter(r["avg_popularity"], r["ild"], s=170, color=color, edgecolor="white",
                   linewidth=1.2, zorder=3)
        ax.annotate(name.replace("\n", " "), (r["avg_popularity"], r["ild"]),
                    textcoords="offset points", xytext=(9, 7), fontsize=10, color=TEXT)
    ax.set_xlabel("Average popularity of recommendations", fontsize=12)
    ax.set_ylabel("Mean Intra-List Diversity (ILD)", fontsize=12)
    ax.set_title("Diversity vs Popularity Trade-off (per-method means)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlim(40, 100)
    ax.set_ylim(-0.01, max(results[k]["ild"] for k in ALGO_KEYS) * 1.3)
    ax.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tradeoff_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ tradeoff_chart.png")


def plot_mmr_lambda_curve(results, output_dir="output"):
    """MMR relevance/diversity trade-off across lambda, with CF and DPP for reference."""
    lam_keys = sorted((k for k in results if k.startswith("mmr_") and k != "mmr_floor"),
                      key=lambda k: float(k.split("_")[1]))
    if not lam_keys:
        return
    lams = [float(k.split("_")[1]) for k in lam_keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, label in ((axes[0], "ild", "Mean ILD"), (axes[1], "niche_pct", "Mean niche %")):
        vals = [results[k][metric] for k in lam_keys]
        sds = [results[k][metric + "_std"] for k in lam_keys]
        ax.errorbar(lams, vals, yerr=sds, fmt="o-", color="#f59e0b", linewidth=2.2, markersize=9,
                    capsize=5, label="MMR (no floor)")
        for key, name, color in SERIES:
            if key in ("content_filtering", "graph_dpp_rerank", "mmr_floor"):
                ax.axhline(results[key][metric], color=color, linestyle="--", linewidth=1.6,
                           label=name.replace("\n", " ") + (" (λ=0.7)" if key == "mmr_floor" else ""))
        ax.set_xticks(lams)
        ax.set_xlabel("λ (1 = pure relevance)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(linestyle="--")
    axes[0].legend(framealpha=0.3, facecolor="#1a1a3e", edgecolor="#3d3d7a", fontsize=9)
    fig.suptitle("MMR λ sweep: relevance/diversity trade-off (error bars = ±1 std)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mmr_lambda_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ mmr_lambda_curve.png")


def generate_all(results, output_dir="output"):
    setup_style()
    os.makedirs(output_dir, exist_ok=True)
    plot_diversity(results, output_dir)
    plot_fairness(results, output_dir)
    plot_popularity(results, output_dir)
    plot_tradeoff(results, output_dir)
    plot_niche_percentage(results, output_dir)
    plot_mmr_lambda_curve(results, output_dir)


def main():
    print("Generating plots...")
    generate_all(load_results())
    print("\n✅ All plots saved to output/")


if __name__ == "__main__":
    main()
