"""
run_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Convenience script: runs the full pipeline in order.

Usage:
  python run_pipeline.py            # integrate new dataset + run experiments
  python run_pipeline.py --skip-dl  # skip dataset download (CSV already exists)
  python run_pipeline.py --plots    # also regenerate matplotlib plots

Steps:
  1. (Optional) Download + integrate the new Kaggle dataset
  2. Run recommendation_engine experiments (8 seeds, K=10)
  3. Print side-by-side metric table (new vs old)
  4. (Optional) Regenerate plots
"""

import argparse
import json
import os
import sys


def backup_old_results():
    """If output/results.json exists from the old run, move it to old_results.json."""
    src  = "output/results.json"
    dest = "output/old_results.json"
    if os.path.exists(src) and not os.path.exists(dest):
        import shutil
        shutil.copy(src, dest)
        print(f"[backup] Saved existing results -> {dest}")
    elif os.path.exists(dest):
        print(f"[backup] {dest} already exists — skipping backup.")


def main():
    parser = argparse.ArgumentParser(description="DiverseTune pipeline runner")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip dataset download/integration (use existing CSV)")
    parser.add_argument("--plots",   action="store_true",
                        help="Regenerate matplotlib plots after experiments")
    args = parser.parse_args()

    # 1. Backup old results so comparison table can load them
    backup_old_results()

    # 2. Dataset integration
    if not args.skip_dl:
        print("\n=== STEP 1: Integrating new dataset ===")
        import integrate_new_dataset
        integrate_new_dataset.main()
    else:
        print("\n=== STEP 1: Skipped (--skip-dl) ===")
        if not os.path.exists("data/song_track.csv"):
            print("ERROR: data/song_track.csv not found. Run without --skip-dl first.")
            sys.exit(1)

    # 3. Run experiments
    print("\n=== STEP 2: Running recommendation experiments ===")
    from recommendation_engine import load_and_preprocess, run_experiment, \
        load_old_results, print_comparison_table
    from collections import Counter

    tracks = load_and_preprocess("data/song_track.csv")
    print(f"\nDataset: {len(tracks)} tracks")
    genre_counts = Counter(t["genre"] for t in tracks)
    print("Genre distribution:")
    for g, cnt in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  {g:<15}: {cnt}")

    summary = run_experiment(tracks, num_seeds=8, K=10)

    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[OK] Results saved -> output/results.json")

    # 4. Print comparison table
    old_summary = load_old_results("output/old_results.json")
    print_comparison_table(summary, old_summary)

    # 5. Optional plots
    if args.plots:
        print("\n=== STEP 3: Regenerating plots ===")
        import generate_plots
        generate_plots.setup_style()
        generate_plots.plot_diversity(summary)
        generate_plots.plot_fairness(summary)
        generate_plots.plot_popularity(summary)
        generate_plots.plot_tradeoff(summary)
        generate_plots.plot_niche_percentage(summary)
        print("[OK] Plots saved to output/")

    print("\n=== PIPELINE COMPLETE ===\n")


if __name__ == "__main__":
    main()
