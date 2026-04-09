"""
Recommendation Engine: Implements Greedy, Similarity-Based, and Hybrid (DPP) algorithms.
Also computes all evaluation metrics (ILD, Gini, Popularity).
"""

import csv
import json
import math
import os
import random
import numpy as np
from collections import Counter

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

FEATURE_COLS = ["danceability", "energy", "valence", "tempo", "acousticness"]
ALL_COLS = FEATURE_COLS + ["popularity"]


def load_and_preprocess(csv_path="data/spotify_tracks.csv"):
    """Load CSV, remove nulls, normalize features using MinMaxScaler."""
    raw = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with any null/empty feature values
            if any(row[c].strip() == "" for c in ALL_COLS):
                continue
            for c in ALL_COLS:
                row[c] = float(row[c])
            raw.append(row)

    print(f"Loaded {len(raw)} tracks (after removing nulls)")

    # MinMaxScaler normalisation
    mins = {c: min(r[c] for r in raw) for c in ALL_COLS}
    maxs = {c: max(r[c] for r in raw) for c in ALL_COLS}

    for row in raw:
        for c in ALL_COLS:
            rng = maxs[c] - mins[c]
            row[f"{c}_norm"] = (row[c] - mins[c]) / rng if rng > 0 else 0.0

    # Build feature vectors (normalised features only, not popularity)
    for row in raw:
        row["feature_vec"] = np.array([row[f"{c}_norm"] for c in FEATURE_COLS])

    return raw


# ─────────────────────────────────────────────
# 2. SIMILARITY COMPUTATION
# ─────────────────────────────────────────────

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def pairwise_cosine_matrix(vecs):
    """Compute pairwise cosine similarity matrix for a list of vectors."""
    n = len(vecs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = cosine_similarity(vecs[i], vecs[j])
            mat[i][j] = s
            mat[j][i] = s
    return mat


# ─────────────────────────────────────────────
# 3.(A) GREEDY ALGORITHM — Popularity-based
# ─────────────────────────────────────────────

def greedy_recommend(tracks, K=10):
    """Rank songs by popularity, return top K."""
    sorted_tracks = sorted(tracks, key=lambda t: t["popularity"], reverse=True)
    return sorted_tracks[:K]


# ─────────────────────────────────────────────
# 3.(B) SIMILARITY-BASED — Content Filtering
# ─────────────────────────────────────────────

def content_filtering_recommend(tracks, liked_indices, K=10):
    """
    For each candidate (not in liked), compute average cosine similarity
    to the liked songs. Return top K most similar.
    """
    liked_set = set(liked_indices)
    liked_vecs = [tracks[i]["feature_vec"] for i in liked_indices]

    scores = []
    for idx, track in enumerate(tracks):
        if idx in liked_set:
            continue
        avg_sim = np.mean([cosine_similarity(track["feature_vec"], lv) for lv in liked_vecs])
        scores.append((idx, avg_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [tracks[i] for i, _ in scores[:K]]


# ─────────────────────────────────────────────
# 3.(C) HYBRID — Similarity + DPP Re‑ranking
# ─────────────────────────────────────────────

def dpp_rerank(candidates, K=10, lambda_div=0.5):
    """
    Determinantal Point Process–based greedy re‑ranking.
    Maximises quality (relevance) while penalising redundancy (similarity).

    L_ij = q_i * q_j * S_ij   (kernel matrix)
    Greedy: iteratively pick item maximising log‑det(L_selected ∪ {item}).
    """
    n = len(candidates)
    if n <= K:
        return candidates

    vecs = [c["feature_vec"] for c in candidates]
    # Relevance scores (normalised similarity already stored)
    qualities = np.array([c.get("_relevance", 0.5) for c in candidates])
    # Ensure positive
    qualities = np.clip(qualities, 0.01, 1.0)

    # Build kernel matrix L
    sim_mat = pairwise_cosine_matrix(vecs)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            L[i][j] = qualities[i] * qualities[j] * sim_mat[i][j]

    # Greedy DPP selection
    selected = []
    remaining = list(range(n))

    for _ in range(K):
        best_idx = None
        best_gain = -float("inf")
        for idx in remaining:
            trial = selected + [idx]
            sub_L = L[np.ix_(trial, trial)]
            # Add small regularisation for numerical stability
            sign, logdet = np.linalg.slogdet(sub_L + 1e-8 * np.eye(len(trial)))
            gain = logdet if sign > 0 else -float("inf")
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def graph_dpp_rerank_recommend(tracks, liked_indices, K=10, N=50, min_niche_pct=0.20):
    """
    Step 1: Get top N candidates via similarity.
    Step 2: DPP re-rank to top K.
    Step 3: Enforce fairness constraint (≥ 20% niche songs, popularity < 40).
    """
    liked_set = set(liked_indices)
    liked_vecs = [tracks[i]["feature_vec"] for i in liked_indices]

    # Step 1: Similarity-based candidate generation
    scores = []
    for idx, track in enumerate(tracks):
        if idx in liked_set:
            continue
        avg_sim = np.mean([cosine_similarity(track["feature_vec"], lv) for lv in liked_vecs])
        scores.append((idx, avg_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = []
    for i, sim in scores[:N]:
        t = dict(tracks[i])
        t["feature_vec"] = tracks[i]["feature_vec"]
        t["_relevance"] = sim
        t["_orig_idx"] = i
        candidates.append(t)

    # Step 2: DPP re-ranking
    reranked = dpp_rerank(candidates, K=K)

    # Step 3: Fairness enforcement — ensure ≥ min_niche_pct niche songs
    niche_count = sum(1 for t in reranked if t["popularity"] < 40)
    required_niche = max(1, int(K * min_niche_pct))

    if niche_count < required_niche:
        needed = required_niche - niche_count
        # Find niche candidates not already selected
        selected_ids = {t["track_id"] for t in reranked}
        niche_pool = [c for c in candidates if c["popularity"] < 40 and c["track_id"] not in selected_ids]
        niche_pool.sort(key=lambda x: x.get("_relevance", 0), reverse=True)

        # Replace least-relevant popular songs with niche songs
        popular_in_list = [i for i, t in enumerate(reranked) if t["popularity"] >= 40]
        popular_in_list.sort(key=lambda i: reranked[i].get("_relevance", 0))

        for j in range(min(needed, len(niche_pool), len(popular_in_list))):
            reranked[popular_in_list[j]] = niche_pool[j]

    return reranked


# ─────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────

def intra_list_diversity(recs):
    """
    ILD = (1 / |L|(|L|-1)) * Σ_{i≠j} (1 - sim(i,j))
    Higher = more diverse.
    """
    n = len(recs)
    if n < 2:
        return 0.0
    vecs = [r["feature_vec"] for r in recs]
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - cosine_similarity(vecs[i], vecs[j])
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def popularity_concentration_index(values):
    """Compute Gini index for a list of values. Lower = more equal."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def popularity_metrics(recs):
    """Compute average popularity and percentage of niche songs (<40)."""
    pops = [r["popularity"] for r in recs]
    avg = np.mean(pops)
    niche_pct = sum(1 for p in pops if p < 40) / len(pops) * 100
    return float(avg), float(niche_pct)


# ─────────────────────────────────────────────
# 5. EXPERIMENT RUNNER
# ─────────────────────────────────────────────

def run_experiment(tracks, num_seeds=8, K=10):
    """
    Run all 3 algorithms across multiple seed sets.
    Return averaged metrics + per-run details.
    """
    results = {
        "greedy": {"ild": [], "gini": [], "avg_pop": [], "niche_pct": [], "runs": []},
        "similarity": {"ild": [], "gini": [], "avg_pop": [], "niche_pct": [], "runs": []},
        "hybrid": {"ild": [], "gini": [], "avg_pop": [], "niche_pct": [], "runs": []},
    }

    # Generate seed sets (5-10 random liked songs each)
    all_indices = list(range(len(tracks)))
    seed_sets = []
    for _ in range(num_seeds):
        size = random.randint(5, 10)
        seed_sets.append(random.sample(all_indices, size))

    for run_idx, liked in enumerate(seed_sets):
        seed_names = [tracks[i]["track_name"] for i in liked]
        print(f"\n── Run {run_idx + 1}/{num_seeds} ──")
        print(f"  Seed songs: {seed_names[:3]}{'...' if len(seed_names) > 3 else ''}")

        # (A) Greedy
        greedy_recs = greedy_recommend(tracks, K)
        g_ild = intra_list_diversity(greedy_recs)
        g_gini = popularity_concentration_index([r["popularity"] for r in greedy_recs])
        g_avg, g_niche = popularity_metrics(greedy_recs)
        results["greedy"]["ild"].append(g_ild)
        results["greedy"]["gini"].append(g_gini)
        results["greedy"]["avg_pop"].append(g_avg)
        results["greedy"]["niche_pct"].append(g_niche)
        results["greedy"]["runs"].append({
            "seed_songs": seed_names,
            "recommendations": [{"name": r["track_name"], "artist": r["artists"],
                                 "genre": r["track_genre"], "popularity": r["popularity"]}
                                for r in greedy_recs]
        })

        # (B) Similarity
        sim_recs = content_filtering_recommend(tracks, liked, K)
        s_ild = intra_list_diversity(sim_recs)
        s_gini = popularity_concentration_index([r["popularity"] for r in sim_recs])
        s_avg, s_niche = popularity_metrics(sim_recs)
        results["similarity"]["ild"].append(s_ild)
        results["similarity"]["gini"].append(s_gini)
        results["similarity"]["avg_pop"].append(s_avg)
        results["similarity"]["niche_pct"].append(s_niche)
        results["similarity"]["runs"].append({
            "seed_songs": seed_names,
            "recommendations": [{"name": r["track_name"], "artist": r["artists"],
                                 "genre": r["track_genre"], "popularity": r["popularity"]}
                                for r in sim_recs]
        })

        # (C) Hybrid (DPP)
        hyb_recs = graph_dpp_rerank_recommend(tracks, liked, K)
        h_ild = intra_list_diversity(hyb_recs)
        h_gini = popularity_concentration_index([r["popularity"] for r in hyb_recs])
        h_avg, h_niche = popularity_metrics(hyb_recs)
        results["hybrid"]["ild"].append(h_ild)
        results["hybrid"]["gini"].append(h_gini)
        results["hybrid"]["avg_pop"].append(h_avg)
        results["hybrid"]["niche_pct"].append(h_niche)
        results["hybrid"]["runs"].append({
            "seed_songs": seed_names,
            "recommendations": [{"name": r["track_name"], "artist": r["artists"],
                                 "genre": r["track_genre"], "popularity": r["popularity"]}
                                for r in hyb_recs]
        })

    # Compute averages
    summary = {}
    for algo in ["greedy", "similarity", "hybrid"]:
        summary[algo] = {
            "ild": round(float(np.mean(results[algo]["ild"])), 4),
            "gini": round(float(np.mean(results[algo]["gini"])), 4),
            "avg_popularity": round(float(np.mean(results[algo]["avg_pop"])), 2),
            "niche_pct": round(float(np.mean(results[algo]["niche_pct"])), 2),
            "runs": results[algo]["runs"],
            "per_run_ild": [round(v, 4) for v in results[algo]["ild"]],
            "per_run_gini": [round(v, 4) for v in results[algo]["gini"]],
            "per_run_avg_pop": [round(v, 2) for v in results[algo]["avg_pop"]],
            "per_run_niche_pct": [round(v, 2) for v in results[algo]["niche_pct"]],
        }

    return summary


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    # Load
    tracks = load_and_preprocess("data/spotify_tracks.csv")

    # Run experiment
    summary = run_experiment(tracks, num_seeds=8, K=10)

    # Print results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS (averaged over all runs)")
    print("=" * 60)

    algo_labels = {"greedy": "Greedy (Popularity)", "similarity": "Similarity (CF)", "hybrid": "Hybrid (DPP)"}
    for algo, label in algo_labels.items():
        s = summary[algo]
        print(f"\nAlgorithm: {label}")
        print(f"  ILD (Diversity):     {s['ild']:.4f}")
        print(f"  Gini Index:          {s['gini']:.4f}")
        print(f"  Avg Popularity:      {s['avg_popularity']:.2f}")
        print(f"  Niche Songs (%):     {s['niche_pct']:.2f}%")

    # Determine best
    best_div = max(summary, key=lambda a: summary[a]["ild"])
    best_fair = min(summary, key=lambda a: summary[a]["gini"])
    print(f"\n{'─' * 60}")
    print(f"🏆 Highest Diversity (ILD):  {algo_labels[best_div]}  ({summary[best_div]['ild']:.4f})")
    print(f"🏆 Best Fairness (Gini):     {algo_labels[best_fair]} ({summary[best_fair]['gini']:.4f})")
    print(f"{'─' * 60}")

    # Save JSON for web frontend
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n✅ Results saved → output/results.json")


if __name__ == "__main__":
    main()
