"""
recommendation_engine.py
─────────────────────────────────────────────────────────────────────────────
Implements Greedy, Content Filtering (Similarity), and Hybrid (DPP)
recommendation algorithms on the new Spotify dataset.

Feature vector (NO audio features):
  - One-hot encoded genre         (11 dimensions)
  - Normalised duration_ms        (1 dimension, pre-computed in CSV)
  - Explicit flag (binary)        (1 dimension)
  - Artist co-occurrence (Jaccard): encoded as a normalised count column
    built at load time from the full artist list.

Popularity is kept OUT of the similarity vector; used only for metrics.
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

# Canonical genre list (11 genres after mapping)
GENRE_LIST = [
    "Pop", "Hip-Hop", "Rock", "Electronic",
    "Classical", "Jazz", "Country", "Latin",
    "Metal", "Indie", "Reggae",
]


def _artist_jaccard_features(tracks):
    """
    Build a per-track artist frequency score.
    We represent each artist as an index in a shared vocabulary, then
    for each track compute a unit L2-normalised vector over artist dims.
    Because tracks typically have 1-2 artists the vectors are very sparse;
    we collapse to a scalar 'artist_rarity_norm' = 1 - (freq/max_freq)
    so that rare artists score high (niche).  Used as a single feature dim.
    """
    # Count how often each artist string appears across the dataset
    artist_counts = Counter()
    for t in tracks:
        for a in t["_artist_list"]:
            artist_counts[a] += 1
    max_count = max(artist_counts.values()) if artist_counts else 1

    for t in tracks:
        # Average normalised frequency of the track's artists
        freqs = [artist_counts[a] / max_count for a in t["_artist_list"]]
        t["artist_pop_norm"] = float(np.mean(freqs)) if freqs else 0.5

    return tracks


def load_and_preprocess(csv_path="data/song_track.csv"):
    """
    Load CSV, build feature vectors from:
      - One-hot genre (11 dims)
      - duration_norm (1 dim, already in CSV)
      - explicit binary (1 dim)
      - artist_pop_norm (1 dim — normalised artist frequency, captures rarity)
    Popularity is loaded but NOT included in the feature vector.
    """
    raw = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Required columns
            if not row.get("genre") or not row.get("name"):
                continue
            try:
                row["popularity"] = float(row["popularity"])
            except (ValueError, TypeError):
                continue

            # Parse numeric fields
            try:
                row["duration_ms"] = float(row.get("duration_ms", 180000) or 180000)
            except (ValueError, TypeError):
                row["duration_ms"] = 180000

            try:
                row["duration_norm"] = float(row.get("duration_norm", 0.5) or 0.5)
            except (ValueError, TypeError):
                row["duration_norm"] = 0.5

            try:
                row["explicit"] = int(float(row.get("explicit", 0) or 0))
            except (ValueError, TypeError):
                row["explicit"] = 0

            # Build artist list (handle comma-separated strings like "['Artist A', 'Artist B']")
            artists_raw = str(row.get("artists", "Unknown"))
            # Strip list formatting if present
            artists_raw = artists_raw.strip("[]").replace("'", "").replace('"', "")
            row["_artist_list"] = [a.strip() for a in artists_raw.split(",") if a.strip()]
            if not row["_artist_list"]:
                row["_artist_list"] = ["Unknown"]

            raw.append(row)

    print(f"Loaded {len(raw)} tracks (after removing nulls)")

    # Build artist rarity feature
    raw = _artist_jaccard_features(raw)

    # One-hot encode genre
    genre_to_idx = {g: i for i, g in enumerate(GENRE_LIST)}

    for row in raw:
        genre_vec = np.zeros(len(GENRE_LIST))
        g_idx = genre_to_idx.get(row["genre"])
        if g_idx is not None:
            genre_vec[g_idx] = 1.0

        row["feature_vec"] = np.concatenate([
            genre_vec,                                  # 11 dims
            [row["duration_norm"]],                     # 1 dim
            [float(row["explicit"])],                   # 1 dim
            [row["artist_pop_norm"]],                   # 1 dim  ← artist freq
        ])  # total: 14 dims

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
    liked_set  = set(liked_indices)
    liked_vecs = [tracks[i]["feature_vec"] for i in liked_indices]

    scores = []
    for idx, track in enumerate(tracks):
        if idx in liked_set:
            continue
        avg_sim = float(np.mean([cosine_similarity(track["feature_vec"], lv) for lv in liked_vecs]))
        scores.append((idx, avg_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [tracks[i] for i, _ in scores[:K]]


# ─────────────────────────────────────────────
# 3.(C) HYBRID — Similarity + DPP Re-ranking
# ─────────────────────────────────────────────

def dpp_rerank(candidates, K=10, lambda_div=0.5):
    """
    Determinantal Point Process-based greedy re-ranking.
    Maximises quality (relevance) while penalising redundancy.
    L_ij = q_i * q_j * S_ij   (kernel matrix)
    Greedy: iteratively pick item maximising log-det(L_selected ∪ {item}).
    """
    n = len(candidates)
    if n <= K:
        return candidates

    vecs      = [c["feature_vec"] for c in candidates]
    qualities = np.array([c.get("_relevance", 0.5) for c in candidates])
    qualities = np.clip(qualities, 0.01, 1.0)

    sim_mat = pairwise_cosine_matrix(vecs)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            L[i][j] = qualities[i] * qualities[j] * sim_mat[i][j]

    selected  = []
    remaining = list(range(n))

    for _ in range(K):
        best_idx  = None
        best_gain = -float("inf")
        for idx in remaining:
            trial  = selected + [idx]
            sub_L  = L[np.ix_(trial, trial)]
            sign, logdet = np.linalg.slogdet(sub_L + 1e-8 * np.eye(len(trial)))
            gain   = logdet if sign > 0 else -float("inf")
            if gain > best_gain:
                best_gain = gain
                best_idx  = idx
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def graph_dpp_rerank_recommend(tracks, liked_indices, K=10, N=50, min_niche_pct=0.20):
    """
    Step 1: Get top N candidates via content similarity.
    Step 2: DPP re-rank to top K.
    Step 3: Enforce fairness constraint (>=20% niche songs, popularity < 40).
    """
    liked_set  = set(liked_indices)
    liked_vecs = [tracks[i]["feature_vec"] for i in liked_indices]

    scores = []
    for idx, track in enumerate(tracks):
        if idx in liked_set:
            continue
        avg_sim = float(np.mean([cosine_similarity(track["feature_vec"], lv) for lv in liked_vecs]))
        scores.append((idx, avg_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = []
    for i, sim in scores[:N]:
        t = dict(tracks[i])
        t["feature_vec"]  = tracks[i]["feature_vec"]
        t["_relevance"]   = sim
        t["_orig_idx"]    = i
        candidates.append(t)

    reranked = dpp_rerank(candidates, K=K)

    # Fairness: ensure >= min_niche_pct niche songs
    niche_count    = sum(1 for t in reranked if t["popularity"] < 40)
    required_niche = max(1, int(K * min_niche_pct))

    if niche_count < required_niche:
        needed       = required_niche - niche_count
        selected_ids = {t["track_id"] for t in reranked}
        niche_pool   = [
            c for c in candidates
            if c["popularity"] < 40 and c["track_id"] not in selected_ids
        ]
        niche_pool.sort(key=lambda x: x.get("_relevance", 0), reverse=True)

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
    ILD = (1 / |L|(|L|-1)) * sum_{i!=j} (1 - sim(i,j))
    Higher = more diverse.
    """
    n = len(recs)
    if n < 2:
        return 0.0
    vecs   = [r["feature_vec"] for r in recs]
    total  = 0.0
    pairs  = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - cosine_similarity(vecs[i], vecs[j])
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def popularity_concentration_index(values):
    """Compute Gini index. Lower = more equal."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n     = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def popularity_metrics(recs):
    """Compute average popularity and % of niche songs (popularity < 40)."""
    pops     = [r["popularity"] for r in recs]
    avg      = float(np.mean(pops))
    niche_pct = sum(1 for p in pops if p < 40) / len(pops) * 100
    return avg, float(niche_pct)


# ─────────────────────────────────────────────
# 5. EXPERIMENT RUNNER
# ─────────────────────────────────────────────

def run_experiment(tracks, num_seeds=8, K=10):
    """
    Run all 3 algorithms across num_seeds randomised seed sets.
    Returns averaged metrics + per-run details.
    """
    results = {
        "greedy":     {"ild": [], "gini": [], "avg_pop": [], "niche_pct": [], "runs": []},
        "content_filtering": {"ild": [], "gini": [], "avg_pop": [], "niche_pct": [], "runs": []},
        "graph_dpp_rerank":     {"ild": [], "gini": [], "avg_pop": [], "niche_pct": [], "runs": []},
    }

    all_indices = list(range(len(tracks)))
    seed_sets   = []
    for _ in range(num_seeds):
        size = random.randint(5, 10)
        seed_sets.append(random.sample(all_indices, size))

    for run_idx, liked in enumerate(seed_sets):
        seed_names = [tracks[i]["name"] for i in liked]
        print(f"\n-- Run {run_idx + 1}/{num_seeds} --")
        print(f"   Seed songs: {seed_names[:3]}{'...' if len(seed_names) > 3 else ''}")

        # (A) Greedy
        greedy_recs = greedy_recommend(tracks, K)
        g_ild  = intra_list_diversity(greedy_recs)
        g_gini = popularity_concentration_index([r["popularity"] for r in greedy_recs])
        g_avg, g_niche = popularity_metrics(greedy_recs)
        results["greedy"]["ild"].append(g_ild)
        results["greedy"]["gini"].append(g_gini)
        results["greedy"]["avg_pop"].append(g_avg)
        results["greedy"]["niche_pct"].append(g_niche)
        results["greedy"]["runs"].append({
            "seed_songs": seed_names,
            "recommendations": [
                {"name": r["name"], "artist": r["artists"],
                 "genre": r["genre"], "popularity": round(r["popularity"], 1)}
                for r in greedy_recs
            ]
        })

        # (B) Similarity
        sim_recs = content_filtering_recommend(tracks, liked, K)
        s_ild  = intra_list_diversity(sim_recs)
        s_gini = popularity_concentration_index([r["popularity"] for r in sim_recs])
        s_avg, s_niche = popularity_metrics(sim_recs)
        results["content_filtering"]["ild"].append(s_ild)
        results["content_filtering"]["gini"].append(s_gini)
        results["content_filtering"]["avg_pop"].append(s_avg)
        results["content_filtering"]["niche_pct"].append(s_niche)
        results["content_filtering"]["runs"].append({
            "seed_songs": seed_names,
            "recommendations": [
                {"name": r["name"], "artist": r["artists"],
                 "genre": r["genre"], "popularity": round(r["popularity"], 1)}
                for r in sim_recs
            ]
        })

        # (C) Hybrid (DPP)
        hyb_recs = graph_dpp_rerank_recommend(tracks, liked, K)
        h_ild  = intra_list_diversity(hyb_recs)
        h_gini = popularity_concentration_index([r["popularity"] for r in hyb_recs])
        h_avg, h_niche = popularity_metrics(hyb_recs)
        results["graph_dpp_rerank"]["ild"].append(h_ild)
        results["graph_dpp_rerank"]["gini"].append(h_gini)
        results["graph_dpp_rerank"]["avg_pop"].append(h_avg)
        results["graph_dpp_rerank"]["niche_pct"].append(h_niche)
        results["graph_dpp_rerank"]["runs"].append({
            "seed_songs": seed_names,
            "recommendations": [
                {"name": r["name"], "artist": r["artists"],
                 "genre": r["genre"], "popularity": round(r["popularity"], 1)}
                for r in hyb_recs
            ]
        })

    # Compute averages
    summary = {}
    for algo in ["greedy", "content_filtering", "graph_dpp_rerank"]:
        summary[algo] = {
            "ild":           round(float(np.mean(results[algo]["ild"])),     4),
            "gini":          round(float(np.mean(results[algo]["gini"])),    4),
            "avg_popularity":round(float(np.mean(results[algo]["avg_pop"])), 2),
            "niche_pct":     round(float(np.mean(results[algo]["niche_pct"])),2),
            "runs":          results[algo]["runs"],
            "per_run_ild":   [round(v, 4) for v in results[algo]["ild"]],
            "per_run_gini":  [round(v, 4) for v in results[algo]["gini"]],
            "per_run_avg_pop":   [round(v, 2) for v in results[algo]["avg_pop"]],
            "per_run_niche_pct": [round(v, 2) for v in results[algo]["niche_pct"]],
        }

    return summary


# ─────────────────────────────────────────────
# 6. COMPARISON TABLE PRINTER
# ─────────────────────────────────────────────

OLD_RESULTS = {
    # Baseline from the 1,960-track audio-feature dataset (12 genres).
    # Update these values if you have the actual old output/results.json.
    "greedy":     {"ild": None, "gini": None, "avg_popularity": None, "niche_pct": None},
    "content_filtering": {"ild": None, "gini": None, "avg_popularity": None, "niche_pct": None},
    "graph_dpp_rerank":     {"ild": None, "gini": None, "avg_popularity": None, "niche_pct": None},
}


def load_old_results(path="output/old_results.json"):
    """Load previously-saved results for side-by-side comparison."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def print_comparison_table(new_summary, old_summary=None):
    """Print new vs old metric table side by side."""
    algo_labels = {
        "greedy":     "Greedy (Popularity)",
        "content_filtering": "Content Filtering",
        "graph_dpp_rerank":     "Graph DPP Rerank",
    }
    metrics = ["ild", "gini", "avg_popularity", "niche_pct"]
    metric_labels = {
        "ild":           "ILD (Diversity)",
        "gini":          "Gini Index",
        "avg_popularity":"Avg Popularity",
        "niche_pct":     "Niche % (<40)",
    }

    has_old = old_summary is not None

    print("\n" + "=" * 74)
    if has_old:
        print(f"{'METRIC':<22} {'OLD (1,960 tracks)':>20}  {'NEW (3,000 tracks)':>20}  {'DELTA':>8}")
    else:
        print(f"{'METRIC':<22} {'NEW (3,000 tracks)':>20}")
    print("=" * 74)

    for algo, label in algo_labels.items():
        print(f"\n  Algorithm: {label}")
        print("  " + "-" * 70)
        for m in metrics:
            new_val = new_summary[algo].get(m)
            lbl = metric_labels[m]
            if has_old and old_summary.get(algo, {}).get(m) is not None:
                old_val = old_summary[algo][m]
                delta   = (new_val - old_val) if (new_val is not None and old_val is not None) else None
                delta_s = f"{delta:+.4f}" if delta is not None else "   N/A"
                print(f"  {lbl:<22} {old_val:>20.4f}  {new_val:>20.4f}  {delta_s:>8}")
            else:
                val_s = f"{new_val:.4f}" if new_val is not None else "N/A"
                print(f"  {lbl:<22} {val_s:>20}")

    print("\n" + "=" * 74)


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    print("Loading dataset …")
    tracks = load_and_preprocess("data/song_track.csv")

    print(f"\nDataset size: {len(tracks)} tracks")
    genre_counts = Counter(t["genre"] for t in tracks)
    print("Genre distribution:")
    for g, cnt in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  {g:<15}: {cnt}")

    print("\nRunning experiments (8 randomised seed sets, K=10) …")
    summary = run_experiment(tracks, num_seeds=8, K=10)

    # Save new results
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[OK] Results saved -> output/results.json")

    # Load old results for comparison (if they exist)
    old_summary = load_old_results("output/old_results.json")

    print_comparison_table(summary, old_summary)

    algo_labels = {
        "greedy": "Greedy (Popularity)",
        "content_filtering": "Content Filtering",
        "graph_dpp_rerank": "Graph DPP Rerank",
    }
    best_div  = max(summary, key=lambda a: summary[a]["ild"])
    best_fair = min(summary, key=lambda a: summary[a]["gini"])
    print(f"  Highest Diversity (ILD) : {algo_labels[best_div]}  ({summary[best_div]['ild']:.4f})")
    print(f"  Best Fairness (Gini)    : {algo_labels[best_fair]} ({summary[best_fair]['gini']:.4f})")
    print()


if __name__ == "__main__":
    main()
