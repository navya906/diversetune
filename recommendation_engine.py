"""
recommendation_engine.py
─────────────────────────────────────────────────────────────────────────────
Implements Greedy, Content Filtering (Similarity), MMR and Hybrid (DPP)
recommendation algorithms on the new Spotify dataset.

Feature vector (NO audio features):
  - One-hot encoded genre         (10 dimensions)
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
from itertools import combinations
from scipy.stats import wilcoxon

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

# Canonical genre list (10 genres; must match TARGET_GENRES in integrate_new_dataset.py)
GENRE_LIST = [
    "Pop", "Hip-Hop", "Rock", "Electronic", "Classical",
    "Jazz", "Latin", "Ambient", "World", "Indie-Folk",
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
      - One-hot genre (10 dims)
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
            genre_vec,                                  # 10 dims
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

def _track_num(t):
    """Numeric part of 'sp_123' -> 123, used as a deterministic tie-breaker."""
    try:
        return int(str(t["track_id"]).split("_")[-1])
    except ValueError:
        return 0


def greedy_recommend(tracks, K=10, liked_indices=()):
    """
    Rank songs by popularity (desc), return top K, excluding the seed songs.
    Ties are broken deterministically by ascending track_id number.
    NOTE: track_ids are assigned in genre-alphabetical order by the dataset
    builder, so a tie is resolved in favour of the alphabetically earlier genre.
    """
    liked = set(liked_indices)
    pool = [t for i, t in enumerate(tracks) if i not in liked]
    pool.sort(key=lambda t: (-t["popularity"], _track_num(t)))
    return pool[:K]


# ─────────────────────────────────────────────
# 3.(B) SIMILARITY-BASED — Content Filtering
# ─────────────────────────────────────────────

def similarity_scores(tracks, liked_indices):
    """
    Relevance of every non-seed track: mean cosine similarity to the seed songs,
    sorted descending (ties keep dataset order). Shared by Content Filtering,
    DPP and MMR so the comparison isolates the re-ranking strategy.
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
    return scores


def retrieve_candidates(tracks, liked_indices, N=50):
    """Top-N most similar non-seed tracks, each annotated with `_relevance`."""
    candidates = []
    for i, sim in similarity_scores(tracks, liked_indices)[:N]:
        t = dict(tracks[i])
        t["_relevance"] = sim
        t["_orig_idx"]  = i
        candidates.append(t)
    return candidates


def content_filtering_recommend(tracks, liked_indices, K=10):
    """Top K tracks by mean cosine similarity to the liked (seed) songs."""
    return [tracks[i] for i, _ in similarity_scores(tracks, liked_indices)[:K]]


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


# ─────────────────────────────────────────────
# 3.(D) MMR — Maximal Marginal Relevance (Carbonell & Goldstein, 1998)
# ─────────────────────────────────────────────

MMR_LAMBDA = 0.7
MMR_LAMBDAS = (0.5, 0.7, 0.9)


def _mmr_select(cands, K, lam):
    r"""
    Greedy MMR selection from a candidate pool:
        next = argmax_{d_i in R\S} [ lam * Sim(d_i, seed)
                                     - (1 - lam) * max_{d_j in S} Sim(d_i, d_j) ]
    Sim(d_i, seed) is the candidate's `_relevance` (mean cosine to the seed songs),
    Sim(d_i, d_j) is cosine on the same feature vectors. Cost: O(N*K) cosine
    evaluations (max-similarity is updated incrementally).
    """
    if len(cands) <= K:
        return list(cands)
    selected  = []
    remaining = list(range(len(cands)))
    max_sim   = np.zeros(len(cands))
    for _ in range(K):
        best = max(remaining,
                   key=lambda i: lam * cands[i]["_relevance"] - (1 - lam) * max_sim[i])
        selected.append(best)
        remaining.remove(best)
        for i in remaining:
            max_sim[i] = max(max_sim[i],
                             cosine_similarity(cands[i]["feature_vec"], cands[best]["feature_vec"]))
    return [cands[i] for i in selected]


def mmr_recommend(tracks, liked_indices, K=10, N=50, lam=MMR_LAMBDA):
    """Plain MMR over the top-N pool (literature definition: NO popularity floor)."""
    return _mmr_select(retrieve_candidates(tracks, liked_indices, N), K, lam)


def mmr_floor_recommend(tracks, liked_indices, K=10, N=50, lam=MMR_LAMBDA, min_niche_pct=0.20):
    """MMR + the identical niche-floor enforcement used by Hybrid DPP (ablation)."""
    cands = retrieve_candidates(tracks, liked_indices, N)
    return enforce_niche_floor(_mmr_select(cands, K, lam), cands, K, min_niche_pct)


NICHE_THRESHOLD = 40   # popularity below this counts as "niche" (also used by the metrics)


def enforce_niche_floor(reranked, candidates, K, min_niche_pct=0.20):
    """
    Fairness constraint shared by Hybrid DPP and MMR+floor: guarantee at least
    max(1, int(K * min_niche_pct)) niche tracks by swapping the least relevant
    non-niche picks for the most relevant unselected niche candidates.
    Best effort: if the candidate pool has too few niche tracks, the floor is missed.
    """
    reranked = list(reranked)
    niche_count    = sum(1 for t in reranked if t["popularity"] < NICHE_THRESHOLD)
    required_niche = max(1, int(K * min_niche_pct))

    if niche_count < required_niche:
        needed       = required_niche - niche_count
        selected_ids = {t["track_id"] for t in reranked}
        niche_pool   = [
            c for c in candidates
            if c["popularity"] < NICHE_THRESHOLD and c["track_id"] not in selected_ids
        ]
        niche_pool.sort(key=lambda x: x.get("_relevance", 0), reverse=True)

        popular_in_list = [i for i, t in enumerate(reranked) if t["popularity"] >= NICHE_THRESHOLD]
        popular_in_list.sort(key=lambda i: reranked[i].get("_relevance", 0))

        for j in range(min(needed, len(niche_pool), len(popular_in_list))):
            reranked[popular_in_list[j]] = niche_pool[j]
    return reranked


def graph_dpp_rerank_recommend(tracks, liked_indices, K=10, N=50, min_niche_pct=0.20):
    """
    Step 1: Get top N candidates via content similarity.
    Step 2: DPP re-rank to top K.
    Step 3: Enforce fairness constraint (>=20% niche songs, popularity < 40).
    """
    candidates = retrieve_candidates(tracks, liked_indices, N)
    reranked = dpp_rerank(candidates, K=K)
    return enforce_niche_floor(reranked, candidates, K, min_niche_pct)


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
    niche_pct = sum(1 for p in pops if p < NICHE_THRESHOLD) / len(pops) * 100
    return avg, float(niche_pct)


# ─────────────────────────────────────────────
# 5. EXPERIMENT RUNNER
# ─────────────────────────────────────────────

SEED = 42          # master RNG seed for drawing seed-song sets (documented in README)
NUM_RUNS = 30      # number of random seed-song sets per experiment
METRICS = ("ild", "gini", "avg_popularity", "niche_pct")

# Every strategy has the same signature: fn(tracks, liked_indices, K) -> list of tracks.
# Seed songs (liked_indices) are excluded from each strategy's candidate pool.
STRATEGIES = {
    "greedy":            lambda tracks, liked, K: greedy_recommend(tracks, K, liked),
    "content_filtering": lambda tracks, liked, K: content_filtering_recommend(tracks, liked, K),
    "graph_dpp_rerank":  lambda tracks, liked, K: graph_dpp_rerank_recommend(tracks, liked, K),
}
STRATEGIES["mmr_floor"] = lambda tracks, liked, K: mmr_floor_recommend(tracks, liked, K)
for _lam in MMR_LAMBDAS:
    STRATEGIES[f"mmr_{_lam}"] = (lambda lam: lambda tracks, liked, K: mmr_recommend(tracks, liked, K, lam=lam))(_lam)


def evaluate(recs):
    """All list-level metrics for one recommendation list."""
    pops = [r["popularity"] for r in recs]
    avg, niche = popularity_metrics(recs)
    return {"ild": intra_list_diversity(recs),
            "gini": popularity_concentration_index(pops),
            "avg_popularity": avg, "niche_pct": niche}


def run_experiment(tracks, num_seeds=NUM_RUNS, K=10, seed=SEED, strategies=None):
    """
    Run every strategy over num_seeds random seed-song sets (5-10 liked songs
    each, drawn from a dedicated RNG(seed) so results are reproducible).
    Returns mean and sample std (ddof=1) per metric, plus per-run values.
    """
    strategies = strategies or STRATEGIES
    rng = random.Random(seed)
    all_indices = list(range(len(tracks)))
    seed_sets = [rng.sample(all_indices, rng.randint(5, 10)) for _ in range(num_seeds)]

    per_run = {name: [] for name in strategies}
    examples = {name: [] for name in strategies}
    for run_idx, liked in enumerate(seed_sets):
        seed_names = [tracks[i]["name"] for i in liked]
        print(f"-- Run {run_idx + 1}/{num_seeds}: {len(liked)} seed songs")
        liked_ids = {tracks[i]["track_id"] for i in liked}
        for name, fn in strategies.items():
            recs = fn(tracks, liked, K)
            assert not liked_ids & {r["track_id"] for r in recs}, f"{name} leaked a seed song"
            per_run[name].append(evaluate(recs))
            examples[name].append({
                "seed_songs": seed_names,
                "recommendations": [
                    {"name": r["name"], "artist": r["artists"],
                     "genre": r["genre"], "popularity": round(r["popularity"], 1)}
                    for r in recs],
            })

    summary = {}
    for name in strategies:
        entry = {"n_runs": num_seeds, "rng_seed": seed, "K": K}
        for m in METRICS:
            entry.update(describe([r[m] for r in per_run[name]], m))
        entry["runs"] = examples[name]
        entry["per_run_ild"] = [round(r["ild"], 4) for r in per_run[name]]
        entry["per_run_gini"] = [round(r["gini"], 4) for r in per_run[name]]
        entry["per_run_avg_pop"] = [round(r["avg_popularity"], 2) for r in per_run[name]]
        entry["per_run_niche_pct"] = [round(r["niche_pct"], 2) for r in per_run[name]]
        summary[name] = entry
    summary["_paired"] = paired_comparisons(per_run)
    return summary


def describe(values, metric):
    """Mean/std plus robust stats (ILD and niche% are bimodal, so median/IQR matter)."""
    v = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {metric: round(float(v.mean()), 4),
            metric + "_std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
            metric + "_median": round(float(med), 4),
            metric + "_q1": round(float(q1), 4),
            metric + "_q3": round(float(q3), 4),
            metric + "_iqr": round(float(q3 - q1), 4)}


def paired_comparisons(per_run):
    """
    For every pair (A, B) of methods and every metric: per-run differences
    A_i - B_i (same seed set i), mean/median difference, and a two-sided paired
    Wilcoxon signed-rank test. p-values are raw (uncorrected); with
    len(pairs) * len(METRICS) tests, apply a multiplicity correction before
    claiming significance for any single one.
    """
    out = {}
    for a, b in combinations(per_run, 2):
        entry = {}
        for m in METRICS:
            x = np.array([r[m] for r in per_run[a]])
            y = np.array([r[m] for r in per_run[b]])
            d = x - y
            n_nonzero = int(np.count_nonzero(np.abs(d) > 1e-12))
            if n_nonzero == 0:
                p = 1.0           # identical in every run
            else:
                p = float(wilcoxon(x, y).pvalue)
            entry[m] = {
                "diffs": [round(float(v), 4) for v in d],
                "mean_diff": round(float(d.mean()), 4),
                "median_diff": round(float(np.median(d)), 4),
                "n_nonzero": n_nonzero,
                "wilcoxon_p": float(f"{p:.3g}"),
                "a_wins": int((d > 1e-12).sum()), "b_wins": int((d < -1e-12).sum()),
            }
        out[f"{a}__minus__{b}"] = entry
    n_tests = len(out) * len(METRICS)
    alpha = 0.05 / n_tests
    for entry in out.values():
        for m in METRICS:
            entry[m]["bonferroni_alpha"] = float(f"{alpha:.3g}")
            entry[m]["significant_bonferroni"] = bool(entry[m]["wilcoxon_p"] < alpha)
    return out


# ─────────────────────────────────────────────
# 6. COMPARISON TABLE PRINTER
# ─────────────────────────────────────────────

def load_old_results(path="baseline/baseline_results.json"):
    """Load the baseline snapshot (pre-fix run, see baseline/README.md)."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


ALGO_LABELS = {
    "greedy":            "Greedy (Popularity)",
    "content_filtering": "Content Filtering",
    "mmr_0.5":           "MMR (lambda=0.5)",
    "mmr_0.7":           "MMR (lambda=0.7)",
    "mmr_0.9":           "MMR (lambda=0.9)",
    "mmr_floor":         "MMR 0.7 + niche floor",
    "graph_dpp_rerank":  "Graph DPP Rerank",
}
METRIC_LABELS = {
    "ild":            "ILD (Diversity)",
    "gini":           "Gini Index",
    "avg_popularity": "Avg Popularity",
    "niche_pct":      "Niche % (<40)",
}


def print_comparison_table(new_summary, old_summary=None):
    """Print new metrics (mean +/- std) next to the baseline snapshot and the delta."""
    has_old = old_summary is not None
    print("\n" + "=" * 86)
    head = f"{'METRIC':<20}{'BASELINE (pre-fix)':>20}{'NEW mean +/- std':>26}{'DELTA':>12}"
    print(head if has_old else f"{'METRIC':<20}{'NEW mean +/- std':>26}")
    print("=" * 86)
    for algo, label in ALGO_LABELS.items():
        if algo not in new_summary:
            continue
        print(f"\n  {label}")
        print("  " + "-" * 82)
        for m in METRIC_LABELS:
            new_val = new_summary[algo][m]
            new_s = f"{new_val:.4f} +/- {new_summary[algo].get(m + '_std', 0):.4f}"
            old_val = (old_summary or {}).get(algo, {}).get(m)
            if has_old and old_val is not None:
                print(f"  {METRIC_LABELS[m]:<18}{old_val:>20.4f}{new_s:>26}{new_val - old_val:>+12.4f}")
            else:
                print(f"  {METRIC_LABELS[m]:<18}{'(not in baseline)' if has_old else '':>20}{new_s:>26}{'N/A' if has_old else '':>12}")
    print("\n" + "=" * 86)


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

    print("\nRunning experiments ({NUM_RUNS} randomised seed sets, K=10, rng seed {SEED}) …")
    summary = run_experiment(tracks, num_seeds=NUM_RUNS, K=10)

    # Save new results
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[OK] Results saved -> output/results.json")

    # Load old results for comparison (if they exist)
    old_summary = load_old_results("baseline/baseline_results.json")

    print_comparison_table(summary, old_summary)

    best_div  = max(ALGO_LABELS.keys() & summary.keys(), key=lambda a: summary[a]["ild"])
    best_fair = min(ALGO_LABELS.keys() & summary.keys(), key=lambda a: summary[a]["gini"])
    print(f"  Highest mean ILD : {ALGO_LABELS[best_div]}  ({summary[best_div]['ild']:.4f})")
    print(f"  Lowest Gini      : {ALGO_LABELS[best_fair]} ({summary[best_fair]['gini']:.4f})")
    print()


if __name__ == "__main__":
    main()
