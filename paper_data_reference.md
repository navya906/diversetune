# Paper data reference — DiverseTune

_Every number below was extracted programmatically from the current files (`output/results.json`, `data/song_track.csv`, `data/dataset_report.md`, `data/genre_mapping.csv`, `repo_analysis.md`, git history) by a script at generation time; nothing was retyped from earlier discussion. Section 8 lists items to double-check or that the repo does not contain._

## 1. Dataset stats

- **Final rows:** 3,000 (`data/song_track.csv`); **genres:** 10; smallest genre 76 (Indie-Folk), largest 470 (Ambient); floor enforced = 50.
- **Popularity:** min 11, max 100, mean 53.04, median 53, std 19.39 (sample std, ddof=1; std not in dataset_report.md), share below 40: 25.2%.

**Pipeline counts** (from `data/dataset_report.md`):

| Step | Rows |
|---|---|
| Raw rows, low+high popularity files | 4,831 |
| Removed as duplicate (normalised name, artists) | 365 |
| After dedup | 4,466 |
| After genre mapping (107 `gaming` tracks dropped) | 4,359 |
| Final sample, no replacement | 3,000 |

**Duplicates before vs after:** the pre-fix CSV committed as `3171e19` had 3,000 rows with **1,490** duplicate (name, artists) rows (popularity 11–68, mean 44.3, 11 genres incl. Metal 19 and Country 11). Current CSV: **0** duplicate (name, artists) rows and 0 duplicate track_ids. Cause of the old duplicates: sampling with replacement (see repo_analysis/commit `6ee86c2`).

**Per-genre distribution** (sampled vs available after dedup+mapping, and niche rate = share with popularity < 40):

| Genre | In sample | Available | Sampled share of available | Niche rate (pop<40) | Mean popularity |
|---|---|---|---|---|---|
| Ambient | 470 | 698 | 67% | 18.5% | 48.4 |
| World | 437 | 647 | 68% | 44.4% | 42.6 |
| Electronic | 385 | 567 | 68% | 24.2% | 51.4 |
| Pop | 379 | 558 | 68% | 25.6% | 60.5 |
| Latin | 370 | 544 | 68% | 37.0% | 50.3 |
| Hip-Hop | 347 | 509 | 68% | 16.1% | 60.2 |
| Rock | 281 | 406 | 69% | 7.8% | 66.1 |
| Jazz | 162 | 223 | 73% | 13.0% | 51.1 |
| Classical | 93 | 116 | 80% | 35.5% | 46.7 |
| Indie-Folk | 76 | 91 | 84% | 21.1% | 57.0 |

Imbalance: Ambient 470 vs Indie-Folk 76 (of 91 available); ratio 6.2:1.

Raw `playlist_genre` → mapped genre (from `data/genre_mapping.csv`, counts after dedup):

| Mapped genre | Raw genres (n tracks) |
|---|---|
| Ambient | ambient (319), lofi (299), wellness (80) |
| Classical | classical (116) |
| DROPPED | gaming (107) |
| Electronic | disco (9), electronic (558) |
| Hip-Hop | funk (28), gospel (34), hip-hop (354), r&b (50), soul (43) |
| Indie-Folk | country (11), folk (64), indie (16) |
| Jazz | blues (78), jazz (145) |
| Latin | brazilian (147), latin (397) |
| Pop | cantopop (27), j-pop (23), k-pop (17), korean (32), mandopop (14), pop (445) |
| Rock | metal (30), punk (69), rock (307) |
| World | afrobeats (72), arabic (184), indian (56), reggae (23), soca (14), turkish (70), world (228) |

**Label-quality audit** (`data/dataset_report.md`): artists with ≥2 tracks: 303; with tracks in >1 genre: 52 (17.2%). Playlist purity is 100% by construction (genre = playlist genre) and does not measure label accuracy.

## 2. Headline results (all seven configurations, from `output/results.json`)

n_runs = 30, K = 10, rng_seed = 42. std is sample std (ddof=1); IQR = Q3 − Q1 (numpy linear-interpolation percentiles).

### ILD

| Method | Mean | Std | Median | Q1 | Q3 | IQR |
|---|---|---|---|---|---|---|
| Greedy | 0.0020 | 0.0000 | 0.0020 | 0.0020 | 0.0020 | 0.0000 |
| Content Filtering | 0.0701 | 0.1659 | 0.0025 | 0.0016 | 0.0108 | 0.0092 |
| MMR λ=0.5 | 0.1203 | 0.2118 | 0.0110 | 0.0048 | 0.1518 | 0.1470 |
| MMR λ=0.7 | 0.1166 | 0.2074 | 0.0075 | 0.0036 | 0.1518 | 0.1483 |
| MMR λ=0.9 | 0.1127 | 0.2031 | 0.0070 | 0.0021 | 0.1457 | 0.1436 |
| MMR 0.7 + niche floor | 0.1098 | 0.2096 | 0.0060 | 0.0034 | 0.0343 | 0.0309 |
| Graph DPP Rerank | 0.1494 | 0.2359 | 0.0196 | 0.0084 | 0.2715 | 0.2631 |

### Gini

| Method | Mean | Std | Median | Q1 | Q3 | IQR |
|---|---|---|---|---|---|---|
| Greedy | 0.0145 | 0.0004 | 0.0144 | 0.0144 | 0.0144 | 0.0000 |
| Content Filtering | 0.1669 | 0.0784 | 0.1578 | 0.1123 | 0.2335 | 0.1212 |
| MMR λ=0.5 | 0.1866 | 0.0825 | 0.1783 | 0.1241 | 0.2601 | 0.1360 |
| MMR λ=0.7 | 0.1857 | 0.0759 | 0.1836 | 0.1242 | 0.2428 | 0.1186 |
| MMR λ=0.9 | 0.1778 | 0.0759 | 0.1706 | 0.1149 | 0.2396 | 0.1247 |
| MMR 0.7 + niche floor | 0.1996 | 0.0633 | 0.1841 | 0.1557 | 0.2448 | 0.0892 |
| Graph DPP Rerank | 0.2046 | 0.0577 | 0.1871 | 0.1624 | 0.2443 | 0.0818 |

### Avg popularity

| Method | Mean | Std | Median | Q1 | Q3 | IQR |
|---|---|---|---|---|---|---|
| Greedy | 93.68 | 0.06 | 93.70 | 93.70 | 93.70 | 0.00 |
| Content Filtering | 53.11 | 14.82 | 57.10 | 44.17 | 62.02 | 17.85 |
| MMR λ=0.5 | 53.35 | 13.13 | 55.25 | 46.52 | 61.17 | 14.65 |
| MMR λ=0.7 | 53.50 | 12.61 | 56.10 | 48.60 | 61.23 | 12.62 |
| MMR λ=0.9 | 52.98 | 14.39 | 57.10 | 44.17 | 61.70 | 17.52 |
| MMR 0.7 + niche floor | 51.67 | 11.48 | 51.25 | 48.42 | 59.85 | 11.43 |
| Graph DPP Rerank | 50.81 | 11.02 | 54.35 | 43.80 | 57.40 | 13.60 |

### Niche %

| Method | Mean | Std | Median | Q1 | Q3 | IQR |
|---|---|---|---|---|---|---|
| Greedy | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Content Filtering | 26.00 | 26.73 | 20.00 | 2.50 | 47.50 | 45.00 |
| MMR λ=0.5 | 26.67 | 22.49 | 20.00 | 10.00 | 40.00 | 30.00 |
| MMR λ=0.7 | 25.67 | 22.23 | 20.00 | 10.00 | 40.00 | 30.00 |
| MMR λ=0.9 | 27.67 | 26.22 | 20.00 | 10.00 | 47.50 | 37.50 |
| MMR 0.7 + niche floor | 31.00 | 17.68 | 20.00 | 20.00 | 40.00 | 20.00 |
| Graph DPP Rerank | 30.67 | 19.46 | 20.00 | 20.00 | 30.00 | 10.00 |

**Niche-floor compliance** (runs with niche % ≥ 20, from `per_run_niche_pct`):

| Method | Runs ≥20% | Runs <20% |
|---|---|---|
| Greedy | 0/30 | 30 |
| Content Filtering | 16/30 | 14 |
| MMR λ=0.5 | 19/30 | 11 |
| MMR λ=0.7 | 17/30 | 13 |
| MMR λ=0.9 | 18/30 | 12 |
| MMR 0.7 + niche floor | 28/30 | 2 |
| Graph DPP Rerank | 28/30 | 2 |

**Distinct Greedy lists across 30 runs:** 4 (Greedy only varies when a seed song falls in its top 10).

## 3. Paired Wilcoxon signed-rank tests (cited comparisons)

Two-sided paired Wilcoxon (`scipy.stats.wilcoxon`, same seed set per pair; zero differences dropped). Bonferroni over all 21 method pairs × 4 metrics = **84 tests**, α/84 = **0.000595** (5.95e-4 expected). Differences are **A − B**; wins = runs where A is larger / smaller (ties excluded).

| A − B | Metric | Mean diff | Median diff | Wins A / B | n non-zero | Raw p | Survives Bonferroni |
|---|---|---|---|---|---|---|---|
| Graph DPP Rerank − Content Filtering | ILD | +0.0792 | +0.0124 | 29 / 1 | 30/30 | 3.86e-07 | **yes** |
| Graph DPP Rerank − Content Filtering | Gini | +0.0377 | +0.0331 | 23 / 7 | 30/30 | 0.00237 | no |
| Graph DPP Rerank − Content Filtering | Avg popularity | -2.30 | -5.10 | 8 / 22 | 30/30 | 0.114 | no |
| Graph DPP Rerank − Content Filtering | Niche % | +4.67 | +10.00 | 17 / 8 | 25/30 | 0.193 | no |
| Graph DPP Rerank − MMR λ=0.7 | ILD | +0.0328 | +0.0075 | 29 / 1 | 30/30 | 3.15e-07 | **yes** |
| Graph DPP Rerank − MMR λ=0.7 | Gini | +0.0189 | +0.0286 | 20 / 10 | 30/30 | 0.0277 | no |
| Graph DPP Rerank − MMR λ=0.7 | Avg popularity | -2.69 | -4.05 | 8 / 22 | 30/30 | 0.046 | no |
| Graph DPP Rerank − MMR λ=0.7 | Niche % | +5.00 | +10.00 | 16 / 5 | 21/30 | 0.194 | no |
| Graph DPP Rerank − MMR 0.7 + niche floor | ILD | +0.0396 | +0.0096 | 28 / 2 | 30/30 | 5.14e-06 | **yes** |
| Graph DPP Rerank − MMR 0.7 + niche floor | Gini | +0.0051 | +0.0099 | 19 / 11 | 30/30 | 0.253 | no |
| Graph DPP Rerank − MMR 0.7 + niche floor | Avg popularity | -0.86 | -2.05 | 12 / 17 | 29/30 | 0.381 | no |
| Graph DPP Rerank − MMR 0.7 + niche floor | Niche % | -0.33 | +0.00 | 8 / 5 | 13/30 | 0.888 | no |
| MMR 0.7 + niche floor − MMR λ=0.7 | ILD | -0.0068 | +0.0000 | 2 / 10 | 12/30 | 0.0499 | no |
| MMR 0.7 + niche floor − MMR λ=0.7 | Gini | +0.0139 | +0.0000 | 12 / 0 | 12/30 | 0.00222 | no |
| MMR 0.7 + niche floor − MMR λ=0.7 | Avg popularity | -1.83 | +0.00 | 0 / 12 | 12/30 | 0.00221 | no |
| MMR 0.7 + niche floor − MMR λ=0.7 | Niche % | +5.33 | +0.00 | 12 / 0 | 12/30 | 0.00149 | no |

Other comparisons discussed in `repo_analysis.md` (same format):

| A − B | Metric | Mean diff | Median diff | Wins A / B | Raw p | Survives |
|---|---|---|---|---|---|---|
| MMR λ=0.7 − Content Filtering | ILD | +0.0465 | +0.0033 | 27 / 1 | 3.76e-05 | **yes** |
| MMR λ=0.7 − Content Filtering | Niche % | -0.33 | -0.00 | 6 / 10 | 0.6 | no |
| Graph DPP Rerank − MMR λ=0.5 | ILD | +0.0291 | +0.0066 | 28 / 2 | 2.08e-05 | **yes** |
| Graph DPP Rerank − MMR λ=0.5 | Niche % | +4.00 | +0.00 | 14 / 6 | 0.242 | no |
| Graph DPP Rerank − MMR λ=0.9 | ILD | +0.0367 | +0.0113 | 30 / 0 | 1.86e-09 | **yes** |
| Graph DPP Rerank − MMR λ=0.9 | Niche % | +3.00 | +10.00 | 16 / 8 | 0.278 | no |
| MMR 0.7 + niche floor − Content Filtering | ILD | +0.0397 | +0.0019 | 25 / 3 | 8.17e-05 | **yes** |
| MMR 0.7 + niche floor − Content Filtering | Niche % | +5.00 | -0.00 | 14 / 8 | 0.0446 | no |

**Bonferroni-surviving results excluding Greedy:** 12; metrics involved: ['ild']. (Greedy differs from every method on all four metrics.)

## 4. Complexity figures

Notation: **n** = tracks in dataset, **m** = seed songs per run, **d** = feature dimension, **N** = candidate-pool size, **K** = list length. Values used here: n = 3000, m = 5–10, d = 14, N = 50, K = 10.

Complexity table (verbatim from `repo_analysis.md`):

| Stage / method | Time | What dominates (from the code) |
|---|---|---|
| **Greedy** | O(n log n) | One sort of the non-seed tracks (`greedy_recommend`); the seed filter is O(n). |
| **Retrieval** (shared by CF, MMR, DPP) | O(n·m·d + n log n) | `similarity_scores`: n tracks × m seeds × one cosine, each O(d), then one sort. |
| **Content Filtering** | O(n·m·d + n log n) | Retrieval, then take the top K. |
| **MMR** (selection only) | O(N·K·d) | `_mmr_select`: K rounds; each scans ≤ N candidates in O(N), then updates the running max-similarity of the remaining candidates with one O(d) cosine each. Treating d as a constant this is the O(N·K) greedy selection. |
| **DPP rerank** (selection only) | O(N²·d + N·K⁴) | (i) `pairwise_cosine_matrix` + building L: N² / 2 cosines, O(N²·d). (ii) The greedy loop calls `np.linalg.slogdet` on a fresh (k+1)×(k+1) submatrix for **every** remaining candidate at **every** step, i.e. N·K determinants of size up to K, each O(k³): Σ_k N·k³ = O(N·K⁴). |
| **Niche floor** (MMR+floor, DPP) | O(N log N) | One sort of the niche pool plus an O(K) swap. |


**Fresh timing re-measurement** (this machine, n=3000, m=8, N=50, K=10; best of several repeats, ms) compared with the figures quoted in `repo_analysis.md`:

| Stage | Fresh (ms) | Quoted in repo_analysis.md |
|---|---|---|
| Retrieval (similarity + sort) | 100.5 | ≈ 100 ms |
| DPP rerank (kernel + selection) | 10.4 | ≈ 11 ms |
| MMR selection | 1.75 | ≈ 1.7 ms |
| Greedy | 1.3 | ≈ 2 ms |

**N-scaling of `dpp_rerank` and `_mmr_select`** (K=10):

| N | DPP (ms) | MMR (ms) |
|---|---|---|
| 25 | 3.5 | 0.76 |
| 50 | 10.1 | 1.73 |
| 100 | 33.0 | 3.70 |
| 200 | 113.8 | 7.56 |

**`slogdet` call counts** (N=50) against Σ_{k<K}(N−k):

| K | calls measured | Σ(N−k) |
|---|---|---|
| 5 | 240 | 240 |
| 10 | 455 | 455 |
| 20 | 810 | 810 |

**Single `slogdet` cost vs matrix size** (µs, mean of 200 calls): 10×10: 3.6, 50×50: 12.9, 100×100: 43.2, 200×200: 777.4, 400×400: 8874.7.

Notes: (i) at K=10 the determinants are ≤10×10 and dominated by call overhead, so wall-clock grows ~N·K and ~N², not K⁴; (ii) the O(N·K²) DPP cost requires an incremental Cholesky update (Chen, Zhang & Zhou, NeurIPS 2018), which this code does not implement; (iii) the experiment recomputes retrieval 6× per seed set (CF, MMR×3, MMR+floor, DPP).

## 5. Figures / plots inventory

Files are in `output/` (git-ignored; regenerate with `python run_pipeline.py --skip-dl --plots`). "Suggested paper figure" is a proposal, not something the repo defines. All bar charts: bars = mean, whiskers = ±1 std, white diamond = median, n=30 runs; five series (Greedy, CF, MMR λ=0.7, MMR+floor, DPP).

| File | Shows | Suggested paper figure | Size (KB) | Last written |
|---|---|---|---|---|
| `diversity_comparison.png` | ILD by method | Fig. 1 (Results: diversity) | 62 | 2026-10-04 19:21 |
| `niche_percentage.png` | Niche-track share (pop<40) by method | Fig. 2 (Results: niche representation) | 72 | 2026-10-04 19:21 |
| `fairness_comparison.png` | Gini of list popularity by method (lower = more equal) | Fig. 3 (or appendix; Greedy trivially lowest) | 74 | 2026-10-04 19:21 |
| `popularity_comparison.png` | Mean popularity of recommendations by method | Fig. 4 (or appendix) | 70 | 2026-10-04 19:21 |
| `tradeoff_chart.png` | Per-method mean ILD vs mean popularity (scatter) | Fig. 5 (Discussion: trade-off) | 86 | 2026-10-04 19:21 |
| `mmr_lambda_curve.png` | MMR λ∈{0.5,0.7,0.9}: mean ILD and niche % vs λ, with CF / MMR+floor / DPP reference lines | Fig. 6 (MMR sensitivity) | 71 | 2026-10-04 19:21 |

Orphans in `output/`: archive_20260929, old_results.json.

## 6. Limitations source data

**Genre skew in niche rate** (computed from `data/song_track.csv`; per-genre niche rates are *not* in `dataset_report.md`): Rock 7.8% (22/281) vs World 44.4% (194/437); full table in Section 1. Overall niche share 25.2%.

**Floor failures** (niche % < 20; floor = max(1, int(K·0.20)) = 2 niche tracks): DPP 2/30 (runs [12, 22]), MMR+floor 2/30 (runs [12, 22]); same runs in both. Without the floor: plain MMR 0.7 13/30, CF 14/30.

**MMR λ curve:** mean ILD 0.1203 (λ=0.5) → 0.1166 (0.7) → 0.1127 (0.9); mean niche % 26.67 → 25.67 → 27.67; ILD std at each λ 0.212 / 0.207 / 0.203 (the 0.120→0.113 range is ≈ 0.04 of one std). Paired ILD tests between λ values (raw p): 0.5 vs 0.7 8.69e-05, 0.5 vs 0.9 1.86e-09, 0.7 vs 0.9 2e-05 (Bonferroni: True/True/True).

**ILD is bimodal / near zero** (runs with ILD < 0.05 | > 0.1): Greedy 30 | 0; Content Filtering 25 | 4; MMR λ=0.5 22 | 8; MMR λ=0.7 22 | 8; MMR λ=0.9 22 | 8; MMR 0.7 + niche floor 23 | 7; Graph DPP Rerank 22 | 8. Medians: Greedy 0.0020, Content Filtering 0.0025, MMR λ=0.5 0.0110, MMR λ=0.7 0.0075, MMR λ=0.9 0.0070, MMR 0.7 + niche floor 0.0060, Graph DPP Rerank 0.0196.

**Data-size vs balance:** 4,359 mappable tracks after dedup; 3,000 sampled (69%); Metal (49 raw) merged into Rock, Country (11) + Indie (17) merged with Folk; 107 `gaming` tracks dropped (see Section 1 table).

**Label noise:** 52 of 303 multi-track artists (17.2%) appear under >1 genre; spot-check mislabels are listed in `data/dataset_report.md` §5.

**Audio features available but unused:** time_signature, speechiness, danceability, energy, mode, instrumentalness, valence, key, tempo, loudness, acousticness, liveness (columns present in the source CSVs).

## 7. Repo / reproducibility references

- **Repository:** https://github.com/navya906/diversetune (branch `Main`).
- **Current HEAD:** `d67113fa97a48a3aa936f0382801bb3e520d078c` (Merge pull request #2 from navya906/phase6-complexity-limitations).
- **Commit history:**

```
d67113f 2026-10-04 Merge pull request #2 from navya906/phase6-complexity-limitations
092b0c7 2026-10-04 Add complexity analysis and Limitations (Phase 6)
8906c4c 2026-10-04 repo_analysis: drop completed items from remaining work
0ad33eb 2026-10-04 Replace stale hard-coded results in the web UI with data rendered from results.json
fbb12f9 2026-10-04 Repo hygiene (Phase 5): gitignore, untrack generated files, README, requirements
6ee86c2 2026-10-04 Fix dataset, algorithms and stats; add MMR and MMR+floor (Phases 1-4)
3171e19 2026-09-29 pre-fix baseline
6f75d28 2026-04-09 Cleanup: Remove pycache, output logs, and add gitignore
47e0041 2026-04-09 initial commit
```
- **Dataset:** Kaggle `solomonameh/spotify-music-dataset`, kagglehub cache path `.../versions/1/` (Kaggle dataset version 1). Files used:
  - `high_popularity_spotify_data.csv`: 1,686 rows, 730,216 bytes, sha256 `ba70ab2da48003ee…`
  - `low_popularity_spotify_data.csv`: 3,145 rows, 1,364,298 bytes, sha256 `6b9d478cc8f5582d…`
- **Seeds / runs:** dataset sampling seed 42 (`integrate_new_dataset.py --seed`, default 42); experiment `SEED = 42`, `NUM_RUNS = 30` (`recommendation_engine.py`), seed-song set sizes drawn uniformly 5–10 from `random.Random(42)`; K = 10, N = 50, niche threshold popularity < 40, niche floor ≥ 20% (2 of 10).
- **Dependencies** (`requirements.txt`): numpy==2.1.3, pandas==2.3.0, scipy==1.15.3, matplotlib==3.10.8, kagglehub==0.3.12; Python 3.12.10.
- **Reproduce:** `pip install -r requirements.txt && python run_pipeline.py --plots` (or `--skip-dl` to reuse `data/song_track.csv`).
- **Baseline snapshot:** `baseline/baseline_results.json` (exists); the pre-fix run (8 seed sets, duplicated rows, popularity capped at 68); **do not cite** — see `baseline/README.md`. Keys: ['greedy', 'content_filtering', 'graph_dpp_rerank'].

## 8. Flags: things to check or not in the repo

- Final commit: the request says `092b0c7`; that is the last *content* commit (Phase 6), but current `Main` HEAD is `d67113f` (the PR #2 merge commit, identical tree). Cite one deliberately.
- Baseline path: the request says `data/baseline_results.json`; the actual path is `baseline/baseline_results.json` (moved in Phase 5 so it stays tracked while `output/` is ignored). `output/old_results.json` is a stale untracked local copy of the same file and is no longer read by any script.
- Duplicate count before dedup: the audit figure "1,488 of 3,000" came from the uncommitted working-tree CSV; the CSV committed as `3171e19` has 1,490. Use 1,490 if you cite the committed pre-fix state, and state which file.
- The 1,488/1,490 figure is for the **old sampled CSV** (duplicates created by sampling with replacement), not duplicates in the raw Kaggle data; the raw merged data had 365 duplicate (name, artists) rows of 4,831.
- Rock 8% / World 44% niche rates are **not in `dataset_report.md`** (it has no per-genre popularity); they are computed here from `data/song_track.csv` (sample, not the full dedup pool).
- Popularity std is not in `dataset_report.md`; computed here from the CSV.
- Timing numbers and `slogdet` counts come from an ad-hoc benchmark; **no benchmark script is committed**, and `output/results.json` does not contain timings. Section 4 re-measured them here. Timings are machine-dependent (cite as indicative, not as constants); call counts are exact.
- Possible wording error in `repo_analysis.md`: it says some ILD differences between MMR λ values are "nominally significant". In `results.json` all three λ-pair ILD differences survive Bonferroni (raw p 8.7e-5, 1.9e-9, 2.0e-5 vs α/84 = 5.95e-4): ILD decreases monotonically with λ, significant but practically tiny (0.120 → 0.113, ≈0.04 std). Describe it that way in the paper ("significant but negligible"), not as "within noise".
- Figure numbers in Section 5 are suggestions; the repo has no paper-figure mapping. `fairness_comparison.png` shows Gini, which is degenerate for Greedy (near-identical popularity), so consider omitting or caveating it.
- `output/` (results.json, PNGs) is git-ignored and exists only locally; the numbers in this document are not recoverable from GitHub without re-running the pipeline (deterministic with seed 42, but re-run before final submission).
- No per-run seed-genre coherence statistic, no equivalence tests (TOST), no multiplicity-corrected p-values (Holm) and no effect sizes (e.g. rank-biserial) are stored; only raw Wilcoxon p-values with a Bonferroni flag.
- No statistical test between methods on **Gini / avg popularity** beyond what `_paired` already contains; if the paper makes claims about them, take them from Section 3 tables only.
- No user study, no audio-feature similarity, no second dataset: nothing in the repo supports claims beyond this single dataset and a genre-based similarity.

### Consistency check of quoted `repo_analysis.md` figures against `results.json`

| Quoted figure | Quoted | Actual | OK |
|---|---|---|---|
| DPP−CF ILD mean diff | 0.079 | 0.0792 | yes |
| MMR+floor−DPP ILD mean diff | -0.04 | -0.0396 | yes |
| MMR+floor niche mean | 31.0 | 31.0000 | yes |
| DPP niche mean | 30.7 | 30.6667 | yes |
| DPP ILD mean | 0.149 | 0.1494 | yes |
| Greedy avg pop | 93.7 | 93.6833 | yes |
| MMR 0.5 ILD | 0.12 | 0.1203 | yes |
| MMR 0.9 ILD | 0.113 | 0.1127 | yes |
| Rock niche rate | 0.08 | 0.0783 | yes |
| World niche rate | 0.44 | 0.4439 | yes |
