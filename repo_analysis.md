# DiverseTune — Repository Analysis

> DAA experimental study: diversity and fairness of music-recommendation re-ranking.
> All numbers below are generated from `output/results.json` (30 runs, K=10, RNG seed 42, 3,000 tracks, 10 genres). The previous analysis (8 runs, defective data) has been withdrawn; its archive is in `output/archive_20260929/` and must not be cited.

## Status

| Component | State |
|---|---|
| Dataset build (`integrate_new_dataset.py`) | Working. Deduped on (name, artists), no sampling with replacement, per-genre floor enforced and asserted. |
| Experiment (`recommendation_engine.py`) | Working. Four method families (7 configurations), 30 paired runs, seed songs excluded and asserted, results reproducible. |
| Plots (`generate_plots.py`) | Working. Five comparison charts plus an MMR λ chart. |
| Web UI / server | Updated. `index.html` renders metrics, comparison and paired-test tables from `output/results.json` (no hard-coded results); `search.html` and `server.py` serve all five methods. Needs `output/` generated first. |
| Tests | None. |
| Validity of conclusions | Limited — see "Caveats". The experiment runs correctly; what it can support is narrower than earlier write-ups claimed. |

## Setup of the experiment

- **Data:** Kaggle `solomonameh/spotify-music-dataset`, low- and high-popularity files merged (popularity 11–100, mean 53.0). 4,831 raw rows → 4,466 after dedup → 3,000 sampled without replacement. Genres: Ambient 470, World 437, Electronic 385, Pop 379, Latin 370, Hip-Hop 347, Rock 281, Jazz 162, Classical 93, Indie-Folk 76. Details and the label-quality audit are in `data/dataset_report.md`.
- **Seed sets:** 30 random sets of 5–10 liked tracks drawn from `random.Random(42)`. Seed tracks are excluded from every method's candidates.
- **Similarity:** cosine over a 14-d vector (10 genre one-hot, normalised duration, explicit flag, artist-rarity). Popularity is never in the similarity.
- **Niche:** popularity < 40 (25.2% of the dataset).
- **Methods:** Greedy (top popularity, ties by track id); Content Filtering (CF); MMR at λ = 0.5 / 0.7 / 0.9 over the top-50 CF pool, no popularity floor; **MMR+floor** (MMR 0.7 plus the same ≥20%-niche enforcement as DPP, shared code `enforce_niche_floor`); Graph DPP Rerank (top-50 pool, greedy log-det MAP, then the niche floor).

## Results (mean ± std over 30 runs)

| Method | ILD | Gini | Avg popularity | Niche % |
|---|---|---|---|---|
| Greedy | 0.002 ± 0.000 | 0.015 ± 0.000 | 93.7 ± 0.1 | 0.0 ± 0.0 |
| Content Filtering | 0.070 ± 0.166 | 0.167 ± 0.078 | 53.1 ± 14.8 | 26.0 ± 26.7 |
| MMR λ=0.5 | 0.120 ± 0.212 | 0.187 ± 0.083 | 53.4 ± 13.1 | 26.7 ± 22.5 |
| MMR λ=0.7 | 0.117 ± 0.207 | 0.186 ± 0.076 | 53.5 ± 12.6 | 25.7 ± 22.2 |
| MMR λ=0.9 | 0.113 ± 0.203 | 0.178 ± 0.076 | 53.0 ± 14.4 | 27.7 ± 26.2 |
| MMR 0.7 + floor | 0.110 ± 0.210 | 0.200 ± 0.063 | 51.7 ± 11.5 | 31.0 ± 17.7 |
| Graph DPP Rerank | 0.149 ± 0.236 | 0.205 ± 0.058 | 50.8 ± 11.0 | 30.7 ± 19.5 |

ILD and niche % are strongly bimodal (std exceeds the mean for ILD), so medians are more representative:

| Method | ILD median [Q1, Q3] | Niche % median [Q1, Q3] | Runs below the 20% niche floor |
|---|---|---|---|
| Greedy | 0.0020 [0.0020, 0.0020] | 0 [0, 0] | 30 / 30 |
| Content Filtering | 0.0025 [0.0016, 0.0108] | 20 [2.5, 47.5] | 14 / 30 |
| MMR λ=0.5 | 0.0110 [0.0048, 0.1518] | 20 [10, 40] | 11 / 30 |
| MMR λ=0.7 | 0.0075 [0.0036, 0.1518] | 20 [10, 40] | 13 / 30 |
| MMR λ=0.9 | 0.0070 [0.0021, 0.1457] | 20 [10, 47.5] | 12 / 30 |
| MMR 0.7 + floor | 0.0060 [0.0034, 0.0343] | 20 [20, 40] | **2 / 30** |
| Graph DPP Rerank | 0.0196 [0.0084, 0.2715] | 20 [20, 30] | **2 / 30** |

In most runs every method except DPP returns a nearly single-genre list (median ILD ≤ 0.011); DPP's median is still only 0.02. Large ILD values come from a minority of runs (8–9 of 30 above 0.1) in which a list spans several genres.

## Significance (paired Wilcoxon signed-rank, same seed set per pair)

All 21 method pairs × 4 metrics = **84 tests**; Bonferroni threshold **α/84 = 5.95e-4** (not α/60: with 7 configurations there are 84 tests, not 60). p-values below are raw. "Survives" means p < 5.95e-4.

| Comparison (A − B) | ILD: mean / median diff, p | Niche %: mean / median diff, p |
|---|---|---|
| DPP − CF | +0.079 / +0.012, p=3.9e-7 **(survives)** | +4.7 / +10, p=0.19 (n.s.) |
| MMR 0.7 − CF | +0.047 / +0.003, p=3.8e-5 **(survives)** | −0.3 / 0, p=0.60 (n.s.) |
| DPP − MMR 0.7 | +0.033 / +0.008, p=3.2e-7 **(survives)** | +5.0 / +10, p=0.19 (n.s.) |
| MMR+floor − MMR 0.7 | −0.007 / 0.000, p=0.0499 (borderline nominal; does **not** survive) | +5.3 / 0, p=0.0015 (nominal only; does **not** survive) |
| MMR+floor − DPP | −0.040 / −0.010, p=5.1e-6 **(survives)** | +0.3 / 0, p=0.89 (n.s.) |
| DPP − MMR 0.5 | +0.029 / +0.007, p=2.1e-5 **(survives)** | +4.0 / 0, p=0.24 (n.s.) |
| DPP − MMR 0.9 | +0.037 / +0.011, p=1.9e-9 **(survives)** | +3.0 / +10, p=0.28 (n.s.) |

Among comparisons not involving Greedy, **the only results that survive Bonferroni are ILD differences.** No niche-%, Gini or popularity difference between CF, MMR, MMR+floor and DPP survives. (Greedy differs from every other method on all four metrics; those results are large and survive.) Nominal-only results (raw p < 0.05 but not surviving) are exploratory: for example DPP vs CF on Gini (p=0.0024, DPP less equal) and MMR+floor vs MMR on Gini/avg-popularity (p≈0.002).

## What the results support

1. **Greedy is a genuinely low-diversity baseline.** Its top 10 are all Pop at popularity 90–100 (ILD 0.002, avg popularity 93.7, 0% niche). The old Greedy ILD of 0.83 was an artifact of the 68-point popularity ceiling and arbitrary tie-breaking.
2. **Re-ranking raises ILD above plain CF, and DPP raises it above MMR — but by small amounts.** All of DPP, MMR and MMR+floor beat CF on ILD, and DPP beats every MMR variant and MMR+floor, all surviving Bonferroni. The median differences are 0.003–0.012 in an ILD range where most lists are almost single-genre; the mean differences (0.03–0.08) come from a minority of runs.
3. **The niche floor, not the DPP kernel, explains DPP's niche behaviour.** MMR+floor matches DPP's niche % (31.0 vs 30.7, p=0.89) and, like DPP, meets the ≥20% floor in 28/30 runs (plain MMR and CF: 11–14 of 30 runs miss it). The cost to ILD of adding the floor to MMR is not detectable (median difference 0; mean −0.007, raw p=0.0499, borderline at the nominal level and far from the Bonferroni threshold).
4. **DPP's kernel does buy a small ILD advantage over MMR+floor** (−0.040 mean / −0.010 median for MMR+floor, p=5.1e-6, survives). This is the one part of DPP's added cost that the data justifies, and it is small in absolute terms.
5. **MMR's λ curve is flat.** ILD moves from 0.120 to 0.113 across λ=0.5→0.9 and niche % stays at 26–28 (see `output/mmr_lambda_curve.png`); the λ differences are within run-to-run noise on niche % and, although some ILD differences between λ values are nominally significant, they are tiny.
6. **DPP is not shown to beat CF on niche representation** (p=0.19) and is nominally *less* equal in Gini. The earlier claim that DPP "decisively wins" is not supported.

## Caveats

- **Genre labels are playlist-level, not track-level.** 17% of artists with ≥ 2 tracks appear under more than one genre, and spot checks show mislabels (see `data/dataset_report.md`). The genre one-hot dominates the similarity vector, so "diversity" here partly measures the diversity of those noisy labels.
- **ILD is almost always near zero** because candidates retrieved by similarity are nearly all the same genre. The experiment therefore has little room to separate re-ranking strategies, which is also why the λ curve is flat.
- **30 runs from random seed sets** are not real user histories; seed sets of 5–10 random tracks are usually genre-incoherent.
- **The niche floor is best-effort** and missed in 2/30 runs for both DPP and MMR+floor when the 50-candidate pool had too few niche tracks.
- **Bonferroni is conservative**; the nominal-only findings should be treated as hypotheses, not results.
- The Greedy tie-break uses track-id order, and ids are assigned in genre-alphabetical order. Only 3 tracks tie at the cutoff here (all Pop), so it does not affect the result, but it would on other data.

## Baseline snapshot

`output/old_results.json` is the archived **pre-fix** run (8 runs, duplicated rows, popularity capped at 68; known defects listed in `output/archive_20260929/README.md`). `run_pipeline.py` uses it for the delta table. Deltas against it show the effect of the data and algorithm fixes, not an algorithm improvement, and the baseline must not be cited as a result.

## Remaining work

- Phase 5 (repo hygiene): `.gitignore`, untrack generated files, delete `integrate_kaggle.py`, remove legacy-key shims in `app.js`, add `requirements.txt` and README.
- Phase 6 (write-up): complexity analysis and Limitations paragraph.
- Update `server.py` / `index.html` / `app.js` to display MMR and MMR+floor.
- Optional: audio-feature-based similarity (energy, valence, danceability and others are in the source data) to replace the noisy genre one-hot.
