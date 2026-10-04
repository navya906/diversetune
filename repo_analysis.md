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
| Validity of conclusions | Limited — see "Limitations". The experiment runs correctly; what it can support is narrower than earlier write-ups claimed. |

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
5. **MMR's λ curve is nearly flat in absolute terms.** Mean ILD falls monotonically from 0.120 to 0.113 as λ increases from 0.5 to 0.9, and niche % stays at 26–28 (see `output/mmr_lambda_curve.png`). All three λ-pair ILD differences survive Bonferroni (raw p = 8.7e-5 for 0.5 vs 0.7, 1.9e-9 for 0.5 vs 0.9, 2.0e-5 for 0.7 vs 0.9, against α/84 = 5.95e-4), so the ILD trend is statistically significant but practically negligible: the full 0.5→0.9 change (0.0076) is about 0.04 of one standard deviation (≈0.21). The niche-% differences between λ values are not significant (raw p = 0.19–0.58).
6. **DPP is not shown to beat CF on niche representation** (p=0.19) and is nominally *less* equal in Gini. The earlier claim that DPP "decisively wins" is not supported.

## Complexity analysis

Notation: **n** = tracks in the dataset (3,000), **m** = seed songs (5–10), **d** = feature dimension (14), **N** = candidate-pool size (50), **K** = list length (10). Every bound below was checked against the code in `recommendation_engine.py`, and the operation counts and timings were measured.

| Stage / method | Time | What dominates (from the code) |
|---|---|---|
| **Greedy** | O(n log n) | One sort of the non-seed tracks (`greedy_recommend`); the seed filter is O(n). |
| **Retrieval** (shared by CF, MMR, DPP) | O(n·m·d + n log n) | `similarity_scores`: n tracks × m seeds × one cosine, each O(d), then one sort. |
| **Content Filtering** | O(n·m·d + n log n) | Retrieval, then take the top K. |
| **MMR** (selection only) | O(N·K·d) | `_mmr_select`: K rounds; each scans ≤ N candidates in O(N), then updates the running max-similarity of the remaining candidates with one O(d) cosine each. Treating d as a constant this is the O(N·K) greedy selection. |
| **DPP rerank** (selection only) | O(N²·d + N·K⁴) | (i) `pairwise_cosine_matrix` + building L: N² / 2 cosines, O(N²·d). (ii) The greedy loop calls `np.linalg.slogdet` on a fresh (k+1)×(k+1) submatrix for **every** remaining candidate at **every** step, i.e. N·K determinants of size up to K, each O(k³): Σ_k N·k³ = O(N·K⁴). |
| **Niche floor** (MMR+floor, DPP) | O(N log N) | One sort of the niche pool plus an O(K) swap. |

Full pipelines are the sum of retrieval and re-ranking, and the two terms scale with different quantities:

- **MMR / MMR+floor:** O(n·m·d + n log n) *retrieval, scales with the dataset* + O(N·K·d) *re-ranking, scales with the pool and list size*.
- **DPP:** O(n·m·d + n log n) *retrieval* + O(N²·d) *kernel construction* + O(N·K⁴) *greedy selection*.

**Confirmed by measurement** (n = 3,000, m = 8, N = 50, K = 10):

| Stage | Time |
|---|---|
| Retrieval (similarity + sort) | ≈ 100 ms |
| DPP rerank (kernel + selection) | ≈ 11 ms |
| MMR selection | ≈ 1.7 ms |
| Greedy | ≈ 2 ms |

- The number of `slogdet` calls equals Σ_{k<K}(N−k) exactly (240, 455, 810 for K = 5, 10, 20 at N = 50), confirming N·K determinants, which is the "N·K" part of N·K⁴.
- **In practice retrieval dominates the whole pipeline.** At these sizes the determinants are tiny (≤ 10×10, about 4 µs each, dominated by call overhead), so the K³ per-determinant cost is invisible: wall-clock time grows roughly with N·K calls and with N² for the Python-loop kernel build (DPP: 10.8 ms at N = 50, 32.7 ms at N = 100, 109 ms at N = 200), not with K⁴. The K⁴ term would only start to dominate for much larger K (the cost of a single `slogdet` jumps from ≈ 40 µs at 100×100 to ≈ 0.8 ms at 200×200).
- The experiment recomputes retrieval separately for CF, the three MMR variants, MMR+floor and DPP (six times per seed set). Sharing one retrieval per seed set would cut experiment time by a large factor without changing any result.

**Note on the O(N·K²) figure.** The commonly quoted O(N·K²) greedy-MAP cost for DPPs applies to an *incremental* implementation, not to this one. This code recomputes each candidate's log-determinant from scratch, hence O(N·K⁴) (the extra K² comes from recomputing a k×k determinant, O(k³), instead of updating one).

### Future work: incremental Cholesky update (O(N·K²))

Fast greedy MAP inference for DPPs (Chen, Zhang & Zhou, NeurIPS 2018) maintains, for each candidate i, a row of the Cholesky factor of the selected-set kernel. When item j is selected, each remaining candidate's row gains one entry, computed in O(k) from its existing row and the new row of the kernel, and the marginal log-determinant gain of i is just log of its squared residual. Each of the K rounds therefore costs O(N·k) instead of N determinants of size k, for O(N·K²) total, on top of the O(N²·d) kernel construction (or O(N·K·d) if kernel rows are computed lazily for selected items only). That would make the re-ranking cost for DPP the same order as the MMR selection times K. Because the greedy log-det objective is unchanged, the selected lists should be identical up to the 1e-8 jitter currently added to the diagonal, which makes this a drop-in speed-up that can be validated by comparing selected lists on the 30 seed sets. Two cheaper engineering fixes are independent of it: vectorising `similarity_scores` as one normalised matrix product (the dominant 100 ms stage) and sharing retrieval across strategies.

## Limitations

**Data.** The source's low-popularity file caps popularity at 68, which compressed the scale to 11–68 and made the old fixed "niche = popularity < 40" threshold meaningless. This was fixed by merging the dataset's high-popularity file (range 11–100, 25.2% of tracks below 40) rather than by redefining niche as a percentile; the < 40 threshold is still an arbitrary absolute cut-off on a skewed, genre-dependent distribution (8% of Rock tracks are niche against 44% of World tracks), so "niche" partly tracks genre. Balancing genres traded dataset size for balance: after de-duplication only 4,359 mappable tracks remain, so reaching a per-genre floor of 50 required merging Metal into Rock and Country and Indie into Indie-Folk, dropping the non-genre `gaming` playlists (107 tracks), and the 3,000-track sample is still uneven (Ambient 470 vs Indie-Folk 76 of 91 available). Genre labels are playlist-level rather than track-level: 17% of artists with at least two tracks appear under more than one genre and the spot checks show clear mislabels. **Method.** Similarity uses no audio features. It is a 14-dimensional vector dominated by a noisy genre one-hot plus duration, an explicit flag and artist rarity, even though the source data contains energy, valence, danceability and similar features. Because candidates retrieved by that similarity are nearly all the same genre, ILD is close to zero in most runs for every method (median ≤ 0.02), which leaves little room to separate re-ranking strategies and is why MMR's λ curve is almost flat. Seed sets are 30 random tracks per run rather than real listening histories, the niche floor is best-effort (missed in 2 of 30 runs for both DPP and MMR+floor), and Greedy's tie-break uses track-id order, which follows genre-alphabetical order (harmless here since only 3 Pop tracks tie at the cutoff, but not in general). **Comparison result.** MMR does not match DPP on ILD at any λ (DPP is higher, p ≤ 2.1e-5, surviving Bonferroni), but the difference is small (median +0.007 to +0.011). On niche representation no pairwise difference between CF, MMR, MMR+floor and DPP survives correction, and MMR+floor reproduces DPP's niche share (31.0 vs 30.7%, p = 0.89) and floor compliance (28/30 runs) at no detectable ILD cost relative to plain MMR. The hard niche floor, not the DPP kernel, therefore accounts for DPP's fairness behaviour, and the kernel's justified contribution is a small ILD gain over MMR+floor (−0.040 mean, p = 5.1e-6). With n = 30 and run-to-run standard deviations as large as the means, these tests cannot establish equivalence where they find no difference, and Bonferroni (α/84) is conservative, so nominal-only findings should be treated as hypotheses.

## Baseline snapshot

`output/old_results.json` is the archived **pre-fix** run (8 runs, duplicated rows, popularity capped at 68; known defects listed in `output/archive_20260929/README.md`). `run_pipeline.py` uses it for the delta table. Deltas against it show the effect of the data and algorithm fixes, not an algorithm improvement, and the baseline must not be cited as a result.

## Remaining work

- Implement the incremental-Cholesky DPP and shared retrieval described under Complexity analysis.
- Optional: audio-feature-based similarity (energy, valence, danceability and others are in the source data) to replace the noisy genre one-hot.
