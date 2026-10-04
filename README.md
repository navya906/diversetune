# DiverseTune

DAA experimental study of how recommendation re-ranking trades relevance against diversity and popularity
fairness on Spotify track data. Methods compared: **Greedy** (popularity), **Content Filtering** (cosine
similarity), **MMR** (λ = 0.5 / 0.7 / 0.9), **MMR + niche floor**, and **Graph DPP Rerank**.

Results and their statistical caveats are in [`repo_analysis.md`](repo_analysis.md); dataset facts and the
label-quality audit are in [`data/dataset_report.md`](data/dataset_report.md).

## Setup

Requires Python 3.12.

```bash
pip install -r requirements.txt
```

The dataset is downloaded from Kaggle (`solomonameh/spotify-music-dataset`) through `kagglehub`; you may need
Kaggle credentials configured for `kagglehub` on first download. `data/song_track.csv` is committed, so the
download is only needed to rebuild it.

## Run

```bash
python run_pipeline.py --plots              # rebuild dataset, run experiments, draw plots
python run_pipeline.py --skip-dl --plots    # reuse data/song_track.csv (no download)
python diagnostic.py                        # re-audit genre labels, appends to data/dataset_report.md
python server.py                            # web UI + API at http://localhost:8000
```

`output/` (results and PNGs) is generated and git-ignored; run the pipeline once before opening the web UI.

## Reproducibility

- Dataset sampling seed: `42` (`integrate_new_dataset.py --seed`).
- Experiment: `SEED = 42`, `NUM_RUNS = 30` in `recommendation_engine.py`. Each run draws 5-10 random seed
  ("liked") songs from `random.Random(42)`; all methods see the same seed sets, so results are paired and
  exactly reproducible.
- Seed songs are excluded from every method's candidates (asserted on every run).
- `python generate_paper_data.py` regenerates `paper_data_reference.md` (every dataset, result, test, timing and figure used in the paper) directly from `output/results.json` and the tracked data files; run the pipeline first.

## What each file does

| File | Purpose |
|---|---|
| `integrate_new_dataset.py` | Downloads and merges the low+high popularity CSVs, dedupes on (name, artists), maps `playlist_genre` to 10 genres, samples 3,000 tracks without replacement with a 50-track genre floor, writes `data/song_track.csv`, `data/genre_mapping.csv`, `data/dataset_report.md`. |
| `diagnostic.py` | Audits genre-label quality on `data/song_track.csv` and appends the findings to the dataset report. |
| `fetch_dataset.py` | Quick inspection utility: downloads the Kaggle dataset and prints columns and shape. |
| `recommendation_engine.py` | Feature vectors, the five methods, metrics (ILD, Gini, avg popularity, niche %), the 30-run experiment loop, descriptive and paired Wilcoxon statistics with Bonferroni flags. |
| `generate_plots.py` | Draws the comparison charts and the MMR λ chart into `output/` from `output/results.json`. |
| `run_pipeline.py` | Orchestrates dataset -> experiments -> results table (vs `baseline/`) -> plots. |
| `generate_paper_data.py` | Compiles all paper statistics (dataset stats, results tables, paired tests, complexity measurements, figure inventory, reproducibility references, open flags) into `paper_data_reference.md`. |
| `server.py` | HTTP server for the web UI and the `/api/search` and `/api/recommend` endpoints. |
| `index.html`, `search.html`, `app.js`, `index.css` | Web frontend. The dashboard tables are rendered from `output/results.json`; the search demo shows all five methods via `server.py`. |
| `baseline/` | Archived pre-fix results used only for the delta table. Not citable. |

## Outputs

`output/results.json` (per-method mean, std, median, quartiles, per-run values, and `_paired` comparisons) and
`diversity_comparison.png`, `fairness_comparison.png`, `popularity_comparison.png`, `niche_percentage.png`,
`tradeoff_chart.png`, `mmr_lambda_curve.png`.

## Metrics

- **ILD**: mean pairwise (1 - cosine) within a list; higher = more diverse.
- **Gini**: inequality of popularity within a list; lower = more equal.
- **Niche %**: share of a list with popularity < 40.
