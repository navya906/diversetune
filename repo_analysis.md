# DiverseTune Repository Analysis

> **Project:** DAA Experimental Study — Music Recommendation Diversity Analysis  
> **Last Updated:** 29 September 2026

---

## 🏗️ Architecture & Working State

The repository is built as a complete end-to-end data pipeline and web application for comparing music recommendation algorithms. 

**Status:** ✅ **Fully Functional End-to-End**

### 1. Data Pipeline (`integrate_new_dataset.py`, `fetch_dataset.py`)
- **Working:** Successfully fetches the `solomonameh/spotify-music-dataset` dataset from Kaggle.
- **Working:** Maps 100+ raw genres into a clean 11-genre taxonomy.
- **Working:** Enforces a stratified sample of 3,000 tracks (minimum 50 per genre), normalizes duration, and outputs cleanly to `data/song_track.csv`.

### 2. Algorithm Engine (`recommendation_engine.py`)
- **Working:** Successfully loads the dataset and builds a 14-dimensional feature vector per track.
- **Working:** Evaluates three algorithms effectively:
  - **Greedy:** Popularity baseline.
  - **Content Filtering:** Cosine similarity.
  - **Graph DPP Rerank:** Determinantal Point Process for diversity with a ≥20% niche fairness constraint.
- **Working:** Evaluation metrics (ILD, Gini, Avg Popularity, Niche %) are calculated across 8 randomized seed sets (K=10) and properly exported to `output/results.json`.

### 3. Visualization (`generate_plots.py`)
- **Working:** Successfully reads the `results.json` file and outputs 5 comparative `matplotlib` charts into `output/`. 
- **Working:** Gracefully falls back if old keys like `similarity` or `hybrid` are present in the JSON, making it robust against schema changes.

### 4. Web Server & UI (`server.py`, `index.html`, `search.html`, `app.js`, `index.css`)
- **Working:** Python `http.server` handles both static file serving and JSON APIs (`/api/search` and `/api/recommend`). 
- **Working:** The frontend logic in `app.js` smoothly interfaces with the backend to provide fast, interactive searches and side-by-side recommendation algorithm comparisons.

### 5. Orchestrator (`run_pipeline.py`)
- **Working:** CLI-driven script that ties the entire process together sequentially, allowing conditional skips (e.g., `--skip-dl`).

---

## 📊 Current Results & Methodology

### Seed Data
The experiment utilizes 8 randomized seed sets, each containing 5–10 "liked" songs drawn uniformly from the 3,000-track dataset. We use random seed data rather than hardcoded playlists to ensure the algorithms are evaluated across a diverse range of starting states (from highly popular mainstream tracks to deep niche cuts). This simulates different types of users accurately.

### Algorithms Used
We compare three fundamentally different approaches to recommendation:
1. **Greedy (Popularity):** Ranks candidate tracks entirely by raw popularity. This simulates the baseline "Top 50" radio model. It is prone to the popularity bias but guarantees highly recognizable tracks.
2. **Content Filtering (Similarity):** Computes the average cosine similarity between a candidate's feature vector (genre, duration, explicit flag, artist rarity) and the liked songs' vectors. This simulates the "More Like This" echo-chamber model.
3. **Graph DPP Rerank (Hybrid):** Retrieves top-50 candidates via content similarity, then applies a Determinantal Point Process (DPP) to re-rank them. DPP explicitly maximizes the log-determinant of a quality-similarity kernel matrix, mathematically balancing relevance (quality) against diversity. Finally, a strict fairness constraint guarantees ≥20% of the list contains niche tracks (popularity < 40).

### Current Outputs (3,000 Tracks, K=10)
Based on the latest run in `output/results.json`, here are the averaged metrics across all 8 seed sets:

| Metric | Greedy | Content Filtering | Graph DPP Rerank |
|---|---|---|---|
| **Intra-List Diversity (ILD) ⬆** | 0.0253 | 0.0099 | **0.0635** |
| **Gini Index (Fairness) ⬇** | 0.1821 | **0.1460** | 0.3673 |
| **Avg Popularity** | 45.42 | 5.35 | 6.57 |
| **Niche Song % (< 40)** | 60.0% | 100.0% | 100.0% |

**Key Takeaways:**
- **Graph DPP** decisively wins on **Diversity (ILD)** (0.0635 vs 0.0099 for pure CF), successfully breaking the "echo chamber" effect of standard similarity models by surfacing mathematically distinct tracks.
- **Content Filtering** performed best on the **Gini Index**, providing the most equal distribution of popularity, but suffered from extremely low diversity (ILD 0.0099), meaning it recommends highly similar, repetitive tracks.
- Both Content Filtering and Graph DPP achieved 100% niche song representation in this run, heavily favoring lesser-known artists compared to the Greedy baseline (60%).

---

## 📂 Inputs & Generated Plots

### Inputs: The Dataset
The pipeline begins by fetching the `solomonameh/spotify-music-dataset` from Kaggle. The raw data is heavily processed into `data/song_track.csv` to serve as the input for the recommendation engine. 
- **Format:** CSV with 3,000 tracks (stratified sample).
- **Columns Available:** `track_id`, `name`, `artists`, `genre` (11 core genres), `popularity` (normalized 0-100), `duration_ms`, `duration_norm`, `explicit`, `album`.
- **Feature Vector Extraction:** During execution, `recommendation_engine.py` converts these inputs into a 14-dimensional feature vector for each track:
  - 11 dimensions for One-Hot Encoded Genres.
  - 1 dimension for Normalized Duration.
  - 1 dimension for Explicit flag (0 or 1).
  - 1 dimension for Artist Rarity (1.0 for niche artists, ~0.0 for mainstream artists like BTS/Drake).

### Outputs: Generated Visualizations
Running the `generate_plots.py` script parses `output/results.json` and outputs 5 PNG charts into the `output/` directory. These are directly displayed on the frontend:
1. **`diversity_comparison.png`:** A bar chart comparing the Intra-List Diversity (ILD) of the three algorithms. (Graph DPP typically wins here).
2. **`fairness_comparison.png`:** A bar chart comparing the Gini Index of popularity distributions. A lower Gini score indicates fairer, more equitable exposure for all artists.
3. **`popularity_comparison.png`:** A bar chart showing the average raw popularity (0-100) of the recommendations produced by each algorithm.
4. **`niche_percentage.png`:** A bar chart showing what percentage of recommended songs classify as "niche" (popularity < 40).
5. **`tradeoff_chart.png`:** A dual-axis line graph plotting Diversity (ILD) against Avg Popularity to visually illustrate the trade-off between recommending recognizable hits and providing high-variance, diverse tracks.

---

## ⚠️ Discrepancies & Technical Debt

While the system is working perfectly, the recent migration from the old `algozee/spotyfy` dataset to the new `solomonameh` dataset left behind some minor inconsistencies. (Note: `cols.json`, `rename.py`, and `fetch_output.txt` were successfully deleted).

### 1. Backward Compatibility Shims
Because the project underwent naming changes mid-flight (e.g., renaming "Similarity" to "Content Filtering"), a few files have shims to handle old naming conventions:
- **`app.js` (Line 36):** `content_filtering: data.content_filtering || data.similarity`
- **`generate_plots.py` (Line 21):** `results.get("content_filtering") or results.get("similarity", {})`
- *Assessment:* These are safe and make the system robust against older `results.json` files, but they exist purely as a historical artifact of the migration.

### 2. Pipeline Edge Cases
- **`output/old_results.json`**: `run_pipeline.py` attempts to copy `results.json` to `old_results.json` for a delta comparison to print in the console. If this file doesn't exist or is empty, the console output simply shows `N/A` for the delta. It's a non-breaking discrepancy but worth noting.

### 3. `integrate_kaggle.py`
- This file was recently rewritten to simply `import integrate_new_dataset` and run its `main()` function. It acts purely as a redirect so that any old scripts or muscle memory calling `python integrate_kaggle.py` don't break. You can eventually delete it once you are used to calling `integrate_new_dataset.py`.

### 4. Hardcoded UI States
- In `app.js` `useFallbackData()`, there is hardcoded JSON data used for testing when the Python server is offline. This mock data uses hardcoded artist names and tracks that might not perfectly align with the new 3,000 track dataset, though it functions correctly for UI design purposes.
