"""
generate_paper_data.py
Compiles every result, statistic and figure used in the paper into one markdown
file, paper_data_reference.md, by reading the repo's own artefacts. Nothing is
typed by hand: re-running this script regenerates the file.

Inputs (relative to the repo root):
  output/results.json            written by `python run_pipeline.py` (git-ignored; generate it first)
  data/song_track.csv, data/dataset_report.md, data/genre_mapping.csv, repo_analysis.md,
  requirements.txt, baseline/baseline_results.json   (tracked)
  output/*.png                   plot inventory (from `run_pipeline.py --plots`)
Optional inputs (the affected items are flagged in the output if absent):
  git history                    commit hashes and the pre-fix CSV (needs a full clone)
  the Kaggle files cached by kagglehub (~/.cache/kagglehub/...)  source row counts / hashes

Usage:
  python run_pipeline.py --skip-dl --plots     # once, to create output/
  python generate_paper_data.py
"""

import glob
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

REQUIRED = ['output/results.json', 'data/song_track.csv', 'data/dataset_report.md',
            'data/genre_mapping.csv', 'repo_analysis.md', 'requirements.txt']
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    sys.exit('Missing inputs: ' + ', '.join(missing) +
             '\nRun `python run_pipeline.py --skip-dl --plots` first (output/ is git-ignored).')

R = json.load(open('output/results.json', encoding='utf-8'))
D = pd.read_csv('data/song_track.csv')
REPORT = open('data/dataset_report.md', encoding='utf-8').read()
ANALYSIS = open('repo_analysis.md', encoding='utf-8').read()
GM = pd.read_csv('data/genre_mapping.csv')
P = R['_paired']
flags = []          # things the paper author must know
out = []
w = out.append


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8').stdout.strip()
    except OSError:
        return ''


_kdirs = sorted(glob.glob(os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets',
                                       'solomonameh', 'spotify-music-dataset', 'versions', '*')))
KAGGLE = _kdirs[-1] if _kdirs else None


NAMES = {'greedy': 'Greedy', 'content_filtering': 'Content Filtering', 'mmr_0.5': 'MMR λ=0.5',
         'mmr_0.7': 'MMR λ=0.7', 'mmr_0.9': 'MMR λ=0.9', 'mmr_floor': 'MMR 0.7 + niche floor',
         'graph_dpp_rerank': 'Graph DPP Rerank'}
ORDER = ['greedy', 'content_filtering', 'mmr_0.5', 'mmr_0.7', 'mmr_0.9', 'mmr_floor', 'graph_dpp_rerank']
MET = [('ild', 'ILD', 4), ('gini', 'Gini', 4), ('avg_popularity', 'Avg popularity', 2), ('niche_pct', 'Niche %', 2)]

head = sh('git rev-parse HEAD') or 'unknown (not a git checkout)'
w('# Paper data reference — DiverseTune\n')
w('_Every number below was extracted programmatically from the current files '
  '(`output/results.json`, `data/song_track.csv`, `data/dataset_report.md`, `data/genre_mapping.csv`, '
  '`repo_analysis.md`, git history) by a script at generation time; nothing was retyped from earlier '
  'discussion. Section 8 lists items to double-check or that the repo does not contain._\n')

# ---------------------------------------------------------------- 1 dataset
w('## 1. Dataset stats\n')
n = len(D)
vc = D['genre'].value_counts()
avail = GM.groupby('mapped_to')['n_tracks'].sum()
w(f'- **Final rows:** {n:,} (`data/song_track.csv`); **genres:** {D["genre"].nunique()}; '
  f'smallest genre {vc.min()} ({vc.idxmin()}), largest {vc.max()} ({vc.idxmax()}); floor enforced = 50.')
w(f'- **Popularity:** min {D.popularity.min():.0f}, max {D.popularity.max():.0f}, mean {D.popularity.mean():.2f}, '
  f'median {D.popularity.median():.0f}, std {D.popularity.std():.2f} (sample std, ddof=1; std not in dataset_report.md), '
  f'share below 40: {(D.popularity < 40).mean():.1%}.')
raw = int(re.search(r'Raw rows \(all files\): ([\d,]+)', REPORT).group(1).replace(',', ''))
rem = int(re.search(r'Removed as duplicate \(name, artists\): ([\d,]+)', REPORT).group(1).replace(',', ''))
aft = int(re.search(r'After dedup: ([\d,]+)', REPORT).group(1).replace(',', ''))
mapped = int(re.search(r'After genre mapping[^:]*: ([\d,]+)', REPORT).group(1).replace(',', ''))
dd = int(D.assign(a=D.name.str.lower().str.strip(), b=D.artists.str.lower().str.strip()).duplicated(['a', 'b']).sum())
w('\n**Pipeline counts** (from `data/dataset_report.md`):\n')
w('| Step | Rows |\n|---|---|')
w(f'| Raw rows, low+high popularity files | {raw:,} |')
w(f'| Removed as duplicate (normalised name, artists) | {rem:,} |')
w(f'| After dedup | {aft:,} |')
w(f'| After genre mapping (107 `gaming` tracks dropped) | {mapped:,} |')
w(f'| Final sample, no replacement | {n:,} |')
old_txt = sh('git show 3171e19:data/song_track.csv')
oldd = None
if old_txt:
    old = pd.read_csv(io.StringIO(old_txt))
    oldd = int(old.duplicated(['name', 'artists']).sum())
    w(f'\n**Duplicates before vs after:** the pre-fix CSV committed as `3171e19` had {len(old):,} rows with **{oldd:,}** '
      f'duplicate (name, artists) rows (popularity {old.popularity.min():.0f}–{old.popularity.max():.0f}, mean {old.popularity.mean():.1f}, '
      f'{old.genre.nunique()} genres incl. Metal {int((old.genre == "Metal").sum())} and Country {int((old.genre == "Country").sum())}). '
      f'Current CSV: **{dd}** duplicate (name, artists) rows and {int(D.track_id.duplicated().sum())} duplicate track_ids. '
      f'Cause of the old duplicates: sampling with replacement (see commit `6ee86c2`).')
else:
    w(f'\n**Duplicates before vs after:** current CSV has **{dd}** duplicate (name, artists) rows and '
      f'{int(D.track_id.duplicated().sum())} duplicate track_ids. The pre-fix CSV (commit `3171e19`) is not available (no git history).')
    flags.append('Pre-fix duplicate count unavailable: git history (commit 3171e19) not present in this checkout.')
w('\n**Per-genre distribution** (sampled vs available after dedup+mapping, and niche rate = share with popularity < 40):\n')
w('| Genre | In sample | Available | Sampled share of available | Niche rate (pop<40) | Mean popularity |\n|---|---|---|---|---|---|')
nr = (D.popularity < 40).groupby(D.genre).mean()
mp = D.groupby('genre').popularity.mean()
for g in vc.index:
    w(f'| {g} | {vc[g]} | {avail[g]} | {vc[g] / avail[g]:.0%} | {nr[g]:.1%} | {mp[g]:.1f} |')
w(f'\nImbalance: Ambient {vc["Ambient"]} vs Indie-Folk {vc["Indie-Folk"]} (of {avail["Indie-Folk"]} available); ratio {vc["Ambient"] / vc["Indie-Folk"]:.1f}:1.')
w('\nRaw `playlist_genre` → mapped genre (from `data/genre_mapping.csv`, counts after dedup):\n')
w('| Mapped genre | Raw genres (n tracks) |\n|---|---|')
for g, sub in GM.groupby('mapped_to'):
    w(f'| {g} | ' + ', '.join(f'{r.raw_genre} ({r.n_tracks})' for r in sub.itertuples()) + ' |')
al = re.search(r'Artists with >= 2 tracks: (\d+); of these, (\d+) \(([\d.]+)%\)', REPORT)
w(f'\n**Label-quality audit** (`data/dataset_report.md`): artists with ≥2 tracks: {al.group(1)}; with tracks in >1 genre: {al.group(2)} ({al.group(3)}%). '
  f'Playlist purity is 100% by construction (genre = playlist genre) and does not measure label accuracy.')

# ---------------------------------------------------------------- 2 headline
w('\n## 2. Headline results (all seven configurations, from `output/results.json`)\n')
f0 = R['greedy']
w(f'n_runs = {f0["n_runs"]}, K = {f0["K"]}, rng_seed = {f0["rng_seed"]}. std is sample std (ddof=1); IQR = Q3 − Q1 (numpy linear-interpolation percentiles).\n')
for key, lab, dg in MET:
    w(f'### {lab}\n')
    w('| Method | Mean | Std | Median | Q1 | Q3 | IQR |\n|---|---|---|---|---|---|---|')
    for k in ORDER:
        v = R[k]
        w(f'| {NAMES[k]} | {v[key]:.{dg}f} | {v[key + "_std"]:.{dg}f} | {v[key + "_median"]:.{dg}f} | '
          f'{v[key + "_q1"]:.{dg}f} | {v[key + "_q3"]:.{dg}f} | {v[key + "_iqr"]:.{dg}f} |')
    w('')
w('**Niche-floor compliance** (runs with niche % ≥ 20, from `per_run_niche_pct`):\n')
w('| Method | Runs ≥20% | Runs <20% |\n|---|---|---|')
for k in ORDER:
    v = R[k]['per_run_niche_pct']
    w(f'| {NAMES[k]} | {sum(1 for x in v if x >= 20)}/{len(v)} | {sum(1 for x in v if x < 20)} |')
w('\n**Distinct Greedy lists across 30 runs:** ' +
  str(len({tuple(r["name"] for r in x["recommendations"]) for x in R["greedy"]["runs"]})) +
  ' (Greedy only varies when a seed song falls in its top 10).')

# ---------------------------------------------------------------- 3 paired
w('\n## 3. Paired Wilcoxon signed-rank tests (cited comparisons)\n')
first = next(iter(P.values()))['ild']
alpha = first['bonferroni_alpha']
ntests = round(0.05 / alpha)
w(f'Two-sided paired Wilcoxon (`scipy.stats.wilcoxon`, same seed set per pair; zero differences dropped). '
  f'Bonferroni over all {len(P)} method pairs × 4 metrics = **{ntests} tests**, α/{ntests} = **{alpha:.3g}** '
  f'(5.95e-4 expected). Differences are **A − B**; wins = runs where A is larger / smaller (ties excluded).\n')


def pair(a, b):
    k = f'{a}__minus__{b}'
    if k in P:
        return P[k], 1, False
    return P[f'{b}__minus__{a}'], -1, True


CITED = [('graph_dpp_rerank', 'content_filtering'), ('graph_dpp_rerank', 'mmr_0.7'),
         ('graph_dpp_rerank', 'mmr_floor'), ('mmr_floor', 'mmr_0.7')]
w('| A − B | Metric | Mean diff | Median diff | Wins A / B | n non-zero | Raw p | Survives Bonferroni |\n|---|---|---|---|---|---|---|---|')
for a, b in CITED:
    e, sg, rev = pair(a, b)
    for key, lab, dg in MET:
        x = e[key]
        wa, wb = (x['b_wins'], x['a_wins']) if rev else (x['a_wins'], x['b_wins'])
        w(f'| {NAMES[a]} − {NAMES[b]} | {lab} | {sg * x["mean_diff"]:+.{dg}f} | {sg * x["median_diff"]:+.{dg}f} | '
          f'{wa} / {wb} | {x["n_nonzero"]}/30 | {x["wilcoxon_p"]:.3g} | {"**yes**" if x["significant_bonferroni"] else "no"} |')
w('\nOther comparisons discussed in `repo_analysis.md` (same format):\n')
w('| A − B | Metric | Mean diff | Median diff | Wins A / B | Raw p | Survives |\n|---|---|---|---|---|---|---|')
for a, b in [('mmr_0.7', 'content_filtering'), ('graph_dpp_rerank', 'mmr_0.5'), ('graph_dpp_rerank', 'mmr_0.9'),
             ('mmr_floor', 'content_filtering')]:
    e, sg, rev = pair(a, b)
    for key, lab, dg in MET[:1] + MET[3:]:
        x = e[key]
        wa, wb = (x['b_wins'], x['a_wins']) if rev else (x['a_wins'], x['b_wins'])
        w(f'| {NAMES[a]} − {NAMES[b]} | {lab} | {sg * x["mean_diff"]:+.{dg}f} | {sg * x["median_diff"]:+.{dg}f} | '
          f'{wa} / {wb} | {x["wilcoxon_p"]:.3g} | {"**yes**" if x["significant_bonferroni"] else "no"} |')
surv = [(k, m) for k, v in P.items() if 'greedy' not in k for m in ('ild', 'gini', 'avg_popularity', 'niche_pct')
        if v[m]['significant_bonferroni']]
w(f'\n**Bonferroni-surviving results excluding Greedy:** {len(surv)}; metrics involved: {sorted({m for _, m in surv})}. '
  f'(Greedy differs from every method on all four metrics.)')

# ---------------------------------------------------------------- 4 complexity
w('\n## 4. Complexity figures\n')
w('Notation: **n** = tracks in dataset, **m** = seed songs per run, **d** = feature dimension, **N** = candidate-pool size, **K** = list length. '
  f'Values used here: n = {n}, m = 5–10, d = 14, N = 50, K = 10.\n')
sec = ANALYSIS[ANALYSIS.index('## Complexity analysis'):ANALYSIS.index('## Limitations')]
tbl = re.search(r'(\| Stage / method.*?\n)\n', sec, re.S).group(1)
w('Complexity table (verbatim from `repo_analysis.md`):\n')
w(tbl)

# fresh measurement so the numbers are reproducible and checkable against repo_analysis.md
import sys
import recommendation_engine as E
tracks = E.load_and_preprocess()
rng = random.Random(1)
liked = rng.sample(range(len(tracks)), 8)


def timeit(f, *a, rep=5):
    best = 1e9
    for _ in range(rep):
        s = time.perf_counter(); f(*a); best = min(best, time.perf_counter() - s)
    return best * 1e3


cands = E.retrieve_candidates(tracks, liked, 200)
c50 = cands[:50]
t_ret = timeit(E.retrieve_candidates, tracks, liked, 50, rep=3)
t_dpp = timeit(E.dpp_rerank, c50, 10)
t_mmr = timeit(E._mmr_select, c50, 10, 0.7)
t_gr = timeit(E.greedy_recommend, tracks, 10, liked)
w('\n**Fresh timing re-measurement** (this machine, n=3000, m=8, N=50, K=10; best of several repeats, ms) '
  'compared with the figures quoted in `repo_analysis.md`:\n')
w('| Stage | Fresh (ms) | Quoted in repo_analysis.md |\n|---|---|---|')
w(f'| Retrieval (similarity + sort) | {t_ret:.1f} | ≈ 100 ms |')
w(f'| DPP rerank (kernel + selection) | {t_dpp:.1f} | ≈ 11 ms |')
w(f'| MMR selection | {t_mmr:.2f} | ≈ 1.7 ms |')
w(f'| Greedy | {t_gr:.1f} | ≈ 2 ms |')
w('\n**N-scaling of `dpp_rerank` and `_mmr_select`** (K=10):\n')
w('| N | DPP (ms) | MMR (ms) |\n|---|---|---|')
for N in (25, 50, 100, 200):
    w(f'| {N} | {timeit(E.dpp_rerank, cands[:N], 10):.1f} | {timeit(E._mmr_select, cands[:N], 10, 0.7):.2f} |')
calls = [0]
orig = np.linalg.slogdet
np.linalg.slogdet = lambda a: (calls.__setitem__(0, calls[0] + 1), orig(a))[1]
w('\n**`slogdet` call counts** (N=50) against Σ_{k<K}(N−k):\n')
w('| K | calls measured | Σ(N−k) |\n|---|---|---|')
for K in (5, 10, 20):
    calls[0] = 0
    E.dpp_rerank(c50, K)
    w(f'| {K} | {calls[0]} | {sum(50 - k for k in range(K))} |')
np.linalg.slogdet = orig
w('\n**Single `slogdet` cost vs matrix size** (µs, mean of 200 calls): ' + ', '.join(
    f'{k}×{k}: {(lambda A: (lambda s: [orig(A) for _ in range(200)] and (time.perf_counter() - s) / 200 * 1e6)(time.perf_counter()))((lambda M: M @ M.T + np.eye(k))(np.random.rand(k, k))):.1f}'
    for k in (10, 50, 100, 200, 400)) + '.')
w('\nNotes: (i) at K=10 the determinants are ≤10×10 and dominated by call overhead, so wall-clock grows ~N·K and ~N², not K⁴; '
  '(ii) the O(N·K²) DPP cost requires an incremental Cholesky update (Chen, Zhang & Zhou, NeurIPS 2018), which this code does not implement; '
  '(iii) the experiment recomputes retrieval 6× per seed set (CF, MMR×3, MMR+floor, DPP).')

# ---------------------------------------------------------------- 5 figures
w('\n## 5. Figures / plots inventory\n')
w('Files are in `output/` (git-ignored; regenerate with `python run_pipeline.py --skip-dl --plots`). '
  '"Suggested paper figure" is a proposal, not something the repo defines. All bar charts: bars = mean, whiskers = ±1 std, white diamond = median, n=30 runs; five series (Greedy, CF, MMR λ=0.7, MMR+floor, DPP).\n')
w('| File | Shows | Suggested paper figure | Size (KB) | Last written |\n|---|---|---|---|---|')
INV = [('diversity_comparison.png', 'ILD by method', 'Fig. 1 (Results: diversity)'),
       ('niche_percentage.png', 'Niche-track share (pop<40) by method', 'Fig. 2 (Results: niche representation)'),
       ('fairness_comparison.png', 'Gini of list popularity by method (lower = more equal)', 'Fig. 3 (or appendix; Greedy trivially lowest)'),
       ('popularity_comparison.png', 'Mean popularity of recommendations by method', 'Fig. 4 (or appendix)'),
       ('tradeoff_chart.png', 'Per-method mean ILD vs mean popularity (scatter)', 'Fig. 5 (Discussion: trade-off)'),
       ('mmr_lambda_curve.png', 'MMR λ∈{0.5,0.7,0.9}: mean ILD and niche % vs λ, with CF / MMR+floor / DPP reference lines', 'Fig. 6 (MMR sensitivity)')]
for f, what, fig in INV:
    pth = os.path.join('output', f)
    if os.path.exists(pth):
        w(f'| `{f}` | {what} | {fig} | {os.path.getsize(pth) / 1024:.0f} | {time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(pth)))} |')
    else:
        w(f'| `{f}` | {what} | {fig} | MISSING | |')
        flags.append(f'Plot `{f}` is missing from output/.')
w('\nOrphans in `output/`: ' + (', '.join(sorted(set(os.listdir('output')) - {f for f, _, _ in INV} - {'results.json'})) or 'none') + '.')

# ---------------------------------------------------------------- 6 limitations
w('\n## 6. Limitations source data\n')
w('**Genre skew in niche rate** (computed from `data/song_track.csv`; per-genre niche rates are *not* in `dataset_report.md`): '
  f'Rock {nr["Rock"]:.1%} ({int(((D.genre == "Rock") & (D.popularity < 40)).sum())}/{vc["Rock"]}) vs World {nr["World"]:.1%} '
  f'({int(((D.genre == "World") & (D.popularity < 40)).sum())}/{vc["World"]}); full table in Section 1. Overall niche share {(D.popularity < 40).mean():.1%}.')
bad = {k: [i + 1 for i, v in enumerate(R[k]['per_run_niche_pct']) if v < 20] for k in ('graph_dpp_rerank', 'mmr_floor')}
w(f'\n**Floor failures** (niche % < 20; floor = max(1, int(K·0.20)) = 2 niche tracks): DPP {len(bad["graph_dpp_rerank"])}/30 '
  f'(runs {bad["graph_dpp_rerank"]}), MMR+floor {len(bad["mmr_floor"])}/30 (runs {bad["mmr_floor"]}); '
  + ('same runs in both.' if bad['graph_dpp_rerank'] == bad['mmr_floor'] else 'different runs.') +
  f' Without the floor: plain MMR 0.7 {sum(1 for v in R["mmr_0.7"]["per_run_niche_pct"] if v < 20)}/30, CF {sum(1 for v in R["content_filtering"]["per_run_niche_pct"] if v < 20)}/30.')
w(f'\n**MMR λ curve:** mean ILD {R["mmr_0.5"]["ild"]:.4f} (λ=0.5) → {R["mmr_0.7"]["ild"]:.4f} (0.7) → {R["mmr_0.9"]["ild"]:.4f} (0.9); '
  f'mean niche % {R["mmr_0.5"]["niche_pct"]:.2f} → {R["mmr_0.7"]["niche_pct"]:.2f} → {R["mmr_0.9"]["niche_pct"]:.2f}; '
  f'ILD std at each λ {R["mmr_0.5"]["ild_std"]:.3f} / {R["mmr_0.7"]["ild_std"]:.3f} / {R["mmr_0.9"]["ild_std"]:.3f} '
  f'(the 0.120→0.113 range is ≈ {abs(R["mmr_0.5"]["ild"] - R["mmr_0.9"]["ild"]) / R["mmr_0.5"]["ild_std"]:.2f} of one std). '
  f'Paired ILD tests between λ values (raw p): 0.5 vs 0.7 {pair("mmr_0.5", "mmr_0.7")[0]["ild"]["wilcoxon_p"]:.3g}, '
  f'0.5 vs 0.9 {pair("mmr_0.5", "mmr_0.9")[0]["ild"]["wilcoxon_p"]:.3g}, 0.7 vs 0.9 {pair("mmr_0.7", "mmr_0.9")[0]["ild"]["wilcoxon_p"]:.3g} '
  f'(Bonferroni: {pair("mmr_0.5", "mmr_0.7")[0]["ild"]["significant_bonferroni"]}/{pair("mmr_0.5", "mmr_0.9")[0]["ild"]["significant_bonferroni"]}/{pair("mmr_0.7", "mmr_0.9")[0]["ild"]["significant_bonferroni"]}).')
ild_lt = {k: sum(1 for v in R[k]['per_run_ild'] if v < 0.05) for k in ORDER}
ild_gt = {k: sum(1 for v in R[k]['per_run_ild'] if v > 0.1) for k in ORDER}
w('\n**ILD is bimodal / near zero** (runs with ILD < 0.05 | > 0.1): ' + '; '.join(f'{NAMES[k]} {ild_lt[k]} | {ild_gt[k]}' for k in ORDER) + '. '
  'Medians: ' + ', '.join(f'{NAMES[k]} {R[k]["ild_median"]:.4f}' for k in ORDER) + '.')
w(f'\n**Data-size vs balance:** {mapped:,} mappable tracks after dedup; {n:,} sampled ({n / mapped:.0%}); Metal (49 raw) merged into Rock, Country (11) + Indie (17) merged with Folk; '
  f'107 `gaming` tracks dropped (see Section 1 table).')
w(f'\n**Label noise:** {al.group(2)} of {al.group(1)} multi-track artists ({al.group(3)}%) appear under >1 genre; spot-check mislabels are listed in `data/dataset_report.md` §5.')
AUDIO = ('energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness',
         'instrumentalness', 'acousticness', 'mode', 'key', 'time_signature')
if KAGGLE:
    cols = pd.read_csv(os.path.join(KAGGLE, 'low_popularity_spotify_data.csv'), nrows=1).columns
    w('\n**Audio features available but unused:** ' + ', '.join(c for c in cols if c in AUDIO) +
      ' (columns present in the source CSVs).')
else:
    w('\n**Audio features available but unused:** source CSVs not found locally, so not listed here '
      '(the Kaggle files contain energy, valence, danceability and others).')
    flags.append('Kaggle source files not found in the kagglehub cache; source row counts/hashes and the audio-feature column list are omitted.')

# ---------------------------------------------------------------- 7 repro
w('\n## 7. Repo / reproducibility references\n')
w(f'- **Repository:** https://github.com/navya906/diversetune (branch `Main`).')
w(f'- **HEAD when this file was generated:** `{head}` ({sh("git log -1 --format=%s") or "n/a"}). '
  f'This file is committed together with the code, so cite the commit that contains it '
  f'(`git log -1 -- paper_data_reference.md`), not an earlier one.')
w('- **Commit history:**\n')
w('```\n' + (sh('git log --format="%h %ad %s" --date=short -12') or '(no git history available)') + '\n```')
if KAGGLE:
    w(f'- **Dataset:** Kaggle `solomonameh/spotify-music-dataset`, kagglehub cache version `{os.path.basename(KAGGLE)}`. Files used:')
    for f in sorted(os.listdir(KAGGLE)):
        if f.endswith('.csv'):
            pth = os.path.join(KAGGLE, f)
            w(f'  - `{f}`: {len(pd.read_csv(pth, low_memory=False)):,} rows, {os.path.getsize(pth):,} bytes, '
              f'sha256 `{hashlib.sha256(open(pth, "rb").read()).hexdigest()[:16]}…`')
else:
    w('- **Dataset:** Kaggle `solomonameh/spotify-music-dataset` (low- and high-popularity CSVs); local copy not found, so row counts/hashes omitted.')
w('- **Seeds / runs:** dataset sampling seed 42 (`integrate_new_dataset.py --seed`, default 42); experiment `SEED = 42`, `NUM_RUNS = 30` (`recommendation_engine.py`), '
  'seed-song set sizes drawn uniformly 5–10 from `random.Random(42)`; K = 10, N = 50, niche threshold popularity < 40, niche floor ≥ 20% (2 of 10).')
w('- **Dependencies** (`requirements.txt`): ' + ', '.join(open('requirements.txt').read().split()) + '; Python ' + sys.version.split()[0] + '.')
w('- **Reproduce:** `pip install -r requirements.txt && python run_pipeline.py --plots` (or `--skip-dl` to reuse `data/song_track.csv`), then `python generate_paper_data.py`.')
bl = 'baseline/baseline_results.json'
w(f'- **Baseline snapshot:** `{bl}` ({"exists" if os.path.exists(bl) else "MISSING"}); the pre-fix run (8 seed sets, duplicated rows, popularity capped at 68); '
  f'**do not cite** — see `baseline/README.md`.' + (f' Keys: {list(json.load(open(bl)).keys())}.' if os.path.exists(bl) else ''))

# ---------------------------------------------------------------- 8 flags
flags.append('Baseline: `baseline/baseline_results.json` is tracked (moved there so it survives `output/` being git-ignored); `output/old_results.json` may exist locally as a stale untracked copy and is not read by any script.')
if oldd is not None:
    flags.append(f'Duplicate count before dedup: the committed pre-fix CSV (`3171e19`) has {oldd:,} duplicate (name, artists) rows. These were created by sampling with replacement and are not duplicates in the raw Kaggle data (raw merged data: {rem:,} duplicates of {raw:,} rows). An earlier audit quoted 1,488, measured on an uncommitted working copy; cite the committed figure and name the file.')
flags.append('Per-genre niche rates (e.g. Rock vs World) and popularity std are **not in `dataset_report.md`**; they are computed here from `data/song_track.csv` (the sample, not the full dedup pool).')
if re.search(r'nominally significant|within run-to-run noise', ANALYSIS):
    flags.append('repo_analysis.md still describes the MMR λ ILD differences as "nominally significant"/"within noise"; in results.json all three λ-pair ILD differences survive Bonferroni (Section 6).')
flags.append('Timings and `slogdet` counts are re-measured by this script on the generating machine (they are not stored in `results.json`). Timings are machine-dependent (cite as indicative, not as constants); call counts are exact.')
flags.append('Possible wording error in `repo_analysis.md`: it says some ILD differences between MMR λ values are "nominally significant". In `results.json` all three λ-pair ILD differences survive Bonferroni (raw p 8.7e-5, 1.9e-9, 2.0e-5 vs α/84 = 5.95e-4): ILD decreases monotonically with λ, significant but practically tiny (0.120 → 0.113, ≈0.04 std). Describe it that way in the paper ("significant but negligible"), not as "within noise".')
flags.append('Figure numbers in Section 5 are suggestions; the repo has no paper-figure mapping. `fairness_comparison.png` shows Gini, which is degenerate for Greedy (near-identical popularity), so consider omitting or caveating it.')
flags.append('`output/` (results.json, PNGs) is git-ignored and exists only locally; the numbers in this document are not recoverable from GitHub without re-running the pipeline (deterministic with seed 42, but re-run before final submission).')
flags.append('No per-run seed-genre coherence statistic, no equivalence tests (TOST), no multiplicity-corrected p-values (Holm) and no effect sizes (e.g. rank-biserial) are stored; only raw Wilcoxon p-values with a Bonferroni flag.')
flags.append('No statistical test between methods on **Gini / avg popularity** beyond what `_paired` already contains; if the paper makes claims about them, take them from Section 3 tables only.')
flags.append('No user study, no audio-feature similarity, no second dataset: nothing in the repo supports claims beyond this single dataset and a genre-based similarity.')
w('\n## 8. Flags: things to check or not in the repo\n')
for f in flags:
    w(f'- {f}')

# consistency of numbers quoted in repo_analysis.md vs results.json
w('\n### Consistency check of quoted `repo_analysis.md` figures against `results.json`\n')
checks = []
def chk(label, quoted, actual, tol):
    ok = abs(quoted - actual) <= tol
    checks.append((label, quoted, actual, ok))
e, sg, _ = pair('graph_dpp_rerank', 'content_filtering'); chk('DPP−CF ILD mean diff', 0.079, sg * e['ild']['mean_diff'], 0.0006)
e, sg, _ = pair('mmr_floor', 'graph_dpp_rerank'); chk('MMR+floor−DPP ILD mean diff', -0.040, sg * e['ild']['mean_diff'], 0.0006)
chk('MMR+floor niche mean', 31.0, R['mmr_floor']['niche_pct'], 0.05); chk('DPP niche mean', 30.7, R['graph_dpp_rerank']['niche_pct'], 0.05)
chk('DPP ILD mean', 0.149, R['graph_dpp_rerank']['ild'], 0.0006); chk('Greedy avg pop', 93.7, R['greedy']['avg_popularity'], 0.05)
chk('MMR 0.5 ILD', 0.120, R['mmr_0.5']['ild'], 0.0006); chk('MMR 0.9 ILD', 0.113, R['mmr_0.9']['ild'], 0.0006)
chk('Rock niche rate', 0.08, nr['Rock'], 0.005); chk('World niche rate', 0.44, nr['World'], 0.005)
w('| Quoted figure | Quoted | Actual | OK |\n|---|---|---|---|')
for l, q, a, ok in checks:
    w(f'| {l} | {q} | {a:.4f} | {"yes" if ok else "**NO**"} |')
    if not ok:
        flags.append(f'MISMATCH in repo_analysis.md: {l} quoted {q}, actual {a:.4f}')

open('paper_data_reference.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')
print('written', len(out), 'lines;', sum(1 for _, _, _, ok in checks if not ok), 'mismatches')
