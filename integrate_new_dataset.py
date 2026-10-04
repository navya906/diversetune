"""
integrate_new_dataset.py
─────────────────────────────────────────────────────────────────────────────
Builds data/song_track.csv for the recommendation engine from the Kaggle
dataset solomonameh/spotify-music-dataset.

  1. Loads BOTH files shipped with the dataset (low + high popularity) so the
     popularity range is the full 11-100 instead of the low file's 11-68 cap.
  2. Maps the dataset's `playlist_genre` values into a 10-genre taxonomy.
  3. Deduplicates on normalised (name, artists), keeping the most popular row.
  4. Samples up to 3,000 tracks WITHOUT replacement, guaranteeing a floor of
     MIN_PER_GENRE tracks per genre (the script fails loudly if it cannot).
  5. Writes data/song_track.csv (with playlist_name / genre_raw provenance
     columns) and data/dataset_report.md.
"""

import kagglehub
import pandas as pd
import numpy as np
import os
import argparse
import sys

# ─────────────────────────────────────────────
# TARGET 10-GENRE TAXONOMY
# Keys are the raw `playlist_genre` values of the source CSVs.
# Genres that could not reach the 50-track floor on their own were merged:
#   Metal (49 raw)  -> Rock;  Country (11) + Indie (17) -> Indie-Folk with Folk.
# `gaming` is a use-case playlist, not a genre, and is dropped.
# ─────────────────────────────────────────────
GENRE_MAP = {
    "pop": "Pop", "k-pop": "Pop", "j-pop": "Pop", "cantopop": "Pop",
    "mandopop": "Pop", "korean": "Pop",

    "hip-hop": "Hip-Hop", "r&b": "Hip-Hop", "soul": "Hip-Hop",
    "funk": "Hip-Hop", "gospel": "Hip-Hop",

    "rock": "Rock", "punk": "Rock", "metal": "Rock",

    "electronic": "Electronic", "disco": "Electronic",

    "classical": "Classical",

    "jazz": "Jazz", "blues": "Jazz",

    "latin": "Latin", "brazilian": "Latin",

    "ambient": "Ambient", "lofi": "Ambient", "wellness": "Ambient",

    "world": "World", "arabic": "World", "turkish": "World",
    "indian": "World", "afrobeats": "World", "reggae": "World",
    "soca": "World",

    "folk": "Indie-Folk", "indie": "Indie-Folk", "country": "Indie-Folk",
}

TARGET_GENRES = [
    "Pop", "Hip-Hop", "Rock", "Electronic", "Classical",
    "Jazz", "Latin", "Ambient", "World", "Indie-Folk",
]
TARGET_TOTAL = 3000
MIN_PER_GENRE = 50


def download_dataset():
    """Return every CSV in the Kaggle dataset (low- AND high-popularity files)."""
    print("Downloading solomonameh/spotify-music-dataset from Kaggle ...")
    path = kagglehub.dataset_download("solomonameh/spotify-music-dataset")
    print(f"  Cached at: {path}")
    csvs = []
    for root, _, files in os.walk(path):
        csvs += [os.path.join(root, f) for f in sorted(files) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError("No CSV found in the downloaded dataset path.")
    for c in csvs:
        print(f"  Using file: {c}")
    return csvs


def load_raw(csv_files):
    """Concatenate all source CSVs into one frame with the project's column names."""
    frames = []
    for f in csv_files:
        d = pd.read_csv(f, low_memory=False)
        print(f"  {os.path.basename(f)}: {d.shape[0]:,} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    rename = {"track_name": "name", "track_artist": "artists",
              "track_album_name": "album", "track_popularity": "popularity",
              "playlist_genre": "genre"}
    missing = [c for c in list(rename) + ["track_id", "duration_ms"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df = df.rename(columns=rename)
    if "explicit" not in df.columns:
        df["explicit"] = 0
    if "playlist_name" not in df.columns:
        df["playlist_name"] = ""
    return df


def _norm(series):
    return series.astype(str).str.lower().str.strip()


def clean_and_dedupe(df):
    """Dedupe on normalised (name, artists), keeping the most popular row."""
    df = df.copy()
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df = df.dropna(subset=["name", "artists", "genre", "popularity"])
    df["popularity"] = df["popularity"].clip(0, 100)
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce").fillna(180000)
    df["explicit"] = df["explicit"].map(
        lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes") else 0).astype(int)

    before = len(df)
    df["_k1"], df["_k2"] = _norm(df["name"]), _norm(df["artists"])
    # popularity desc, then track_id asc -> deterministic choice among ties
    df = df.sort_values(["popularity", "track_id"], ascending=[False, True])
    df = df.drop_duplicates(subset=["_k1", "_k2"], keep="first")
    df = df.drop(columns=["_k1", "_k2"])
    dupes = before - len(df)
    print(f"After dedup on (name, artists): {len(df):,}  (dropped {dupes:,})")
    return df, dupes


def map_genres(df):
    df = df.copy()
    df["genre_raw"] = _norm(df["genre"])
    df["genre"] = df["genre_raw"].map(GENRE_MAP)

    audit = []
    for raw in sorted(df["genre_raw"].unique()):
        sub = df[df["genre_raw"] == raw]
        audit.append({
            "raw_genre": raw,
            "mapped_to": GENRE_MAP.get(raw, "DROPPED"),
            "n_tracks": len(sub),
            "sample_artists": " | ".join(sub["artists"].dropna().unique()[:5]),
        })
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(audit).to_csv("data/genre_mapping.csv", index=False)
    print("[Audit] Wrote data/genre_mapping.csv")

    dropped = df[df["genre"].isna()]["genre_raw"].value_counts()
    if not dropped.empty:
        print("Dropped (unmapped) raw genres:", dropped.to_dict())
    return df.dropna(subset=["genre"])


def allocate(avail, total, floor):
    """Per-genre sample sizes: proportional, never above availability, >= floor."""
    avail = avail.astype(int)
    short = avail[avail < floor]
    if not short.empty:
        raise ValueError(f"Genres below the {floor}-track floor: {short.to_dict()}. "
                         "Merge or drop them in GENRE_MAP.")
    if avail.sum() <= total:
        return avail.copy()
    alloc = pd.Series(floor, index=avail.index)
    # distribute the remainder proportionally to spare capacity (largest remainder)
    spare = avail - floor
    quota = spare / spare.sum() * (total - alloc.sum())
    alloc += np.floor(quota).astype(int)
    rem = int(total - alloc.sum())
    for g in (quota - np.floor(quota)).sort_values(ascending=False).index[:rem]:
        alloc[g] += 1
    return alloc


def stratified_sample(df, seed=42, total=TARGET_TOTAL):
    alloc = allocate(df["genre"].value_counts(), total, MIN_PER_GENRE)
    frames = [df[df["genre"] == g].sample(n=int(n), replace=False, random_state=seed)
              for g, n in alloc.items()]
    result = pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"\n--- Genre distribution after sampling (no replacement, floor {MIN_PER_GENRE}) ---")
    print(result["genre"].value_counts().to_string())
    return result


def build_csv(df):
    df = df.copy().sort_values(["genre", "name", "artists"]).reset_index(drop=True)
    df["track_id"] = ["sp_" + str(i) for i in range(len(df))]
    rng = df["duration_ms"].max() - df["duration_ms"].min()
    df["duration_norm"] = (df["duration_ms"] - df["duration_ms"].min()) / rng if rng > 0 else 0.0
    out_cols = ["track_id", "name", "artists", "genre", "popularity",
                "duration_ms", "duration_norm", "explicit", "album",
                "playlist_name", "genre_raw"]
    os.makedirs("data", exist_ok=True)
    df[out_cols].to_csv("data/song_track.csv", index=False)
    print(f"\n[OK] Wrote {len(df):,} tracks -> data/song_track.csv")
    return df


def verify(final):
    """Hard checks on the written CSV; the report records these numbers."""
    stats = {
        "rows": len(final),
        "dup_name_artist": int(final.assign(a=_norm(final["name"]), b=_norm(final["artists"]))
                               .duplicated(["a", "b"]).sum()),
        "dup_track_id": int(final["track_id"].duplicated().sum()),
        "min_genre": int(final["genre"].value_counts().min()),
    }
    assert stats["dup_name_artist"] == 0, "duplicate (name, artists) rows remain"
    assert stats["dup_track_id"] == 0
    assert stats["min_genre"] >= MIN_PER_GENRE, "genre floor violated"
    return stats


def write_dataset_report(csvs, n_raw, dupes, n_dedup, n_mapped, final, stats, seed):
    """Dataset facts only; quality findings are appended by diagnostic.py."""
    pop = final["popularity"]
    with open("data/dataset_report.md", "w", encoding="utf-8") as f:
        f.write("# Dataset Report\n\n")
        f.write("_Generated by `integrate_new_dataset.py`; the quality section is "
                "appended by `diagnostic.py`._\n\n")
        f.write("## Source and pipeline\n")
        f.write("- Source: Kaggle `solomonameh/spotify-music-dataset`, files: "
                + ", ".join(f"`{os.path.basename(c)}`" for c in csvs) + "\n")
        f.write(f"- Sampling seed: {seed}\n")
        f.write(f"- Raw rows (all files): {n_raw:,}\n")
        f.write(f"- Removed as duplicate (name, artists): {dupes:,}\n")
        f.write(f"- After dedup: {n_dedup:,}\n")
        f.write(f"- After genre mapping (unmapped genres dropped): {n_mapped:,}\n")
        f.write(f"- Final sample (without replacement): {stats['rows']:,}\n\n")
        f.write("## Checks on `data/song_track.csv`\n")
        f.write(f"- Duplicate (name, artists) rows: {stats['dup_name_artist']}\n")
        f.write(f"- Duplicate track_id rows: {stats['dup_track_id']}\n")
        f.write(f"- Smallest genre: {stats['min_genre']} tracks (floor = {MIN_PER_GENRE})\n\n")
        f.write("## Genre distribution\n| Genre | Count |\n|---|---|\n")
        for g, c in final["genre"].value_counts().items():
            f.write(f"| {g} | {c} |\n")
        f.write("\n## Popularity\n")
        f.write(f"- Min: {pop.min():.0f}\n- Max: {pop.max():.0f}\n")
        f.write(f"- Mean: {pop.mean():.2f}\n- Median: {pop.median():.0f}\n")
        f.write(f"- Share below 40 (the engine's niche threshold): {(pop < 40).mean():.1%}\n")


def main(seed=42):
    csvs = download_dataset()
    raw = load_raw(csvs)
    n_raw = len(raw)
    df, dupes = clean_and_dedupe(raw)
    n_dedup = len(df)
    df = map_genres(df)
    n_mapped = len(df)
    df = stratified_sample(df, seed=seed)
    build_csv(df)

    final = pd.read_csv("data/song_track.csv")
    stats = verify(final)
    write_dataset_report(csvs, n_raw, dupes, n_dedup, n_mapped, final, stats, seed)
    print("\n[DONE] Dataset integration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    main(seed=args.seed)
