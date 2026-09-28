"""
integrate_new_dataset.py
─────────────────────────────────────────────────────────────────────────────
Downloads solomonameh/spotify-music-dataset from Kaggle, cleans it, maps
genres to the project's 12-genre taxonomy, stratified-samples to 3,000 tracks,
and writes data/song_track.csv for the recommendation engine.

Run once before running recommendation_engine.py / server.py.
"""

import kagglehub
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# TARGET 12-GENRE TAXONOMY (original dataset)
# ─────────────────────────────────────────────
GENRE_MAP = {
    # Pop
    "pop": "Pop", "pop-film": "Pop", "k-pop": "Pop", "j-pop": "Pop",
    "power-pop": "Pop", "synth-pop": "Pop", "indie-pop": "Pop",
    "electro": "Pop", "cantopop": "Pop", "mandopop": "Pop",
    "chill": "Pop", "happy": "Pop", "children": "Pop", "gospel": "Pop",
    "comedy": "Pop", "holidays": "Pop", "disney": "Pop",
    "show-tunes": "Pop", "kids": "Pop", "summer": "Pop",
    "movies": "Pop", "romance": "Pop",

    # Hip-Hop
    "hip-hop": "Hip-Hop", "rap": "Hip-Hop", "trap": "Hip-Hop",
    "r-n-b": "Hip-Hop", "soul": "Hip-Hop", "funk": "Hip-Hop",

    # Rock
    "rock": "Rock", "alt-rock": "Rock", "hard-rock": "Rock",
    "punk-rock": "Rock", "punk": "Rock", "grunge": "Rock",
    "emo": "Rock", "garage": "Rock", "goth": "Rock",
    "psych-rock": "Rock", "rock-n-roll": "Rock", "rockabilly": "Rock",
    "road-trip": "Rock",

    # Electronic
    "edm": "Electronic", "electronic": "Electronic", "dance": "Electronic",
    "techno": "Electronic", "house": "Electronic", "deep-house": "Electronic",
    "progressive-house": "Electronic", "trance": "Electronic",
    "dubstep": "Electronic", "drum-and-bass": "Electronic",
    "club": "Electronic", "disco": "Electronic", "dancehall": "Electronic",
    "afrobeat": "Electronic", "party": "Electronic",

    # Classical
    "classical": "Classical", "opera": "Classical", "piano": "Classical",
    "acoustic": "Classical", "new-age": "Classical", "sleep": "Classical",
    "study": "Classical", "ambient": "Classical",

    # Jazz
    "jazz": "Jazz", "blues": "Jazz", "bossanova": "Jazz", "brazil": "Jazz",

    # Country
    "country": "Country", "honky-tonk": "Country", "bluegrass": "Country",

    # Latin
    "latin": "Latin", "salsa": "Latin", "samba": "Latin",
    "reggaeton": "Latin", "spanish": "Latin", "forro": "Latin",
    "pagode": "Latin", "mpb": "Latin", "sertanejo": "Latin",
    "tango": "Latin",

    # Metal
    "metal": "Metal", "heavy-metal": "Metal", "black-metal": "Metal",
    "death-metal": "Metal", "metalcore": "Metal", "grindcore": "Metal",

    # Indie
    "indie": "Indie", "alternative": "Indie", "folk": "Indie",
    "singer-songwriter": "Indie", "british": "Indie", "sad": "Indie",

    # Reggae / World
    "reggae": "Reggae", "world-music": "Reggae", "ska": "Reggae",
    "turkish": "Reggae", "german": "Reggae", "french": "Reggae",
    "swedish": "Reggae", "iranian": "Reggae", "malay": "Reggae",
    "anime": "Reggae", "j-dance": "Reggae", "j-idol": "Reggae",
    "j-rock": "Reggae",
}

TARGET_GENRES = [
    "Pop", "Hip-Hop", "Rock", "Electronic",
    "Classical", "Jazz", "Country", "Latin",
    "Metal", "Indie", "Reggae",
]
TARGET_TOTAL = 3000
MIN_PER_GENRE = 50


def download_dataset():
    print("Downloading solomonameh/spotify-music-dataset from Kaggle ...")
    path = kagglehub.dataset_download("solomonameh/spotify-music-dataset")
    print(f"  Cached at: {path}")

    csv_file = None
    for root, dirs, files in os.walk(path):
        for fname in sorted(files):
            if fname.endswith(".csv"):
                candidate = os.path.join(root, fname)
                if csv_file is None or os.path.getsize(candidate) > os.path.getsize(csv_file):
                    csv_file = candidate

    if csv_file is None:
        raise FileNotFoundError("No CSV found in the downloaded dataset path.")

    print(f"  Using file: {csv_file}")
    return csv_file


def load_and_clean(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"\n--- Raw dataset ---")
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")

    col_lower = {c.lower(): c for c in df.columns}

    def pick(candidates):
        for c in candidates:
            if c in col_lower:
                return col_lower[c]
        return None

    id_col      = pick(["id", "track_id", "spotify_id"])
    name_col    = pick(["name", "track_name", "title"])
    genre_col   = pick(["genre", "track_genre", "genres", "playlist_genre"])
    artists_col = pick(["artists", "artist", "artist_name", "track_artist"])
    album_col   = pick(["album", "album_name", "track_album_name"])
    pop_col     = pick(["popularity", "pop", "track_popularity"])
    dur_col     = pick(["duration_ms", "duration"])
    exp_col     = pick(["explicit"])

    required = {"id": id_col, "name": name_col, "genre": genre_col,
                "artists": artists_col, "popularity": pop_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        print(f"\nWARN: Could not find columns for: {missing}")
        print(f"      Available: {list(df.columns)}")
        raise KeyError(f"Missing required columns: {missing}")

    rename = {}
    if id_col and id_col != "id":               rename[id_col]      = "id"
    if name_col != "name":                      rename[name_col]    = "name"
    if genre_col != "genre":                    rename[genre_col]   = "genre"
    if artists_col != "artists":                rename[artists_col] = "artists"
    if pop_col != "popularity":                 rename[pop_col]     = "popularity"
    if dur_col and dur_col != "duration_ms":    rename[dur_col]     = "duration_ms"
    if exp_col and exp_col != "explicit":       rename[exp_col]     = "explicit"
    if album_col and album_col != "album":      rename[album_col]   = "album"

    if rename:
        df = df.rename(columns=rename)

    keep = ["id", "name", "genre", "artists", "popularity"]
    for c in ["duration_ms", "explicit", "album"]:
        if c in df.columns:
            keep.append(c)

    df = df[[c for c in keep if c in df.columns]].copy()

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["name", "artists"])
    print(f"After dedup (name+artists): {len(df):,}  (dropped {before - len(df):,})")

    # Drop nulls
    df = df.dropna(subset=["genre", "popularity"])
    print(f"After dropping null genre/popularity: {len(df):,}")

    # Coerce types
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df = df.dropna(subset=["popularity"])
    df["popularity"] = df["popularity"].clip(0, 100)

    if "duration_ms" in df.columns:
        df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce").fillna(180000)
    else:
        df["duration_ms"] = 180000

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].map(
            lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes") else 0
        ).fillna(0).astype(int)
    else:
        df["explicit"] = 0

    print(f"\n--- Popularity describe ---")
    print(df["popularity"].describe().round(2))

    print(f"\n--- Raw genre value_counts (top 30) ---")
    print(df["genre"].value_counts().head(30).to_string())

    return df


def map_genres(df):
    df = df.copy()
    df["genre_raw"] = df["genre"].str.strip().str.lower()
    df["genre"] = df["genre_raw"].map(GENRE_MAP)

    unmapped = df[df["genre"].isna()]["genre_raw"].value_counts()
    if not unmapped.empty:
        print(f"\nWARN: Unmapped genres (will be dropped):")
        print(unmapped.head(20).to_string())

    df = df.dropna(subset=["genre"])
    df = df.drop(columns=["genre_raw"])

    print(f"\n--- Genre distribution after mapping ---")
    print(df["genre"].value_counts().to_string())
    print(f"\nTotal mapped tracks: {len(df):,}")
    return df


def stratified_sample(df, total=TARGET_TOTAL):
    genre_counts  = df["genre"].value_counts()
    present_genres = genre_counts.index.tolist()

    proportions = genre_counts / genre_counts.sum()
    alloc = (proportions * total).astype(int)

    remainder = total - alloc.sum()
    for g in genre_counts.index:
        if remainder <= 0:
            break
        alloc[g] += 1
        remainder -= 1

    # Enforce minimum
    for g in present_genres:
        available = len(df[df["genre"] == g])
        if available < MIN_PER_GENRE:
            print(f"  WARN: Genre '{g}' has only {available} tracks (below N={MIN_PER_GENRE}).")
        alloc[g] = max(alloc[g], min(MIN_PER_GENRE, available))

    # Re-normalise if minimums push total over target
    alloc_total = alloc.sum()
    if alloc_total > total:
        excess = int(alloc_total - total)
        for g in genre_counts.index:
            trim = min(int(alloc[g]) - MIN_PER_GENRE, excess)
            if trim > 0:
                alloc[g] -= trim
                excess   -= trim
            if excess <= 0:
                break

    frames = []
    for g in present_genres:
        subset = df[df["genre"] == g]
        n = int(alloc.get(g, 0))
        if n <= 0:
            continue
        replace = n > len(subset)
        sampled = subset.sample(n=n, replace=replace, random_state=42)
        frames.append(sampled)

    result = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n--- Genre distribution after stratified sampling to {total} ---")
    gc = result["genre"].value_counts()
    print(gc.to_string())

    print(f"\n--- N>={MIN_PER_GENRE} candidate check ---")
    for g, cnt in gc.items():
        status = "OK " if cnt >= MIN_PER_GENRE else "LOW"
        print(f"  [{status}]  {g}: {cnt}")

    return result


def build_csv(df):
    df = df.copy()
    df["track_id"] = ["sp_" + str(i) for i in range(len(df))]

    dur_min = df["duration_ms"].min()
    dur_max = df["duration_ms"].max()
    dur_range = dur_max - dur_min
    df["duration_norm"] = (
        (df["duration_ms"] - dur_min) / dur_range
        if dur_range > 0 else 0.0
    )

    out_cols = [
        "track_id", "name", "artists", "genre",
        "popularity", "duration_ms", "duration_norm", "explicit"
    ]
    if "album" in df.columns:
        out_cols.append("album")

    os.makedirs("data", exist_ok=True)
    df[out_cols].to_csv("data/song_track.csv", index=False)
    print(f"\n[OK] Wrote {len(df):,} tracks -> data/song_track.csv")
    print(f"     Columns: {out_cols}")

    print(f"\n--- Popularity stats (final {TARGET_TOTAL}-track sample) ---")
    print(df["popularity"].describe().round(2))

    return df


def main():
    csv_file = download_dataset()
    df = load_and_clean(csv_file)
    df = map_genres(df)
    df = stratified_sample(df, total=TARGET_TOTAL)
    build_csv(df)
    print("\n[DONE] Dataset integration complete. Run recommendation_engine.py next.\n")


if __name__ == "__main__":
    main()
