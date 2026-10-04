"""
diagnostic.py
Audits genre labelling quality of data/song_track.csv and appends the findings
to data/dataset_report.md (replacing any previous "Quality diagnostics" section).

Reads only the project CSV (which carries `playlist_name` and `genre_raw`
provenance columns), so no machine-specific paths are needed.

Usage:  python diagnostic.py
"""

import pandas as pd

CSV = "data/song_track.csv"
REPORT = "data/dataset_report.md"
SECTION = "## Quality diagnostics"
# Artists whose tracks were mis-genred in the pre-fix dataset.
SPOT_CHECK_ARTISTS = ["Burna Boy", "Zinoleesky", "Michael Jackson", "Céline Dion", "Polo G"]


def run_diagnostics():
    df = pd.read_csv(CSV)
    out = [SECTION, "", f"Computed on `{CSV}` ({len(df):,} tracks).", ""]

    # 1. Completeness
    missing = int(df["playlist_name"].isna().sum())
    out += ["### 1. Completeness",
            f"- Missing `playlist_name`: {missing} / {len(df)}",
            f"- Distinct playlists: {df['playlist_name'].nunique()}", ""]

    # 2. Exact agreement between playlist_name and raw playlist_genre
    name_c = df["playlist_name"].astype(str).str.lower().str.strip()
    raw_c = df["genre_raw"].astype(str).str.lower().str.strip()
    exact = int((name_c == raw_c).sum())
    out += ["### 2. playlist_name vs playlist_genre (exact match)",
            f"- Exact matches: {exact} / {len(df)} ({exact / len(df):.2%})",
            "- Playlist names are free-text marketing titles (e.g. 'Meditative Vibes'), "
            "so a near-zero exact-match rate is expected and is NOT evidence of bad labels. "
            "Check 3 is the meaningful test.", ""]

    # 3. Within-playlist consistency of the mapped genre (purity)
    g = df.groupby("playlist_name")["genre"]
    sizes = g.size()
    top_share = g.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    big = sizes[sizes >= 10].index
    weighted = (top_share[big] * sizes[big]).sum() / sizes[big].sum()
    out += ["### 3. Playlist purity (share of a playlist's tracks in its majority mapped genre)",
            "- NOTE: this is true by construction (every track inherits its playlist's single "
            "`playlist_genre`), so it verifies the mapping code, not label accuracy. "
            "Checks 5 and 6 measure label accuracy.",
            f"- Track-weighted purity over playlists with >= 10 tracks ({len(big)} playlists): "
            f"{weighted:.1%}",
            f"- Playlists with purity < 70%: {int((top_share[big] < 0.70).sum())} of {len(big)}", ""]
    low = top_share[big][top_share[big] < 0.70].sort_values().head(10)
    if len(low):
        out += ["Least pure playlists:", "", "| Playlist | Tracks | Purity | Genres present |", "|---|---|---|---|"]
        for name, p in low.items():
            mix = df[df["playlist_name"] == name]["genre"].value_counts().head(3)
            out.append(f"| {name} | {sizes[name]} | {p:.0%} | "
                       + ", ".join(f"{k} ({v})" for k, v in mix.items()) + " |")
        out.append("")

    # 4. Raw-genre -> mapped-genre sanity: raw genres spread over several mapped genres
    #    would signal a mapping collision (should be none, mapping is a function).
    spread = df.groupby("genre_raw")["genre"].nunique()
    out += ["### 4. Raw genre -> mapped genre",
            f"- Raw genres present: {df['genre_raw'].nunique()}; "
            f"raw genres mapped to more than one genre: {int((spread > 1).sum())}", ""]

    # 6 (computed here, printed after 5): artists whose tracks carry several genres
    per_artist = df.groupby("artists")["genre"].agg(["nunique", "size"])
    multi = per_artist[(per_artist["size"] >= 2)]
    cross = int((multi["nunique"] > 1).sum())

    # 5. Spot check of previously mis-genred artists
    out += ["### 5. Spot check of previously mis-genred artists", "",
            "| Artist | Track | Raw genre | Mapped genre | Playlist |", "|---|---|---|---|---|"]
    found = 0
    for artist in SPOT_CHECK_ARTISTS:
        sub = df[df["artists"].astype(str).str.contains(artist, case=False, na=False, regex=False)]
        for _, r in sub.head(3).iterrows():
            found += 1
            out.append(f"| {artist} | {r['name']} | {r['genre_raw']} | {r['genre']} | {r['playlist_name']} |")
    if not found:
        out.append("| _none of these artists are in the final sample_ | | | | |")
    out.append("")

    out += ["### 6. Artist-level label consistency",
            f"- Artists with >= 2 tracks: {len(multi)}; of these, {cross} "
            f"({cross / max(len(multi), 1):.1%}) have tracks in more than one genre.",
            "", "### Verdict",
            "- Duplicates and the genre floor are fixed and verified by `verify()` in the pipeline.",
            "- Genre is the genre of the *playlist* a track was scraped from, not an artist- or "
            "track-level label. The spot check shows clear mislabels (e.g. Burna Boy under "
            "Ambient, Michael Jackson under Jazz via a 'Classic Blues' playlist). The genre "
            "one-hot is therefore a noisy proxy for musical style; see Limitations.", ""]

    text = "\n".join(out)
    print(text)

    with open(REPORT, encoding="utf-8") as f:
        report = f.read()
    if SECTION in report:
        report = report[:report.index(SECTION)].rstrip() + "\n\n"
    else:
        report = report.rstrip() + "\n\n"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report + text + "\n")
    print(f"\n[OK] Appended findings to {REPORT}")


if __name__ == "__main__":
    run_diagnostics()
