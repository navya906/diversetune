"""
integrate_kaggle.py
─────────────────────────────────────────────────────────────────────────────
Legacy entry-point for dataset integration.

Previously used the algozee/spotyfy dataset with old column names:
  track_name, track_genre, artist_name, streams, days, pos, audio features

Now updated for the solomonameh/spotify-music-dataset with new column schema:
  name, genre, artists, popularity, duration_ms, explicit, album

All integration logic lives in integrate_new_dataset.py.
This script delegates to it so any old call-sites continue to work.

Usage:
    python integrate_kaggle.py
"""

import integrate_new_dataset

if __name__ == "__main__":
    integrate_new_dataset.main()

