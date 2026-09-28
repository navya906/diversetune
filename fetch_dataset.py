"""
fetch_dataset.py — Quick inspection utility for the new Kaggle dataset.
Downloads solomonameh/spotify-music-dataset and prints column names + shape.
Run this first to verify the download works before running integrate_new_dataset.py.
"""

import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("solomonameh/spotify-music-dataset")
print(f"Dataset cached at: {path}")

for root, dirs, files in os.walk(path):
    for f in sorted(files):
        if f.endswith(".csv"):
            file_path = os.path.join(root, f)
            df = pd.read_csv(file_path, nrows=5)
            print(f"\nFile: {f}")
            print("COLUMNS:", ", ".join(df.columns.tolist()))
            print("SHAPE  :", pd.read_csv(file_path, low_memory=False).shape)
            print(df.head(2).to_string())
            break
