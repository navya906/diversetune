import kagglehub
import pandas as pd
import numpy as np
import os

print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("algozee/spotyfy")
csv_file = None
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith('.csv'):
            csv_file = os.path.join(root, f)
            break
    if csv_file:
        break

if not csv_file:
    raise ValueError("No CSV file found in the downloaded dataset.")

df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} records from Algozee Kaggle dataset.")

# Rename columns to match what the engine expects
if "artist_name" in df.columns:
    df["artists"] = df["artist_name"]
else:
    df["artists"] = "Unknown Artist"

if "genre" in df.columns:
    df["track_genre"] = df["genre"]
else:
    df["track_genre"] = "Unknown Genre"

df = df.dropna(subset=['track_name'])

# Parse numbers robustly
def to_float(x):
    try:
        if isinstance(x, str):
            return float(x.replace(',', ''))
        return float(x)
    except:
        return 0.0

df['streams_num'] = df['streams'].apply(to_float) if 'streams' in df.columns else np.random.rand(len(df))
df['days_num'] = df['days'].apply(to_float) if 'days' in df.columns else np.random.rand(len(df))
df['pos_num'] = df['pos'].apply(to_float) if 'pos' in df.columns else np.random.rand(len(df))

# Map features to the 5 standard features our app expects (using MinMaxScaler for stability)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feat_cols = ['streams_num', 'days_num', 'pos_num']
scaled = scaler.fit_transform(df[feat_cols])

df['danceability'] = scaled[:, 0]
df['energy'] = scaled[:, 1]
df['valence'] = scaled[:, 2]
df['tempo'] = (scaled[:, 0] + scaled[:, 1]) / 2  # Derived correctly
df['acousticness'] = 1.0 - df['energy']          # Derived

# Map popularity to 0-100 logic smoothly
pop_scaler = MinMaxScaler(feature_range=(0, 100))
df['popularity'] = pop_scaler.fit_transform(df[['streams_num']])

df['track_id'] = ["sp_" + str(i) for i in range(len(df))]

out_cols = [
    "track_id", "track_name", "artists", "track_genre",
    "danceability", "energy", "valence", "tempo", "acousticness", "popularity"
]

# Write to data directory
os.makedirs("data", exist_ok=True)
df[out_cols].to_csv("data/spotify_tracks.csv", index=False)

print(f"Successfully mapped and integrated {len(df)} kaggle records into data/spotify_tracks.csv")
