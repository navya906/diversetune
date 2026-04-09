import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("algozee/spotyfy")
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith('.csv'):
            file_path = os.path.join(root, f)
            df = pd.read_csv(file_path)
            print("COLUMNS:")
            print(", ".join(df.columns.tolist()))
            break
