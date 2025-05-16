import os
import pandas as pd
from glob import glob

csv_files = glob('./2005_2024/*build*.csv')

dataframes = [pd.read_csv(file) for file in csv_files]

all_columns = sorted(set(col for df in dataframes for col in df.columns))

# Ensure all dataframes have the same columns, filling missing ones with False
for i, df in enumerate(dataframes):
    missing_cols = [col for col in all_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = False  # Fill missing columns with False

combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_csv('./2005_2024/All_prefectures_buildings.csv', index=False)

print(f"Combined {len(csv_files)} files into 'All_prefectures_buildings.csv'")
