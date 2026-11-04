import pandas as pd
import numpy as np
from pathlib import Path

def calculate_and_append_cv(
    load_file: str | Path,
    master_file: str | Path,
    output_file: str | Path
):
    print(f"Loading time series data from {load_file}...")
    try:
        # Dynamic index assignment from load profiles
        df_load = pd.read_csv(load_file, sep="\t")
        id_col = df_load.columns[0]
        df_load.set_index(id_col, inplace=True)
    except Exception as e:
        print(f"Error loading time series: {e}")
        return

    print("Calculating Means and Standard Deviations...")
    # Calculate row-wise mean and sample standard deviation
    means = np.nanmean(df_load.values, axis=1)
    stds = np.nanstd(df_load.values, axis=1, ddof=1)

    print("Computing Coefficient of Variation (CV)...")
    cv_values = np.divide(
        stds, 
        means, 
        out=np.zeros_like(stds, dtype=float), 
        where=means != 0
    )

    df_cv = pd.DataFrame({
        'Id': df_load.index.astype(str),
        'CV': cv_values
    })

    print(f"Loading master data from {master_file}...")
    try:
        df_master = pd.read_csv(master_file, sep="\t") 
    except Exception as e:
        print(f"Error loading master data: {e}")
        return

    df_master['Id'] = df_master['Id'].astype(str)

    print("Merging CV into master data...")
    df_merged = df_master.merge(df_cv, on='Id', how='left')

    df_merged.to_csv(output_file, index=False, sep="\t") 
    print(f"Success! Enriched master data saved to {output_file}")

if __name__ == "__main__":
    calculate_and_append_cv(
        load_file=r'example\Path\data.tsv',
        master_file=r'example\Path\data.csv',
        output_file=r'example\Path\data.csv'
    )