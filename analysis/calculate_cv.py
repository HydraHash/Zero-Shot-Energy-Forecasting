import pandas as pd
import numpy as np
from pathlib import Path

def calculate_and_append_cv_and_repeatability(
    load_file: str | Path,
    master_file: str | Path,
    output_file: str | Path
):
    print(f"Loading master data from {master_file}...")
    try:
        df_master = pd.read_csv(master_file, sep="\t") 
    except Exception as e:
        print(f"Error loading master data: {e}")
        return

    df_master['Id'] = df_master['Id'].astype(str)
    master_ids = set(df_master['Id'])

    print(f"Loading time series data from {load_file}...")
    try:
        df_load = pd.read_csv(load_file, sep="\t", index_col=0)
        df_load.index = df_load.index.astype(str)
    except Exception as e:
        print(f"Error loading time series: {e}")
        return

    # Restrict to master IDs if applicable
    df_load_matched = df_load.loc[df_load.index.isin(master_ids)] if master_ids else df_load
    load_values = df_load_matched.values[:, :35136]

    print("Calculating Means and Standard Deviations for CV...")
    means = np.nanmean(load_values, axis=1)
    stds = np.nanstd(load_values, axis=1, ddof=1)

    print("Computing Coefficient of Variation (CV)...")
    cv_values = np.divide(
        stds, 
        means, 
        out=np.zeros_like(stds, dtype=float), 
        where=means != 0
    )

    print("Computing Daily Intraday Repeatability Scores (Workdays)...")
    time_axis = pd.date_range(start="2016-01-01 00:00:00", periods=35136, freq="15min")
    weekdays = time_axis.strftime("%A")
    workday_mask = ~np.isin(weekdays, ["Saturday", "Sunday"])
    
    # 261 workdays x 96 fifteen-minute intervals
    workday_values = load_values[:, workday_mask]
    n_companies = len(df_load_matched)
    n_workdays = np.sum(workday_mask) // 96
    
    arr = workday_values.reshape(n_companies, n_workdays, 96)
    mins = np.nanmin(arr, axis=2, keepdims=True)
    maxs = np.nanmax(arr, axis=2, keepdims=True)
    ptp = maxs - mins
    ptp[ptp == 0] = 1.0
    norm_arr = (arr - mins) / ptp

    norm_arr_centered = norm_arr - np.nanmean(norm_arr, axis=2, keepdims=True)
    norms = np.sqrt(np.nansum(norm_arr_centered**2, axis=2, keepdims=True))
    norms[norms == 0] = 1.0
    norm_arr_normed = norm_arr_centered / norms

    corr_matrices = np.matmul(norm_arr_normed, norm_arr_normed.transpose(0, 2, 1))
    triu_indices = np.triu_indices(n_workdays, k=1)
    upper_triangles = corr_matrices[:, triu_indices[0], triu_indices[1]]
    repeatability_scores = np.nanmedian(upper_triangles, axis=1)

    df_enrichment = pd.DataFrame({
        'Id': df_load_matched.index.astype(str),
        'CV': cv_values,
        'repeatability_score': np.round(repeatability_scores, 6)
    })

    print("Merging metrics into master data...")
    # Drop existing columns if present to avoid duplication
    cols_to_drop = [c for c in ['CV', 'repeatability_score'] if c in df_master.columns]
    if cols_to_drop:
        df_master.drop(columns=cols_to_drop, inplace=True)

    df_merged = df_master.merge(df_enrichment, on='Id', how='left')

    df_merged.to_csv(output_file, index=False, sep="\t") 
    print(f"Success! Enriched master data saved to {output_file}")


calculate_and_append_cv = calculate_and_append_cv_and_repeatability

if __name__ == "__main__":
    calculate_and_append_cv(
        load_file=r"path/to/load_profiles_enriched.tsv",
        master_file=r"path/to/master_data_imputed.csv",
        output_file=r"path/to/master_data_cv_repeat.csv"
    )