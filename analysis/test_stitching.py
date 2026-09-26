import pandas as pd
from pathlib import Path
import logging
import numpy as np

def load_and_stitch_predictions(base_output_dir: str | Path) -> pd.DataFrame:
    base_path = Path(base_output_dir)
    all_batches = []
    
    # 1. Recursively find all Parquet files inside the base directory
    parquet_files = list(base_path.rglob("*.parquet"))
    
    if not parquet_files:
        logging.warning(f"No parquet files found in {base_path}")
        return pd.DataFrame()

    logging.info(f"Found {len(parquet_files)} batch files. Starting aggregation...")

    for file_path in parquet_files:
        # Extract context from the directory tree 
        # e.g., .../c1_lag7_horizon7_step7/google_cnn/batch_0.parquet
        model_name = file_path.parent.name
        combo_name = file_path.parent.parent.name
        
        # Read the file using the robust PyArrow engine
        df_batch = pd.read_parquet(file_path, engine="pyarrow")
        
        # Inject the metadata directly into the DataFrame
        df_batch["model"] = model_name
        df_batch["combo"] = combo_name
        
        all_batches.append(df_batch)

    # 2. Vertical Concatenation
    # Stacks all companies, models, and configurations into one massive table
    full_df = pd.concat(all_batches, ignore_index=True)
    
    # 3. Stitching and Chronological Sorting
    # This guarantees that the timeline for every single company is perfectly ordered
    full_df.sort_values(
        by=["combo", "model", "id", "horizon_start"], 
        ascending=[True, True, True, True],
        inplace=True
    )
    
    full_df.reset_index(drop=True, inplace=True)
    logging.info(f"Successfully stitched {len(full_df)} total prediction horizons.")
    
    return full_df


def inspect_stitched_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        print("DataFrame is empty.")
        return

    print("=" * 60)
    print("STITCHED DATA QUALITY & STRUCTURE AUDIT")
    print("=" * 60)
    print(f"Total Rows (Horizons): {len(df):,}")
    print(f"Total Columns:         {len(df.columns)}")
    print(f"Columns:               {list(df.columns)}")
    print(f"Memory Usage:          {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print("-" * 60)

    # 1. Distinct Entity Counts
    combos = df["combo"].unique().tolist()
    models = df["model"].unique().tolist()
    n_companies = df["id"].nunique()

    print(f"Combinations ({len(combos)}): {combos}")
    print(f"Models ({len(models)}):       {models}")
    print(f"Unique Company IDs:  {n_companies:,}")
    print("-" * 60)

    # 2. Horizon Sequence Length Audit (List validation)
    # Check if actual and predicted arrays match expected horizon sizes
    df["_actual_len"] = df["actual_values"].apply(lambda v: len(v) if hasattr(v, "__len__") else -1)
    df["_pred_len"] = df["predicted_values"].apply(lambda v: len(v) if hasattr(v, "__len__") else -1)

    print("Vector Length Distribution:")
    for combo in combos:
        sub = df[df["combo"] == combo]
        actual_lens = sub["_actual_len"].unique()
        pred_lens = sub["_pred_len"].unique()
        print(f"  [{combo}]")
        print(f"    - actual_values length(s):    {actual_lens}")
        print(f"    - predicted_values length(s): {pred_lens}")

    # Check for mismatches between actuals and predictions
    mismatches = (df["_actual_len"] != df["_pred_len"]).sum()
    null_counts = df[["id", "combo", "model", "horizon_start", "actual_values", "predicted_values"]].isnull().sum()

    print("-" * 60)
    print(f"Array Length Mismatches: {mismatches}")
    print("Missing / Null Values per Column:")
    for col, count in null_counts.items():
        print(f"  - {col}: {count}")

    # Clean up temporary inspection columns
    df.drop(columns=["_actual_len", "_pred_len"], inplace=True)
    print("=" * 60)


def export_combo_to_tsv(df: pd.DataFrame, target_combo: str, output_tsv_path: str | Path) -> None:
    output_path = Path(output_tsv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Filter strictly to the desired combination
    sub_df = df[df["combo"] == target_combo].copy()
    if sub_df.empty:
        available = df["combo"].unique().tolist()
        raise ValueError(f"Combination '{target_combo}' not found in data. Available: {available}")

    # 2. Serialize vector lists into compact comma-separated strings for TSV formatting
    # e.g., [0.24, 0.51, ...] -> "0.24,0.51,..."
    sub_df["actual_values"] = sub_df["actual_values"].apply(
        lambda vals: ",".join(map(str, vals)) if hasattr(vals, "__iter__") else str(vals)
    )
    sub_df["predicted_values"] = sub_df["predicted_values"].apply(
        lambda vals: ",".join(map(str, vals)) if hasattr(vals, "__iter__") else str(vals)
    )

    # 3. Write tab-delimited file
    sub_df.to_csv(output_path, sep="\t", index=False)
    print(f"Exported {len(sub_df):,} rows for combination '{target_combo}' to: {output_path.resolve()}")


def append_metrics(df):
    # Calculate MSE and MAE for each individual horizon window
    mses, maes = [], []
    
    for actuals, preds in zip(df['actual_values'], df['predicted_values']):
        act_arr = np.array(actuals)
        pred_arr = np.array(preds)
        
        # Calculate raw errors
        errors = act_arr - pred_arr
        
        mses.append(np.mean(errors ** 2))
        maes.append(np.mean(np.abs(errors)))
        
    df['horizon_mse'] = mses
    df['horizon_mae'] = maes
    
    return df


def calculate_wins(master_df):
    # 1. Aggregate both MAE and MSE at the company level
        company_metrics = master_df.groupby(['id', 'combo', 'model'])[['horizon_mae', 'horizon_mse']].mean().reset_index()
    
        # 2. Global summary: Average MAE and MSE across ALL companies per combo & model
        global_summary = company_metrics.groupby(['combo', 'model'])[['horizon_mae', 'horizon_mse']].mean().reset_index()
        print("--- Global Average Performance ---")
        print(global_summary.sort_values(by=['horizon_mae', 'horizon_mse']))#['combo', 'horizon_mae']))
    
        # 3. Pivot for direct company-level comparison 
        # This creates multi-level columns (e.g., 'horizon_mae' -> 'model_A', 'model_B', etc.)
        comparison_df = company_metrics.pivot(
            index=['id', 'combo'], 
            columns='model', 
            values=['horizon_mae', 'horizon_mse']
        )
        
        # Flatten the MultiIndex columns for easier querying
        comparison_df.columns = [f"{metric}_{model}" for metric, model in comparison_df.columns]
        comparison_df.reset_index(inplace=True)
    
        # 4. Determine the "Winner" (lowest MAE) for each company and combination
        # Isolate just the MAE columns to find the minimum across the 5 models
        mae_cols = [col for col in comparison_df.columns if col.startswith('horizon_mae_')]
        
        # idxmin returns the column name with the lowest value, we strip 'horizon_mae_' to get the model name
        comparison_df['best_model'] = comparison_df[mae_cols].idxmin(axis=1).str.replace('horizon_mae_', '')
    
        # 5. Output the Win Rates
        print("\n--- Company-Level Win Rate (Lowest MAE) ---")
        for combo in comparison_df['combo'].unique():
            combo_data = comparison_df[comparison_df['combo'] == combo]
            wins = combo_data['best_model'].value_counts()
            total = len(combo_data)
            
            print(f"\nWin distribution for {combo} (Total Companies: {total}):")
            for model_name, count in wins.items():
                print(f"  - {model_name}: {count} wins ({count/total*100:.1f}%)")


if __name__ == "__main__":
    PREDICTION_DIR = r"C:\Users\lsb-ps\Documents\predictions_overview_lean\4_HistoricAvg\small_historicAvg_prediction_output\c1_lag1_horizon1_step1"
    #PREDICTION_DIR = r"C:\Users\lsb-ps\Documents\predictions_overview_lean\5_DLinear\small_dlinear_prediction_output\c1_lag1_horizon1_step1"
    #PREDICTION_DIR = r"H:\model_Timer\small_timer_prediction_output"


    master_df = load_and_stitch_predictions(PREDICTION_DIR)
    print(master_df.head())

    inspect_stitched_dataframe(master_df)

    #master_df = append_metrics(master_df)

    #calculate_wins(master_df)

    #export_combo_to_tsv(df=master_df, target_combo="c1_lag1_horizon1_step1", output_tsv_path="./analysis_c1_lag1.tsv")
