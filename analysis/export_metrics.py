import ast
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import gc


def append_metrics(df: pd.DataFrame) -> pd.DataFrame:
    mses, maes, smapes, biases, rmses, das = [], [], [], [], [], []
    
    for actuals, preds in zip(df['actual_values'], df['predicted_values']):
        # Parse string representations back into Python lists if necessary
        if isinstance(actuals, str):
            actuals = ast.literal_eval(actuals)
        if isinstance(preds, str):
            preds = ast.literal_eval(preds)
            
        # Convert to float arrays for mathematical operations
        act_arr = np.array(actuals, dtype=float)
        pred_arr = np.array(preds, dtype=float)
        
        errors = act_arr - pred_arr
        
        # Core & Scale-Free Metrics
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mse)
        bias = np.mean(pred_arr - act_arr)
        
        # sMAPE computation
        denominator = (np.abs(act_arr) + np.abs(pred_arr)) / 2
        smape = np.mean(np.divide(np.abs(errors), denominator, out=np.zeros_like(errors, dtype=float), where=denominator!=0)) * 100
        
        # Directional Accuracy
        if len(act_arr) > 1:
            da = np.mean(np.sign(np.diff(act_arr)) == np.sign(np.diff(pred_arr))) * 100
        else:
            da = np.nan # Undefined for single-step horizons
        
        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)
        biases.append(bias)
        smapes.append(smape)
        das.append(da)
        
    df['horizon_mse'] = mses
    df['horizon_mae'] = maes
    df['horizon_rmse'] = rmses
    df['horizon_bias'] = biases
    df['horizon_smape'] = smapes
    df['horizon_da'] = das

    return df


def process_and_export_batches(base_output_dir: str | Path):
    base_path = Path(base_output_dir)
    
    # Group parquet files by their immediate parent directory
    batch_groups = defaultdict(list)
    parquet_files = list(base_path.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {base_path}")
        return
        
    for file_path in parquet_files:
        batch_groups[file_path.parent].append(file_path)
        
    print(f"Found {len(batch_groups)} unique model/combination folders to process.")

    # Iteratively process each folder
    for parquet_dir, files in batch_groups.items():
        print(f"Processing {len(files)} files in {parquet_dir}")
        
        # Extract metadata from the directory tree
        # size_folder / combination_folder / model_folder / batch.parquet
        model_name = parquet_dir.name
        combo_name = parquet_dir.parent.name
        size_folder_name = parquet_dir.parent.parent.name
        
        size = "small" if "small" in size_folder_name.lower() else "normal"
        combo_prefix = combo_name.split("_")[0]
        
        # Load and stitch batches for this specific iteration
        all_batches = []
        for file_path in files:
            df_batch = pd.read_parquet(file_path, engine="pyarrow")
            all_batches.append(df_batch)
            
        df = pd.concat(all_batches, ignore_index=True)
        # Sort chronologically per company
        df.sort_values(by=["id", "horizon_start"], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = append_metrics(df)
        
        # Aggregate metrics at the company level
        agg_cols = ['horizon_smape', 'horizon_mae', 'horizon_rmse', 'horizon_bias', 'horizon_da']
        company_metrics = df.groupby(['id'])[agg_cols].mean().reset_index()
        
        output_filename = f"metrics_{model_name}_{size}_{combo_prefix}.tsv"
        out_path = parquet_dir.parent.parent / output_filename
        
        company_metrics.to_csv(out_path, sep="\t", index=False)
        print(f"Exported metrics to {out_path}")
        
        # Force memory cleanup before the next folder
        del df
        del company_metrics
        del all_batches
        gc.collect()

if __name__ == "__main__":
    PREDICTION_DIR = r"C:\Users\lsb-ps\Documents\predictions_overview_lean\1_TSMixer"
    process_and_export_batches(PREDICTION_DIR)