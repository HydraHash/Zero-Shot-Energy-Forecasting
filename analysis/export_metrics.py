import ast
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import wilcoxon
import gc
import warnings

# Calculates raw MAE for all models independently
def export_all_raw_maes(base_output_dir: str | Path, output_file: str | Path):
    base_path = Path(base_output_dir)
    parquet_files = list(base_path.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {base_path}")
        return
        
    mae_records = []
    
    for file_path in parquet_files:
        model_name = file_path.parent.name
        df = pd.read_parquet(file_path, engine="pyarrow", columns=['id', 'horizon_start', 'actual_values', 'predicted_values'])
        
        for _, row in df.iterrows():
            actuals = row['actual_values']
            preds = row['predicted_values']
            
            if isinstance(actuals, str): actuals = ast.literal_eval(actuals)
            if isinstance(preds, str): preds = ast.literal_eval(preds)
                
            act_arr = np.array(actuals, dtype=float)
            pred_arr = np.array(preds, dtype=float)
    
            mae = np.mean(np.abs(act_arr - pred_arr))
            mae_records.append({
                'id': row['id'],
                'horizon_start': row['horizon_start'],
                'model_name': model_name,
                'mae': mae
            })
            
    mae_df = pd.DataFrame(mae_records)
    mae_df.to_csv(output_file, index=False)
    print(f"Raw MAEs saved to {output_file}")


def append_advanced_metrics(df: pd.DataFrame, baseline_mae_df: pd.DataFrame) -> pd.DataFrame:
    rmses, smapes_mean, smapes_median, rmaes, das = [], [], [], [], []
    
    # Merge baseline MAEs to align exactly on ID and horizon_start
    df = df.merge(baseline_mae_df, on=['id', 'horizon_start'], how='left')
    
    for _, row in df.iterrows():
        actuals = row['actual_values']
        preds = row['predicted_values']
        baseline_mae = row['baseline_mae']
        
        if isinstance(actuals, str): actuals = ast.literal_eval(actuals)
        if isinstance(preds, str): preds = ast.literal_eval(preds)
            
        act_arr = np.array(actuals, dtype=float)
        pred_arr = np.array(preds, dtype=float)
        errors = act_arr - pred_arr
        
        model_mae = np.mean(np.abs(errors))
        
        # 1. rMAE (Relative MAE)
        if pd.isna(baseline_mae):
            rmae = np.nan
        elif baseline_mae == 0 and model_mae == 0:
            rmae = 0.0  # Both predictions perfectly matched a flat zero load
        elif baseline_mae == 0 and model_mae > 0:
            rmae = np.inf  # Baseline was perfect, zero-shot model hallucinated load
        else:
            rmae = model_mae / baseline_mae
            
        # 2. RMSE
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # 3. sMAPE (Mean & Median)
        denominator = (np.abs(act_arr) + np.abs(pred_arr)) / 2
        smape_arr = np.divide(np.abs(errors), denominator, out=np.zeros_like(errors, dtype=float), where=denominator != 0) * 100
        smape_mean = np.mean(smape_arr)
        smape_median = np.median(smape_arr)
        
        # 4. Directional Accuracy (DA)
        if len(act_arr) > 1:
            da = np.mean(np.sign(np.diff(act_arr)) == np.sign(np.diff(pred_arr))) * 100
        else:
            da = np.nan
            
        rmaes.append(rmae)
        rmses.append(rmse)
        smapes_mean.append(smape_mean)
        smapes_median.append(smape_median)
        das.append(da)
        
    df['horizon_rmae'] = rmaes
    df['horizon_rmse'] = rmses
    df['horizon_smape_mean'] = smapes_mean
    df['horizon_smape_median'] = smapes_median
    df['horizon_da'] = das

    df.drop(columns=['baseline_mae'], inplace=True)
    return df


def process_and_export_batches(base_output_dir: str | Path, mae_file: str | Path, baseline_model_name: str):
    base_path = Path(base_output_dir)
    
    all_maes = pd.read_csv(mae_file)
    baseline_maes = all_maes[all_maes['model_name'] == baseline_model_name][['id', 'horizon_start', 'mae']]
    baseline_maes.rename(columns={'mae': 'baseline_mae'}, inplace=True)
    
    if baseline_maes.empty:
        warnings.warn(f"No MAE data found for baseline '{baseline_model_name}'. rMAE will result in NaN.")
    
    batch_groups = defaultdict(list)
    parquet_files = list(base_path.rglob("*.parquet"))
    
    for file_path in parquet_files:
        batch_groups[file_path.parent].append(file_path)

    for parquet_dir, files in batch_groups.items():
        model_name = parquet_dir.name
        combo_name = parquet_dir.parent.name
        size_folder_name = parquet_dir.parent.parent.name
        
        size = "small" if "small" in size_folder_name.lower() else "normal"
        combo_prefix = combo_name.split("_")[0]
        output_filename = f"metrics_{model_name}_{size}_{combo_prefix}.tsv"
        out_path = parquet_dir.parent.parent / output_filename
        
        if out_path.exists():
            print(f"Skipping {model_name} - {output_filename} already exists.")
            continue

        print(f"Processing metrics for {model_name}...")
        all_batches = [pd.read_parquet(f, engine="pyarrow") for f in files]
        df = pd.concat(all_batches, ignore_index=True)
        df.sort_values(by=["id", "horizon_start"], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = append_advanced_metrics(df, baseline_maes)
        
        # Aggregate logic
        agg_funcs = {
            'horizon_rmae': ['mean', 'median'],
            'horizon_rmse': ['mean', 'median'],
            'horizon_smape_mean': ['mean'],
            'horizon_smape_median': ['median'],
            'horizon_da': ['mean']
        }
        company_metrics = df.groupby('id').agg(agg_funcs)
        
        # Flatten MultiIndex columns
        company_metrics.columns = ['_'.join(col).strip() for col in company_metrics.columns.values]
        company_metrics.reset_index(inplace=True)
        
        company_metrics.to_csv(out_path, sep="\t", index=False)
        print(f"Exported {output_filename}")
        
        del df, company_metrics, all_batches
        gc.collect()


def export_unified_wilcoxon_tests(base_output_dir: str | Path, baseline_model_name: str, target_metrics: list[str] = None, alpha: float = 0.05):
    if target_metrics is None:
        target_metrics = ['horizon_smape_median', 'horizon_rmae_median']
        
    base_path = Path(base_output_dir)
    output_file = base_path / "master_wilcoxon_tests.parquet"
    
    # Implementation of the requested skip-check
    if output_file.exists():
        print(f"Skipping statistical testing - '{output_file.name}' already exists.")
        return
        
    tsv_files = list(base_path.rglob("metrics_*.tsv"))
    if not tsv_files:
        print("No TSV metric files found for statistical testing.")
        return
        
    # 1. Parse filename metadata to strictly isolate evaluation cohorts
    file_metadata = []
    for f in tsv_files:
        parts = f.stem.split('_')
        if len(parts) < 4:
            continue
        combo_prefix = parts[-1]
        size = parts[-2]
        # Rejoin model name in case it contained underscores itself
        model = "_".join(parts[1:-2]) 
        file_metadata.append({'path': f, 'model': model, 'size': size, 'combo': combo_prefix})
        
    df_files = pd.DataFrame(file_metadata)
    baseline_files = df_files[df_files['model'] == baseline_model_name]
    
    if baseline_files.empty:
        warnings.warn(f"No baseline TSVs found for '{baseline_model_name}'.")
        return
        
    results = []
    print(f"-Running  Wilcoxon Testing across {len(target_metrics)} metrics")
    
    # 2. Iterate through each exact evaluation cohort (matching size and combo)
    for _, b_row in baseline_files.iterrows():
        size = b_row['size']
        combo = b_row['combo']
        b_path = b_row['path']
        
        df_baseline = pd.read_csv(b_path, sep="\t")
        
        # Filter competing models within this identical cohort
        competing_files = df_files[
            (df_files['size'] == size) & 
            (df_files['combo'] == combo) & 
            (df_files['model'] != baseline_model_name)
        ]
        
        for _, m_row in competing_files.iterrows():
            model_name = m_row['model']
            df_model = pd.read_csv(m_row['path'], sep="\t")
            
            # Merge on company ID to guarantee strict pairing
            df_paired = df_model.merge(df_baseline, on='id', suffixes=('_model', '_baseline'))
            
            # 3. Iterate over the requested target metrics
            for metric in target_metrics:
                metric_model = f"{metric}_model"
                metric_baseline = f"{metric}_baseline"
                
                if metric_model not in df_paired.columns or metric_baseline not in df_paired.columns:
                    continue
                    
                # Strict cleaning: filter out inf (from rMAE zero-division) and NaN
                valid_mask = np.isfinite(df_paired[metric_model]) & np.isfinite(df_paired[metric_baseline])
                df_clean = df_paired[valid_mask]
                n_dropped = len(df_paired) - len(df_clean)
                
                if len(df_clean) < 10:
                    print(f"Skipping {model_name} ({metric}): Insufficient valid samples.")
                    continue
                    
                # 4. Statistical computation
                stat, p_value = wilcoxon(df_clean[metric_model], df_clean[metric_baseline], zero_method='wilcox')
                median_diff = np.median(df_clean[metric_model] - df_clean[metric_baseline])
                
                is_sig = p_value < alpha
                if is_sig:
                    conclusion = "Model is Superior" if median_diff < 0 else "Baseline is Superior"
                else:
                    conclusion = "No Significant Difference"
                    
                results.append({
                    'target_metric': metric,
                    'model_name': model_name,
                    'baseline_model': baseline_model_name,
                    'size_category': size,
                    'combo_prefix': combo,
                    'n_valid_samples': len(df_clean),
                    'dropped_infs_nans': n_dropped,
                    'median_difference': float(median_diff),
                    'w_statistic': float(stat),
                    'p_value': float(p_value),
                    'is_significant': bool(is_sig),
                    'conclusion': conclusion
                })
                
    # 5. Compile and export master file
    if results:
        results_df = pd.DataFrame(results)
        # Using pyarrow engine for fastest I/O and strict type preservation
        results_df.to_parquet(output_file, engine='pyarrow', index=False)
        print(f"Wilcoxon tests successfully exported to {output_file.name}")
    else:
        print("No valid paired tests could be executed across the provided files.")


if __name__ == "__main__":
    PREDICTION_DIR = r"C:\Users\lsb-ps\Documents\predictions_overview_lean"
    MAE_FILE = Path(PREDICTION_DIR) / "all_models_raw_mae.csv"
    
    # Must explicitly match the exact folder name of the baseline model
    BASELINE_NAME = "SeasonNaive"
    TARGET_METRICS = ['horizon_rmae_median', 'horizon_smape_median']
    
    if not MAE_FILE.exists():
        print("Calculating raw mae for all models...")
        export_all_raw_maes(PREDICTION_DIR, MAE_FILE)

    #print("Calculating all metrics...")
    #process_and_export_batches(PREDICTION_DIR, MAE_FILE, BASELINE_NAME)

    export_unified_wilcoxon_tests(PREDICTION_DIR, BASELINE_NAME, TARGET_METRICS)