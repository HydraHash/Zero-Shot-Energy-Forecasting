import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
        model_name = file_path.parent.name
        combo_name = file_path.parent.parent.name
        
        # Read the file using the robust PyArrow engine
        df_batch = pd.read_parquet(file_path, engine="pyarrow")
        
        # Inject the metadata directly into the DataFrame
        df_batch["model"] = model_name
        df_batch["combo"] = combo_name
        
        all_batches.append(df_batch)

    # Stacks all companies, models, and configurations into one massive table
    full_df = pd.concat(all_batches, ignore_index=True)
    
    # 3. Stitching and Chronological Sorting
    full_df.sort_values(by=["combo", "model", "id", "horizon_start"], ascending=[True, True, True, True], inplace=True)
    
    full_df.reset_index(drop=True, inplace=True)
    logging.info(f"Successfully stitched {len(full_df)} total prediction horizons.")
    
    return full_df

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


def basic_measures() -> None:
    # Weekly sales data
    actual    = np.array([200, 180, 220, 190, 210, 230, 250])
    predicted = np.array([210, 170, 215, 200, 205, 240, 245])

    # Training data for MASE baseline
    train = np.array([160, 175, 185, 195, 200, 190, 205, 198])

    # --- Core metrics ---
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    bias = np.mean(predicted - actual)
    r2   = r2_score(actual, predicted)

    # --- Scale-free metrics ---
    mape_val  = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape_val = np.mean(np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100
    mase_val  = mae / np.mean(np.abs(np.diff(train)))

    # --- Direction ---
    da = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100

    print("="*40)
    print("   MODEL EVALUATION REPORT")
    print("="*40)
    print(f"RMSE:  {rmse:.2f}")
    print(f"MAE:   {mae:.2f}")
    print(f"Bias:  {bias:.2f}  ({'overestimates' if bias > 0 else 'underestimates'})")
    print(f"R²:    {r2:.4f}")
    print(f"MAPE:  {mape_val:.2f}%")
    print(f"sMAPE: {smape_val:.2f}%")
    print(f"MASE:  {mase_val:.3f}  ({'beats' if mase_val < 1 else 'does NOT beat'} naïve)")
    print(f"DA:    {da:.1f}%")

if __name__ == "__main__":
    PREDICTION_DIR = r"H:\model_Timer\small_timer_prediction_output"

    master_df = load_and_stitch_predictions(PREDICTION_DIR)
    print(master_df.head())