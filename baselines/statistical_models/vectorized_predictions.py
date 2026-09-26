import os
import re
import numpy as np
import pandas as pd
import logging
from system_data_psutil import ResourceProfiler
from pathlib import Path
from datetime import timedelta

# =====================================================================
# 1. Parameter Configuration Matrix
# =====================================================================
OUTPUT_BASE_DIR = Path("./small_seasonnaiv_prediction_output")

logging.basicConfig(filename="small_seasonnaiv_statistical_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

EXPERIMENTAL_COMBINATIONS = {
    # WARNING: To look back 7 days for Mon/Sat/Sun, input_days MUST be >= 7.
    #1:  {"input_days": 7,  "horizon_days": 1, "step_days": 1}, 
    2:  {"input_days": 7,  "horizon_days": 7, "step_days": 7}, 
}

# =====================================================================
# 2. Vectorized Forecast Methods 
# =====================================================================
def forecast_seasonal_naive(y_train_series, horizon_start_dt, horizon_days):
    """
    Vectorized seasonal naive:
    - Tue, Wed, Thu, Fri -> Copies previous day (T-1)
    - Sat, Sun, Mon -> Copies exact same day from last week (T-7)
    """
    forecast = []
    train_len = len(y_train_series)
    
    for day_offset in range(horizon_days):
        target_date = horizon_start_dt + timedelta(days=day_offset)
        target_weekday = target_date.weekday() # 0=Mon, 6=Sun
        
        # Determine Lookback Logic
        if target_weekday in [0, 5, 6]:
            lookback_days = 7
        else:
            lookback_days = 1
            
        lookback_steps = lookback_days * 96
        
        # Safety fallback if context is too short
        if train_len < lookback_steps:
            lookback_steps = 96 # Fallback to yesterday
            logging.info(r"Fallback to yesterday was activated!")
            
        # Extract the exact 96-step 24h block via numpy slicing
        start_idx = train_len - lookback_steps
        historical_slice = y_train_series.iloc[start_idx : start_idx + 96].values
        forecast.extend(historical_slice)
        
    return np.array(forecast)

def forecast_historical_average(y_train_series, horizon_days):
    """
    Calculates the mean for each 15-min slot across the entire lookback context.
    """
    # Reshape the training series into days (N days, 96 intervals)
    # Then take the mean across the days (axis 0)
    daily_matrix = y_train_series.values.reshape(-1, 96)
    average_day = daily_matrix.mean(axis=0)
    
    # Tile the average day out to match the required horizon days
    return np.tile(average_day, horizon_days)

# =====================================================================
# 3. Optimized Master Pipeline
# =====================================================================
def run_master_pipeline(df_all_series, config):
    x_steps = config["input_days"] * 96
    y_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    
    total_steps = len(df_all_series)
    strategy_data = {"seasonal_naive": [], "historic_average": []}
    
    # Iterate over sliding windows first, then companies (more memory efficient)
    for start_idx in range(0, total_steps - x_steps - y_steps + 1, s_steps):
        origin_idx = start_idx + x_steps
        end_idx = origin_idx + y_steps
        
        # Timestamps for metadata
        horizon_start_dt = df_all_series.index[origin_idx]
        horizon_end_dt = df_all_series.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        # Slice train and actual target blocks globally for all companies at once
        train_block = df_all_series.iloc[start_idx:origin_idx]
        actual_block = df_all_series.iloc[origin_idx:end_idx]
        
        for company_id in df_all_series.columns:
            y_train_slice = train_block[company_id]
            y_actual_slice = actual_block[company_id].values
            
            # Base metadata enriched for evaluation
            row_meta = {
                "id": company_id,
                "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_weekday": target_weekday_name,
                "horizon_days": config["horizon_days"]
            }
            
            # 1. Seasonal Naive
            pred_naive = forecast_seasonal_naive(y_train_slice, horizon_start_dt, config["horizon_days"])
            row_naive = row_meta.copy()
            row_naive["actual_values"] = y_actual_slice.tolist() # Store ground truth
            row_naive["predicted_values"] = pred_naive.tolist()
            strategy_data["seasonal_naive"].append(row_naive)
            
            # 2. Historical Average
            pred_avg = forecast_historical_average(y_train_slice, config["horizon_days"])
            row_avg = row_meta.copy()
            row_avg["actual_values"] = y_actual_slice.tolist()
            row_avg["predicted_values"] = pred_avg.tolist()
            strategy_data["historic_average"].append(row_avg)
            
    return strategy_data

# =====================================================================
# 4. Vectorized Data Loader
# =====================================================================
def load_vea_profiles(file_path) -> pd.DataFrame:
    """
    Returns a transposed, continuous Datetime-indexed DataFrame where 
    columns are company IDs and rows are 15-minute intervals.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.pkl':
        raw_df = pd.read_pickle(file_path)
    else:
        raw_df = pd.read_csv(file_path, sep="\t")

    id_col = raw_df.columns[0]
    raw_df.set_index(id_col, inplace=True)

    # Parse timestamps using list comprehension (much faster)
    parsed_timestamps = [
        pd.to_datetime(f"{re.search(r'(\d{4}-\d{2}-\d{2})', str(c)).group(1)} {re.search(r'(\d{2}:\d{2})-\d{2}:\d{2}', str(c)).group(1)}")
        for c in raw_df.columns
    ]

    # Transpose and assign index globally
    df_t = raw_df.T
    df_t.index = pd.DatetimeIndex(parsed_timestamps)
    
    # Force frequency and fill missing values globally
    df_t = df_t.asfreq("15min").ffill().bfill()
    
    logging.info(f"Successfully loaded continuous grid: {df_t.shape[1]} companies across {df_t.shape[0]} timestamps.")
    return df_t

# =====================================================================
# 5. Main execution block
# =====================================================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched_small.tsv"
    
    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)
    BATCH_SIZE = 268
    company_ids = loaded_profiles_df.columns
    
    logging.info(f"\nStarting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['input_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        naive_dir = combo_output_path / "seasonal_naive"
        avg_dir = combo_output_path / "historic_average"
        naive_dir.mkdir(parents=True, exist_ok=True)
        avg_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing {combo_dir_name}...")
        
        with ResourceProfiler(run_name=combo_dir_name, input_file_path=DATA_PATH, output_dir_path=combo_output_path):
            for i in range(0, len(company_ids), BATCH_SIZE):
                batch_cols = company_ids[i : i + BATCH_SIZE]
                batch_df = loaded_profiles_df[batch_cols]
                
                logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
                
                results = run_master_pipeline(batch_df, config)
                
                # Convert dictionaries to DataFrames and save directly to disk
                if results["seasonal_naive"]:
                    df_naive = pd.DataFrame(results["seasonal_naive"])
                    df_naive.to_parquet(naive_dir / f"batch_{i}.parquet", index=False, compression="snappy", engine="pyarrow")

                    if i == 0:
                        tsv_path = naive_dir / f"batch_{i}_.tsv"
                        df_naive.to_csv(tsv_path, sep="\t", index=False)
                    
                    # Free memory immediately
                    del df_naive 
                    
                if results["historic_average"]:
                    df_avg = pd.DataFrame(results["historic_average"])
                    df_avg.to_parquet(avg_dir / f"batch_{i}.parquet", index=False, compression="snappy", engine="pyarrow")

                    if i == 0:
                        tsv_path = avg_dir / f"batch_{i}_.tsv"
                        df_avg.to_csv(tsv_path, sep="\t", index=False)
                    
                    del df_avg 
                del results
            
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")