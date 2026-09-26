import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sktime.forecasting.naive import NaiveForecaster

# =====================================================================
# 1. Parameter Configuration Matrix & Load Paths
# =====================================================================
OUTPUT_BASE_DIR = Path("./prediction_output")

EXPERIMENTAL_COMBINATIONS = {
    1:  {"input_days": 1,  "horizon_days": 1, "step_days": 1}, #day-day
    2:  {"input_days": 7,  "horizon_days": 7, "step_days": 7}, #week-week
}

# =====================================================================
# 2. Forecast Methods for given input context and forecast horizon steps
# =====================================================================
def forecast_seasonal_naive_german_grid(y_train, horizon_steps):
    if y_train.index.freq is None:
        y_train = y_train.asfreq("15min")

    last_train_timestamp = y_train.index[-1]
    target_start_timestamp = last_train_timestamp + timedelta(minutes=15)
    target_weekday = target_start_timestamp.weekday() # 0=Mon, 5=Sat, 6=Sun
    
    # Preferred seasonal period
    if target_weekday in [0, 5, 6]:
        desired_sp = 96 * 7  # 672 steps (1 week)
    else:
        desired_sp = 96      # 96 steps (1 day)

    # SAFETY CAVEAT: Capping sp so it never exceeds available training context
    sp_setting = min(desired_sp, len(y_train))

    forecaster = NaiveForecaster(strategy="last", sp=sp_setting)
    forecaster.fit(y_train)
    fh = np.arange(1, horizon_steps + 1)
    return forecaster.predict(fh).values

def forecast_historical_average(y_train, horizon_steps):
    # 15-minute historical intra-day average for each specific 15-min interval across lookback context
    # forecasts the mean of all values in a season from last window will be forecasted
    if y_train.index.freq is None:
        y_train = y_train.asfreq("15min")

    sp_setting = min(96, len(y_train))

    forecaster = NaiveForecaster(strategy="mean", sp=sp_setting)
    forecaster.fit(y_train)
    fh = np.arange(1, horizon_steps + 1)
    return forecaster.predict(fh).values

# =====================================================================
# 3. Sliding Window over all loaded entries for one parameter combination
# =====================================================================
def run_master_pipeline(series_dict, combo_id, config):
    x_steps = config["input_days"] * 96
    y_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    
    strategy_data = {"seasonal_naive": [], "historic_average": []}
    
    for company_id, y_series in series_dict.items():
        total_steps = len(y_series)
        
        # Sliding Window Loop Slices
        for start_idx in range(0, total_steps - x_steps - y_steps + 1, s_steps):
            origin_idx = start_idx + x_steps
            end_idx = origin_idx + y_steps
            
            y_train_slice = y_series.iloc[start_idx:origin_idx]
            
            # Extract Metadata Markers for the row structure
            origin_dt = y_train_slice.index[-1]
            horizon_start_dt = y_train_slice.index[-1] + timedelta(minutes=15)
            horizon_end_dt = y_series.index[end_idx - 1]
            
            row_meta = {
                "id": company_id,
                "origin_date": origin_dt.strftime("%m-%d"),
                "horizon_start": horizon_start_dt.strftime("%m-%d"),
                "horizon_end": horizon_end_dt.strftime("%m-%d")
            }
            
            # Execute Strategy 1: Seasonal Naïve
            pred_naive = forecast_seasonal_naive_german_grid(y_train_slice, y_steps)
            row_naive = row_meta.copy()
            for step_idx, val in enumerate(pred_naive, start=1):
                row_naive[f"step_{step_idx}"] = val
            strategy_data["seasonal_naive"].append(row_naive)
            
            # Execute Strategy 2: Historical Average
            pred_avg = forecast_historical_average(y_train_slice, y_steps)
            row_avg = row_meta.copy()
            for step_idx, val in enumerate(pred_avg, start=1):
                row_avg[f"step_{step_idx}"] = val
            strategy_data["historic_average"].append(row_avg)
            
    return strategy_data

# =====================================================================
# 4. Load enriched load data from pkl file
# =====================================================================
def load_vea_profiles_from_tsv(file_path) -> dict:
    """
    Parses wide-format load profiles where column headers follow the composite format:
    'YYYY-MM-DD|DayOfWeek|HH:MM-HH:MM'
    
    Returns:
        dict: { company_id_str: pd.Series(values, index=DatetimeIndex(freq='15min')) }
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found at path: {file_path}")

    print(f"Loading raw VEA load profiles from: {file_path} ...")
    
    # Load dataset (TSV or PKL depending on extension)
    if file_path.suffix.lower() == '.pkl':
        raw_df = pd.read_pickle(file_path)
    else:
        raw_df = pd.read_csv(file_path, sep="\t")

    # Set company ID as index
    id_col = raw_df.columns[0]
    raw_df.set_index(id_col, inplace=True)

    # Extract all 15-minute timeslot column headers
    time_columns = raw_df.columns

    # Parse timestamps from headers: '2016-01-01|Friday|00:00-00:15' -> '2016-01-01 00:00'
    parsed_timestamps = []
    for col_header in time_columns:
        # Match 'YYYY-MM-DD' and the start time 'HH:MM'
        match = re.match(r"(\d{4}-\d{2}-\d{2})\|[^|]+\|(\d{2}:\d{2})-\d{2}:\d{2}", str(col_header))
        if match:
            date_str, start_time_str = match.groups()
            parsed_timestamps.append(pd.to_datetime(f"{date_str} {start_time_str}"))
        else:
            raise ValueError(f"Header '{col_header}' does not match pattern 'YYYY-MM-DD|Weekday|HH:MM-HH:MM'")

    datetime_index = pd.DatetimeIndex(parsed_timestamps)

    # Optional subsetting for fast local testing
    #target_ids = raw_df.index[:max_companies] if max_companies else raw_df.index

    series_map = {}
    for c_id in raw_df.index:
        # Extract row values for company and coerce to numeric floats
        load_values = pd.to_numeric(raw_df.loc[c_id, time_columns].values, errors='coerce')
        
        # Build Pandas Series with explicit 15min frequency required by sktime
        s = pd.Series(data=load_values, index=datetime_index)
        s = s[~s.index.duplicated(keep='first')].sort_index()
        s = s.asfreq("15min")

        # Fill potential missing values
        if s.isna().any():
            s = s.ffill().bfill()

        series_map[str(c_id)] = s

    print(f"Successfully loaded {len(series_map)} company series into 15-minute sktime format.")
    return series_map

# =====================================================================
# 5. MAIN PIPELINE EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched.tsv"
    loaded_profiles = load_vea_profiles_from_tsv(file_path=DATA_PATH)
    
    print(f"\nStarting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} parameter matrix grids...")
    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"combo_{combo_id}_X{config['input_days']}_Y{config['horizon_days']}_S{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        combo_output_path.mkdir(parents=True, exist_ok=True)
        
        print(f" -> Processing {combo_dir_name}...")
        
        # Execute rolling forecasts for current setup combo block
        results = run_master_pipeline(loaded_profiles, combo_id, config)
        
        # Save down output configurations safely per model target strategy
        for strategy_name, matrix_records in results.items():
            if not matrix_records:
                continue
                
            matrix_df = pd.DataFrame(matrix_records)
            
            # File baseline destination identifiers
            parquet_file = combo_output_path / f"{strategy_name}.parquet"
            tsv_file = combo_output_path / f"{strategy_name}.tsv"
            
            # Export Matrix Array Formats
            matrix_df.to_parquet(parquet_file, index=False, compression="snappy")
            matrix_df.to_csv(tsv_file, sep="\t", index=False)
            
    print(f"\nExecution loop complete. High-level path structure saved to: '{OUTPUT_BASE_DIR.absolute()}'")