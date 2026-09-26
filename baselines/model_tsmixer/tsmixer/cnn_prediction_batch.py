import os
import re
import time
import psutil
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from pathlib import Path
from system_data_psutil import ResourceProfiler

# =====================================================================
# 1. Parameter Configuration Matrix
# =====================================================================
OUTPUT_BASE_DIR = Path("./cnn_prediction_output")

logging.basicConfig(filename="cnn_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

EXPERIMENTAL_COMBINATIONS = {
    1: {"train_split_days": 30, "lag_days": 1, "horizon_days": 1, "step_days": 1},
    2: {"train_split_days": 30,  "lag_days": 7, "horizon_days": 7, "step_days": 7},
}


# =====================================================================
# 2. Original CNN Code
# =====================================================================
class CNNModel(tf.keras.Model):
    def __init__(self, n_channel, pred_len, kernel_size):
        super().__init__()
        self.cnn = tf.keras.layers.Conv1D(
            n_channel, kernel_size, padding='same', input_shape=(None, n_channel)
        )
        self.dense = tf.keras.layers.Dense(pred_len)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        x = self.cnn(x)
        x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
        x = self.dense(x)
        x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
        return x  # [Batch, Output length, Channel]


def create_lag_features(series_values, lag_steps, horizon_steps):
    X, y = [], []
    for i in range(len(series_values) - lag_steps - horizon_steps + 1):
        X.append(series_values[i : i + lag_steps])
        y.append(series_values[i + lag_steps : i + lag_steps + horizon_steps])
    return np.array(X), np.array(y)


# =====================================================================
# 3. Training Pipeline for single training, rolling inference
# =====================================================================
def run_cnn_pipeline(batch_df, config):
    train_split_steps = config["train_split_days"] * 96
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    
    total_steps = len(batch_df)
    strategy_data = {"google_cnn": []}
    
    # ---------------------------------------------------------
    # PHASE 1: Build the Global Training Dataset
    # ---------------------------------------------------------
    # Extract only the chronological training block
    train_df = batch_df.iloc[:train_split_steps]
    
    global_X_train, global_y_train = [], []
    for company_id in batch_df.columns:
        series_vals = train_df[company_id].values.ravel()
        X_train, y_train = create_lag_features(series_vals, lag_steps, horizon_steps)
        global_X_train.append(X_train)
        global_y_train.append(y_train)
        
    X_train_matrix = np.vstack(global_X_train)
    y_train_matrix = np.vstack(global_y_train)
    
    # Format for Keras: [Batch, TimeSteps, Channel=1]
    X_train_matrix = np.expand_dims(X_train_matrix, axis=-1)
    y_train_matrix = np.expand_dims(y_train_matrix, axis=-1)
    

    tf.keras.backend.clear_session() 
    model = CNNModel(n_channel=1, pred_len=horizon_steps, kernel_size=3)
    model.compile(optimizer='adam', loss='mse')
    
    logging.info(f"      -> Phase 1: Training Global CNN on first {config['train_split_days']} days...")
    
    model.fit(X_train_matrix, y_train_matrix, epochs=10, batch_size=64, verbose=0)  #from 264 due to memory error
    
    logging.info(f"      -> Phase 2: Generating rolling predictions for the remaining days...")
    
    # Set the sliding window to start exactly where the training data ends.
    # We step back by `lag_steps` to grab the historical features needed to predict the first horizon.
    eval_start_idx = train_split_steps - lag_steps
    
    for start_idx in range(eval_start_idx, total_steps - lag_steps - horizon_steps + 1, s_steps):
        origin_idx = start_idx + lag_steps
        end_idx = origin_idx + horizon_steps
        
        horizon_start_dt = batch_df.index[origin_idx]
        horizon_end_dt = batch_df.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        # 1. Slice the features and the ground truth
        lag_block = batch_df.iloc[start_idx:origin_idx]
        actual_block = batch_df.iloc[origin_idx:end_idx]
        
        # 2. Extract features for all 500 companies simultaneously
        X_test_batch = np.array([lag_block[cid].values.ravel() for cid in batch_df.columns])
        X_test_batch = np.expand_dims(X_test_batch, axis=-1) # Shape: [500, lag_steps, 1]
        
        # 3. Model Inference: Predict the entire 500-plant batch in one pass!
        preds = model.predict(X_test_batch, verbose=0) # Shape: [500, horizon_steps, 1]
        preds = preds.squeeze(axis=-1)                 # Shape: [500, horizon_steps]
        
        # 4. Map the predictions back to the Parquet/TSV format
        for i, company_id in enumerate(batch_df.columns):
            y_actual_slice = actual_block[company_id].values.ravel()
            pred_cnn = preds[i]
            
            row_cnn = {
                "id": company_id,
                "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_weekday": target_weekday_name,
                "horizon_days": config["horizon_days"],
                "actual_values": y_actual_slice.tolist(),
                "predicted_values": pred_cnn.tolist()
            }
            strategy_data["google_cnn"].append(row_cnn)
            
    return strategy_data


# =====================================================================
# 4. Vectorized Data Loader
# =====================================================================
def load_vea_profiles(file_path) -> pd.DataFrame:
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.pkl':
        raw_df = pd.read_pickle(file_path)
    else:
        raw_df = pd.read_csv(file_path, sep="\t")

    # Set company ID (first column) as index
    id_col = raw_df.columns[0]
    raw_df.set_index(id_col, inplace=True)

    valid_time_cols = []
    parsed_timestamps = []

    # Combined regex pattern for 'YYYY-MM-DD|Weekday|HH:MM-HH:MM' or simple 'YYYY-MM-DD HH:MM'
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}).*?(\d{1,2}:\d{2})")

    for col in raw_df.columns:
        col_str = str(col)
        match = pattern.search(col_str)
        if match:
            date_part, time_part = match.groups()
            parsed_timestamps.append(pd.to_datetime(f"{date_part} {time_part}"))
            valid_time_cols.append(col)
        else:
            logging.info(f" -> [Warning] Skipping non-timestamp column header: '{col}'")

    # Filter DataFrame to keep only valid timeslot columns
    raw_df = raw_df[valid_time_cols]

    # Transpose and assign DatetimeIndex globally
    df_t = raw_df.T
    df_t.index = pd.DatetimeIndex(parsed_timestamps)
    
    # Sort index chronologically, handle duplicates, force frequency, and fill gaps
    df_t = df_t[~df_t.index.duplicated(keep='first')].sort_index()
    df_t = df_t.loc[:, ~df_t.columns.duplicated(keep='first')]
    df_t = df_t.asfreq("15min").ffill().bfill()
    
    logging.info(f"Successfully loaded continuous grid: {df_t.shape[1]} companies across {df_t.shape[0]} timestamps.")
    return df_t


# =====================================================================
# 5. Main Execution Block
# =====================================================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched.tsv"
    
    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)
    BATCH_SIZE = 500
    company_ids = loaded_profiles_df.columns
    
    logging.info(f"\nStarting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['lag_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        cnn_dir = combo_output_path / "google_cnn"
        cnn_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing {combo_dir_name}...")
        
        with ResourceProfiler(run_name=combo_dir_name, input_file_path=DATA_PATH, output_dir_path=combo_output_path):
            for i in range(0, len(company_ids), BATCH_SIZE):
                batch_cols = company_ids[i : i + BATCH_SIZE]
                batch_df = loaded_profiles_df[batch_cols]
                logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
                
                results = run_cnn_pipeline(batch_df, config)
                
                if results["google_cnn"]:
                    df_cnn = pd.DataFrame(results["google_cnn"])
                    df_cnn.to_parquet(cnn_dir / f"batch_{i}.parquet", index=False, compression="snappy")

                    if i == 0:
                        tsv_path = cnn_dir / f"batch_{i}_.tsv"
                        df_cnn.to_csv(tsv_path, sep="\t", index=False)
                    
                    del df_cnn 
                del results
                
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")