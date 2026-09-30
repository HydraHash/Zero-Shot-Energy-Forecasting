import os
import re
import time
import psutil
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from tensorflow.keras import layers
from pathlib import Path
from system_data_psutil import ResourceProfiler
from tsmixer_revin import RevNorm

# Parameter Configuration Matrix
OUTPUT_BASE_DIR = Path("./small_outdom_prediction_output")

logging.basicConfig(filename="small_outdom_tsmixer_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

EXPERIMENTAL_COMBINATIONS = {
    1: {"train_split_days": 30, "lag_days": 1, "horizon_days": 1, "step_days": 1},
    2: {"train_split_days": 30,  "lag_days": 7, "horizon_days": 7, "step_days": 7},
    3: {"train_split_days": 30,  "lag_days": 7, "horizon_days": 1, "step_days": 1},
}

# TSMixer + RevIN Code (Google Research)
def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer."""
    if norm_type == 'L':
        norm_layer_1 = layers.LayerNormalization(axis=[-2, -1])
        norm_layer_2 = layers.LayerNormalization(axis=[-2, -1])
    else:
        norm_layer_1 = layers.BatchNormalization(axis=-1)
        norm_layer_2 = layers.BatchNormalization(axis=-1)

    # Temporal Linear Block
    x = norm_layer_1(inputs)
    # FIX: Use Keras Permute layer instead of tf.transpose
    x = layers.Permute((2, 1))(x)  # [Batch, Channel, Input Length]
    
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    
    # FIX: Use Keras Permute layer instead of tf.transpose
    x = layers.Permute((2, 1))(x)  # [Batch, Input Length, Channel]
    
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feature Linear Block
    x = norm_layer_2(res)
    x = layers.Dense(ff_dim, activation=activation)(x)  # [Batch, Input Length, FF_Dim]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    
    return x + res


def build_tsmixer_revin_model(input_shape, pred_len, norm_type='L', activation='relu', n_block=2, dropout=0.1, ff_dim=64, target_slice=None):
    """Build TSMixer with Reversible Instance Normalization model."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs  # [Batch, Input Length, Channel]
    
    rev_norm = RevNorm(axis=-2)
    x = rev_norm(x, mode='norm')
    
    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, ff_dim)

    if target_slice:
        x = x[:, :, target_slice]

    # FIX: Use Keras Permute layer instead of tf.transpose
    x = layers.Permute((2, 1))(x)  # [Batch, Channel, Input Length]
    x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    
    # FIX: Use Keras Permute layer instead of tf.transpose
    outputs = layers.Permute((2, 1))(x)  # [Batch, Output Length, Channel]
    
    outputs = rev_norm(outputs, mode='denorm', target_slice=target_slice)
    
    return tf.keras.Model(inputs, outputs)


def create_lag_features(series_values, lag_steps, horizon_steps):
    X, y = [], []
    for i in range(len(series_values) - lag_steps - horizon_steps + 1):
        X.append(series_values[i : i + lag_steps])
        y.append(series_values[i + lag_steps : i + lag_steps + horizon_steps])
    return np.array(X), np.array(y)


def run_zs_inference(vea_df: pd.DataFrame, config: dict, pretrained_model):
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    total_steps = len(vea_df)
    
    strategy_data = {"TSMixer": []}
    
    logging.info("Running Zero-Shot Inference on VEA Dataset...")
    
    # Notice we start exactly at index 0. No train_split_days offset.
    for start_idx in range(0, total_steps - lag_steps - horizon_steps + 1, s_steps):
        origin_idx = start_idx + lag_steps
        end_idx = origin_idx + horizon_steps
        
        horizon_start_dt = vea_df.index[origin_idx]
        horizon_end_dt = vea_df.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        lag_block = vea_df.iloc[start_idx:origin_idx]
        actual_block = vea_df.iloc[origin_idx:end_idx]
        
        X_test_batch = np.array([lag_block[cid].values.ravel() for cid in vea_df.columns])
        X_test_batch = np.expand_dims(X_test_batch, axis=-1) 
        
        # Inference using the frozen UCI weights
        preds = pretrained_model.predict(X_test_batch, verbose=0) 
        preds = preds.squeeze(axis=-1)                 
        
        for i, company_id in enumerate(vea_df.columns):
            row = {
                "id": company_id,
                "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_weekday": target_weekday_name,
                "horizon_days": config["horizon_days"],
                "actual_values": actual_block[company_id].values.ravel().tolist(),
                "predicted_values": preds[i].tolist()
            }
            strategy_data["TSMixer"].append(row)
            
    return strategy_data


def load_train_dataset(file_path, max_clients: int = 50, months: int = 4) -> pd.DataFrame:
    logging.info("Loading UCI Electricity Dataset for Training...")

    load_cols = [0] + list(range(1, max_clients + 1))
    
    df = pd.read_csv(file_path, sep=';', decimal=',', index_col=0, parse_dates=True, usecols=load_cols, dtype=np.float32)
    load_rows = months * 30 * 96
    df = df.tail(load_rows)
    df = df.resample('15min').ffill().fillna(0)
    
    logging.info(f"UCI Dataset loaded: {df.shape[1]} clients, {df.shape[0]} timestamps.")
    return df


def train_tsmixer(uci_df: pd.DataFrame, lag_steps: int, horizon_steps: int):
    global_X_train, global_y_train = [], []
    
    logging.info("Extracting training sequences from training data...")
    for client_id in uci_df.columns:
        series_vals = uci_df[client_id].values.ravel()
        X_train, y_train = create_lag_features(series_vals, lag_steps, horizon_steps)
        global_X_train.append(X_train)
        global_y_train.append(y_train)
        
    X_train_matrix = np.vstack(global_X_train)
    y_train_matrix = np.vstack(global_y_train)
    
    X_train_matrix = np.expand_dims(X_train_matrix, axis=-1)
    y_train_matrix = np.expand_dims(y_train_matrix, axis=-1)
    
    tf.keras.backend.clear_session()
    
    model = build_tsmixer_revin_model(
        input_shape=(lag_steps, 1),
        pred_len=horizon_steps,
        norm_type='L',
        activation='relu',
        n_block=2,   
        dropout=0.1, 
        ff_dim=64,   
    )
    model.compile(optimizer='adam', loss='mse', jit_compile=True)
    logging.info("Training Zero-Shot TSMixer on UCI Dataset...")
    model.fit(X_train_matrix, y_train_matrix, epochs=3, batch_size=1024, verbose=1)
    
    return model


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


# Main Execution Block
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched_small.tsv"
    TRAIN_DATA_PATH = r"C:\Users\lsb-ps\Documents\electricityloaddiagrams20112014\LD2011_2014.txt"
    
    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)
    train_profiles_df = load_train_dataset(TRAIN_DATA_PATH)
    BATCH_SIZE = 268 #268 due to memory error
    company_ids = loaded_profiles_df.columns
    
    logging.info(f"\nStarting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['lag_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        tsmixer_dir = combo_output_path / "TSMixer"
        tsmixer_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing {combo_dir_name}...")

        trained_tsmixer = train_tsmixer(train_profiles_df, config["lag_days"] * 96, config["horizon_days"] * 96)

        for i in range(0, len(company_ids), BATCH_SIZE):
            batch_cols = company_ids[i : i + BATCH_SIZE]
            batch_df = loaded_profiles_df[batch_cols]
            logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
            
            results = run_zs_inference(batch_df, config, trained_tsmixer)
            
            if results["TSMixer"]:
                df_tsmixer = pd.DataFrame(results["TSMixer"])
                df_tsmixer.to_parquet(tsmixer_dir / f"batch_{i}.parquet", index=False, compression="snappy")

                if i == 0:
                    tsv_path = tsmixer_dir / f"batch_{i}_.tsv"
                    df_tsmixer.to_csv(tsv_path, sep="\t", index=False)
                
                del df_tsmixer 
            del results
                
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")