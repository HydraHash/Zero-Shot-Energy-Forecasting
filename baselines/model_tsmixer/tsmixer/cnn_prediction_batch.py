import re
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from pathlib import Path




EXPERIMENTAL_COMBINATIONS = {
    1: {"lag_days": 1, "horizon_days": 1, "step_days": 1},
    2: {"lag_days": 7, "horizon_days": 7, "step_days": 7},
    3: {"lag_days": 7, "horizon_days": 1, "step_days": 1},
    4: {"lag_days": 14, "horizon_days": 1, "step_days": 1}
}

# TSMixer CNN Code from Google Research
class CNNModel(tf.keras.Model):
    def __init__(self, n_channel, pred_len, kernel_size):
        super().__init__()
        self.cnn = tf.keras.layers.Conv1D(
            n_channel, kernel_size, padding='same', input_shape=(None, n_channel)
        )
        self.dense = tf.keras.layers.Dense(pred_len)

    def call(self, x):
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


def train_cnn(uci_df: pd.DataFrame, lag_steps: int, horizon_steps: int):
    global_X_train, global_y_train = [], []
    for company_id in uci_df.columns:
        series_vals = uci_df[company_id].values.ravel()
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
    
    logging.info(f"Training Global CNN on UCI Dataset")
    model.fit(X_train_matrix, y_train_matrix, epochs=3, batch_size=128, verbose=0)  #from 264 due to memory error

    return model


def run_zs_interference(vea_df: pd.DataFrame, config: dict, pretrained_model):
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    total_steps = len(vea_df)
    
    strategy_data = {"CNN": []}
    
    for start_idx in range(0, total_steps - lag_steps - horizon_steps + 1, s_steps):
        origin_idx = start_idx + lag_steps
        end_idx = origin_idx + horizon_steps
        
        horizon_start_dt = batch_df.index[origin_idx]
        horizon_end_dt = batch_df.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        lag_block = vea_df.iloc[start_idx:origin_idx]
        actual_block = vea_df.iloc[origin_idx:end_idx]

        X_test_batch = np.array([lag_block[cid].values.ravel() for cid in vea_df.columns])
        X_test_batch = np.expand_dims(X_test_batch, axis=-1) # Shape: [500, lag_steps, 1]

        # Inference using the frozen UCI weights
        preds = pretrained_model.predict(X_test_batch, verbose=0) # Shape: [500, horizon_steps, 1]
        preds = preds.squeeze(axis=-1) # Shape: [500, horizon_steps]
        
        # Map the predictions back to the Parquet/TSV format
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
            strategy_data["CNN"].append(row_cnn)
            
    return strategy_data


def load_train_dataset(file_path, max_clients: int = 50, months: int = 4) -> pd.DataFrame:
    load_cols = [0] + list(range(1, max_clients + 1))
    
    df = pd.read_csv(file_path, sep=';', decimal=',', index_col=0, parse_dates=True, usecols=load_cols)
    df = df.astype(np.float32)
    load_rows = months * 30 * 96
    df = df.tail(load_rows)
    df = df.resample('15min').ffill().fillna(0)
    
    logging.info(f"UCI Dataset loaded: {df.shape[1]} clients, {df.shape[0]} timestamps.")
    return df


def load_vea_profiles(file_path) -> pd.DataFrame:
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.pkl':
        raw_df = pd.read_pickle(file_path)
    else:
        raw_df = pd.read_csv(file_path, sep="\t")

    id_col = raw_df.columns[0]
    raw_df.set_index(id_col, inplace=True)

    valid_time_cols = []
    parsed_timestamps = []
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

    raw_df = raw_df[valid_time_cols]
    df_t = raw_df.T
    df_t.index = pd.DatetimeIndex(parsed_timestamps)
    
    df_t = df_t[~df_t.index.duplicated(keep='first')].sort_index()
    df_t = df_t.loc[:, ~df_t.columns.duplicated(keep='first')]
    df_t = df_t.asfreq("15min").ffill().bfill()
    
    logging.info(f"Successfully loaded continuous grid: {df_t.shape[1]} companies across {df_t.shape[0]} timestamps.")
    return df_t


if __name__ == "__main__":
    DATA_PATH = r"example\Path\data.tsv"
    TRAIN_DATA_PATH = r"example\Data\Path.txt"
    BATCH_SIZE = 268
    OUTPUT_BASE_DIR = Path("./c4_cnn_prediction_output")

    logging.basicConfig(filename="c4_cnn_log.txt",
                        filemode='a',
                        format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)
    train_profiles_df = load_train_dataset(TRAIN_DATA_PATH)0
    company_ids = loaded_profiles_df.columns
    
    logging.info(f"\nStarting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['lag_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        cnn_dir = combo_output_path / "CNN"
        cnn_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing {combo_dir_name}...")

        trained_cnn = train_cnn(train_profiles_df, config["lag_days"] * 96, config["horizon_days"] * 96)
        
        for i in range(0, len(company_ids), BATCH_SIZE):
            batch_cols = company_ids[i : i + BATCH_SIZE]
            batch_df = loaded_profiles_df[batch_cols]
            logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
            
            results = run_zs_interference(batch_df, config, trained_cnn)
            
            if results["CNN"]:
                df_cnn = pd.DataFrame(results["CNN"])
                df_cnn.to_parquet(cnn_dir / f"batch_{i}.parquet", index=False, compression="snappy", engine="pyarrow")

                if i == 0:
                    tsv_path = cnn_dir / f"batch_{i}_.tsv"
                    df_cnn.to_csv(tsv_path, sep="\t", index=False)
                
                del df_cnn 
            del results
                
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")