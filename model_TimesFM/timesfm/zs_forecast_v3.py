import os
import re
import time
import psutil
import threading
import numpy as np
import pandas as pd
import torch
import torch.quantization
import logging
import timesfm
from pathlib import Path
from system_data_psutil import ResourceProfiler

# Optimize float32 matrix operations
torch.set_float32_matmul_precision("high")
torch.set_num_threads(20)

OUTPUT_BASE_DIR = Path("./verysmall_timesfm3_prediction_output")

logging.basicConfig(filename="verysmall_timesfm3_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


EXPERIMENTAL_COMBINATIONS = {
    1: {"lag_days": 1, "horizon_days": 1, "step_days": 1},
    2: {"lag_days": 7, "horizon_days": 7, "step_days": 7},
    3: {"lag_days": 7, "horizon_days": 1, "step_days": 7},
}

def run_timesfm_pipeline(batch_df, config, model):
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    
    total_steps = len(batch_df)
    strategy_data = {"TimesFM": []}
    
    # Start sliding window once we have enough context (lag_steps)
    for start_idx in range(0, total_steps - lag_steps - horizon_steps + 1, s_steps):
        origin_idx = start_idx + lag_steps
        end_idx = origin_idx + horizon_steps
        
        horizon_start_dt = batch_df.index[origin_idx]
        horizon_end_dt = batch_df.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        # 1. Slice features (context) and actual ground truth
        lag_block = batch_df.iloc[start_idx:origin_idx]
        actual_block = batch_df.iloc[origin_idx:end_idx]
        
        # 2. Extract inputs as a list of 1D numpy arrays for all 500 companies in batch
        inputs_list = [lag_block[cid].values.ravel() for cid in batch_df.columns]

        batch_outputs = list(model.predict_batch(inputs_list, horizon=horizon_steps))

        # 4. Map predictions back to the unified Parquet/TSV format
        for i, company_id in enumerate(batch_df.columns):
            y_actual_slice = actual_block[company_id].values.ravel()
            
            # Extract the point forecast attribute from the output object
            pred_timesfm = batch_outputs[i].forecast
            
            row_timesfm = {
                "id": company_id,
                "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_weekday": target_weekday_name,
                "horizon_days": config["horizon_days"],
                "actual_values": y_actual_slice.tolist(),
                "predicted_values": pred_timesfm.tolist()
            }
            strategy_data["TimesFM"].append(row_timesfm)
            
    return strategy_data


def load_vea_profiles(file_path) -> pd.DataFrame:
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.pkl':
        raw_df = pd.read_pickle(file_path)
    else:
        raw_df = pd.read_csv(file_path, sep="\t")

    first_col_name = str(raw_df.columns[0])
    if not re.search(r"\d{4}-\d{2}-\d{2}", first_col_name):
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

    raw_df = raw_df[valid_time_cols]

    df_t = raw_df.T
    df_t.index = pd.DatetimeIndex(parsed_timestamps)
    
    df_t = df_t[~df_t.index.duplicated(keep='first')].sort_index()
    df_t = df_t.loc[:, ~df_t.columns.duplicated(keep='first')]
    df_t = df_t.asfreq("15min").ffill().bfill()
    
    logging.info(f"Successfully loaded continuous grid: {df_t.shape[1]} companies across {df_t.shape[0]} timestamps.")
    return df_t


if __name__ == "__main__":
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched_very_small.tsv"
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    logging.info(f" HARDWARE DETECTED: {device_str.upper()}")
    if device_str == "cuda":
        logging.info(f" GPU NAME: {torch.cuda.get_device_name(0)}")
        logging.info(f" CUDA VERSION: {torch.version.cuda}")
    
    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)
    
    # High Batch Size since CPU is more efficient with fewer loops
    BATCH_SIZE = 512 
    company_ids = loaded_profiles_df.columns
    
    logging.info("\nLoading TimesFM 3.0 PyTorch Checkpoint from Hugging Face...")
    tfm_model = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=BATCH_SIZE)    

    try:
        tfm_model = tfm_model.to(device)
    except AttributeError:
        if hasattr(tfm_model, 'model'):
            tfm_model.model = tfm_model.model.to(device)
            
    if device_str == "cpu":
        logging.info("\nApplying CPU Dynamic Quantization and XLA JIT Compilation...")
        try:
            if hasattr(tfm_model, 'model'):
                # 1. Dynamic INT8 Quantization (Reduces memory bandwidth & speeds up linear layers)
                tfm_model.model = torch.quantization.quantize_dynamic(
                    tfm_model.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                # 2. PyTorch JIT Compile (Fuses CPU operations)
                tfm_model.model = torch.compile(tfm_model.model)
                logging.info(" -> CPU Optimizations applied successfully.")
        except Exception as e:
            logging.info(f" -> [Warning] Could not apply CPU optimizations: {e}")
            
    logging.info(f"\nStarting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['lag_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        timesfm_dir = combo_output_path / "TimesFM"
        timesfm_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"\n -> Processing {combo_dir_name}...")
        
        with ResourceProfiler(run_name=combo_dir_name, input_file_path=DATA_PATH, output_dir_path=combo_output_path):
            for i in range(0, len(company_ids), BATCH_SIZE):
                batch_cols = company_ids[i : i + BATCH_SIZE]
                batch_df = loaded_profiles_df[batch_cols]
                logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
                
                results = run_timesfm_pipeline(batch_df, config, tfm_model)
                
                if results["TimesFM"]:
                    df_timesfm = pd.DataFrame(results["TimesFM"])
                    df_timesfm.to_parquet(timesfm_dir / f"batch_{i}.parquet", index=False, compression="snappy")

                    if i == 0:
                        tsv_path = timesfm_dir / f"batch_{i}_.tsv"
                        df_timesfm.to_csv(tsv_path, sep="\t", index=False)
                    
                    del df_timesfm 
                del results
                
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")