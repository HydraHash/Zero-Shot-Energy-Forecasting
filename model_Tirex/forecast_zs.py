import os
import re
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path

# TiRex Import
from tirex import load_model

from system_data_psutil import ResourceProfiler

# Optimize CPU/GPU execution
torch.set_float32_matmul_precision("high")
torch.set_num_threads(20)

OUTPUT_BASE_DIR = Path("./smallc3_tirex_prediction_output")

logging.basicConfig(filename="smallc3_tirex_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

EXPERIMENTAL_COMBINATIONS = {
    #1: {"lag_days": 1, "horizon_days": 1, "step_days": 1},
    #2: {"lag_days": 7, "horizon_days": 7, "step_days": 7},
    3: {"lag_days": 7, "horizon_days": 1, "step_days": 1},
}


def run_tirex_pipeline(batch_df, config, model, device_str):
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    total_steps = len(batch_df)
    
    strategy_data = {"nxai_tirex": []}
    
    # Sliding Window Inference
    for start_idx in range(0, total_steps - lag_steps - horizon_steps + 1, s_steps):
        origin_idx = start_idx + lag_steps
        end_idx = origin_idx + horizon_steps
        
        horizon_start_dt = batch_df.index[origin_idx]
        horizon_end_dt = batch_df.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        # 1. Slice features (context) and actual ground truth
        lag_block = batch_df.iloc[start_idx:origin_idx]
        actual_block = batch_df.iloc[origin_idx:end_idx]
        
        # 2. Extract inputs into a 2D PyTorch Tensor: [Batch, Lookback_Length]
        context_arrays = [lag_block[cid].values.ravel() for cid in batch_df.columns]
        
        # TiRex's internal forecast method automatically moves batches to the model's device during generation
        input_tensor = torch.tensor(np.array(context_arrays), dtype=torch.float32)
        
        # 3. Generate Forecasts
        with torch.no_grad():
            # TiRex natively returns quantiles and a point forecast (mean)
            # Output type is set to "numpy" for easy slicing.
            quantiles, means = model.forecast(
                context=input_tensor, 
                prediction_length=horizon_steps, 
                batch_size=32,          # Native batching to prevent VRAM spikes
                output_type="numpy"
            )
            
        # 4. Map predictions back to the unified Parquet/TSV format
        for i, company_id in enumerate(batch_df.columns):
            y_actual_slice = actual_block[company_id].values.ravel().tolist()
            
            row = {
                "id": company_id,
                "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_weekday": target_weekday_name,
                "horizon_days": config["horizon_days"],
                "actual_values": y_actual_slice,
                "predicted_values": means[i].tolist() # Extract the point forecast
            }
            strategy_data["nxai_tirex"].append(row)
            
    return strategy_data


# =====================================================================
# 2. Vectorized Data Loader
# =====================================================================
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


# =====================================================================
# 3. Main Execution Block
# =====================================================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched_small.tsv"
    
    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)
    BATCH_SIZE = 268 
    company_ids = loaded_profiles_df.columns
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"HARDWARE DETECTED: {device_str.upper()}")
    logging.info("Loading NX-AI TiRex from Hugging Face...")
    
    if device_str == "cuda":
        try:
            # First attempt: custom CUDA kernels (Requires C++ Toolkit)
            logging.info("Attempting to load TiRex with custom CUDA xLSTM backend...")
            tirex_model = load_model("NX-AI/TiRex", backend="cuda")
        except Exception as e:
            # Fallback attempt: Standard PyTorch GPU execution
            logging.warning(f"Custom CUDA compile failed: {e}")
            logging.info("Falling back to pure PyTorch backend on CUDA...")
            tirex_model = load_model("NX-AI/TiRex", backend="torch", device="cuda")
    else:
        # Final fallback: Standard CPU execution
        logging.info("Loading TiRex with pure PyTorch backend on CPU...")
        tirex_model = load_model("NX-AI/TiRex", backend="torch", device="cpu")
        
        logging.info("Applying basic PyTorch JIT Compilation for CPU...")
        try:
            tirex_model.model = torch.compile(tirex_model.model)
        except Exception as e:
            logging.info(f" -> [Warning] Could not apply torch.compile: {e}")
            
    # -------------------------------------------------------------

    logging.info(f"Starting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['lag_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        tirex_dir = combo_output_path / "nxai_tirex"
        tirex_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing {combo_dir_name}...")
        
        with ResourceProfiler(run_name=combo_dir_name, input_file_path=DATA_PATH, output_dir_path=combo_output_path):
            for i in range(0, len(company_ids), BATCH_SIZE):
                batch_cols = company_ids[i : i + BATCH_SIZE]
                batch_df = loaded_profiles_df[batch_cols]
                logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
                
                results = run_tirex_pipeline(batch_df, config, tirex_model, device_str)
                
                if results["nxai_tirex"]:
                    df_out = pd.DataFrame(results["nxai_tirex"])
                    df_out.to_parquet(tirex_dir / f"batch_{i}.parquet", index=False, compression="snappy")
                    if i == 0:
                        df_out.to_csv(tirex_dir / f"batch_{i}_.tsv", sep="\t", index=False)
                    del df_out
                del results
                
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")