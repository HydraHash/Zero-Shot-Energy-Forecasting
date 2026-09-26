import os
import re
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path

# MOMENT Import
from momentfm import MOMENTPipeline

from system_data_psutil import ResourceProfiler

# Optimize CPU/GPU execution
torch.set_float32_matmul_precision("high")
torch.set_num_threads(20)

OUTPUT_BASE_DIR = Path("./small_moment_prediction_output")

logging.basicConfig(filename="moment_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

EXPERIMENTAL_COMBINATIONS = {
    # MOMENT operates on a fixed 512-step window. Ensure Lag + Horizon <= 512.
    1: {"lag_days": 1, "horizon_days": 1, "step_days": 1},
    2: {"lag_days": 7, "horizon_days": 7, "step_days": 7}, 
}

# =====================================================================
# 1. MOMENT Zero-Shot Pipeline
# =====================================================================
def run_moment_pipeline(batch_df, config, model, device):
    MOMENT_SEQ_LEN = 512
    CHUNK_SIZE = 336  # Predict in 3.5-day chunks to leave 176 steps for context
    
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    total_steps = len(batch_df)
    
    strategy_data = {"autonlab_moment": []}
    
    # ---------------------------------------------------------
    # Sliding Window Inference
    # ---------------------------------------------------------
    for start_idx in range(0, total_steps - lag_steps - horizon_steps + 1, s_steps):
        origin_idx = start_idx + lag_steps
        end_idx = origin_idx + horizon_steps
        
        horizon_start_dt = batch_df.index[origin_idx]
        horizon_end_dt = batch_df.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        lag_block = batch_df.iloc[start_idx:origin_idx]
        actual_block = batch_df.iloc[origin_idx:end_idx]
        
        # 1. Extract true historical context into a 2D array
        context_arrays = [lag_block[cid].values.ravel() for cid in batch_df.columns]
        rolling_context = torch.tensor(np.array(context_arrays), dtype=torch.float32)
        
        preds = []
        remaining_steps = horizon_steps
        
        # 2. Autoregressive loop to bypass the 512-step limit
        while remaining_steps > 0:
            current_chunk = min(remaining_steps, CHUNK_SIZE)
            needed_history = MOMENT_SEQ_LEN - current_chunk
            
            # Slice the most recent history from our rolling context
            if rolling_context.shape[1] >= needed_history:
                history_tensor = rolling_context[:, -needed_history:]
                pad_size = 0
            else:
                # If we have less history than needed (e.g., early in the dataset), pad with zeros
                pad_size = needed_history - rolling_context.shape[1]
                pad_tensor = torch.zeros((rolling_context.shape[0], pad_size), dtype=torch.float32)
                history_tensor = torch.cat([pad_tensor, rolling_context], dim=1)
                
            future_zeros = torch.zeros((rolling_context.shape[0], current_chunk), dtype=torch.float32)
            
            # Combine [History] + [Masked Future]. MOMENT expects shape: [Batch, Channels (1), Seq_Len (512)]
            input_tensor = torch.cat([history_tensor, future_zeros], dim=1).unsqueeze(1).to(device)
            
            # Create Input Mask: 1 = observed, 0 = missing/padding
            input_mask = torch.ones((rolling_context.shape[0], MOMENT_SEQ_LEN), dtype=torch.float32).to(device)
            if pad_size > 0:
                input_mask[:, :pad_size] = 0  # Ignore front padding
            
            # Mask the future horizon to force MOMENT's reconstruction head to impute it
            input_mask[:, -current_chunk:] = 0  
            
            # 3. Generate Chunk Forecast
            with torch.no_grad():
                output = model(x_enc=input_tensor, input_mask=input_mask)
                
                # Output shape: [Batch, 1, 512]. Extract the predicted chunk.
                chunk_forecast = output.reconstruction[:, 0, -current_chunk:].cpu()
                
            # Store the predictions
            preds.append(chunk_forecast.numpy().astype(np.float32))
            
            rolling_context = torch.cat([rolling_context, chunk_forecast], dim=1)
            
            remaining_steps -= current_chunk
            
        # 4. Concatenate all generated chunks into the full horizon array
        point_forecasts = np.concatenate(preds, axis=1)
            
        # 5. Map predictions back to the unified Parquet/TSV format
        for i, company_id in enumerate(batch_df.columns):
            y_actual_slice = actual_block[company_id].values.ravel().tolist()
            
            row = {
                "id": company_id,
                "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_weekday": target_weekday_name,
                "horizon_days": config["horizon_days"],
                "actual_values": y_actual_slice,
                "predicted_values": point_forecasts[i].tolist()
            }
            strategy_data["autonlab_moment"].append(row)
            
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
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched.tsv"
    
    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)
    BATCH_SIZE = 500 
    company_ids = loaded_profiles_df.columns
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    logging.info(f"HARDWARE DETECTED: {device_str.upper()}")
    
    # -------------------------------------------------------------
    # MODEL INITIALIZATION (MOMENT)
    # -------------------------------------------------------------
    logging.info("Loading MOMENT-1-small from Hugging Face...")
    
    moment_model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", model_kwargs={'task_name': 'reconstruction',})
    moment_model.init()
    moment_model = moment_model.to(device)
    moment_model.eval()
    
    logging.info(f"Starting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['lag_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        moment_dir = combo_output_path / "autonlab_moment"
        moment_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing {combo_dir_name}...")
        
        with ResourceProfiler(run_name=combo_dir_name, input_file_path=DATA_PATH, output_dir_path=combo_output_path):
            for i in range(0, len(company_ids), BATCH_SIZE):
                batch_cols = company_ids[i : i + BATCH_SIZE]
                batch_df = loaded_profiles_df[batch_cols]
                logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
                
                results = run_moment_pipeline(batch_df, config, moment_model, device)
                
                if results["autonlab_moment"]:
                    df_out = pd.DataFrame(results["autonlab_moment"])
                    # Engine forced to pyarrow to prevent 32-bit overflow
                    df_out.to_parquet(moment_dir / f"batch_{i}.parquet", index=False, compression="snappy", engine="pyarrow")
                    if i == 0:
                        df_out.to_csv(moment_dir / f"batch_{i}_.tsv", sep="\t", index=False)
                    del df_out
                del results
                
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")