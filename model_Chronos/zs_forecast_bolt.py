import re
import numpy as np
import pandas as pd
import torch
import logging
from chronos import ChronosBoltPipeline
from pathlib import Path

# Optimize CPU/GPU execution
torch.set_float32_matmul_precision("high")
torch.set_num_threads(20)


EXPERIMENTAL_COMBINATIONS = {
    1: {"lag_days": 1, "horizon_days": 1, "step_days": 1},
    2: {"lag_days": 7, "horizon_days": 7, "step_days": 7},
    3: {"lag_days": 7, "horizon_days": 1, "step_days": 1},
    4: {"lag_days": 14, "horizon_days": 1, "step_days": 1}
}


def run_chronos_bolt_pipeline(batch_df, config, model, model_name):
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    total_steps = len(batch_df)
    
    strategy_data = {f"{model_name}": []}
    CHUNK_SIZE = 48 
    
    for start_idx in range(0, total_steps - lag_steps - horizon_steps + 1, s_steps):
        origin_idx = start_idx + lag_steps
        end_idx = origin_idx + horizon_steps
        
        horizon_start_dt = batch_df.index[origin_idx]
        horizon_end_dt = batch_df.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        lag_block = batch_df.iloc[start_idx:origin_idx]
        actual_block = batch_df.iloc[origin_idx:end_idx]
        
        context_arrays = [lag_block[cid].values.ravel() for cid in batch_df.columns]
        
        # Chronos-Bolt requires 2D shape: [Batch, Context_Length]
        rolling_context = torch.tensor(np.array(context_arrays), dtype=torch.float32)
        
        preds = []
        remaining_steps = horizon_steps
        
        while remaining_steps > 0:
            current_chunk = min(remaining_steps, CHUNK_SIZE)
            
            # Predict exact 0.5 quantile
            quantiles, _ = model.predict_quantiles(rolling_context, prediction_length=current_chunk, quantile_levels=[0.5])
            median_2d = quantiles[:, :, 0]
            preds.append(median_2d.cpu().numpy())
            
            # Concatenate along the Time Dimension 
            chunk_tensor = median_2d.cpu().float()
            rolling_context = torch.cat([rolling_context, chunk_tensor], dim=1)
            remaining_steps -= current_chunk
            
        point_forecast = np.concatenate(preds, axis=1)
        
        for i, company_id in enumerate(batch_df.columns):
            row = {
                "id": company_id,
                "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_weekday": target_weekday_name,
                "horizon_days": config["horizon_days"],
                "actual_values": actual_block[company_id].values.ravel().tolist(),
                "predicted_values": point_forecast[i].tolist()
            }
            strategy_data[f"{model_name}"].append(row)
            
    return strategy_data


def run_modelsize(model_name, data_path, batch_size, loaded_profiles_df):
    OUTPUT_BASE_DIR = Path(f"./triplec3_{model_name}_prediction_output")
    company_ids = loaded_profiles_df.columns
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"HARDWARE DETECTED: {device_map.upper()}")
    
    logging.info(f"\nStarting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['lag_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        bolt_dir = combo_output_path / f"{model_name}"
        bolt_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing {combo_dir_name}...")
        logging.info(f"\nLoading Chronos-{model_name} Checkpoint...")
        if "21" in model_name: hf_name = "mini"
        elif "43" in model_name: hf_name = "small"
        else: hf_name = "tiny"

        logging.info(f"Running Model chronos-bolt-{hf_name} now")
        chronos_bolt = ChronosBoltPipeline.from_pretrained(f"amazon/chronos-bolt-{hf_name}", device_map=device_map, dtype=torch.bfloat16)
        
        for i in range(0, len(company_ids), batch_size):
            batch_cols = company_ids[i : i + batch_size]
            batch_df = loaded_profiles_df[batch_cols]
            logging.info(f"    [Chronos-Bolt] Running Batch {i} to {i + len(batch_cols)}...")
            
            results = run_chronos_bolt_pipeline(batch_df, config, chronos_bolt, model_name)
            
            if results[f"{model_name}"]:
                df_c2 = pd.DataFrame(results[f"{model_name}"])
                df_c2.to_parquet(bolt_dir / f"batch_{i}.parquet", index=False, compression="snappy", engine="pyarrow")
                if i == 0:
                    df_c2.to_csv(bolt_dir / f"batch_{i}_.tsv", sep="\t", index=False)
                del df_c2
            del results
            
        del chronos_bolt
                
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")


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
    DATA_PATH = r"example\Path\data.tsv"
    BATCH_SIZE = 500

    logging.basicConfig(filename="triplec3_chronosBolt_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)

    run_modelsize("Bolt9m", DATA_PATH, BATCH_SIZE, loaded_profiles_df)
    logging.info("----------------------------------------------------------------------------------------------------------------------------------")

    run_modelsize("Bolt21m", DATA_PATH, BATCH_SIZE, loaded_profiles_df)
    logging.info("----------------------------------------------------------------------------------------------------------------------------------")

    run_modelsize("Bolt43m", DATA_PATH, BATCH_SIZE, loaded_profiles_df)
    logging.info("----------------------------------------------------------------------------------------------------------------------------------")
