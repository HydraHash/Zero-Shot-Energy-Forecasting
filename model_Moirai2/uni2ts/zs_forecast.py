import re
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# Optimize CPU/GPU execution
torch.set_float32_matmul_precision("high")
torch.set_num_threads(20)


EXPERIMENTAL_COMBINATIONS = {
    1: {"lag_days": 1, "horizon_days": 1, "step_days": 1},
    2: {"lag_days": 7, "horizon_days": 7, "step_days": 7},
    3: {"lag_days": 7, "horizon_days": 1, "step_days": 1},
    4: {"lag_days": 14, "horizon_days": 1, "step_days": 1}
}


def run_moirai_pipeline(batch_df, config, predictor):
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    total_steps = len(batch_df)

    strategy_data = {"Moirai": []}
    
    for start_idx in range(0, total_steps - lag_steps - horizon_steps + 1, s_steps):
        origin_idx = start_idx + lag_steps
        end_idx = origin_idx + horizon_steps
        
        horizon_start_dt = batch_df.index[origin_idx]
        horizon_end_dt = batch_df.index[end_idx - 1]
        target_weekday_name = horizon_start_dt.strftime("%A")
        
        # Slice features (context) and actual ground truth
        lag_block = batch_df.iloc[start_idx:origin_idx]
        actual_block = batch_df.iloc[origin_idx:end_idx]
        
        # Convert raw pandas data into a GluonTS ListDataset
        start_dt = lag_block.index[0]
        ds_list = [
            {
                "target": lag_block[cid].values.astype(np.float32), 
                "start": start_dt, 
                "item_id": cid
            } 
            for cid in batch_df.columns
        ]
        input_ds = ListDataset(ds_list, freq="15min")
        forecasts = list(predictor.predict(input_ds))
        
        # Map predictions back to the unified Parquet/TSV format
        for i, company_id in enumerate(batch_df.columns):
            y_actual_slice = actual_block[company_id].values.ravel().tolist()
            
            # Extract the median from the probabilistic samples
            pred_median = forecasts[i].quantile(0.5)
            
            row = {
                "id": company_id,
                "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_weekday": target_weekday_name,
                "horizon_days": config["horizon_days"],
                "actual_values": y_actual_slice,
                "predicted_values": pred_median.tolist()
            }
            strategy_data["Moirai"].append(row)
            
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
    DATA_PATH = r"example\Path\data.tsv"
    BATCH_SIZE = 268
    MODEL_NAME = "Salesforce/moirai-2.0-R-small" #/moirai-1.1-R-small   /moirai-moe-1.0-R-small
    OUTPUT_BASE_DIR = Path("./smallc1_moirai20_prediction_output")

    logging.basicConfig(filename="smallc1_moirai20_log.txt",
                        filemode='a',
                        format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    
    loaded_profiles_df = load_vea_profiles(file_path=DATA_PATH)
    company_ids = loaded_profiles_df.columns
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logging.info(f"HARDWARE DETECTED: {device_str.upper()}")

    logging.info(f"Starting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_lag{config['lag_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        
        moirai_dir = combo_output_path / "Moirai"
        moirai_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing {combo_dir_name}...")
        logging.info(f"Loading {MODEL_NAME}")
        
        # Distinguished Execution API based on model
        if "moirai-2.0" in MODEL_NAME:
            moirai_module = Moirai2Module.from_pretrained(MODEL_NAME).to(device)
            moirai_model = Moirai2Forecast(
                module=moirai_module,
                prediction_length=config["horizon_days"] * 96,
                context_length=config["lag_days"] * 96,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        else:
            moirai_module = MoiraiModule.from_pretrained(MODEL_NAME).to(device)
            moirai_model = MoiraiForecast(
                module=moirai_module,
                prediction_length=config["horizon_days"] * 96,
                context_length=config["lag_days"] * 96,
                patch_size="auto",
                num_samples=20,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        
        # predictor attached to the correct device
        predictor = moirai_model.create_predictor(batch_size=32, device=device)
        
        for i in range(0, len(company_ids), BATCH_SIZE):
            batch_cols = company_ids[i : i + BATCH_SIZE]
            batch_df = loaded_profiles_df[batch_cols]
            logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
            
            results = run_moirai_pipeline(batch_df, config, predictor)
            
            if results["Moirai"]:
                df_out = pd.DataFrame(results["Moirai"])
                df_out.to_parquet(moirai_dir / f"batch_{i}.parquet", index=False, compression="snappy")
                if i == 0:
                    df_out.to_csv(moirai_dir / f"batch_{i}_.tsv", sep="\t", index=False)
                del df_out
            del results
                
        del moirai_model
        del predictor
        del moirai_module
                
    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")