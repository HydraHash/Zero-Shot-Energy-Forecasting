import os
import re
import time
import psutil
import threading
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from system_data_psutil import ResourceProfiler

# Optimize float32 matrix operations
torch.set_float32_matmul_precision("high")
torch.set_num_threads(20)

# =====================================================================
# 1. Parameter Configuration Matrix
# =====================================================================
OUTPUT_BASE_DIR = Path("./test_dlinear_prediction_output")

logging.basicConfig(filename="test_dlinear_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

EXPERIMENTAL_COMBINATIONS = {
    1: {"train_split_days": 30, "lag_days": 1, "horizon_days": 1, "step_days": 1},
    2: {"train_split_days": 30, "lag_days": 7, "horizon_days": 7, "step_days": 7},
}

# =====================================================================
# 2. DLinear PyTorch Implementation
# =====================================================================
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, Channel, Length]
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=2)
        return self.avg(x_pad)

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        # x: [Batch, Length, Channel]
        x_perm = x.permute(0, 2, 1)
        moving_mean = self.moving_avg(x_perm).permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean

class DLinearModel(nn.Module):
    """Decomposition Linear Model for Univariate Time Series[cite: 3]."""
    def __init__(self, seq_len, pred_len, moving_avg=25):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = SeriesDecomp(moving_avg)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, 1]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [Batch, Pred_Len, 1]

def create_lag_features(series_values, lag_steps, horizon_steps):
    X, y = [], []
    for i in range(len(series_values) - lag_steps - horizon_steps + 1):
        X.append(series_values[i : i + lag_steps])
        y.append(series_values[i + lag_steps : i + lag_steps + horizon_steps])
    return np.array(X), np.array(y)

# =====================================================================
# 3. Training & Inference Pipeline
# =====================================================================
def run_dlinear_pipeline(batch_df, config, device):
    train_split_steps = config["train_split_days"] * 96
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    
    total_steps = len(batch_df)
    strategy_data = {"dlinear": []}
    
    # ---------------------------------------------------------
    # PHASE 1: Build Global Training Dataset
    # ---------------------------------------------------------
    train_df = batch_df.iloc[:train_split_steps]
    global_X_train, global_y_train = [], []
    for company_id in batch_df.columns:
        series_vals = train_df[company_id].values.ravel()
        X_train, y_train = create_lag_features(series_vals, lag_steps, horizon_steps)
        global_X_train.append(X_train)
        global_y_train.append(y_train)
        
    X_train_matrix = np.vstack(global_X_train)
    y_train_matrix = np.vstack(global_y_train)
    
    X_tensor = torch.tensor(X_train_matrix, dtype=torch.float32).unsqueeze(-1)  # [N, Lag, 1]
    y_tensor = torch.tensor(y_train_matrix, dtype=torch.float32).unsqueeze(-1)  # [N, Horizon, 1]
    
    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # ---------------------------------------------------------
    # PHASE 2: Train Model Once
    # ---------------------------------------------------------
    model = DLinearModel(seq_len=lag_steps, pred_len=horizon_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    logging.info(f"      -> Phase 1: Training Global DLinear on first {config['train_split_days']} days...")
    model.train()
    for epoch in range(5):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    logging.info(f"      -> Phase 2: Generating rolling predictions...")
    model.eval()
    eval_start_idx = train_split_steps - lag_steps
    
    with torch.no_grad():
        for start_idx in range(eval_start_idx, total_steps - lag_steps - horizon_steps + 1, s_steps):
            origin_idx = start_idx + lag_steps
            end_idx = origin_idx + horizon_steps
            
            horizon_start_dt = batch_df.index[origin_idx]
            horizon_end_dt = batch_df.index[end_idx - 1]
            target_weekday_name = horizon_start_dt.strftime("%A")
            
            lag_block = batch_df.iloc[start_idx:origin_idx]
            actual_block = batch_df.iloc[origin_idx:end_idx]
            
            X_test_np = np.array([lag_block[cid].values.ravel() for cid in batch_df.columns])
            X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(-1).to(device)
            
            preds = model(X_test_tensor).squeeze(-1).cpu().numpy()  # [500, Horizon]
            
            for i, company_id in enumerate(batch_df.columns):
                row_dlinear = {
                    "id": company_id,
                    "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "horizon_weekday": target_weekday_name,
                    "horizon_days": config["horizon_days"],
                    "actual_values": actual_block[company_id].values.ravel().tolist(),
                    "predicted_values": preds[i].tolist()
                }
                strategy_data["dlinear"].append(row_dlinear)
                
    return strategy_data

# =====================================================================
# 4. Data Loader & Execution Block
# =====================================================================
def load_vea_profiles(file_path) -> pd.DataFrame:
    file_path = Path(file_path)
    raw_df = pd.read_pickle(file_path) if file_path.suffix.lower() == '.pkl' else pd.read_csv(file_path, sep="\t")

    first_col_name = str(raw_df.columns[0])
    if not re.search(r"\d{4}-\d{2}-\d{2}", first_col_name):
        raw_df.set_index(raw_df.columns[0], inplace=True)

    valid_time_cols, parsed_timestamps = [], []
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}).*?(\d{1,2}:\d{2})")

    for col in raw_df.columns:
        match = pattern.search(str(col))
        if match:
            date_p, time_p = match.groups()
            parsed_timestamps.append(pd.to_datetime(f"{date_p} {time_p}"))
            valid_time_cols.append(col)

    raw_df = raw_df[valid_time_cols]
    df_t = raw_df.T
    df_t.index = pd.DatetimeIndex(parsed_timestamps)
    df_t = df_t[~df_t.index.duplicated(keep='first')].sort_index()
    df_t = df_t.loc[:, ~df_t.columns.duplicated(keep='first')]

    logging.info(f"Successfully loaded continuous grid: {df_t.shape[1]} companies across {df_t.shape[0]} timestamps.")
    return df_t.asfreq("15min").ffill().bfill()


if __name__ == "__main__":
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched_very_small.tsv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loaded_profiles_df = load_vea_profiles(DATA_PATH)
    BATCH_SIZE = 500
    company_ids = loaded_profiles_df.columns

    logging.info(f"\nStarting evaluation loop across {len(EXPERIMENTAL_COMBINATIONS)} configurations...")

    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_train{config['train_split_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        model_dir = combo_output_path / "dlinear"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing DLinear for {combo_dir_name}...")
        with ResourceProfiler(run_name=combo_dir_name, input_file_path=DATA_PATH, output_dir_path=combo_output_path):
            for i in range(0, len(company_ids), BATCH_SIZE):
                batch_cols = company_ids[i : i + BATCH_SIZE]
                batch_df = loaded_profiles_df[batch_cols]
                logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
                
                results = run_dlinear_pipeline(batch_df, config, device)
                if results["dlinear"]:
                    df_out = pd.DataFrame(results["dlinear"])
                    df_out.to_parquet(model_dir / f"batch_{i}.parquet", index=False, compression="snappy")
                    if i == 0:
                        df_out.to_csv(model_dir / f"batch_{i}_.tsv", sep="\t", index=False)
                    del df_out
                del results

    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")