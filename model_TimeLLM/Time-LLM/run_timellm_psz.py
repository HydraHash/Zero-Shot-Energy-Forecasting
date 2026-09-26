#Pseudo Zero-Shot Approach (In-domain fine tuning)
import os
import re
import time
import math
import psutil
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from pathlib import Path
from system_data_psutil import ResourceProfiler

# Optimize float32 matrix operations
torch.set_float32_matmul_precision("high")
torch.set_num_threads(20)


# =====================================================================
# 1. Parameter Configuration Matrix
# =====================================================================
OUTPUT_BASE_DIR = Path("./smallpsz_prediction_output_timellm")

logging.basicConfig(filename="smallpsz_timellm_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

EXPERIMENTAL_COMBINATIONS = {
    1: {"train_split_days": 30, "lag_days": 7, "horizon_days": 1, "step_days": 1},
    2: {"train_split_days": 30, "lag_days": 7, "horizon_days": 7, "step_days": 7},
}

# =====================================================================
# 2. Time-LLM Modular Components
# =====================================================================
class StandardNormalize(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True)
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
            return (x - self.mean) / self.stdev
        elif mode == 'denorm':
            return (x * self.stdev) + self.mean

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq_Len, 1]
        x = x.permute(0, 2, 1)  # [Batch, 1, Seq_Len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [Batch, 1, Num_Patches, Patch_Len]
        x = self.value_embedding(x)
        return self.dropout(x.squeeze(1))  # [Batch, Num_Patches, d_model]

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_llm, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        d_keys = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_embedding, source_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        q = self.query_projection(target_embedding).view(B, L, H, -1)
        k = self.key_projection(source_embedding).view(S, H, -1)
        v = self.value_projection(source_embedding).view(S, H, -1)

        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("blhe,she->bhls", q, k)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", attn, v).reshape(B, L, -1)
        return self.out_projection(out)

class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0.1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))

class TimeLLMModel(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8, d_model=32, d_ff=64, n_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_ff = d_ff

        # GPT-2 Backbone
        self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
        self.gpt2_config.num_hidden_layers = 4  # Truncated for lightweight compute
        self.llm_model = GPT2Model.from_pretrained('openai-community/gpt2', config=self.gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze LLM Backbone
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.d_llm = self.gpt2_config.n_embd
        self.normalize_layer = StandardNormalize()
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout=0.1)
        
        # Word Embedding Alignment
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.mapping_layer = nn.Linear(self.word_embeddings.shape[0], 500)
        self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, d_ff, self.d_llm)

        self.patch_nums = int((seq_len - patch_len) / stride + 1)
        self.output_projection = FlattenHead(self.d_ff * self.patch_nums, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, 1]
        x_norm = self.normalize_layer(x, mode='norm')
        
        # Patch & Reprogramming
        enc_out = self.patch_embedding(x_norm)
        source_embeds = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        reprogrammed_patches = self.reprogramming_layer(enc_out, source_embeds)
        
        # Prompt Prefix
        prompt_text = ["Electricity load profile time series forecasting:"] * x.shape[0]
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", padding=True).input_ids.to(x.device)
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids)
        
        # Concatenate and pass through frozen LLM
        combined_embeds = torch.cat([prompt_embeds, reprogrammed_patches], dim=1)
        hidden_states = self.llm_model(inputs_embeds=combined_embeds).last_hidden_state
        
        # Extract features and project to forecast horizon
        patch_hidden = hidden_states[:, -self.patch_nums:, :self.d_ff]
        out = self.output_projection(patch_hidden).unsqueeze(-1)
        
        return self.normalize_layer(out, mode='denorm')

def create_lag_features(series_values, lag_steps, horizon_steps):
    X, y = [], []
    for i in range(len(series_values) - lag_steps - horizon_steps + 1):
        X.append(series_values[i : i + lag_steps])
        y.append(series_values[i + lag_steps : i + lag_steps + horizon_steps])
    return np.array(X), np.array(y)

# =====================================================================
# 3. Training & Inference Pipeline
# =====================================================================
def run_timellm_indomain_pipeline(batch_df, config, device):
    train_split_steps = config["train_split_days"] * 96
    lag_steps = config["lag_days"] * 96
    horizon_steps = config["horizon_days"] * 96
    s_steps = config["step_days"] * 96
    
    total_steps = len(batch_df)
    strategy_data = {"timellm_indomain": []}
    
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
        
    X_tensor = torch.tensor(np.vstack(global_X_train), dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(np.vstack(global_y_train), dtype=torch.float32).unsqueeze(-1)
    
    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # ---------------------------------------------------------
    # PHASE 2: Train Adapter Bridge Layers
    # ---------------------------------------------------------
    model = TimeLLMModel(seq_len=lag_steps, pred_len=horizon_steps).to(device)
    
    # Only optimize trainable adapter bridge layers
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    criterion = nn.MSELoss()
    
    logging.info(f"      -> Phase 1: Fine-tuning Time-LLM adapters on first {config['train_split_days']} days...")
    model.train()
    for epoch in range(3):
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
            
            # Predict in sub-chunks if memory is constrained
            preds = model(X_test_tensor).squeeze(-1).cpu().numpy()
            
            for i, company_id in enumerate(batch_df.columns):
                row_timellm = {
                    "id": company_id,
                    "horizon_start": horizon_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "horizon_end": horizon_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "horizon_weekday": target_weekday_name,
                    "horizon_days": config["horizon_days"],
                    "actual_values": actual_block[company_id].values.ravel().tolist(),
                    "predicted_values": preds[i].tolist()
                }
                strategy_data["timellm_indomain"].append(row_timellm)
                
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
    return df_t.asfreq("15min").ffill().bfill()

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\load_profiles_enriched_small.tsv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loaded_profiles_df = load_vea_profiles(DATA_PATH)
    BATCH_SIZE = 268
    company_ids = loaded_profiles_df.columns
    
    for combo_id, config in EXPERIMENTAL_COMBINATIONS.items():
        combo_dir_name = f"c{combo_id}_train{config['train_split_days']}_horizon{config['horizon_days']}_step{config['step_days']}"
        combo_output_path = OUTPUT_BASE_DIR / combo_dir_name
        model_dir = combo_output_path / "timellm_indomain"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n -> Processing Time-LLM In-Domain (PSZ) for {combo_dir_name}...")
        with ResourceProfiler(run_name=combo_dir_name, input_file_path=DATA_PATH, output_dir_path=combo_output_path):
            for i in range(0, len(company_ids), BATCH_SIZE):
                batch_cols = company_ids[i : i + BATCH_SIZE]
                batch_df = loaded_profiles_df[batch_cols]
                logging.info(f"    Running Batch {i} to {i + len(batch_cols)}...")
                
                results = run_timellm_indomain_pipeline(batch_df, config, device)
                if results["timellm_indomain"]:
                    df_out = pd.DataFrame(results["timellm_indomain"])
                    df_out.to_parquet(model_dir / f"batch_{i}.parquet", index=False, compression="snappy")
                    if i == 0:
                        df_out.to_csv(model_dir / f"batch_{i}_.tsv", sep="\t", index=False)
                    del df_out
                del results

    logging.info(f"\nExecution loop complete. Results saved to: '{OUTPUT_BASE_DIR.absolute()}'")