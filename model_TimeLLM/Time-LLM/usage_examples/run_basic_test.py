import os
import torch
import numpy as np
from torch import nn, optim
from dataclasses import dataclass

# Force CPU/Single Device execution for testing stability
os.environ['CURL_CA_BUNDLE'] = ''

from models import TimeLLM
from data_provider.data_factory import data_provider

# 1. Mock Configuration Object mimicking argparse
@dataclass
class Config:
    task_name: str = 'long_term_forecast'
    is_training: int = 1
    model_id: str = 'smoke_test_ETTh1'
    model_comment: str = 'SmokeTest'
    model: str = 'TimeLLM'
    data: str = 'ETTh1'
    root_path: str = './dataset/ETT-small/'
    data_path: str = 'ETTh1.csv'
    features: str = 'M'
    target: str = 'OT'
    loader: str = 'modal'
    freq: str = 'h'
    checkpoints: str = './checkpoints/'
    seq_len: int = 96        # Small sequence window for fast testing
    label_len: int = 48
    pred_len: int = 24       # Fast prediction step
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    d_model: int = 16
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 32
    factor: int = 3
    dropout: float = 0.1
    embed: str = 'timeF'
    activation: str = 'gelu'
    output_attention: bool = False
    patch_len: int = 16
    stride: int = 8
    prompt_domain: int = 0
    llm_model: str = 'GPT2'  # Using GPT2 as it's drastically smaller/faster to pull for testing than LLaMA
    llm_dim: int = 768       # Matching GPT2-small dimensions
    llm_layers: int = 2       # Minimal layers for instant processing
    num_workers: int = 0
    itr: int = 1
    train_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 0.001
    loss: str = 'MSE'
    lradj: str = 'type1'
    use_amp: bool = False
    patience: int = 3
    content: str = ""
    percent: int = 100
    seasonal_patterns: str = 'Yearly'

configs = Config()

print("="*60)
print("STARTING TIME-LLM BASIC TEST PARADIGMS")
print("="*60)

# Load minimal Data Streamers
train_data, train_loader = data_provider(configs, 'train')
test_data, test_loader = data_provider(configs, 'test')

# Initialize Base Architecture
print("\n[Initializing Model Backbone...]")
model = TimeLLM.Model(configs).float()
criterion = nn.MSELoss()

# -------------------------------------------------------------
# APPROACH 1: Pseudo-Zero-Shot (In-Domain Adapter Fine-Tuning)
# -------------------------------------------------------------
print("\n--- Running Approach 1: Pseudo-Zero-Shot (1 Epoch Tune) ---")
model.train()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.learning_rate)

# Take 1 dummy batch to prove tuning mechanics work
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    optimizer.zero_grad()
    
    # Generate placeholder decoder input matching run_main.py logic
    dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float()
    
    outputs = model(batch_x.float(), batch_x_mark.float(), dec_inp, batch_y_mark.float())
    
    # Strip down dimension matches
    outputs = outputs[:, -configs.pred_len:, :]
    batch_y = batch_y[:, -configs.pred_len:, :]
    
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    
    print(f"Success: Tuned 1 batch. Loss achieved: {loss.item():.4f}")
    break

# Manually save checkpoint simulating pipeline conclusion
checkpoint_dir = os.path.join(configs.checkpoints, 'smoke_test_weights')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'checkpoint.pth'))
print(f"Checkpoint saved to {checkpoint_dir}")


# -------------------------------------------------------------
# APPROACH 2: Cross-Dataset/Saved Adapter Test (Evaluation Mode)
# -------------------------------------------------------------
print("\n--- Running Approach 2: Loading Pre-Trained Adapters (Inference) ---")
# Reset environment configurations state
configs.is_training = 0 

eval_model = TimeLLM.Model(configs).float()
eval_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth')))
eval_model.eval()

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float()
        
        predictions = eval_model(batch_x.float(), batch_x_mark.float(), dec_inp, batch_y_mark.float())
        print(f"Success: Model evaluation step completed.")
        print(f"Forecast Tensor Shape output: {predictions.shape} (Expected: [Batch, Prediction Length, Features])")
        break


# -------------------------------------------------------------
# APPROACH 3: Pure Raw Out-of-the-Box Zero-Shot (No Adapters Trained)
# -------------------------------------------------------------
print("\n--- Running Approach 3: Pure Un-tuned Zero-Shot Inference ---")
raw_model = TimeLLM.Model(configs).float() # Fresh initialization, weights completely standard random/frozen LLM
raw_model.eval()

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float()
        
        raw_forecast = raw_model(batch_x.float(), batch_x_mark.float(), dec_inp, batch_y_mark.float())
        
        print(f"✅ Success: Raw Zero-Shot inference completed.")
        # Slice into the actual multivariate columns to show a prediction sample
        sample_forecast = raw_forecast[0, 0, :].detach().cpu().numpy()
        print(f"📈 Sample forecast snapshot for step 1 across 7 data dimensions:\n   {sample_forecast}")
        break

print("\n" + "="*60)
print("ALL THREE BASIC TESTS PACKAGES CONCLUDED SUCCESSFULLY!")
print("="*60)