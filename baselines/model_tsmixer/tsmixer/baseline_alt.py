import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

print("="*60)
print("MODERN PYTORCH TS-MIXER LOCAL BASELINE RUN")
print("="*60)

# 1. Load Local Data Directly
csv_path = "./datasets/ETTh1.csv"
if not os.path.exists(csv_path):
    csv_path = "./dataset/ETT-small/ETTh1.csv" # Fallback mirror path Check

df = pd.read_csv(csv_path).set_index('date')
target_data = df.values # Multivariate array shape: [Total_Timesteps, 7]

# Chronological Train / Test split
train_end = 12 * 30 * 24
scaler = StandardScaler()
train_scaled = scaler.fit_transform(target_data[:train_end])
all_scaled = scaler.transform(target_data)

SEQ_LEN, PRED_LEN = 96, 24
NUM_FEATURES = target_data.shape[-1]

def create_windows(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

X, y = create_windows(all_scaled, SEQ_LEN, PRED_LEN)
split_idx = train_end
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 2. Clean, Modern TSMixer Layer Definition in PyTorch
class TSMixerBlock(nn.Module):
    def __init__(self, seq_len, num_features, ff_dim):
        super().__init__()
        self.temporal_norm = nn.LayerNorm([seq_len, num_features])
        self.temporal_fc = nn.Linear(seq_len, seq_len)
        
        self.feature_norm = nn.LayerNorm([seq_len, num_features])
        self.feature_fc1 = nn.Linear(num_features, ff_dim)
        self.feature_fc2 = nn.Linear(ff_dim, num_features)
        self.act = nn.ReLU()

    def forward(self, x):
        # Time Mixing Step
        res = x
        x = self.temporal_norm(x)
        x = x.transpose(1, 2) # [Batch, Features, Seq_Len]
        x = self.act(self.temporal_fc(x))
        x = x.transpose(1, 2) # [Batch, Seq_Len, Features]
        res = x + res
        
        # Feature Mixing Step
        x = self.feature_norm(res)
        x = self.act(self.feature_fc1(x))
        x = self.feature_fc2(x)
        return x + res

class TSMixer(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, ff_dim=64, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([TSMixerBlock(seq_len, num_features, ff_dim) for _ in range(n_blocks)])
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2) # [Batch, Features, Seq_Len]
        x = self.projection(x) # [Batch, Features, Pred_Len]
        return x.transpose(1, 2) # [Batch, Pred_Len, Features]

# 3. Fast Train Execution Run (1 Quick Epoch)
model = TSMixer(SEQ_LEN, PRED_LEN, NUM_FEATURES)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Running training iteration pass...")
model.train()
# Single batch trial for instant verification
optimizer.zero_grad()
output = model(X_train[:32])
loss = criterion(output, y_train[:32])
loss.backward()
optimizer.step()
print(f"Training step completed smoothly. Initial loss: {loss.item():.4f}")

# 4. Generate & Visualize Forecast
model.eval()
with torch.no_grad():
    sample_pred = model(X_test[0:1]).cpu().numpy()[0]

# Denormalize just the final channel column (OT) for plotting actual values
history_unscaled = scaler.inverse_transform(X_test[0].numpy())[:, -1]
actual_unscaled = scaler.inverse_transform(y_test[0].numpy())[:, -1]
pred_unscaled = scaler.inverse_transform(sample_pred)[:, -1]

plt.figure(figsize=(10, 4))
plt.plot(range(SEQ_LEN), history_unscaled, label="Historical Lookback Context", color="royalblue")
plt.plot(range(SEQ_LEN, SEQ_LEN + PRED_LEN), actual_unscaled, label="Ground Truth (Actuals)", color="green")
plt.plot(range(SEQ_LEN, SEQ_LEN + PRED_LEN), pred_unscaled, label="PyTorch TSMixer Forecast", color="crimson", linestyle="--")
plt.axvline(x=SEQ_LEN, color='black', linestyle=':', alpha=0.5)
plt.title("TSMixer Local PyTorch Verification Baseline")
plt.legend()
plt.tight_layout()
plt.show()