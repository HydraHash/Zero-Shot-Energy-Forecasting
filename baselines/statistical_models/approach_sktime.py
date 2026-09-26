import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.naive import NaiveForecaster

# =====================================================================
# 1. DATA PREPARATION & TIME INDEXING
# =====================================================================
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
df = pd.read_csv(url, parse_dates=['date'])

df = df.set_index('date')
df = df.asfreq('h') 

y = df['OT']

# Let's cleanly define your window cuts:
plot_start = 120
context_length = 168  # 7 days of lookback history
pred_length = 24     # Day-ahead horizon

# y_train should capture the historical context window up to the forecast origin
y_train = y.iloc[plot_start : plot_start + context_length]  # Length: 168

# y_test captures the immediate subsequent 24 hours
y_test = y.iloc[plot_start + context_length : plot_start + context_length + pred_length]  # Length: 24

# Define the Forecasting Horizon relative to the end of y_train
fh = np.arange(1, pred_length + 1)

# =====================================================================
# 2. SKTIME BASELINE MODELING
# =====================================================================
# Baseline A: Seasonal Naïve (Strategy "last" with a seasonal period of 24)
forecaster_naive = NaiveForecaster(strategy="last", sp=24)
forecaster_naive.fit(y_train)
pred_seasonal_naive = forecaster_naive.predict(fh)

# Baseline B: Historical Average (Strategy "mean" with a seasonal period of 24)
forecaster_avg = NaiveForecaster(strategy="mean", sp=24)
forecaster_avg.fit(y_train)
pred_hist_average = forecaster_avg.predict(fh)

# =====================================================================
# 3. METRIC EVALUATION (Sanity Check)
# =====================================================================
def get_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

rmse_naive = get_rmse(y_test, pred_seasonal_naive)
rmse_avg = get_rmse(y_test, pred_hist_average)

print(f"--- ETTh1 Univariate Baseline Results ---")
print(f"Seasonal Naïve (sp=24) RMSE:    {rmse_naive:.4f}")
print(f"Historical Average (sp=24) RMSE: {rmse_avg:.4f}")

# =====================================================================
# 4. VISUALIZATION
# =====================================================================
plt.figure(figsize=(14, 6))

# Plot a small portion of the historical training data leading up to the forecast
plt.plot(y.iloc[plot_start:context_length], label="Past History (Context)", color="gray", alpha=0.5)
plt.plot(y_train.iloc[-48:], label="Immediate History (Last 48h)", color="black", linewidth=1.5)

# Plot Ground Truth
plt.plot(y_test, label="Ground Truth (Actuals)", color="green", linewidth=2, linestyle="-")

# Plot Baselines
plt.plot(pred_seasonal_naive, label=f"Seasonal Naïve (RMSE: {rmse_naive:.2f})", color="red", linestyle="--", linewidth=2)
plt.plot(pred_hist_average, label=f"Historical Average (RMSE: {rmse_avg:.2f})", color="blue", linestyle="-.", linewidth=2)

# Formatting the Chart
plt.title("ETTh1 Univariate Baseline Comparison (sktime Pipeline Evaluation)", fontsize=14, fontweight='bold')
plt.xlabel("Timeline", fontsize=12)
plt.ylabel("Oil Temperature (OT)", fontsize=12)
plt.axvline(x=y_train.index[-1], color="purple", linestyle=":", label="Forecast Origin Step", linewidth=2)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left")
plt.tight_layout()

# Render the plot
plt.show()