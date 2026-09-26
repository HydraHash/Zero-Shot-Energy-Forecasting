import pandas as pd
import numpy as np

# 1. Fetch the ETTh1 dataset directly from the official repository
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
df = pd.read_csv(url, parse_dates=['date'])

# For this univariate example, let's look at the target feature: 'OT' (Oil Temperature)
series = df['OT'].values 

# 2. Configure our Time-Series Dimensions
# ETTh1 is hourly data. 
# Your industrial dataset is 15-min data, so you would change these numbers accordingly!
SEASON_LENGTH = 24       # 24 hours per day
CONTEXT_WINDOW = 168     # 7 days of history (7 * 24)
HORIZON = 24             # 24-hour day-ahead forecast

# Let's extract a sample test slice
# x_history: the data the model is allowed to look at
# y_truth: the actual future values we want to evaluate against
x_history = series[:CONTEXT_WINDOW]
y_truth = series[CONTEXT_WINDOW : CONTEXT_WINDOW + HORIZON]

# ==========================================
# APPROACH 1: SEASONAL NAÏVE IMPLEMENTATION
# ==========================================
def seasonal_naive_forecast(history, horizon, season_length):
    """
    Projects the last complete seasonal cycle forward into the horizon.
    If the horizon is longer than one season, it repeats the cycle.
    """
    # Get the last complete cycle from our history
    last_cycle = history[-season_length:]
    
    # Repeat the cycle as many times as needed to fill the horizon
    repetitions = int(np.ceil(horizon / season_length))
    forecast = np.tile(last_cycle, repetitions)[:horizon]
    
    return forecast

# ==========================================
# APPROACH 2: HISTORICAL AVERAGE IMPLEMENTATION
# ==========================================
def historical_average_forecast(history, horizon, season_length):
    """
    Calculates the mean for each specific hour of the day across 
    all historical days available in the context window.
    """
    # Reshape history into a 2D matrix where rows = days, columns = hour of day
    # Note: Assumes history length is a perfect multiple of season length
    num_days = len(history) // season_length
    reshaped_history = history[-num_days * season_length:].reshape(num_days, season_length)
    
    # Calculate the average profile for a single day
    average_day_profile = np.mean(reshaped_history, axis=0)
    
    # Tile the average day profile to fit the target horizon
    repetitions = int(np.ceil(horizon / season_length))
    forecast = np.tile(average_day_profile, repetitions)[:horizon]
    
    return forecast

# 3. Generate Predictions
pred_seasonal_naive = seasonal_naive_forecast(x_history, HORIZON, SEASON_LENGTH)
pred_hist_average = historical_average_forecast(x_history, HORIZON, SEASON_LENGTH)

# ==========================================
# EVALUATION (The Sanity Check)
# ==========================================
def evaluate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

rmse_naive = evaluate_rmse(y_truth, pred_seasonal_naive)
rmse_avg = evaluate_rmse(y_truth, pred_hist_average)

print(f"--- Evaluation Results (Horizon: {HORIZON}h) ---")
print(f"Seasonal Naïve RMSE:    {rmse_naive:.4f}")
print(f"Historical Average RMSE: {rmse_avg:.4f}")