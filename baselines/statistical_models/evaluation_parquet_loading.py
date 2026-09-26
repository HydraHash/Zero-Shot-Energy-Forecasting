import pandas as pd

# Pandas automatically detects all batch files in this folder, 
# reads them, and combines them into one massive DataFrame instantly.
df_naive_all = pd.read_parquet("./prediction_output/combo_1_X7_Y1_S1/seasonal_naive/")

print(df_naive_all.shape)
# Output: (1825000, 6) -> All 5,000 plants over 365 days perfectly reassembled.