import pandas as pd
import numpy as np

# load your data first
data = pd.read_pickle("step1_raw.pkl")

# 1. Returns
data["return"] = data["close"].pct_change()

# 2. Amihud Illiquidity
data["amihud"] = abs(data["return"]) / data["quote_volume"].replace(0, np.nan)

# 3. Volume Imbalance (buy pressure vs total volume)
data["imbalance"] = (data["taker_buy_volume"] / data["volume"].replace(0, np.nan)) * 2 - 1

# 4. Rolling Volatility (1hr window = 60 1-min bars)
data["volatility"] = data["return"].rolling(60).std()

# 5. Rolling Amihud (smoothed)
data["amihud_smooth"] = data["amihud"].rolling(60).mean()

# drop nulls from rolling windows
data = data.dropna().reset_index(drop=True)

print(data[["timestamp","return","amihud","imbalance","volatility"]].head())

data.to_pickle("step2_features.pkl")