import pandas as pd
import zipfile, os

folder = "data/"
dfs = []

for f in sorted(os.listdir(folder)):
    if f.endswith(".zip"):
        with zipfile.ZipFile(folder + f) as z:
            csv_name = z.namelist()[0]
            df = pd.read_csv(z.open(csv_name), header=None)
            dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

data.columns = [
    "timestamp", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
data = data.drop(columns=["close_time", "ignore"])
data = data.sort_values("timestamp").reset_index(drop=True)

print(data.shape)
print(data.head())


data.to_pickle("step1_raw.pkl")


