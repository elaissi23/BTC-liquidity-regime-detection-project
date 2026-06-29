# ============ 1. LOAD DATA ============
import pandas as pd
import numpy as np
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
data.columns = ["timestamp","open","high","low","close","volume",
                "close_time","quote_volume","trades",
                "taker_buy_volume","taker_buy_quote_volume","ignore"]
data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
data = data.drop(columns=["close_time","ignore"])
data = data.sort_values("timestamp").reset_index(drop=True)

# ============ 2. FEATURES ============
data["return"] = data["close"].pct_change()
data["amihud"] = abs(data["return"]) / data["quote_volume"].replace(0, np.nan)
data["imbalance"] = (data["taker_buy_volume"] / data["volume"].replace(0, np.nan)) * 2 - 1
data["volatility"] = data["return"].rolling(60).std()
data["amihud_smooth"] = data["amihud"].rolling(60).mean()
data = data.dropna().reset_index(drop=True)

# ============ 3. GMM REGIMES ============
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

features = ["return", "amihud_smooth", "imbalance", "volatility"]
X = data[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GaussianMixture(n_components=3, random_state=42)
model.fit(X_scaled)
data["regime"] = model.predict(X_scaled)
regime_map = {0: "normal", 1: "stressed", 2: "crisis"}
data["regime_label"] = data["regime"].map(regime_map)

print(data.groupby("regime")[features].mean().to_string())
print(data["regime_label"].value_counts())

# ============ 4. VISUALIZE ============
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

color_map = {"normal": "green", "stressed": "orange", "crisis": "red"}
colors = data["regime_label"].map(color_map)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

ax1.scatter(data["timestamp"], data["close"], c=colors, s=0.1, alpha=0.6)
ax1.set_ylabel("BTC Price (USDT)", fontsize=12)
ax1.set_title("BTC Price Colored by Liquidity Regime", fontsize=14)

legend = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Normal'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Stressed'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Crisis'),
]
ax1.legend(handles=legend, loc="upper left")

ax2.plot(data["timestamp"], data["volatility"], color="gray", linewidth=0.3, alpha=0.7)
ax2.set_ylabel("Volatility", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_title("Rolling Volatility", fontsize=14)
ax2.fill_between(data["timestamp"], data["volatility"],
                 where=data["regime_label"] == "crisis",
                 color="red", alpha=0.3, label="Crisis")
ax2.fill_between(data["timestamp"], data["volatility"],
                 where=data["regime_label"] == "stressed",
                 color="orange", alpha=0.2, label="Stressed")
ax2.legend(loc="upper left")

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("regime_plot.png", dpi=150)
plt.show()
print("✅ saved as regime_plot.png")