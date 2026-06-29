import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# ---- 1. Load HMM results ----
data=exec(open("features.py").read())
data = pd.read_pickle("data_with_hmm_regimes.pkl")

# ---- 2. Colors ----
color_map = {"normal": "green", "stressed": "orange", "crisis": "red"}
colors = data["regime_label"].map(color_map)

# ---- 3. Plot ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Top: price colored by regime
ax1.scatter(data["timestamp"], data["close"], c=colors, s=0.1, alpha=0.6)
ax1.set_ylabel("BTC Price (USDT)", fontsize=12)
ax1.set_title("BTC Price Colored by HMM Liquidity Regime", fontsize=14)

legend = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Normal'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Stressed'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Crisis'),
]
ax1.legend(handles=legend, loc="upper left")

# Bottom: volatility shaded by regime
ax2.plot(data["timestamp"], data["volatility"], color="gray", linewidth=0.3, alpha=0.7)
ax2.set_ylabel("Volatility", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_title("Rolling Volatility (HMM Regimes)", fontsize=14)

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
plt.savefig("regime_plot_hmm.png", dpi=150)
plt.show()
print("✅ saved as regime_plot_hmm.png")
