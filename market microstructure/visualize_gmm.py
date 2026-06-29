exec(open("features.py").read())
exec(open("regime_gmm.py").read())


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---- Color map ----
color_map = {"normal": "green", "stressed": "orange", "crisis": "red"}
colors = data["regime_label"].map(color_map)

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# --- Top: BTC Price colored by regime ---
ax1.scatter(data["timestamp"], data["close"], 
            c=colors, s=0.1, alpha=0.6)
ax1.set_ylabel("BTC Price (USDT)", fontsize=12)
ax1.set_title("BTC Price Colored by Liquidity Regime", fontsize=14)

# legend
from matplotlib.lines import Line2D
legend = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Normal'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Stressed'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Crisis'),
]
ax1.legend(handles=legend, loc="upper left")

# --- Bottom: Volatility by regime ---
ax2.plot(data["timestamp"], data["volatility"], color="gray", linewidth=0.3, alpha=0.7)
ax2.set_ylabel("Volatility", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_title("Rolling Volatility", fontsize=14)

# shade crisis periods on volatility plot
crisis = data[data["regime_label"] == "crisis"]
ax2.fill_between(data["timestamp"], data["volatility"],
                 where=data["regime_label"] == "crisis",
                 color="red", alpha=0.3, label="Crisis")
ax2.fill_between(data["timestamp"], data["volatility"],
                 where=data["regime_label"] == "stressed",
                 color="orange", alpha=0.2, label="Stressed")
ax2.legend(loc="upper left")

# format x axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("regime_plot.png", dpi=150)
plt.show()
print("✅ saved as regime_plot.png")