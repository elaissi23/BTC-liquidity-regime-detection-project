exec(open("features.py").read())

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

features = ["return", "amihud_smooth", "imbalance", "volatility"]
X = data[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GaussianMixture(n_components=3, random_state=42)
model.fit(X_scaled)
data["regime"] = model.predict(X_scaled)

print("\n=== Regime Means ===")
print(data.groupby("regime")[features].mean().to_string())
print("\n=== Regime Counts ===")
regime_map = {0: "normal", 1: "stressed", 2: "crisis"}
data["regime_label"] = data["regime"].map(regime_map)
print(data["regime_label"].value_counts())