

import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
data = pd.read_pickle("step2_features.pkl") 
data.to_pickle("step2_features.pkl")

# ---- . Fit HMM ----
features = ["return", "amihud", "imbalance", "volatility"]
X = data[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=1000,
    random_state=42
)
model.fit(X_scaled)
data["regime"] = model.predict(X_scaled)

# ---- . Interpret ----
print("\n=== Regime Means ===")
print(data.groupby("regime")[features].mean().to_string())

print("\n=== Regime Counts ===")
print(data["regime"].value_counts())

print("\n=== Transition Matrix ===")
print(pd.DataFrame(
    model.transmat_,
    index=[f"from_{i}" for i in range(3)],
    columns=[f"to_{i}" for i in range(3)]
).round(3))

# ---- . Auto-label by volatility ----
vol_order = data.groupby("regime")["volatility"].mean().sort_values().index
label_map = {vol_order[0]: "normal", vol_order[1]: "stressed", vol_order[2]: "crisis"}
data["regime_label"] = data["regime"].map(label_map)

print("\n=== Regime Labels ===")
print(data["regime_label"].value_counts())

data.to_pickle("step3_hmm_regimes.pkl")
