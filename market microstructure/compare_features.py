import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ---- Load already-computed regimes (instant, no 4-feature refit) ----
data = pd.read_pickle("step3_hmm_regimes.pkl")

# ---- Fit a volatility-ONLY HMM (fast — just 1 feature) ----
X_vol = data[["volatility"]].values
X_vol_scaled = StandardScaler().fit_transform(X_vol)

model_vol = GaussianHMM(n_components=3, n_iter=1000, random_state=42)
model_vol.fit(X_vol_scaled)
data["regime_vol_only"] = model_vol.predict(X_vol_scaled)

# ---- Align both label sets by volatility ranking ----
def relabel_by_vol(df, col):
    order = df.groupby(col)["volatility"].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order)}
    return df[col].map(mapping)

data["regime_ranked"] = relabel_by_vol(data, "regime")
data["regime_vol_ranked"] = relabel_by_vol(data, "regime_vol_only")

# ---- Compare ----
agreement = (data["regime_ranked"] == data["regime_vol_ranked"]).mean()
print(f"\nAgreement (volatility-aligned): {agreement:.2%}")