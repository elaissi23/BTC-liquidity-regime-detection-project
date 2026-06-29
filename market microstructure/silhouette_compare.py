import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ---- load finished HMM result (no slow refit) ----
data = pd.read_pickle("step3_hmm_regimes.pkl")

features = ["return", "amihud", "imbalance", "volatility"]
X = StandardScaler().fit_transform(data[features].values)

# ---- HMM labels are already in the pickle (column 'regime') ----
data["regime_hmm"] = data["regime"]

# ---- GMM is fast, refit it here so both live in one dataframe ----
data["regime_gmm"] = GaussianMixture(n_components=3, random_state=42).fit_predict(X)

# ---- sample 20k rows: silhouette on 1.5M rows would hang ----
sample_idx = data.sample(20000, random_state=42).index
mask = data.index.isin(sample_idx)
X_sample = X[mask]

for col in ["regime_gmm", "regime_hmm"]:
    labels = data.loc[mask, col]
    score = silhouette_score(X_sample, labels)
    print(f"{col:12s} silhouette: {score:.4f}")