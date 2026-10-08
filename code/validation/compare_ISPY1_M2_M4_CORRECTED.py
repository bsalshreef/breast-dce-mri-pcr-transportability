import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# Load frozen external predictions
m2 = pd.read_csv("ISPY1_M2_EXTERNAL_predictions.csv")
m4 = pd.read_csv("ISPY1_M4_EXTERNAL_predictions_CORRECTED.csv")

# Keep only required columns and merge by patient ID
a = m2[["pid", "pCR", "M2_prob"]].copy()
b = m4[["pid", "pCR", "M4_prob"]].copy()

d = a.merge(
    b,
    on="pid",
    how="inner",
    suffixes=("_M2", "_M4")
)

# Verify outcome agreement
assert len(d) == 167
assert np.array_equal(
    d["pCR_M2"].to_numpy(),
    d["pCR_M4"].to_numpy()
)

y = d["pCR_M2"].astype(int).to_numpy()
p2 = d["M2_prob"].to_numpy()
p4 = d["M4_prob"].to_numpy()

def metrics(y, p):
    return np.array([
        roc_auc_score(y, p),
        average_precision_score(y, p),
        brier_score_loss(y, p),
    ])

m2_point = metrics(y, p2)
m4_point = metrics(y, p4)
delta_point = m4_point - m2_point

# Paired patient-level bootstrap
rng = np.random.default_rng(42)
boot = []

while len(boot) < 5000:
    idx = rng.integers(0, len(y), len(y))

    if len(np.unique(y[idx])) < 2:
        continue

    boot.append(
        metrics(y[idx], p4[idx]) -
        metrics(y[idx], p2[idx])
    )

boot = np.asarray(boot)

low = np.quantile(boot, 0.025, axis=0)
high = np.quantile(boot, 0.975, axis=0)

# Two-sided bootstrap p-value
pvals = np.array([
    min(
        1.0,
        2 * min(
            np.mean(boot[:, j] <= 0),
            np.mean(boot[:, j] >= 0)
        )
    )
    for j in range(3)
])

names = ["AUROC", "AUPRC", "Brier"]

print("Matched N =", len(y))
print("pCR+ =", int(y.sum()))
print()
print("Differences are M4 - M2")
print()

for j, name in enumerate(names):
    print(name)
    print("M2 =", m2_point[j])
    print("M4 =", m4_point[j])
    print("Delta =", delta_point[j])
    print("95% CI =", [low[j], high[j]])
    print("Bootstrap p =", pvals[j])
    print()