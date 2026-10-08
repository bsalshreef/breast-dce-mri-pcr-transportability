import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# Load frozen external predictions
d = pd.read_csv("ISPY1_M4_EXTERNAL_predictions_CORRECTED.csv")

y = d["pCR"].astype(int).to_numpy()
p = d["M4_prob"].to_numpy()

# Point estimates
auroc = roc_auc_score(y, p)
auprc = average_precision_score(y, p)
brier = brier_score_loss(y, p)

# Patient-level bootstrap
rng = np.random.default_rng(42)
results = []

while len(results) < 5000:
    idx = rng.integers(0, len(y), len(y))

    # AUROC requires both outcome classes
    if len(np.unique(y[idx])) < 2:
        continue

    results.append(
        [
            roc_auc_score(y[idx], p[idx]),
            average_precision_score(y[idx], p[idx]),
            brier_score_loss(y[idx], p[idx]),
        ]
    )

results = np.asarray(results)

ci_auroc = np.quantile(results[:, 0], [0.025, 0.975])
ci_auprc = np.quantile(results[:, 1], [0.025, 0.975])
ci_brier = np.quantile(results[:, 2], [0.025, 0.975])

print("N =", len(y))
print("pCR+ =", int(y.sum()))
print()
print("AUROC =", auroc)
print("AUROC 95% CI =", ci_auroc)
print()
print("AUPRC =", auprc)
print("AUPRC 95% CI =", ci_auprc)
print()
print("Brier =", brier)
print("Brier 95% CI =", ci_brier)