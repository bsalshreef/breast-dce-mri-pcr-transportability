import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# Internal locked OOF predictions: I-SPY2
internal = pd.read_csv("M2_locked_OOF_predictions.csv")

# External frozen predictions: I-SPY1
external = pd.read_csv("ISPY1_M2_EXTERNAL_predictions.csv")

# Detect internal prediction column
pred_candidates = [
    c for c in internal.columns
    if "pred" in c.lower() or "prob" in c.lower()
]

print("Internal columns:", list(internal.columns))
print("Candidate prediction columns:", pred_candidates)

# Expected locked M2 prediction column
if "M2_pred" in internal.columns:
    int_col = "M2_pred"
elif "M2_prob" in internal.columns:
    int_col = "M2_prob"
else:
    raise ValueError(
        "Could not identify M2 prediction column in internal OOF file."
    )

y_int = internal["pCR"].astype(int).to_numpy()
p_int = internal[int_col].to_numpy()

y_ext = external["pCR"].astype(int).to_numpy()
p_ext = external["M2_prob"].to_numpy()

def metrics(y, p):
    return np.array([
        roc_auc_score(y, p),
        average_precision_score(y, p),
        brier_score_loss(y, p),
    ])

int_point = metrics(y_int, p_int)
ext_point = metrics(y_ext, p_ext)
delta_point = ext_point - int_point

rng = np.random.default_rng(42)
boot = []

while len(boot) < 5000:
    ii = rng.integers(0, len(y_int), len(y_int))
    ie = rng.integers(0, len(y_ext), len(y_ext))

    if len(np.unique(y_int[ii])) < 2:
        continue
    if len(np.unique(y_ext[ie])) < 2:
        continue

    boot.append(
        metrics(y_ext[ie], p_ext[ie]) -
        metrics(y_int[ii], p_int[ii])
    )

boot = np.asarray(boot)

low = np.quantile(boot, 0.025, axis=0)
high = np.quantile(boot, 0.975, axis=0)

names = ["AUROC", "AUPRC", "Brier"]

print()
print("Internal I-SPY2 N =", len(y_int))
print("External I-SPY1 N =", len(y_ext))
print("Differences are external - internal")
print()

for j, name in enumerate(names):
    print(name)
    print("Internal =", int_point[j])
    print("External =", ext_point[j])
    print("Delta =", delta_point[j])
    print("95% CI =", [low[j], high[j]])
    print()