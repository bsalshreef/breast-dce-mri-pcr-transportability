import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# Internal locked OOF predictions: I-SPY2
internal = pd.read_csv("M4_LOCKED_OOF_predictions.csv")

# External frozen predictions: I-SPY1
external = pd.read_csv("ISPY1_M4_EXTERNAL_predictions_CORRECTED.csv")

# Detect internal M4 prediction column
if "M4_pred" in internal.columns:
    int_col = "M4_pred"
elif "M4_prob" in internal.columns:
    int_col = "M4_prob"
else:
    raise ValueError(
        "Could not identify M4 prediction column in internal OOF file. "
        f"Columns: {list(internal.columns)}"
    )

# External prediction column
if "M4_prob" in external.columns:
    ext_col = "M4_prob"
elif "M4_pred" in external.columns:
    ext_col = "M4_pred"
else:
    raise ValueError(
        "Could not identify M4 prediction column in I-SPY1 file."
    )

y_int = internal["pCR"].astype(int).to_numpy()
p_int = internal[int_col].to_numpy()

y_ext = external["pCR"].astype(int).to_numpy()
p_ext = external[ext_col].to_numpy()

def metrics(y, p):
    return np.array([
        roc_auc_score(y, p),
        average_precision_score(y, p),
        brier_score_loss(y, p),
    ])

# Point estimates
int_point = metrics(y_int, p_int)
ext_point = metrics(y_ext, p_ext)

# Delta = external - internal
delta_point = ext_point - int_point

# Independent patient-level bootstrap
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
print("Internal pCR+ =", int(y_int.sum()))
print()

print("External I-SPY1 N =", len(y_ext))
print("External pCR+ =", int(y_ext.sum()))
print()

print("Differences are I-SPY1 external - I-SPY2 internal")
print()

for j, name in enumerate(names):
    print(name)
    print("Internal =", int_point[j])
    print("External =", ext_point[j])
    print("Delta =", delta_point[j])
    print("95% CI =", [low[j], high[j]])
    print()