import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 2026

CLINICAL_FILE = "ISPY2_MRI_master_manifest.csv"
FOLDS_FILE = "ISPY2_CV5_FOLDS_LOCKED.csv"


# ============================================================
# LOAD AND MERGE
# ============================================================

clin = pd.read_csv(CLINICAL_FILE)
folds = pd.read_csv(FOLDS_FILE)

data = clin.merge(
    folds[["PatientID", "CV_Fold"]],
    on="PatientID",
    how="inner"
)

# Complete-case sensitivity analysis: derive exclusions from the original source metadata
SOURCE_METADATA_FILE = "BreastDCEDL_metadata_min_crop.csv"
source_meta = pd.read_csv(SOURCE_METADATA_FILE)
source_spy2 = source_meta[source_meta["dataset"].astype(str).str.lower().eq("spy2")].copy()

if len(source_spy2) != 982:
    raise ValueError(f"Expected 982 I-SPY2 source records, found {len(source_spy2)}")

missing_age_pids = set(source_spy2.loc[source_spy2["age"].isna(), "pid"].astype(str))
if len(missing_age_pids) != 3:
    raise ValueError(f"Expected 3 source-metadata missing-age cases, found {len(missing_age_pids)}")

data = data[~data["pid"].astype(str).isin(missing_age_pids)].copy()

if len(data) != 979:
    raise ValueError(f"Expected 979 patients, found {len(data)}")

if data["PatientID"].nunique() != 979:
    raise ValueError("Duplicate PatientID detected.")

if (data["tum_vol"] <= 0).any():
    raise ValueError("Non-positive tumor volume detected.")

data["log_tum_vol"] = np.log(data["tum_vol"])

features = ["age", "HR", "HER2", "log_tum_vol"]

X = data[features].astype(float).values
y = data["pCR"].astype(int).values
outer_folds = data["CV_Fold"].values

if np.isnan(X).any():
    raise ValueError("NaN detected.")

if np.isinf(X).any():
    raise ValueError("Inf detected.")


print("=" * 75)
print("M2 CLINICAL MODEL — LOCKED 5-FOLD CV")
print("=" * 75)
print("N:", len(y))
print("pCR positive:", int(y.sum()))
print("pCR negative:", int(len(y) - y.sum()))
print("Features:", features)
print("\nFold counts:")
print(data["CV_Fold"].value_counts().sort_index())


# ============================================================
# LOCKED OUTER CV
# Scaling is fit ONLY on training fold.
# ============================================================

oof_pred = np.full(len(data), np.nan)
fold_results = []

for fold in sorted(np.unique(outer_folds)):

    train = outer_folds != fold
    test = outer_folds == fold

    X_train = X[train]
    y_train = y[train]

    X_test = X[test]
    y_test = y[test]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000
    )

    model.fit(X_train_s, y_train)

    pred = model.predict_proba(X_test_s)[:, 1]
    oof_pred[test] = pred

    auc = roc_auc_score(y_test, pred)
    ap = average_precision_score(y_test, pred)
    bs = brier_score_loss(y_test, pred)

    fold_results.append({
        "fold": fold,
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "test_positive": int(y_test.sum()),
        "test_negative": int(len(y_test) - y_test.sum()),
        "outer_AUROC": auc,
        "outer_AUPRC": ap,
        "outer_Brier": bs
    })

    print("\n" + "=" * 75)
    print("OUTER FOLD", fold)
    print("=" * 75)
    print("Train N:", int(train.sum()))
    print("Test N :", int(test.sum()))
    print(f"AUROC : {auc:.4f}")
    print(f"AUPRC : {ap:.4f}")
    print(f"Brier : {bs:.4f}")


if np.isnan(oof_pred).any():
    raise RuntimeError("Missing OOF predictions.")


# ============================================================
# OVERALL OOF PERFORMANCE
# ============================================================

auc = roc_auc_score(y, oof_pred)
ap = average_precision_score(y, oof_pred)
bs = brier_score_loss(y, oof_pred)


# Calibration intercept and slope
eps = 1e-6
p = np.clip(oof_pred, eps, 1 - eps)
lp = np.log(p / (1 - p)).reshape(-1, 1)

cal = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=5000
)

cal.fit(lp, y)

cal_intercept = float(cal.intercept_[0])
cal_slope = float(cal.coef_[0][0])


# ============================================================
# BOOTSTRAP CIs — FIXED OOF PREDICTIONS
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

boot_auc = []
boot_ap = []
boot_bs = []

n = len(y)

for _ in range(2000):

    idx = rng.integers(0, n, n)

    yb = y[idx]
    pb = oof_pred[idx]

    if len(np.unique(yb)) < 2:
        continue

    boot_auc.append(roc_auc_score(yb, pb))
    boot_ap.append(average_precision_score(yb, pb))
    boot_bs.append(brier_score_loss(yb, pb))


def ci(x):
    return np.percentile(x, [2.5, 97.5])


auc_ci = ci(boot_auc)
ap_ci = ci(boot_ap)
bs_ci = ci(boot_bs)


# ============================================================
# SAVE
# ============================================================

pd.DataFrame({
    "PatientID": data["PatientID"],
    "pCR": y,
    "CV_Fold": outer_folds,
    "M2_pred": oof_pred
}).to_csv(
    "M2_CC979_OOF_predictions.csv",
    index=False
)

pd.DataFrame(fold_results).to_csv(
    "M2_CC979_fold_results.csv",
    index=False
)

pd.DataFrame([{
    "model": "M2_clinical_CC979_sensitivity",
    "N": len(y),
    "AUROC": auc,
    "AUROC_CI_low": auc_ci[0],
    "AUROC_CI_high": auc_ci[1],
    "AUPRC": ap,
    "AUPRC_CI_low": ap_ci[0],
    "AUPRC_CI_high": ap_ci[1],
    "Brier": bs,
    "Brier_CI_low": bs_ci[0],
    "Brier_CI_high": bs_ci[1],
    "Calibration_intercept": cal_intercept,
    "Calibration_slope": cal_slope
}]).to_csv(
    "M2_CC979_performance_summary.csv",
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("M2 OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 75)

print(f"AUROC: {auc:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
print(f"AUPRC: {ap:.4f} [{ap_ci[0]:.4f}, {ap_ci[1]:.4f}]")
print(f"Brier: {bs:.4f} [{bs_ci[0]:.4f}, {bs_ci[1]:.4f}]")
print(f"Calibration intercept: {cal_intercept:.4f}")
print(f"Calibration slope: {cal_slope:.4f}")

print("\nFiles saved:")
print(" - M2_CC979_OOF_predictions.csv")
print(" - M2_CC979_fold_results.csv")
print(" - M2_CC979_performance_summary.csv")

print("\n" + "=" * 75)
print("M2 LOCKED CV COMPLETE")
print("=" * 75)