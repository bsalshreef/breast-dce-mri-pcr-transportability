import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# ============================================================
# M3 LOCKED CV
# MRI-only radiomics model
# Uses EXACT SAME locked outer folds as M2/M4
# ============================================================

SEED = 20261003
N_BOOT = 2000

RAD_FILE = "ISPY2_radiomics_FINAL_982.csv"
FOLD_FILE = "ISPY2_CV5_FOLDS_LOCKED.csv"

OUT_PRED = "M3_LOCKED_OOF_predictions.csv"
OUT_FOLD = "M3_LOCKED_fold_results.csv"
OUT_SUMMARY = "M3_LOCKED_performance_summary.csv"

ID_COL = "PatientID"
OUTCOME = "pCR"

print("=" * 80)
print("M3 MRI-ONLY RADIOMICS — LOCKED 5-FOLD CV")
print("=" * 80)

# ------------------------------------------------------------
# 1. Load radiomics
# ------------------------------------------------------------
rad = pd.read_csv(RAD_FILE)

if ID_COL not in rad.columns:
    raise ValueError(f"{RAD_FILE}: missing {ID_COL}")

if OUTCOME not in rad.columns:
    raise ValueError(f"{RAD_FILE}: missing {OUTCOME}")

rad[ID_COL] = rad[ID_COL].astype(str)
rad[OUTCOME] = pd.to_numeric(rad[OUTCOME], errors="raise").astype(int)

print(f"Radiomics rows: {len(rad)}")
print(f"Unique patients: {rad[ID_COL].nunique()}")

if rad[ID_COL].duplicated().any():
    raise ValueError("Duplicate PatientID found in radiomics file.")

# ------------------------------------------------------------
# 2. Load LOCKED folds
# ------------------------------------------------------------
folds = pd.read_csv(FOLD_FILE)

required = [ID_COL, OUTCOME, "CV_Fold"]
missing = [c for c in required if c not in folds.columns]
if missing:
    raise ValueError(f"{FOLD_FILE}: missing columns {missing}")

folds[ID_COL] = folds[ID_COL].astype(str)
folds[OUTCOME] = pd.to_numeric(folds[OUTCOME], errors="raise").astype(int)
folds["CV_Fold"] = pd.to_numeric(
    folds["CV_Fold"], errors="raise"
).astype(int)

print(f"Locked fold rows: {len(folds)}")
print(f"Locked unique patients: {folds[ID_COL].nunique()}")
print("Locked folds:", sorted(folds["CV_Fold"].unique().tolist()))

if folds[ID_COL].duplicated().any():
    raise ValueError("Duplicate PatientID found in locked fold file.")

# ------------------------------------------------------------
# 3. Merge and QC
# ------------------------------------------------------------
df = rad.merge(
    folds[[ID_COL, OUTCOME, "CV_Fold"]],
    on=ID_COL,
    how="inner",
    suffixes=("_rad", "_locked")
)

if len(df) != 982:
    raise ValueError(
        f"Expected 982 matched patients, found {len(df)}."
    )

if df[ID_COL].nunique() != 982:
    raise ValueError("Expected 982 unique patients after merge.")

disagree = int(
    (df[f"{OUTCOME}_rad"] != df[f"{OUTCOME}_locked"]).sum()
)

print(f"Matched patients: {len(df)}")
print(f"pCR disagreements: {disagree}")

if disagree != 0:
    raise ValueError("pCR mismatch between radiomics and locked fold file.")

df[OUTCOME] = df[f"{OUTCOME}_locked"]

# ------------------------------------------------------------
# 4. Identify radiomics features
# ------------------------------------------------------------
exclude_cols = {
    ID_COL,
    "pCR",
    "pCR_rad",
    "pCR_locked",
    "CV_Fold"
}

feature_cols = []

for c in rad.columns:
    if c in exclude_cols:
        continue

    vals = pd.to_numeric(rad[c], errors="coerce")

    # Keep columns that are genuinely numeric
    if vals.notna().sum() > 0:
        feature_cols.append(c)
        df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"Candidate radiomics features: {len(feature_cols)}")

if len(feature_cols) == 0:
    raise ValueError("No numeric radiomics features detected.")

X = df[feature_cols].copy()
y = df[OUTCOME].to_numpy(dtype=int)

# Remove globally unusable columns only:
# all missing, all non-finite, or constant across entire dataset.
# This uses no outcome information.
usable = []

for c in feature_cols:
    v = pd.to_numeric(X[c], errors="coerce")
    finite = v[np.isfinite(v)]

    if len(finite) == 0:
        continue

    if finite.nunique() <= 1:
        continue

    usable.append(c)

feature_cols = usable
X = X[feature_cols].copy()

print(f"Usable radiomics features: {len(feature_cols)}")

# ------------------------------------------------------------
# 5. Hyperparameter grid
# ------------------------------------------------------------
K_GRID = [5, 10, 20, 30, 50]
K_GRID = [k for k in K_GRID if k <= len(feature_cols)]

if len(K_GRID) == 0:
    K_GRID = [min(5, len(feature_cols))]

C_GRID = [0.01, 0.1, 1.0, 10.0]

print("K grid:", K_GRID)
print("C grid:", C_GRID)

# ------------------------------------------------------------
# Helper: training-only preprocessing
# ------------------------------------------------------------
def prepare_train_test(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Median imputation estimated ONLY from training data
    medians = X_train.median(axis=0)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Handle inf after training medians
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medians2 = X_train.median(axis=0)
    X_train = X_train.fillna(medians2)
    X_test = X_test.fillna(medians2)

    # If any column still unusable in this training fold, remove it
    good_cols = []

    for c in X_train.columns:
        tr = X_train[c].to_numpy(dtype=float)

        if not np.all(np.isfinite(tr)):
            continue

        if np.nanstd(tr) == 0:
            continue

        good_cols.append(c)

    X_train = X_train[good_cols]
    X_test = X_test[good_cols]

    return X_train, X_test


# ------------------------------------------------------------
# 6. Locked outer CV
# ------------------------------------------------------------
oof_pred = np.full(len(df), np.nan, dtype=float)
fold_results = []

locked_fold_values = sorted(df["CV_Fold"].unique())

for fold_value in locked_fold_values:

    print()
    print("=" * 80)
    print(f"OUTER LOCKED FOLD {fold_value}")
    print("=" * 80)

    test_idx = np.where(df["CV_Fold"].to_numpy() == fold_value)[0]
    train_idx = np.where(df["CV_Fold"].to_numpy() != fold_value)[0]

    X_train_raw = X.iloc[train_idx].copy()
    X_test_raw = X.iloc[test_idx].copy()

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Train N:", len(train_idx))
    print("Test N :", len(test_idx))
    print("Train positives:", int(y_train.sum()))
    print("Test positives :", int(y_test.sum()))

    # --------------------------------------------------------
    # INNER CV — training patients only
    # --------------------------------------------------------
    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED + int(fold_value)
    )

    best_score = -np.inf
    best_k = None
    best_C = None

    for k in K_GRID:
        for C in C_GRID:

            inner_scores = []

            for inner_train_rel, inner_val_rel in inner_cv.split(
                X_train_raw, y_train
            ):
                Xi_train_raw = X_train_raw.iloc[inner_train_rel].copy()
                Xi_val_raw = X_train_raw.iloc[inner_val_rel].copy()

                yi_train = y_train[inner_train_rel]
                yi_val = y_train[inner_val_rel]

                Xi_train, Xi_val = prepare_train_test(
                    Xi_train_raw,
                    Xi_val_raw
                )

                if Xi_train.shape[1] == 0:
                    continue

                k_use = min(k, Xi_train.shape[1])

                scaler = StandardScaler()
                Xi_train_scaled = scaler.fit_transform(Xi_train)
                Xi_val_scaled = scaler.transform(Xi_val)

                selector = SelectKBest(
                    score_func=f_classif,
                    k=k_use
                )

                Xi_train_sel = selector.fit_transform(
                    Xi_train_scaled,
                    yi_train
                )

                Xi_val_sel = selector.transform(
                    Xi_val_scaled
                )

                model = LogisticRegression(
                    C=C,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED
                )

                model.fit(Xi_train_sel, yi_train)

                pred = model.predict_proba(Xi_val_sel)[:, 1]

                score = roc_auc_score(yi_val, pred)
                inner_scores.append(score)

            if len(inner_scores) == 0:
                continue

            mean_score = float(np.mean(inner_scores))

            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_C = C

    if best_k is None or best_C is None:
        raise RuntimeError(
            f"Could not determine hyperparameters for fold {fold_value}"
        )

    print(
        "Best parameters:",
        {"model__C": best_C, "select__k": best_k}
    )
    print(f"Inner CV AUROC: {best_score:.4f}")

    # --------------------------------------------------------
    # Fit selected model to entire outer training set
    # --------------------------------------------------------
    X_train, X_test = prepare_train_test(
        X_train_raw,
        X_test_raw
    )

    k_use = min(best_k, X_train.shape[1])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(
        score_func=f_classif,
        k=k_use
    )

    X_train_sel = selector.fit_transform(
        X_train_scaled,
        y_train
    )

    X_test_sel = selector.transform(
        X_test_scaled
    )

    model = LogisticRegression(
        C=best_C,
        solver="liblinear",
        max_iter=5000,
        random_state=SEED
    )

    model.fit(X_train_sel, y_train)

    pred = model.predict_proba(X_test_sel)[:, 1]

    oof_pred[test_idx] = pred

    fold_auc = roc_auc_score(y_test, pred)
    fold_ap = average_precision_score(y_test, pred)
    fold_brier = brier_score_loss(y_test, pred)

    print(f"Outer AUROC: {fold_auc:.4f}")
    print(f"Outer AUPRC: {fold_ap:.4f}")
    print(f"Outer Brier: {fold_brier:.4f}")

    fold_results.append({
        "fold": int(fold_value),
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "test_positive": int(y_test.sum()),
        "test_negative": int(len(y_test) - y_test.sum()),
        "best_k": int(k_use),
        "best_C": float(best_C),
        "inner_best_AUROC": float(best_score),
        "outer_AUROC": float(fold_auc),
        "outer_AUPRC": float(fold_ap),
        "outer_Brier": float(fold_brier)
    })

# ------------------------------------------------------------
# 7. OOF completeness
# ------------------------------------------------------------
if np.isnan(oof_pred).any():
    raise RuntimeError(
        f"OOF predictions incomplete: {np.isnan(oof_pred).sum()} missing."
    )

print()
print("OOF predictions complete:", len(oof_pred))

# ------------------------------------------------------------
# 8. Overall performance
# ------------------------------------------------------------
auc = roc_auc_score(y, oof_pred)
ap = average_precision_score(y, oof_pred)
brier = brier_score_loss(y, oof_pred)

# Calibration intercept and slope
eps = 1e-6
p_clip = np.clip(oof_pred, eps, 1 - eps)
logit_pred = np.log(p_clip / (1 - p_clip))

cal_model = LogisticRegression(
    C=1e10,
    solver="lbfgs",
    max_iter=5000
)

cal_model.fit(
    logit_pred.reshape(-1, 1),
    y
)

cal_intercept = float(cal_model.intercept_[0])
cal_slope = float(cal_model.coef_[0][0])

# ------------------------------------------------------------
# 9. Patient-level bootstrap CIs
# ------------------------------------------------------------
rng = np.random.default_rng(SEED)

boot_auc = []
boot_ap = []
boot_brier = []

n = len(y)

for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)

    yb = y[idx]
    pb = oof_pred[idx]

    if len(np.unique(yb)) < 2:
        continue

    boot_auc.append(
        roc_auc_score(yb, pb)
    )

    boot_ap.append(
        average_precision_score(yb, pb)
    )

    boot_brier.append(
        brier_score_loss(yb, pb)
    )

def ci95(values):
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5))
    )

auc_ci = ci95(boot_auc)
ap_ci = ci95(boot_ap)
brier_ci = ci95(boot_brier)

print()
print("=" * 80)
print("M3 LOCKED OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 80)

print(
    f"AUROC: {auc:.4f} "
    f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]"
)

print(
    f"AUPRC: {ap:.4f} "
    f"[{ap_ci[0]:.4f}, {ap_ci[1]:.4f}]"
)

print(
    f"Brier: {brier:.4f} "
    f"[{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]"
)

print(f"Calibration intercept: {cal_intercept:.4f}")
print(f"Calibration slope: {cal_slope:.4f}")

# ------------------------------------------------------------
# 10. Save outputs
# ------------------------------------------------------------
pred_df = pd.DataFrame({
    ID_COL: df[ID_COL],
    OUTCOME: y,
    "CV_Fold": df["CV_Fold"].astype(int),
    "M3_pred": oof_pred
})

pred_df.to_csv(
    OUT_PRED,
    index=False
)

pd.DataFrame(fold_results).to_csv(
    OUT_FOLD,
    index=False
)

summary = pd.DataFrame([{
    "model": "M3_LOCKED",
    "N": len(y),
    "pCR_positive": int(y.sum()),
    "pCR_negative": int(len(y) - y.sum()),
    "AUROC": auc,
    "AUROC_CI_low": auc_ci[0],
    "AUROC_CI_high": auc_ci[1],
    "AUPRC": ap,
    "AUPRC_CI_low": ap_ci[0],
    "AUPRC_CI_high": ap_ci[1],
    "Brier": brier,
    "Brier_CI_low": brier_ci[0],
    "Brier_CI_high": brier_ci[1],
    "calibration_intercept": cal_intercept,
    "calibration_slope": cal_slope
}])

summary.to_csv(
    OUT_SUMMARY,
    index=False
)

print()
print("Files saved:")
print(" -", OUT_PRED)
print(" -", OUT_FOLD)
print(" -", OUT_SUMMARY)

print()
print("=" * 80)
print("M3 LOCKED CV COMPLETE")
print("=" * 80)

# ============================================================
# 10. FINAL FULL-DATA REFIT FOR EXTERNAL VALIDATION
# ============================================================

print()
print("=" * 80)
print("M3 FINAL FULL-DATA REFIT")
print("=" * 80)

# Choose final hyperparameters from locked nested-CV results.
# Rule: maximize mean inner-CV AUROC across the five outer folds
# for each (k, C) combination represented by the selected results.
fold_results_df = pd.DataFrame(fold_results)

# Conservative consensus fallback:
# use the most frequently selected C and k only if needed.
final_C = float(
    fold_results_df["best_C"].value_counts().index[0]
)
final_k = int(
    fold_results_df["best_k"].value_counts().index[0]
)

print("Final k:", final_k)
print("Final C:", final_C)

# Prepare ALL 982 I-SPY2 patients.
X_full_raw = df[feature_cols].copy()
y_full = df[OUTCOME].to_numpy(dtype=int)

# Full-data preprocessing is allowed here because this model is
# fitted only AFTER internal validation is complete and will be
# used for future locked external validation.
X_full = X_full_raw.replace([np.inf, -np.inf], np.nan)

full_medians = X_full.median(axis=0)
X_full = X_full.fillna(full_medians)

good_cols_full = []
for c in X_full.columns:
    vals = X_full[c].to_numpy(dtype=float)
    if np.all(np.isfinite(vals)) and np.nanstd(vals) > 0:
        good_cols_full.append(c)

X_full = X_full[good_cols_full]

final_scaler = StandardScaler()
X_full_scaled = final_scaler.fit_transform(X_full)

final_k = min(final_k, X_full_scaled.shape[1])

final_selector = SelectKBest(
    score_func=f_classif,
    k=final_k
)

X_full_selected = final_selector.fit_transform(
    X_full_scaled,
    y_full
)

final_model = LogisticRegression(
    C=final_C,
    solver="liblinear",
    max_iter=5000,
    random_state=SEED
)

final_model.fit(X_full_selected, y_full)

selected_features = np.array(good_cols_full)[
    final_selector.get_support()
]

pd.DataFrame({
    "selected_feature": selected_features
}).to_csv(
    "M3_FINAL_selected_features.csv",
    index=False
)

print("Full-data N:", len(y_full))
print("pCR positive:", int(y_full.sum()))
print("Number of usable radiomics features:", len(good_cols_full))
print("Number selected:", len(selected_features))
print("Final model fitted successfully.")
print("Saved: M3_FINAL_selected_features.csv")
print("=" * 80)