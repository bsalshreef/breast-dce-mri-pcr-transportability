import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 2026

RADIOMICS_FILE = "ISPY2_radiomics_FINAL_982.csv"
CLINICAL_FILE = "ISPY2_MRI_master_manifest.csv"
FOLDS_FILE = "ISPY2_CV5_FOLDS_LOCKED.csv"

CLINICAL_COLS = ["age", "HR", "HER2", "log_tum_vol"]


# ============================================================
# LOAD DATA
# ============================================================

rad = pd.read_csv(RADIOMICS_FILE)
clin = pd.read_csv(CLINICAL_FILE)
folds = pd.read_csv(FOLDS_FILE)

print("=" * 80)
print("M4 LOCKED: CLINICAL + RADIOMICS")
print("=" * 80)

print("Radiomics:", rad.shape)
print("Clinical:", clin.shape)
print("Folds:", folds.shape)


# ============================================================
# PREPARE CLINICAL DATA
# ============================================================

clin = clin.copy()

if (clin["tum_vol"] <= 0).any():
    raise ValueError("Non-positive tumor volume detected.")

clin["log_tum_vol"] = np.log(clin["tum_vol"])

clinical_keep = [
    "PatientID",
    "pCR",
    "age",
    "HR",
    "HER2",
    "log_tum_vol"
]

clin = clin[clinical_keep]


# ============================================================
# MERGE
# ============================================================

data = rad.merge(
    clin,
    on="PatientID",
    how="inner",
    suffixes=("_rad", "_clin")
)

data = data.merge(
    folds[["PatientID", "CV_Fold"]],
    on="PatientID",
    how="inner"
)

if len(data) != 982:
    raise ValueError(f"Expected 982 patients after merge, found {len(data)}")

if data["PatientID"].nunique() != 982:
    raise ValueError("Patient IDs are not unique.")

if (data["pCR_rad"] != data["pCR_clin"]).any():
    raise ValueError("pCR mismatch between radiomics and clinical data.")

data["pCR"] = data["pCR_rad"].astype(int)

print("\nMerged N:", len(data))
print("Unique patients:", data["PatientID"].nunique())
print("pCR counts:")
print(data["pCR"].value_counts().sort_index())
print("Locked folds:")
print(data["CV_Fold"].value_counts().sort_index())


# ============================================================
# DEFINE FEATURES
# ============================================================

exclude = {
    "PatientID",
    "pCR_rad",
    "pCR_clin",
    "pCR",
    "CV_Fold",
    "age",
    "HR",
    "HER2",
    "log_tum_vol"
}

radiomics_cols = [
    c for c in rad.columns
    if c not in ["PatientID", "pCR"]
]

print("\nClinical features:", CLINICAL_COLS)
print("Radiomics features:", len(radiomics_cols))

if len(radiomics_cols) != 321:
    print("WARNING: expected 321 radiomics features, found",
          len(radiomics_cols))


# ============================================================
# CUSTOM TRANSFORMER:
# Select K radiomics features ONLY,
# while always retaining clinical predictors
# ============================================================

class ClinicalRadiomicsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_clinical=4, k=20):
        self.n_clinical = n_clinical
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)

        X_rad = X[:, self.n_clinical:]

        self.selector_ = SelectKBest(
            score_func=f_classif,
            k=min(self.k, X_rad.shape[1])
        )

        self.selector_.fit(X_rad, y)

        return self

    def transform(self, X):
        X = np.asarray(X)

        X_clin = X[:, :self.n_clinical]
        X_rad = X[:, self.n_clinical:]

        X_rad_selected = self.selector_.transform(X_rad)

        return np.hstack([X_clin, X_rad_selected])


# ============================================================
# DESIGN MATRIX
# clinical predictors FIRST, radiomics SECOND
# ============================================================

feature_cols = CLINICAL_COLS + radiomics_cols

X = data[feature_cols].astype(float).values
y = data["pCR"].values
patient_ids = data["PatientID"].values

outer_folds = data["CV_Fold"].values

# Locked-fold integrity checks
expected_folds = {1, 2, 3, 4, 5}
observed_folds = set(pd.Series(outer_folds).dropna().astype(int).unique())
if observed_folds != expected_folds:
    raise ValueError(
        f"Locked CV folds must be exactly {sorted(expected_folds)}, "
        f"found {sorted(observed_folds)}"
    )
if pd.Series(outer_folds).isna().any():
    raise ValueError("Missing locked CV fold assignments detected.")



if np.isnan(X).any():
    raise ValueError("NaN detected in M4 design matrix.")

if np.isinf(X).any():
    raise ValueError("Inf detected in M4 design matrix.")


# ============================================================
# PIPELINE
#
# IMPORTANT:
# feature selection and scaling are fit inside training data.
# ============================================================

pipeline = Pipeline([
    (
        "select",
        ClinicalRadiomicsSelector(
            n_clinical=len(CLINICAL_COLS)
        )
    ),
    ("scale", StandardScaler()),
    (
        "model",
        LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE
        )
    )
])


param_grid = {
    "select__k": [10, 20, 40],
    "model__C": [0.01, 0.1, 1.0, 10.0]
}


# ============================================================
# NESTED CV
# ============================================================

oof_pred = np.full(len(data), np.nan)

fold_results = []

unique_folds = sorted(np.unique(outer_folds))

for fold in unique_folds:

    print("\n" + "=" * 80)
    print(f"OUTER LOCKED FOLD {fold}")
    print("=" * 80)

    test_mask = outer_folds == fold
    train_mask = ~test_mask

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("Train N:", len(y_train))
    print("Test N :", len(y_test))
    print("Train positives:", int(y_train.sum()))
    print("Test positives :", int(y_test.sum()))

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE + int(fold)
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    pred = search.predict_proba(X_test)[:, 1]

    oof_pred[test_mask] = pred

    fold_auc = roc_auc_score(y_test, pred)
    fold_auprc = average_precision_score(y_test, pred)
    fold_brier = brier_score_loss(y_test, pred)

    best_k = search.best_params_["select__k"]
    best_c = search.best_params_["model__C"]

    print("Best parameters:", search.best_params_)
    print("Inner CV AUROC:", round(search.best_score_, 4))
    print("Outer AUROC:", round(fold_auc, 4))
    print("Outer AUPRC:", round(fold_auprc, 4))
    print("Outer Brier:", round(fold_brier, 4))

    fold_results.append({
        "fold": fold,
        "train_n": len(y_train),
        "test_n": len(y_test),
        "test_positive": int(y_test.sum()),
        "test_negative": int(len(y_test) - y_test.sum()),
        "best_k": best_k,
        "best_C": best_c,
        "inner_best_AUROC": search.best_score_,
        "outer_AUROC": fold_auc,
        "outer_AUPRC": fold_auprc,
        "outer_Brier": fold_brier
    })


# ============================================================
# CHECK OOF COMPLETENESS
# ============================================================

if np.isnan(oof_pred).any():
    raise RuntimeError("Missing OOF predictions detected.")

print("\nOOF predictions complete:", len(oof_pred))


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

auc = roc_auc_score(y, oof_pred)
auprc = average_precision_score(y, oof_pred)
brier = brier_score_loss(y, oof_pred)


# ============================================================
# CALIBRATION INTERCEPT AND SLOPE
# logistic calibration:
# y ~ intercept + slope * logit(prediction)
# ============================================================

eps = 1e-6
p_clip = np.clip(oof_pred, eps, 1 - eps)

logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)

cal_model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=5000
)

cal_model.fit(logit_p, y)

cal_intercept = float(cal_model.intercept_[0])
cal_slope = float(cal_model.coef_[0][0])


# ============================================================
# PATIENT-LEVEL BOOTSTRAP CIs
# fixed OOF predictions
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

B = 2000

boot_auc = []
boot_auprc = []
boot_brier = []

n = len(y)

for _ in range(B):

    idx = rng.integers(0, n, n)

    y_b = y[idx]
    p_b = oof_pred[idx]

    if len(np.unique(y_b)) < 2:
        continue

    boot_auc.append(
        roc_auc_score(y_b, p_b)
    )

    boot_auprc.append(
        average_precision_score(y_b, p_b)
    )

    boot_brier.append(
        brier_score_loss(y_b, p_b)
    )


def ci(values):
    return np.percentile(values, [2.5, 97.5])


auc_ci = ci(boot_auc)
auprc_ci = ci(boot_auprc)
brier_ci = ci(boot_brier)


# ============================================================
# SAVE OUTPUTS
# ============================================================

pred_df = pd.DataFrame({
    "PatientID": patient_ids,
    "pCR": y,
    "CV_Fold": outer_folds,
    "M4_pred": oof_pred
})

pred_df.to_csv(
    "M4_LOCKED_OOF_predictions.csv",
    index=False
)

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    "M4_LOCKED_fold_results.csv",
    index=False
)

summary = pd.DataFrame([{
    "model": "M4_clinical_plus_radiomics",
    "N": len(y),
    "pCR_positive": int(y.sum()),
    "pCR_negative": int(len(y) - y.sum()),
    "AUROC": auc,
    "AUROC_CI_low": auc_ci[0],
    "AUROC_CI_high": auc_ci[1],
    "AUPRC": auprc,
    "AUPRC_CI_low": auprc_ci[0],
    "AUPRC_CI_high": auprc_ci[1],
    "Brier": brier,
    "Brier_CI_low": brier_ci[0],
    "Brier_CI_high": brier_ci[1],
    "Calibration_intercept": cal_intercept,
    "Calibration_slope": cal_slope
}])

summary.to_csv(
    "M4_LOCKED_performance_summary.csv",
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("M4 LOCKED OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 80)

print(f"AUROC: {auc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"Brier: {brier:.4f}")
print(f"Calibration intercept: {cal_intercept:.4f}")
print(f"Calibration slope: {cal_slope:.4f}")

print("\n95% bootstrap CIs")

print(
    f"AUROC: {auc:.4f} "
    f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]"
)

print(
    f"AUPRC: {auprc:.4f} "
    f"[{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]"
)

print(
    f"Brier: {brier:.4f} "
    f"[{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]"
)

print("\nFiles saved:")
print(" - M4_OOF_predictions.csv")
print(" - M4_fold_results.csv")
print(" - M4_performance_summary.csv")

print("\n" + "=" * 80)
print("M4 LOCKED CV COMPLETE")
print("=" * 80)