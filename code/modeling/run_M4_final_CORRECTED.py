import numpy as np
import pandas as pd
import joblib

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 2026

RADIOMICS_FILE = "ISPY2_radiomics_FINAL_982.csv"
CLINICAL_FILE = "ISPY2_MRI_master_manifest.csv"

OUTPUT_BUNDLE = "M4_FINAL_CORRECTED_model_bundle.joblib"
OUTPUT_FEATURES = "M4_FINAL_CORRECTED_selected_features.csv"

CLINICAL_COLS = ["age", "HR", "HER2", "log_tum_vol"]

# Locked choice derived from the completed nested CV:
# k=20 was selected in 3/5 outer folds.
# C=0.1 follows the original deterministic tie-breaking rule.
FINAL_K_RADIOMICS = 20
FINAL_C = 0.1


# ============================================================
# LOAD DEVELOPMENT DATA ONLY
# ============================================================

rad = pd.read_csv(RADIOMICS_FILE)
clin = pd.read_csv(CLINICAL_FILE)

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
# MERGE I-SPY2 DEVELOPMENT DATA
# ============================================================

data = rad.merge(
    clin,
    on="PatientID",
    how="inner",
    suffixes=("_rad", "_clin")
)

if len(data) != 982:
    raise ValueError(
        f"Expected 982 patients after merge, found {len(data)}"
    )

if data["PatientID"].nunique() != 982:
    raise ValueError("Patient IDs are not unique.")

if (data["pCR_rad"] != data["pCR_clin"]).any():
    raise ValueError(
        "pCR mismatch between radiomics and clinical data."
    )

data["pCR"] = data["pCR_rad"].astype(int)


# ============================================================
# DEFINE FEATURES
# ============================================================

radiomics_cols = [
    c for c in rad.columns
    if c not in ["PatientID", "pCR"]
]

if len(radiomics_cols) != 321:
    raise ValueError(
        f"Expected 321 radiomics features, found "
        f"{len(radiomics_cols)}"
    )

X_clin = data[CLINICAL_COLS].astype(float)
X_rad = data[radiomics_cols].astype(float)

y = data["pCR"].values


# ============================================================
# INTEGRITY CHECKS
# ============================================================

if X_clin.isna().any().any():
    raise ValueError("NaN detected in clinical predictors.")

if X_rad.isna().any().any():
    raise ValueError("NaN detected in radiomics predictors.")

if np.isinf(X_clin.to_numpy()).any():
    raise ValueError("Inf detected in clinical predictors.")

if np.isinf(X_rad.to_numpy()).any():
    raise ValueError("Inf detected in radiomics predictors.")

if len(np.unique(y)) != 2:
    raise ValueError("Outcome must be binary.")


# ============================================================
# CORRECTED FEATURE SELECTION
#
# IMPORTANT:
# The four clinical predictors are ALWAYS retained.
# SelectKBest is applied ONLY to the radiomics predictors,
# matching the M4 locked nested-CV model definition.
# ============================================================

rad_selector = SelectKBest(
    score_func=f_classif,
    k=FINAL_K_RADIOMICS
)

X_rad_selected = rad_selector.fit_transform(
    X_rad,
    y
)

selected_radiomics = np.array(radiomics_cols)[
    rad_selector.get_support()
].tolist()

if len(selected_radiomics) != FINAL_K_RADIOMICS:
    raise RuntimeError(
        "Unexpected number of selected radiomics features."
    )


# ============================================================
# COMBINE:
# 4 FIXED CLINICAL + 20 SELECTED RADIOMICS
# ============================================================

selected_features = CLINICAL_COLS + selected_radiomics

X_final = data[selected_features].astype(float)

if X_final.shape[1] != 4 + FINAL_K_RADIOMICS:
    raise RuntimeError(
        f"Expected 24 final predictors, found "
        f"{X_final.shape[1]}"
    )


# ============================================================
# STANDARDIZATION
# ============================================================

final_scaler = StandardScaler()

X_final_scaled = final_scaler.fit_transform(
    X_final
)


# ============================================================
# FINAL LOGISTIC REGRESSION
# ============================================================

final_model = LogisticRegression(
    penalty="l2",
    C=FINAL_C,
    solver="liblinear",
    max_iter=5000,
    random_state=RANDOM_STATE
)

final_model.fit(
    X_final_scaled,
    y
)


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

feature_type = (
    ["clinical"] * len(CLINICAL_COLS)
    + ["radiomics"] * len(selected_radiomics)
)

pd.DataFrame({
    "selected_feature": selected_features,
    "feature_type": feature_type
}).to_csv(
    OUTPUT_FEATURES,
    index=False
)


# ============================================================
# SAVE CORRECTED MODEL BUNDLE
# ============================================================

bundle = {
    "scaler": final_scaler,
    "model": final_model,
    "input_features": selected_features,
    "clinical_features": CLINICAL_COLS,
    "selected_radiomics": selected_radiomics,
    "final_k_radiomics": FINAL_K_RADIOMICS,
    "final_C": FINAL_C,
    "random_state": RANDOM_STATE,
    "development_cohort": "I-SPY2",
    "development_N": len(data),
    "pCR_positive": int(y.sum()),
    "pCR_negative": int(len(y) - y.sum()),
    "correction_note":
        "Corrected full-data M4 refit. Four clinical predictors "
        "are forced-in and SelectKBest is applied only to "
        "radiomics, matching the locked nested-CV M4 definition."
}

joblib.dump(
    bundle,
    OUTPUT_BUNDLE
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)
print("M4 FINAL CORRECTED REFIT")
print("=" * 80)

print("Development cohort: I-SPY2")
print("N:", len(data))
print("pCR positive:", int(y.sum()))
print("pCR negative:", int(len(y) - y.sum()))

print()
print("Clinical predictors forced in:")
for f in CLINICAL_COLS:
    print(" -", f)

print()
print(
    "Selected radiomics:",
    len(selected_radiomics)
)

for f in selected_radiomics:
    print(" -", f)

print()
print(
    "Total final predictors:",
    len(selected_features)
)

print("Final k radiomics:", FINAL_K_RADIOMICS)
print("Final C:", FINAL_C)

print()
print("Saved:", OUTPUT_FEATURES)
print("Saved:", OUTPUT_BUNDLE)

print()
print("CORRECTED M4 FINAL REFIT COMPLETE")
print("=" * 80)