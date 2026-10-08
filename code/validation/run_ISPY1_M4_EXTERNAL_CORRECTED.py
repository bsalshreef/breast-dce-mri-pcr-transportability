import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "ISPY1_EXTERNAL_READY.csv"
MODEL_FILE = "M4_FINAL_CORRECTED_model_bundle.joblib"

OUTPUT_FILE = "ISPY1_M4_EXTERNAL_predictions_CORRECTED.csv"

CLINICAL_COLS = ["age", "HR", "HER2", "log_tum_vol"]


# ============================================================
# LOAD DATA AND CORRECTED MODEL
# ============================================================

d = pd.read_csv(DATA_FILE)
bundle = joblib.load(MODEL_FILE)

scaler = bundle["scaler"]
model = bundle["model"]

input_features = list(bundle["input_features"])
clinical_features = list(bundle["clinical_features"])
selected_radiomics = list(bundle["selected_radiomics"])

print("=" * 80)
print("I-SPY1 EXTERNAL VALIDATION — CORRECTED M4")
print("=" * 80)

print("Input rows:", len(d))
print("Corrected model predictors:", len(input_features))
print("Clinical predictors:", len(clinical_features))
print("Selected radiomics:", len(selected_radiomics))
print("Final C:", bundle["final_C"])


# ============================================================
# VERIFY MODEL DEFINITION
# ============================================================

if clinical_features != CLINICAL_COLS:
    raise ValueError(
        f"Unexpected clinical predictors: {clinical_features}"
    )

if len(input_features) != 24:
    raise ValueError(
        f"Expected 24 model predictors, found {len(input_features)}"
    )

if len(selected_radiomics) != 20:
    raise ValueError(
        f"Expected 20 selected radiomics, found "
        f"{len(selected_radiomics)}"
    )

for c in input_features:
    if c not in d.columns:
        raise ValueError(
            f"Required external predictor missing: {c}"
        )


# ============================================================
# COMPLETE-CASE EXTERNAL COHORT
#
# Same clinical completeness requirement used previously:
# age, HR, HER2, log_tum_vol must be available.
#
# Radiomics required by the frozen model must also be finite.
# ============================================================

required = ["pCR"] + input_features

ext = d.copy()

for c in required:
    ext[c] = pd.to_numeric(ext[c], errors="coerce")

complete_mask = ext[required].notna().all(axis=1)

finite_mask = np.isfinite(
    ext[required].to_numpy(dtype=float)
).all(axis=1)

keep = complete_mask & finite_mask

ext = ext.loc[keep].copy().reset_index(drop=True)

print()
print("Complete external cases:", len(ext))
print("Excluded:", len(d) - len(ext))

if len(ext) != 167:
    raise ValueError(
        f"Expected 167 complete I-SPY1 cases, found {len(ext)}"
    )

if ext["pCR"].nunique() != 2:
    raise ValueError("External pCR outcome is not binary.")

print("pCR positive:", int(ext["pCR"].sum()))
print("pCR negative:", int((ext["pCR"] == 0).sum()))


# ============================================================
# FROZEN EXTERNAL INFERENCE
#
# IMPORTANT:
# No fitting, feature selection, scaling estimation,
# tuning, or recalibration is performed on I-SPY1.
# ============================================================

X_ext = ext[input_features].astype(float)

X_ext_scaled = scaler.transform(X_ext)

pred = model.predict_proba(X_ext_scaled)[:, 1]

if not np.isfinite(pred).all():
    raise RuntimeError("Non-finite predictions generated.")

if ((pred < 0) | (pred > 1)).any():
    raise RuntimeError("Invalid probability generated.")


# ============================================================
# PERFORMANCE
# ============================================================

y = ext["pCR"].astype(int).to_numpy()

auc = roc_auc_score(y, pred)
auprc = average_precision_score(y, pred)
brier = brier_score_loss(y, pred)

mean_pred = float(np.mean(pred))
observed = float(np.mean(y))


# ============================================================
# CALIBRATION INTERCEPT AND SLOPE
# y ~ intercept + slope * logit(pred)
# ============================================================

eps = 1e-6
p_clip = np.clip(pred, eps, 1 - eps)

logit_p = np.log(
    p_clip / (1 - p_clip)
).reshape(-1, 1)

cal_model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=5000
)

cal_model.fit(logit_p, y)

cal_intercept = float(cal_model.intercept_[0])
cal_slope = float(cal_model.coef_[0][0])


# ============================================================
# SAVE CORRECTED EXTERNAL PREDICTIONS
# ============================================================

if "pid" in ext.columns:
    id_values = ext["pid"].astype(str)
elif "PatientID_x" in ext.columns:
    id_values = ext["PatientID_x"].astype(str)
else:
    raise ValueError("No usable patient identifier found.")

out = pd.DataFrame({
    "pid": id_values,
    "pCR": y,
    "M4_prob": pred
})

if out["pid"].duplicated().any():
    raise ValueError(
        "Duplicate patient identifiers detected."
    )

out.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 80)
print("CORRECTED M4 EXTERNAL PERFORMANCE")
print("=" * 80)

print(f"N: {len(y)}")
print(f"pCR positive: {int(y.sum())}")
print(f"pCR negative: {int((y == 0).sum())}")

print()
print(f"AUROC: {auc:.10f}")
print(f"AUPRC: {auprc:.10f}")
print(f"Brier: {brier:.10f}")

print()
print(f"Observed pCR rate: {observed:.10f}")
print(f"Mean predicted probability: {mean_pred:.10f}")
print(f"Calibration intercept: {cal_intercept:.10f}")
print(f"Calibration slope: {cal_slope:.10f}")

print()
print("Prediction minimum:", float(pred.min()))
print("Prediction maximum:", float(pred.max()))

print()
print("Saved:", OUTPUT_FILE)
print()
print("NO EXTERNAL MODEL FITTING OR RECALIBRATION PERFORMED")
print("=" * 80)