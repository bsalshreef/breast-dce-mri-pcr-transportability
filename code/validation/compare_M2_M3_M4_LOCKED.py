import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# ============================================================
# SETTINGS
# ============================================================

SEED = 20261003
N_BOOT = 5000

M2_FILE = "M2_locked_OOF_predictions.csv"
M3_FILE = "M3_LOCKED_OOF_predictions.csv"
M4_FILE = "M4_LOCKED_OOF_predictions.csv"

rng = np.random.default_rng(SEED)


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 78)
print("LOADING M2 / M3 / M4 OUT-OF-FOLD PREDICTIONS")
print("=" * 78)

m2_raw = pd.read_csv(M2_FILE)
m3_raw = pd.read_csv(M3_FILE)
m4_raw = pd.read_csv(M4_FILE)

print("\nM2 columns:", m2_raw.columns.tolist())
print("M3 columns:", m3_raw.columns.tolist())
print("M4 columns:", m4_raw.columns.tolist())


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

required_m2 = ["PatientID", "pCR", "M2_pred"]
required_m3 = ["PatientID", "pCR", "M3_pred"]
required_m4 = ["PatientID", "pCR", "M4_pred"]

for c in required_m2:
    if c not in m2_raw.columns:
        raise ValueError(f"M2 missing required column: {c}")

for c in required_m3:
    if c not in m3_raw.columns:
        raise ValueError(f"M3 missing required column: {c}")

for c in required_m4:
    if c not in m4_raw.columns:
        raise ValueError(f"M4 missing required column: {c}")


m2 = m2_raw[["PatientID", "pCR", "M2_pred"]].copy()
m3 = m3_raw[["PatientID", "pCR", "M3_pred"]].copy()
m4 = m4_raw[["PatientID", "pCR", "M4_pred"]].copy()

m2.columns = ["PatientID", "pCR_M2", "M2"]
m3.columns = ["PatientID", "pCR_M3", "M3"]
m4.columns = ["PatientID", "pCR_M4", "M4"]


# ============================================================
# BASIC FILE QC
# ============================================================

print("\n" + "=" * 78)
print("INDIVIDUAL FILE QC")
print("=" * 78)

for name, df in [("M2", m2), ("M3", m3), ("M4", m4)]:

    print(f"\n{name}")
    print("-" * 40)
    print("Rows:", len(df))
    print("Unique patients:", df["PatientID"].nunique())
    print("Duplicate PatientID:", df["PatientID"].duplicated().sum())
    print("Missing cells:", int(df.isna().sum().sum()))

    if df["PatientID"].duplicated().any():
        raise ValueError(f"{name}: duplicate PatientID detected")

    pred_col = name

    if not np.isfinite(df[pred_col]).all():
        raise ValueError(f"{name}: non-finite predictions detected")

    if ((df[pred_col] < 0) | (df[pred_col] > 1)).any():
        raise ValueError(f"{name}: predictions outside [0,1]")


# ============================================================
# MERGE SAME PATIENTS
# ============================================================

merged = (
    m2
    .merge(m3, on="PatientID", how="inner", validate="one_to_one")
    .merge(m4, on="PatientID", how="inner", validate="one_to_one")
)

print("\n" + "=" * 78)
print("MATCHING QC")
print("=" * 78)

print("Matched patients:", len(merged))
print("Unique patients:", merged["PatientID"].nunique())

if len(merged) != 982:
    raise ValueError(
        f"Expected 982 matched patients, found {len(merged)}"
    )

# Outcome consistency
d23 = int((merged["pCR_M2"] != merged["pCR_M3"]).sum())
d24 = int((merged["pCR_M2"] != merged["pCR_M4"]).sum())

print("M2 vs M3 pCR disagreements:", d23)
print("M2 vs M4 pCR disagreements:", d24)

if d23 != 0 or d24 != 0:
    raise ValueError("pCR outcome disagreement between model files")

merged["pCR"] = merged["pCR_M2"].astype(int)

print("\npCR counts:")
print(merged["pCR"].value_counts().sort_index())

n_neg = int((merged["pCR"] == 0).sum())
n_pos = int((merged["pCR"] == 1).sum())

print("\npCR negative:", n_neg)
print("pCR positive:", n_pos)
print("pCR prevalence:", merged["pCR"].mean())

if n_neg != 666 or n_pos != 316:
    raise ValueError(
        f"Unexpected pCR counts: negative={n_neg}, positive={n_pos}"
    )


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def metric_values(y, p):
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
    }


def bootstrap_model_ci(y, p, n_boot=N_BOOT):
    """
    Patient-level bootstrap of fixed OOF predictions.

    These CIs quantify sampling uncertainty conditional on
    the already-generated out-of-fold predictions.
    """

    n = len(y)

    aucs = []
    aprs = []
    briers = []

    for _ in range(n_boot):

        idx = rng.integers(0, n, n)

        yb = y[idx]
        pb = p[idx]

        # AUROC undefined if bootstrap sample has one class
        if len(np.unique(yb)) < 2:
            continue

        aucs.append(roc_auc_score(yb, pb))
        aprs.append(average_precision_score(yb, pb))
        briers.append(brier_score_loss(yb, pb))

    return {
        "AUROC_low": np.percentile(aucs, 2.5),
        "AUROC_high": np.percentile(aucs, 97.5),

        "AUPRC_low": np.percentile(aprs, 2.5),
        "AUPRC_high": np.percentile(aprs, 97.5),

        "Brier_low": np.percentile(briers, 2.5),
        "Brier_high": np.percentile(briers, 97.5),
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

y = merged["pCR"].to_numpy()

predictions = {
    "M2": merged["M2"].to_numpy(),
    "M3": merged["M3"].to_numpy(),
    "M4": merged["M4"].to_numpy(),
}

performance_rows = []

print("\n" + "=" * 78)
print("OVERALL OOF PERFORMANCE")
print("=" * 78)

for model_name, pred in predictions.items():

    point = metric_values(y, pred)
    ci = bootstrap_model_ci(y, pred)

    row = {
        "Model": model_name,

        "AUROC": point["AUROC"],
        "AUROC_low": ci["AUROC_low"],
        "AUROC_high": ci["AUROC_high"],

        "AUPRC": point["AUPRC"],
        "AUPRC_low": ci["AUPRC_low"],
        "AUPRC_high": ci["AUPRC_high"],

        "Brier": point["Brier"],
        "Brier_low": ci["Brier_low"],
        "Brier_high": ci["Brier_high"],
    }

    performance_rows.append(row)

    print(f"\n{model_name}")
    print("-" * 40)

    print(
        f"AUROC: {point['AUROC']:.4f} "
        f"[{ci['AUROC_low']:.4f}, {ci['AUROC_high']:.4f}]"
    )

    print(
        f"AUPRC: {point['AUPRC']:.4f} "
        f"[{ci['AUPRC_low']:.4f}, {ci['AUPRC_high']:.4f}]"
    )

    print(
        f"Brier: {point['Brier']:.4f} "
        f"[{ci['Brier_low']:.4f}, {ci['Brier_high']:.4f}]"
    )


performance_df = pd.DataFrame(performance_rows)

performance_df.to_csv(
    "M2_M3_M4_performance_summary.csv",
    index=False
)


# ============================================================
# PAIRED BOOTSTRAP COMPARISON
# ============================================================

def paired_bootstrap(y, pred_a, pred_b, n_boot=N_BOOT):
    """
    Computes B - A using identical bootstrap patients.

    Positive delta AUROC/AUPRC favors B.
    Negative delta Brier favors B.
    """

    n = len(y)

    point_a = metric_values(y, pred_a)
    point_b = metric_values(y, pred_b)

    point_delta = {
        "AUROC": point_b["AUROC"] - point_a["AUROC"],
        "AUPRC": point_b["AUPRC"] - point_a["AUPRC"],
        "Brier": point_b["Brier"] - point_a["Brier"],
    }

    delta_auc = []
    delta_ap = []
    delta_brier = []

    for _ in range(n_boot):

        idx = rng.integers(0, n, n)

        yb = y[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]

        if len(np.unique(yb)) < 2:
            continue

        delta_auc.append(
            roc_auc_score(yb, pb) -
            roc_auc_score(yb, pa)
        )

        delta_ap.append(
            average_precision_score(yb, pb) -
            average_precision_score(yb, pa)
        )

        delta_brier.append(
            brier_score_loss(yb, pb) -
            brier_score_loss(yb, pa)
        )

    result = {}

    for metric, arr in [
        ("AUROC", delta_auc),
        ("AUPRC", delta_ap),
        ("Brier", delta_brier),
    ]:

        arr = np.asarray(arr)

        result[metric] = point_delta[metric]
        result[f"{metric}_low"] = np.percentile(arr, 2.5)
        result[f"{metric}_high"] = np.percentile(arr, 97.5)

        # Two-sided bootstrap sign-based p-value
        p_lower = np.mean(arr <= 0)
        p_upper = np.mean(arr >= 0)

        result[f"{metric}_p"] = min(
            1.0,
            2 * min(p_lower, p_upper)
        )

    return result


comparisons = [
    ("M2", "M3"),
    ("M2", "M4"),
    ("M3", "M4"),
]

comparison_rows = []

print("\n" + "=" * 78)
print("PAIRED MODEL COMPARISONS")
print("Delta = SECOND MODEL minus FIRST MODEL")
print("=" * 78)

for model_a, model_b in comparisons:

    res = paired_bootstrap(
        y,
        predictions[model_a],
        predictions[model_b]
    )

    print(f"\n{model_b} - {model_a}")
    print("-" * 40)

    print(
        f"Delta AUROC: {res['AUROC']:+.4f} "
        f"[{res['AUROC_low']:+.4f}, "
        f"{res['AUROC_high']:+.4f}], "
        f"p={res['AUROC_p']:.4f}"
    )

    print(
        f"Delta AUPRC: {res['AUPRC']:+.4f} "
        f"[{res['AUPRC_low']:+.4f}, "
        f"{res['AUPRC_high']:+.4f}], "
        f"p={res['AUPRC_p']:.4f}"
    )

    print(
        f"Delta Brier: {res['Brier']:+.4f} "
        f"[{res['Brier_low']:+.4f}, "
        f"{res['Brier_high']:+.4f}], "
        f"p={res['Brier_p']:.4f}"
    )

    comparison_rows.append({
        "Model_A": model_a,
        "Model_B": model_b,

        "Delta_AUROC_B_minus_A": res["AUROC"],
        "Delta_AUROC_low": res["AUROC_low"],
        "Delta_AUROC_high": res["AUROC_high"],
        "Delta_AUROC_p": res["AUROC_p"],

        "Delta_AUPRC_B_minus_A": res["AUPRC"],
        "Delta_AUPRC_low": res["AUPRC_low"],
        "Delta_AUPRC_high": res["AUPRC_high"],
        "Delta_AUPRC_p": res["AUPRC_p"],

        "Delta_Brier_B_minus_A": res["Brier"],
        "Delta_Brier_low": res["Brier_low"],
        "Delta_Brier_high": res["Brier_high"],
        "Delta_Brier_p": res["Brier_p"],
    })


comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    "M2_M3_M4_paired_comparisons.csv",
    index=False
)


# ============================================================
# SAVE MASTER PREDICTION TABLE
# ============================================================

master = merged[
    [
        "PatientID",
        "pCR",
        "M2",
        "M3",
        "M4"
    ]
].copy()

master.to_csv(
    "M2_M3_M4_matched_OOF_predictions.csv",
    index=False
)


# ============================================================
# PRIMARY INCREMENTAL-VALUE RESULT
# ============================================================

primary = comparison_df[
    (comparison_df["Model_A"] == "M2") &
    (comparison_df["Model_B"] == "M4")
].iloc[0]

print("\n" + "=" * 78)
print("PRIMARY INCREMENTAL-VALUE ANALYSIS")
print("M4 (clinical + radiomics) versus M2 (clinical)")
print("=" * 78)

print(
    f"Delta AUROC = "
    f"{primary['Delta_AUROC_B_minus_A']:+.4f} "
    f"[{primary['Delta_AUROC_low']:+.4f}, "
    f"{primary['Delta_AUROC_high']:+.4f}] "
    f"p={primary['Delta_AUROC_p']:.4f}"
)

print(
    f"Delta AUPRC = "
    f"{primary['Delta_AUPRC_B_minus_A']:+.4f} "
    f"[{primary['Delta_AUPRC_low']:+.4f}, "
    f"{primary['Delta_AUPRC_high']:+.4f}] "
    f"p={primary['Delta_AUPRC_p']:.4f}"
)

print(
    f"Delta Brier = "
    f"{primary['Delta_Brier_B_minus_A']:+.4f} "
    f"[{primary['Delta_Brier_low']:+.4f}, "
    f"{primary['Delta_Brier_high']:+.4f}] "
    f"p={primary['Delta_Brier_p']:.4f}"
)


# ============================================================
# AUTOMATED INTERPRETATION FLAGS
# ============================================================

print("\n" + "=" * 78)
print("INTERPRETATION FLAGS")
print("=" * 78)

auc_low = primary["Delta_AUROC_low"]
auc_high = primary["Delta_AUROC_high"]

ap_low = primary["Delta_AUPRC_low"]
ap_high = primary["Delta_AUPRC_high"]

brier_low = primary["Delta_Brier_low"]
brier_high = primary["Delta_Brier_high"]

if auc_low > 0:
    print(
        "AUROC: M4 shows a positive paired improvement over M2."
    )
elif auc_high < 0:
    print(
        "AUROC: M4 performs worse than M2."
    )
else:
    print(
        "AUROC: CI includes zero; no clear paired improvement."
    )

if ap_low > 0:
    print(
        "AUPRC: M4 shows a positive paired improvement over M2."
    )
elif ap_high < 0:
    print(
        "AUPRC: M4 performs worse than M2."
    )
else:
    print(
        "AUPRC: CI includes zero; no clear paired improvement."
    )

# For Brier, lower is better.
if brier_high < 0:
    print(
        "Brier: M4 shows lower prediction error than M2."
    )
elif brier_low > 0:
    print(
        "Brier: M4 shows higher prediction error than M2."
    )
else:
    print(
        "Brier: CI includes zero; no clear paired improvement."
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

with open(
    "M2_M3_M4_primary_result.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("I-SPY2 PAIRED OOF MODEL COMPARISON\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"N = {len(merged)} "
        f"(pCR+={n_pos}, pCR-={n_neg})\n\n"
    )

    f.write(
        "Primary comparison: "
        "M4 clinical+radiomics versus M2 clinical\n\n"
    )

    f.write(
        f"Delta AUROC: "
        f"{primary['Delta_AUROC_B_minus_A']:+.4f} "
        f"[{primary['Delta_AUROC_low']:+.4f}, "
        f"{primary['Delta_AUROC_high']:+.4f}], "
        f"p={primary['Delta_AUROC_p']:.4f}\n"
    )

    f.write(
        f"Delta AUPRC: "
        f"{primary['Delta_AUPRC_B_minus_A']:+.4f} "
        f"[{primary['Delta_AUPRC_low']:+.4f}, "
        f"{primary['Delta_AUPRC_high']:+.4f}], "
        f"p={primary['Delta_AUPRC_p']:.4f}\n"
    )

    f.write(
        f"Delta Brier: "
        f"{primary['Delta_Brier_B_minus_A']:+.4f} "
        f"[{primary['Delta_Brier_low']:+.4f}, "
        f"{primary['Delta_Brier_high']:+.4f}], "
        f"p={primary['Delta_Brier_p']:.4f}\n"
    )

    f.write(
        "\nNote: bootstrap intervals are based on paired "
        "patient-level resampling of fixed out-of-fold "
        "predictions and therefore quantify conditional "
        "sampling uncertainty rather than full "
        "model-development uncertainty.\n"
    )


print("\n" + "=" * 78)
print("FILES SAVED")
print("=" * 78)

print("M2_M3_M4_performance_summary.csv")
print("M2_M3_M4_paired_comparisons.csv")
print("M2_M3_M4_matched_OOF_predictions.csv")
print("M2_M3_M4_primary_result.txt")

print("\n" + "=" * 78)
print("ANALYSIS COMPLETE")
print("=" * 78)
