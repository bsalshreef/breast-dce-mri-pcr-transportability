import pandas as pd
import numpy as np

# ============================================================
# LOAD ANALYTIC COHORTS
# ============================================================

# I-SPY2: locked analytic cohort
spy2 = pd.read_csv("ISPY2_MRI_master_manifest.csv")[
    ["PatientID", "pCR", "age", "HR", "HER2", "tum_vol"]
].copy()

# Source metadata
source = pd.read_csv("BreastDCEDL_metadata_min_crop.csv")

# I-SPY2 master manifest stores tumor volume at exactly
# 1000 x the source-metadata scale, as verified in the audit.
# Restore source-metadata scale for descriptive reporting.
spy2["tum_vol"] = spy2["tum_vol"] / 1000.0
spy2["Cohort"] = "I-SPY2"


# I-SPY1: exactly the 167 externally evaluated patients
spy1_ready = pd.read_csv("ISPY1_EXTERNAL_READY.csv")
spy1_pred = pd.read_csv("ISPY1_M2_EXTERNAL_predictions.csv")

spy1 = spy1_pred[["pid"]].merge(
    spy1_ready[
        ["pid", "pCR", "age", "HR", "HER2", "tum_vol"]
    ],
    on="pid",
    how="left",
    validate="one_to_one"
)

spy1["tum_vol"] = spy1["tum_vol"] / 1000.0
spy1["Cohort"] = "I-SPY1"


# Duke: exactly the 297 externally evaluated patients
duke_pred = pd.read_csv(
    "DUKE_M2_EXTERNAL_predictions_UNIT_ALIGNED.csv"
)

duke_source = source[source["dataset"].eq("duke")].copy()

duke = duke_pred[["PatientID"]].merge(
    duke_source[
        ["pid", "pCR", "age", "HR", "HER2", "tum_vol"]
    ],
    left_on="PatientID",
    right_on="pid",
    how="left",
    validate="one_to_one"
)

duke["Cohort"] = "Duke"


# ============================================================
# SAFETY CHECKS
# ============================================================

assert len(spy2) == 982
assert len(spy1) == 167
assert len(duke) == 297

assert int(spy2["pCR"].sum()) == 316
assert int(spy1["pCR"].sum()) == 47
assert int(duke["pCR"].sum()) == 62

for name, d in [
    ("I-SPY2", spy2),
    ("I-SPY1", spy1),
    ("Duke", duke),
]:
    for col in ["pCR", "age", "HR", "HER2", "tum_vol"]:
        if d[col].isna().any():
            raise ValueError(
                f"{name}: missing values detected in {col}"
            )


# ============================================================
# SUMMARY FUNCTIONS
# ============================================================

def median_iqr(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    q1 = x.quantile(0.25)
    med = x.median()
    q3 = x.quantile(0.75)
    return f"{med:.2f} ({q1:.2f}-{q3:.2f})"


def n_pct(x):
    x = pd.to_numeric(x, errors="coerce")
    n = int((x == 1).sum())
    total = int(x.notna().sum())
    pct = 100 * n / total
    return f"{n} ({pct:.1f}%)"


def outcome_n_pct(d):
    n = int(d["pCR"].sum())
    total = len(d)
    pct = 100 * n / total
    return f"{n} ({pct:.1f}%)"


# ============================================================
# BUILD TABLE
# ============================================================

cohorts = {
    "I-SPY2 (N=982)": spy2,
    "I-SPY1 (N=167)": spy1,
    "Duke (N=297)": duke,
}

rows = []

def add_row(characteristic, function):
    row = {"Characteristic": characteristic}
    for label, d in cohorts.items():
        row[label] = function(d)
    rows.append(row)


add_row(
    "Age, years, median (IQR)",
    lambda d: median_iqr(d["age"])
)

add_row(
    "HR-positive, n (%)",
    lambda d: n_pct(d["HR"])
)

add_row(
    "HER2-positive, n (%)",
    lambda d: n_pct(d["HER2"])
)

add_row(
    "Tumor volume, source units, median (IQR)",
    lambda d: median_iqr(d["tum_vol"])
)

add_row(
    "pCR, n (%)",
    outcome_n_pct
)

table1 = pd.DataFrame(rows)

table1.to_csv(
    "TABLE1_COHORT_CHARACTERISTICS.csv",
    index=False
)

print()
print("TABLE 1 CREATED SUCCESSFULLY")
print("File: TABLE1_COHORT_CHARACTERISTICS.csv")
print()
print(table1.to_string(index=False))

print()
print("Analytic cohort checks:")
print("I-SPY2:", len(spy2), "patients; pCR+ =", int(spy2.pCR.sum()))
print("I-SPY1:", len(spy1), "patients; pCR+ =", int(spy1.pCR.sum()))
print("Duke:", len(duke), "patients; pCR+ =", int(duke.pCR.sum()))
