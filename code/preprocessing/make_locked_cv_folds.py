import pandas as pd
from sklearn.model_selection import StratifiedKFold

SRC = "ISPY2_imaging_manifest_LOCKED.csv"
OUT = "ISPY2_CV5_FOLDS_LOCKED.csv"

SEED = 20261003
N_SPLITS = 5

d = pd.read_csv(SRC).sort_values("PatientID").reset_index(drop=True)

assert len(d) == 982
assert d["PatientID"].nunique() == 982
assert d["pCR"].notna().all()
assert set(d["pCR"].unique()) == {0,1}

d["CV_Fold"] = -1

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

for fold, (_, test_idx) in enumerate(cv.split(d, d["pCR"]), start=1):
    d.loc[test_idx, "CV_Fold"] = fold

assert (d["CV_Fold"] >= 1).all()
assert d["PatientID"].duplicated().sum() == 0

folds = d.groupby("CV_Fold")["pCR"].agg(
    N="size",
    pCR_Positive="sum"
)
folds["pCR_Negative"] = folds["N"] - folds["pCR_Positive"]
folds["pCR_Rate"] = folds["pCR_Positive"] / folds["N"]

print("=== LOCKED 5-FOLD CV ===")
print("Seed:", SEED)
print("Patients:", len(d))
print("Unique IDs:", d["PatientID"].nunique())
print()
print(folds)
print()
print("Total pCR+:", int(d["pCR"].sum()))
print("Total pCR-:", int((d["pCR"] == 0).sum()))

d[["PatientID","pCR","CV_Fold"]].to_csv(OUT, index=False)

print()
print("SAVED:", OUT)
