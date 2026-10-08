import pandas as pd
from pathlib import Path

SRC = Path("ISPY2_DCE_phase_map.csv")
OUT = Path("ISPY2_imaging_manifest_LOCKED.csv")

d = pd.read_csv(SRC)

required = [
    "PatientID", "pCR",
    "DCE1_Acq", "DCE1_File",
    "DCE2_Acq", "DCE2_File",
    "DCE3_Acq", "DCE3_File",
    "Mask_File"
]

missing_cols = [c for c in required if c not in d.columns]
if missing_cols:
    raise RuntimeError(f"Missing columns: {missing_cols}")

m = d[required].copy()

# Core integrity checks
assert len(m) == 982, f"Expected 982 patients, found {len(m)}"
assert m["PatientID"].nunique() == 982, "Duplicate PatientID detected"
assert m["PatientID"].notna().all(), "Missing PatientID"
assert m["pCR"].notna().all(), "Missing pCR"
assert set(m["pCR"].unique()).issubset({0,1}), "Invalid pCR values"

file_cols = ["DCE1_File","DCE2_File","DCE3_File","Mask_File"]
assert m[file_cols].notna().all().all(), "Missing imaging filenames"

# Ensure three distinct DCE acquisitions per patient
assert (
    m[["DCE1_Acq","DCE2_Acq","DCE3_Acq"]]
    .nunique(axis=1)
    .eq(3)
    .all()
), "Non-distinct DCE acquisitions detected"

# Deterministic order
m = m.sort_values("PatientID").reset_index(drop=True)

m.to_csv(OUT, index=False)

print("=== LOCKED IMAGING MANIFEST ===")
print("Patients:", len(m))
print("Unique IDs:", m["PatientID"].nunique())
print("pCR positive:", int((m["pCR"] == 1).sum()))
print("pCR negative:", int((m["pCR"] == 0).sum()))
print("Missing values:", int(m.isna().sum().sum()))
print("Duplicate IDs:", int(m["PatientID"].duplicated().sum()))
print()
print("Acquisition patterns:")
print(
    m.groupby(["DCE1_Acq","DCE2_Acq","DCE3_Acq"])
     .size()
     .sort_values(ascending=False)
)
print()
print("SAVED:", OUT)
