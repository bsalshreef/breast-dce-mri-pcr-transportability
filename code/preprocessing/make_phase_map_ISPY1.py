import pandas as pd
import re

INPUT = "ISPY1_MRI_integrity_FINAL.csv"
OUTPUT = "ISPY1_DCE_phase_map.csv"

d = pd.read_csv(INPUT)

def parse_dce(value):
    files = [f.strip() for f in str(value).split(";") if f.strip()]
    result = []

    for f in files:
        m = re.search(r"_aqc_([0-9]+)", f)
        if m is None:
            raise ValueError(f"Cannot identify acquisition number: {f}")
        result.append((int(m.group(1)), f))

    return sorted(result, key=lambda z: z[0])

p = d["DCE_Files"].apply(parse_dce)
counts = p.apply(len)

print("=== PHASE MAP CHECK ===")
print("Patients:", len(d))
print("Exactly 3 DCE:", int((counts == 3).sum()))
print("Other:", int((counts != 3).sum()))

if not (counts == 3).all():
    bad = d.loc[counts != 3, ["PatientID", "DCE_Files"]]
    bad.to_csv("ISPY1_DCE_phase_map_ERRORS.csv", index=False)
    raise RuntimeError("STOP: some patients do not have exactly 3 DCE files")

d["DCE1_Acq"] = p.apply(lambda x: x[0][0])
d["DCE2_Acq"] = p.apply(lambda x: x[1][0])
d["DCE3_Acq"] = p.apply(lambda x: x[2][0])

d["DCE1_File"] = p.apply(lambda x: x[0][1])
d["DCE2_File"] = p.apply(lambda x: x[1][1])
d["DCE3_File"] = p.apply(lambda x: x[2][1])

print()
print("=== ACQUISITION PATTERNS ===")
print(
    d.groupby(["DCE1_Acq", "DCE2_Acq", "DCE3_Acq"])
     .size()
     .sort_values(ascending=False)
)

d.to_csv(OUTPUT, index=False)

print()
print("SAVED:", OUTPUT)
print("ROWS:", len(d))

