import os
import pandas as pd
from radiomics import featureextractor

BASE = r"BreastDCEDL_ISPY2_min_crop"
DCE_DIR = os.path.join(BASE, "dce")
MASK_DIR = os.path.join(BASE, "mask")

INPUT = "ISPY2_DCE_phase_map.csv"
OUTPUT = "ISPY2_radiomics_FULL.csv"
FAILURES = "ISPY2_radiomics_FULL_failures.csv"

df = pd.read_csv(INPUT)

extractor = featureextractor.RadiomicsFeatureExtractor()

# Resume support
if os.path.exists(OUTPUT):
    old = pd.read_csv(OUTPUT)
    completed = set(old["PatientID"].astype(str))
    results = old.to_dict("records")
    print(f"Resuming: {len(completed)} already completed")
else:
    completed = set()
    results = []

failures = []

total = len(df)

for idx, row in df.iterrows():

    pid = str(row["PatientID"])

    if pid in completed:
        continue

    print(f"[{idx+1}/{total}] {pid}", flush=True)

    try:
        mask = os.path.join(MASK_DIR, str(row["Mask_File"]))

        if not os.path.exists(mask):
            raise FileNotFoundError(f"MASK missing: {mask}")

        record = {
            "PatientID": pid,
            "pCR": row["pCR"]
        }

        for phase in ["DCE1", "DCE2", "DCE3"]:

            image = os.path.join(
                DCE_DIR,
                str(row[f"{phase}_File"])
            )

            if not os.path.exists(image):
                raise FileNotFoundError(
                    f"{phase} missing: {image}"
                )

            raw = extractor.execute(image, mask)

            features = {
                k: float(v)
                for k, v in raw.items()
                if k.startswith("original_")
            }

            for name, value in features.items():
                record[f"{phase}_{name}"] = value

        results.append(record)
        completed.add(pid)

        # Checkpoint after every successful patient
        pd.DataFrame(results).to_csv(
            OUTPUT,
            index=False
        )

        print("  SUCCESS", flush=True)

    except Exception as e:

        failures.append({
            "PatientID": pid,
            "error": str(e)
        })

        pd.DataFrame(failures).to_csv(
            FAILURES,
            index=False
        )

        print("  FAILED:", e, flush=True)

print("\n================================")
print("FULL EXTRACTION COMPLETE")
print("Successful:", len(results))
print("Failed:", len(failures))
print("Expected:", total)
print("================================")