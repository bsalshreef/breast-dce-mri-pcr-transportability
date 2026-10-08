import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# ============================================================
# FIGURE 1 — INTERNAL DISCRIMINATION
# Locked I-SPY2 OOF predictions only
# ============================================================

m2 = pd.read_csv("M2_locked_OOF_predictions.csv")
m3 = pd.read_csv("M3_LOCKED_OOF_predictions.csv")
m4 = pd.read_csv("M4_LOCKED_OOF_predictions.csv")

# Merge exact paired predictions
d = (
    m2[["PatientID", "pCR", "M2_pred"]]
    .merge(
        m3[["PatientID", "pCR", "M3_pred"]],
        on=["PatientID", "pCR"],
        validate="one_to_one"
    )
    .merge(
        m4[["PatientID", "pCR", "M4_pred"]],
        on=["PatientID", "pCR"],
        validate="one_to_one"
    )
)

assert len(d) == 982
assert int(d["pCR"].sum()) == 316

y = d["pCR"].astype(int)

models = [
    ("M2: Clinical", "M2_pred"),
    ("M3: MRI/radiomics", "M3_pred"),
    ("M4: Clinical + MRI", "M4_pred"),
]

# ============================================================
# PANEL A — ROC CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 6.2))

for label, col in models:
    fpr, tpr, _ = roc_curve(y, d[col])
    auc = roc_auc_score(y, d[col])

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{label} (AUROC = {auc:.3f})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Chance"
)

ax.set_xlabel("False-positive rate", fontsize=11)
ax.set_ylabel("True-positive rate", fontsize=11)
ax.set_title(
    "Internal discrimination in I-SPY2\nLocked out-of-fold predictions",
    fontsize=12
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    "FIGURE1A_INTERNAL_ROC.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "FIGURE1A_INTERNAL_ROC.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# PANEL B — PRECISION–RECALL CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 6.2))

for label, col in models:
    precision, recall, _ = precision_recall_curve(y, d[col])
    ap = average_precision_score(y, d[col])

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{label} (AUPRC = {ap:.3f})"
    )

prevalence = y.mean()

ax.axhline(
    prevalence,
    linestyle="--",
    linewidth=1,
    label=f"pCR prevalence = {prevalence:.3f}"
)

ax.set_xlabel("Recall (sensitivity)", fontsize=11)
ax.set_ylabel("Precision (positive predictive value)", fontsize=11)
ax.set_title(
    "Internal precision–recall performance in I-SPY2\n"
    "Locked out-of-fold predictions",
    fontsize=12
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    "FIGURE1B_INTERNAL_PR.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "FIGURE1B_INTERNAL_PR.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# VERIFY METRICS AGAINST LOCKED RESULTS
# ============================================================

print()
print("FIGURE 1 CREATED SUCCESSFULLY")
print("Analytic N:", len(d))
print("pCR+:", int(y.sum()))
print()

for label, col in models:
    print(
        label,
        "| AUROC:",
        f"{roc_auc_score(y, d[col]):.6f}",
        "| AUPRC:",
        f"{average_precision_score(y, d[col]):.6f}"
    )

print()
print("Created:")
print("FIGURE1A_INTERNAL_ROC.png")
print("FIGURE1A_INTERNAL_ROC.pdf")
print("FIGURE1B_INTERNAL_PR.png")
print("FIGURE1B_INTERNAL_PR.pdf")