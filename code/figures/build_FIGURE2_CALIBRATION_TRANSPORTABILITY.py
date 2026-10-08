import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Figure 2 — Calibration Transportability
# Uses LOCKED predictions only. No model refitting.
# ============================================================

def calibration_points(y, p, n_bins=10):
    """
    Quantile-based calibration bins.
    Returns mean predicted probability, observed event rate,
    and number of observations per bin.
    """
    df = pd.DataFrame({
        "y": np.asarray(y, dtype=float),
        "p": np.asarray(p, dtype=float)
    }).dropna()

    # qcut may drop duplicated bin edges if necessary
    df["bin"] = pd.qcut(
        df["p"],
        q=min(n_bins, len(df)),
        duplicates="drop"
    )

    out = (
        df.groupby("bin", observed=True)
        .agg(
            mean_pred=("p", "mean"),
            observed=("y", "mean"),
            n=("y", "size")
        )
        .reset_index(drop=True)
    )

    return out


# ------------------------------------------------------------
# Load locked predictions
# ------------------------------------------------------------

m2_internal = pd.read_csv("M2_locked_OOF_predictions.csv")
m4_internal = pd.read_csv("M4_LOCKED_OOF_predictions.csv")

m2_ispy1 = pd.read_csv("ISPY1_M2_EXTERNAL_predictions.csv")
m4_ispy1 = pd.read_csv("ISPY1_M4_EXTERNAL_predictions_CORRECTED.csv")

m2_duke = pd.read_csv(
    "DUKE_M2_EXTERNAL_predictions_UNIT_ALIGNED.csv"
)


# ------------------------------------------------------------
# Basic integrity checks
# ------------------------------------------------------------

assert len(m2_internal) == 982
assert len(m4_internal) == 982
assert len(m2_ispy1) == 167
assert len(m4_ispy1) == 167
assert len(m2_duke) == 297

assert m2_internal["M2_pred"].notna().all()
assert m4_internal["M4_pred"].notna().all()
assert m2_ispy1["M2_prob"].notna().all()
assert m4_ispy1["M4_prob"].notna().all()
assert m2_duke["M2_prob"].notna().all()


# ------------------------------------------------------------
# Generate calibration points
# ------------------------------------------------------------

m2_int_cal = calibration_points(
    m2_internal["pCR"],
    m2_internal["M2_pred"]
)

m2_i1_cal = calibration_points(
    m2_ispy1["pCR"],
    m2_ispy1["M2_prob"]
)

m2_duke_cal = calibration_points(
    m2_duke["pCR"],
    m2_duke["M2_prob"]
)

m4_int_cal = calibration_points(
    m4_internal["pCR"],
    m4_internal["M4_pred"]
)

m4_i1_cal = calibration_points(
    m4_ispy1["pCR"],
    m4_ispy1["M4_prob"]
)


# ------------------------------------------------------------
# Save calibration-point data for audit/reproducibility
# ------------------------------------------------------------

def save_points(df, model, cohort, filename):
    x = df.copy()
    x.insert(0, "cohort", cohort)
    x.insert(0, "model", model)
    x.to_csv(filename, index=False)


save_points(
    m2_int_cal,
    "M2",
    "I-SPY2 internal OOF",
    "FIGURE2A_M2_ISPY2_calibration_points.csv"
)

save_points(
    m2_i1_cal,
    "M2",
    "I-SPY1 external",
    "FIGURE2A_M2_ISPY1_calibration_points.csv"
)

save_points(
    m2_duke_cal,
    "M2",
    "Duke external",
    "FIGURE2A_M2_DUKE_calibration_points.csv"
)

save_points(
    m4_int_cal,
    "M4",
    "I-SPY2 internal OOF",
    "FIGURE2B_M4_ISPY2_calibration_points.csv"
)

save_points(
    m4_i1_cal,
    "M4",
    "I-SPY1 external",
    "FIGURE2B_M4_ISPY1_calibration_points.csv"
)


# ------------------------------------------------------------
# Figure 2A — M2
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.2, 6.5))

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    linewidth=1.2,
    label="Perfect calibration"
)

ax.plot(
    m2_int_cal["mean_pred"],
    m2_int_cal["observed"],
    marker="o",
    linewidth=1.8,
    label="I-SPY2 internal OOF (N=982)"
)

ax.plot(
    m2_i1_cal["mean_pred"],
    m2_i1_cal["observed"],
    marker="s",
    linewidth=1.8,
    label="I-SPY1 external (N=167)"
)

ax.plot(
    m2_duke_cal["mean_pred"],
    m2_duke_cal["observed"],
    marker="^",
    linewidth=1.8,
    label="Duke external (N=297)"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed pCR proportion")

ax.set_title(
    "M2 calibration transportability\n"
    "Clinical model"
)

ax.legend(frameon=False)
ax.grid(alpha=0.20)

fig.tight_layout()

fig.savefig(
    "FIGURE2A_M2_CALIBRATION_TRANSPORTABILITY.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "FIGURE2A_M2_CALIBRATION_TRANSPORTABILITY.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# Figure 2B — M4
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.2, 6.5))

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    linewidth=1.2,
    label="Perfect calibration"
)

ax.plot(
    m4_int_cal["mean_pred"],
    m4_int_cal["observed"],
    marker="o",
    linewidth=1.8,
    label="I-SPY2 internal OOF (N=982)"
)

ax.plot(
    m4_i1_cal["mean_pred"],
    m4_i1_cal["observed"],
    marker="s",
    linewidth=1.8,
    label="I-SPY1 external (N=167)"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed pCR proportion")

ax.set_title(
    "M4 calibration transportability\n"
    "Clinical + MRI model"
)

ax.legend(frameon=False)
ax.grid(alpha=0.20)

fig.tight_layout()

fig.savefig(
    "FIGURE2B_M4_CALIBRATION_TRANSPORTABILITY.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "FIGURE2B_M4_CALIBRATION_TRANSPORTABILITY.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# Console audit summary
# ------------------------------------------------------------

print("Figure 2 generated successfully.")
print()
print("M2:")
print(
    "I-SPY2 N =", len(m2_internal),
    "| observed =", m2_internal["pCR"].mean(),
    "| mean predicted =", m2_internal["M2_pred"].mean()
)
print(
    "I-SPY1 N =", len(m2_ispy1),
    "| observed =", m2_ispy1["pCR"].mean(),
    "| mean predicted =", m2_ispy1["M2_prob"].mean()
)
print(
    "Duke N =", len(m2_duke),
    "| observed =", m2_duke["pCR"].mean(),
    "| mean predicted =", m2_duke["M2_prob"].mean()
)

print()
print("M4:")
print(
    "I-SPY2 N =", len(m4_internal),
    "| observed =", m4_internal["pCR"].mean(),
    "| mean predicted =", m4_internal["M4_pred"].mean()
)
print(
    "I-SPY1 N =", len(m4_ispy1),
    "| observed =", m4_ispy1["pCR"].mean(),
    "| mean predicted =", m4_ispy1["M4_prob"].mean()
)

print()
print("IMPORTANT:")
print(
    "M4 is intentionally not evaluated in Duke because the Duke archive "
    "does not provide segmentation masks semantically compatible with "
    "the tumor masks used to derive the locked M4 radiomic features."
)
