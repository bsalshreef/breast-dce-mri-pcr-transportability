import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Figure 2B - M4 Calibration Transportability
# CORRECTED v2
#
# Uses:
#   - Locked I-SPY2 M4 OOF predictions
#   - Corrected frozen-model I-SPY1 M4 predictions
#
# No model fitting or recalibration is performed here.
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
# Load predictions
# ------------------------------------------------------------

m4_internal = pd.read_csv(
    "M4_LOCKED_OOF_predictions.csv"
)

m4_ispy1 = pd.read_csv(
    "ISPY1_M4_EXTERNAL_predictions_CORRECTED.csv"
)


# ------------------------------------------------------------
# Integrity checks
# ------------------------------------------------------------

assert len(m4_internal) == 982
assert len(m4_ispy1) == 167

assert m4_internal["pCR"].notna().all()
assert m4_internal["M4_pred"].notna().all()

assert m4_ispy1["pCR"].notna().all()
assert m4_ispy1["M4_prob"].notna().all()

assert set(m4_internal["pCR"].unique()).issubset({0, 1})
assert set(m4_ispy1["pCR"].unique()).issubset({0, 1})


# ------------------------------------------------------------
# Generate calibration points
# ------------------------------------------------------------

m4_int_cal = calibration_points(
    m4_internal["pCR"],
    m4_internal["M4_pred"]
)

m4_i1_cal = calibration_points(
    m4_ispy1["pCR"],
    m4_ispy1["M4_prob"]
)


# ------------------------------------------------------------
# Save corrected calibration-point data
# ------------------------------------------------------------

def save_points(df, model, cohort, filename):
    x = df.copy()
    x.insert(0, "cohort", cohort)
    x.insert(0, "model", model)
    x.to_csv(filename, index=False)


save_points(
    m4_int_cal,
    "M4",
    "I-SPY2 internal OOF",
    "FIGURE2B_M4_ISPY2_calibration_points_CORRECTED_v2.csv"
)

save_points(
    m4_i1_cal,
    "M4",
    "I-SPY1 external",
    "FIGURE2B_M4_ISPY1_calibration_points_CORRECTED_v2.csv"
)


# ------------------------------------------------------------
# Figure 2B - M4
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
    "FIGURE2B_M4_CALIBRATION_TRANSPORTABILITY_CORRECTED_v2.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "FIGURE2B_M4_CALIBRATION_TRANSPORTABILITY_CORRECTED_v2.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# Console audit summary
# ------------------------------------------------------------

print("Figure 2B CORRECTED v2 generated successfully.")
print()

print(
    "I-SPY2 internal:",
    "N =", len(m4_internal),
    "| pCR+ =", int(m4_internal["pCR"].sum()),
    "| observed =", m4_internal["pCR"].mean(),
    "| mean predicted =", m4_internal["M4_pred"].mean()
)

print(
    "I-SPY1 external:",
    "N =", len(m4_ispy1),
    "| pCR+ =", int(m4_ispy1["pCR"].sum()),
    "| observed =", m4_ispy1["pCR"].mean(),
    "| mean predicted =", m4_ispy1["M4_prob"].mean()
)

print()
print("No model fitting or recalibration performed.")
print("Historical Figure 2 files were not overwritten.")