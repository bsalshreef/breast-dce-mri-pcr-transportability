import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Figure 3 — Transportability Forest Plot
# FINAL CLEAN VERSION
#
# Source:
#   TABLE4_TRANSPORTABILITY.csv
#
# IMPORTANT:
#   Visualization only.
#   No model fitting.
#   No bootstrap.
#   No metric recalculation.
# ============================================================

SOURCE = "TABLE4_TRANSPORTABILITY_M4_CORRECTED_v2.csv"

df = pd.read_csv(SOURCE)

required = [
    "Comparison",
    "Metric",
    "Delta (95% CI)"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Parse the LOCKED estimates
# Example:
# 0.075 (-0.015 to 0.157)
# ============================================================

pattern = re.compile(
    r"^\s*"
    r"([-+]?\d*\.?\d+)"
    r"\s*\(\s*"
    r"([-+]?\d*\.?\d+)"
    r"\s+to\s+"
    r"([-+]?\d*\.?\d+)"
    r"\s*\)\s*$"
)

parsed = []

for value in df["Delta (95% CI)"]:

    match = pattern.match(str(value))

    if match is None:
        raise ValueError(
            f"Could not parse locked estimate: {value}"
        )

    parsed.append(
        (
            float(match.group(1)),
            float(match.group(2)),
            float(match.group(3))
        )
    )


df[["Delta", "CI_low", "CI_high"]] = pd.DataFrame(
    parsed,
    index=df.index
)


# ============================================================
# Integrity checks
# ============================================================

if len(df) != 9:
    raise ValueError(
        f"Expected 9 locked rows; found {len(df)}"
    )

expected_metrics = {
    "Delta AUROC",
    "Delta AUPRC",
    "Delta Brier"
}

if set(df["Metric"]) != expected_metrics:
    raise ValueError(
        "Unexpected metric labels."
    )


# ============================================================
# Readable labels
# ============================================================

comparison_map = {
    "M2 | I-SPY1 - I-SPY2":
        "M2: I-SPY1 − I-SPY2",

    "M2 | Duke - I-SPY2":
        "M2: Duke − I-SPY2",

    "M4 | I-SPY1 - I-SPY2":
        "M4: I-SPY1 − I-SPY2"
}

metric_map = {
    "Delta AUROC": "AUROC",
    "Delta AUPRC": "AUPRC",
    "Delta Brier": "Brier"
}

df["Comparison_label"] = (
    df["Comparison"]
    .map(comparison_map)
    .fillna(df["Comparison"])
)

df["Metric_label"] = (
    df["Metric"]
    .map(metric_map)
)


# ============================================================
# Preserve locked table order
# Reverse only for visual top-to-bottom presentation
# ============================================================

df = df.reset_index(drop=True)

plot_df = (
    df.iloc[::-1]
    .reset_index(drop=True)
)

y = np.arange(len(plot_df))


# ============================================================
# Confidence interval lengths
# ============================================================

left_error = (
    plot_df["Delta"] -
    plot_df["CI_low"]
)

right_error = (
    plot_df["CI_high"] -
    plot_df["Delta"]
)

xerr = np.vstack(
    [left_error, right_error]
)


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.5, 7.4)
)


# ------------------------------------------------------------
# Reference line: no change
# No label and NO legend, avoiding overlap.
# ------------------------------------------------------------

ax.axvline(
    x=0,
    linestyle="--",
    linewidth=1.2
)


# ------------------------------------------------------------
# Forest estimates
# ------------------------------------------------------------

ax.errorbar(
    plot_df["Delta"],
    y,
    xerr=xerr,
    fmt="o",
    markersize=6,
    capsize=4,
    linewidth=1.4
)


# ============================================================
# Y labels
# ============================================================

labels = []

for row in plot_df.itertuples():

    labels.append(
        f"{row.Comparison_label} | "
        f"{row.Metric_label}"
    )

ax.set_yticks(y)

ax.set_yticklabels(
    labels,
    fontsize=10
)


# ============================================================
# Numerical estimate annotations
# ============================================================

x_min = min(
    plot_df["CI_low"].min(),
    -0.30
)

x_max_data = max(
    plot_df["CI_high"].max(),
    0.20
)

text_x = 0.235

for i, row in plot_df.iterrows():

    estimate_text = (
        f"{row['Delta']:+.3f} "
        f"({row['CI_low']:+.3f} to "
        f"{row['CI_high']:+.3f})"
    )

    ax.text(
        text_x,
        i,
        estimate_text,
        va="center",
        ha="left",
        fontsize=9.5
    )


# ============================================================
# Axis formatting
# ============================================================

ax.set_xlim(
    -0.31,
    0.46
)

ax.set_xlabel(
    "Change in performance (external − internal)",
    fontsize=11
)

ax.set_title(
    "Transportability of locked models across external cohorts\n"
    "Point estimates with 95% confidence intervals",
    fontsize=13
)

ax.grid(
    axis="x",
    alpha=0.20
)


# ============================================================
# Explicit zero-line label OUTSIDE plotting rows
# ============================================================

ax.text(
    0,
    -0.085,
    "No change (Δ = 0)",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=8.5
)


# ============================================================
# Footnote
# ============================================================

fig.text(
    0.02,
    0.015,
    "Δ = external − I-SPY2 internal performance. "
    "AUPRC is prevalence-sensitive; Brier-score differences "
    "should be interpreted cautiously across cohorts with "
    "differing outcome prevalence.",
    ha="left",
    va="bottom",
    fontsize=8.5
)


# Leave room for footnote and zero-line annotation
fig.tight_layout(
    rect=[0, 0.075, 1, 1]
)


# ============================================================
# Save final figure
# ============================================================

fig.savefig(
    "FIGURE3_TRANSPORTABILITY_FOREST_CORRECTED_v2.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "FIGURE3_TRANSPORTABILITY_FOREST_CORRECTED_v2.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "FIGURE3_TRANSPORTABILITY_FOREST.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Audit data
# ============================================================

audit_cols = [
    "Comparison",
    "Metric",
    "Delta",
    "CI_low",
    "CI_high",
    "Note"
]

df[audit_cols].to_csv(
    "FIGURE3_TRANSPORTABILITY_FOREST_DATA_CORRECTED_v2.csv",
    index=False
)


# ============================================================
# Console verification
# ============================================================

print(
    "Figure 3 FINAL clean version generated successfully."
)

print(
    "Source:",
    SOURCE
)

print(
    "Rows:",
    len(df)
)

print()

print(
    df[
        [
            "Comparison",
            "Metric",
            "Delta",
            "CI_low",
            "CI_high"
        ]
    ].to_string(index=False)
)

print()

print(
    "Visualization only: no model fitting, bootstrap, "
    "or performance recalculation was performed."
)