#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from magnetpy.paper_figs._paths import example_data_path

input_dir = example_data_path("noise_test")
output_file = input_dir / "noise_residuals.png"
input_file = input_dir / "parameters" / "noisy_test_parameters.xlsx"

# ----------------------------------------------------------
# Load + preprocess
# ----------------------------------------------------------
if not input_file.exists():
    raise FileNotFoundError(
        f"Expected noise parameter table at {input_file}. "
        "Add the file under example_data/noise_test/parameters to run this figure script."
    )

df = pd.read_excel(input_file)

for c in ["X-axis", "Y-axis", "Z-axis", "Max RMSE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["X-axis", "Y-axis", "Z-axis", "Max RMSE"]).copy()

# log10 transform (safe)
eps = max(df["Max RMSE"].min(), 1e-20)
df["log10_RMSE"] = np.log10(df["Max RMSE"] + eps)

# total noise magnitude
df["noise_mag"] = np.sqrt(
    df["X-axis"]**2 + df["Y-axis"]**2 + df["Z-axis"]**2
)

def infer_orientation(row, tol=1e-9):
    parts = []
    if abs(row["X-axis"]) > tol:
        parts.append("X")
    if abs(row["Y-axis"]) > tol:
        parts.append("Y")
    if abs(row["Z-axis"]) > tol:
        parts.append("Z")
    return "".join(parts) if parts else "none"

df["orientation"] = df.apply(infer_orientation, axis=1)

wanted = ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]
df = df[df["orientation"].isin(wanted)].copy()
df["orientation"] = pd.Categorical(df["orientation"], categories=wanted, ordered=True)

# ----------------------------------------------------------
# Figure: 3 rows, 1 column
# ----------------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 9), sharex=True)

group1 = ["X", "Y", "Z"]
group2 = ["XY", "XZ", "YZ"]
group3 = ["XYZ"]

groups = [group1, group2, group3]
titles = ["(a.) Single-axis noise", "(b.) Dual-axis noise", "(c.) Triple-axis noise"]

# Sequential colormap sampling
# (pick mid to high values for usable contrast)
cmap1 = plt.cm.plasma
cmap2 = plt.cm.viridis
cmap3 = plt.cm.seismic

# sample N colors per group
palette1 = [cmap1(i) for i in np.linspace(0.4, 0.85, len(group1))]
palette2 = [cmap2(i) for i in np.linspace(0.25, 0.95, len(group2))]
palette3 = [cmap3(i) for i in np.linspace(0.9, 0.95, len(group3))]

palettes = [palette1, palette2, palette3]

for ax, grp, title, pal in zip(axes, groups, titles, palettes):
    for ori, color in zip(grp, pal):
        sub = df[df["orientation"] == ori].copy()
        if sub.empty:
            continue

        # sort for nicer plotting
        sub = sub.sort_values("noise_mag")
        x = sub["noise_mag"].to_numpy()
        y = sub["log10_RMSE"].to_numpy()

        # scatter points
        ax.scatter(x, y, color=color, alpha=0.5, s=30)

        # fit line: log10_RMSE = a * noise_mag + b
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, deg=1)
            a, b = coeffs
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = a * x_fit + b

            ax.plot(
                x_fit,
                y_fit,
                color=color,
                linewidth=2,
                label=f"{ori}"
            )
        else:
            # if only one point, still show a label
            ax.plot([], [], color=color, label=f"{ori} (single point)")

    ax.set_ylim([-0.5, 4.5])
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Noise axis", loc="upper right", fontsize=12, title_fontsize=12)

axes[-1].set_xlabel("Total noise magnitude [%]", fontsize=14)
for ax in axes:
    ax.set_ylabel("log10(Max RMSE)", fontsize=14)

plt.tight_layout()
plt.show()

fig.savefig(output_file, dpi=300, bbox_inches="tight")
