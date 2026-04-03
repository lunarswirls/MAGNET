#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from magnetpy.paper_figs._paths import example_data_path

input_dir = example_data_path("sine_wave_test")
output_file = input_dir / "sine_residuals.png"
linear = False  # use linear scale instead of log10

ORIENTATIONS = ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]

# Orientation inference from filename/stem.
# Supports stems containing: "_X_only_", "_Y_only_", "_Z_only_", "_XY_", "_XZ_", "_YZ_", "_XYZ_"
def infer_orientation(path: Path) -> str | None:
    s = path.stem.upper()
    if "XYZ" in s:
        return "XYZ"
    if "XY" in s:
        return "XY"
    if "XZ" in s:
        return "XZ"
    if "YZ" in s:
        return "YZ"
    if "X_ONLY" in s or "X-ONLY" in s or "XONLY" in s:
        return "X"
    if "Y_ONLY" in s or "Y-ONLY" in s or "YONLY" in s:
        return "Y"
    if "Z_ONLY" in s or "Z-ONLY" in s or "ZONLY" in s:
        return "Z"
    return None


def safe_log10(a: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """log10 with floor to avoid -inf."""
    a = np.asarray(a, dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    a = np.where(a <= floor, floor, a)
    return np.log10(a)


def load_orientation_tables(input_dir: Path) -> Dict[str, pd.DataFrame]:
    files = sorted(list(input_dir.glob("*.xlsx")) + list(input_dir.glob("*.xls")))
    if not files:
        raise FileNotFoundError(f"No files found in: {input_dir}")

    tables: Dict[str, pd.DataFrame] = {}
    for fp in files:
        orient = infer_orientation(fp)
        if orient is None or orient not in ORIENTATIONS:
            continue

        df = pd.read_excel(fp)
        # Validate required columns
        required = ["Sine amplitude", "Sine frequency", "Sine phase", "Max RMSE"]
        missing_col = [c for c in required if c not in df.columns]
        if missing_col:
            raise ValueError(f"{fp.name} missing required columns: {missing_col}")

        # Coerce numerics
        df["Sine amplitude"] = pd.to_numeric(df["Sine amplitude"], errors="coerce")
        df["Sine frequency"] = pd.to_numeric(df["Sine frequency"], errors="coerce")
        df["Sine phase"] = pd.to_numeric(df["Sine phase"], errors="coerce")
        df["Max RMSE"] = pd.to_numeric(df["Max RMSE"], errors="coerce")

        tables[orient] = df

    return tables


def compute_grids(df: pd.DataFrame, amps: np.ndarray, freqs: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
    """Return (median_over_phase, p95_over_phase) grids on (freqs x amps)."""
    d = df.dropna(subset=["Sine amplitude", "Sine frequency", "Sine phase", "Max RMSE"])
    if d.empty:
        shape = (freqs.size, amps.size)
        return np.full(shape, np.nan), np.full(shape, np.nan)

    grp = d.groupby(["Sine frequency", "Sine amplitude"])["Max RMSE"]
    med = grp.median().unstack("Sine amplitude").reindex(index=freqs, columns=amps).to_numpy()
    p95 = grp.quantile(0.95).unstack("Sine amplitude").reindex(index=freqs, columns=amps).to_numpy()
    return med, p95


def make_publication_7x2(tables: Dict[str, pd.DataFrame], outpath: Path, *, use_log10: bool = True,) -> None:
    # Build union amplitude/frequency axes across all orientations present
    amp_vals = []
    freq_vals = []
    for df in tables.values():
        amp_vals.append(pd.to_numeric(df["Sine amplitude"], errors="coerce").dropna().unique())
        freq_vals.append(pd.to_numeric(df["Sine frequency"], errors="coerce").dropna().unique())

    amps = np.sort(np.unique(np.concatenate(amp_vals))) if amp_vals else np.array([])
    freqs = np.sort(np.unique(np.concatenate(freq_vals))) if freq_vals else np.array([])
    if amps.size == 0 or freqs.size == 0:
        raise ValueError("No valid amplitude/frequency values found in the provided files.")

    # Compute grids per orientation and gather for global color scale
    grids: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    all_vals = []
    for o in ORIENTATIONS:
        df = tables.get(o)
        if df is None:
            med = np.full((freqs.size, amps.size), np.nan)
            p95 = np.full((freqs.size, amps.size), np.nan)
        else:
            med, p95 = compute_grids(df, amps, freqs)

        if use_log10:
            med = safe_log10(med)
            p95 = safe_log10(p95)

        grids[o] = (med, p95)
        all_vals.append(med.ravel())
        all_vals.append(p95.ravel())

    all_vals = np.concatenate(all_vals)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise ValueError("All grid values are NaN; check Max RMSE column and numeric parsing.")

    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    # Figure
    fig, axes = plt.subplots(len(ORIENTATIONS), 2, figsize=(9.0, 2.0 * len(ORIENTATIONS)), constrained_layout=True)
    last_im = None

    for i, o in enumerate(ORIENTATIONS):
        med, p95 = grids[o]

        for j, (arr, title) in enumerate([(med, "Median over phase"), (p95, "95th percentile over phase")]):
            ax = axes[i, j]
            last_im = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                extent=[amps.min(), amps.max(), freqs.min(), freqs.max()],
                vmin=vmin,
                vmax=vmax,
            )

            # Titles only on top row
            if i == 0:
                ax.set_title(title)

            # Y labels: orientation + frequency label on left; blank on right
            if j == 0:
                ax.set_ylabel(f"{o}\nFrequency (Hz)")
            else:
                ax.set_ylabel("")

            # X labels only on bottom row
            if i == len(ORIENTATIONS) - 1:
                ax.set_xlabel("Amplitude (nT)")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

    cbar_label = "log10(Max RMSE)" if use_log10 else "Max RMSE"
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.82, label=cbar_label)

    fig.suptitle(f"RMS error surfaces per exitation axis", y=1.02, fontsize=18)
    # fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


input_dir = Path(input_dir).expanduser().resolve()
outpath = Path(output_file).expanduser().resolve()

tables = load_orientation_tables(input_dir)

# Warn if some orientations are missing, still produces figure with blank rows.
missing = [o for o in ORIENTATIONS if o not in tables]
if missing:
    print(f"Warning: missing orientations (rows will be blank): {missing}")

make_publication_7x2(tables, outpath, use_log10=(not linear))
print(f"Wrote: {outpath}")
