#!/usr/bin/env python3
"""Unified Lichens + Pepsin plots for the poster.

Generates two figure types per dataset, with **identical axes, sizes,
styling, and pure-black text** so the poster reader can compare them
directly:

  - Wavelength selection heatmap (8 excitations × 31 emissions, union grid)
  - Accuracy envelope (kNN, x = K bands, y = accuracy)

Run once per font-size variant. Output:

  Showcase_Poster/dataset_plots/font_small/{lichens|pepsin}_{heatmap|envelope}.{png,pdf}
  Showcase_Poster/dataset_plots/font_medium/...
  Showcase_Poster/dataset_plots/font_large/...
"""
from __future__ import annotations
import json, warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_BASE = ROOT / "Showcase_Poster" / "dataset_plots"

# ----------------------------------------------------------------------
# Unified grids — both heatmaps share these
# ----------------------------------------------------------------------
EX_GRID = [310, 325, 340, 365, 385, 400, 415, 430]      # union, 8 rows
EM_GRID = list(range(420, 730, 10))                     # 420..720, 31 cols
EM_TICK_EVERY = 2                                        # label every other emission

# ----------------------------------------------------------------------
# Envelope axis bounds — both envelopes share these
# ----------------------------------------------------------------------
ENV_XLIM = (0, 200)
ENV_YLIM = (0.30, 1.00)

# ----------------------------------------------------------------------
# Style constants — pure black text, no grey
# ----------------------------------------------------------------------
BLACK = "#000000"
GRID_GREY = "#CFD3D8"
FILL_BLUE = "#7AAEDB"
LINE_BLACK = "#000000"
BEST_GREEN = "#1B7A2B"
BASE_RED = "#C0392B"
PEAK_GOLD = "#FFC72C"

# ----------------------------------------------------------------------
# Font-size profiles
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FontProfile:
    name: str
    axis_label: int
    tick: int
    legend: int
    peak_text: int
    cbar_label: int

PROFILES = [
    FontProfile("font_small",  axis_label=12, tick=10, legend=9,  peak_text=10, cbar_label=10),
    FontProfile("font_medium", axis_label=14, tick=12, legend=11, peak_text=12, cbar_label=12),
    FontProfile("font_large",  axis_label=17, tick=15, legend=13, peak_text=14, cbar_label=14),
]


def force_black_axes(ax):
    """Push every axis-related text/line/spine to pure black."""
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", colors=BLACK, which="both")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(BLACK)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    if ax.get_title():
        ax.title.set_color(BLACK)


# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------
def _load_above_baseline(results_csv: Path):
    df = pd.read_csv(results_csv)
    bl = df[df["config"] == "BASELINE"].iloc[0]
    bl_acc = bl["accuracy"]
    sel = df[df["config"] != "BASELINE"]
    above = sel[sel["accuracy"] > bl_acc]
    if len(above) < 5:
        above = sel.nlargest(20, "accuracy")
    return df, bl_acc, above


def build_importance_matrix(experiments_dir: Path, above_configs):
    """Return shape (len(EX_GRID), len(EM_GRID)) mean importance + valid mask."""
    imp = np.zeros((len(EX_GRID), len(EM_GRID)))
    cnt = 0
    for cfg in above_configs:
        wf = experiments_dir / cfg / "wavelengths.json"
        if not wf.exists():
            continue
        wls = json.loads(wf.read_text())
        n = max(len(wls), 1)
        for w in wls:
            ex = int(w["excitation"])
            em = int(round(w["emission"] / 10) * 10)
            if ex in EX_GRID and em in EM_GRID:
                i = EX_GRID.index(ex)
                j = EM_GRID.index(em)
                imp[i, j] += 1.0 - (w["rank"] - 1) / n
        cnt += 1
    if cnt:
        imp /= cnt

    # Validity: project-canonical Rayleigh mask (matches generate_figures.py).
    #   1st order:  em < ex + 40
    #   2nd order:  |em - 2*ex| < 40
    rayleigh_valid = np.ones_like(imp, dtype=bool)
    for i, ex in enumerate(EX_GRID):
        for j, em in enumerate(EM_GRID):
            if em < ex + 40 or abs(em - 2 * ex) < 40:
                rayleigh_valid[i, j] = False

    # Acquisition validity: rows whose excitation was never sampled
    acquired = np.array([
        any(int(w["excitation"]) == ex
            for cfg in above_configs
            for w in json.loads((experiments_dir / cfg / "wavelengths.json").read_text())
            if (experiments_dir / cfg / "wavelengths.json").exists())
        for ex in EX_GRID
    ])
    valid = rayleigh_valid.copy()
    for i, ok in enumerate(acquired):
        if not ok:
            valid[i, :] = False

    return imp, valid


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_heatmap(imp, valid, out_path: Path, prof: FontProfile,
                 vmax_local: float, ex_labels=None):
    """Per-dataset normalization: vmax = max within this dataset.

    If ``ex_labels`` is given (a subset of EX_GRID), the matrix is sliced
    to those rows so the omitted excitations don't appear on the y-axis.
    """
    if ex_labels is None:
        ex_labels = EX_GRID
    if ex_labels is not EX_GRID:
        keep = [EX_GRID.index(e) for e in ex_labels]
        imp = imp[keep, :]
        valid = valid[keep, :]

    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=300)
    disp = imp.astype(float).copy()
    disp[~valid] = np.nan

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="#D3D3D3")
    masked = np.ma.masked_invalid(disp)
    im = ax.pcolormesh(masked, cmap=cmap, vmin=0, vmax=vmax_local,
                       edgecolors="face", linewidth=0, rasterized=True)

    # X ticks (every other emission)
    xt = [i + 0.5 for i in range(0, len(EM_GRID), EM_TICK_EVERY)]
    xtl = [str(EM_GRID[i]) for i in range(0, len(EM_GRID), EM_TICK_EVERY)]
    ax.set_xticks(xt)
    ax.set_xticklabels(xtl, rotation=90, fontsize=prof.tick)

    # Y ticks (every excitation in the chosen subset)
    yt = [i + 0.5 for i in range(len(ex_labels))]
    ax.set_yticks(yt)
    ax.set_yticklabels([str(e) for e in ex_labels], fontsize=prof.tick)

    ax.set_xlabel("Emission Wavelength (nm)", fontsize=prof.axis_label,
                  color=BLACK)
    ax.set_ylabel("Excitation Wavelength (nm)", fontsize=prof.axis_label,
                  color=BLACK)

    cbar = plt.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Relative Importance",
                   fontsize=prof.cbar_label, color=BLACK)
    cbar.ax.tick_params(labelsize=prof.tick, colors=BLACK)
    cbar.outline.set_edgecolor(BLACK)

    force_black_axes(ax)
    # Lock layout so both heatmaps save at identical pixel dimensions.
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_envelope(results_csv: Path, baseline_acc: float,
                  out_path: Path, prof: FontProfile):
    df = pd.read_csv(results_csv)
    df = df[df["config"] != "BASELINE"]
    stats = (df.groupby("n_features")
               .agg(min_acc=("accuracy", "min"),
                    max_acc=("accuracy", "max"),
                    mean_acc=("accuracy", "mean"))
               .reset_index().sort_values("n_features"))

    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=300)
    ax.fill_between(stats["n_features"], stats["min_acc"], stats["max_acc"],
                    alpha=0.30, color=FILL_BLUE, label="Range (min–max)",
                    edgecolor="none")
    ax.plot(stats["n_features"], stats["mean_acc"], color=LINE_BLACK,
            linewidth=2.0, label="Mean")
    ax.plot(stats["n_features"], stats["max_acc"], color=BEST_GREEN,
            linewidth=2.2, linestyle="--", label="Best")
    ax.axhline(y=baseline_acc, color=BASE_RED, linestyle="--",
               linewidth=1.6, label=f"Baseline ({baseline_acc:.2%})")

    bi = stats["max_acc"].idxmax()
    bn = int(stats.loc[bi, "n_features"])
    ba = float(stats.loc[bi, "max_acc"])
    ax.scatter([bn], [ba], s=260, marker="*", color=PEAK_GOLD,
               edgecolor=BEST_GREEN, linewidth=1.6, zorder=6)
    ax.text(bn + 5, ba - 0.04,
            f"Peak: {ba:.2%}\nn = {bn}",
            fontsize=prof.peak_text, fontweight="bold", color=BEST_GREEN,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=BEST_GREEN, alpha=0.92))

    ax.set_xlim(*ENV_XLIM)
    ax.set_ylim(*ENV_YLIM)
    ax.set_xlabel("Number of Selected Bands",
                  fontsize=prof.axis_label, color=BLACK)
    ax.set_ylabel("Classification Accuracy",
                  fontsize=prof.axis_label, color=BLACK)
    ax.tick_params(axis="both", which="major", labelsize=prof.tick,
                   colors=BLACK)
    ax.grid(True, color=GRID_GREY, alpha=0.7, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=prof.legend, framealpha=0.95,
              edgecolor=BLACK, labelcolor=BLACK)

    force_black_axes(ax)
    # Lock layout so both envelopes save at identical pixel dimensions.
    # right=0.97 leaves room for the "200" x-tick at large fonts.
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
DATASETS = {
    "lichens": dict(
        results_csv = RESULTS / "Lichens_Dataset_1_MasterRun" / "results.csv",
        experiments = RESULTS / "Lichens_Dataset_1_MasterRun" / "experiments",
    ),
    "pepsin": dict(
        results_csv = RESULTS / "Collagen_Pepsin_Normalized" / "results.csv",
        experiments = RESULTS / "Collagen_Pepsin_Normalized" / "experiments",
    ),
}


def main():
    # --- compute heatmap matrices once (data is font-independent) ---
    matrices = {}
    bl_accs = {}
    for name, paths in DATASETS.items():
        df, bl_acc, above = _load_above_baseline(paths["results_csv"])
        bl_accs[name] = bl_acc
        imp, valid = build_importance_matrix(paths["experiments"],
                                             above["config"].tolist())
        matrices[name] = (imp, valid)
        print(f"{name}: baseline={bl_acc:.4f}, "
              f"above-baseline configs={len(above)}, "
              f"matrix max={np.nanmax(np.where(valid, imp, np.nan)):.3f}")

    # per-dataset normalization: each heatmap reaches its own brightest
    vmax_per_dataset = {
        name: float(np.nanmax(np.where(v, imp, np.nan)))
        for name, (imp, v) in matrices.items()
    }
    for k, v in vmax_per_dataset.items():
        print(f"  {k} vmax = {v:.3f}")

    # Pepsin-only excitations (drops 415, 430 — never sampled)
    PEPSIN_EX = [310, 325, 340, 365, 385, 400]

    # --- render each font profile ---
    for prof in PROFILES:
        out_dir = OUT_BASE / prof.name
        for name, paths in DATASETS.items():
            imp, valid = matrices[name]
            plot_heatmap(imp, valid,
                         out_dir / f"{name}_heatmap",
                         prof, vmax_per_dataset[name])
            plot_envelope(paths["results_csv"], bl_accs[name],
                          out_dir / f"{name}_envelope",
                          prof)
        # Extra Pepsin variant without the 415/430 rows on the axis
        imp_p, valid_p = matrices["pepsin"]
        plot_heatmap(imp_p, valid_p,
                     out_dir / "pepsin_heatmap_native",
                     prof, vmax_per_dataset["pepsin"],
                     ex_labels=PEPSIN_EX)
        print(f"[{prof.name}] wrote 5 plots → {out_dir}")


if __name__ == "__main__":
    main()
