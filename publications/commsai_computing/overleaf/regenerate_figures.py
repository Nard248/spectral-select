#!/usr/bin/env python3
"""Regenerate ALL data plots for the manuscript with one unified style.

Every plot uses the SAME font family (sans-serif / DejaVu Sans) and the SAME
absolute font sizes. Figures are generated at their on-page physical width
(single-column ~3.4 in, full-width ~7.0 in) so that, after LaTeX includes them
at scale ~= 1, every label renders at the same physical point size on the page.

Outputs (into ./figures/, overwriting the paper figures):
  accuracy_envelope.png      lichen accuracy envelope        (single column)
  collagen_envelope.png      collagen accuracy envelope      (single column)
  robustness_histogram.png   lichen 13-band robustness       (single column)
  wavelength_heatmap.png     lichen importance heatmap       (full width)
  collagen_heatmap.png       collagen importance heatmap     (full width)

Run:
  .venv/bin/python CommsAIComputing_Overleaf/regenerate_figures.py
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

# ---------------------------------------------------------------------------
# UNIFIED STYLE  — one font family, one set of sizes, for every figure.
# ---------------------------------------------------------------------------
AXIS_LABEL = 10      # axis titles
TICK       = 9       # tick labels (and colorbar ticks)
LEGEND     = 9       # legend text
ANNOT      = 9       # in-plot annotations

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],   # universal, identical on every machine
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": AXIS_LABEL,
    "axes.labelsize": AXIS_LABEL,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "legend.fontsize": LEGEND,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.9,
    "savefig.facecolor": "white",
})

# Physical widths matching the on-page display size (IEEEtran two-column)
W_COL  = 3.40        # single-column figures (\includegraphics width=0.48\textwidth)
W_FULL = 7.02        # full-width figures      (\includegraphics width=\textwidth)

# Colors (kept from the original figures)
BLACK     = "#000000"
GRID_GREY = "#CFD3D8"
FILL_BLUE = "#7AAEDB"
BEST_GREEN= "#1B7A2B"
BASE_RED  = "#C0392B"
PEAK_GOLD = "#FFC72C"
HIST_BLUE = "#9DC3E6"
LEARN_PURPLE = "#6A1B9A"

# Shared spectral grids for the heatmaps
EX_GRID = [310, 325, 340, 365, 385, 400, 415, 430]   # lichen: 8 excitations
EM_GRID = list(range(420, 730, 10))                  # 420..720
EM_TICK_EVERY = 2
PEPSIN_EX = [310, 325, 340, 365, 385, 400]           # collagen: 6 excitations

ENV_XLIM = (0, 200)
ENV_YLIM = (0.30, 1.00)


def _force_black(ax):
    for s in ax.spines.values():
        s.set_color(BLACK)
    ax.tick_params(colors=BLACK, which="both")
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)


# ---------------------------------------------------------------------------
# Data loaders (ported from poster_dataset_plots_unified.py)
# ---------------------------------------------------------------------------
def _load_above_baseline(results_csv: Path):
    df = pd.read_csv(results_csv)
    bl = df[df["config"] == "BASELINE"].iloc[0]
    sel = df[df["config"] != "BASELINE"]
    above = sel[sel["accuracy"] > bl["accuracy"]]
    if len(above) < 5:
        above = sel.nlargest(20, "accuracy")
    return df, float(bl["accuracy"]), above


def build_importance_matrix(experiments_dir: Path, above_configs, ex_grid):
    imp = np.zeros((len(ex_grid), len(EM_GRID)))
    cnt = 0
    for cfg in above_configs:
        wf = experiments_dir / cfg / "wavelengths.json"
        if not wf.exists():
            continue
        wls = json.loads(wf.read_text())
        n = max(len(wls), 1)
        for w in wls:
            ex = int(w["excitation"])
            # Snap emission with FLOOR (not round): Ex=415 nm has a 5 nm-offset
            # grid (455, 465, ...); rounding aliases two emissions into one cell
            # and skips others, producing a black checkerboard. Floor maps them
            # to consecutive grid cells (455->450, 465->460, ...).
            em = int(w["emission"] // 10) * 10
            if ex in ex_grid and em in EM_GRID:
                imp[ex_grid.index(ex), EM_GRID.index(em)] += 1.0 - (w["rank"] - 1) / n
        cnt += 1
    if cnt:
        imp /= cnt
    valid = np.ones_like(imp, dtype=bool)
    for i, ex in enumerate(ex_grid):
        for j, em in enumerate(EM_GRID):
            if em < ex + 40 or abs(em - 2 * ex) < 40:
                valid[i, j] = False
    return imp, valid


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------
def plot_envelope(results_csv, baseline_acc, out_png):
    df = pd.read_csv(results_csv)
    df = df[df["config"] != "BASELINE"]
    s = (df.groupby("n_features")
           .agg(lo=("accuracy", "min"), hi=("accuracy", "max"), mu=("accuracy", "mean"))
           .reset_index().sort_values("n_features"))

    fig, ax = plt.subplots(figsize=(W_COL, 2.35))
    ax.fill_between(s["n_features"], s["lo"], s["hi"], alpha=0.30,
                    color=FILL_BLUE, label="Range (min–max)", edgecolor="none")
    ax.plot(s["n_features"], s["mu"], color=BLACK, lw=1.4, label="Mean")
    ax.plot(s["n_features"], s["hi"], color=BEST_GREEN, lw=1.6, ls="--", label="Best")
    ax.axhline(baseline_acc, color=BASE_RED, ls="--", lw=1.2,
               label=f"Baseline ({baseline_acc:.1%})")

    bi = s["hi"].idxmax()
    bn, ba = int(s.loc[bi, "n_features"]), float(s.loc[bi, "hi"])
    ax.scatter([bn], [ba], s=120, marker="*", color=PEAK_GOLD,
               edgecolor=BEST_GREEN, lw=1.0, zorder=6)
    ax.text(bn + 6, ba - 0.05, f"Peak: {ba:.1%}\nn = {bn}", fontsize=ANNOT,
            fontweight="bold", color=BEST_GREEN, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=BEST_GREEN, alpha=0.92))

    ax.set_xlim(*ENV_XLIM); ax.set_ylim(*ENV_YLIM)
    ax.set_xlabel("Number of Selected Bands")
    ax.set_ylabel("Classification Accuracy")
    ax.grid(True, color=GRID_GREY, alpha=0.7, lw=0.5)
    ax.legend(loc="lower right", framealpha=0.95, edgecolor=BLACK)
    _force_black(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  {out_png.name}: baseline={baseline_acc:.3f} peak={ba:.3f}@{bn}")


def plot_heatmap(imp, valid, out_png, ex_labels):
    # Raw mean importance on a fixed 0..1 colour scale (matches the original
    # generate_figures.py and the caption: 1.0 = consistently top-ranked).
    disp = imp.astype(float).copy()
    m = np.nanmax(np.where(valid, disp, np.nan))
    disp[~valid] = np.nan

    n_rows = len(ex_labels)
    h = 1.05 + 0.21 * n_rows           # height scales with #excitations
    fig, ax = plt.subplots(figsize=(W_FULL, h))
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="#D3D3D3")
    im = ax.pcolormesh(np.ma.masked_invalid(disp), cmap=cmap, vmin=0, vmax=1.0,
                       edgecolors="face", linewidth=0, rasterized=True)

    xt = list(range(0, len(EM_GRID), EM_TICK_EVERY))
    ax.set_xticks([i + 0.5 for i in xt])
    ax.set_xticklabels([str(EM_GRID[i]) for i in xt], rotation=90)
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels([str(e) for e in ex_labels])
    ax.set_xlabel("Emission Wavelength (nm)")
    ax.set_ylabel("Excitation Wavelength (nm)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.92, pad=0.015)
    cbar.ax.tick_params(labelsize=TICK, colors=BLACK)
    cbar.outline.set_edgecolor(BLACK)
    _force_black(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  {out_png.name}: rows={n_rows} max_imp={m:.3f}")


def plot_robustness(rand_acc, learned_acc, out_png):
    mean, std = float(np.mean(rand_acc)), float(np.std(rand_acc))
    median, mx = float(np.median(rand_acc)), float(np.max(rand_acc))
    z = (learned_acc - mean) / std

    fig, ax = plt.subplots(figsize=(W_COL, 2.45))
    ax.hist(rand_acc, bins=40, color=HIST_BLUE, edgecolor="white", linewidth=0.3,
            label="Random (10,000)")
    ax.axvline(mean, color=BASE_RED, ls="--", lw=1.2, label=f"Random mean ({mean:.1%})")
    ax.axvline(learned_acc, color=LEARN_PURPLE, ls="-", lw=1.8,
               label=f"Learned ({learned_acc:.1%})")
    ax.set_xlabel("Classification Accuracy")
    ax.set_ylabel("Count")
    ax.set_xlim(0.30, 1.0)
    ax.grid(True, axis="y", color=GRID_GREY, alpha=0.6, lw=0.5)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor=BLACK)
    # sigma annotation, placed just left of the learned line, mid-height, clear of legend
    ax.text(learned_acc - 0.02, ax.get_ylim()[1] * 0.55,
            f"{z:.1f}σ", fontsize=ANNOT, fontweight="bold",
            color=LEARN_PURPLE, ha="right", va="center")
    _force_black(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  {out_png.name}: mean={mean:.3f} max={mx:.3f} learned={learned_acc:.3f} ({z:.1f}sigma)")


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    lich = RESULTS / "Lichens_Dataset_1_MasterRun"
    coll = RESULTS / "Collagen_Pepsin_Normalized"

    print("Envelopes:")
    _, bl_l, _ = _load_above_baseline(lich / "results.csv")
    plot_envelope(lich / "results.csv", bl_l, OUT / "accuracy_envelope.png")
    _, bl_c, _ = _load_above_baseline(coll / "results.csv")
    plot_envelope(coll / "results.csv", bl_c, OUT / "collagen_envelope.png")

    print("Heatmaps:")
    _, _, above_l = _load_above_baseline(lich / "results.csv")
    imp_l, val_l = build_importance_matrix(lich / "experiments",
                                           above_l["config"].tolist(), EX_GRID)
    plot_heatmap(imp_l, val_l, OUT / "wavelength_heatmap.png", EX_GRID)

    _, _, above_c = _load_above_baseline(coll / "results.csv")
    imp_c, val_c = build_importance_matrix(coll / "experiments",
                                           above_c["config"].tolist(), PEPSIN_EX)
    plot_heatmap(imp_c, val_c, OUT / "collagen_heatmap.png", PEPSIN_EX)

    print("Robustness:")
    rb = pd.read_csv(ROOT / "archive" / "wavelength_analysis" / "Results" /
                     "robustness" / "robustness_13bands_results.csv")
    # learned 13-band selection = 0.9016 (PCA selection, main results table & paper text)
    plot_robustness(rb["accuracy"].values, 0.9016, OUT / "robustness_histogram.png")

    print("\nAll five data plots regenerated with the unified style.")


if __name__ == "__main__":
    main()
