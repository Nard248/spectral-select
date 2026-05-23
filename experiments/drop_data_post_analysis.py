#!/usr/bin/env python3
"""
Drop Data — Post-Sweep Analysis & Visualization
================================================
Reads the sweep results CSV produced by drop_data_selection_sweep.py and
generates the figures the user will visually verify.

Outputs (all under results/Drop_Data_Selection_Sweep/figures/):
  best_per_variant.csv             - winning method per variant by silhouette
  best_method_grid.png             - heatmap of silhouette across variant x method x n
  selected_bands_<variant>_<method>_n<N>.png
                                   - selected (ex, em) overlaid on the
                                     mean spectrum for each excitation
  drop_clusters_<variant>_<method>_n<N>.png
                                   - per-drop spectra coloured by k-means in
                                     the selected-band subspace + scatter of
                                     drops in the first 2 PCs of that subspace

Usage: .venv/bin/python experiments/drop_data_post_analysis.py
       (run after drop_data_selection_sweep.py)
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
SWEEP_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Selection_Sweep"
FIG_ROOT = SWEEP_ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

EXCITATIONS = [310.0, 325.0, 340.0, 365.0, 385.0, 400.0, 415.0]
EXCITATION_COLORS = plt.cm.viridis(np.linspace(0, 0.95, len(EXCITATIONS)))

LABEL_RE = re.compile(r"(\d+)/(\d+)")


def parse_band_list(s):
    """Robustly parse the 'selected_bands' CSV column.

    The column is stored as a Python list-of-string repr like
    "['310/470', '325/420']". We extract the (ex, em) pairs by regex —
    no eval needed.
    """
    if isinstance(s, list):
        items = s
    else:
        items = [m.group(0) for m in LABEL_RE.finditer(str(s))]
    out = []
    for item in items:
        m = LABEL_RE.search(item) if isinstance(item, str) else None
        if m:
            out.append((float(m.group(1)), float(m.group(2))))
    return out


def load_drop_spectra(variant: str) -> tuple[np.ndarray, list[dict]]:
    spec = np.load(SWEEP_ROOT / variant / "drop_mean_spectra.npy")
    bands_meta = json.loads((SWEEP_ROOT / variant / "bands.json").read_text())
    return spec, bands_meta


def best_method_per_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Pick winning (method, n) per variant by silhouette_max, ignoring random_*."""
    df_real = df[~df["method"].str.startswith("random_")].copy()
    idx = df_real.groupby("variant")["silhouette_max"].idxmax()
    out = df_real.loc[idx].sort_values("silhouette_max", ascending=False)
    out.to_csv(FIG_ROOT / "best_per_variant.csv", index=False)
    return out


def plot_method_heatmap(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["method_group"] = df_plot["method"].apply(
        lambda m: "random" if m.startswith("random_") else m
    )
    if "random" in df_plot["method_group"].unique():
        rng = (
            df_plot[df_plot["method_group"] == "random"]
            .groupby(["variant", "n_bands"])["silhouette_max"]
            .mean()
            .reset_index()
        )
        rng["method_group"] = "random_mean"
        df_plot = pd.concat([
            df_plot[df_plot["method_group"] != "random"],
            rng.rename(columns={"silhouette_max": "silhouette_max"}),
        ], ignore_index=True)
    pivot = df_plot.pivot_table(
        index=["variant", "method_group"],
        columns="n_bands",
        values="silhouette_max",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"{v}/{m}" for v, m in pivot.index], fontsize=8)
    ax.set_xlabel("n bands")
    ax.set_title("Silhouette across variant x method x n bands")
    fig.colorbar(im, ax=ax, label="silhouette_max")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "best_method_grid.png", dpi=140)
    plt.close(fig)


def plot_selected_bands_overlay(
    variant: str, method: str, n: int, bands_selected: list[tuple[float, float]]
) -> None:
    spec, bands_meta = load_drop_spectra(variant)
    by_ex_idx: dict[float, list[int]] = {ex: [] for ex in EXCITATIONS}
    by_ex_em: dict[float, list[float]] = {ex: [] for ex in EXCITATIONS}
    for i, b in enumerate(bands_meta):
        by_ex_idx[b["excitation_nm"]].append(i)
        by_ex_em[b["excitation_nm"]].append(b["emission_nm"])

    fig, axes = plt.subplots(
        len(EXCITATIONS), 1, figsize=(11, 1.5 * len(EXCITATIONS)), sharex=True,
    )
    for ax, ex, color in zip(axes, EXCITATIONS, EXCITATION_COLORS):
        idxs = by_ex_idx[ex]
        ems = by_ex_em[ex]
        if not idxs:
            ax.axis("off")
            continue
        mean_spec = spec[:, idxs].mean(axis=0)
        ax.plot(ems, mean_spec, color=color, linewidth=1.6,
                label=f"Ex={ex:.0f} mean across drops")
        chosen_em = [em for ex_b, em in bands_selected if ex_b == ex]
        if chosen_em:
            ax.scatter(chosen_em,
                       [np.interp(em, ems, mean_spec) for em in chosen_em],
                       s=80, edgecolor="red", facecolor="none", linewidth=2,
                       zorder=5, label=f"selected ({len(chosen_em)})")
        ax.set_ylabel("Intensity", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Emission lambda (nm)")
    fig.suptitle(f"Selected bands  -  variant={variant}  method={method}  n={n}")
    fig.tight_layout()
    out_path = FIG_ROOT / f"selected_bands_{variant}_{method}_n{n}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_drop_clusters(
    variant: str, method: str, n: int,
    bands_selected: list[tuple[float, float]], silhouette_k: int,
) -> None:
    spec, bands_meta = load_drop_spectra(variant)
    sel_idx = []
    for ex, em in bands_selected:
        for i, b in enumerate(bands_meta):
            if abs(b["excitation_nm"] - ex) < 1e-6 and abs(b["emission_nm"] - em) < 1e-6:
                sel_idx.append(i)
                break
    sel_idx = np.array(sel_idx)
    sub = spec[:, sel_idx]
    n_drops = sub.shape[0]

    k = max(2, min(silhouette_k or 3, n_drops - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sub)
    labels = km.labels_

    if sub.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=0).fit(sub)
        proj = pca.transform(sub)
    else:
        proj = np.column_stack([sub[:, 0], np.zeros(n_drops)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(k, 3)))
    for d in range(n_drops):
        axes[0].plot(np.arange(sub.shape[1]), sub[d],
                     color=cmap[labels[d] % len(cmap)],
                     linewidth=1.2, alpha=0.85)
    axes[0].set_xticks(range(sub.shape[1]))
    axes[0].set_xticklabels([f"{int(b[0])}/{int(b[1])}" for b in bands_selected],
                            rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Intensity")
    axes[0].set_title(f"Per-drop spectra in selected bands (k-means, k={k})")
    axes[0].grid(alpha=0.3)

    for d in range(n_drops):
        axes[1].scatter(proj[d, 0], proj[d, 1],
                        color=cmap[labels[d] % len(cmap)], s=80, edgecolor="black")
        axes[1].annotate(str(d + 1), (proj[d, 0], proj[d, 1]), fontsize=8,
                         xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title(f"Drops in PC space of selected bands  (n={n}, k={k})")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Drop clustering  -  variant={variant}  method={method}  n={n}")
    fig.tight_layout()
    out_path = FIG_ROOT / f"drop_clusters_{variant}_{method}_n{n}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    df_path = SWEEP_ROOT / "sweep_results.csv"
    if not df_path.exists():
        raise SystemExit(f"Missing {df_path}. Run drop_data_selection_sweep.py first.")
    df = pd.read_csv(df_path)
    print(f"[post] loaded {len(df)} rows from sweep_results.csv")

    plot_method_heatmap(df)
    print(f"[post] wrote {FIG_ROOT / 'best_method_grid.png'}")

    best = best_method_per_variant(df)
    print(f"[post] best per variant:")
    print(best[["variant", "method", "n_bands", "silhouette_max",
                "median_pairwise_sam"]].to_string(index=False))

    targets: list[tuple[str, str, int]] = []
    for _, row in best.iterrows():
        targets.append((row["variant"], row["method"], int(row["n_bands"])))
    for variant in df["variant"].unique():
        sub = df[(df["variant"] == variant) & (df["method"] == "ae_perturb") & (df["n_bands"] == 5)]
        if not sub.empty:
            targets.append((variant, "ae_perturb", 5))
    targets = list(dict.fromkeys(targets))

    for variant, method, n in targets:
        sub = df[(df["variant"] == variant) & (df["method"] == method) & (df["n_bands"] == n)]
        if sub.empty:
            continue
        bands_selected = parse_band_list(sub["selected_bands"].iloc[0])
        plot_selected_bands_overlay(variant, method, n, bands_selected)
        sil_k_val = sub["silhouette_k"].iloc[0]
        sil_k = int(sil_k_val) if not pd.isna(sil_k_val) else 3
        plot_drop_clusters(variant, method, n, bands_selected, sil_k)
        print(f"[post] wrote selected_bands + drop_clusters for {variant}/{method}/n{n}")


if __name__ == "__main__":
    main()
