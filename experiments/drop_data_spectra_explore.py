#!/usr/bin/env python3
"""
Drop Data — Spectrum-Driven Exploration
=========================================
Forget silhouette scores. We have prior knowledge that the 18 detected drops
should fall into ~3 spectral types. This script:

  1. Plots the per-drop spectra so we can see the 3 types by eye.
  2. Hierarchically clusters drops with k=3 to identify the archetypes.
  3. Plots each archetype's per-excitation spectrum + drops within the type.
  4. Builds a "discriminative band" map: for each (excitation, emission)
     band, computes inter-type variance / intra-type variance (F-ratio).
     The bands with the highest F-ratio are the ones that *actually*
     distinguish the 3 types -- the gold-standard for selection.
  5. Compares each method's selected bands against this discriminative map
     and produces a single numeric score per method:
         retrieval_at_n = mean F-ratio rank percentile of the selected n bands.
  6. Generates a concrete numerical traceback: for the 3 archetypes, dump
     a CSV of (band, archetype_1_intensity, archetype_2_intensity,
     archetype_3_intensity, between_var, within_var, F_ratio) sorted by
     F_ratio so we can read off the most informative bands by eye.

Outputs (under results/Drop_Data_Spectra_Explore/<variant>/):
  per_drop_overview.png            -- all 18 drops, one row per excitation
  archetypes_and_members.png       -- the 3 archetypes + members per type
  discriminative_band_map.png      -- F-ratio heatmap (excitation x emission)
  band_table.csv                   -- numeric traceback, sorted by F-ratio
  method_vs_groundtruth.csv        -- per-method retrieval-at-n scores
  method_selections_overlay.png    -- each method's bands on the F-ratio map
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
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
SWEEP_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Selection_Sweep"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Spectra_Explore"

K_TYPES = 3                       # prior knowledge: 3 distinct drop types
EXCITATIONS = [310.0, 325.0, 340.0, 365.0, 385.0, 400.0, 415.0]
LABEL_RE = re.compile(r"(\d+)/(\d+)")


def parse_band_list(s):
    items = [m.group(0) for m in LABEL_RE.finditer(str(s))]
    out = []
    for item in items:
        m = LABEL_RE.search(item)
        if m:
            out.append((float(m.group(1)), float(m.group(2))))
    return out


def load_variant(variant: str):
    spec = np.load(SWEEP_ROOT / variant / "drop_mean_spectra.npy")
    bands = json.loads((SWEEP_ROOT / variant / "bands.json").read_text())
    return spec, bands


def cluster_drops_into_types(drop_spectra: np.ndarray, k: int = K_TYPES):
    """Cluster drops with two methods, return whichever gives more balanced groups."""
    norm = drop_spectra / (np.linalg.norm(drop_spectra, axis=1, keepdims=True) + 1e-12)
    Z = linkage(norm, method="ward")
    h_labels = fcluster(Z, t=k, criterion="maxclust") - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(norm)
    return h_labels, km.labels_


def plot_per_drop_overview(spec, bands_meta, types, out_path):
    """All 18 drop spectra, one subplot per excitation, color = type."""
    n_drops = spec.shape[0]
    by_ex_idx = {ex: [] for ex in EXCITATIONS}
    by_ex_em = {ex: [] for ex in EXCITATIONS}
    for i, b in enumerate(bands_meta):
        by_ex_idx[b["excitation_nm"]].append(i)
        by_ex_em[b["excitation_nm"]].append(b["emission_nm"])
    type_colors = plt.cm.tab10(np.linspace(0, 1, max(K_TYPES, 3)))
    fig, axes = plt.subplots(len(EXCITATIONS), 1, figsize=(11, 1.7 * len(EXCITATIONS)),
                             sharex=False, squeeze=False)
    for ax, ex in zip(axes[:, 0], EXCITATIONS):
        idxs = by_ex_idx[ex]
        ems = by_ex_em[ex]
        for d in range(n_drops):
            ax.plot(ems, spec[d, idxs], color=type_colors[types[d]],
                    linewidth=1.0, alpha=0.85, label=f"type {types[d]}")
        ax.set_title(f"Ex={int(ex)} nm  -  {n_drops} drops, colored by type", fontsize=9)
        ax.set_xlabel("Emission lambda (nm)")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.3)
        # dedupe legend
        h, l = ax.get_legend_handles_labels()
        seen = {}
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
        ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right")
    fig.suptitle("Per-drop spectra colored by inferred type (k=3)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_archetypes_with_members(spec, bands_meta, types, out_path):
    by_ex_idx = {ex: [] for ex in EXCITATIONS}
    by_ex_em = {ex: [] for ex in EXCITATIONS}
    for i, b in enumerate(bands_meta):
        by_ex_idx[b["excitation_nm"]].append(i)
        by_ex_em[b["excitation_nm"]].append(b["emission_nm"])
    type_colors = plt.cm.tab10(np.linspace(0, 1, K_TYPES))
    archetypes = np.stack([spec[types == t].mean(axis=0) for t in range(K_TYPES)])

    fig, axes = plt.subplots(K_TYPES, len(EXCITATIONS),
                             figsize=(2.4 * len(EXCITATIONS), 2.0 * K_TYPES),
                             sharey="row")
    for t in range(K_TYPES):
        member_idx = np.where(types == t)[0]
        for ci, ex in enumerate(EXCITATIONS):
            ax = axes[t, ci]
            idxs = by_ex_idx[ex]
            ems = by_ex_em[ex]
            for d in member_idx:
                ax.plot(ems, spec[d, idxs], color=type_colors[t], alpha=0.35,
                        linewidth=0.9)
            ax.plot(ems, archetypes[t, idxs], color=type_colors[t], linewidth=2.4)
            if t == 0:
                ax.set_title(f"Ex={int(ex)} nm", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"Type {t}\n(n={len(member_idx)})\nIntensity", fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("3 archetypal drop spectra (thick) with members (thin)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return archetypes


def f_ratio_per_band(spec: np.ndarray, types: np.ndarray) -> dict:
    """One-way ANOVA F-ratio per band: between-type / within-type variance."""
    n_total, B = spec.shape
    overall = spec.mean(axis=0)
    between = np.zeros(B)
    within = np.zeros(B)
    for t in range(K_TYPES):
        members = spec[types == t]
        if len(members) == 0:
            continue
        type_mean = members.mean(axis=0)
        between += len(members) * (type_mean - overall) ** 2
        within += ((members - type_mean) ** 2).sum(axis=0)
    df_between = K_TYPES - 1
    df_within = n_total - K_TYPES
    ms_between = between / df_between
    ms_within = within / max(df_within, 1) + 1e-12
    f_ratio = ms_between / ms_within
    return dict(
        f_ratio=f_ratio, between=between, within=within,
        ms_between=ms_between, ms_within=ms_within,
    )


def plot_discriminative_map(bands_meta, f_ratio, out_path):
    """Heatmap of F-ratio with rows=excitation, cols=emission_nm."""
    excitations = sorted({b["excitation_nm"] for b in bands_meta})
    emissions = sorted({b["emission_nm"] for b in bands_meta})
    grid = np.full((len(excitations), len(emissions)), np.nan)
    for i, b in enumerate(bands_meta):
        r = excitations.index(b["excitation_nm"])
        c = emissions.index(b["emission_nm"])
        grid[r, c] = f_ratio[i]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(grid, aspect="auto", cmap="magma",
                   extent=[emissions[0], emissions[-1],
                           len(excitations) - 0.5, -0.5],
                   interpolation="nearest")
    ax.set_yticks(range(len(excitations)))
    ax.set_yticklabels([f"{int(e)} nm" for e in excitations])
    ax.set_xlabel("Emission wavelength (nm)")
    ax.set_ylabel("Excitation wavelength")
    ax.set_title("Discriminative-band map: F-ratio (between-type / within-type) per band")
    fig.colorbar(im, ax=ax, label="F-ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_band_table(bands_meta, f_stats, out_path, archetypes):
    rows = []
    for i, b in enumerate(bands_meta):
        row = {
            "excitation_nm": b["excitation_nm"],
            "emission_nm": b["emission_nm"],
            "f_ratio": float(f_stats["f_ratio"][i]),
            "between_var": float(f_stats["ms_between"][i]),
            "within_var": float(f_stats["ms_within"][i]),
        }
        for t in range(K_TYPES):
            row[f"type{t}_intensity"] = float(archetypes[t, i])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("f_ratio", ascending=False)
    df.to_csv(out_path, index=False)
    return df


def score_method_against_groundtruth(
    df_sweep: pd.DataFrame, bands_meta, f_ratio: np.ndarray, variant: str
) -> pd.DataFrame:
    """For each (method, n) row, compute mean F-ratio rank percentile of selected bands.

    Higher = the method picked bands that the F-ratio ranks highly.
    """
    label_to_idx = {f"{int(b['excitation_nm'])}/{int(b['emission_nm'])}": i
                    for i, b in enumerate(bands_meta)}
    f_ranks = np.argsort(np.argsort(-f_ratio))     # 0 = best, len-1 = worst
    n_total = len(f_ratio)
    sub = df_sweep[df_sweep["variant"] == variant].copy()
    out_rows = []
    for _, row in sub.iterrows():
        bands = parse_band_list(row["selected_bands"])
        sel_idx = [label_to_idx.get(f"{int(ex)}/{int(em)}") for ex, em in bands]
        sel_idx = [i for i in sel_idx if i is not None]
        if not sel_idx:
            continue
        ranks = f_ranks[sel_idx]
        percentile = 1.0 - (ranks.mean() / max(n_total - 1, 1))
        mean_f = float(f_ratio[sel_idx].mean())
        max_f = float(f_ratio[sel_idx].max())
        out_rows.append({
            "variant": variant,
            "method": row["method"],
            "n_bands": row["n_bands"],
            "mean_f_ratio_of_selected": mean_f,
            "max_f_ratio_of_selected": max_f,
            "rank_percentile": percentile,
            "selected_bands": row["selected_bands"],
            "silhouette_max": row["silhouette_max"],
        })
    return pd.DataFrame(out_rows)


def plot_method_selections_overlay(
    bands_meta, f_ratio, df_method_scores, variant, n_target, out_path
):
    """Show each method's selected bands as crosses on the F-ratio map."""
    excitations = sorted({b["excitation_nm"] for b in bands_meta})
    emissions = sorted({b["emission_nm"] for b in bands_meta})
    grid = np.full((len(excitations), len(emissions)), np.nan)
    for i, b in enumerate(bands_meta):
        r = excitations.index(b["excitation_nm"])
        c = emissions.index(b["emission_nm"])
        grid[r, c] = f_ratio[i]
    methods = [m for m in df_method_scores["method"].unique()
               if not m.startswith("random_")]

    fig, axes = plt.subplots(len(methods), 1,
                             figsize=(13, 1.8 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        im = ax.imshow(grid, aspect="auto", cmap="magma",
                       extent=[emissions[0] - 5, emissions[-1] + 5,
                               len(excitations) - 0.5, -0.5],
                       interpolation="nearest", vmin=0,
                       vmax=np.nanpercentile(f_ratio, 99))
        sub = df_method_scores[
            (df_method_scores["method"] == method) &
            (df_method_scores["n_bands"] == n_target)
        ]
        if not sub.empty:
            bands = parse_band_list(sub["selected_bands"].iloc[0])
            for ex, em in bands:
                r = excitations.index(ex)
                ax.scatter(em, r, marker="x", color="cyan", s=80,
                           linewidth=2, zorder=5)
            mean_f = sub["mean_f_ratio_of_selected"].iloc[0]
            pct = sub["rank_percentile"].iloc[0]
            ax.set_title(f"{method}  (n={n_target}, mean F={mean_f:.2f}, "
                         f"rank pct={pct*100:.0f}%)", fontsize=9)
        ax.set_yticks(range(len(excitations)))
        ax.set_yticklabels([f"{int(e)}" for e in excitations])
        ax.set_ylabel("Ex (nm)")
    axes[-1].set_xlabel("Emission wavelength (nm)")
    fig.suptitle(f"Selected bands vs F-ratio map  -  variant={variant}, n={n_target}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_variant(variant: str, df_sweep: pd.DataFrame):
    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[explore] === variant: {variant} ===")
    spec, bands = load_variant(variant)
    print(f"[explore] drop_spectra: {spec.shape}")

    h_labels, k_labels = cluster_drops_into_types(spec)
    counts_h = np.bincount(h_labels, minlength=K_TYPES)
    counts_k = np.bincount(k_labels, minlength=K_TYPES)
    types = h_labels if counts_h.min() >= max(counts_k.min(), 1) else k_labels
    print(f"[explore] hierarchical counts={counts_h}, kmeans counts={counts_k} -> using "
          f"{'hierarchical' if types is h_labels else 'kmeans'}")
    np.save(out_dir / "drop_types.npy", types)

    plot_per_drop_overview(spec, bands, types, out_dir / "per_drop_overview.png")
    archetypes = plot_archetypes_with_members(
        spec, bands, types, out_dir / "archetypes_and_members.png"
    )

    f_stats = f_ratio_per_band(spec, types)
    plot_discriminative_map(bands, f_stats["f_ratio"],
                            out_dir / "discriminative_band_map.png")
    df_bands = write_band_table(bands, f_stats, out_dir / "band_table.csv", archetypes)
    print(f"[explore] top-10 discriminative bands:")
    print(df_bands.head(10).to_string(index=False))

    df_methods = score_method_against_groundtruth(df_sweep, bands, f_stats["f_ratio"], variant)
    df_methods.to_csv(out_dir / "method_vs_groundtruth.csv", index=False)

    n_target = 5
    plot_method_selections_overlay(
        bands, f_stats["f_ratio"], df_methods, variant, n_target,
        out_dir / f"method_selections_overlay_n{n_target}.png",
    )
    n_target = 3
    plot_method_selections_overlay(
        bands, f_stats["f_ratio"], df_methods, variant, n_target,
        out_dir / f"method_selections_overlay_n{n_target}.png",
    )

    pivot = df_methods.pivot_table(
        index="method", columns="n_bands",
        values="mean_f_ratio_of_selected",
    )
    print(f"\n[explore] mean F-ratio of selected bands per (method, n):")
    print(pivot.round(2).to_string())
    return df_methods


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df_sweep = pd.read_csv(SWEEP_ROOT / "sweep_results.csv")
    print(f"[explore] sweep_results: {len(df_sweep)} rows")

    all_methods = []
    for v in ["full", "dark_norm_mask", "dark_norm", "dark", "raw"]:
        rows = run_variant(v, df_sweep)
        all_methods.append(rows)
    df_all = pd.concat(all_methods, ignore_index=True)
    df_all.to_csv(OUT_ROOT / "all_methods_vs_groundtruth.csv", index=False)
    print(f"\n[explore] wrote {OUT_ROOT / 'all_methods_vs_groundtruth.csv'}")


if __name__ == "__main__":
    main()
