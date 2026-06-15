#!/usr/bin/env python3
"""
Drop Data — Latent-dimension config sweep (Fix A.5)
=====================================================
Tests whether changing how the AE perturbation method selects which latent
dimensions to perturb (and how many) recovers high-F-ratio bands.

Configs tested (cartesian product):
  dimension_selection_method : {activation, variance, pca}
  n_important_dimensions     : {3, 5, 8, 12}

For each variant + config, loads the cached AE weights (no retraining) and
re-runs only the influence-scoring stage. Each combo's selection is scored
against the F-ratio ground truth from spectra_explore.

Outputs:
  results/Drop_Data_Dim_Sweep/<variant>/dim_sweep.csv
  results/Drop_Data_Dim_Sweep/<variant>/dim_sweep.png  (heatmap)
  results/Drop_Data_Dim_Sweep/summary.csv
  results/Drop_Data_Dim_Sweep/best_config.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral_select import Analyzer, Config, SpectraData  # noqa: E402

PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
SWEEP_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Selection_Sweep"
EXPLORE_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Spectra_Explore"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Dim_Sweep"
MODEL_ROOT = PROJECT_ROOT / "model_output"

VARIANTS = ["full", "dark_norm_mask", "dark_norm", "dark", "raw"]
DIM_METHODS = ["activation", "variance", "pca"]
N_DIMS_GRID = [3, 5, 8, 12]
N_BAND_GRID = list(range(3, 11))


def load_band_truth(variant: str):
    bands_meta = json.loads((SWEEP_ROOT / variant / "bands.json").read_text())
    table = pd.read_csv(EXPLORE_ROOT / variant / "band_table.csv")
    f_by_key = {(int(r.excitation_nm), int(r.emission_nm)): float(r.f_ratio)
                for r in table.itertuples()}
    bands = [(b["excitation_nm"], b["emission_idx"], b["emission_nm"])
             for b in bands_meta]
    f_ratio = np.array([f_by_key.get((int(ex), int(em)), 0.0)
                        for ex, _, em in bands])
    return bands, f_ratio


def get_influence_for_config(variant, data, dim_method, n_dims):
    model_path = MODEL_ROOT / f"DropData_sweep_{variant}" / "model.pth"
    cfg = Config(
        sample_name=f"DropData_dim_{variant}_{dim_method}_{n_dims}",
        n_bands_to_select=10,
        device="mps",
        training_epochs=1,
        patch_size=16,
        patch_stride=8,
        n_baseline_patches=30,
        n_important_dimensions=n_dims,
        dimension_selection_method=dim_method,
        save_visualizations=False,
        save_tiff_layers=False,
        save_detailed_results=False,
        model_path=model_path,
        output_dir=OUT_ROOT / variant / f"{dim_method}_n{n_dims}",
    )
    analyzer = Analyzer(cfg)
    analyzer.fit(data)
    inf = analyzer.influence_matrix or {}
    out = {}
    for ex in data.excitation_wavelengths:
        if ex in inf:
            out[float(ex)] = np.asarray(inf[ex], dtype=float)
        elif str(ex) in inf:
            out[float(ex)] = np.asarray(inf[str(ex)], dtype=float)
    return out


def flatten_influence(influence, bands_order):
    out = np.zeros(len(bands_order), dtype=float)
    for i, (ex, em_idx, _) in enumerate(bands_order):
        if ex in influence and em_idx < len(influence[ex]):
            out[i] = float(influence[ex][em_idx])
    return out


def evaluate_top_n(scores, f_ratio, n):
    sel = np.argsort(-scores)[:n]
    sub = f_ratio[sel]
    n_total = len(f_ratio)
    rank_pct = 1.0 - (np.argsort(np.argsort(-f_ratio))[sel].mean()
                      / max(n_total - 1, 1))
    return dict(
        mean_f=float(sub.mean()),
        max_f=float(sub.max()),
        rank_percentile=float(rank_pct),
        selected_idx=sel.tolist(),
    )


def run_variant(variant: str) -> pd.DataFrame:
    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[dim] === variant: {variant} ===")
    pkl_path = PROC_ROOT / variant / "spectra_data.pkl"
    data = SpectraData.from_pickle(pkl_path)
    bands, f_ratio = load_band_truth(variant)

    rows = []
    for dim_method in DIM_METHODS:
        for n_dims in N_DIMS_GRID:
            try:
                t0 = time.time()
                inf_dict = get_influence_for_config(variant, data, dim_method, n_dims)
                inf = flatten_influence(inf_dict, bands)
                dt = time.time() - t0
                for n in N_BAND_GRID:
                    m = evaluate_top_n(inf, f_ratio, n)
                    rows.append({
                        "variant": variant,
                        "dim_method": dim_method,
                        "n_dims": n_dims,
                        "n_bands": n,
                        "mean_f": m["mean_f"],
                        "max_f": m["max_f"],
                        "rank_percentile": m["rank_percentile"],
                        "selected_bands": [
                            f"{int(bands[i][0])}/{int(bands[i][2])}"
                            for i in m["selected_idx"]
                        ],
                    })
                pulled = [r["mean_f"] for r in rows
                          if r["dim_method"] == dim_method
                          and r["n_dims"] == n_dims
                          and r["n_bands"] == 5]
                print(f"[dim] {dim_method:11s} n_dims={n_dims:2d}  "
                      f"mean_F(n=5)={pulled[0]:.2f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"[dim] {dim_method} n_dims={n_dims} failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dim_sweep.csv", index=False)

    pivot = df.pivot_table(index=["dim_method", "n_dims"],
                           columns="n_bands", values="mean_f")
    print(f"\n[dim] mean F-ratio per (dim_method, n_dims) x n_bands:")
    print(pivot.round(2).to_string())

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"{m}/n={n}" for m, n in pivot.index], fontsize=9)
    ax.set_xlabel("n bands selected")
    ax.set_title(f"AE-perturb mean F-ratio: dim_method x n_important_dims  -  {variant}")
    fig.colorbar(im, ax=ax, label="mean F-ratio")
    fig.tight_layout()
    fig.savefig(out_dir / "dim_sweep.png", dpi=140)
    plt.close(fig)
    return df


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for v in VARIANTS:
        try:
            all_rows.append(run_variant(v))
        except Exception as e:
            print(f"[dim] variant {v} failed: {e}")
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(OUT_ROOT / "all_variants.csv", index=False)

    summary = (df_all.groupby(["variant", "dim_method", "n_dims"])["mean_f"]
                     .mean().reset_index())
    summary.to_csv(OUT_ROOT / "summary.csv", index=False)

    best = (summary.sort_values(["variant", "mean_f"], ascending=[True, False])
                   .groupby("variant").head(1))
    print(f"\n[dim] === best (dim_method, n_dims) per variant ===")
    print(best.to_string(index=False))
    best.to_json(OUT_ROOT / "best_config.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
