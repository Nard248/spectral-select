#!/usr/bin/env python3
"""
Drop Data — SNR-weighted re-ranking of AE perturbation influence
=================================================================
Fix A test: weight per-band influence by per-band SNR/variance and check
whether the F-ratio of the top-N selections rises above the unweighted
baseline (and above random aggregate).

For each variant:
  1. Load SpectraData + cached AE model from the sweep (no retraining).
  2. Run analyzer.fit -> influence_matrix.
  3. Compute per-band variance and SNR over the in-mask pixels.
  4. Try several weightings on the influence vector.
  5. Re-rank, score against F-ratio truth from spectra_explore.

Outputs: results/Drop_Data_SNR_Rerank/<variant>/comparison.{csv,png}
         results/Drop_Data_SNR_Rerank/summary_meanF.csv
         results/Drop_Data_SNR_Rerank/best_weighting.json
"""
from __future__ import annotations

import json
import os
import sys
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
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_SNR_Rerank"
MODEL_ROOT = PROJECT_ROOT / "model_output"

VARIANTS = ["full", "dark_norm_mask", "dark_norm", "dark", "raw"]
N_GRID = list(range(3, 11))


def load_band_truth(variant: str) -> dict:
    bands_meta = json.loads((SWEEP_ROOT / variant / "bands.json").read_text())
    table = pd.read_csv(EXPLORE_ROOT / variant / "band_table.csv")
    f_by_key = {(int(r.excitation_nm), int(r.emission_nm)): float(r.f_ratio)
                for r in table.itertuples()}
    bands = [(b["excitation_nm"], b["emission_idx"], b["emission_nm"])
             for b in bands_meta]
    f_ratio = np.array([
        f_by_key.get((int(ex), int(em)), 0.0) for ex, _, em in bands
    ])
    return dict(bands=bands, f_ratio=f_ratio)


def get_influence_matrix(variant: str, data: SpectraData) -> dict[float, np.ndarray]:
    model_path = MODEL_ROOT / f"DropData_sweep_{variant}" / "model.pth"
    cfg = Config(
        sample_name=f"DropData_snr_{variant}",
        n_bands_to_select=10,
        device="mps",
        training_epochs=1,
        patch_size=16,
        patch_stride=8,
        n_baseline_patches=30,
        n_important_dimensions=12,
        save_visualizations=False,
        save_tiff_layers=False,
        save_detailed_results=False,
        model_path=model_path,
        output_dir=OUT_ROOT / variant / "ae_run",
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"No cached AE model for {variant} at {model_path}. Run the sweep first."
        )
    print(f"[snr] {variant}: loading cached AE from {model_path}")
    analyzer = Analyzer(cfg)
    analyzer.fit(data)
    inf = analyzer.influence_matrix or {}
    out: dict[float, np.ndarray] = {}
    for ex in data.excitation_wavelengths:
        if ex in inf:
            out[float(ex)] = np.asarray(inf[ex], dtype=float)
        elif str(ex) in inf:
            out[float(ex)] = np.asarray(inf[str(ex)], dtype=float)
    return out


def per_band_stats(data: SpectraData):
    bands = []
    var_list = []
    snr_list = []
    H, W = data.spatial_shape
    if data.mask is not None:
        mask = data.mask.astype(bool)
    else:
        mask = np.ones((H, W), dtype=bool)
        mask[187:, :] = False
    flat_idx = np.flatnonzero(mask.ravel())
    for ex in data.excitation_wavelengths:
        ed = data.get_excitation(ex)
        cube_flat = ed.cube.reshape(-1, ed.cube.shape[2])[flat_idx]
        for i, em in enumerate(ed.emission_wavelengths):
            col = cube_flat[:, i]
            mu = float(col.mean())
            sigma = float(col.std() + 1e-12)
            var = float(col.var())
            snr = mu / sigma
            bands.append((float(ex), int(i), float(em)))
            var_list.append(var)
            snr_list.append(snr)
    return np.array(var_list), np.array(snr_list), bands


def flatten_influence(influence, bands_order):
    out = np.zeros(len(bands_order), dtype=float)
    for i, (ex, em_idx, _) in enumerate(bands_order):
        if ex in influence and em_idx < len(influence[ex]):
            out[i] = float(influence[ex][em_idx])
    return out


def rank_percentile(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def apply_weightings(inf, var, snr):
    snr_db = 20.0 * np.log10(np.maximum(snr, 1e-3))
    snr_db = np.clip(snr_db, 0.0, None)
    return {
        "unweighted": inf,
        "var":       inf * var,
        "log_var":   inf * np.log1p(var),
        "sqrt_var":  inf * np.sqrt(var),
        "rank_var":  inf * rank_percentile(var),
        "snr_db":    inf * snr_db,
    }


def rank_top_n(scores, n):
    return np.argsort(-scores)[:n]


def evaluate_ranking(selected, f_ratio):
    sub = f_ratio[selected]
    n_total = len(f_ratio)
    rank_pct = 1.0 - (np.argsort(np.argsort(-f_ratio))[selected].mean()
                      / max(n_total - 1, 1))
    return {
        "mean_f": float(sub.mean()),
        "max_f": float(sub.max()),
        "min_f": float(sub.min()),
        "rank_percentile": float(rank_pct),
    }


def run_variant(variant: str) -> pd.DataFrame:
    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[snr] === variant: {variant} ===")
    pkl_path = PROC_ROOT / variant / "spectra_data.pkl"
    data = SpectraData.from_pickle(pkl_path)
    truth = load_band_truth(variant)
    influence = get_influence_matrix(variant, data)

    var, snr, bands = per_band_stats(data)
    inf = flatten_influence(influence, bands)
    print(f"[snr] inf={inf.shape}, var=[{var.min():.3g},{var.max():.3g}], "
          f"snr=[{snr.min():.2f},{snr.max():.2f}]")
    f_ratio = truth["f_ratio"]
    weightings = apply_weightings(inf, var, snr)

    rows = []
    for n in N_GRID:
        for name, score in weightings.items():
            sel = rank_top_n(score, n)
            metrics = evaluate_ranking(sel, f_ratio)
            rows.append({
                "variant": variant,
                "weighting": name,
                "n_bands": n,
                **metrics,
                "selected": [
                    f"{int(bands[i][0])}/{int(bands[i][2])}" for i in sel
                ],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison.csv", index=False)
    print(f"[snr] mean F-ratio per weighting (averaged over n=3..10):")
    print(df.groupby("weighting")["mean_f"].mean().sort_values(ascending=False).round(2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for w in df["weighting"].unique():
        sub = df[df["weighting"] == w].sort_values("n_bands")
        axes[0].plot(sub["n_bands"], sub["mean_f"], marker="o",
                     label=w, linewidth=1.6, alpha=0.85)
        axes[1].plot(sub["n_bands"], sub["rank_percentile"], marker="o",
                     label=w, linewidth=1.6, alpha=0.85)
    axes[0].set_title(f"Mean F-ratio of selected bands  -  {variant}")
    axes[0].set_xlabel("n bands")
    axes[0].set_ylabel("Mean F-ratio")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_title(f"Rank percentile  -  {variant}")
    axes[1].set_xlabel("n bands")
    axes[1].set_ylabel("Rank percentile")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=140)
    plt.close(fig)
    return df


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for v in VARIANTS:
        try:
            df = run_variant(v)
            all_rows.append(df)
        except Exception as e:
            print(f"[snr] variant {v} failed: {e}")
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(OUT_ROOT / "all_variants.csv", index=False)
    summary = (df_all.groupby(["variant", "weighting"])["mean_f"].mean()
                     .unstack().round(2))
    summary.to_csv(OUT_ROOT / "summary_meanF.csv")
    print(f"\n[snr] === FINAL SUMMARY (mean F across n=3..10) ===")
    print(summary)
    best = (df_all.groupby(["variant", "weighting"])["mean_f"].mean()
                  .reset_index()
                  .sort_values(["variant", "mean_f"], ascending=[True, False])
                  .groupby("variant").head(1))
    print(f"\n[snr] best weighting per variant:")
    print(best.to_string(index=False))
    best.to_json(OUT_ROOT / "best_weighting.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
