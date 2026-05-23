#!/usr/bin/env python3
"""
Drop Data — Test the variance-inversion fix
==============================================
Hypothesis: the analyzer's default normalization_method="variance" divides
influence by band variance. On Drop Data the discriminative bands ALSO
have high variance, so this normalization inverts the ranking.

Test: re-run AE-perturb with each normalization_method and check if mean
F-ratio of the top-N selections rises above random (~55) on the well-
preprocessed variants.
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
from scipy.stats import spearmanr

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral_select import Analyzer, Config, SpectraData  # noqa: E402

PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
SWEEP_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Selection_Sweep"
EXPLORE_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Spectra_Explore"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Norm_Fix"
MODEL_ROOT = PROJECT_ROOT / "model_output"

VARIANTS = ["full", "dark_norm_mask", "dark_norm", "dark", "raw"]
NORMALIZATIONS = ["variance", "max_per_excitation", "none"]
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


def get_influence(variant, data, normalization):
    model_path = MODEL_ROOT / f"DropData_sweep_{variant}" / "model.pth"
    cfg = Config(
        sample_name=f"DropData_norm_{variant}_{normalization}",
        n_bands_to_select=10,
        device="mps",
        training_epochs=1,
        patch_size=16,
        patch_stride=8,
        n_baseline_patches=30,
        n_important_dimensions=12,
        normalization_method=normalization,
        save_visualizations=False,
        save_tiff_layers=False,
        save_detailed_results=False,
        model_path=model_path,
        output_dir=OUT_ROOT / variant / normalization,
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


def flatten(infl, bands):
    out = np.zeros(len(bands), dtype=float)
    for i, (ex, idx, _) in enumerate(bands):
        if ex in infl and idx < len(infl[ex]):
            out[i] = float(infl[ex][idx])
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
    print(f"\n[norm-fix] === variant: {variant} ===")
    pkl_path = PROC_ROOT / variant / "spectra_data.pkl"
    data = SpectraData.from_pickle(pkl_path)
    bands, f_ratio = load_band_truth(variant)

    rows = []
    for norm in NORMALIZATIONS:
        try:
            inf_dict = get_influence(variant, data, norm)
            inf = flatten(inf_dict, bands)
            rho, _ = spearmanr(inf, f_ratio)
            print(f"[norm-fix] norm={norm:18s}  Spearman rho(inf, F)={rho:+.3f}")
            for n in N_BAND_GRID:
                m = evaluate_top_n(inf, f_ratio, n)
                rows.append({
                    "variant": variant,
                    "normalization": norm,
                    "n_bands": n,
                    "mean_f": m["mean_f"],
                    "max_f": m["max_f"],
                    "rank_percentile": m["rank_percentile"],
                    "spearman_rho": float(rho),
                    "selected_bands": [
                        f"{int(bands[i][0])}/{int(bands[i][2])}"
                        for i in m["selected_idx"]
                    ],
                })
        except Exception as e:
            print(f"[norm-fix] norm={norm} failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "norm_fix.csv", index=False)
    pivot = df.pivot_table(index="normalization", columns="n_bands", values="mean_f")
    print(f"[norm-fix] mean F-ratio per (normalization, n_bands):")
    print(pivot.round(2).to_string())
    return df


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for v in VARIANTS:
        try:
            all_rows.append(run_variant(v))
        except Exception as e:
            print(f"[norm-fix] {v} failed: {e}")
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(OUT_ROOT / "all_variants.csv", index=False)
    summary = (df_all.groupby(["variant", "normalization"])["mean_f"]
                     .mean().unstack().round(2))
    summary.to_csv(OUT_ROOT / "summary.csv")
    print(f"\n[norm-fix] === FINAL SUMMARY (mean F across n=3..10) ===")
    print(summary)
    rho_summary = (df_all.groupby(["variant", "normalization"])["spearman_rho"]
                          .mean().unstack().round(3))
    rho_summary.to_csv(OUT_ROOT / "spearman_summary.csv")
    print(f"\n[norm-fix] Spearman rho (AE influence vs F-ratio) per normalization:")
    print(rho_summary)


if __name__ == "__main__":
    main()
