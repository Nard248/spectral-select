#!/usr/bin/env python3
"""
Drop Data — AE influence vs F-ratio direct comparison
=======================================================
Fast diagnostic. For each variant: dump per-band AE perturbation influence
side by side with F-ratio (between-type / within-type) and report Spearman
rho. If they're orthogonal, perturbation is measuring something other than
discriminative power -> need a different ranking criterion (Fix C).
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
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Influence_Diagnostic"
MODEL_ROOT = PROJECT_ROOT / "model_output"


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


def get_influence(variant: str, data: SpectraData):
    model_path = MODEL_ROOT / f"DropData_sweep_{variant}" / "model.pth"
    cfg = Config(
        sample_name=f"DropData_diag_{variant}",
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
        output_dir=OUT_ROOT / variant / "ae",
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


def make_grid(values, bands_meta):
    excitations = sorted({b[0] for b in bands_meta})
    emissions = sorted({b[2] for b in bands_meta})
    grid = np.full((len(excitations), len(emissions)), np.nan)
    for v, b in zip(values, bands_meta):
        r = excitations.index(b[0])
        c = emissions.index(b[2])
        grid[r, c] = v
    return grid, excitations, emissions


def run_variant(variant: str):
    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[diag] === variant: {variant} ===")
    pkl_path = PROC_ROOT / variant / "spectra_data.pkl"
    data = SpectraData.from_pickle(pkl_path)
    bands, f_ratio = load_band_truth(variant)
    infl = flatten(get_influence(variant, data), bands)

    rho, p = spearmanr(infl, f_ratio)
    print(f"[diag] Spearman rho(influence, F-ratio) = {rho:.3f} (p={p:.2e})")

    df = pd.DataFrame({
        "excitation_nm": [b[0] for b in bands],
        "emission_nm": [b[2] for b in bands],
        "ae_influence": infl,
        "f_ratio": f_ratio,
        "ae_rank": np.argsort(np.argsort(-infl)) + 1,
        "f_rank":  np.argsort(np.argsort(-f_ratio)) + 1,
    })
    df.to_csv(out_dir / "influence_vs_fratio.csv", index=False)

    inf_grid, excitations, emissions = make_grid(infl, bands)
    f_grid, _, _ = make_grid(f_ratio, bands)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    im0 = axes[0, 0].imshow(inf_grid, aspect="auto", cmap="viridis",
                            extent=[emissions[0]-5, emissions[-1]+5,
                                    len(excitations)-0.5, -0.5],
                            interpolation="nearest")
    axes[0, 0].set_yticks(range(len(excitations)))
    axes[0, 0].set_yticklabels([f"{int(e)}" for e in excitations])
    axes[0, 0].set_xlabel("Emission lambda (nm)")
    axes[0, 0].set_ylabel("Excitation (nm)")
    axes[0, 0].set_title("AE perturbation influence (higher = picked)")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(f_grid, aspect="auto", cmap="magma",
                            extent=[emissions[0]-5, emissions[-1]+5,
                                    len(excitations)-0.5, -0.5],
                            interpolation="nearest")
    axes[0, 1].set_yticks(range(len(excitations)))
    axes[0, 1].set_yticklabels([f"{int(e)}" for e in excitations])
    axes[0, 1].set_xlabel("Emission lambda (nm)")
    axes[0, 1].set_ylabel("Excitation (nm)")
    axes[0, 1].set_title("F-ratio (higher = discriminates 3 types)")
    fig.colorbar(im1, ax=axes[0, 1])

    axes[1, 0].scatter(f_ratio, infl, s=14, alpha=0.6)
    axes[1, 0].set_xlabel("F-ratio")
    axes[1, 0].set_ylabel("AE influence")
    axes[1, 0].set_title(f"Per-band scatter  (Spearman rho = {rho:.3f})")
    axes[1, 0].grid(alpha=0.3)

    rank_diff = (df["f_rank"] - df["ae_rank"]).values
    axes[1, 1].hist(rank_diff, bins=40)
    axes[1, 1].set_xlabel("F-ratio rank - AE-influence rank")
    axes[1, 1].set_ylabel("Number of bands")
    axes[1, 1].set_title("How far apart are the rankings?")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axvline(0, color="red", linestyle="--", alpha=0.5)

    fig.suptitle(f"AE influence vs F-ratio  -  variant={variant}")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic.png", dpi=140)
    plt.close(fig)
    return rho


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rhos = {}
    for v in ["full", "dark_norm_mask", "dark_norm", "dark", "raw"]:
        try:
            rhos[v] = run_variant(v)
        except Exception as e:
            print(f"[diag] {v} failed: {e}")
    summary = pd.Series(rhos, name="spearman_rho_inf_vs_F")
    print("\n[diag] Spearman correlation (AE influence vs F-ratio) per variant:")
    print(summary.round(3))
    summary.to_csv(OUT_ROOT / "correlation_summary.csv")


if __name__ == "__main__":
    main()
