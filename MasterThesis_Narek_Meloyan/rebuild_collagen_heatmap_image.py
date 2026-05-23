"""Re-render the Collagen Sponges wavelength importance heatmap in the same
visual format as the Lichens heatmap currently used in the thesis.

Format alignment with `figures/wavelength_heatmap.png` (Lichens):
- Banner aspect ratio (figsize ~ (10, 4))
- Axis labels: "Excitation Wavelength (nm)" / "Emission Wavelength (nm)"
- Colorbar: 0.0 -> 1.0, no text label (only numeric ticks)
- Single grey for Rayleigh-invalid cells (no light/dark distinction)
- No "Cutoff regions" legend at the bottom
- No title

Input: results/Pepsin_Paper_Figures/wavelength_heatmap_v2_importance.csv
Output: MasterThesis_Narek_Meloyan/figures/collagen_sponges/wavelength_heatmap.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPORTANCE_CSV = (ROOT / "results" / "Pepsin_Paper_Figures"
                  / "wavelength_heatmap_v2_importance.csv")
OUT_PATH = (ROOT / "MasterThesis_Narek_Meloyan" / "figures"
            / "collagen_sponges" / "wavelength_heatmap.png")

# Rayleigh-cutoff rule (same as preprocessor): em < ex + 40 OR |em - 2*ex| < 40
CUTOFF_OFFSET_NM = 40


def is_valid_pair(ex: int, em: int) -> bool:
    if em < ex + CUTOFF_OFFSET_NM:
        return False
    if abs(em - 2 * ex) < CUTOFF_OFFSET_NM:
        return False
    return True


def main() -> None:
    df = pd.read_csv(IMPORTANCE_CSV, index_col=0)
    excitations = [int(x) for x in df.index]
    emissions = [int(x) for x in df.columns]
    importance = df.values.astype(float)

    # Build invalid-cell mask via Rayleigh cutoffs
    invalid = np.zeros_like(importance, dtype=bool)
    for i, ex in enumerate(excitations):
        for j, em in enumerate(emissions):
            if not is_valid_pair(ex, em):
                invalid[i, j] = True
    display = importance.copy()
    display[invalid] = np.nan

    # Plot — matching the Lichens banner layout
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="#D3D3D3")  # single grey for invalid (matches Lichens)
    masked = np.ma.masked_invalid(display)
    im = ax.pcolormesh(masked, cmap=cmap, vmin=0.0, vmax=1.0,
                       edgecolors="face", linewidth=0, rasterized=True)

    # Ticks every 10 nm on emission; per-excitation on the y-axis
    ax.set_xticks([j + 0.5 for j in range(len(emissions))])
    ax.set_xticklabels([str(e) for e in emissions], rotation=90, fontsize=11)
    ax.set_yticks([i + 0.5 for i in range(len(excitations))])
    ax.set_yticklabels([str(e) for e in excitations], fontsize=11)

    ax.set_xlabel("Emission Wavelength (nm)", fontsize=13)
    ax.set_ylabel("Excitation Wavelength (nm)", fontsize=13)

    # Colorbar: 0–1, numeric only (no label string — matches Lichens)
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
