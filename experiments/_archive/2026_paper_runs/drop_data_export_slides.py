#!/usr/bin/env python3
"""
Drop Data — Export selected (excitation, emission) slides as PNG
==================================================================
Takes the AE-perturb selections produced under the fixed
normalization_method="max_per_excitation" config (from drop_data_norm_fix.py)
and exports each chosen band as a spatial-image PNG.

For each variant and each n in {3, 5, 8, 10}:
  results/Drop_Data_Best_Slides/<variant>/n<N>/
      _montage.png                  - all n bands side-by-side
      rank01_ex<EX>_em<EM>nm.png    - individual band slices
      rank02_ex<EX>_em<EM>nm.png
      ...
  results/Drop_Data_Best_Slides/<variant>/_all_bands_collage.png
                                    - union of bands across n=3..10

Each band is shown as the dark-subtracted intensity image (so the actual
fluorescence is visible), with drop ROI outlines overlaid.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
RAW_CACHE = PROC_ROOT / "raw"
INSPECT_DIR = PROJECT_ROOT / "results" / "Drop_Data_Inspection"
NORM_FIX_DIR = PROJECT_ROOT / "results" / "Drop_Data_Norm_Fix"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Best_Slides"

VARIANTS = ["full", "dark_norm_mask", "dark_norm", "dark", "raw"]
N_VALUES = [3, 5, 8, 10]
LABEL_RE = re.compile(r"(\d+)/(\d+)")

# Use the recommended cube per excitation produced by drop_data_montage.py
def load_recommended_cubes() -> dict[float, np.ndarray]:
    rec = json.loads((INSPECT_DIR / "recommended_cubes.json").read_text())
    cubes: dict[float, np.ndarray] = {}
    for ex_str, info in rec["recommended_per_excitation"].items():
        cubes[float(ex_str)] = np.load(RAW_CACHE / f"{info['stem']}.npy")
    return cubes


def emission_band_index(ex_nm: float, em_nm: float) -> int:
    """All cubes are 31 bands at 420..720 nm in 10 nm steps."""
    return int(round((em_nm - 420.0) / 10.0))


def parse_band_list(s):
    items = [m.group(0) for m in LABEL_RE.finditer(str(s))]
    out = []
    for it in items:
        m = LABEL_RE.search(it)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def get_drop_outlines(drop_labels: np.ndarray) -> np.ndarray:
    """Boolean mask of drop boundaries for overlay."""
    boundaries = np.zeros_like(drop_labels, dtype=bool)
    for d in range(1, int(drop_labels.max()) + 1):
        m = drop_labels == d
        eroded = ndi.binary_erosion(m, iterations=1)
        boundaries |= (m & ~eroded)
    return boundaries


def render_slide(
    image: np.ndarray, ex_nm: int, em_nm: int, rank: int, outlines: np.ndarray,
    out_path: Path, title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    finite = image[np.isfinite(image)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    vmin = float(max(np.percentile(finite, 1), 0.0)) if finite.size else 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    im = ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
    if outlines is not None:
        # Draw drop outlines as a translucent cyan overlay
        rgba = np.zeros((*outlines.shape, 4))
        rgba[outlines, 0] = 0.0
        rgba[outlines, 1] = 1.0
        rgba[outlines, 2] = 1.0
        rgba[outlines, 3] = 0.85
        ax.imshow(rgba)
    ax.set_title(
        f"Rank {rank}  -  Ex={ex_nm} nm,  Em={em_nm} nm{title_suffix}\n"
        f"intensity range [{vmin:.1f}, {vmax:.1f}] (1-99.5 pct)",
        fontsize=11,
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="Dark-subtracted intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_montage(
    cubes_corrected: dict[float, np.ndarray], bands: list[tuple[int, int]],
    drop_labels: np.ndarray, outlines: np.ndarray, out_path: Path,
    title: str,
) -> None:
    n = len(bands)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows),
                              squeeze=False)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            ax = axes[r][c]
            if i >= n:
                ax.axis("off")
                continue
            ex_nm, em_nm = bands[i]
            cube = cubes_corrected[float(ex_nm)]
            band_idx = emission_band_index(ex_nm, em_nm)
            band_idx = min(max(band_idx, 0), cube.shape[2] - 1)
            img = cube[:, :, band_idx]
            finite = img[np.isfinite(img)]
            vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
            vmin = float(max(np.percentile(finite, 1), 0.0)) if finite.size else 0.0
            ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)
            rgba = np.zeros((*outlines.shape, 4))
            rgba[outlines, 1:3] = 1.0
            rgba[outlines, 3] = 0.85
            ax.imshow(rgba)
            ax.set_title(f"Rank {i+1}: Ex={ex_nm}/Em={em_nm}", fontsize=10)
            ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def export_for_variant(variant: str, cubes: dict[float, np.ndarray],
                       background: np.ndarray, drop_labels: np.ndarray) -> None:
    norm_csv = NORM_FIX_DIR / variant / "norm_fix.csv"
    if not norm_csv.exists():
        print(f"[slides] no norm_fix.csv for {variant}, skipping")
        return
    df = pd.read_csv(norm_csv)
    df = df[df["normalization"] == "max_per_excitation"]
    if df.empty:
        print(f"[slides] no max_per_excitation rows for {variant}, skipping")
        return

    # Pre-compute dark-subtracted cubes once
    cubes_corrected = {}
    for ex, cube in cubes.items():
        cc = cube.astype(np.float32) - background.astype(np.float32)
        np.maximum(cc, 0, out=cc)
        cubes_corrected[ex] = cc

    outlines = get_drop_outlines(drop_labels)
    var_dir = OUT_ROOT / variant
    var_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[slides] === variant: {variant} ===")

    union: dict[tuple[int, int], int] = {}   # band -> first n where it appears
    for n in N_VALUES:
        sub = df[df["n_bands"] == n]
        if sub.empty:
            continue
        bands = parse_band_list(sub["selected_bands"].iloc[0])
        n_dir = var_dir / f"n{n}"
        n_dir.mkdir(exist_ok=True)
        print(f"[slides] n={n}: {bands}")

        # individual rank PNGs
        for rank_idx, (ex_nm, em_nm) in enumerate(bands, start=1):
            cube = cubes_corrected[float(ex_nm)]
            band_idx = emission_band_index(ex_nm, em_nm)
            band_idx = min(max(band_idx, 0), cube.shape[2] - 1)
            img = cube[:, :, band_idx]
            png_path = (n_dir /
                        f"rank{rank_idx:02d}_ex{ex_nm}_em{em_nm}nm.png")
            render_slide(img, ex_nm, em_nm, rank_idx, outlines, png_path,
                          title_suffix=f"  ({variant}, n={n})")
            union.setdefault((ex_nm, em_nm), n)

        # montage of this n's bands
        mean_f = float(sub["mean_f"].iloc[0])
        render_montage(cubes_corrected, bands, drop_labels, outlines,
                        n_dir / "_montage.png",
                        title=f"AE-perturb best slides  -  variant={variant}, n={n}, mean F-ratio={mean_f:.1f}")

    # Collage of every unique band that appeared across n=3..10
    if union:
        unique_bands = list(union.keys())
        unique_bands.sort(key=lambda b: union[b])  # earliest n first
        render_montage(cubes_corrected, unique_bands, drop_labels, outlines,
                        var_dir / "_all_bands_collage.png",
                        title=f"All unique selected bands across n=3..10  -  {variant}")
        print(f"[slides] {variant}: {len(unique_bands)} unique bands across all n")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cubes = load_recommended_cubes()
    background = np.load(RAW_CACHE / "Background.npy")
    drop_labels = np.load(PROC_ROOT / "drop_labels.npy")
    print(f"[slides] background loaded {background.shape}, "
          f"{len(cubes)} cubes, {int(drop_labels.max())} drops")

    for v in VARIANTS:
        export_for_variant(v, cubes, background, drop_labels)
    print(f"\n[slides] all PNGs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
