#!/usr/bin/env python3
"""
Drop Data — Export selected slides for the CROPPED pipeline
==============================================================
Same as drop_data_export_slides.py but reads:
  - cropped recommended cubes (rows < 187 only)
  - cropped drop labels
  - selections from Drop_Data_Cropped_Sweep/<variant>/ae_perturb_results.csv

Outputs go to results/Drop_Data_Best_Slides_Cropped/<variant>/n<N>/.
Each PNG shows the dark-subtracted band image WITH NO RULER and the drop
ROI outlines overlaid.
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
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
PROC_CROP_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data Cropped"
RAW_CACHE = PROC_ROOT / "raw"
INSPECT_DIR = PROJECT_ROOT / "results" / "Drop_Data_Inspection"
SWEEP_CROP_DIR = PROJECT_ROOT / "results" / "Drop_Data_Cropped_Sweep"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Best_Slides_Cropped"

VARIANTS = ["full_cr", "dark_norm_mask_cr", "dark_norm_cr", "dark_cr", "raw_cr"]
N_VALUES = [3, 5, 8, 10]
RULER_ROW_START = 175             # match cropped pipeline
LABEL_RE = re.compile(r"(\d+)/(\d+)")


def load_cropped_cubes() -> dict[float, np.ndarray]:
    rec = json.loads((INSPECT_DIR / "recommended_cubes.json").read_text())
    cubes: dict[float, np.ndarray] = {}
    for ex_str, info in rec["recommended_per_excitation"].items():
        full = np.load(RAW_CACHE / f"{info['stem']}.npy")
        cubes[float(ex_str)] = full[:RULER_ROW_START, :, :]
    return cubes


def load_cropped_background() -> np.ndarray:
    full = np.load(RAW_CACHE / "Background.npy")
    return full[:RULER_ROW_START, :, :]


def emission_band_index(em_nm: float) -> int:
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
    boundaries = np.zeros_like(drop_labels, dtype=bool)
    for d in range(1, int(drop_labels.max()) + 1):
        m = drop_labels == d
        eroded = ndi.binary_erosion(m, iterations=1)
        boundaries |= (m & ~eroded)
    return boundaries


def render_slide(image, ex_nm, em_nm, rank, outlines, out_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    finite = image[np.isfinite(image)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    vmin = float(max(np.percentile(finite, 1), 0.0)) if finite.size else 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    im = ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
    rgba = np.zeros((*outlines.shape, 4))
    rgba[outlines, 1:3] = 1.0
    rgba[outlines, 3] = 0.85
    ax.imshow(rgba)
    ax.set_title(
        f"Rank {rank}  -  Ex={ex_nm} nm,  Em={em_nm} nm{title_suffix}\n"
        f"intensity range [{vmin:.1f}, {vmax:.1f}] (1-99.5 pct)  -  ruler-free crop",
        fontsize=11,
    )
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="Dark-subtracted intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_montage(cubes, bands, outlines, out_path, title):
    n = len(bands)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.8 * rows),
                              squeeze=False)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            ax = axes[r][c]
            if i >= n:
                ax.axis("off"); continue
            ex_nm, em_nm = bands[i]
            cube = cubes[float(ex_nm)]
            band_idx = emission_band_index(em_nm)
            band_idx = min(max(band_idx, 0), cube.shape[2] - 1)
            img = cube[:, :, band_idx]
            finite = img[np.isfinite(img)]
            vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
            vmin = float(max(np.percentile(finite, 1), 0.0)) if finite.size else 0.0
            ax.imshow(img, cmap="magma", vmin=vmin,
                      vmax=vmax if vmax > vmin else vmin + 1)
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


def export_for_variant(variant: str, cubes_corrected: dict[float, np.ndarray],
                       drop_labels: np.ndarray) -> None:
    csv = SWEEP_CROP_DIR / variant / "ae_perturb_results.csv"
    if not csv.exists():
        print(f"[slides-cr] no AE results for {variant}, skipping")
        return
    df = pd.read_csv(csv)
    outlines = get_drop_outlines(drop_labels)

    var_dir = OUT_ROOT / variant
    var_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[slides-cr] === variant: {variant} ===")

    union: dict[tuple[int, int], int] = {}
    for n in N_VALUES:
        sub = df[df["n_bands"] == n]
        if sub.empty:
            continue
        bands = parse_band_list(sub["selected_bands"].iloc[0])
        n_dir = var_dir / f"n{n}"
        n_dir.mkdir(exist_ok=True)
        mean_f = float(sub["mean_f"].iloc[0])
        print(f"[slides-cr]   n={n}  mean_F={mean_f:.2f}  bands={bands}")

        for rank_idx, (ex_nm, em_nm) in enumerate(bands, start=1):
            cube = cubes_corrected[float(ex_nm)]
            band_idx = emission_band_index(em_nm)
            band_idx = min(max(band_idx, 0), cube.shape[2] - 1)
            img = cube[:, :, band_idx]
            png_path = n_dir / f"rank{rank_idx:02d}_ex{ex_nm}_em{em_nm}nm.png"
            render_slide(img, ex_nm, em_nm, rank_idx, outlines, png_path,
                         title_suffix=f"  ({variant}, n={n})")
            union.setdefault((ex_nm, em_nm), n)

        render_montage(
            cubes_corrected, bands, outlines, n_dir / "_montage.png",
            f"AE-perturb (cropped) best slides  -  variant={variant}, n={n}, mean F={mean_f:.1f}"
        )

    if union:
        unique_bands = sorted(union.keys(), key=lambda b: union[b])
        render_montage(
            cubes_corrected, unique_bands, outlines,
            var_dir / "_all_bands_collage.png",
            f"All unique selected bands across n=3..10  -  {variant}",
        )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cubes = load_cropped_cubes()
    background = load_cropped_background()
    drop_labels = np.load(PROC_CROP_ROOT / "drop_labels.npy")
    print(f"[slides-cr] loaded {len(cubes)} cropped cubes  shape="
          f"{next(iter(cubes.values())).shape}  n_drops={int(drop_labels.max())}")

    cubes_corrected = {}
    for ex, cube in cubes.items():
        cc = cube.astype(np.float32) - background.astype(np.float32)
        np.maximum(cc, 0, out=cc)
        cubes_corrected[ex] = cc

    for v in VARIANTS:
        export_for_variant(v, cubes_corrected, drop_labels)
    print(f"\n[slides-cr] all PNGs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
