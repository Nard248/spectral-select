#!/usr/bin/env python3
"""
Pepsin — Swap Embedded Figures In-Place (Preserves User Text Edits)
=====================================================================
Does NOT regenerate either docx. Only replaces the image blobs inside
them, preserving any text edits made since they were originally generated.

Fixes:
  1. Abstract figure: peak annotation moved off the data band into the
     empty lower-right region.
  2. Summary-doc heatmap: a single uniform gray now marks all cutoff
     cells (was light/dark split by cutoff type).

Target images inside each docx:
  - Abstract: word/media/image1.png  (accuracy envelope)
  - Summary:  word/media/image5.png  (wavelength heatmap)
"""

import json
import zipfile
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "Pepsin_Paper_Figures"

ABSTRACT_DOCX = OUT_DIR / "IASIM2026_Pepsin_Abstract.docx"
SUMMARY_DOCX = OUT_DIR / "Pepsin_Collagen_Wavelength_Selection_Report.docx"

ABSTRACT_IMAGE_NAME = "word/media/image1.png"
SUMMARY_HEATMAP_NAME = "word/media/image5.png"

PIPELINE_CSV = PROJECT_ROOT / "results" / "Collagen_Pepsin_Normalized" / "results.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "Collagen_Pepsin_Normalized" / "experiments"

CUTOFF_OFFSET_NM = 30
EM_GRID = list(range(420, 730, 10))
EX_GRID = [310, 325, 340, 365, 385, 400]

# One uniform gray for all cutoff / physically-invalid cells.
CUTOFF_GRAY = '#B0B0B0'


def build_abstract_figure(out_path):
    df = pd.read_csv(PIPELINE_CSV)
    bl = df[df['config'] == 'BASELINE'].iloc[0]
    sel = df[df['config'] != 'BASELINE']
    stats = sel.groupby('n_features').agg(
        mn=('accuracy', 'min'), mx=('accuracy', 'max'), mean=('accuracy', 'mean')
    ).reset_index().sort_values('n_features')

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.fill_between(stats['n_features'], stats['mn'], stats['mx'],
                    alpha=0.25, color='#4A90D9', label='Range (min\u2013max)')
    ax.plot(stats['n_features'], stats['mean'], color='black', lw=1.8, label='Mean')
    ax.plot(stats['n_features'], stats['mx'],
            color='#1B7A2B', lw=2.2, linestyle='--', label='Best')
    ax.axhline(y=bl['accuracy'], color='red', linestyle='--', lw=1.3,
               label=f'Baseline ({bl["accuracy"]:.1%})')

    bi = stats['mx'].idxmax()
    bn, ba = stats.loc[bi, 'n_features'], stats.loc[bi, 'mx']
    ax.scatter([bn], [ba], color='gold', s=180, zorder=7, marker='*',
               edgecolor='#1B7A2B', lw=1.4)

    # Label in the empty lower-right region with arrow to the star.
    ax.annotate(f"Peak {ba:.1%}\n({int(bn)} bands)",
                xy=(bn, ba),
                xytext=(95, 0.56),
                fontsize=10, fontweight='bold', color='#1B7A2B',
                ha='center', va='center',
                arrowprops=dict(arrowstyle='-|>', color='#1B7A2B', lw=1.2,
                                shrinkA=1, shrinkB=6),
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='#1B7A2B', alpha=0.95),
                zorder=8)

    ax.set_xlabel('Number of Bands Selected', fontsize=11)
    ax.set_ylabel('Classification Accuracy', fontsize=11)
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
    ax.set_xlim(0, 160)
    ax.set_ylim(0.45, 0.92)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path.name} ({out_path.stat().st_size/1024:.0f} KB)")


def rayleigh_invalid(ex, em, offset=CUTOFF_OFFSET_NM):
    return em < ex + offset


def second_order_invalid(ex, em, offset=CUTOFF_OFFSET_NM):
    return (2 * ex - offset) <= em <= (2 * ex + offset)


def is_invalid(ex, em):
    return rayleigh_invalid(ex, em) or second_order_invalid(ex, em)


def load_importance():
    df = pd.read_csv(PIPELINE_CSV)
    baseline = df[df['config'] == 'BASELINE'].iloc[0]
    non_bl = df[df['config'] != 'BASELINE']
    above = non_bl[non_bl['accuracy'] > baseline['accuracy']]
    if len(above) < 5:
        above = non_bl.nlargest(30, 'accuracy')
    use = set(above['config'].values)

    imp = np.zeros((len(EX_GRID), len(EM_GRID)))
    cnt = 0
    for cfg in use:
        wf = EXPERIMENTS_DIR / cfg / "wavelengths.json"
        if not wf.exists():
            continue
        with open(wf) as f:
            wls = json.load(f)
        n = max(len(wls), 1)
        for w in wls:
            ex, em = w['excitation'], w['emission']
            if ex not in EX_GRID:
                continue
            emr = int(round(em / 10) * 10)
            if emr not in EM_GRID:
                continue
            i = EX_GRID.index(ex)
            j = EM_GRID.index(emr)
            imp[i, j] += 1.0 - (w['rank'] - 1) / n
        cnt += 1
    if cnt > 0:
        imp /= cnt
    return imp, cnt


def build_heatmap_figure(out_path):
    importance, n_cfg = load_importance()

    display = importance.copy().astype(float)
    invalid = np.zeros_like(importance, dtype=bool)
    for i, ex in enumerate(EX_GRID):
        for j, em in enumerate(EM_GRID):
            if is_invalid(ex, em):
                invalid[i, j] = True
                display[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(11, 4))

    # Uniform gray background for all cutoff cells
    bg = np.ones((len(EX_GRID), len(EM_GRID), 3))
    gray_rgb = np.array(mcolors.to_rgb(CUTOFF_GRAY))
    for i in range(len(EX_GRID)):
        for j in range(len(EM_GRID)):
            if invalid[i, j]:
                bg[i, j] = gray_rgb
    ax.imshow(bg, origin='lower', aspect='auto',
              extent=[0, len(EM_GRID), 0, len(EX_GRID)],
              interpolation='nearest')

    # Inferno over valid cells; masked cells transparent so bg shows through
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    masked = np.ma.masked_invalid(display)
    vmax = max(0.01, np.nanmax(display))
    im = ax.pcolormesh(masked, cmap=cmap, vmin=0, vmax=vmax,
                       edgecolors='face', linewidth=0, rasterized=True)

    ax.set_xticks([i + 0.5 for i in range(len(EM_GRID))])
    ax.set_xticklabels([str(e) for e in EM_GRID], rotation=90, fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(len(EX_GRID))])
    ax.set_yticklabels([int(e) for e in EX_GRID], fontsize=11)
    ax.set_xlabel('Emission Wavelength (nm)', fontsize=13)
    ax.set_ylabel('Excitation (nm)', fontsize=13)
    ax.set_xlim(0, len(EM_GRID))
    ax.set_ylim(0, len(EX_GRID))

    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Mean Importance (rank-weighted)', fontsize=11)

    legend = [
        Patch(facecolor=CUTOFF_GRAY, edgecolor='black',
              label='Cutoff regions (physically invalid: Rayleigh or 2nd-order)')
    ]
    ax.legend(handles=legend, loc='upper center',
              bbox_to_anchor=(0.5, -0.25), ncol=1, fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path.name} ({out_path.stat().st_size/1024:.0f} KB)")


def replace_image_in_docx(docx_path, image_name, new_image_path):
    """Rewrite the zip swapping only one media file; keep everything else."""
    if not docx_path.exists():
        raise FileNotFoundError(f"docx not found: {docx_path}")

    new_bytes = new_image_path.read_bytes()

    backup = docx_path.with_suffix('.docx.bak')
    if not backup.exists():
        shutil.copy2(docx_path, backup)
        print(f"  Backup: {backup.name}")

    tmp = docx_path.with_suffix('.docx.tmp')
    with zipfile.ZipFile(docx_path, 'r') as zin:
        names = zin.namelist()
        if image_name not in names:
            raise KeyError(
                f"{image_name} not found in {docx_path.name}; "
                f"available media: {[n for n in names if 'media' in n]}"
            )
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                if name == image_name:
                    zout.writestr(name, new_bytes)
                else:
                    zout.writestr(name, zin.read(name))

    tmp.replace(docx_path)
    print(f"  Replaced {image_name} in {docx_path.name}")
    print(f"    Text content untouched; only the image changed.")


def main():
    print("=" * 86)
    print("Pepsin — In-Place Figure Fix (preserves user text edits)")
    print("=" * 86)

    for p in (ABSTRACT_DOCX, SUMMARY_DOCX):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    tmp = OUT_DIR / "_figure_swaps"
    tmp.mkdir(exist_ok=True)

    print("\n[1/4] Building new abstract figure (peak label moved)...")
    new_abs = tmp / "iasim_figure_fixed.png"
    build_abstract_figure(new_abs)

    print("\n[2/4] Building new heatmap (uniform gray)...")
    new_heat = tmp / "wavelength_heatmap_uniform_gray.png"
    build_heatmap_figure(new_heat)

    print("\n[3/4] Swapping figure in abstract docx...")
    replace_image_in_docx(ABSTRACT_DOCX, ABSTRACT_IMAGE_NAME, new_abs)

    print("\n[4/4] Swapping heatmap in summary docx...")
    replace_image_in_docx(SUMMARY_DOCX, SUMMARY_HEATMAP_NAME, new_heat)

    # Refresh the standalone PNGs so downstream consumers see the fix too
    (OUT_DIR / "iasim_figure.png").write_bytes(new_abs.read_bytes())
    (OUT_DIR / "wavelength_heatmap_v2.png").write_bytes(new_heat.read_bytes())

    print("\n" + "=" * 86)
    print("DONE — both documents updated in place. Backups at *.docx.bak")
    print(f"  Abstract: {ABSTRACT_DOCX}")
    print(f"  Summary:  {SUMMARY_DOCX}")
    print("=" * 86)


if __name__ == '__main__':
    main()
