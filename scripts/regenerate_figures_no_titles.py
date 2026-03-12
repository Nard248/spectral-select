#!/usr/bin/env python3
"""
Regenerate paper figures WITHOUT embedded titles.
=================================================
Titles should come from LaTeX captions only, matching the clean style
of Figures 3 (LichensRGB) and 5 (training curves via TikZ).

This script regenerates:
  1. robustness_histogram.png  (Figure 8)
  2. accuracy_envelope.png     (Figure 6)
  3. wavelength_heatmap.png    (Figure 9)
  4. roi_overlay.png           (Figure 4)
  5. classification_*.png      (Figure 7 panels)

Output: Paper Source/paper/figures-updated/
"""

import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Paper Source" / "paper" / "figures-updated"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_DIR = PROJECT_ROOT / "results" / "Lichens_Dataset_1_MasterRun"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
ROBUSTNESS_CSV = (
    PROJECT_ROOT / "archive" / "wavelength_analysis" / "Results"
    / "robustness" / "robustness_13bands_results.csv"
)
PAPER_FIGURES_DIR = RESULTS_DIR / "paper_figures"

BASELINE_ACCURACY = 0.8815
LEARNED_ACCURACY = 0.9016  # 13-band learned selection

CLASS_COLORS = {
    1: '#FF0000',   # Class 1 = Red
    3: '#0000FF',   # Class 2 = Blue
    6: '#00C800',   # Class 3 = Green
    7: '#FFA500',   # Class 4 = Orange/Yellow
}
CLASS_PAPER_NAMES = {1: 'Class 1', 3: 'Class 2', 6: 'Class 3', 7: 'Class 4'}

generated = []


# ═════════════════════════════════════════════════════════════════════════════
# 1. ROBUSTNESS HISTOGRAM (Figure 8)
# ═════════════════════════════════════════════════════════════════════════════

def regenerate_robustness_histogram():
    """Robustness histogram without embedded title."""
    print("\n[1/5] Robustness histogram...")

    if not ROBUSTNESS_CSV.exists():
        print(f"  SKIP: {ROBUSTNESS_CSV} not found")
        return

    df = pd.read_csv(ROBUSTNESS_CSV)
    accuracies = df['accuracy'].values

    mean_acc = accuracies.mean()
    median_acc = np.median(accuracies)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(accuracies, bins=50, color='steelblue', edgecolor='black',
            alpha=0.7, label='Random selections', zorder=2)

    ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_acc:.1%}', zorder=3)
    ax.axvline(median_acc, color='darkorange', linestyle='--', linewidth=2,
               label=f'Median: {median_acc:.1%}', zorder=3)
    ax.axvline(LEARNED_ACCURACY, color='purple', linestyle='-', linewidth=2.5,
               label=f'Learned: {LEARNED_ACCURACY:.1%}', zorder=4)

    ax.annotate(f'{LEARNED_ACCURACY:.1%}',
                xy=(LEARNED_ACCURACY, ax.get_ylim()[1] * 0.05),
                xytext=(LEARNED_ACCURACY - 0.08, ax.get_ylim()[1] * 0.4),
                fontsize=14, fontweight='bold', color='purple',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    ax.set_xlabel('Classification Accuracy', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    # NO title
    ax.legend(fontsize=12, loc='upper center')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = OUTPUT_DIR / "robustness_histogram.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    generated.append(out.name)
    print(f"  Saved: {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. ACCURACY ENVELOPE (Figure 6)
# ═════════════════════════════════════════════════════════════════════════════

def regenerate_accuracy_envelope():
    """Accuracy envelope without embedded title."""
    print("\n[2/5] Accuracy envelope...")

    csv_path = RESULTS_DIR / "results.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df = df[df['config'] != 'BASELINE']

    stats_df = df.groupby('n_bands_to_select').agg({
        'accuracy': ['min', 'max', 'mean', 'std']
    }).reset_index()
    stats_df.columns = ['n_bands', 'min', 'max', 'mean', 'std']

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(stats_df['n_bands'], stats_df['min'], stats_df['max'],
                    alpha=0.25, color='#4A90D9', label='Range (min-max)')
    ax.plot(stats_df['n_bands'], stats_df['mean'],
            color='black', linewidth=2, label='Mean')
    ax.plot(stats_df['n_bands'], stats_df['max'],
            color='#1B7A2B', linewidth=2.5, linestyle='--', label='Best')
    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--',
               linewidth=1.5, label=f'Baseline ({BASELINE_ACCURACY:.2%})')

    # Mark peak
    best_idx = stats_df['max'].idxmax()
    best_n = stats_df.loc[best_idx, 'n_bands']
    best_acc = stats_df.loc[best_idx, 'max']

    ax.plot([best_n, best_n], [0.20, best_acc],
            color='#1B7A2B', linestyle=':', linewidth=1.5, zorder=4)
    ax.scatter([best_n], [best_acc], color='gold', s=250, zorder=6,
               marker='*', edgecolor='#1B7A2B', linewidth=1.5)

    mid_y = (0.20 + best_acc) / 2
    ax.text(best_n + 5, mid_y, f"Peak: {best_acc:.2%}\nn = {int(best_n)}",
            fontsize=14, fontweight='bold', color='#1B7A2B', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#1B7A2B', alpha=0.9))

    ax.set_xlabel('Number of Bands Selected', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    # NO title
    ax.legend(loc='lower right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(0, 185)
    ax.set_ylim(0.20, 0.98)

    plt.tight_layout()
    out = OUTPUT_DIR / "accuracy_envelope.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    generated.append(out.name)
    print(f"  Saved: {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. WAVELENGTH HEATMAP (Figure 9)
# ═════════════════════════════════════════════════════════════════════════════

def load_wavelength_data() -> Dict:
    """Load wavelength JSON files from all experiment directories."""
    wavelengths_data = {}
    if not EXPERIMENTS_DIR.exists():
        return wavelengths_data

    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        wl_file = exp_dir / "wavelengths.json"
        if wl_file.exists():
            with open(wl_file, 'r') as f:
                wavelengths_data[exp_dir.name] = json.load(f)
    return wavelengths_data


def regenerate_wavelength_heatmap():
    """Wavelength importance heatmap without embedded title."""
    print("\n[3/5] Wavelength heatmap...")

    csv_path = RESULTS_DIR / "results.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df = df[df['config'] != 'BASELINE']

    wavelengths_data = load_wavelength_data()
    if not wavelengths_data:
        print("  SKIP: No wavelength data found")
        return

    def extract_group_key(config_name):
        match = re.match(r'^bands_\d+_(.+)$', config_name)
        return match.group(1) if match else None

    pca_df = df[df['dimension_selection_method'] == 'pca']
    above_baseline = pca_df[pca_df['accuracy'] > BASELINE_ACCURACY]
    if len(above_baseline) == 0:
        above_baseline = pca_df

    above_baseline_groups = set()
    for _, row in above_baseline.iterrows():
        gk = extract_group_key(row['config'])
        if gk:
            above_baseline_groups.add(gk)

    print(f"  Found {len(above_baseline_groups)} above-baseline config groups")

    group_rankings = {}
    for group_key in above_baseline_groups:
        group_configs = {}
        for config_name, wl_data in wavelengths_data.items():
            gk = extract_group_key(config_name)
            if gk == group_key:
                match = re.match(r'^bands_(\d+)_', config_name)
                if match:
                    n_bands = int(match.group(1))
                    group_configs[n_bands] = (config_name, wl_data)
        if group_configs:
            max_bands = max(group_configs.keys())
            _, wl_data = group_configs[max_bands]
            group_rankings[group_key] = wl_data

    emission_grid = list(range(420, 730, 10))
    excitation_grid = [310, 325, 340, 365, 385, 400, 415, 430]

    def is_valid_pair(ex, em):
        if em < ex + 40:
            return False
        if abs(em - 2 * ex) < 40:
            return False
        return True

    valid_mask = np.zeros((len(excitation_grid), len(emission_grid)), dtype=bool)
    for i, ex in enumerate(excitation_grid):
        for j, em in enumerate(emission_grid):
            valid_mask[i, j] = is_valid_pair(ex, em)

    importance_matrix = np.zeros((len(excitation_grid), len(emission_grid)))
    count_matrix = np.zeros((len(excitation_grid), len(emission_grid)))

    for group_key, wavelengths in group_rankings.items():
        max_rank = len(wavelengths)
        if max_rank == 0:
            continue
        for wl in wavelengths:
            ex = wl['excitation']
            em = wl['emission']
            if ex in excitation_grid:
                i = excitation_grid.index(ex)
            else:
                continue
            em_rounded = int(em // 10) * 10
            if em_rounded in emission_grid:
                j = emission_grid.index(em_rounded)
            else:
                continue
            importance = 1.0 - (wl['rank'] - 1) / max_rank
            importance_matrix[i, j] += importance
            count_matrix[i, j] += 1

    n_groups = len(group_rankings)
    mean_importance = importance_matrix / n_groups if n_groups > 0 else importance_matrix

    fig, ax = plt.subplots(figsize=(10, 4))

    display_matrix = mean_importance.copy().astype(float)
    display_matrix[~valid_mask] = np.nan

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color='#D3D3D3')

    masked_matrix = np.ma.masked_invalid(display_matrix)
    im = ax.pcolormesh(masked_matrix, cmap=cmap, vmin=0, vmax=1,
                       edgecolors='face', linewidth=0, rasterized=True)

    ax.set_xticks([i + 0.5 for i in range(len(emission_grid))])
    ax.set_xticklabels([str(e) for e in emission_grid], rotation=90, fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(len(excitation_grid))])
    ax.set_yticklabels([int(e) for e in excitation_grid], fontsize=12)
    ax.tick_params(which='minor', length=0)

    ax.set_xlabel('Emission Wavelength (nm)', fontsize=13)
    ax.set_ylabel('Excitation Wavelength (nm)', fontsize=13)
    # NO title

    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Mean Importance Score (0 = low, 1 = high)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    out = OUTPUT_DIR / "wavelength_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    generated.append(out.name)
    print(f"  Saved: {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. ROI OVERLAY (Figure 4) — from existing paper_figures data
# ═════════════════════════════════════════════════════════════════════════════

def regenerate_roi_overlay():
    """Re-render ROI overlay without embedded title.

    Uses the paper_metrics.json and the baseline classification map
    from the existing paper_figures directory.
    """
    print("\n[4/5] ROI overlay...")

    metrics_path = PAPER_FIGURES_DIR / "paper_metrics.json"
    if not metrics_path.exists():
        print(f"  SKIP: {metrics_path} not found (run rerun_knn_for_paper.py first)")
        return

    # Load the existing roi_overlay image and crop off the title
    # The title is at the top of the image
    existing = PAPER_FIGURES_DIR / "roi_overlay_baseline.png"
    if not existing.exists():
        print(f"  SKIP: {existing} not found")
        return

    from PIL import Image

    img = Image.open(existing)
    arr = np.array(img)

    # The title occupies roughly the top ~8-10% of the image.
    # Detect: find first row that isn't all-white (title area is white bg with text)
    # Look for the first row where less than 95% of pixels are white
    height = arr.shape[0]
    for row_idx in range(height):
        row = arr[row_idx]
        if row.ndim == 2:
            white_fraction = np.mean(row > 250)
        else:
            white_fraction = np.mean(np.all(row[:, :3] > 250, axis=1))
        if white_fraction < 0.90:
            break

    # Crop from slightly above the content start to capture the full image
    crop_start = max(0, row_idx - 5)
    cropped = arr[crop_start:]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cropped)
    ax.axis('off')

    plt.tight_layout(pad=0)
    out = OUTPUT_DIR / "roi_overlay.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.close()
    generated.append(out.name)
    print(f"  Saved: {out.name} (cropped from existing)")


# ═════════════════════════════════════════════════════════════════════════════
# 5. CLASSIFICATION MAPS (Figure 7 panels) — crop from existing
# ═════════════════════════════════════════════════════════════════════════════

def regenerate_classification_maps():
    """Re-render classification map panels without embedded titles."""
    print("\n[5/5] Classification maps...")

    from PIL import Image

    panels = [
        ("classification_baseline.png", "classification_192bands.png"),
        ("classification_best_80.png", "classification_80bands.png"),
        ("classification_efficient_9.png", "classification_9bands.png"),
    ]

    for src_name, dst_name in panels:
        src = PAPER_FIGURES_DIR / src_name
        if not src.exists():
            print(f"  SKIP: {src} not found")
            continue

        img = Image.open(src)
        arr = np.array(img)

        # Detect title area (white rows at top)
        height = arr.shape[0]
        for row_idx in range(height):
            row = arr[row_idx]
            if row.ndim == 2:
                white_fraction = np.mean(row > 250)
            else:
                white_fraction = np.mean(np.all(row[:, :3] > 250, axis=1))
            if white_fraction < 0.90:
                break

        crop_start = max(0, row_idx - 5)
        cropped = arr[crop_start:]

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(cropped)
        ax.axis('off')

        plt.tight_layout(pad=0)
        out = OUTPUT_DIR / dst_name
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
        plt.close()
        generated.append(dst_name)
        print(f"  Saved: {dst_name} (cropped from {src_name})")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("REGENERATE PAPER FIGURES (NO EMBEDDED TITLES)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    regenerate_robustness_histogram()
    regenerate_accuracy_envelope()
    regenerate_wavelength_heatmap()
    regenerate_roi_overlay()
    regenerate_classification_maps()

    print("\n" + "=" * 60)
    print(f"DONE — {len(generated)} figures generated:")
    for name in generated:
        print(f"  {name}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
