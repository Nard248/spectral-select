#!/usr/bin/env python3
"""
Re-run KNN Classification for Paper Figures
=============================================
Generates publication-ready classification maps and detailed metrics
for the best configurations from the master run.

Configs tested:
  1. Baseline: all 192 bands (88.15% accuracy)
  2. Best overall: 80 bands, PCA dim1 (95.23% accuracy)
  3. Most efficient >= baseline: 9 bands, PCA dim3 (89.43% accuracy)

Note: This script uses pickle to load spectral data files (.pkl) which is
required for the scientific hyperspectral data format used in this project.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
import time

warnings.filterwarnings('ignore')

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

np.random.seed(42)

from spectral_select import SpectraData
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score,
    classification_report, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score
)
from scipy import ndimage
from PIL import Image

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = PROJECT_ROOT / "Data" / "processed" / "Lichens Dataset 1"
RESULTS_DIR = PROJECT_ROOT / "results" / "Lichens_Dataset_1_MasterRun"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
OUTPUT_DIR = RESULTS_DIR / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configs to test
CONFIGS = {
    "baseline": {
        "name": "Baseline (all 192 bands)",
        "short_name": "192 bands",
        "wavelengths_file": None,  # Use all bands
        "n_bands": 192,
    },
    "best_80": {
        "name": "Best: PCA, 80 bands",
        "short_name": "80 bands",
        "wavelengths_file": EXPERIMENTS_DIR / "bands_80_pca_dim_1_perc_mag_medium_non" / "wavelengths.json",
        "n_bands": 80,
    },
    "efficient_9": {
        "name": "Most Efficient: PCA, 9 bands",
        "short_name": "9 bands",
        "wavelengths_file": EXPERIMENTS_DIR / "bands_9_pca_dim_3_abso_mag_medium_max" / "wavelengths.json",
        "n_bands": 9,
    },
}

# Class colors matching the paper style
CLASS_COLORS = {
    1: '#FF0000',   # Class 0 in paper = Class 1 in data = Red
    3: '#0000FF',   # Class 1 in paper = Class 3 in data = Blue
    6: '#00C800',   # Class 2 in paper = Class 6 in data = Green
    7: '#FFA500',   # Class 3 in paper = Class 7 in data = Orange/Yellow
}

CLASS_PAPER_NAMES = {1: 'Class 1', 3: 'Class 2', 6: 'Class 3', 7: 'Class 4'}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Dict[str, Any]:
    """Load the full hyperspectral dataset."""
    print("Loading hyperspectral data...")
    t0 = time.time()
    spectra_data = SpectraData.from_pickle(DATA_DIR / "spectra_masked.pkl")

    full_data = {
        'excitation_wavelengths': list(spectra_data.excitations.keys()),
        'data': {}
    }

    for ex_nm, ex_data in spectra_data.excitations.items():
        wavelengths = ex_data.emission_wavelengths
        if hasattr(wavelengths, 'tolist'):
            wavelengths = wavelengths.tolist()
        full_data['data'][str(ex_nm)] = {
            'cube': ex_data.cube,
            'wavelengths': wavelengths
        }

    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Excitations: {full_data['excitation_wavelengths']}")

    first_ex = str(full_data['excitation_wavelengths'][0])
    shape = full_data['data'][first_ex]['cube'].shape
    print(f"  Spatial: {shape[0]} x {shape[1]}")

    total_bands = sum(len(full_data['data'][str(ex)]['wavelengths'])
                      for ex in full_data['excitation_wavelengths'])
    print(f"  Total EEM pairs: {total_bands}")

    # Print emission bands per excitation
    for ex in full_data['excitation_wavelengths']:
        wl = full_data['data'][str(ex)]['wavelengths']
        print(f"    Ex {ex} nm: {len(wl)} emission bands ({min(wl):.0f}-{max(wl):.0f} nm)")

    return full_data


def load_ground_truth() -> Tuple[np.ndarray, List[dict], Dict]:
    """Load ground truth mask and ROI regions."""
    mask_path = DATA_DIR / "class_mask.png"
    roi_path = DATA_DIR / "roi_regions.json"

    # Load mask
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)

    # Load ROI data
    with open(roi_path, 'r') as f:
        roi_data = json.load(f)

    # Build ground truth
    ground_truth = np.full(mask_array.shape[:2], -1, dtype=int)
    class_info = {}

    for cls in roi_data['classes']:
        color = tuple(cls['color'])
        class_id = cls['id']
        class_info[class_id] = {'name': cls['name'], 'color': color}

        if mask_array.shape[-1] == 4:
            mask = np.all(mask_array[:, :, :3] == color, axis=2)
        else:
            mask = np.all(mask_array == color, axis=2)

        ground_truth[mask] = class_id
        pixel_count = np.sum(mask)
        print(f"  {CLASS_PAPER_NAMES.get(class_id, f'Class {class_id}')}: {pixel_count:,} pixels")

    total_labeled = np.sum(ground_truth >= 0)
    print(f"  Total labeled pixels: {total_labeled:,}")
    print(f"  Image dimensions: {ground_truth.shape}")

    # ROI regions
    roi_regions = []
    for roi in roi_data['regions']:
        rect = roi['rect']
        roi_regions.append({
            'name': roi['class_name'],
            'class_id': roi['class_id'],
            'coords': (rect['row_min'], rect['row_max'], rect['col_min'], rect['col_max'])
        })

    return ground_truth, roi_regions, class_info


def extract_wavelength_subset(full_data: dict, wavelength_combos: List[dict]) -> dict:
    """Extract data subset using selected wavelengths."""
    subset = {'data': {}, 'excitation_wavelengths': []}

    combos_by_ex = {}
    for c in wavelength_combos:
        ex = c['excitation']
        combos_by_ex.setdefault(ex, []).append(c['emission'])

    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        ex_data = full_data['data'][ex_str]
        wavelengths = np.array(ex_data['wavelengths'])

        matching = None
        for combo_ex, emissions in combos_by_ex.items():
            if abs(float(ex) - float(combo_ex)) < 1.0:
                matching = emissions
                break

        if matching is None:
            continue

        indices, wl_values = [], []
        for em in matching:
            distances = np.abs(wavelengths - float(em))
            idx = np.argmin(distances)
            if distances[idx] < 10 and idx not in indices:
                indices.append(idx)
                wl_values.append(wavelengths[idx])

        if indices:
            subset['data'][ex_str] = {
                'cube': ex_data['cube'][:, :, indices],
                'wavelengths': wl_values
            }
            subset['excitation_wavelengths'].append(ex)

    return subset


# =============================================================================
# KNN CLASSIFICATION
# =============================================================================

def run_knn_classification(
    data: dict,
    roi_regions: List[dict],
    ground_truth: np.ndarray,
    n_neighbors: int = 5
) -> Tuple[np.ndarray, dict]:
    """Run KNN classification on valid pixels."""

    first_ex = str(data['excitation_wavelengths'][0])
    height, width = data['data'][first_ex]['cube'].shape[:2]

    valid_mask = ground_truth >= 0
    valid_coords = np.argwhere(valid_mask)

    # Build features
    print("  Building feature matrix...")
    features = []
    for y, x in valid_coords:
        pixel = []
        for ex in data['excitation_wavelengths']:
            cube = data['data'][str(ex)]['cube']
            pixel.extend(cube[y, x, :])
        features.append(pixel)

    X_full = np.array(features)
    n_features = X_full.shape[1]

    if np.any(np.isnan(X_full)):
        X_full = np.nan_to_num(X_full, nan=0.0)

    # Training data from ROIs
    print("  Extracting training data from ROIs...")
    X_train, y_train = [], []
    for roi in roi_regions:
        row_min, row_max, col_min, col_max = roi['coords']
        for y in range(row_min, row_max):
            for x in range(col_min, col_max):
                if 0 <= y < height and 0 <= x < width and valid_mask[y, x]:
                    idx = np.where((valid_coords[:, 0] == y) & (valid_coords[:, 1] == x))[0]
                    if len(idx) > 0:
                        X_train.append(X_full[idx[0]])
                        y_train.append(roi['class_id'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  Training samples: {len(X_train)}, Features: {n_features}")

    # Train and predict
    print("  Training KNN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    predictions = knn.predict(X_full_scaled)

    # Reconstruct map
    cluster_map = np.full((height, width), -1, dtype=int)
    for i, (y, x) in enumerate(valid_coords):
        cluster_map[y, x] = predictions[i]

    # Compute all metrics
    y_true = ground_truth[valid_mask]
    y_pred = cluster_map[valid_mask]

    unique_classes = sorted(np.unique(y_true))

    metrics = {
        'n_features': n_features,
        'n_train_samples': len(X_train),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'ari': adjusted_rand_score(y_true, y_pred),
        'nmi': normalized_mutual_info_score(y_true, y_pred),
    }

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)

    for i, cls_id in enumerate(unique_classes):
        paper_name = CLASS_PAPER_NAMES.get(cls_id, f"Class {cls_id}")
        metrics[f'precision_{paper_name}'] = per_class_precision[i]
        metrics[f'recall_{paper_name}'] = per_class_recall[i]
        metrics[f'f1_{paper_name}'] = per_class_f1[i]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    metrics['confusion_matrix'] = cm

    # Classification report
    report = classification_report(y_true, y_pred, labels=unique_classes,
                                   target_names=[CLASS_PAPER_NAMES.get(c, f'C{c}') for c in unique_classes],
                                   output_dict=True)
    metrics['classification_report'] = report

    return cluster_map, metrics


# =============================================================================
# OBJECT-LEVEL ANALYSIS
# =============================================================================

def compute_object_metrics(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray
) -> List[dict]:
    """Compute per-object classification metrics."""
    unique_classes = sorted(set(np.unique(ground_truth)) - {-1})

    objects = []
    for cls_id in unique_classes:
        class_mask = ground_truth == cls_id
        labeled, n_components = ndimage.label(class_mask)

        for obj_idx in range(1, n_components + 1):
            obj_mask = labeled == obj_idx
            obj_size = np.sum(obj_mask)

            if obj_size < 100:
                continue

            obj_gt = ground_truth[obj_mask]
            obj_pred = cluster_map[obj_mask]
            valid = obj_pred >= 0

            if np.sum(valid) == 0:
                continue

            acc = accuracy_score(obj_gt[valid], obj_pred[valid])

            y_coords, x_coords = np.where(obj_mask)
            objects.append({
                'class_id': cls_id,
                'class_name': CLASS_PAPER_NAMES.get(cls_id, f'Class {cls_id}'),
                'object_id': len(objects) + 1,
                'n_pixels': int(obj_size),
                'accuracy': acc,
                'center_y': int(np.mean(y_coords)),
                'center_x': int(np.mean(x_coords)),
            })

    return objects


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_classification_map(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray,
    title: str,
    save_path: Path,
    show_accuracy: bool = True,
    accuracy: float = None
):
    """Create a publication-ready classification map."""
    unique_classes = sorted(set(np.unique(ground_truth)) - {-1})

    # Create colormap
    color_list = []
    for cls_id in unique_classes:
        hex_color = CLASS_COLORS.get(cls_id, '#808080')
        color_list.append(mcolors.hex2color(hex_color))

    cmap = mcolors.ListedColormap(color_list)

    # Map class IDs to sequential indices for coloring
    display_map = np.full_like(cluster_map, -1, dtype=float)
    for i, cls_id in enumerate(unique_classes):
        display_map[cluster_map == cls_id] = i

    display_map[cluster_map < 0] = np.nan

    # Sized for IEEE ~0.32\textwidth display (~2.24" wide)
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # Create masked array for proper NaN handling
    masked_display = np.ma.masked_invalid(display_map)

    ax.imshow(masked_display, cmap=cmap, vmin=0, vmax=len(unique_classes) - 1,
              interpolation='nearest')

    if show_accuracy and accuracy is not None:
        ax.set_title(f'{title}\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold', pad=8)
    else:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=8)

    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


def create_comparison_figure(
    results: dict,
    save_path: Path
):
    """Create a 3-panel comparison figure for the paper."""
    configs_to_show = ['baseline', 'best_80', 'efficient_9']

    # Sized for IEEE full-width display (~7" wide)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    unique_classes = sorted(set(np.unique(results['baseline']['ground_truth'])) - {-1})

    color_list = []
    for cls_id in unique_classes:
        hex_color = CLASS_COLORS.get(cls_id, '#808080')
        color_list.append(mcolors.hex2color(hex_color))
    cmap = mcolors.ListedColormap(color_list)

    titles_for_panel = [
        "Baseline (all 192 bands)",
        "Best: PCA, 80 bands",
        "Most Efficient: PCA, 9 bands",
    ]
    labels_for_panel = [
        "(a) Baseline: 192 bands",
        "(b) 80 bands (58% reduction)",
        "(c) 9 bands (95% reduction)",
    ]

    for idx, (config_key, title, label) in enumerate(zip(configs_to_show, titles_for_panel, labels_for_panel)):
        ax = axes[idx]
        r = results[config_key]
        cluster_map = r['cluster_map']
        acc = r['metrics']['accuracy']

        display_map = np.full_like(cluster_map, -1, dtype=float)
        for i, cls_id in enumerate(unique_classes):
            display_map[cluster_map == cls_id] = i
        display_map[cluster_map < 0] = np.nan

        masked_display = np.ma.masked_invalid(display_map)
        ax.imshow(masked_display, cmap=cmap, vmin=0, vmax=len(unique_classes) - 1,
                  interpolation='nearest')
        # Use consistent 2-line title for equal heights
        ax.set_title(f'{title}\nAccuracy: {acc:.2%}', fontsize=13, fontweight='bold')
        ax.set_xlabel(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(w_pad=1.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved comparison figure: {save_path.name}")


def create_roi_overlay(
    cluster_map: np.ndarray,
    roi_regions: List[dict],
    object_metrics: List[dict],
    overall_accuracy: float,
    save_path: Path,
    title: str = None
):
    """Create ROI overlay with per-object accuracy labels."""
    unique_classes = sorted(set(np.unique(cluster_map)) - {-1})

    color_list = [mcolors.hex2color(CLASS_COLORS.get(c, '#808080')) for c in unique_classes]
    cmap = mcolors.ListedColormap(color_list)

    display_map = np.full_like(cluster_map, -1, dtype=float)
    for i, cls_id in enumerate(unique_classes):
        display_map[cluster_map == cls_id] = i
    display_map[cluster_map < 0] = np.nan

    # Sized for IEEE single-column (~3.35" display width)
    fig, ax = plt.subplots(figsize=(7, 6))
    masked_display = np.ma.masked_invalid(display_map)
    ax.imshow(masked_display, cmap=cmap, vmin=0, vmax=len(unique_classes) - 1,
              interpolation='nearest')

    # Build set of ROI bounding boxes for overlap detection
    roi_boxes = []
    for roi in roi_regions:
        row_min, row_max, col_min, col_max = roi['coords']
        roi_boxes.append((row_min, row_max, col_min, col_max))

        rect = plt.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min,
                              fill=False, edgecolor='white', linewidth=2, linestyle='-')
        ax.add_patch(rect)

    # Per-object accuracy labels — offset when overlapping an ROI
    for obj in object_metrics:
        cx, cy = obj['center_x'], obj['center_y']

        # Check if object center is inside any ROI
        in_roi = False
        for row_min, row_max, col_min, col_max in roi_boxes:
            if row_min <= cy <= row_max and col_min <= cx <= col_max:
                in_roi = True
                break

        color = '#2ecc71' if obj['accuracy'] > 0.8 else '#f39c12' if obj['accuracy'] > 0.5 else '#e74c3c'

        if in_roi:
            # Place label below the ROI to keep the ROI rectangle visible
            label_y = row_max + 18
            ax.text(cx, label_y, f"{obj['accuracy']:.0%}",
                    fontsize=12, ha='center', va='top', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.85))
        else:
            ax.text(cx, cy, f"{obj['accuracy']:.0%}",
                    fontsize=14, ha='center', va='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.85))

    if title:
        ax.set_title(f'{title}\nOverall Accuracy: {overall_accuracy:.2%}',
                     fontsize=16, fontweight='bold')
    ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=mcolors.hex2color(CLASS_COLORS[c]),
                             label=CLASS_PAPER_NAMES.get(c, f'Class {c}'))
                       for c in unique_classes]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved ROI overlay: {save_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("KNN RE-RUN FOR PAPER FIGURES")
    print("=" * 70)

    # Load data
    full_data = load_data()
    print("\nLoading ground truth...")
    ground_truth, roi_regions, class_info = load_ground_truth()

    # Print dataset info for paper
    print("\n" + "=" * 70)
    print("DATASET INFORMATION (for paper Table I)")
    print("=" * 70)
    unique_classes = sorted(set(np.unique(ground_truth)) - {-1})
    total_labeled = np.sum(ground_truth >= 0)
    for cls_id in unique_classes:
        count = np.sum(ground_truth == cls_id)
        print(f"  {CLASS_PAPER_NAMES[cls_id]}: {count:,} pixels ({count/total_labeled*100:.1f}%)")
    print(f"  Total labeled: {total_labeled:,}")

    # Run KNN for each config
    results = {}

    for config_key, config in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"CONFIG: {config['name']}")
        print(f"{'='*70}")

        # Select data subset
        if config['wavelengths_file'] is None:
            data = full_data
            print("  Using all bands (baseline)")
        else:
            with open(config['wavelengths_file'], 'r') as f:
                wavelength_combos = json.load(f)
            data = extract_wavelength_subset(full_data, wavelength_combos)
            n_bands = sum(len(data['data'][str(ex)]['wavelengths'])
                          for ex in data['excitation_wavelengths'])
            print(f"  Selected {n_bands} bands from {len(data['excitation_wavelengths'])} excitations")

        # Run KNN
        t0 = time.time()
        cluster_map, metrics = run_knn_classification(data, roi_regions, ground_truth)
        elapsed = time.time() - t0
        print(f"  Classification time: {elapsed:.1f}s")

        # Object metrics
        object_metrics = compute_object_metrics(cluster_map, ground_truth)

        results[config_key] = {
            'cluster_map': cluster_map,
            'metrics': metrics,
            'object_metrics': object_metrics,
            'ground_truth': ground_truth,
            'config': config,
        }

        # Print metrics
        print(f"\n  METRICS:")
        print(f"    Accuracy:          {metrics['accuracy']:.4f}")
        print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"    Precision (wtd):   {metrics['precision']:.4f}")
        print(f"    Recall (wtd):      {metrics['recall']:.4f}")
        print(f"    F1 (wtd):          {metrics['f1']:.4f}")
        print(f"    Cohen's Kappa:     {metrics['kappa']:.4f}")
        print(f"    MCC:               {metrics['mcc']:.4f}")
        print(f"    ARI:               {metrics['ari']:.4f}")
        print(f"    NMI:               {metrics['nmi']:.4f}")

        # Per-class
        print(f"\n  PER-CLASS METRICS:")
        for cls_id in unique_classes:
            paper_name = CLASS_PAPER_NAMES[cls_id]
            p = metrics.get(f'precision_{paper_name}', 0)
            r = metrics.get(f'recall_{paper_name}', 0)
            f = metrics.get(f'f1_{paper_name}', 0)
            print(f"    {paper_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")

        # Confusion matrix
        print(f"\n  CONFUSION MATRIX:")
        cm = metrics['confusion_matrix']
        header = "     " + " ".join([f"{CLASS_PAPER_NAMES[c]:>7s}" for c in unique_classes])
        print(f"    {header}")
        for i, cls_id in enumerate(unique_classes):
            row = " ".join([f"{cm[i,j]:>7d}" for j in range(len(unique_classes))])
            print(f"    {CLASS_PAPER_NAMES[cls_id]:>5s} {row}")

        # Object metrics summary
        if object_metrics:
            accs = [o['accuracy'] for o in object_metrics]
            print(f"\n  OBJECT-LEVEL SUMMARY ({len(object_metrics)} objects):")
            print(f"    Mean accuracy:  {np.mean(accs):.3f}")
            print(f"    Std deviation:  {np.std(accs):.3f}")
            print(f"    Min:            {min(accs):.3f}")
            print(f"    Max:            {max(accs):.3f}")

            # Per-object detail
            for obj in sorted(object_metrics, key=lambda x: x['object_id']):
                print(f"    Object {obj['object_id']:2d} ({obj['class_name']:>7s}): "
                      f"acc={obj['accuracy']:.3f}, pixels={obj['n_pixels']:,}")

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    # Individual classification maps
    for config_key, r in results.items():
        create_classification_map(
            r['cluster_map'], r['ground_truth'],
            r['config']['name'],
            OUTPUT_DIR / f"classification_{config_key}.png",
            accuracy=r['metrics']['accuracy']
        )

    # 3-panel comparison (main paper figure)
    create_comparison_figure(
        results,
        OUTPUT_DIR / "classification_comparison_3panel.png"
    )
    # Also save as PDF
    create_comparison_figure(
        results,
        OUTPUT_DIR / "classification_comparison_3panel.pdf"
    )

    # ROI overlays
    for config_key, r in results.items():
        create_roi_overlay(
            r['cluster_map'], roi_regions, r['object_metrics'],
            r['metrics']['accuracy'],
            OUTPUT_DIR / f"roi_overlay_{config_key}.png",
            title=r['config']['name']
        )

    # =========================================================================
    # SAVE METRICS TO JSON
    # =========================================================================
    print(f"\n{'='*70}")
    print("SAVING METRICS")
    print(f"{'='*70}")

    metrics_output = {}
    for config_key, r in results.items():
        m = {k: v for k, v in r['metrics'].items()
             if k not in ('confusion_matrix', 'classification_report')}
        m['confusion_matrix'] = r['metrics']['confusion_matrix'].tolist()
        m['object_metrics'] = r['object_metrics']
        m['config_name'] = r['config']['name']
        m['n_bands'] = r['config']['n_bands']
        metrics_output[config_key] = m

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_DIR / "paper_metrics.json", 'w') as f:
        json.dump(metrics_output, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved metrics to: {OUTPUT_DIR / 'paper_metrics.json'}")

    # Object comparison table
    print(f"\n{'='*70}")
    print("OBJECT-LEVEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Obj':>4s} {'Class':>8s} {'Baseline':>10s} {'80 bands':>10s} {'9 bands':>10s} {'Improvement':>12s}")
    print("-" * 60)

    baseline_objs = {o['object_id']: o for o in results['baseline']['object_metrics']}
    best_objs = {o['object_id']: o for o in results['best_80']['object_metrics']}
    eff_objs = {o['object_id']: o for o in results['efficient_9']['object_metrics']}

    for obj_id in sorted(baseline_objs.keys()):
        b = baseline_objs.get(obj_id, {})
        e80 = best_objs.get(obj_id, {})
        e9 = eff_objs.get(obj_id, {})

        b_acc = b.get('accuracy', 0)
        e80_acc = e80.get('accuracy', 0)
        e9_acc = e9.get('accuracy', 0)
        improvement = (e80_acc - b_acc) / b_acc * 100 if b_acc > 0 else 0

        print(f"{obj_id:>4d} {b.get('class_name', '?'):>8s} "
              f"{b_acc:>10.3f} {e80_acc:>10.3f} {e9_acc:>10.3f} "
              f"{improvement:>+11.1f}%")

    print(f"\n{'='*70}")
    print("DONE - All figures saved to:", OUTPUT_DIR)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
