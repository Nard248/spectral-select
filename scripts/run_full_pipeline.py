#!/usr/bin/env python3
"""
Full Pipeline for 4D Multi-Excitation Hyperspectral Wavelength Selection and Classification
============================================================================================

This script runs a complete pipeline similar to wavelengthSelectionV2-2 using the new
spectral_select module. It performs:
1. Data loading from pickle files or im3 cubes
2. Ground truth extraction from PNG mask with class colors
3. Multiple wavelength selection configurations
4. KNN classification using ROI regions as training data
5. Validation against ground truth
6. Comprehensive visualization and metrics export

Usage:
    python run_full_pipeline.py --data /path/to/data.pkl --mask /path/to/mask.png
    python run_full_pipeline.py --data /path/to/data_folder --mask /path/to/mask.png
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Import spectral_select module
from spectral_select import Config, Analyzer, SpectraData, Visualizer

# Import sklearn for KNN classification
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


# ============================================================================
# Configuration Generation
# ============================================================================

def generate_configurations(
    n_bands_list: List[int] = None,
    diversity_methods: List[str] = None,
    lambda_values: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Generate a list of configurations for wavelength selection experiments.

    Args:
        n_bands_list: List of number of bands to select
        diversity_methods: List of diversity methods to try
        lambda_values: List of lambda values for diversity weighting

    Returns:
        List of configuration dictionaries
    """
    if n_bands_list is None:
        n_bands_list = [10, 15, 20, 25, 30, 40, 50]

    if diversity_methods is None:
        diversity_methods = ['none', 'mmr', 'dpp', 'spectral_spread']

    if lambda_values is None:
        lambda_values = [0.3, 0.5, 0.7]

    configs = []

    for n_bands in n_bands_list:
        # No diversity constraint
        configs.append({
            'name': f'bands_{n_bands}_no_diversity',
            'n_bands_to_select': n_bands,
            'use_diversity_constraint': False,
            'diversity_method': 'mmr',
            'lambda_diversity': 0.5
        })

        # With diversity constraints
        for method in diversity_methods:
            if method == 'none':
                continue
            for lambda_val in lambda_values:
                configs.append({
                    'name': f'bands_{n_bands}_{method}_lambda_{lambda_val:.1f}',
                    'n_bands_to_select': n_bands,
                    'use_diversity_constraint': True,
                    'diversity_method': method,
                    'lambda_diversity': lambda_val
                })

    return configs


# ============================================================================
# Data Loading
# ============================================================================

def load_hyperspectral_data(data_path: Path) -> Dict[str, Any]:
    """
    Load hyperspectral data from pickle file or directory.

    Supports:
    - Direct .pkl file path
    - Directory containing .pkl files (uses first one found)
    - Directory containing .im3 files (raw data)

    Args:
        data_path: Path to pickle file or directory

    Returns:
        Data dictionary with excitation wavelengths and cubes
    """
    data_path = Path(data_path)

    # Case 1: Direct pickle file
    if data_path.suffix == '.pkl':
        print(f"Loading pickle file: {data_path}")
        spectra_data = SpectraData.from_pickle(data_path)
        return _spectra_data_to_dict(spectra_data)

    # Case 2: Directory
    elif data_path.is_dir():
        # Check for pickle files first
        pkl_files = list(data_path.glob("*.pkl"))
        im3_files = list(data_path.glob("*.im3"))

        if pkl_files:
            # Use the first pickle file (or one with 'data' in name if available)
            pkl_file = pkl_files[0]
            for pf in pkl_files:
                if 'data' in pf.stem.lower() and 'cutoff' in pf.stem.lower():
                    pkl_file = pf
                    break
            print(f"Loading pickle from directory: {pkl_file}")
            spectra_data = SpectraData.from_pickle(pkl_file)
            return _spectra_data_to_dict(spectra_data)

        elif im3_files:
            # Load raw .im3 files
            print(f"Loading raw .im3 files from: {data_path}")
            spectra_data = SpectraData.from_raw(data_path)
            return _spectra_data_to_dict(spectra_data)

        else:
            raise ValueError(
                f"No .pkl or .im3 files found in directory: {data_path}\n"
                f"Contents: {[f.name for f in data_path.iterdir()][:10]}"
            )

    else:
        raise ValueError(f"Path does not exist or unsupported format: {data_path}")


def _spectra_data_to_dict(spectra_data: SpectraData) -> Dict[str, Any]:
    """Convert SpectraData to V2-compatible dictionary format."""
    data = {
        'excitation_wavelengths': list(spectra_data.excitations.keys()),
        'data': {},
        'metadata': spectra_data.metadata or {}
    }

    for ex_nm, ex_data in spectra_data.excitations.items():
        wavelengths = ex_data.emission_wavelengths
        if hasattr(wavelengths, 'tolist'):
            wavelengths = wavelengths.tolist()
        data['data'][str(ex_nm)] = {
            'cube': ex_data.cube,
            'wavelengths': wavelengths
        }

    return data


def extract_ground_truth_from_png(
    png_path: Path,
    background_colors: List[Tuple[int, ...]] = None,
    target_shape: Tuple[int, int] = None,
    min_pixel_count: int = 100
) -> Tuple[np.ndarray, Dict, List[Tuple[int, ...]]]:
    """
    Extract ground truth labels from a PNG mask with colored regions.

    Args:
        png_path: Path to PNG mask file
        background_colors: List of RGBA tuples to treat as background
        target_shape: Optional target shape for resizing (height, width)
        min_pixel_count: Minimum pixels for a color to be considered a class
                        (helps filter out anti-aliasing artifacts)

    Returns:
        - ground_truth: 2D array with class labels (-1 for background)
        - color_mapping: Dictionary mapping colors to class indices
        - class_colors: List of unique class colors
    """
    from PIL import Image

    img = Image.open(png_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    img_array = np.array(img)

    if target_shape and img_array.shape[:2] != target_shape:
        img_resized = img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        img_array = np.array(img_resized)

    if background_colors is None:
        background_colors = [
            (0, 0, 0, 255),      # Black
            (255, 255, 255, 255), # White
            (24, 24, 24, 255),   # Dark gray
            (168, 168, 168, 255), # Light gray
            (0, 0, 0, 0),        # Transparent
        ]

    # Find unique colors with their counts
    pixels = img_array.reshape(-1, 4)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # Filter out background colors and colors with few pixels (anti-aliasing)
    class_colors = []
    for color, count in zip(unique_colors, counts):
        color_tuple = tuple(int(c) for c in color)  # Convert to Python int

        # Skip colors with too few pixels (likely anti-aliasing artifacts)
        if count < min_pixel_count:
            continue

        # Check if this is a background color (exact match with small tolerance)
        is_background = False
        for bg in background_colors:
            # Calculate RGB difference (ignore alpha for comparison)
            rgb_diff = sum(abs(color_tuple[i] - bg[i]) for i in range(3))
            if rgb_diff < 30:  # Total RGB difference < 30 means likely background
                is_background = True
                break

        if not is_background:
            class_colors.append(color_tuple)

    # Create ground truth map
    height, width = img_array.shape[:2]
    ground_truth = np.full((height, width), -1, dtype=int)
    color_mapping = {}

    for class_idx, color in enumerate(class_colors):
        mask = np.all(np.abs(img_array.astype(int) - np.array(color)) < 10, axis=2)
        ground_truth[mask] = class_idx
        color_mapping[color] = class_idx

    print(f"Ground truth extracted:")
    print(f"  Shape: {ground_truth.shape}")
    print(f"  Classes: {len(class_colors)}")
    for color, idx in color_mapping.items():
        count = np.sum(ground_truth == idx)
        print(f"    Class {idx}: {count:,} pixels, color={color[:3]}")

    return ground_truth, color_mapping, class_colors


# ============================================================================
# ROI Definition and KNN Classification
# ============================================================================

@dataclass
class ROIRegion:
    """Definition of a Region of Interest for training."""
    name: str
    coords: Tuple[int, int, int, int]  # (y_start, y_end, x_start, x_end)
    color: str
    class_id: int = -1  # Assigned during mapping to ground truth


def auto_detect_rois_from_ground_truth(
    ground_truth: np.ndarray,
    n_samples_per_class: int = 500,
    min_region_size: int = 50
) -> List[ROIRegion]:
    """
    Automatically detect ROI regions from ground truth mask.

    Args:
        ground_truth: 2D array with class labels
        n_samples_per_class: Target number of samples per class
        min_region_size: Minimum region size to consider

    Returns:
        List of ROIRegion objects
    """
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FF00FF', '#00FFFF',
              '#FFA500', '#800080', '#008000', '#FFC0CB']

    unique_classes = np.unique(ground_truth)
    unique_classes = unique_classes[unique_classes >= 0]  # Exclude background

    rois = []

    for class_id in unique_classes:
        class_mask = ground_truth == class_id

        # Find connected components
        labeled, n_components = ndimage.label(class_mask)

        # Find largest component
        component_sizes = ndimage.sum(class_mask, labeled, range(1, n_components + 1))
        if len(component_sizes) == 0:
            continue

        largest_component = np.argmax(component_sizes) + 1
        component_mask = labeled == largest_component

        # Get bounding box
        y_indices, x_indices = np.where(component_mask)
        if len(y_indices) == 0:
            continue

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Shrink slightly to ensure pure samples
        margin = 5
        y_min = min(y_min + margin, y_max - margin)
        y_max = max(y_max - margin, y_min + margin)
        x_min = min(x_min + margin, x_max - margin)
        x_max = max(x_max - margin, x_min + margin)

        roi = ROIRegion(
            name=f'Class_{class_id}',
            coords=(y_min, y_max, x_min, x_max),
            color=colors[int(class_id) % len(colors)],
            class_id=int(class_id)
        )
        rois.append(roi)

    print(f"Auto-detected {len(rois)} ROI regions from ground truth")
    for roi in rois:
        print(f"  {roi.name}: coords={roi.coords}, class_id={roi.class_id}")

    return rois


def run_knn_classification(
    data: Dict[str, Any],
    roi_regions: List[ROIRegion],
    ground_truth: np.ndarray = None,
    n_neighbors: int = 5
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run KNN classification using ROI regions as training data.

    Args:
        data: Hyperspectral data dictionary
        roi_regions: List of ROI regions for training
        ground_truth: Optional ground truth for metrics
        n_neighbors: Number of neighbors for KNN

    Returns:
        - cluster_map: 2D classification map
        - metrics: Dictionary of metrics
    """
    # Concatenate spectral data
    spectral_features = []
    coords = []

    # Get spatial dimensions from first excitation
    first_ex = str(data['excitation_wavelengths'][0])
    height, width = data['data'][first_ex]['cube'].shape[:2]

    # Build feature matrix
    for y in range(height):
        for x in range(width):
            pixel_features = []
            for ex in data['excitation_wavelengths']:
                ex_str = str(ex)
                cube = data['data'][ex_str]['cube']
                pixel_features.extend(cube[y, x, :])
            spectral_features.append(pixel_features)
            coords.append((y, x))

    X_full = np.array(spectral_features)
    coords = np.array(coords)

    # Extract training data from ROIs
    X_train_list = []
    y_train_list = []

    for roi in roi_regions:
        y_start, y_end, x_start, x_end = roi.coords

        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if 0 <= y < height and 0 <= x < width:
                    idx = y * width + x
                    X_train_list.append(X_full[idx])
                    y_train_list.append(roi.class_id)

    if len(X_train_list) == 0:
        raise ValueError("No training samples found in ROI regions")

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    print(f"  Training KNN with {len(X_train)} samples, {len(np.unique(y_train))} classes")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    # Predict
    predictions = knn.predict(X_full_scaled)

    # Reconstruct classification map
    cluster_map = np.full((height, width), -1, dtype=int)
    for i, (y, x) in enumerate(coords):
        cluster_map[y, x] = predictions[i]

    # Calculate metrics
    metrics = {
        'n_features': X_full.shape[1],
        'n_training_samples': len(X_train),
        'n_classes': len(np.unique(y_train))
    }

    if ground_truth is not None:
        # Mask valid pixels
        valid_mask = ground_truth >= 0
        y_true = ground_truth[valid_mask]
        y_pred = cluster_map[valid_mask]

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

    # Calculate clustering metrics on sample
    sample_size = min(10000, len(X_full_scaled))
    sample_indices = np.random.choice(len(X_full_scaled), sample_size, replace=False)
    X_sample = X_full_scaled[sample_indices]
    pred_sample = predictions[sample_indices]

    if len(np.unique(pred_sample)) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X_sample, pred_sample)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_sample, pred_sample)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_sample, pred_sample)
        except:
            pass

    return cluster_map, metrics


# ============================================================================
# Wavelength Selection Integration
# ============================================================================

def run_wavelength_selection(
    data_path: Path,
    mask_path: Path,
    sample_name: str,
    config_params: Dict[str, Any],
    output_dir: Path
) -> Tuple[List[Dict], 'Analyzer']:
    """
    Run wavelength selection using spectral_select module.

    Args:
        data_path: Path to data
        mask_path: Path to mask
        sample_name: Sample identifier
        config_params: Configuration parameters
        output_dir: Output directory

    Returns:
        - wavelength_combinations: List of selected wavelength combinations
        - analyzer: Fitted analyzer instance
    """
    # Create config
    config = Config(
        sample_name=sample_name,
        data_path=str(data_path),
        mask_path=str(mask_path),
        output_dir=str(output_dir),
        n_bands_to_select=config_params.get('n_bands_to_select', 30),
        use_diversity_constraint=config_params.get('use_diversity_constraint', False),
        diversity_method=config_params.get('diversity_method', 'mmr'),
        lambda_diversity=config_params.get('lambda_diversity', 0.5),
        save_visualizations=False,
        save_tiff_layers=False
    )

    # Run analysis
    analyzer = Analyzer(config)

    # Load data using same logic as load_hyperspectral_data
    data_path_obj = Path(config.data_path)
    if data_path_obj.suffix == '.pkl':
        data = SpectraData.from_pickle(data_path_obj)
    elif data_path_obj.is_dir():
        pkl_files = list(data_path_obj.glob("*.pkl"))
        im3_files = list(data_path_obj.glob("*.im3"))
        if pkl_files:
            pkl_file = pkl_files[0]
            for pf in pkl_files:
                if 'data' in pf.stem.lower() and 'cutoff' in pf.stem.lower():
                    pkl_file = pf
                    break
            data = SpectraData.from_pickle(pkl_file)
        elif im3_files:
            data = SpectraData.from_raw(data_path_obj)
        else:
            raise ValueError(f"No .pkl or .im3 files in {data_path_obj}")
    else:
        raise ValueError(f"Invalid data path: {data_path_obj}")

    if mask_path.exists():
        mask = np.load(mask_path) if mask_path.suffix == '.npy' else None
        if mask is not None:
            data.mask = mask

    # Fit
    analyzer.fit(data)

    # Get selected bands
    selected_bands = analyzer.get_selected_bands()

    # Convert to wavelength combinations format
    wavelength_combinations = []
    for band in selected_bands:
        combination = {
            'excitation': band.excitation_nm,
            'emission': band.emission_nm,
            'combination_name': f"Ex{band.excitation_nm:.0f}_Em{band.emission_nm:.1f}",
            'influence_score': band.influence_score,
            'rank': band.rank
        }
        wavelength_combinations.append(combination)

    return wavelength_combinations, analyzer


def extract_wavelength_subset(
    full_data: Dict[str, Any],
    wavelength_combinations: List[Dict],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Extract subset of data using selected excitation-emission wavelength combinations.

    Args:
        full_data: Full hyperspectral data dictionary
        wavelength_combinations: List of selected combinations
        verbose: Print information

    Returns:
        Subset data dictionary
    """
    subset_data = {
        'data': {},
        'metadata': full_data.get('metadata', {}),
        'excitation_wavelengths': [],
        'selected_combinations': wavelength_combinations
    }

    total_bands_original = 0
    total_bands_selected = 0

    # Group combinations by excitation wavelength
    combos_by_excitation = {}
    for combo in wavelength_combinations:
        ex = combo['excitation']
        em = combo['emission']
        if ex not in combos_by_excitation:
            combos_by_excitation[ex] = []
        combos_by_excitation[ex].append(em)

    # Process each excitation wavelength
    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        ex_data = full_data['data'][ex_str]

        original_wavelengths = np.array(ex_data['wavelengths'])
        original_cube = ex_data['cube']

        total_bands_original += len(original_wavelengths)

        # Check if this excitation has any selected combinations
        matching_emissions = None
        for combo_ex, emissions in combos_by_excitation.items():
            if abs(float(ex) - float(combo_ex)) < 1.0:
                matching_emissions = emissions
                break

        if matching_emissions is None:
            continue

        # Find indices of selected emission wavelengths
        selected_indices = []
        selected_wl_values = []

        for target_em in matching_emissions:
            target_em = float(target_em)
            distances = np.abs(original_wavelengths - target_em)
            closest_idx = np.argmin(distances)

            if distances[closest_idx] < 10 and closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
                selected_wl_values.append(original_wavelengths[closest_idx])

        if selected_indices:
            subset_cube = original_cube[:, :, selected_indices]
            total_bands_selected += len(selected_indices)

            subset_data['data'][ex_str] = {
                'cube': subset_cube,
                'wavelengths': selected_wl_values,
                **{k: v for k, v in ex_data.items() if k not in ['cube', 'wavelengths']}
            }

            if ex not in subset_data['excitation_wavelengths']:
                subset_data['excitation_wavelengths'].append(ex)

    if verbose:
        reduction_pct = (1 - total_bands_selected / total_bands_original) * 100 if total_bands_original > 0 else 0
        print(f"  Data reduction: {total_bands_original} -> {total_bands_selected} bands ({reduction_pct:.1f}% reduction)")

    return subset_data


# ============================================================================
# Visualization
# ============================================================================

def create_classification_visualization(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray,
    roi_regions: List[ROIRegion],
    metrics: Dict[str, float],
    config_name: str,
    output_dir: Path
):
    """
    Create visualization comparing classification to ground truth.

    Args:
        cluster_map: Classification result
        ground_truth: Ground truth labels
        roi_regions: ROI regions used for training
        metrics: Classification metrics
        config_name: Configuration name
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Classification map
    n_classes = len(roi_regions)
    colors = [roi.color for roi in roi_regions]
    cmap = mcolors.ListedColormap(colors)

    im1 = axes[0].imshow(cluster_map, cmap=cmap, vmin=0, vmax=n_classes-1)
    axes[0].set_title(f'Classification\nAccuracy: {metrics.get("accuracy", 0):.4f}')
    axes[0].axis('off')

    # Ground truth
    im2 = axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=n_classes-1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Difference map
    diff = (cluster_map != ground_truth).astype(int)
    diff[ground_truth < 0] = -1  # Mark background

    diff_cmap = mcolors.ListedColormap(['gray', 'white', 'red'])
    im3 = axes[2].imshow(diff, cmap=diff_cmap, vmin=-1, vmax=1)
    axes[2].set_title('Difference (Red = Errors)')
    axes[2].axis('off')

    # Add colorbar
    cbar = plt.colorbar(im1, ax=axes[:2], orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_ticks(range(n_classes))
    cbar.set_ticklabels([roi.name for roi in roi_regions])

    plt.suptitle(f'{config_name}\nF1: {metrics.get("f1_weighted", 0):.4f}, Kappa: {metrics.get("cohen_kappa", 0):.4f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / f'{config_name}_classification.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_plots(
    results_df: pd.DataFrame,
    output_dir: Path
):
    """
    Create summary visualization plots.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Output directory
    """
    # Accuracy vs number of bands
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Accuracy vs bands
    ax = axes[0, 0]
    ax.scatter(results_df['n_features'], results_df['accuracy'],
               c=results_df['accuracy'], cmap='viridis', s=100, edgecolors='black')
    ax.set_xlabel('Number of Spectral Features')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy vs. Feature Count')
    ax.grid(True, alpha=0.3)

    # Highlight best
    best_idx = results_df['accuracy'].idxmax()
    ax.scatter(results_df.loc[best_idx, 'n_features'],
               results_df.loc[best_idx, 'accuracy'],
               s=200, color='red', marker='*', zorder=5, label='Best')
    ax.legend()

    # 2. F1 vs bands
    ax = axes[0, 1]
    ax.scatter(results_df['n_features'], results_df['f1_weighted'],
               c=results_df['f1_weighted'], cmap='viridis', s=100, edgecolors='black')
    ax.set_xlabel('Number of Spectral Features')
    ax.set_ylabel('F1 Score (Weighted)')
    ax.set_title('F1 Score vs. Feature Count')
    ax.grid(True, alpha=0.3)

    # 3. Metrics comparison bar chart
    ax = axes[1, 0]
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

    # Get best and baseline
    best_row = results_df.loc[results_df['accuracy'].idxmax()]
    baseline_row = results_df[results_df['config_name'] == 'BASELINE_FULL_DATA']

    if len(baseline_row) > 0:
        baseline_row = baseline_row.iloc[0]
        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, [best_row[m] for m in metrics], width, label='Best Config', color='green')
        ax.bar(x + width/2, [baseline_row[m] for m in metrics], width, label='Baseline', color='blue')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel('Score')
        ax.set_title('Best Config vs. Baseline')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

    # 4. Data reduction vs accuracy
    ax = axes[1, 1]
    if 'data_reduction_pct' in results_df.columns:
        ax.scatter(results_df['data_reduction_pct'], results_df['accuracy'],
                   c=results_df['n_features'], cmap='plasma', s=100, edgecolors='black')
        ax.set_xlabel('Data Reduction (%)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs. Data Reduction')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='plasma',
                                   norm=plt.Normalize(results_df['n_features'].min(),
                                                     results_df['n_features'].max()))
        plt.colorbar(sm, ax=ax, label='# Features')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion matrix for best config
    print(f"\nBest configuration: {best_row['config_name']}")
    print(f"  Accuracy: {best_row['accuracy']:.4f}")
    print(f"  F1: {best_row['f1_weighted']:.4f}")
    print(f"  Features: {best_row['n_features']}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full wavelength selection and classification pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to data (pickle or directory)')
    parser.add_argument('--mask', type=str, required=True, help='Path to ground truth PNG mask')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--sample-name', type=str, default='sample', help='Sample name for output files')
    parser.add_argument('--max-configs', type=int, default=None, help='Maximum configurations to run')
    parser.add_argument('--n-bands', type=str, default='10,20,30,40', help='Comma-separated list of n_bands to test')

    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data)
    mask_path = Path(args.mask)
    sample_name = args.sample_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / 'results' / f'{sample_name}_pipeline_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FULL WAVELENGTH SELECTION PIPELINE")
    print("=" * 80)
    print(f"  Data: {data_path}")
    print(f"  Mask: {mask_path}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    full_data = load_hyperspectral_data(data_path)

    # Count bands
    total_bands = sum(
        len(full_data['data'][str(ex)]['wavelengths'])
        for ex in full_data['excitation_wavelengths']
    )
    print(f"  Excitations: {len(full_data['excitation_wavelengths'])}")
    print(f"  Total bands: {total_bands}")

    # Get spatial dimensions
    first_ex = str(full_data['excitation_wavelengths'][0])
    height, width = full_data['data'][first_ex]['cube'].shape[:2]
    print(f"  Spatial dimensions: {height} x {width}")

    # Extract ground truth
    print("\n" + "=" * 80)
    print("EXTRACTING GROUND TRUTH")
    print("=" * 80)

    ground_truth, color_mapping, class_colors = extract_ground_truth_from_png(
        mask_path,
        target_shape=(height, width)
    )

    n_classes = len(class_colors)

    # Auto-detect ROI regions
    print("\n" + "=" * 80)
    print("DETECTING ROI REGIONS")
    print("=" * 80)

    roi_regions = auto_detect_rois_from_ground_truth(ground_truth)

    # Create mask for wavelength selection (non-background pixels)
    analysis_mask = ground_truth >= 0
    mask_path_npy = output_dir / 'analysis_mask.npy'
    np.save(mask_path_npy, analysis_mask)

    # Generate configurations
    n_bands_list = [int(x) for x in args.n_bands.split(',')]
    configurations = generate_configurations(n_bands_list=n_bands_list)

    if args.max_configs:
        configurations = configurations[:args.max_configs]

    print(f"\n{len(configurations)} configurations to run")

    # Run baseline (full data)
    print("\n" + "=" * 80)
    print("BASELINE: Full Data Classification")
    print("=" * 80)

    cluster_map_baseline, metrics_baseline = run_knn_classification(
        full_data, roi_regions, ground_truth
    )

    # Create baseline visualization
    create_classification_visualization(
        cluster_map_baseline, ground_truth, roi_regions,
        metrics_baseline, 'BASELINE_FULL_DATA', output_dir
    )

    # Store results
    results = [{
        'config_name': 'BASELINE_FULL_DATA',
        'n_bands_selected': total_bands,
        'n_features': metrics_baseline['n_features'],
        'data_reduction_pct': 0.0,
        **{k: v for k, v in metrics_baseline.items() if isinstance(v, (int, float))}
    }]

    # Run configurations
    print("\n" + "=" * 80)
    print("RUNNING WAVELENGTH SELECTION CONFIGURATIONS")
    print("=" * 80)

    for i, config in enumerate(configurations):
        config_name = config['name']
        print(f"\n[{i+1}/{len(configurations)}] {config_name}")

        try:
            # Run wavelength selection
            wavelength_combinations, analyzer = run_wavelength_selection(
                data_path, mask_path_npy, sample_name, config,
                output_dir / 'wavelength_selection' / config_name
            )

            print(f"  Selected {len(wavelength_combinations)} wavelength combinations")

            # Extract subset
            subset_data = extract_wavelength_subset(full_data, wavelength_combinations)

            # Run classification
            cluster_map, metrics = run_knn_classification(
                subset_data, roi_regions, ground_truth
            )

            # Create visualization
            config_output_dir = output_dir / 'experiments' / config_name
            config_output_dir.mkdir(parents=True, exist_ok=True)

            create_classification_visualization(
                cluster_map, ground_truth, roi_regions,
                metrics, config_name, config_output_dir
            )

            # Store results
            data_reduction = (1 - metrics['n_features'] / metrics_baseline['n_features']) * 100
            results.append({
                'config_name': config_name,
                'n_bands_selected': len(wavelength_combinations),
                'n_features': metrics['n_features'],
                'data_reduction_pct': data_reduction,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'config_name': config_name,
                'error': str(e)
            })

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False, na_position='last')

    # Save to Excel
    excel_path = output_dir / 'pipeline_results.xlsx'
    results_df.to_excel(excel_path, index=False)
    print(f"  Results saved to: {excel_path}")

    # Save to CSV
    csv_path = output_dir / 'pipeline_results.csv'
    results_df.to_csv(csv_path, index=False)

    # Create summary plots
    create_summary_plots(results_df, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid_results = results_df[~results_df['accuracy'].isna()]
    if len(valid_results) > 0:
        best = valid_results.iloc[0]
        baseline = results_df[results_df['config_name'] == 'BASELINE_FULL_DATA'].iloc[0]

        print(f"\nBaseline (full data):")
        print(f"  Features: {baseline['n_features']}")
        print(f"  Accuracy: {baseline['accuracy']:.4f}")
        print(f"  F1: {baseline['f1_weighted']:.4f}")

        print(f"\nBest configuration: {best['config_name']}")
        print(f"  Features: {best['n_features']} ({best['data_reduction_pct']:.1f}% reduction)")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  F1: {best['f1_weighted']:.4f}")

        acc_change = (best['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
        print(f"\n  Accuracy change: {acc_change:+.2f}%")

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
