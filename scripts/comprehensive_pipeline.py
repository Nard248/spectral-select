#!/usr/bin/env python3
"""
Comprehensive Wavelength Selection Pipeline
============================================
Full pipeline with ALL V2-2 functionality using the new spectral_select module.

Features:
- Multiple wavelength selection configurations
- KNN classification using ROI regions
- Object-wise segmentation analysis
- Comprehensive metrics (accuracy, F1, precision, recall, kappa, purity, ARI, NMI)
- Paper-ready visualizations
- Detailed Excel exports with all configuration parameters
- Per-experiment folder structure with visualizations

Usage:
    python comprehensive_pipeline.py --data /path/to/data --mask /path/to/mask.png
    python comprehensive_pipeline.py --data /path/to/data.pkl --mask /path/to/mask.png --sample Lichens
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import warnings
import time

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

# Import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


# ============================================================================
# Timer Utilities
# ============================================================================

class PerformanceTimer:
    """Context manager for timing operations."""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ROIRegion:
    """Definition of a Region of Interest for training."""
    name: str
    coords: Tuple[int, int, int, int]  # (y_start, y_end, x_start, x_end)
    color: str
    class_id: int = -1


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    n_bands_to_select: int
    use_diversity_constraint: bool = False
    diversity_method: str = 'mmr'
    lambda_diversity: float = 0.5
    perturbation_method: str = 'percentile'
    perturbation_magnitudes: List[int] = field(default_factory=lambda: [10, 20, 30])
    n_important_dimensions: int = 15
    normalization_method: str = 'variance'
    min_distance_nm: float = 15.0


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config_name: str
    n_bands_selected: int
    n_features: int
    data_reduction_pct: float
    # Supervised metrics
    accuracy: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    cohen_kappa: float = 0.0
    # Clustering metrics
    purity: float = 0.0
    ari: float = 0.0
    nmi: float = 0.0
    # Timing
    selection_time: float = 0.0
    clustering_time: float = 0.0
    speedup_factor: float = 0.0
    # Object-wise metrics
    mean_object_accuracy: float = 0.0
    std_object_accuracy: float = 0.0
    # Config parameters
    use_diversity_constraint: bool = False
    diversity_method: str = 'none'
    lambda_diversity: float = 0.0
    # Error handling
    error: str = None


# ============================================================================
# Configuration Generation
# ============================================================================

def generate_configurations(
    n_bands_list: List[int] = None,
    diversity_methods: List[str] = None,
    lambda_values: List[float] = None,
    include_no_diversity: bool = True
) -> List[ExperimentConfig]:
    """Generate experiment configurations."""
    if n_bands_list is None:
        n_bands_list = [10, 15, 20, 25, 30, 40, 50]

    if diversity_methods is None:
        diversity_methods = ['mmr', 'dpp', 'spectral_spread']

    if lambda_values is None:
        lambda_values = [0.3, 0.5, 0.7]

    configs = []

    for n_bands in n_bands_list:
        # No diversity constraint
        if include_no_diversity:
            configs.append(ExperimentConfig(
                name=f'bands_{n_bands}_no_diversity',
                n_bands_to_select=n_bands,
                use_diversity_constraint=False,
                diversity_method='none',
                lambda_diversity=0.0
            ))

        # With diversity constraints
        for method in diversity_methods:
            for lambda_val in lambda_values:
                configs.append(ExperimentConfig(
                    name=f'bands_{n_bands}_{method}_lambda_{lambda_val:.1f}',
                    n_bands_to_select=n_bands,
                    use_diversity_constraint=True,
                    diversity_method=method,
                    lambda_diversity=lambda_val
                ))

    return configs


# ============================================================================
# Data Loading
# ============================================================================

def load_hyperspectral_data(data_path: Path) -> Tuple[Dict[str, Any], SpectraData]:
    """Load hyperspectral data and return both dict format and SpectraData."""
    data_path = Path(data_path)

    if data_path.suffix == '.pkl':
        print(f"Loading pickle file: {data_path}")
        spectra_data = SpectraData.from_pickle(data_path)
    elif data_path.is_dir():
        pkl_files = list(data_path.glob("*.pkl"))
        im3_files = list(data_path.glob("*.im3"))

        if pkl_files:
            pkl_file = pkl_files[0]
            for pf in pkl_files:
                if 'data' in pf.stem.lower() and 'cutoff' in pf.stem.lower():
                    pkl_file = pf
                    break
            print(f"Loading pickle from directory: {pkl_file}")
            spectra_data = SpectraData.from_pickle(pkl_file)
        elif im3_files:
            print(f"Loading raw .im3 files from: {data_path}")
            spectra_data = SpectraData.from_raw(data_path)
        else:
            raise ValueError(f"No .pkl or .im3 files found in: {data_path}")
    else:
        raise ValueError(f"Invalid path: {data_path}")

    # Convert to dict format for compatibility
    data_dict = _spectra_data_to_dict(spectra_data)

    return data_dict, spectra_data


def _spectra_data_to_dict(spectra_data: SpectraData) -> Dict[str, Any]:
    """Convert SpectraData to dictionary format."""
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
    """Extract ground truth labels from PNG mask."""
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
            (0, 0, 0, 255), (255, 255, 255, 255),
            (24, 24, 24, 255), (168, 168, 168, 255), (0, 0, 0, 0),
        ]

    pixels = img_array.reshape(-1, 4)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    class_colors = []
    for color, count in zip(unique_colors, counts):
        color_tuple = tuple(int(c) for c in color)

        if count < min_pixel_count:
            continue

        is_background = False
        for bg in background_colors:
            rgb_diff = sum(abs(color_tuple[i] - bg[i]) for i in range(3))
            if rgb_diff < 30:
                is_background = True
                break

        if not is_background:
            class_colors.append(color_tuple)

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
# ROI Detection
# ============================================================================

def auto_detect_rois_from_ground_truth(
    ground_truth: np.ndarray,
    class_colors: List[Tuple[int, ...]] = None
) -> List[ROIRegion]:
    """Automatically detect ROI regions from ground truth."""
    default_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FF00FF',
                      '#00FFFF', '#FFA500', '#800080', '#008000', '#FFC0CB']

    unique_classes = np.unique(ground_truth)
    unique_classes = unique_classes[unique_classes >= 0]

    rois = []

    for class_id in unique_classes:
        class_mask = ground_truth == class_id
        labeled, n_components = ndimage.label(class_mask)

        component_sizes = ndimage.sum(class_mask, labeled, range(1, n_components + 1))
        if len(component_sizes) == 0:
            continue

        largest_component = np.argmax(component_sizes) + 1
        component_mask = labeled == largest_component

        y_indices, x_indices = np.where(component_mask)
        if len(y_indices) == 0:
            continue

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        margin = 5
        y_min = min(y_min + margin, y_max - margin)
        y_max = max(y_max - margin, y_min + margin)
        x_min = min(x_min + margin, x_max - margin)
        x_max = max(x_max - margin, x_min + margin)

        # Use class color if available
        if class_colors and int(class_id) < len(class_colors):
            color = '#{:02x}{:02x}{:02x}'.format(*class_colors[int(class_id)][:3])
        else:
            color = default_colors[int(class_id) % len(default_colors)]

        roi = ROIRegion(
            name=f'Class_{class_id}',
            coords=(y_min, y_max, x_min, x_max),
            color=color,
            class_id=int(class_id)
        )
        rois.append(roi)

    print(f"Auto-detected {len(rois)} ROI regions from ground truth")
    for roi in rois:
        print(f"  {roi.name}: coords={roi.coords}, class_id={roi.class_id}")

    return rois


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate clustering purity."""
    contingency = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    for i, true_label in enumerate(np.unique(y_true)):
        for j, pred_label in enumerate(np.unique(y_pred)):
            contingency[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return np.sum(np.max(contingency, axis=0)) / len(y_true)


def calculate_all_metrics(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray = None
) -> Dict[str, float]:
    """Calculate all classification and clustering metrics."""
    if valid_mask is None:
        valid_mask = ground_truth >= 0

    y_true = ground_truth[valid_mask]
    y_pred = cluster_map[valid_mask]

    # Filter out invalid predictions
    valid_idx = y_pred >= 0
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]

    if len(y_true) == 0:
        return {k: 0.0 for k in ['accuracy', 'precision_weighted', 'recall_weighted',
                                  'f1_weighted', 'cohen_kappa', 'purity', 'ari', 'nmi']}

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'purity': calculate_purity(y_true, y_pred),
        'ari': adjusted_rand_score(y_true, y_pred),
        'nmi': normalized_mutual_info_score(y_true, y_pred)
    }

    return metrics


def calculate_object_metrics(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray,
    labeled_objects: np.ndarray,
    num_objects: int
) -> List[Dict[str, Any]]:
    """Calculate per-object metrics."""
    object_metrics = []

    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_objects == obj_id
        y_true = ground_truth[obj_mask]
        y_pred = cluster_map[obj_mask]

        if len(y_true) == 0 or np.all(y_true == -1):
            continue

        valid_idx = y_true != -1
        y_true_valid = y_true[valid_idx]
        y_pred_valid = y_pred[valid_idx]

        if len(y_true_valid) == 0:
            continue

        obj_accuracy = accuracy_score(y_true_valid, y_pred_valid)
        true_class = np.unique(y_true_valid)[0] if len(np.unique(y_true_valid)) == 1 else -1

        object_metrics.append({
            'object_id': obj_id,
            'num_pixels': int(np.sum(obj_mask)),
            'true_class': int(true_class),
            'accuracy': float(obj_accuracy)
        })

    return object_metrics


# ============================================================================
# KNN Classification
# ============================================================================

def run_knn_classification(
    data: Dict[str, Any],
    roi_regions: List[ROIRegion],
    ground_truth: np.ndarray = None,
    n_neighbors: int = 5,
    export_concat_data: bool = False,
    concat_data_path: Path = None
) -> Tuple[np.ndarray, Dict[str, float], int]:
    """Run KNN classification using ROI regions as training data."""

    # Get spatial dimensions
    first_ex = str(data['excitation_wavelengths'][0])
    height, width = data['data'][first_ex]['cube'].shape[:2]

    # Build feature matrix
    spectral_features = []
    coords = []

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
    n_features = X_full.shape[1]

    # Export concatenated data if requested
    if export_concat_data and concat_data_path:
        df_concat = pd.DataFrame(X_full)
        df_concat['y'] = [c[0] for c in coords]
        df_concat['x'] = [c[1] for c in coords]
        df_concat.to_csv(concat_data_path, index=False)

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

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    predictions = knn.predict(X_full_scaled)

    # Reconstruct classification map
    cluster_map = np.full((height, width), -1, dtype=int)
    for i, (y, x) in enumerate(coords):
        cluster_map[y, x] = predictions[i]

    # Calculate metrics
    metrics = {'n_features': n_features, 'n_training_samples': len(X_train)}

    if ground_truth is not None:
        metrics.update(calculate_all_metrics(cluster_map, ground_truth))

    return cluster_map, metrics, n_features


# ============================================================================
# Wavelength Selection
# ============================================================================

def run_wavelength_selection(
    data_path: Path,
    mask_path: Path,
    sample_name: str,
    config: ExperimentConfig,
    model_path: Path,
    output_dir: Path
) -> Tuple[List[Dict], Any]:
    """Run wavelength selection using spectral_select module."""

    # Create config for spectral_select
    ss_config = Config(
        sample_name=sample_name,
        data_path=str(data_path),
        mask_path=str(mask_path),
        model_path=str(model_path),
        output_dir=str(output_dir),
        n_bands_to_select=config.n_bands_to_select,
        use_diversity_constraint=config.use_diversity_constraint,
        diversity_method=config.diversity_method if config.use_diversity_constraint else 'mmr',
        lambda_diversity=config.lambda_diversity,
        perturbation_method=config.perturbation_method,
        perturbation_magnitudes=config.perturbation_magnitudes,
        n_important_dimensions=config.n_important_dimensions,
        normalization_method=config.normalization_method,
        min_distance_nm=config.min_distance_nm,
        save_visualizations=False,
        save_tiff_layers=False,
        n_baseline_patches=10
    )

    # Run analysis
    analyzer = Analyzer(ss_config)

    # Load data
    data_path_obj = Path(ss_config.data_path)
    if data_path_obj.suffix == '.pkl':
        spectra_data = SpectraData.from_pickle(data_path_obj)
    elif data_path_obj.is_dir():
        pkl_files = list(data_path_obj.glob("*.pkl"))
        if pkl_files:
            pkl_file = pkl_files[0]
            for pf in pkl_files:
                if 'data' in pf.stem.lower() and 'cutoff' in pf.stem.lower():
                    pkl_file = pf
                    break
            spectra_data = SpectraData.from_pickle(pkl_file)
        else:
            spectra_data = SpectraData.from_raw(data_path_obj)
    else:
        raise ValueError(f"Invalid data path: {data_path_obj}")

    # Load mask
    mask_path_obj = Path(ss_config.mask_path)
    if mask_path_obj.exists():
        if mask_path_obj.suffix == '.npy':
            spectra_data.mask = np.load(mask_path_obj)

    # Fit analyzer
    analyzer.fit(spectra_data)

    # Get result with selected bands
    result = analyzer.result
    if result is None:
        raise ValueError("Analyzer fit() did not produce a result")

    # Convert to wavelength combinations format
    wavelength_combinations = []
    for band in result.selected_bands:
        combination = {
            'excitation': float(band.excitation_nm),
            'emission': float(band.emission_nm),
            'combination_name': f"Ex{band.excitation_nm:.0f}_Em{band.emission_nm:.1f}",
            'influence_score': float(band.influence_score),
            'rank': int(band.rank)
        }
        wavelength_combinations.append(combination)

    return wavelength_combinations, analyzer


def extract_wavelength_subset(
    full_data: Dict[str, Any],
    wavelength_combinations: List[Dict],
    verbose: bool = True
) -> Dict[str, Any]:
    """Extract subset of data using selected wavelength combinations."""
    subset_data = {
        'data': {},
        'metadata': full_data.get('metadata', {}),
        'excitation_wavelengths': [],
        'selected_combinations': wavelength_combinations
    }

    total_bands_original = 0
    total_bands_selected = 0

    # Group by excitation
    combos_by_excitation = {}
    for combo in wavelength_combinations:
        ex = combo['excitation']
        em = combo['emission']
        if ex not in combos_by_excitation:
            combos_by_excitation[ex] = []
        combos_by_excitation[ex].append(em)

    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        ex_data = full_data['data'][ex_str]

        original_wavelengths = np.array(ex_data['wavelengths'])
        original_cube = ex_data['cube']
        total_bands_original += len(original_wavelengths)

        matching_emissions = None
        for combo_ex, emissions in combos_by_excitation.items():
            if abs(float(ex) - float(combo_ex)) < 1.0:
                matching_emissions = emissions
                break

        if matching_emissions is None:
            continue

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
# Visualization Functions
# ============================================================================

def create_roi_colormap(roi_regions: List[ROIRegion]):
    """Create colormap from ROI colors."""
    colors = [roi.color for roi in roi_regions]
    return mcolors.ListedColormap(colors)


def plot_classification_comparison(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray,
    roi_regions: List[ROIRegion],
    metrics: Dict[str, float],
    config_name: str,
    save_path: Path
):
    """Create classification comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    n_classes = len(roi_regions)
    cmap = create_roi_colormap(roi_regions)

    # Classification
    im1 = axes[0].imshow(cluster_map, cmap=cmap, vmin=0, vmax=n_classes-1)
    axes[0].set_title(f'Classification\nAccuracy: {metrics.get("accuracy", 0):.4f}')
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=n_classes-1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Difference
    diff = (cluster_map != ground_truth).astype(int)
    diff[ground_truth < 0] = -1
    diff_cmap = mcolors.ListedColormap(['gray', 'white', 'red'])
    axes[2].imshow(diff, cmap=diff_cmap, vmin=-1, vmax=1)
    axes[2].set_title('Errors (Red)')
    axes[2].axis('off')

    plt.suptitle(f'{config_name}\nF1: {metrics.get("f1_weighted", 0):.4f}, Kappa: {metrics.get("cohen_kappa", 0):.4f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roi_overlay_with_accuracy(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray,
    roi_regions: List[ROIRegion],
    overall_accuracy: float,
    save_path: Path,
    title: str = None
):
    """Create ROI overlay visualization with accuracy."""
    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = create_roi_colormap(roi_regions)
    ax.imshow(cluster_map, cmap=cmap, vmin=0, vmax=len(roi_regions)-1)

    # Draw ROI rectangles
    for roi in roi_regions:
        y_start, y_end, x_start, x_end = roi.coords
        rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                            fill=False, edgecolor=roi.color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x_start, y_start - 5, roi.name, fontsize=10, color=roi.color, fontweight='bold')

    ax.set_title(f'{title or "Classification"}\nOverall Accuracy: {overall_accuracy:.4f}',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_enumerated_objects(
    ground_truth: np.ndarray,
    labeled_objects: np.ndarray,
    num_objects: int,
    save_path: Path
):
    """Plot ground truth with enumerated objects."""
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(ground_truth, cmap='tab20')

    # Add object labels
    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_objects == obj_id
        if np.any(obj_mask):
            y_coords, x_coords = np.where(obj_mask)
            cy, cx = np.mean(y_coords), np.mean(x_coords)
            ax.text(cx, cy, str(obj_id), fontsize=12, ha='center', va='center',
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))

    ax.set_title(f'Ground Truth with {num_objects} Enumerated Objects', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_object_accuracy_overlay(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray,
    roi_regions: List[ROIRegion],
    labeled_objects: np.ndarray,
    object_metrics: List[Dict],
    overall_accuracy: float,
    save_path: Path,
    title: str = None
):
    """Plot classification with per-object accuracy labels."""
    fig, ax = plt.subplots(figsize=(14, 10))

    cmap = create_roi_colormap(roi_regions)
    ax.imshow(cluster_map, cmap=cmap, vmin=0, vmax=len(roi_regions)-1)

    # Add per-object accuracy labels
    for obj in object_metrics:
        obj_mask = labeled_objects == obj['object_id']
        if np.any(obj_mask):
            y_coords, x_coords = np.where(obj_mask)
            cy, cx = np.mean(y_coords), np.mean(x_coords)

            color = 'green' if obj['accuracy'] > 0.8 else 'yellow' if obj['accuracy'] > 0.5 else 'red'
            ax.text(cx, cy, f"{obj['accuracy']:.2f}", fontsize=10, ha='center', va='center',
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

    ax.set_title(f'{title or "Object-wise Accuracy"}\nOverall: {overall_accuracy:.4f}',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Summary Visualization Functions
# ============================================================================

def create_summary_visualizations(
    results_df: pd.DataFrame,
    output_dir: Path
):
    """Create comprehensive summary visualizations."""

    # Filter valid results
    valid_df = results_df[~results_df['accuracy'].isna() & (results_df['error'].isna())].copy()

    if len(valid_df) < 2:
        print("  Not enough valid results for summary visualizations")
        return

    # 1. Combinations vs all metrics
    print("  Creating combinations vs metrics plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metrics_to_plot = ['accuracy', 'precision_weighted', 'recall_weighted',
                       'f1_weighted', 'cohen_kappa', 'purity']

    for ax, metric in zip(axes.flatten(), metrics_to_plot):
        if metric not in valid_df.columns:
            continue

        scatter = ax.scatter(valid_df['n_features'], valid_df[metric],
                           c=valid_df[metric], cmap='viridis', s=80, edgecolors='black')

        # Highlight best
        best_idx = valid_df[metric].idxmax()
        ax.scatter(valid_df.loc[best_idx, 'n_features'], valid_df.loc[best_idx, metric],
                  s=200, color='red', marker='*', zorder=5, label=f'Best: {valid_df.loc[best_idx, metric]:.3f}')

        # Highlight baseline
        baseline = valid_df[valid_df['config_name'] == 'BASELINE_FULL_DATA']
        if len(baseline) > 0:
            ax.scatter(baseline['n_features'].values[0], baseline[metric].values[0],
                      s=150, color='blue', marker='s', zorder=5, label=f'Baseline: {baseline[metric].values[0]:.3f}')

        ax.set_xlabel('Number of Features')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Wavelength Selection Impact on All Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'combinations_vs_all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Pareto frontier for accuracy
    print("  Creating Pareto frontier plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes, ['accuracy', 'f1_weighted']):
        if metric not in valid_df.columns:
            continue

        x = valid_df['n_features'].values
        y = valid_df[metric].values

        ax.scatter(x, y, c=y, cmap='viridis', s=80, edgecolors='black', alpha=0.7)

        # Find Pareto frontier
        sorted_idx = np.argsort(x)
        pareto_x, pareto_y = [x[sorted_idx[0]]], [y[sorted_idx[0]]]
        max_y = y[sorted_idx[0]]

        for i in sorted_idx[1:]:
            if y[i] > max_y:
                pareto_x.append(x[i])
                pareto_y.append(y[i])
                max_y = y[i]

        ax.plot(pareto_x, pareto_y, 'r-', linewidth=2, label='Pareto Frontier')
        ax.scatter(pareto_x, pareto_y, s=120, color='red', marker='D', zorder=5)

        ax.set_xlabel('Number of Features (Complexity)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Pareto Frontier: {metric.replace("_", " ").title()} vs Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontiers.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Metrics correlation matrix
    print("  Creating correlation matrix...")
    metrics_cols = ['accuracy', 'precision_weighted', 'recall_weighted',
                    'f1_weighted', 'cohen_kappa', 'purity', 'ari', 'nmi']
    available_cols = [c for c in metrics_cols if c in valid_df.columns]

    if len(available_cols) > 1:
        corr_matrix = valid_df[available_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, ax=ax)
        ax.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Individual metric plots
    print("  Creating individual metric plots...")
    for metric in metrics_to_plot:
        if metric not in valid_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(valid_df['n_features'], valid_df[metric],
                           s=100, alpha=0.7, c=valid_df[metric],
                           cmap='viridis', edgecolors='black')

        # Trend line
        if len(valid_df) >= 3:
            try:
                z = np.polyfit(valid_df['n_features'], valid_df[metric], 2)
                p = np.poly1d(z)
                x_trend = np.linspace(valid_df['n_features'].min(), valid_df['n_features'].max(), 100)
                ax.plot(x_trend, p(x_trend), 'r-', alpha=0.5, linewidth=2, label='Trend')
            except:
                pass

        # Highlight best and baseline
        best_idx = valid_df[metric].idxmax()
        ax.scatter(valid_df.loc[best_idx, 'n_features'], valid_df.loc[best_idx, metric],
                  s=200, color='red', marker='*', edgecolors='darkred', linewidth=2,
                  label=f'Best: {valid_df.loc[best_idx, metric]:.3f}', zorder=5)

        baseline = valid_df[valid_df['config_name'] == 'BASELINE_FULL_DATA']
        if len(baseline) > 0:
            ax.scatter(baseline['n_features'].values[0], baseline[metric].values[0],
                      s=200, color='blue', marker='s', edgecolors='darkblue', linewidth=2,
                      label=f'Baseline: {baseline[metric].values[0]:.3f}', zorder=5)

        ax.set_xlabel('Number of Spectral Features', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'Wavelength Selection Impact on {metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Add statistics
        corr = valid_df['n_features'].corr(valid_df[metric])
        ax.text(0.02, 0.02, f'Correlation: {corr:.3f}\nSamples: {len(valid_df)}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_dir / f'combinations_vs_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  All summary visualizations saved to: {output_dir}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive wavelength selection and classification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python comprehensive_pipeline.py --data Data/processed/Lichens --mask mask.png
    python comprehensive_pipeline.py --data data.pkl --mask mask.png --sample Lichens --max-configs 5
    python comprehensive_pipeline.py --data Data/raw --mask mask.png --n-bands 10,20,30,40,50
        """
    )
    parser.add_argument('--data', type=str, required=True, help='Path to data (pickle or directory)')
    parser.add_argument('--mask', type=str, required=True, help='Path to ground truth PNG mask')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--sample', type=str, default='Sample', help='Sample name')
    parser.add_argument('--max-configs', type=int, default=None, help='Maximum configurations to run')
    parser.add_argument('--n-bands', type=str, default='10,20,30,40,50', help='Comma-separated n_bands to test')
    parser.add_argument('--diversity-methods', type=str, default='mmr,spectral_spread',
                       help='Comma-separated diversity methods')
    parser.add_argument('--lambda-values', type=str, default='0.3,0.5', help='Comma-separated lambda values')
    parser.add_argument('--no-diversity-configs', action='store_true', help='Include no-diversity configs')
    parser.add_argument('--export-concat-data', action='store_true', help='Export concatenated spectral data')

    args = parser.parse_args()

    # Parse parameters
    n_bands_list = [int(x.strip()) for x in args.n_bands.split(',')]
    diversity_methods = [x.strip() for x in args.diversity_methods.split(',')]
    lambda_values = [float(x.strip()) for x in args.lambda_values.split(',')]

    # Setup paths
    data_path = Path(args.data)
    mask_path = Path(args.mask)
    sample_name = args.sample

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / 'results' / f'{sample_name}_pipeline_{timestamp}'

    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_results_dir = output_dir / 'paper-results'
    paper_results_dir.mkdir(exist_ok=True)
    concat_data_dir = output_dir / 'concat-data'
    concat_data_dir.mkdir(exist_ok=True)
    experiments_dir = output_dir / 'experiments'
    experiments_dir.mkdir(exist_ok=True)
    analysis_summary_dir = output_dir / 'analysis_summary'
    analysis_summary_dir.mkdir(exist_ok=True)
    supervised_metrics_dir = output_dir / 'supervised_metrics'
    supervised_metrics_dir.mkdir(exist_ok=True)
    summary_viz_dir = output_dir / 'summary_visualizations'
    summary_viz_dir.mkdir(exist_ok=True)
    model_dir = output_dir / 'model_output' / sample_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE WAVELENGTH SELECTION PIPELINE")
    print("=" * 80)
    print(f"  Data: {data_path}")
    print(f"  Mask: {mask_path}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    full_data, spectra_data = load_hyperspectral_data(data_path)

    total_bands = sum(len(full_data['data'][str(ex)]['wavelengths']) for ex in full_data['excitation_wavelengths'])
    print(f"  Excitations: {len(full_data['excitation_wavelengths'])}")
    print(f"  Total bands: {total_bands}")

    first_ex = str(full_data['excitation_wavelengths'][0])
    height, width = full_data['data'][first_ex]['cube'].shape[:2]
    print(f"  Spatial dimensions: {height} x {width}")

    # ========================================================================
    # Extract Ground Truth
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING GROUND TRUTH")
    print("=" * 80)

    ground_truth, color_mapping, class_colors = extract_ground_truth_from_png(
        mask_path, target_shape=(height, width)
    )

    n_classes = len(class_colors)

    # ========================================================================
    # Detect ROI Regions
    # ========================================================================
    print("\n" + "=" * 80)
    print("DETECTING ROI REGIONS")
    print("=" * 80)

    roi_regions = auto_detect_rois_from_ground_truth(ground_truth, class_colors)

    # Create mask for wavelength selection
    analysis_mask = ground_truth >= 0
    mask_path_npy = output_dir / 'analysis_mask.npy'
    np.save(mask_path_npy, analysis_mask)

    # ========================================================================
    # Object-wise Segmentation
    # ========================================================================
    print("\n" + "=" * 80)
    print("OBJECT-WISE SEGMENTATION")
    print("=" * 80)

    foreground_mask = ground_truth >= 0
    labeled_objects, num_objects = ndimage.label(foreground_mask)
    print(f"Found {num_objects} spatially separated objects in ground truth")

    # Save enumerated objects visualization
    plot_enumerated_objects(ground_truth, labeled_objects, num_objects,
                           paper_results_dir / 'ground_truth_enumerated_objects.png')

    # ========================================================================
    # Generate Configurations
    # ========================================================================
    configurations = generate_configurations(
        n_bands_list=n_bands_list,
        diversity_methods=diversity_methods,
        lambda_values=lambda_values,
        include_no_diversity=not args.no_diversity_configs
    )

    if args.max_configs:
        configurations = configurations[:args.max_configs]

    print(f"\n{len(configurations)} configurations to run")

    # ========================================================================
    # Baseline: Full Data Classification
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE: Full Data Classification")
    print("=" * 80)

    with PerformanceTimer() as baseline_timer:
        cluster_map_baseline, metrics_baseline, n_features_baseline = run_knn_classification(
            full_data, roi_regions, ground_truth,
            export_concat_data=args.export_concat_data,
            concat_data_path=concat_data_dir / 'BASELINE_FULL_DATA_concat.csv' if args.export_concat_data else None
        )

    # Calculate object metrics for baseline
    baseline_object_metrics = calculate_object_metrics(
        cluster_map_baseline, ground_truth, labeled_objects, num_objects
    )

    # Save baseline visualizations
    baseline_exp_dir = experiments_dir / 'BASELINE_FULL_DATA'
    baseline_exp_dir.mkdir(exist_ok=True)

    plot_classification_comparison(
        cluster_map_baseline, ground_truth, roi_regions, metrics_baseline,
        'BASELINE_FULL_DATA', baseline_exp_dir / 'BASELINE_classification.png'
    )

    plot_roi_overlay_with_accuracy(
        cluster_map_baseline, ground_truth, roi_regions, metrics_baseline['accuracy'],
        paper_results_dir / 'BASELINE_roi_overlay.png', 'BASELINE - Full Data'
    )

    plot_object_accuracy_overlay(
        cluster_map_baseline, ground_truth, roi_regions, labeled_objects,
        baseline_object_metrics, metrics_baseline['accuracy'],
        baseline_exp_dir / 'BASELINE_object_accuracy.png', 'BASELINE - Object Accuracy'
    )

    # Save baseline object metrics
    pd.DataFrame(baseline_object_metrics).to_csv(
        baseline_exp_dir / 'BASELINE_object_metrics.csv', index=False
    )

    # Save baseline supervised metrics
    with open(supervised_metrics_dir / 'BASELINE_supervised_metrics.json', 'w') as f:
        json.dump(metrics_baseline, f, indent=2, default=str)

    print(f"\nBaseline Results:")
    print(f"  Features: {n_features_baseline}")
    print(f"  Accuracy: {metrics_baseline['accuracy']:.4f}")
    print(f"  F1 (weighted): {metrics_baseline['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa: {metrics_baseline['cohen_kappa']:.4f}")
    print(f"  Purity: {metrics_baseline['purity']:.4f}")
    print(f"  ARI: {metrics_baseline['ari']:.4f}")
    print(f"  NMI: {metrics_baseline['nmi']:.4f}")
    print(f"  [TIME] Classification: {baseline_timer.elapsed:.2f}s")

    mean_obj_acc = np.mean([o['accuracy'] for o in baseline_object_metrics])
    std_obj_acc = np.std([o['accuracy'] for o in baseline_object_metrics])
    print(f"  Mean object accuracy: {mean_obj_acc:.4f} +/- {std_obj_acc:.4f}")

    # Store results
    results = [ExperimentResult(
        config_name='BASELINE_FULL_DATA',
        n_bands_selected=total_bands,
        n_features=n_features_baseline,
        data_reduction_pct=0.0,
        accuracy=metrics_baseline['accuracy'],
        precision_weighted=metrics_baseline['precision_weighted'],
        recall_weighted=metrics_baseline['recall_weighted'],
        f1_weighted=metrics_baseline['f1_weighted'],
        cohen_kappa=metrics_baseline['cohen_kappa'],
        purity=metrics_baseline['purity'],
        ari=metrics_baseline['ari'],
        nmi=metrics_baseline['nmi'],
        selection_time=0.0,
        clustering_time=baseline_timer.elapsed,
        speedup_factor=1.0,
        mean_object_accuracy=mean_obj_acc,
        std_object_accuracy=std_obj_acc,
        use_diversity_constraint=False,
        diversity_method='none',
        lambda_diversity=0.0
    )]

    # Accumulate all object metrics
    all_object_metrics = []
    for obj in baseline_object_metrics:
        all_object_metrics.append({
            **obj,
            'config_name': 'BASELINE_FULL_DATA',
            'n_features': n_features_baseline
        })

    # ========================================================================
    # Run Wavelength Selection Configurations
    # ========================================================================
    print("\n" + "=" * 80)
    print("RUNNING WAVELENGTH SELECTION CONFIGURATIONS")
    print("=" * 80)

    model_path = model_dir / 'best_hyperspectral_model.pth'

    for i, config in enumerate(tqdm(configurations, desc="Running configurations")):
        print(f"\n[{i+1}/{len(configurations)}] {config.name}")

        exp_dir = experiments_dir / config.name
        exp_dir.mkdir(exist_ok=True)

        try:
            # Wavelength selection
            with PerformanceTimer() as selection_timer:
                wavelength_combinations, analyzer = run_wavelength_selection(
                    data_path if data_path.suffix == '.pkl' else data_path,
                    mask_path_npy,
                    sample_name,
                    config,
                    model_path,
                    exp_dir / 'wavelength_selection'
                )

            print(f"  Selected {len(wavelength_combinations)} wavelength combinations")
            print(f"  [TIME] Selection: {selection_timer.elapsed:.2f}s")

            # Extract subset
            subset_data = extract_wavelength_subset(full_data, wavelength_combinations)

            # Classification
            with PerformanceTimer() as cluster_timer:
                cluster_map, metrics, n_features = run_knn_classification(
                    subset_data, roi_regions, ground_truth,
                    export_concat_data=args.export_concat_data,
                    concat_data_path=concat_data_dir / f'{config.name}_concat.csv' if args.export_concat_data else None
                )

            # Calculate object metrics
            object_metrics = calculate_object_metrics(
                cluster_map, ground_truth, labeled_objects, num_objects
            )

            # Save visualizations
            plot_classification_comparison(
                cluster_map, ground_truth, roi_regions, metrics,
                config.name, exp_dir / f'{config.name}_classification.png'
            )

            plot_roi_overlay_with_accuracy(
                cluster_map, ground_truth, roi_regions, metrics['accuracy'],
                paper_results_dir / f'{config.name}_roi_overlay.png', config.name
            )

            plot_object_accuracy_overlay(
                cluster_map, ground_truth, roi_regions, labeled_objects,
                object_metrics, metrics['accuracy'],
                exp_dir / f'{config.name}_object_accuracy.png', f'{config.name} - Object Accuracy'
            )

            # Save metrics
            pd.DataFrame(object_metrics).to_csv(exp_dir / f'{config.name}_object_metrics.csv', index=False)

            with open(supervised_metrics_dir / f'{config.name}_supervised_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2, default=str)

            # Save wavelength combinations
            with open(exp_dir / f'{config.name}_wavelength_combinations.json', 'w') as f:
                json.dump(wavelength_combinations, f, indent=2, default=str)

            # Calculate summary
            data_reduction = (1 - n_features / n_features_baseline) * 100
            speedup = baseline_timer.elapsed / cluster_timer.elapsed if cluster_timer.elapsed > 0 else 0
            mean_obj_acc = np.mean([o['accuracy'] for o in object_metrics]) if object_metrics else 0
            std_obj_acc = np.std([o['accuracy'] for o in object_metrics]) if object_metrics else 0

            print(f"  Features: {n_features} ({data_reduction:.1f}% reduction)")
            print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")
            print(f"  [TIME] Classification: {cluster_timer.elapsed:.2f}s | Speedup: {speedup:.2f}x")

            # Store result
            results.append(ExperimentResult(
                config_name=config.name,
                n_bands_selected=len(wavelength_combinations),
                n_features=n_features,
                data_reduction_pct=data_reduction,
                accuracy=metrics['accuracy'],
                precision_weighted=metrics['precision_weighted'],
                recall_weighted=metrics['recall_weighted'],
                f1_weighted=metrics['f1_weighted'],
                cohen_kappa=metrics['cohen_kappa'],
                purity=metrics.get('purity', 0),
                ari=metrics.get('ari', 0),
                nmi=metrics.get('nmi', 0),
                selection_time=selection_timer.elapsed,
                clustering_time=cluster_timer.elapsed,
                speedup_factor=speedup,
                mean_object_accuracy=mean_obj_acc,
                std_object_accuracy=std_obj_acc,
                use_diversity_constraint=config.use_diversity_constraint,
                diversity_method=config.diversity_method,
                lambda_diversity=config.lambda_diversity
            ))

            # Accumulate object metrics
            for obj in object_metrics:
                all_object_metrics.append({
                    **obj,
                    'config_name': config.name,
                    'n_features': n_features
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

            results.append(ExperimentResult(
                config_name=config.name,
                n_bands_selected=0,
                n_features=0,
                data_reduction_pct=0,
                error=str(e),
                use_diversity_constraint=config.use_diversity_constraint,
                diversity_method=config.diversity_method,
                lambda_diversity=config.lambda_diversity
            ))

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Convert results to DataFrame
    results_df = pd.DataFrame([asdict(r) for r in results])
    results_df = results_df.sort_values('accuracy', ascending=False, na_position='last')

    # Save to Excel with multiple sheets
    excel_path = output_dir / 'pipeline_results.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All_Results', index=False)

        # Summary sheet
        summary = results_df[['config_name', 'n_features', 'data_reduction_pct',
                             'accuracy', 'f1_weighted', 'cohen_kappa', 'purity', 'ari', 'nmi',
                             'mean_object_accuracy', 'selection_time', 'clustering_time', 'speedup_factor']].copy()
        summary.to_excel(writer, sheet_name='Summary', index=False)

        # Configuration parameters sheet
        config_params = results_df[['config_name', 'n_bands_selected', 'use_diversity_constraint',
                                   'diversity_method', 'lambda_diversity']].copy()
        config_params.to_excel(writer, sheet_name='Config_Parameters', index=False)

    print(f"  Results saved to: {excel_path}")

    # Save CSV
    results_df.to_csv(output_dir / 'pipeline_results.csv', index=False)

    # Save object metrics
    if all_object_metrics:
        all_obj_df = pd.DataFrame(all_object_metrics)
        all_obj_df.to_csv(analysis_summary_dir / 'all_object_metrics.csv', index=False)

        # Per-config summary
        per_config = all_obj_df.groupby('config_name').agg({
            'accuracy': ['mean', 'std', 'count']
        }).round(4)
        per_config.columns = ['mean_accuracy', 'std_accuracy', 'n_objects']
        per_config.to_csv(analysis_summary_dir / 'per_config_object_summary.csv')

    # ========================================================================
    # Summary Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("=" * 80)

    create_summary_visualizations(results_df, summary_viz_dir)

    # ========================================================================
    # Print Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    valid_results = results_df[results_df['error'].isna()].copy()

    if len(valid_results) > 0:
        best = valid_results.iloc[0]
        baseline = valid_results[valid_results['config_name'] == 'BASELINE_FULL_DATA']

        if len(baseline) > 0:
            baseline = baseline.iloc[0]
            print(f"\nBaseline (full data):")
            print(f"  Features: {baseline['n_features']}")
            print(f"  Accuracy: {baseline['accuracy']:.4f}")
            print(f"  F1: {baseline['f1_weighted']:.4f}")

        print(f"\nBest configuration: {best['config_name']}")
        print(f"  Features: {best['n_features']} ({best['data_reduction_pct']:.1f}% reduction)")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  F1: {best['f1_weighted']:.4f}")
        print(f"  Cohen's Kappa: {best['cohen_kappa']:.4f}")

        if len(baseline) > 0:
            acc_change = (best['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
            print(f"\n  Accuracy change vs baseline: {acc_change:+.2f}%")

    print(f"\nResults directory: {output_dir}")
    print(f"  - paper-results/: Paper-ready visualizations")
    print(f"  - experiments/: Per-experiment folders")
    print(f"  - summary_visualizations/: Summary plots")
    print(f"  - analysis_summary/: Aggregated metrics")
    print(f"  - supervised_metrics/: JSON metric files")
    print(f"  - pipeline_results.xlsx: Full results spreadsheet")
    print("=" * 80)


if __name__ == '__main__':
    main()
