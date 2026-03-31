#!/usr/bin/env python3
"""
Lichens Wavelength Selection Pipeline

A clean script to run wavelength selection and classification experiments
on the Lichens dataset using the spectral_select module.

Usage:
    python scripts/run_lichens_pipeline.py --retrain        # Train new model
    python scripts/run_lichens_pipeline.py --use-existing   # Use latest model
    python scripts/run_lichens_pipeline.py --model PATH     # Use specific model

Note: This script uses pickle to load spectral data files (.pkl) which is
required for the scientific hyperspectral data format used in this project.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from itertools import product
import warnings
import time

warnings.filterwarnings('ignore')

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Set random seeds
np.random.seed(42)
import random
random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Import project modules
from spectral_select import Config, Analyzer, SpectraData
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ParameterSpace:
    """Define the parameter space for experiments.

    FULL EXPLORATION CONFIG:
    - n_bands: 5 to 180 (14 values)
    - dimension_selection: all 3 methods
    - n_important_dimensions: [1, 3, 5]
    - perturbation_method: all 3 methods
    - perturbation_magnitudes: medium + high intensity (2 variants)
    - normalization: all 3 methods
    - diversity: mmr with lambda=0.5 (fixed)

    Total: 14 × 3 × 3 × 3 × 2 × 3 = 2,268 experiments
    """

    # Number of bands to select:
    #   1-20 step 1, then 22-60 step 2, then 65-180 step 5
    n_bands_to_select: Union[int, List[int]] = field(
        default_factory=lambda: (
            list(range(1, 21, 1)) +      # 1-20 step 1:  [1,2,3,...,20]     = 20 values
            list(range(22, 61, 2)) +     # 22-60 step 2: [22,24,26,...,60]  = 20 values
            list(range(65, 181, 5))      # 65-180 step 5: [65,70,75,...,180] = 24 values
        )                                # Total: 64 values
    )

    # Dimension selection - PCA only (best from paper)
    # TODO: uncomment for full sweep: default_factory=lambda: ["variance", "pca"]
    dimension_selection_method: Union[str, List[str]] = field(
        default_factory=lambda: ["pca", 'variance']
    )

    # Important dimensions
    n_important_dimensions: Union[int, List[int]] = field(
        default_factory=lambda: [1, 3]
    )

    # Perturbation method - ALL methods
    perturbation_method: Union[str, List[str]] = field(
        default_factory=lambda: ["percentile", "absolute_range"]
    )

    # Perturbation magnitudes - 2 variants: medium and high intensity
    # This will be handled specially - see PERTURBATION_MAGNITUDE_VARIANTS below
    perturbation_magnitudes: List[int] = field(default_factory=lambda: [30, 40, 50])

    # Normalization - ALL methods
    normalization_method: Union[str, List[str]] = field(
        default_factory=lambda: ["variance", "max_per_excitation", "none"]
    )

    # Diversity - FIXED (mmr with lambda=0.5)
    use_diversity_constraint: Union[bool, List[bool]] = True
    diversity_method: Union[str, List[str]] = "mmr"
    lambda_diversity: Union[float, List[float]] = 0.5
    min_distance_nm: Union[float, List[float]] = 15.0

    # Training (only used if retraining)
    training_epochs: int = 60


# Perturbation magnitude variants - medium only (best from paper)
# TODO: uncomment for full sweep: "high": [50, 60, 70],
PERTURBATION_MAGNITUDE_VARIANTS = {
    # "medium": [30, 40, 50],
    "high": [50, 60, 70]
}


def generate_experiment_configs(param_space: ParameterSpace, use_magnitude_variants: bool = True) -> List[dict]:
    """Generate all experiment configurations from parameter space.

    Args:
        param_space: Parameter space configuration
        use_magnitude_variants: If True, uses PERTURBATION_MAGNITUDE_VARIANTS (medium + high)
                               If False, uses param_space.perturbation_magnitudes only
    """
    params = {}
    for field_name in ['n_bands_to_select', 'dimension_selection_method', 'n_important_dimensions',
                       'perturbation_method', 'normalization_method', 'use_diversity_constraint',
                       'diversity_method', 'lambda_diversity', 'min_distance_nm']:
        value = getattr(param_space, field_name)
        params[field_name] = [value] if not isinstance(value, list) else value

    # Handle perturbation magnitudes - use variants if enabled
    if use_magnitude_variants and PERTURBATION_MAGNITUDE_VARIANTS:
        params['perturbation_magnitudes'] = list(PERTURBATION_MAGNITUDE_VARIANTS.values())
        magnitude_names = list(PERTURBATION_MAGNITUDE_VARIANTS.keys())
    else:
        params['perturbation_magnitudes'] = [param_space.perturbation_magnitudes]
        magnitude_names = ["default"]

    params['training_epochs'] = [param_space.training_epochs]

    keys = list(params.keys())
    values = [params[k] for k in keys]

    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))

        # Skip invalid: diversity disabled but method/lambda varies
        if not config['use_diversity_constraint']:
            if config['diversity_method'] != 'mmr' or config['lambda_diversity'] != 0.5:
                continue

        # Determine magnitude variant name
        mag_tuple = tuple(config['perturbation_magnitudes'])
        mag_name = "default"
        for name, mags in PERTURBATION_MAGNITUDE_VARIANTS.items():
            if tuple(mags) == mag_tuple:
                mag_name = name
                break

        # Generate descriptive name
        name_parts = [
            f"bands_{config['n_bands_to_select']}",
            f"{config['dimension_selection_method'][:3]}",
            f"dim_{config['n_important_dimensions']}",
            f"{config['perturbation_method'][:4]}",
            f"mag_{mag_name}",
            f"{config['normalization_method'][:3]}",
        ]

        config['name'] = "_".join(name_parts)
        config['magnitude_variant'] = mag_name
        configs.append(config)

    return configs


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Timer:
    """Context manager for timing."""
    def __init__(self):
        self.elapsed = 0
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def find_latest_model(results_dir: Path, pattern: str = "Lichens") -> Optional[Path]:
    """Find the most recent trained model."""
    if not results_dir.exists():
        return None

    for subdir in sorted(results_dir.iterdir(), reverse=True):
        if subdir.is_dir() and pattern in subdir.name:
            for model_path in ["model/autoencoder_model.pth", "model_output/*/model.pth"]:
                for match in subdir.glob(model_path):
                    if match.exists():
                        return match
    return None


def load_hyperspectral_data(data_path: Path) -> Tuple[Dict[str, Any], SpectraData]:
    """Load hyperspectral data in both dict and SpectraData formats."""
    print(f"Loading data from: {data_path}")

    # SpectraData.from_pickle loads the project's spectral data format
    spectra_data = SpectraData.from_pickle(data_path)

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

    return full_data, spectra_data


def load_ground_truth(mask_path: Path, roi_path: Path) -> Tuple[np.ndarray, List[dict], Dict]:
    """Load ground truth mask and ROI regions."""
    from PIL import Image

    # Load mask
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)

    # Load ROI data
    with open(roi_path, 'r') as f:
        roi_data = json.load(f)

    # Build ground truth from colors
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
        print(f"  Class {class_id} ({cls['name']}): {np.sum(mask):,} pixels")

    # Convert ROI regions
    roi_regions = []
    for roi in roi_data['regions']:
        rect = roi['rect']
        roi_regions.append({
            'name': roi['class_name'],
            'class_id': roi['class_id'],
            'coords': (rect['row_min'], rect['row_max'], rect['col_min'], rect['col_max'])
        })

    return ground_truth, roi_regions, class_info


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def run_wavelength_selection(
    data_path: Path,
    mask_path: Path,
    model_path: Path,
    output_dir: Path,
    sample_name: str,
    exp_config: dict,
    retrain: bool = False
) -> Tuple[List[dict], Any]:
    """Run wavelength selection with given configuration."""

    # Determine training epochs
    if retrain or not model_path.exists():
        epochs = exp_config['training_epochs']
        print(f"  Training new model ({epochs} epochs)...")
    else:
        epochs = 1  # Won't train if model exists
        print(f"  Using existing model: {model_path}")

    config = Config(
        sample_name=sample_name,
        data_path=str(data_path),
        mask_path=str(mask_path),
        model_path=str(model_path),
        output_dir=str(output_dir),

        n_bands_to_select=exp_config['n_bands_to_select'],
        n_important_dimensions=exp_config['n_important_dimensions'],
        dimension_selection_method=exp_config['dimension_selection_method'],
        perturbation_method=exp_config['perturbation_method'],
        perturbation_magnitudes=exp_config['perturbation_magnitudes'],
        normalization_method=exp_config['normalization_method'],
        use_diversity_constraint=exp_config['use_diversity_constraint'],
        diversity_method=exp_config['diversity_method'],
        lambda_diversity=exp_config['lambda_diversity'],
        min_distance_nm=exp_config['min_distance_nm'],
        training_epochs=epochs,

        save_visualizations=False,
        save_tiff_layers=False,
    )

    analyzer = Analyzer(config)
    spectra = SpectraData.from_pickle(data_path)
    spectra.mask = np.load(mask_path)

    analyzer.fit(spectra)

    result = analyzer.result
    if result is None:
        raise ValueError("Analyzer did not produce a result")

    wavelength_combos = []
    for band in result.selected_bands:
        wavelength_combos.append({
            'excitation': float(band.excitation_nm),
            'emission': float(band.emission_nm),
            'combination_name': f"Ex{band.excitation_nm:.0f}_Em{band.emission_nm:.1f}",
            'influence_score': float(band.influence_score),
            'rank': int(band.rank)
        })

    return wavelength_combos, analyzer


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


def run_knn_classification(
    data: dict,
    roi_regions: List[dict],
    ground_truth: np.ndarray,
    n_neighbors: int = 5
) -> Tuple[np.ndarray, dict]:
    """Run KNN classification on valid pixels only."""

    first_ex = str(data['excitation_wavelengths'][0])
    height, width = data['data'][first_ex]['cube'].shape[:2]

    # Valid pixels only
    valid_mask = ground_truth >= 0
    valid_coords = np.argwhere(valid_mask)

    # Build features for valid pixels
    features = []
    for y, x in valid_coords:
        pixel = []
        for ex in data['excitation_wavelengths']:
            cube = data['data'][str(ex)]['cube']
            pixel.extend(cube[y, x, :])
        features.append(pixel)

    X_full = np.array(features)
    n_features = X_full.shape[1]

    # Handle NaN
    if np.any(np.isnan(X_full)):
        X_full = np.nan_to_num(X_full, nan=0.0)

    # Training data from ROIs
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

    # Train and predict
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

    # Metrics
    y_true = ground_truth[valid_mask]
    y_pred = cluster_map[valid_mask]

    metrics = {
        'n_features': n_features,
        'n_train_samples': len(X_train),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'ari': adjusted_rand_score(y_true, y_pred),
        'nmi': normalized_mutual_info_score(y_true, y_pred),
    }

    return cluster_map, metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    param_space: ParameterSpace,
    model_mode: str = "use-existing",  # "retrain", "use-existing", or path
    sample_name: str = "Lichens_Dataset_1"
):
    """Run the complete pipeline."""

    print("=" * 70)
    print("LICHENS WAVELENGTH SELECTION PIPELINE")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model mode: {model_mode}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Setup paths
    data_file = data_dir / "spectra_masked.pkl"
    mask_file = data_dir / "class_mask.png"
    roi_file = data_dir / "roi_regions.json"

    # Verify files exist
    for name, path in [("Data", data_file), ("Mask", mask_file), ("ROI", roi_file)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        print(f"  {name}: OK")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    experiments_dir = output_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    # Determine model path
    if model_mode == "retrain":
        model_path = model_dir / "autoencoder_model.pth"
        retrain = True
    elif model_mode == "use-existing":
        existing = find_latest_model(PROJECT_ROOT / "results")
        if existing:
            model_path = existing
            retrain = False
            print(f"\nUsing existing model: {model_path}")
        else:
            print("\nNo existing model found, will train new one.")
            model_path = model_dir / "autoencoder_model.pth"
            retrain = True
    else:
        model_path = Path(model_mode)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        retrain = False

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    full_data, spectra_data = load_hyperspectral_data(data_file)
    ground_truth, roi_regions, class_info = load_ground_truth(mask_file, roi_file)

    # Save analysis mask
    analysis_mask = ground_truth >= 0
    analysis_mask_path = output_dir / "analysis_mask.npy"
    np.save(analysis_mask_path, analysis_mask)

    total_bands = sum(len(full_data['data'][str(ex)]['wavelengths'])
                      for ex in full_data['excitation_wavelengths'])
    print(f"\nData summary:")
    print(f"  Excitations: {len(full_data['excitation_wavelengths'])}")
    print(f"  Total bands: {total_bands}")
    print(f"  Valid pixels: {np.sum(analysis_mask):,}")

    # Baseline classification
    print("\n" + "=" * 70)
    print("BASELINE CLASSIFICATION (Full Data)")
    print("=" * 70)

    with Timer() as baseline_timer:
        cluster_map_baseline, metrics_baseline = run_knn_classification(
            full_data, roi_regions, ground_truth
        )

    print(f"\nBaseline Results:")
    print(f"  Features: {metrics_baseline['n_features']}")
    print(f"  Accuracy: {metrics_baseline['accuracy']:.4f}")
    print(f"  F1 Score: {metrics_baseline['f1']:.4f}")
    print(f"  Kappa: {metrics_baseline['kappa']:.4f}")
    print(f"  Time: {baseline_timer.elapsed:.2f}s")

    # Generate experiment configs
    configs = generate_experiment_configs(param_space)
    print(f"\n{len(configs)} experiment configuration(s) to run")

    # Results storage
    results = [{
        'config': 'BASELINE',
        'n_bands_selected': total_bands,
        **metrics_baseline,
        'reduction_pct': 0.0,
        'selection_time': 0.0,
        'classification_time': baseline_timer.elapsed
    }]

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTS")
    print("=" * 70)

    for i, exp_config in enumerate(configs):
        config_name = exp_config['name']
        print(f"\n[{i+1}/{len(configs)}] {config_name}")
        print("-" * 50)

        exp_dir = experiments_dir / config_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Wavelength selection
            with Timer() as sel_timer:
                wavelength_combos, analyzer = run_wavelength_selection(
                    data_path=data_file,
                    mask_path=analysis_mask_path,
                    model_path=model_path,
                    output_dir=exp_dir,
                    sample_name=sample_name,
                    exp_config=exp_config,
                    retrain=retrain and i == 0  # Only retrain on first config
                )

            print(f"  Selected {len(wavelength_combos)} bands in {sel_timer.elapsed:.2f}s")

            # Save wavelengths
            with open(exp_dir / "wavelengths.json", 'w') as f:
                json.dump(wavelength_combos, f, indent=2)

            # Extract subset and classify
            subset_data = extract_wavelength_subset(full_data, wavelength_combos)

            with Timer() as cls_timer:
                cluster_map, metrics = run_knn_classification(
                    subset_data, roi_regions, ground_truth
                )

            reduction = (1 - metrics['n_features'] / metrics_baseline['n_features']) * 100

            print(f"  Features: {metrics['n_features']} ({reduction:.1f}% reduction)")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Kappa: {metrics['kappa']:.4f}")

            # Store results
            results.append({
                'config': config_name,
                'n_bands_selected': len(wavelength_combos),
                **metrics,
                'reduction_pct': reduction,
                **{k: v for k, v in exp_config.items() if k != 'name'},
                'selection_time': sel_timer.elapsed,
                'classification_time': cls_timer.elapsed
            })

            # After first successful run with retrain, use the new model
            if retrain and i == 0:
                new_model = model_dir / "autoencoder_model.pth"
                if new_model.exists():
                    model_path = new_model
                    retrain = False

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'config': config_name, 'error': str(e)})

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)
    results_df.to_excel(output_dir / "results.xlsx", index=False)

    print(f"\nResults saved to: {output_dir}")
    print("\nFinal Summary:")
    print(results_df[['config', 'n_features', 'reduction_pct', 'accuracy', 'f1', 'kappa']].to_string())

    return results_df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Lichens Wavelength Selection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use defaults from ParameterSpace class (edit the class to change defaults)
    python scripts/run_lichens_pipeline.py

    # Override specific parameters via CLI
    python scripts/run_lichens_pipeline.py --n-bands 10,20,30 --retrain
        """
    )

    parser.add_argument('--data-dir', type=str,
                        default=str(PROJECT_ROOT / "Data" / "processed" / "Lichens Dataset 1"),
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results/Lichens_TIMESTAMP)')

    # Model mode
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--retrain', action='store_true',
                             help='Train a new model')
    model_group.add_argument('--use-existing', action='store_true', default=True,
                             help='Use latest existing model (default)')
    model_group.add_argument('--model', type=str,
                             help='Path to specific model file')

    # Parameter overrides (None = use class defaults)
    parser.add_argument('--n-bands', type=str, default=None,
                        help='Override n_bands (comma-separated, e.g., "5,10,15")')
    parser.add_argument('--n-dims', type=str, default=None,
                        help='Override n_important_dimensions (comma-separated or single value)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override training epochs')

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / f"Lichens_Dataset_1_{timestamp}"

    # Determine model mode
    if args.retrain:
        model_mode = "retrain"
    elif args.model:
        model_mode = args.model
    else:
        model_mode = "use-existing"

    # Start with class defaults
    param_space = ParameterSpace()

    # Override only if CLI args provided
    if args.n_bands is not None:
        n_bands_list = [int(x.strip()) for x in args.n_bands.split(',')]
        param_space.n_bands_to_select = n_bands_list

    if args.n_dims is not None:
        if ',' in args.n_dims:
            param_space.n_important_dimensions = [int(x.strip()) for x in args.n_dims.split(',')]
        else:
            param_space.n_important_dimensions = [int(args.n_dims)]

    if args.epochs is not None:
        param_space.training_epochs = args.epochs

    # Show what we're using
    print("\nParameter Space (from class defaults + CLI overrides):")
    print(f"  n_bands_to_select: {param_space.n_bands_to_select}")
    print(f"  dimension_selection_method: {param_space.dimension_selection_method}")
    print(f"  n_important_dimensions: {param_space.n_important_dimensions}")
    print(f"  normalization_method: {param_space.normalization_method}")
    print(f"  diversity_method: {param_space.diversity_method}")
    print(f"  lambda_diversity: {param_space.lambda_diversity}")
    print(f"  training_epochs: {param_space.training_epochs}")

    # Run pipeline
    run_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        param_space=param_space,
        model_mode=model_mode
    )


if __name__ == "__main__":
    main()
