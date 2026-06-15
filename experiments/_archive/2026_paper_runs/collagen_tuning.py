#!/usr/bin/env python3
"""
Collagen Dataset Tuning Experiments
====================================
Targeted experiments to close the gap between baseline (83.1%) and
wavelength-selected accuracy on the Sponges Acid Group 1 dataset.

Problem: This dataset lacks exposure time and lamp power metadata,
causing ~4x intensity imbalance across excitations (310nm: mean=1224
vs 400nm: mean=313). The autoencoder sees artificially dominant bands
in high-intensity excitations, biasing wavelength selection.

Experiments:
  1. More bands (80, 100, 130, 149) to see if the gap closes with scale
  2. Per-excitation normalization — equalize intensity across excitations
     before autoencoder training, so selection is based on spectral shape
  3. Proxy exposure normalization — use Lichens_2 exposure ratios as a
     rough correction (same instrument, 6 overlapping excitations)
  4. Best config sweep — combine the best normalization with fine-grained
     band counts around the sweet spot

Note: pickle is used here for loading scientific hyperspectral data (.pkl),
which is the standard format in this project's data pipeline.

Usage:
    python experiments/collagen_tuning.py --experiment all
    python experiments/collagen_tuning.py --experiment more_bands
    python experiments/collagen_tuning.py --experiment per_excitation_norm
    python experiments/collagen_tuning.py --experiment proxy_exposure
    python experiments/collagen_tuning.py --experiment fine_sweep
"""

import json
import argparse
import pickle
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
import time

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

np.random.seed(42)
import random
random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

from spectral_select import Config, Analyzer, SpectraData
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score
)
from PIL import Image


# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = PROJECT_ROOT / "Data" / "processed" / "Sponges Acid Group 1"
OUTPUT_BASE = PROJECT_ROOT / "results" / "Sponges_Acid_Tuning"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(pkl_path: Path):
    """Load spectra data and return both raw dict and SpectraData."""
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)  # nosec — trusted local scientific data

    spectra = SpectraData.from_pickle(pkl_path)

    full_data = {
        'data': {},
        'excitation_wavelengths': raw['excitation_wavelengths'],
        'mask': raw.get('mask'),
    }
    for ex_nm, ex_data in spectra.excitations.items():
        wls = ex_data.emission_wavelengths
        if hasattr(wls, 'tolist'):
            wls = wls.tolist()
        full_data['data'][str(ex_nm)] = {
            'cube': ex_data.cube,
            'wavelengths': wls,
        }
    return full_data, spectra


def load_ground_truth(mask_path: Path, roi_path: Path):
    """Load ground truth mask and ROI regions."""
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)

    with open(roi_path, 'r') as f:
        roi_data = json.load(f)

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

    roi_regions = []
    for roi in roi_data['regions']:
        rect = roi['rect']
        roi_regions.append({
            'name': roi['class_name'],
            'class_id': roi['class_id'],
            'coords': (rect['row_min'], rect['row_max'],
                       rect['col_min'], rect['col_max']),
        })

    return ground_truth, roi_regions, class_info


def normalize_per_excitation(raw_data: dict) -> dict:
    """Normalize each excitation cube independently to [0, 1].

    This removes the intensity imbalance caused by different exposure
    times across excitations, so the autoencoder sees spectral shape
    rather than absolute intensity.
    """
    data = copy.deepcopy(raw_data)
    mask = data.get('mask')
    if mask is not None:
        valid = mask > 0
    else:
        valid = None

    for ex_key in data['data']:
        cube = data['data'][ex_key]['cube'].astype(np.float64)
        if valid is not None:
            vals = cube[valid]
            vals = vals[~np.isnan(vals)]
        else:
            vals = cube[~np.isnan(cube)]

        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            cube = (cube - vmin) / (vmax - vmin)
        data['data'][ex_key]['cube'] = cube

    return data


def normalize_proxy_exposure(raw_data: dict) -> dict:
    """Apply proxy exposure normalization using Lichens_2 ratios.

    The Lichens_2 dataset was acquired on the same instrument with known
    exposure times. We use those ratios as an approximate correction for
    the Collagen data which lacks this metadata.

    Lichens_2 exposures (ms):
        310: 5432.9, 325: 924.52, 340: 656.47,
        365: 97.79,  385: 48.22,  400: 12.5

    We divide each excitation's cube by its exposure time, then multiply
    by the max exposure to keep values in a reasonable range.
    """
    lichens2_exposures = {
        310.0: 5432.9,
        325.0: 924.52,
        340.0: 656.47,
        365.0: 97.79,
        385.0: 48.22,
        400.0: 12.5,
    }
    max_exp = max(lichens2_exposures.values())

    data = copy.deepcopy(raw_data)
    for ex_key in data['data']:
        ex_nm = float(ex_key)
        if ex_nm in lichens2_exposures:
            factor = max_exp / lichens2_exposures[ex_nm]
            data['data'][ex_key]['cube'] = (
                data['data'][ex_key]['cube'].astype(np.float64) * factor
            )
            print(f"  Ex {ex_nm}nm: multiplied by {factor:.2f}x "
                  f"(proxy exposure correction)")
        else:
            print(f"  Ex {ex_nm}nm: no proxy exposure available, unchanged")

    return data


# =============================================================================
# KNN CLASSIFICATION (same as master experiment)
# =============================================================================

def run_knn_classification(full_data, roi_regions, ground_truth,
                           selected_wavelengths=None):
    """Run KNN classification using ROI-based train/test split."""
    excitations = sorted(full_data['data'].keys(), key=float)
    valid_mask = ground_truth >= 0

    if selected_wavelengths is None:
        band_list = []
        for ex in excitations:
            cube = full_data['data'][ex]['cube']
            for b in range(cube.shape[2]):
                band_list.append(cube[:, :, b])
    else:
        band_list = []
        for wl in selected_wavelengths:
            ex_key = str(wl['excitation_nm'])
            band_idx = wl['emission_band_index']
            cube = full_data['data'][ex_key]['cube']
            band_list.append(cube[:, :, band_idx])

    features = np.stack(band_list, axis=-1)
    n_features = features.shape[2]

    roi_mask = np.zeros_like(ground_truth, dtype=bool)
    for roi in roi_regions:
        r_min, r_max, c_min, c_max = roi['coords']
        roi_mask[r_min:r_max, c_min:c_max] = True

    train_mask = valid_mask & roi_mask
    test_mask = valid_mask & ~roi_mask

    X_train = features[train_mask].reshape(-1, n_features)
    y_train = ground_truth[train_mask]
    X_test = features[test_mask].reshape(-1, n_features)
    y_test = ground_truth[test_mask]

    train_valid = ~np.any(np.isnan(X_train), axis=1)
    test_valid = ~np.any(np.isnan(X_test), axis=1)
    X_train, y_train = X_train[train_valid], y_train[train_valid]
    X_test, y_test = X_test[test_valid], y_test[test_valid]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'ari': adjusted_rand_score(y_test, y_pred),
        'nmi': normalized_mutual_info_score(y_test, y_pred),
        'n_train': len(y_train),
        'n_test': len(y_test),
    }
    return metrics


def run_single_config(spectra_data, full_data, roi_regions, ground_truth,
                      model_path, n_bands, dim_method, n_dims,
                      perturb_method, norm_method, mag_list,
                      output_dir, retrain=False, epochs=60):
    """Run a single wavelength selection + classification experiment."""
    config = Config(
        sample_name="Sponges_Acid",
        model_path=str(model_path),
        output_dir=str(output_dir),
        n_bands_to_select=n_bands,
        dimension_selection_method=dim_method,
        n_important_dimensions=n_dims,
        perturbation_method=perturb_method,
        perturbation_magnitudes=mag_list,
        normalization_method=norm_method,
        use_diversity_constraint=True,
        diversity_method="mmr",
        lambda_diversity=0.5,
        min_distance_nm=15.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_seed=42,
        training_epochs=epochs,
    )

    analyzer = Analyzer(config)
    analyzer.fit(spectra_data)
    bands = analyzer.get_wavelengths()

    selected_wls = [b.to_dict() for b in bands]
    metrics = run_knn_classification(
        full_data, roi_regions, ground_truth, selected_wls
    )

    return metrics, selected_wls


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_more_bands(spectra_data, full_data, roi_regions, ground_truth,
                          model_path, output_dir):
    """Experiment 1: Try higher band counts to see where the gap closes."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: More Bands (80, 100, 130, 149)")
    print("=" * 70)

    results = []
    for n_bands in [80, 100, 130, 149]:
        for norm in ["none", "max_per_excitation"]:
            name = f"bands_{n_bands}_pca_dim3_{norm}"
            print(f"\n[{name}]")
            exp_dir = output_dir / name
            exp_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            metrics, wls = run_single_config(
                spectra_data, full_data, roi_regions, ground_truth,
                model_path=model_path,
                n_bands=n_bands, dim_method="pca", n_dims=3,
                perturb_method="absolute_range", norm_method=norm,
                mag_list=[50, 60, 70], output_dir=exp_dir,
            )
            elapsed = time.time() - t0

            metrics['config'] = name
            metrics['n_bands'] = n_bands
            metrics['norm'] = norm
            metrics['time_s'] = elapsed
            results.append(metrics)
            print(f"  Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"Kappa={metrics['kappa']:.4f}  ({elapsed:.1f}s)")

    return results


def experiment_per_excitation_norm(full_data_orig, roi_regions, ground_truth,
                                   output_dir):
    """Experiment 2: Normalize each excitation independently, retrain model."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Per-Excitation Normalization")
    print("  Each excitation cube normalized to [0,1] independently")
    print("  This removes exposure time bias from band selection")
    print("=" * 70)

    norm_data = normalize_per_excitation(full_data_orig)

    norm_pkl = output_dir / "spectra_per_ex_norm.pkl"
    norm_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(norm_pkl, 'wb') as f:
        pickle.dump(norm_data, f)  # nosec — saving our own processed data

    spectra_norm = SpectraData.from_pickle(norm_pkl)
    model_path = output_dir / "model" / "autoencoder_per_ex_norm.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for n_bands in [5, 10, 20, 30, 50, 80, 100]:
        for norm in ["none", "max_per_excitation"]:
            name = f"perex_bands_{n_bands}_{norm}"
            print(f"\n[{name}]")
            exp_dir = output_dir / name
            exp_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            metrics, wls = run_single_config(
                spectra_norm, norm_data, roi_regions, ground_truth,
                model_path=model_path,
                n_bands=n_bands, dim_method="pca", n_dims=3,
                perturb_method="absolute_range", norm_method=norm,
                mag_list=[50, 60, 70], output_dir=exp_dir,
                retrain=False, epochs=60,
            )
            elapsed = time.time() - t0

            metrics['config'] = name
            metrics['n_bands'] = n_bands
            metrics['norm'] = norm
            metrics['preprocess'] = 'per_excitation'
            metrics['time_s'] = elapsed
            results.append(metrics)
            print(f"  Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"Kappa={metrics['kappa']:.4f}  ({elapsed:.1f}s)")

    print(f"\n[BASELINE on per-ex normalized data]")
    baseline = run_knn_classification(norm_data, roi_regions, ground_truth)
    print(f"  Acc={baseline['accuracy']:.4f}  "
          f"F1={baseline['f1']:.4f}  "
          f"Kappa={baseline['kappa']:.4f}")
    baseline['config'] = 'BASELINE_per_ex_norm'
    baseline['n_bands'] = 149
    baseline['norm'] = 'baseline'
    baseline['preprocess'] = 'per_excitation'
    baseline['time_s'] = 0
    results.append(baseline)

    return results


def experiment_proxy_exposure(full_data_orig, roi_regions, ground_truth,
                              output_dir):
    """Experiment 3: Use Lichens_2 exposure ratios as proxy correction."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Proxy Exposure Normalization (Lichens_2 ratios)")
    print("  Same instrument — using known exposure times as correction")
    print("=" * 70)

    proxy_data = normalize_proxy_exposure(full_data_orig)

    proxy_pkl = output_dir / "spectra_proxy_exposure.pkl"
    proxy_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(proxy_pkl, 'wb') as f:
        pickle.dump(proxy_data, f)  # nosec — saving our own processed data

    spectra_proxy = SpectraData.from_pickle(proxy_pkl)
    model_path = output_dir / "model" / "autoencoder_proxy_exp.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for n_bands in [5, 10, 20, 30, 50, 80, 100]:
        for norm in ["none", "max_per_excitation"]:
            name = f"proxy_bands_{n_bands}_{norm}"
            print(f"\n[{name}]")
            exp_dir = output_dir / name
            exp_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            metrics, wls = run_single_config(
                spectra_proxy, proxy_data, roi_regions, ground_truth,
                model_path=model_path,
                n_bands=n_bands, dim_method="pca", n_dims=3,
                perturb_method="absolute_range", norm_method=norm,
                mag_list=[50, 60, 70], output_dir=exp_dir,
                retrain=False, epochs=60,
            )
            elapsed = time.time() - t0

            metrics['config'] = name
            metrics['n_bands'] = n_bands
            metrics['norm'] = norm
            metrics['preprocess'] = 'proxy_exposure'
            metrics['time_s'] = elapsed
            results.append(metrics)
            print(f"  Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"Kappa={metrics['kappa']:.4f}  ({elapsed:.1f}s)")

    print(f"\n[BASELINE on proxy-exposure normalized data]")
    baseline = run_knn_classification(proxy_data, roi_regions, ground_truth)
    print(f"  Acc={baseline['accuracy']:.4f}  "
          f"F1={baseline['f1']:.4f}  "
          f"Kappa={baseline['kappa']:.4f}")
    baseline['config'] = 'BASELINE_proxy_exp'
    baseline['n_bands'] = 149
    baseline['norm'] = 'baseline'
    baseline['preprocess'] = 'proxy_exposure'
    baseline['time_s'] = 0
    results.append(baseline)

    return results


def experiment_fine_sweep(spectra_data, full_data, roi_regions, ground_truth,
                          model_path, output_dir):
    """Experiment 4: Fine-grained sweep around best configs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Fine-Grained Sweep")
    print("  Testing activation dim selection + more dim counts")
    print("=" * 70)

    results = []
    configs = [
        # (n_bands, dim_method, n_dims, perturb, norm)
        # Try activation method (not tested in original run)
        (10, "activation", 5, "percentile", "none"),
        (20, "activation", 5, "percentile", "none"),
        (30, "activation", 5, "percentile", "none"),
        (50, "activation", 5, "percentile", "none"),
        (10, "activation", 10, "percentile", "none"),
        (20, "activation", 10, "percentile", "none"),
        (30, "activation", 10, "percentile", "none"),
        (50, "activation", 10, "percentile", "none"),
        # Try higher n_dims with PCA
        (10, "pca", 5, "absolute_range", "none"),
        (20, "pca", 5, "absolute_range", "none"),
        (30, "pca", 5, "absolute_range", "none"),
        (50, "pca", 5, "absolute_range", "none"),
        (10, "pca", 10, "absolute_range", "none"),
        (20, "pca", 10, "absolute_range", "none"),
        (30, "pca", 10, "absolute_range", "none"),
        (50, "pca", 10, "absolute_range", "none"),
        # Medium magnitudes with best config
        (10, "pca", 3, "absolute_range", "none"),
        (20, "pca", 3, "absolute_range", "none"),
        (30, "pca", 3, "absolute_range", "none"),
        (50, "pca", 3, "absolute_range", "none"),
    ]

    mag_list = [50, 60, 70]

    for n_bands, dim_m, n_dims, pert, norm in configs:
        name = f"fine_{n_bands}_{dim_m[:3]}_d{n_dims}_{pert[:4]}_{norm[:3]}"
        print(f"\n[{name}]")
        exp_dir = output_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        metrics, wls = run_single_config(
            spectra_data, full_data, roi_regions, ground_truth,
            model_path=model_path,
            n_bands=n_bands, dim_method=dim_m, n_dims=n_dims,
            perturb_method=pert, norm_method=norm,
            mag_list=mag_list, output_dir=exp_dir,
        )
        elapsed = time.time() - t0

        metrics['config'] = name
        metrics['n_bands'] = n_bands
        metrics['dim_method'] = dim_m
        metrics['n_dims'] = n_dims
        metrics['norm'] = norm
        metrics['time_s'] = elapsed
        results.append(metrics)
        print(f"  Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  "
              f"Kappa={metrics['kappa']:.4f}  ({elapsed:.1f}s)")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collagen dataset tuning experiments',
    )
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'more_bands', 'per_excitation_norm',
                                 'proxy_exposure', 'fine_sweep'],
                        help='Which experiment to run')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to existing model (skip retraining)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data_file = DATA_DIR / "spectra_masked.pkl"
    mask_file = DATA_DIR / "class_mask.png"
    roi_file = DATA_DIR / "roi_regions.json"

    full_data, spectra_data = load_data(data_file)
    ground_truth, roi_regions, class_info = load_ground_truth(mask_file, roi_file)

    # Baseline
    print("\n" + "=" * 70)
    print("BASELINE (all 149 bands, raw data)")
    print("=" * 70)
    baseline = run_knn_classification(full_data, roi_regions, ground_truth)
    print(f"  Accuracy: {baseline['accuracy']:.4f}")
    print(f"  F1:       {baseline['f1']:.4f}")
    print(f"  Kappa:    {baseline['kappa']:.4f}")

    # Model path
    if args.model:
        model_path = Path(args.model)
    else:
        prev_model = PROJECT_ROOT / "results" / "Sponges_Acid_Group1" / "model" / "autoencoder_model.pth"
        alt_models = list((PROJECT_ROOT / "results").rglob("*Sponges*/**/autoencoder_model.pth"))
        if prev_model.exists():
            model_path = prev_model
        elif alt_models:
            model_path = alt_models[0]
        else:
            model_path = output_dir / "model" / "autoencoder_model.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nModel: {model_path}")

    all_results = [
        {**baseline, 'config': 'BASELINE_raw', 'n_bands': 149,
         'norm': 'baseline', 'time_s': 0}
    ]

    # Run requested experiments
    experiments = {
        'more_bands': lambda: experiment_more_bands(
            spectra_data, full_data, roi_regions, ground_truth,
            model_path, output_dir / "exp1_more_bands"),
        'per_excitation_norm': lambda: experiment_per_excitation_norm(
            full_data, roi_regions, ground_truth,
            output_dir / "exp2_per_excitation"),
        'proxy_exposure': lambda: experiment_proxy_exposure(
            full_data, roi_regions, ground_truth,
            output_dir / "exp3_proxy_exposure"),
        'fine_sweep': lambda: experiment_fine_sweep(
            spectra_data, full_data, roi_regions, ground_truth,
            model_path, output_dir / "exp4_fine_sweep"),
    }

    if args.experiment == 'all':
        to_run = list(experiments.keys())
    else:
        to_run = [args.experiment]

    for exp_name in to_run:
        results = experiments[exp_name]()
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — ALL RESULTS")
    print("=" * 70)

    df = pd.DataFrame(all_results)
    cols = ['config', 'n_bands', 'accuracy', 'f1', 'kappa', 'time_s']
    available_cols = [c for c in cols if c in df.columns]
    df_sorted = df[available_cols].sort_values('accuracy', ascending=False)
    print(df_sorted.to_string(index=False))

    # Save
    csv_path = output_dir / "tuning_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Top 5
    print("\n" + "-" * 70)
    print("TOP 5 CONFIGURATIONS:")
    print("-" * 70)
    for i, row in df_sorted.head(5).iterrows():
        print(f"  {row['config']:50s}  "
              f"Acc={row['accuracy']:.4f}  F1={row['f1']:.4f}")

    baseline_acc = baseline['accuracy']
    best_acc = df_sorted.iloc[0]['accuracy']
    print(f"\nBaseline: {baseline_acc:.4f}")
    print(f"Best:     {best_acc:.4f} ({best_acc - baseline_acc:+.4f})")


if __name__ == "__main__":
    main()
