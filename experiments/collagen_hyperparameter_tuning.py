#!/usr/bin/env python3
"""
Collagen Hyperparameter Tuning Experiments
============================================
Systematic exploration of autoencoder architecture, training, patch extraction,
KNN classifier, and diversity constraint parameters on the Collagen Acetic Acid
dataset (normalized with exposure + lamp power correction).

This script does NOT modify any existing code or results. All experiments
write to their own output directories under results/Collagen_Hyperparam_Tuning/.

Axes explored:
  1. Autoencoder capacity (k1, k3 filter counts) — requires retraining
  2. Training duration & regularization (epochs, dropout, sparsity) — requires retraining
  3. Baseline patch extraction (patch_size, n_patches) — reuses existing model
  4. KNN classifier parameters (k, weights) — reuses existing model + bands
  5. Diversity constraint tuning (lambda, min_distance) — reuses existing model

All axes use the normalized (exposure + power corrected) Collagen Acetic Acid
data from Data/processed/Collagen_Acetic_Acid/spectra_masked.pkl.

Pickle is used for loading the project's scientific hyperspectral data format.

Usage:
    python experiments/collagen_hyperparameter_tuning.py --axis all
    python experiments/collagen_hyperparameter_tuning.py --axis autoencoder
    python experiments/collagen_hyperparameter_tuning.py --axis training
    python experiments/collagen_hyperparameter_tuning.py --axis patches
    python experiments/collagen_hyperparameter_tuning.py --axis knn
    python experiments/collagen_hyperparameter_tuning.py --axis diversity
"""

import json
import argparse
import pickle  # required for SpectraData .pkl format
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List
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
    accuracy_score, f1_score,
    cohen_kappa_score,
)
from PIL import Image

DATA_DIR = PROJECT_ROOT / "Data" / "processed" / "Collagen_Acetic_Acid"
OUTPUT_BASE = PROJECT_ROOT / "results" / "Collagen_Hyperparam_Tuning"
TEST_BANDS = [5, 10, 15, 20]


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    spectra = SpectraData.from_pickle(pkl_path)
    full_data = {
        'data': {}, 'excitation_wavelengths': raw['excitation_wavelengths'],
        'mask': raw.get('mask'),
    }
    for ex_nm, ex_data in spectra.excitations.items():
        wls = ex_data.emission_wavelengths
        if hasattr(wls, 'tolist'):
            wls = wls.tolist()
        full_data['data'][str(ex_nm)] = {'cube': ex_data.cube, 'wavelengths': wls}
    return full_data, spectra


def load_ground_truth(mask_path, roi_path):
    mask_array = np.array(Image.open(mask_path))
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
            'name': roi['class_name'], 'class_id': roi['class_id'],
            'coords': (rect['row_min'], rect['row_max'],
                       rect['col_min'], rect['col_max']),
        })
    return ground_truth, roi_regions, class_info


def run_knn(full_data, roi_regions, ground_truth,
            selected_wavelengths=None, n_neighbors=5, weights='uniform'):
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
            band_list.append(full_data['data'][ex_key]['cube'][:, :, band_idx])
    features = np.stack(band_list, axis=-1)
    n_feat = features.shape[2]
    roi_mask = np.zeros_like(ground_truth, dtype=bool)
    for roi in roi_regions:
        r0, r1, c0, c1 = roi['coords']
        roi_mask[r0:r1, c0:c1] = True
    X_tr = features[valid_mask & roi_mask].reshape(-1, n_feat)
    y_tr = ground_truth[valid_mask & roi_mask]
    X_te = features[valid_mask & ~roi_mask].reshape(-1, n_feat)
    y_te = ground_truth[valid_mask & ~roi_mask]
    ok_tr = ~np.any(np.isnan(X_tr), axis=1)
    ok_te = ~np.any(np.isnan(X_te), axis=1)
    X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]
    X_te, y_te = X_te[ok_te], y_te[ok_te]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    return {
        'accuracy': accuracy_score(y_te, y_pred),
        'f1': f1_score(y_te, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_te, y_pred),
    }


def select_bands(spectra_data, config):
    analyzer = Analyzer(config)
    analyzer.fit(spectra_data)
    return [b.to_dict() for b in analyzer.get_wavelengths()]


def _find_model():
    for p in [
        PROJECT_ROOT / "results" / "Collagen_Acetic_Acid_Normalized" / "model" / "autoencoder_model.pth",
        PROJECT_ROOT / "results" / "Collagen_Acetic_Acid_LowBands" / "model" / "autoencoder_model.pth",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("No existing model. Run --axis autoencoder first.")


# ── AXIS 1: Autoencoder Capacity ──────────────────────────────────────────

def axis_autoencoder(spectra, full, roi, gt, out):
    print("\n" + "=" * 70)
    print("AXIS 1: Autoencoder Capacity (k1, k3)")
    print("=" * 70)
    cfgs = [(10, 10, "small"), (20, 20, "default"), (40, 40, "large"), (20, 40, "bottleneck")]
    results = []
    for k1, k3, label in cfgs:
        model_path = out / f"model_k1{k1}_k3{k3}.pth"
        for nb in TEST_BANDS:
            name = f"cap_{label}_b{nb}"
            print(f"\n  [{name}] k1={k1}, k3={k3}")
            d = out / name; d.mkdir(parents=True, exist_ok=True)
            cfg = Config(
                sample_name="Collagen_Acid", model_path=str(model_path),
                output_dir=str(d), n_bands_to_select=nb,
                dimension_selection_method="pca", n_important_dimensions=1,
                perturbation_method="absolute_range",
                perturbation_magnitudes=[50, 60, 70],
                normalization_method="max_per_excitation",
                use_diversity_constraint=True, diversity_method="mmr",
                lambda_diversity=0.5, min_distance_nm=15.0,
                device="cpu", random_seed=42, training_epochs=60,
                model_k1=k1, model_k3=k3,
            )
            t0 = time.time()
            wls = select_bands(spectra, cfg)
            m = run_knn(full, roi, gt, wls)
            dt = time.time() - t0
            m.update({'config': name, 'n_bands': nb, 'k1': k1, 'k3': k3,
                      'label': label, 'axis': 'autoencoder', 'time_s': dt})
            results.append(m)
            print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  ({dt:.1f}s)")
    return results


# ── AXIS 2: Training Duration & Regularization ────────────────────────────

def axis_training(spectra, full, roi, gt, out):
    print("\n" + "=" * 70)
    print("AXIS 2: Training Duration & Regularization")
    print("=" * 70)
    cfgs = [
        (60,  0.5, 1.0, "60ep_d05_s10"),
        (100, 0.5, 1.0, "100ep_d05_s10"),
        (60,  0.2, 1.0, "60ep_d02_s10"),
        (60,  0.3, 1.0, "60ep_d03_s10"),
        (60,  0.5, 0.1, "60ep_d05_s01"),
        (60,  0.5, 2.0, "60ep_d05_s20"),
        (100, 0.2, 0.5, "100ep_d02_s05"),
    ]
    results = []
    for ep, drop, sp_w, label in cfgs:
        model_path = out / f"model_{label}.pth"
        for nb in TEST_BANDS:
            name = f"train_{label}_b{nb}"
            print(f"\n  [{name}] ep={ep}, drop={drop}, sp_w={sp_w}")
            d = out / name; d.mkdir(parents=True, exist_ok=True)
            cfg = Config(
                sample_name="Collagen_Acid", model_path=str(model_path),
                output_dir=str(d), n_bands_to_select=nb,
                dimension_selection_method="pca", n_important_dimensions=1,
                perturbation_method="absolute_range",
                perturbation_magnitudes=[50, 60, 70],
                normalization_method="max_per_excitation",
                use_diversity_constraint=True, diversity_method="mmr",
                lambda_diversity=0.5, min_distance_nm=15.0,
                device="cpu", random_seed=42, training_epochs=ep,
                model_dropout_rate=drop, model_sparsity_weight=sp_w,
            )
            t0 = time.time()
            wls = select_bands(spectra, cfg)
            m = run_knn(full, roi, gt, wls)
            dt = time.time() - t0
            m.update({'config': name, 'n_bands': nb, 'epochs': ep,
                      'dropout': drop, 'sparsity_weight': sp_w,
                      'label': label, 'axis': 'training', 'time_s': dt})
            results.append(m)
            print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  ({dt:.1f}s)")
    return results


# ── AXIS 3: Patch Extraction ──────────────────────────────────────────────

def axis_patches(spectra, full, roi, gt, out, model=None):
    print("\n" + "=" * 70)
    print("AXIS 3: Baseline Patch Extraction")
    print("=" * 70)
    mp = model or _find_model()
    cfgs = [
        (8,  100, 4,  "tiny_8px"),
        (16, 75,  8,  "small_16px"),
        (32, 50,  16, "default_32px"),
        (16, 50,  8,  "small_fewer"),
        (8,  200, 4,  "tiny_many"),
    ]
    results = []
    for ps, np_, stride, label in cfgs:
        for nb in TEST_BANDS:
            name = f"patch_{label}_b{nb}"
            print(f"\n  [{name}] size={ps}, n={np_}, stride={stride}")
            d = out / name; d.mkdir(parents=True, exist_ok=True)
            cfg = Config(
                sample_name="Collagen_Acid", model_path=str(mp),
                output_dir=str(d), n_bands_to_select=nb,
                dimension_selection_method="pca", n_important_dimensions=1,
                perturbation_method="absolute_range",
                perturbation_magnitudes=[50, 60, 70],
                normalization_method="max_per_excitation",
                use_diversity_constraint=True, diversity_method="mmr",
                lambda_diversity=0.5, min_distance_nm=15.0,
                device="cpu", random_seed=42,
                n_baseline_patches=np_, patch_size=ps, patch_stride=stride,
                training_epochs=60,
            )
            t0 = time.time()
            wls = select_bands(spectra, cfg)
            m = run_knn(full, roi, gt, wls)
            dt = time.time() - t0
            m.update({'config': name, 'n_bands': nb, 'patch_size': ps,
                      'n_patches': np_, 'label': label,
                      'axis': 'patches', 'time_s': dt})
            results.append(m)
            print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  ({dt:.1f}s)")
    return results


# ── AXIS 4: KNN Classifier ────────────────────────────────────────────────

def axis_knn(spectra, full, roi, gt, out, model=None):
    print("\n" + "=" * 70)
    print("AXIS 4: KNN Classifier Parameters")
    print("=" * 70)
    mp = model or _find_model()
    knn_cfgs = [
        (3, 'uniform'), (3, 'distance'),
        (5, 'uniform'), (5, 'distance'),
        (7, 'uniform'), (7, 'distance'),
        (11, 'uniform'), (11, 'distance'),
    ]
    results = []
    print("\n  Baselines (all bands):")
    for k, w in knn_cfgs:
        m = run_knn(full, roi, gt, n_neighbors=k, weights=w)
        label = f"k{k}_{w}"
        m.update({'config': f"BASELINE_{label}", 'n_bands': 149,
                  'k': k, 'weights': w, 'axis': 'knn', 'time_s': 0})
        results.append(m)
        print(f"    BASELINE k={k:2d} {w:8s}: Acc={m['accuracy']:.4f}")

    for nb in TEST_BANDS:
        d = out / f"sel_b{nb}"; d.mkdir(parents=True, exist_ok=True)
        cfg = Config(
            sample_name="Collagen_Acid", model_path=str(mp),
            output_dir=str(d), n_bands_to_select=nb,
            dimension_selection_method="pca", n_important_dimensions=1,
            perturbation_method="absolute_range",
            perturbation_magnitudes=[50, 60, 70],
            normalization_method="max_per_excitation",
            use_diversity_constraint=True, diversity_method="mmr",
            lambda_diversity=0.5, min_distance_nm=15.0,
            device="cpu", random_seed=42, training_epochs=60,
        )
        print(f"\n  Selecting {nb} bands...")
        wls = select_bands(spectra, cfg)
        for k, w in knn_cfgs:
            label = f"k{k}_{w}"
            name = f"knn_{label}_b{nb}"
            m = run_knn(full, roi, gt, wls, n_neighbors=k, weights=w)
            m.update({'config': name, 'n_bands': nb, 'k': k, 'weights': w,
                      'axis': 'knn', 'time_s': 0})
            results.append(m)
            print(f"    k={k:2d} {w:8s}: Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    return results


# ── AXIS 5: Diversity Constraints ─────────────────────────────────────────

def axis_diversity(spectra, full, roi, gt, out, model=None):
    print("\n" + "=" * 70)
    print("AXIS 5: Diversity Constraint Tuning")
    print("=" * 70)
    mp = model or _find_model()
    cfgs = [
        (False, "none",         0.0, 0.0,  "no_diversity"),
        (True,  "mmr",          0.2, 15.0, "mmr_02"),
        (True,  "mmr",          0.3, 15.0, "mmr_03"),
        (True,  "mmr",          0.5, 15.0, "mmr_05"),
        (True,  "mmr",          0.7, 15.0, "mmr_07"),
        (True,  "min_distance", 0.5, 10.0, "mindist_10"),
        (True,  "min_distance", 0.5, 20.0, "mindist_20"),
        (True,  "min_distance", 0.5, 30.0, "mindist_30"),
    ]
    results = []
    for use_div, method, lam, min_d, label in cfgs:
        for nb in TEST_BANDS:
            name = f"div_{label}_b{nb}"
            print(f"\n  [{name}]")
            d = out / name; d.mkdir(parents=True, exist_ok=True)
            cfg = Config(
                sample_name="Collagen_Acid", model_path=str(mp),
                output_dir=str(d), n_bands_to_select=nb,
                dimension_selection_method="pca", n_important_dimensions=1,
                perturbation_method="absolute_range",
                perturbation_magnitudes=[50, 60, 70],
                normalization_method="max_per_excitation",
                use_diversity_constraint=use_div,
                diversity_method=method if use_div else "mmr",
                lambda_diversity=lam if use_div else 0.5,
                min_distance_nm=min_d if use_div else 15.0,
                device="cpu", random_seed=42, training_epochs=60,
            )
            t0 = time.time()
            wls = select_bands(spectra, cfg)
            m = run_knn(full, roi, gt, wls)
            dt = time.time() - t0
            m.update({'config': name, 'n_bands': nb, 'use_diversity': use_div,
                      'div_method': method, 'lambda': lam, 'min_dist': min_d,
                      'label': label, 'axis': 'diversity', 'time_s': dt})
            results.append(m)
            print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  ({dt:.1f}s)")
    return results


# ── SUMMARY & MAIN ────────────────────────────────────────────────────────

def print_summary(all_results, output_dir):
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("BEST RESULTS PER AXIS AND BAND COUNT")
    print("=" * 70)
    for axis_name in sorted(df['axis'].unique()):
        adf = df[df['axis'] == axis_name]
        print(f"\n--- {axis_name.upper()} ---")
        best = adf.loc[adf.groupby('n_bands')['accuracy'].idxmax()].sort_values('n_bands')
        for _, r in best.iterrows():
            print(f"  {int(r['n_bands']):3d} bands | Acc={r['accuracy']:.4f} | "
                  f"F1={r['f1']:.4f} | {r['config']}")

    print("\n" + "=" * 70)
    print("OVERALL TOP 15 (excluding baselines)")
    print("=" * 70)
    non_bl = df[~df['config'].str.contains('BASELINE')]
    for _, r in non_bl.nlargest(15, 'accuracy').iterrows():
        print(f"  {int(r['n_bands']):3d} bands | Acc={r['accuracy']:.4f} | "
              f"F1={r['f1']:.4f} | [{r['axis']}] {r['config']}")

    csv_path = output_dir / "hyperparameter_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Collagen hyperparameter tuning')
    parser.add_argument('--axis', type=str, default='all',
                        choices=['all', 'autoencoder', 'training', 'patches',
                                 'knn', 'diversity'])
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_BASE / ts
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    full, spectra = load_data(DATA_DIR / "spectra_masked.pkl")
    gt, roi, ci = load_ground_truth(DATA_DIR / "class_mask.png",
                                     DATA_DIR / "roi_regions.json")

    print("\n" + "=" * 70)
    print("BASELINE (all 149 bands, k=5 uniform)")
    print("=" * 70)
    bl = run_knn(full, roi, gt)
    print(f"  Acc={bl['accuracy']:.4f}  F1={bl['f1']:.4f}  Kappa={bl['kappa']:.4f}")

    existing_model = Path(args.model) if args.model else None
    all_results = [{**bl, 'config': 'BASELINE', 'n_bands': 149,
                    'axis': 'baseline', 'time_s': 0}]

    dispatch = {
        'autoencoder': lambda: axis_autoencoder(spectra, full, roi, gt,
                                                 out / "axis1_autoencoder"),
        'training':    lambda: axis_training(spectra, full, roi, gt,
                                              out / "axis2_training"),
        'patches':     lambda: axis_patches(spectra, full, roi, gt,
                                             out / "axis3_patches", existing_model),
        'knn':         lambda: axis_knn(spectra, full, roi, gt,
                                         out / "axis4_knn", existing_model),
        'diversity':   lambda: axis_diversity(spectra, full, roi, gt,
                                              out / "axis5_diversity", existing_model),
    }

    to_run = list(dispatch.keys()) if args.axis == 'all' else [args.axis]
    for name in to_run:
        all_results.extend(dispatch[name]())

    print_summary(all_results, out)


if __name__ == "__main__":
    main()
