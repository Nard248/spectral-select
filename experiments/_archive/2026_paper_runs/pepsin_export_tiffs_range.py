#!/usr/bin/env python3
"""
Pepsin — TIFF Export for n_bands = 5..30
===========================================
Runs wavelength selection at every band count in [5, 30] using the existing
trained autoencoder and a fixed best-known parameter set, then exports each
selection as an ImageJ-compatible multi-page TIFF stack plus individual
per-band TIFFs. Also runs the same ROI-based KNN evaluation used in the
pipeline so we know each export's accuracy.

The project uses pickle for its scientific .pkl format (SpectraData).
"""

import json
import pickle
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tifffile
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

warnings.filterwarnings('ignore')

import random
np.random.seed(42); random.seed(42); torch.manual_seed(42)

from spectral_select import Analyzer, Config, SpectraData

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "processed" / "Collagen Pepsin"
DATA_MASKED_PKL = DATA_DIR / "spectra_masked.pkl"
DATA_UNMASKED_PKL = DATA_DIR / "spectra_unmasked.pkl"
MASK_PNG = DATA_DIR / "class_mask.png"
ROI_JSON = DATA_DIR / "roi_regions.json"

PIPELINE_DIR = PROJECT_ROOT / "results" / "Collagen_Pepsin_Normalized"
MODEL_PATH = PIPELINE_DIR / "model" / "autoencoder_model.pth"
TIFF_OUT_DIR = PIPELINE_DIR / "exported_tiffs_5_to_30"
TIFF_OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_PARAMS = dict(
    dimension_selection_method="pca",
    n_important_dimensions=3,
    perturbation_method="absolute_range",
    perturbation_magnitudes=[30, 40, 50],
    normalization_method="max_per_excitation",
    use_diversity_constraint=True,
    diversity_method="mmr",
    lambda_diversity=0.5,
    min_distance_nm=15.0,
    device="cpu",
    random_seed=42,
    training_epochs=60,
)

BAND_COUNTS = list(range(5, 31))  # 5..30 inclusive = 26 counts


def load_full_data():
    with open(DATA_MASKED_PKL, 'rb') as f:
        raw = pickle.load(f)
    spectra = SpectraData.from_pickle(DATA_MASKED_PKL)
    full = {'data': {}, 'excitation_wavelengths': raw['excitation_wavelengths'],
            'mask': raw.get('mask')}
    for ex_nm, ex_d in spectra.excitations.items():
        wls = ex_d.emission_wavelengths
        if hasattr(wls, 'tolist'):
            wls = wls.tolist()
        full['data'][str(ex_nm)] = {'cube': ex_d.cube, 'wavelengths': wls}
    return full, spectra


def load_unmasked_data():
    path = DATA_UNMASKED_PKL if DATA_UNMASKED_PKL.exists() else DATA_MASKED_PKL
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_ground_truth():
    arr = np.array(Image.open(MASK_PNG))
    with open(ROI_JSON) as f:
        rd = json.load(f)
    gt = np.full(arr.shape[:2], -1, dtype=int)
    for cls in rd['classes']:
        color = tuple(cls['color'])
        m = (np.all(arr[:, :, :3] == color, axis=2)
             if arr.shape[-1] >= 3 else np.all(arr == color, axis=2))
        gt[m] = cls['id']
    rois = []
    for roi in rd['regions']:
        r = roi['rect']
        rois.append({
            'name': roi['class_name'],
            'class_id': roi['class_id'],
            'coords': (r['row_min'], r['row_max'], r['col_min'], r['col_max']),
        })
    return gt, rois


def run_selection(spectra, n_bands, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(sample_name="Collagen_Pepsin", model_path=str(MODEL_PATH),
                 output_dir=str(out_dir), n_bands_to_select=n_bands,
                 **FIXED_PARAMS)
    analyzer = Analyzer(cfg)
    analyzer.fit(spectra)
    return [b.to_dict() for b in analyzer.get_wavelengths()]


def evaluate_knn(full_data, rois, gt, selected):
    excitations = sorted(full_data['data'].keys(), key=float)
    valid_mask = gt >= 0
    if selected is None:
        bands = [full_data['data'][ex]['cube'][:, :, b]
                 for ex in excitations
                 for b in range(full_data['data'][ex]['cube'].shape[2])]
    else:
        bands = [full_data['data'][str(wl['excitation_nm'])]['cube'][:, :, wl['emission_band_index']]
                 for wl in selected]
    feat = np.stack(bands, axis=-1)
    roi_mask = np.zeros_like(gt, dtype=bool)
    for roi in rois:
        r0, r1, c0, c1 = roi['coords']
        roi_mask[r0:r1, c0:c1] = True
    Xtr = feat[valid_mask & roi_mask].reshape(-1, feat.shape[2])
    ytr = gt[valid_mask & roi_mask]
    Xte = feat[valid_mask & ~roi_mask].reshape(-1, feat.shape[2])
    yte = gt[valid_mask & ~roi_mask]
    ok_tr = ~np.any(np.isnan(Xtr), axis=1)
    ok_te = ~np.any(np.isnan(Xte), axis=1)
    Xtr, ytr = Xtr[ok_tr], ytr[ok_tr]
    Xte, yte = Xte[ok_te], yte[ok_te]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(Xtr, ytr)
    yp = knn.predict(Xte)
    return {'accuracy': accuracy_score(yte, yp),
            'f1': f1_score(yte, yp, average='weighted'),
            'kappa': cohen_kappa_score(yte, yp)}


def extract_slice(raw_data, ex_nm, em_nm):
    ex_key = None
    for k in raw_data['data'].keys():
        if abs(float(k) - ex_nm) < 1e-3:
            ex_key = k; break
    if ex_key is None:
        return None
    cube = raw_data['data'][ex_key]['cube']
    wls = raw_data['data'][ex_key]['wavelengths']
    if hasattr(wls, 'tolist'):
        wls = wls.tolist()
    for i, w in enumerate(wls):
        if abs(w - em_nm) < 1.0:
            return cube[:, :, i]
    em_sn = int(em_nm // 10) * 10
    for i, w in enumerate(wls):
        if int(w // 10) * 10 == em_sn:
            return cube[:, :, i]
    return None


def export_tiffs(n_bands, selected, raw_unmasked, metrics):
    label = f"{n_bands:02d}_bands"
    slices, slice_labels, recs = [], [], []
    for wl in sorted(selected, key=lambda w: w['rank']):
        ex, em, rank = float(wl['excitation_nm']), float(wl['emission_nm']), wl['rank']
        img = extract_slice(raw_unmasked, ex, em)
        if img is None:
            print(f"    WARNING: missing Ex={ex} Em={em}")
            continue
        slices.append(img)
        slice_labels.append(f"Rank{rank}_Ex{int(ex)}_Em{int(em)}")
        recs.append({'rank': int(rank), 'excitation': ex, 'emission': em})
    if not slices:
        return None
    stack = np.stack(slices, axis=0).astype(np.float32)
    stack = np.nan_to_num(stack, nan=0.0)
    combined = TIFF_OUT_DIR / f"selected_{label}.tif"
    tifffile.imwrite(str(combined), stack, imagej=True,
                     metadata={'Labels': slice_labels},
                     photometric='minisblack')
    indiv = TIFF_OUT_DIR / f"selected_{label}_individual"
    indiv.mkdir(parents=True, exist_ok=True)
    for lbl, img in zip(slice_labels, stack):
        tifffile.imwrite(str(indiv / f"{lbl}.tif"), img,
                         imagej=True, photometric='minisblack')
    return {'n_bands': n_bands,
            'accuracy': metrics['accuracy'], 'f1': metrics['f1'],
            'kappa': metrics['kappa'],
            'combined_tiff': str(combined.relative_to(PROJECT_ROOT)),
            'individual_dir': str(indiv.relative_to(PROJECT_ROOT)),
            'wavelengths': recs}


def main():
    print("=" * 86)
    print("PEPSIN — TIFF EXPORT for n_bands = 5..30")
    print("=" * 86)
    print(f"  Model:  {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Output: {TIFF_OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("  Fixed params: pca / dim3 / absolute_range / max_per_ex / mmr(0.5)")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    print("\n  Loading data...")
    full_data, spectra = load_full_data()
    raw_unmasked = load_unmasked_data()
    gt, rois = load_ground_truth()

    print("\n  Baseline (all bands, KNN k=5)...")
    bl = evaluate_knn(full_data, rois, gt, None)
    print(f"    Acc={bl['accuracy']:.2%}  F1={bl['f1']:.2%}  Kappa={bl['kappa']:.4f}")

    manifest = []
    work = TIFF_OUT_DIR / "_workspace"
    work.mkdir(parents=True, exist_ok=True)

    print(f"\n  Running {len(BAND_COUNTS)} selections (n={BAND_COUNTS[0]}..{BAND_COUNTS[-1]})...")
    for n in BAND_COUNTS:
        t0 = time.time()
        sel = run_selection(spectra, n, work / f"sel_{n:02d}")
        m = evaluate_knn(full_data, rois, gt, sel)
        rec = export_tiffs(n, sel, raw_unmasked, m)
        d = m['accuracy'] - bl['accuracy']
        mark = "**" if d > 0 else "  "
        print(f"    n={n:2d}  Acc={m['accuracy']:.2%}  F1={m['f1']:.2%}  "
              f"Kappa={m['kappa']:.4f}  vs bl {d:+.2%} {mark}  ({time.time()-t0:.1f}s)")
        if rec:
            manifest.append(rec)

    mf = TIFF_OUT_DIR / "EXPORT_MANIFEST.json"
    with open(mf, 'w') as f:
        json.dump({'baseline': bl,
                   'fixed_params': {k: str(v) for k, v in FIXED_PARAMS.items()},
                   'band_counts_exported': BAND_COUNTS,
                   'exports': manifest}, f, indent=2)

    summ = pd.DataFrame([{
        'n_bands': r['n_bands'], 'accuracy': r['accuracy'],
        'f1': r['f1'], 'kappa': r['kappa'],
        'delta_vs_baseline': r['accuracy'] - bl['accuracy'],
        'reduction_pct': (1 - r['n_bands'] / 158) * 100,
        'combined_tiff': r['combined_tiff']} for r in manifest])
    sc = TIFF_OUT_DIR / "EXPORT_SUMMARY.csv"
    summ.to_csv(sc, index=False)

    print(f"\n  Manifest: {mf.relative_to(PROJECT_ROOT)}")
    print(f"  Summary:  {sc.relative_to(PROJECT_ROOT)}")
    print(f"  {len(manifest)} TIFF stacks + "
          f"{sum(len(r['wavelengths']) for r in manifest)} individual TIFFs")

    print("\n  Top 10 by accuracy:")
    for i, r in enumerate(sorted(manifest, key=lambda r: -r['accuracy'])[:10], 1):
        print(f"    {i:>2}. n={r['n_bands']:2d}  Acc={r['accuracy']:.2%}  "
              f"F1={r['f1']:.2%}  Kappa={r['kappa']:.4f}")

    print("\n" + "=" * 86)
    print("DONE")
    print("=" * 86)


if __name__ == '__main__':
    main()
