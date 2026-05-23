#!/usr/bin/env python3
"""
Collagen Classifier Comparison & Expanded ROI Experiment
=========================================================
Tests multiple classifiers (KNN, SVM, Random Forest, Gradient Boosting,
LDA, MLP) with both original small ROIs and expanded ROIs covering all
9 sponge positions.

Goal: Determine if the 83.1% baseline ceiling is due to KNN limitations
or inherent in the data.

Two ROI versions tested:
  - Original: 3 small boxes from top row only (1,628 px, 3.5%)
  - Expanded: 9 large boxes from all rows (38,843 px, 83.2%)

This project uses pickle for its scientific hyperspectral data format (.pkl).
All pickle files are generated and consumed locally.

Usage:
    python experiments/collagen_classifier_comparison.py
    python experiments/collagen_classifier_comparison.py --roi original
    python experiments/collagen_classifier_comparison.py --roi expanded
    python experiments/collagen_classifier_comparison.py --roi both
"""

import json, argparse, pickle, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
np.random.seed(42)
import random; random.seed(42)
import torch; torch.manual_seed(42)

from spectral_select import Config, Analyzer, SpectraData
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from PIL import Image

DATA_DIR = PROJECT_ROOT / "Data" / "processed" / "Collagen_Acetic_Acid"
OUTPUT_BASE = PROJECT_ROOT / "results" / "Collagen_Classifier_Comparison"
TEST_BANDS = [3, 5, 8, 10, 15, 20, 30, 50]

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:  # nosec - local scientific data
        raw = pickle.load(f)
    spectra = SpectraData.from_pickle(pkl_path)
    fd = {'data': {}, 'excitation_wavelengths': raw['excitation_wavelengths'],
          'mask': raw.get('mask')}
    for ex_nm, exd in spectra.excitations.items():
        wls = exd.emission_wavelengths
        if hasattr(wls, 'tolist'): wls = wls.tolist()
        fd['data'][str(ex_nm)] = {'cube': exd.cube, 'wavelengths': wls}
    return fd, spectra

def load_ground_truth(mask_path, roi_path):
    ma = np.array(Image.open(mask_path))
    with open(roi_path, 'r') as f: rd = json.load(f)
    gt = np.full(ma.shape[:2], -1, dtype=int)
    for cls in rd['classes']:
        c = tuple(cls['color'])
        m = np.all(ma[:,:,:3] == c, axis=2) if ma.shape[-1]==4 else np.all(ma==c, axis=2)
        gt[m] = cls['id']
    rois = []
    for roi in rd['regions']:
        r = roi['rect']
        rois.append({'name': roi['class_name'], 'class_id': roi['class_id'],
                     'coords': (r['row_min'], r['row_max'], r['col_min'], r['col_max'])})
    return gt, rois

def extract_features(full_data, gt, rois, sel_wls=None):
    exs = sorted(full_data['data'].keys(), key=float)
    vm = gt >= 0
    if sel_wls is None:
        bl = [full_data['data'][ex]['cube'][:,:,b]
              for ex in exs for b in range(full_data['data'][ex]['cube'].shape[2])]
    else:
        bl = [full_data['data'][str(w['excitation_nm'])]['cube'][:,:,w['emission_band_index']]
              for w in sel_wls]
    feat = np.stack(bl, axis=-1); nf = feat.shape[2]
    rm = np.zeros_like(gt, dtype=bool)
    for roi in rois:
        r0,r1,c0,c1 = roi['coords']; rm[r0:r1, c0:c1] = True
    Xtr = feat[vm & rm].reshape(-1,nf); ytr = gt[vm & rm]
    Xte = feat[vm & ~rm].reshape(-1,nf); yte = gt[vm & ~rm]
    ok_tr = ~np.any(np.isnan(Xtr),1); ok_te = ~np.any(np.isnan(Xte),1)
    Xtr,ytr = Xtr[ok_tr],ytr[ok_tr]; Xte,yte = Xte[ok_te],yte[ok_te]
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    return Xtr, ytr, Xte, yte

def get_classifiers():
    return {
        'KNN_k5':      (lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),),
        'KNN_k11':     (lambda: KNeighborsClassifier(n_neighbors=11, n_jobs=-1),),
        'KNN_k11_dist':(lambda: KNeighborsClassifier(n_neighbors=11, weights='distance', n_jobs=-1),),
        'SVM_rbf':     (lambda: SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42),),
        'SVM_linear':  (lambda: SVC(kernel='linear', C=1.0, random_state=42),),
        'RF_100':      (lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),),
        'RF_300':      (lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),),
        'GBM':         (lambda: GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                            learning_rate=0.1, random_state=42),),
        'LDA':         (lambda: LinearDiscriminantAnalysis(),),
        'MLP':         (lambda: MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500,
                                              random_state=42, early_stopping=True),),
    }

def run_clf(factory, Xtr, ytr, Xte, yte):
    clf = factory()
    t0 = time.time()
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    return {'accuracy': accuracy_score(yte,yp), 'f1': f1_score(yte,yp,average='weighted'),
            'kappa': cohen_kappa_score(yte,yp), 'train_time_s': time.time()-t0,
            'n_train': len(ytr), 'n_test': len(yte)}

def select_bands(spectra, nb, mp, od):
    od.mkdir(parents=True, exist_ok=True)
    cfg = Config(sample_name="Collagen_Acid", model_path=str(mp), output_dir=str(od),
                 n_bands_to_select=nb, dimension_selection_method="pca",
                 n_important_dimensions=1, perturbation_method="absolute_range",
                 perturbation_magnitudes=[50,60,70], normalization_method="max_per_excitation",
                 use_diversity_constraint=True, diversity_method="mmr",
                 lambda_diversity=0.5, min_distance_nm=15.0, device="cpu",
                 random_seed=42, training_epochs=60)
    a = Analyzer(cfg); a.fit(spectra)
    return [b.to_dict() for b in a.get_wavelengths()]

def find_model():
    for p in [PROJECT_ROOT/"results"/"Collagen_Acetic_Acid_Normalized"/"model"/"autoencoder_model.pth",
              PROJECT_ROOT/"results"/"Collagen_Acetic_Acid_LowBands"/"model"/"autoencoder_model.pth"]:
        if p.exists(): return p
    raise FileNotFoundError("No existing model found.")

def run_experiment(fd, spectra, gt, rois, roi_label, mp, od):
    print(f"\n{'='*70}\nROI CONFIG: {roi_label}\n{'='*70}")
    clfs = get_classifiers()
    rm = np.zeros_like(gt, dtype=bool)
    for roi in rois:
        r0,r1,c0,c1 = roi['coords']; rm[r0:r1,c0:c1] = True
    print(f"  Train: {np.sum(gt[rm]>=0):,} px | Test: {np.sum(gt[~rm]>=0):,} px")

    results = []

    # Baseline
    print(f"\n--- BASELINE (all 149 bands) ---")
    Xtr,ytr,Xte,yte = extract_features(fd, gt, rois)
    for cn,(cf,) in clfs.items():
        m = run_clf(cf, Xtr,ytr,Xte,yte)
        m.update({'config':f"BASELINE_{cn}", 'classifier':cn, 'n_bands':149,
                  'roi':roi_label, 'band_selection':'all'})
        results.append(m)
        print(f"  {cn:15s}: Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
              f"Kappa={m['kappa']:.4f}  ({m['train_time_s']:.2f}s)")

    # Select bands once per count
    band_sel = {}
    for nb in TEST_BANDS:
        print(f"\n--- Selecting {nb} bands ---")
        wls = select_bands(spectra, nb, mp, od/f"sel_{nb}")
        band_sel[nb] = wls

    for nb in TEST_BANDS:
        wls = band_sel[nb]
        print(f"\n--- {nb} bands ---")
        Xtr,ytr,Xte,yte = extract_features(fd, gt, rois, wls)
        for cn,(cf,) in clfs.items():
            m = run_clf(cf, Xtr,ytr,Xte,yte)
            m.update({'config':f"{nb}b_{cn}", 'classifier':cn, 'n_bands':nb,
                      'roi':roi_label, 'band_selection':'autoencoder'})
            results.append(m)
            print(f"  {cn:15s}: Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
                  f"Kappa={m['kappa']:.4f}")
    return results

def print_summary(all_results, od):
    df = pd.DataFrame(all_results)
    for rl in df['roi'].unique():
        rd = df[df['roi']==rl]
        print(f"\n{'='*70}\nRESULTS: {rl}\n{'='*70}")
        bl = rd[rd['band_selection']=='all'].sort_values('accuracy', ascending=False)
        print(f"\n  BASELINES (all 149 bands):")
        for _,r in bl.iterrows():
            print(f"    {r['classifier']:15s}: Acc={r['accuracy']:.4f}  "
                  f"F1={r['f1']:.4f}  Kappa={r['kappa']:.4f}")
        knn_bl = bl[bl['classifier']=='KNN_k5']['accuracy'].values[0]
        sel = rd[rd['band_selection']=='autoencoder']
        print(f"\n  BEST CLASSIFIER PER BAND COUNT (KNN_k5 baseline={knn_bl:.4f}):")
        for nb in sorted(sel['n_bands'].unique()):
            nbd = sel[sel['n_bands']==nb]
            best = nbd.loc[nbd['accuracy'].idxmax()]
            d = best['accuracy'] - knn_bl
            mark = " ** EXCEEDS KNN BASELINE **" if d > 0 else ""
            print(f"    {int(nb):3d} bands | {best['classifier']:15s} | "
                  f"Acc={best['accuracy']:.4f} ({d:+.4f}){mark}")
        exc = sel[sel['accuracy'] > knn_bl]
        if len(exc) > 0:
            print(f"\n  ALL CONFIGS EXCEEDING KNN BASELINE ({knn_bl:.4f}):")
            for _,r in exc.sort_values('accuracy', ascending=False).head(20).iterrows():
                print(f"    {int(r['n_bands']):3d} bands | {r['classifier']:15s} | "
                      f"Acc={r['accuracy']:.4f} ({r['accuracy']-knn_bl:+.4f})")
    csv = od / "classifier_comparison.csv"
    df.to_csv(csv, index=False)
    print(f"\nSaved: {csv}")
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roi', default='both', choices=['original','expanded','both'])
    args = p.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    od = OUTPUT_BASE / ts; od.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    fd, spectra = load_data(DATA_DIR / "spectra_masked.pkl")
    mp = find_model(); print(f"Model: {mp}")
    all_r = []
    if args.roi in ('original','both'):
        gt, roi = load_ground_truth(DATA_DIR/"class_mask.png", DATA_DIR/"roi_regions.json")
        all_r.extend(run_experiment(fd, spectra, gt, roi, "original_small", mp, od/"original"))
    if args.roi in ('expanded','both'):
        gt, roi = load_ground_truth(DATA_DIR/"class_mask.png", DATA_DIR/"roi_regions_expanded.json")
        all_r.extend(run_experiment(fd, spectra, gt, roi, "expanded_all_rows", mp, od/"expanded"))
    print_summary(all_r, od)

if __name__ == "__main__":
    main()
