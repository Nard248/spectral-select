#!/usr/bin/env python3
"""
Drop Data — Radiometric re-run: per-pixel KNN-5 accuracy (the paper's primary
blind metric). For each radiometric variant and each selection method, take the
top-K bands, build per-pixel features for in-drop pixels, label each pixel by
its drop's Ward type, and run 5-fold stratified KNN-5 CV.

Reuses the rankings already written by drop_data_radiometric_rerun.py — no AE
retraining. Mirrors the metric that produced the old run's 0.964 headline so the
two runs are directly comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral_select import SpectraData  # noqa: E402
import drop_data_selection_sweep as sweep  # noqa: E402

PROC_OUT_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data Radiometric"
PROC_CROP_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data Cropped"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Radiometric"

VARIANTS = ["rad_best", "rad_best_detrend", "rad_merged", "rad_merged_detrend"]
METHODS = ["ae_perturb", "variance", "pca_loading", "sam_greedy", "spa", "mcuve"]
N_GRID = list(range(3, 11))
RANDOM_SEEDS = [0, 1, 2, 3, 4]


def pixel_features_and_labels(data, bands, drop_labels, drop_types):
    """In-drop pixels x all-bands features; label = Ward type of the drop."""
    by_ex = {float(ex): data.get_excitation(ex).cube for ex in data.excitation_wavelengths}
    sel = drop_labels > 0
    flat_lab = drop_labels[sel]                       # (P,) drop id 1..16
    y = drop_types[flat_lab - 1].astype(int)          # (P,) type 0..2
    P = flat_lab.size
    X = np.empty((P, len(bands)), dtype=np.float32)
    rows, cols = np.nonzero(sel)
    for c, b in enumerate(bands):
        X[:, c] = by_ex[b.excitation_nm][rows, cols, b.emission_idx]
    return X, y


def knn_cv(X, y, k_neighbors=5, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = KNeighborsClassifier(n_neighbors=k_neighbors)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def main():
    drop_labels = np.load(PROC_CROP_ROOT / "drop_labels.npy").astype(np.int32)
    rows = []
    for variant in VARIANTS:
        vdir = OUT_ROOT / variant
        data = SpectraData.from_pickle(PROC_OUT_ROOT / variant / "spectra_data.pkl")
        bands = sweep.build_band_catalog(data)
        label_to_idx = {b.label(): i for i, b in enumerate(bands)}
        drop_types = np.load(vdir / "drop_types.npy")
        X, y = pixel_features_and_labels(data, bands, drop_labels, drop_types)
        print(f"[knn] {variant}: X={X.shape}, classes={np.bincount(y).tolist()}")

        rankings = {}
        for m in METHODS:
            labels = json.loads((vdir / f"ranking_{m}.json").read_text())
            rankings[m] = [label_to_idx[l] for l in labels]
        for m, order in rankings.items():
            for n in N_GRID:
                acc = knn_cv(X[:, order[:n]], y)
                rows.append({"variant": variant, "method": m, "n_bands": n, "knn_acc": acc})
        # random baseline (mean over seeds)
        for n in N_GRID:
            accs = []
            for s in RANDOM_SEEDS:
                order = sweep.rank_random(len(bands), s)[:n]
                accs.append(knn_cv(X[:, order], y))
            rows.append({"variant": variant, "method": "random", "n_bands": n,
                         "knn_acc": float(np.mean(accs))})
        print(f"[knn]   {variant} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "knn_results.csv", index=False)

    print("\n=== per-pixel KNN-5 accuracy (mean over k=3..10) ===")
    piv = df.groupby(["variant", "method"])["knn_acc"].mean().unstack("method").round(3)
    cols = [c for c in ["ae_perturb", "variance", "pca_loading", "spa",
                        "sam_greedy", "mcuve", "random"] if c in piv.columns]
    print(piv[cols].to_string())

    print("\n=== KNN-5 accuracy at K=5 ===")
    k5 = df[df["n_bands"] == 5].pivot_table(index="variant", columns="method",
                                            values="knn_acc").round(3)
    print(k5[cols].to_string())
    print(f"\n[knn] wrote {OUT_ROOT/'knn_results.csv'}")


if __name__ == "__main__":
    main()
