"""Does the perturbation-AE band selection PRESERVE CLASSIFICATION on a synthetic 4D ME-HSI cube?

This reframes validation away from "did it hit the emission peak" toward the real downstream task:
can you classify the material at each pixel from the selected bands? Mirrors the published protocol
(KNN accuracy on selected vs all bands), on a labelled, balanced synthetic dataset with ground truth.

It also runs the normalization ablation for the perturbation-AE (influence normalization
variance|none|max_per_excitation) to probe WHY the default selection avoids emission peaks, and
reports peak_recovery next to classification F1 so the two questions sit side by side.

Run:  python reports/classification_experiment.py
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import tempfile

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from spectraforge import (
    AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig, from_fpbase_payload, render,
)
from spectraforge.scenegen import make_labeled_scene
from spectraforge.validation import validate_selection

FIX = pathlib.Path(__file__).parent / "fpbase_spectra"
DYES = [("EBFP2", 385.0, 0.56, 1.0), ("EGFP", 489.0, 0.60, 1.0), ("mCherry", 587.0, 0.22, 0.9)]
BUDGET = 12


def _load(name, qy, eps):
    return from_fpbase_payload(json.loads((FIX / f"{name}.json").read_text()),
                               quantum_yield=qy, extinction=eps)


def build_dataset(seed, noise):
    library = {name: _load(name, qy, eps) for name, ex, qy, eps in DYES}
    mats = [Material(name.lower(), {name: 1.0}) for name in library]
    acq = AcquisitionConfig(excitations=[ex for _, ex, _, _ in DYES], em_min=400, em_max=700, em_step=5)
    scene, labels = make_labeled_scene(mats, 64, 64, seed)
    spectra, gt = render(scene, library, acq, artifacts=noise,
                         physics=PhysicsConfig(psf_sigma_px=1.5, autofluorescence=0.02), seed=seed)
    return spectra, gt, labels.ravel(), acq


def feature_matrix(spectra):
    """(n_pixels, n_ex*n_em) features + a column->(ex, em_nm) map."""
    cols, colmap = [], []
    for ex in spectra.excitation_wavelengths:
        exd = spectra.get_excitation(ex)
        cube = exd.cube.reshape(-1, exd.cube.shape[-1])          # (pixels, n_em)
        for b, em in enumerate(exd.emission_wavelengths):
            cols.append(cube[:, b])
            colmap.append((float(ex), float(em)))
    return np.column_stack(cols), colmap


def cols_for_bands(colmap, bands, tol=3.0):
    """Map a list of (ex, em_nm) selections to feature-column indices."""
    out = []
    for ex, em in bands:
        best, bestd = None, tol + 1
        for j, (cex, cem) in enumerate(colmap):
            if abs(cex - ex) <= 1.0 and abs(cem - em) < bestd:
                best, bestd = j, abs(cem - em)
        if best is not None:
            out.append(best)
    return sorted(set(out))


def knn_macro_f1(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    knn = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr), ytr)
    return f1_score(yte, knn.predict(sc.transform(Xte)), average="macro")


def ae_selector(spectra, norm):
    from spectral_select import Analyzer, Config
    cfg = Config(sample_name="v", n_important_dimensions=18, n_bands_to_select=BUDGET,
                 perturbation_method="percentile", normalization_method=norm,
                 use_diversity_constraint=True, training_epochs=30, device="cpu",
                 random_seed=0, output_dir=pathlib.Path(tempfile.mkdtemp()))
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(cfg)
        a.fit(spectra)
        return [(b.excitation_nm, b.emission_nm) for b in a.get_wavelengths()]


def peak_neighbourhood_bands(gt, acq):
    """The BUDGET bands nearest to the true emission peaks (the 'is the neighbourhood enough?' test)."""
    grid = acq.emission_grid()
    peaks = []
    for fname, perex in gt.per_fluorophore_spectra.items():
        best_ex = max(perex, key=lambda e: float(np.max(perex[e])))
        peaks.append((float(best_ex), float(grid[int(np.argmax(perex[best_ex]))])))
    bands = []
    offset = 0
    while len(bands) < BUDGET:                                   # peak, then peak±step, ...
        for ex, pk in peaks:
            for s in ({0} if offset == 0 else {-offset, offset}):
                em = pk + s * (grid[1] - grid[0])
                if grid.min() <= em <= grid.max():
                    bands.append((ex, float(em)))
        offset += 1
    return bands[:BUDGET]


def main():
    seeds = [1, 2, 3]
    noise = ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01)
    rng = np.random.default_rng(0)
    print("=" * 92)
    print(f"Classification on a labelled balanced 4D ME-HSI cube (KNN macro-F1, {BUDGET}-band budget)")
    print("Dyes:", ", ".join(n for n, *_ in DYES), "| 3 classes (argmax material) | mean over 3 scenes")
    print("=" * 92)
    rows = {k: {"f1": [], "peak": []} for k in
            ["all bands", "AE (variance norm)", "AE (no norm)", "variance-ranking", "random", "peak-neighbourhood"]}

    for seed in seeds:
        spectra, gt, y, acq = build_dataset(seed, noise)
        X, colmap = feature_matrix(spectra)

        def record(name, cols, bands=None):
            rows[name]["f1"].append(knn_macro_f1(X[:, cols], y, seed))
            if bands is not None:
                rows[name]["peak"].append(validate_selection(gt, bands, tol_nm=10)["peak_recovery"])

        record("all bands", list(range(X.shape[1])))
        for norm, label in [("variance", "AE (variance norm)"), ("none", "AE (no norm)")]:
            bands = ae_selector(spectra, norm)
            record(label, cols_for_bands(colmap, bands), bands)
        var_cols = list(np.argsort(X.var(0))[::-1][:BUDGET])
        record("variance-ranking", var_cols, [colmap[c] for c in var_cols])
        rand_bands = [(float(rng.choice(acq.excitations)), float(rng.choice(acq.emission_grid()))) for _ in range(BUDGET)]
        record("random", cols_for_bands(colmap, rand_bands), rand_bands)
        pk_bands = peak_neighbourhood_bands(gt, acq)
        record("peak-neighbourhood", cols_for_bands(colmap, pk_bands), pk_bands)

    print(f"{'band selection':<24}{'KNN macro-F1':>14}{'peak_recovery':>16}")
    for name, d in rows.items():
        f1 = np.mean(d["f1"])
        peak = f"{np.mean(d['peak']):.2f}" if d["peak"] else "   -"
        print(f"{name:<24}{f1:>14.3f}{peak:>16}")
    print("-" * 92)
    print("Question 1 (does the AE selection classify?): compare AE rows vs random/all/peak-neighbourhood.")
    print("Question 2 (why off-peak?): compare AE (variance norm) vs AE (no norm) on peak_recovery.")


if __name__ == "__main__":
    main()
