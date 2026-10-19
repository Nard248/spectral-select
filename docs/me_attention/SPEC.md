# ME-Attention study — spec, protocol and fairness rules

Lead-owned. Contracts live in `src/me_attention/types.py` (`MEData`, `Estimator`, `ModelSpec`) — do not change them;
ask the lead. Worktree: `/Users/narekmeloyan/PycharmProjects/spectral-select-attention`, branch `research/me-attention`
(based on Gevorg's `origin/feature/me-smartattention`, commit 584ed77). `Data/` is a symlink to the main checkout's data.

Python: `/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/.venv/bin/python` with
`PYTHONPATH=src`. Tests: `PYTHONPATH=src <python> -m pytest tests/me_attention -o addopts="" -q`.
Device: Apple M4 Pro, 14 cores, 48 GB, `mps` available. No new third-party dependencies (no einops, xgboost, lightgbm):
use torch / numpy / scipy / scikit-learn only.

## Goal

A supervised pixel classifier for multi-excitation (ME) fluorescence hyperspectral images, built around attention, that
beats existing classifiers by a margin — **measured under a fair, pre-specified protocol**. If the fair protocol shows no
margin, we report that. Two tracks, reported separately:
- **spectral** — each pixel classified from its own (excitation, emission) measurements;
- **spatial** — each pixel classified from a P x P patch around it (spectral-spatial).

## Datasets

| Key | Source | Classes | Specimens | Notes |
|---|---|---|---|---|
| `collagen_acetic` | `Data/processed/Collagen_Acetic_Acid/` (class_mask.png, roi_regions.json, spectra_masked.pkl) | 3 | 3/class | **development set** (architecture exploration only; not in the final sweep tables except the bug-fix report) |
| `collagen_pepsin` | `Data/processed/Collagen Pepsin/` (same format) | 3 | 3/class | final |
| `lichens1` | `Data/processed/Lichens Dataset 1/` (same format) | 4 | 4/class | final; 8 excitations; 415 nm sampled at 455,465,... (5 nm offset grid) |
| `lichens2` | `Data/processed/Lichens_2/` — `lichens_2_class_mask.npy` (int16, -1 bg, classes 0..4), `lichens_2_class_info.json` (names), cubes from `data_cutoff_40nm_exposure_max_power_min.pkl` (`data[str(ex)]["cube"/"wavelengths"]`) | 5 | 2–10 regions/class | final; "specimens" = spatially disjoint regions (connected components ≥ 100 px); class 4 has only 2 regions / ~400 px |

Specimens = 8-connected components of each class mask, components < 100 px dropped (count reported).
Wavelengths: per-excitation vector in key `wavelengths` (nm). Emission grids are irregular (gaps, 5-nm offsets) —
models must not assume a regular grid unless they build one from `grid_index`.

## Protocol (pre-specified; do not change after seeing outer-test numbers)

**Outer CV — whole-specimen, class-covering folds.** k = min specimens per class. For each repeat r = 0..R-1: shuffle
each class's specimens with seed r and deal them round-robin into k folds, so every test fold contains ≥ 1 specimen of
every class. Assert this. R = 5 repeats (pepsin k=3 → 15 units; lichens1 k=4 → 20; lichens2 k=2 → 10).

**Inner validation (tuning + early stopping).** Inside each outer-train set, for every class hold out one specimen
(random, seeded) if the class has ≥ 2 training specimens; if it has only 1, hold out a spatially contiguous ~25% block
of that specimen (split along its longer axis). Inner-train = rest.

**Tuning.** Random search, **identical budget T for every model** (same T across baselines and ours), sampled from the
model's `space`; score = inner-val balanced accuracy. Retrain the chosen config on the full outer-train (neural models:
epochs = best inner epoch, scaled by train-size ratio) and evaluate once on the outer test fold.

**Preprocessing** is part of every model's search space, with the same options for all models (fit on training only):
`norm ∈ {none, l2, area, max}` per pixel (applied per excitation block or globally — `norm_scope ∈ {global, per_ex}`),
`log1p ∈ {False, True}`, then per-feature standardisation. Implemented once in `me_attention/preprocess.py`.

**Neural training (shared by all neural models, baselines and ours):** AdamW, cosine schedule with warmup, specimen-
balanced sampling (`samples_per_specimen`), label smoothing, max epochs E_max, early stopping on inner-val balanced
accuracy, same augmentation options (`aug ∈ {none, scale, scale+noise}`: per-pixel intensity scale ~ logU(0.7,1.4),
Gaussian noise 0–2% of feature std). Implemented once in `me_attention/train.py`. Mixed precision off (mps).

**Metrics** (labels fixed to 0..C-1, `zero_division=0`): pixel balanced accuracy (primary), macro-F1, pixel accuracy,
specimen-averaged accuracy (mean over test specimens of their pixel accuracy), specimen majority-vote accuracy,
per-class recall, confusion matrix. Specimen vote is only reported for whole-specimen splits.

**Statistics.** Unit = (repeat, fold). For our best model vs each baseline per dataset: mean ± std, paired
Wilcoxon signed-rank, Nadeau–Bengio corrected resampled t-test (test/train ratio), bootstrap 95% CI of the paired
difference. Holm correction across baselines. Across datasets: report each separately (no pooling) + win/tie/loss.

**Development hygiene.** Architecture/mechanism exploration uses only `collagen_acetic` (full CV is fine there) and
inner-validation scores of the final datasets. Outer-test numbers of the final datasets are produced once, by the final
sweep, after the search spaces are frozen in `docs/me_attention/FROZEN_SPACES.md`.

## Package layout (`src/me_attention/`)

| Module | Owner | Content |
|---|---|---|
| `types.py` | lead | contracts |
| `data.py` | A | loaders for the 4 datasets → `MEData`; `to_grid(X, data) -> [n,E,M]` + mask; `load(key)` registry |
| `protocol.py` | A | outer folds, inner split, `make_patches(data, idx, P) -> [n,P,P,F]` (zero outside centre pixel's specimen), metrics, stats |
| `preprocess.py` | B | shared preprocessing transformer (fit/transform, spectral + patch inputs) |
| `train.py` | B | shared torch training loop (`TorchEstimator` base implementing `Estimator`) |
| `models/baselines_spectral.py` | B | LR, linear SVM, RBF-SVM, RF, ExtraTrees, HistGradientBoosting, KNN, MLP, 1D-CNN, SpectralFormer (pixel-wise, GSE + CAF), vanilla band-token transformer |
| `models/baselines_spatial.py` | D | patch-mean + LR/RBF-SVM/RF, 2D-CNN, 3D-CNN, HybridSN, SpectralFormer patch-wise, SSFTT |
| `models/attention.py` | C | our attention family (see below) |
| `models/registry.py` | lead/C | `MODELS: dict[str, ModelSpec]` |
| `harness.py` | D | sweep runner (dataset × model × repeat × fold), resumable, JSONL results, timing |
| `analysis.py` | E | tables, stats, figures |

Gevorg's original module `src/spectral_select/me_smartattention/` gets the minimal bug fixes (A) so the bug-fix report
can show before/after; the fixed `MESmartAttention` is registered as model `smartattention_v0` (ours, spectral).

## Our attention family (C) — mechanisms to explore

All operate on the *measurement set* of a pixel (F valid (ex, em) values) and must handle irregular grids.
1. **v0** — Gevorg's model on the correctly aligned grid (`edge_mode` zero/discard), as the starting point.
2. **Measurement tokens** — one token per valid (ex, em) measurement: value embedding + continuous wavelength encoding
   (Fourier features of ex_nm, em_nm and Stokes shift em−ex, or learned per-feature embedding); full self-attention;
   CLS or attention pooling.
3. **Axial (factorised) attention** on the E x M grid with key-padding masks for missing cells: alternate attention
   along emission (within an excitation) and along excitation (same emission).
4. **Excitation tokens** — each excitation's emission spectrum embedded (masked linear / small 1D conv) into one token;
   attention across excitations (cross-excitation interactions), optional emission-patch sub-tokens.
5. **Wavelength-distance relative position bias** (bias = f(|Δem|, |Δex|), learned MLP) instead of/in addition to
   absolute encodings.
6. **Token gating / sparse attention** (learned per-token gate with L1 or top-k) — doubles as an attention-derived
   band-importance ranking we can compare with spectral-select.
7. Regularisers relevant to cross-specimen generalisation: token dropout (random measurement masking), intensity-scale
   augmentation, stochastic depth.
Spatial (D/C): **pixel-set encoder → spatial attention** — shared spectral encoder (best spectral variant) per patch
pixel, then attention over the P² pixel tokens (centre-pixel query or CLS), masked outside the specimen.

## Outputs

Results JSONL: `results/me_attention/<sweep>/<dataset>/<model>.jsonl` (one line per (repeat, fold): config, inner score,
best epoch, all metrics, per-specimen predictions summary, timings). Never commit large artefacts; figures go to
`publications/me_attention/figures/`.
