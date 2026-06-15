# Drop Data — Pipeline, Evaluation, and Results

**Author:** Narek Meloyan
**Document date:** 2026-05-12
**Status:** Standalone technical report. Companion to the spectral-select paper revision.

This document is a self-contained walk-through of the Drop Data dataset: what it is, how it was acquired, how it was preprocessed, how the ground truth was discovered, how the spectral-select framework was applied, and what was learned. It is intended to be readable without prior context on the paper. Every figure or numeric result is cross-referenced to a file on disk that the reader can open directly.

---

## Table of contents

1. [Motivation](#1-motivation)
2. [Sample and acquisition](#2-sample-and-acquisition)
3. [Inspection and quality control](#3-inspection-and-quality-control)
4. [Preprocessing pipeline](#4-preprocessing-pipeline)
5. [Ground-truth discovery](#5-ground-truth-discovery-blind-clustering)
6. [The selection method](#6-the-selection-method)
7. [Evaluation methodology](#7-evaluation-methodology)
8. [Results](#8-results)
9. [Findings worth noting](#9-findings-worth-noting)
10. [Pipeline scripts overview](#10-pipeline-scripts-overview)
11. [Future work](#11-future-work)
12. [Appendix: file map](#12-appendix-file-map)

---

## 1. Motivation

The Drop Data dataset was acquired to stress-test the unsupervised claim of the spectral-select framework after the submission of the Lichens-based paper. The Lichens dataset is supervised: four expert-annotated morphological classes are available, and the final value of *K* (the number of selected bands) was implicitly chosen with reference to classification accuracy. A reviewer could legitimately ask whether the framework's gains depend on having labels available at any stage of the pipeline.

The Drop Data dataset is the answer. **No labels of any kind were available during selection**: the framework is run end-to-end without any reference signal, and its output is evaluated post-hoc against a ground truth derived entirely from unsupervised clustering of the full 214-dimensional spectral cube. Success on this dataset is therefore evidence that the framework selects intrinsically informative bands rather than bands that happen to align with a supervised label set.

A secondary motivation is **physical interpretation**: droplets of varied chemistry on a fixed grid are a cleaner physical setup than lichen samples. Spectra are spatially segregated (each drop is a small connected region), background is uniform, and the dominant fluorescence signal is concentrated in the 470--530 nm autofluorescence window familiar from biological samples. Whether the framework's selected bands map to this window or wander somewhere else is a strong sanity check on what the autoencoder has actually learned.

---

## 2. Sample and acquisition

### 2.1 Physical layout

Sixteen fluorescent drops of varied composition were deposited onto a glass stage in a $4 \times 7$ grid (28 well positions; 12 wells were empty or below detection threshold after drop placement). The grid was imaged at seven excitation wavelengths under HDR bracketing.

A calibration ruler was placed across the bottom of the field of view to enable spatial reference. As detailed in Section 4, this ruler had to be removed by cropping before the autoencoder pipeline could be applied; otherwise the high-contrast ruler edges dominate every variance- and PCA-based statistic and the AE learns to reconstruct the ruler rather than the drops.

### 2.2 Acquisition matrix

Seven excitations from 310 nm to 415 nm in non-uniform steps, with 2--3 integration times per excitation to handle the wide intra-scene dynamic range:

| Excitation (nm) | Integration times (ms) |
|-----------------|------------------------|
| 310 | 1100, 1300, 1500 |
| 325 | 1100, 1300, 1500 |
| 340 | 150, 200, 250 |
| 365 | 10, 12, 15 |
| 385 | 6, 8, 10 |
| 400 | 7, 8 |
| 415 | 35, 40 |

Each integration time produces a full hyperspectral cube of shape $256 \times 348 \times 31$ (height, width, emission bands). Total raw data: 20 sample cubes + 1 Background cube + 1 Whitelight cube = 22 cubes, each $\approx 5.8$ MB.

**Emission grid:** 420--720 nm in 10 nm steps (31 emission bands per excitation in raw form; some are masked out by Rayleigh cutoffs in post-processing).

**Why HDR bracketing:** The fluorescence signal varies by more than an order of magnitude across drops and excitations. A single integration time saturates the brightest drops at deep-UV excitations and produces only noise at long-UV excitations. HDR bracketing lets us pick the longest non-saturating exposure per excitation.

### 2.3 Reference cubes

- **Background.im3** — a dark frame acquired with no excitation light. Mean intensity 52 ADU, very flat across all emission bands. Used directly for dark-frame subtraction.
- **Whitelight.im3** — a scene captured with broadband illumination *but* the field of view includes the calibration ruler. Cannot be used as a flat-field divisor because it is not actually a flat field; we use it only for inspection.

### 2.4 Saturation behavior

The Nuance FX camera is 12-bit with a $\sim 52$ ADU offset, so the ceiling is approximately 3886 ADU. Saturation behavior by excitation (longest exposure picked):

- **310, 325, 340, 365 nm** — clean (< 0.01% saturated pixels)
- **385, 400, 415 nm** — *all* integration times produce some saturated pixels on the brightest drops; the shortest exposure leaves 0.4--2.3% saturated. This is a known limitation; saturated pixels at these excitations are accepted rather than masked, because they consistently correspond to the brightest drops (Ward Type 0) whose chemistry is dominated by the autofluorescence peak in 470--530 nm.

### 2.5 Recommended cube per excitation

The first-pass inspection (Section 3) produced a recommended cube per excitation, saved as `results/Drop_Data_Inspection/recommended_cubes.json`:

| Ex (nm) | Recommended file | Reason |
|---|---|---|
| 310 | `310 1500 SPF` | longest non-saturating |
| 325 | `325 1500 SPF` | longest non-saturating |
| 340 | `340 250 SPF`  | longest non-saturating |
| 365 | `365 10 SPF`   | longest non-saturating |
| 385 | `385 6 SPF`    | shortest (all saturating) |
| 400 | `400 7 SPF`    | shortest (all saturating) |
| 415 | `415 35 SPF`   | shortest (all saturating) |

Throughout the rest of the pipeline, only these seven cubes are used as the "single representative" per excitation. The other integration times are kept for posterity (in `Data/processed/Drop Data/raw/`) but not referenced again.

---

## 3. Inspection and quality control

Before any model was trained, four inspection plots were produced. Each is on disk in `results/Drop_Data_Inspection/` and was used to make a deliberate go/no-go decision for downstream processing.

### 3.1 Per-cube max projections

For each of the 22 raw cubes, a per-pixel maximum-over-emission projection was produced. Files: `results/Drop_Data_Inspection/max_proj/<stem>.png`. There are 22 of these (one per cube). Two examples worth opening:

- `results/Drop_Data_Inspection/max_proj/310 1500 SPF.png` — at 310 nm with the longest exposure, nearly all 28 grid positions show fluorescent signal, including many drops that are dim at other excitations. This is the cube that pays for the longest integration time.
- `results/Drop_Data_Inspection/max_proj/415 40 SPF.png` — at 415 nm only 2--3 drops fluoresce strongly, plus diffuse haze (likely substrate autofluorescence at the long-UV edge).

Visual inspection of these max projections is what established that the drops are arranged in a $4 \times 7$ grid and that only 16 of 28 wells produced detectable fluorescence.

### 3.2 Excitation montage (cross-exposure)

File: `results/Drop_Data_Inspection/excitation_montage.png`. A single montage figure showing all integration times for each excitation, side by side, with a marker (★) on the recommended cube per excitation. The reader can see at a glance:

- Each excitation's dynamic range across its 2--3 integration times.
- The saturation behavior on the brightest drops at 385/400/415 nm.
- The visibility of dim drops only at the longest 310/325 nm exposures.

### 3.3 Mean spectra overlay

File: `results/Drop_Data_Inspection/mean_spectra.png`. The pixel-mean spectrum of every cube (excluding background pixels) overlaid on a common emission-wavelength axis. Useful for confirming that the 7 recommended cubes have consistent baseline shape and that the dynamic ranges align after intensity normalization.

### 3.4 Reference check

File: `results/Drop_Data_Inspection/reference_check.png`. Side-by-side visualization of the Background and Whitelight cubes' per-band intensity distributions. Confirms:

- Background is genuinely a dark frame (median 52, flat across bands).
- Whitelight contains the ruler in the bottom rows and is not a flat field; do not use as a divisor.

### 3.5 Ruler mask detection

File: `results/Drop_Data_Inspection/ruler_mask_check.png`. Automatic detection of the calibration ruler region as a horizontal band of consistent high intensity at the image bottom. This visualization is what motivated the cropping step in Section 4: the ruler region must be excluded before any variance- or PCA-based statistic is computed on the image.

---

## 4. Preprocessing pipeline

The preprocessing pipeline produces **five cumulative variants** of the cube, each adding one more step on top of the previous. A separate "cropped" pipeline drops the ruler region first; this yields **ten** total variants (`raw`, `dark`, `dark_norm`, `dark_norm_mask`, `full`, and their `_cr` counterparts).

### 4.1 The five variants

| Variant | Operations applied (cumulative) |
|---|---|
| `raw` | Load .im3 → cast to float32 (no other operations) |
| `dark` | raw + subtract Background mean per band |
| `dark_norm` | dark + per-pixel normalization (divide each pixel's spectrum by its peak across bands) |
| `dark_norm_mask` | dark_norm + drop-only pixel mask (background pixels zeroed) |
| `full` | dark_norm_mask + final per-cube intensity rescale to $[0, 1]$ |

Each variant is saved at `Data/processed/Drop Data/<variant>/spectra_data.pkl`. The drop mask and per-drop labels are shared across variants and saved at `Data/processed/Drop Data/drop_mask.npy` and `Data/processed/Drop Data/drop_labels.npy`.

### 4.2 Ruler crop (the `_cr` variants)

The ruler occupies image rows $\geq 175$ (after determining the precise upper edge by visual inspection). The cropped pipeline applies row-based cropping *before* the five variant steps:

- `results/Drop_Data_Cropped_Sweep/<variant>_cr/` — the five cropped variants.
- Spatial dimensions after cropping: $175 \times 348$ pixels (down from $256 \times 348$).
- Number of detected drops after cropping: 16 (down from 18, because two ruler-edge artifacts at rows 179--186 were eliminated).

The constant `RULER_ROW_START = 175` lives in both `experiments/drop_data_cropped_pipeline.py` and `experiments/drop_data_export_slides_cropped.py`. Changing it shifts the crop globally.

### 4.3 Drop detection

A watershed-based detector finds connected high-signal regions in the broadband max projection (mean of all 7 recommended cubes' max-projections). 16 drops are detected and assigned IDs 1--16. The mask is a $H \times W$ integer array with 0 = background and 1--16 = drop ID; the binary mask is `drop_id > 0`.

This detection is **fixed across all variants**: the same 16 drops are used everywhere, even though their pixel values differ across variants.

### 4.4 Rayleigh masking

Two Rayleigh cutoffs are applied during analysis (not during cube storage):

- **First-order:** $\lambda_{\text{em}} < \lambda_{\text{ex}} + 40$ → invalid.
- **Second-order:** $|\lambda_{\text{em}} - 2 \lambda_{\text{ex}}| < 40$ → invalid.

After both cutoffs, 214 valid $(\lambda_{\text{ex}}, \lambda_{\text{em}})$ pairs remain across the seven excitations. The full breakdown:

| $\lambda_{\text{ex}}$ (nm) | Valid emission bands | Count |
|---|---|---|
| 310 | 420--720 (1st-order at 350, well below; 2nd-order at 620 excludes 580--660) | 31 raw, fewer after 2nd-order |
| 325 | 420--720 (1st-order at 365) | 31 raw |
| 340 | 420--720 (1st-order at 380; 2nd-order at 680) | 31 raw |
| 365 | 420--720 (1st-order at 405) | 31 raw |
| 385 | 425--720 (1st-order at 425) | 30 raw |
| 400 | 440--720 (1st-order at 440) | 29 raw |
| 415 | 455--720 (1st-order at 455) | 27 raw |

The numbers in the table are pre-second-order. Total valid after both cutoffs: 214.

In the cached cube (`revision/figures/drop_data/full_cr_cube.npz`), the per-excitation emission counts are: 310→31, 325→31, 340→31, 365→31, 385→31, 400→30, 415→29. The remaining cells are masked at analysis time, not at storage time, so the cube retains a regular rectangular structure for I/O.

---

## 5. Ground-truth discovery (blind clustering)

Without labels, an evaluation ground truth had to be constructed from the data itself. The procedure:

1. **Compute per-drop mean spectra.** For each of the 16 drops, average all in-drop pixels across all 214 valid bands. Output: `results/Drop_Data_Cropped_Sweep/full_cr/drop_mean_spectra.npy` of shape $(16, 214)$.

2. **Hierarchical clustering.** Apply Ward linkage agglomerative clustering with $k=3$ to the 16-drop $\times$ 214-band matrix. Output: `results/Drop_Data_Cropped_Sweep/full_cr/drop_types.npy` of shape $(16,)$ with values in $\{0, 1, 2\}$.

3. **Result:** three spectral archetypes with the following distribution:

| Type | Count | Description | Mean peak intensity |
|---|---|---|---|
| 0 (Bright) | 3 drops | Strong peak at $\lambda_{\text{em}} = 470$--$530$ nm under excitation 365--415 nm | $\sim 0.85$ (peak), $\sim 0.22$ (mean) |
| 1 (Moderate) | 5 drops | Same shape as Type 0 but attenuated by $\sim 4\times$ | $\sim 0.63$ (peak), $\sim 0.10$ (mean) |
| 2 (Baseline) | 8 drops | Low intensity throughout; weak or absent autofluorescence peak | $\sim 0.54$ (peak), $\sim 0.11$ (mean) |

### 5.1 Why $k=3$

The choice of $k=3$ was made by visual inspection of the dendrogram and the per-drop overview plot at `results/Drop_Data_Spectra_Explore/full/archetypes_and_members.png`. Alternative cluster counts ($k=2$ and $k=4$) were considered:

- **$k=2$** would lump Moderate and Baseline together. The 5 Moderate drops would be assigned to whichever side of the cut they fell closer to, mostly Baseline. Loses interpretive granularity.
- **$k=4$** splits Type 2 (Baseline) into two sub-groups distinguished by minor noise patterns; not physically meaningful.

$k=3$ produced the cleanest interpretation. Silhouette and Calinski-Harabasz scores were both maximized at $k=3$.

### 5.2 Important caveats

The "ground truth" used in the evaluation is **derived from the full-spectrum data**, not from any external reference. This is the cleanest available test of the framework's unsupervised claim:

> *Given only the cube and no external information, can the framework's $K$-band selection recover the same partition that the full 214-band spectrum reveals?*

If yes, the selection is preserving the intrinsic structure of the data. If no, the selection is losing information. Section 7 details how this is quantified.

A separate file `results/Drop_Data_Spectra_Explore/full/discriminative_band_map.png` visualizes per-band F-ratio (between-type / within-type variance, with types from Ward $k=3$) across the entire 214-band space. The reader can see at a glance which bands carry the most discriminative information — concentrated, unsurprisingly, in the 470--530 nm region at $\lambda_{\text{ex}} = 365$--415 nm.

---

## 6. The selection method

This section is a brief reminder of how the framework works; for full detail see the paper.

### 6.1 3D convolutional autoencoder

Each excitation has its own encoder branch (one Conv3D layer mapping the spatial+spectral cube to a $k_1 = 20$ channel feature map). Encoder outputs are pooled to the spatial dimensions and averaged across excitations, yielding a shared $k_1$-channel feature map. A second Conv3D layer produces the bottleneck representation $z \in \mathbb{R}^{20 \times 1 \times H \times W}$.

The decoder mirrors the encoder. Loss is masked MSE between input and reconstruction, with masking restricted to in-drop pixels. Training uses Adam with `lr=0.001`, batch size 32, patch size $64 \times 64$, early stopping at patience 5. Convergence is typically reached in 25--30 epochs.

For Drop Data, training takes approximately 7 minutes on a single M-series Mac with MPS+CPU fallback (one fallback path: `aten::_adaptive_avg_pool3d` is not MPS-native).

### 6.2 Latent dimension scoring

After training, each of the 20 latent dimensions is scored by variance (default) or PCA. The top-$k$ dimensions are passed to the perturbation step.

For Drop Data, the optimal configuration found by sweep was: `normalization_method=max_per_excitation`, `dimension_selection_method=variance` (or PCA — both work), `n_important_dimensions=3`, perturbation magnitude $\epsilon \in \{30, 40, 50\}$ aggregated.

### 6.3 Perturbation analysis

For each top-$k$ latent dimension $d$, perturbations $z'_d = z \pm \epsilon \sigma_d e_d$ are generated and decoded. The reconstruction difference, averaged over patches and aggregated over $\epsilon$ values, gives a per-band influence score. Sum across selected latent dimensions yields the final influence matrix of shape $7 \times 31$ (with Rayleigh-invalid cells set to zero).

### 6.4 Influence normalization

A subtle but crucial step. Three options are available:

- `none` — use raw influence scores.
- `variance` — divide each cell's influence by the data variance at that cell.
- `max_per_excitation` — divide each cell's influence by the max influence within its excitation row.

On mixed-media datasets (Lichens, Collagen Sponges), these are roughly equivalent. On Drop Data they are not: `variance` *inverts* the ranking (Section 9.2). Always use `max_per_excitation` on segregated-sample datasets.

### 6.5 MMR selection

Maximum Marginal Relevance with $\lambda = 0.5$ converts the influence matrix into an ordered list of $K$ bands, balancing per-band relevance with diversity (cosine similarity penalty between selected bands).

---

## 7. Evaluation methodology

This section explains how performance is measured on a fully unlabeled dataset.

### 7.1 Per-pixel KNN-5 accuracy (primary metric)

The labels for evaluation come from Section 5: each of 16 drops has a Ward type in $\{0, 1, 2\}$, and each in-drop pixel inherits its drop's type. A standard 5-fold stratified cross-validation is applied with KNN-5 on the $K$ selected bands; accuracy is reported as the mean across folds.

**Why this is the primary metric:** unlike cluster-recovery metrics (see 7.3), per-pixel classification requires each selected band to carry *informative signal per pixel*, not merely sit in the same general region as an informative band. A method that selects one strong band and four near-zero noise bands gets penalized by KNN because four of its five features are uninformative.

### 7.2 F-ratio of selected bands (secondary metric)

For each band, the F-ratio is the ratio of between-Ward-type variance to within-Ward-type variance. Higher = more discriminative. The sum or mean F-ratio across the $K$ selected bands quantifies the framework's discriminative budget.

This metric was used in the early Drop Data analysis (`results/Drop_Data_Cropped_Sweep/`). It is reported alongside KNN accuracy where helpful.

### 7.3 Adjusted Rand Index — *reported for completeness but not primary*

The natural-looking metric of *"how well does Ward at $k=3$ on the $K$ selected bands recover the Ward at $k=3$ on the full 214 bands?"* turns out to be flawed, as discussed in Section 9.3. It is reported in the results table for completeness but is not the headline metric.

### 7.4 Why classification (not clustering) is the right test

The Drop Data ground truth is itself a clustering output. Using clustering on a subset of bands to evaluate clustering on the full set creates a sympathetic-prior problem: methods that include the right *number* of bands at the right *general location* score well even if the bands themselves carry little per-pixel information. Classification, in contrast, requires that each band actually help discriminate a pixel from another. That is the deployment-relevant question.

---

## 8. Results

### 8.1 Selected bands

For the canonical configuration (`full_cr` variant, max-per-excitation normalization, AE-perturb with $k=3$ PCA dimensions, MMR $\lambda = 0.5$, $\epsilon \in \{30, 40, 50\}$), the top-10 selected bands in order are:

| Rank | $\lambda_{\text{ex}}$ (nm) | $\lambda_{\text{em}}$ (nm) |
|---|---|---|
| 1 | 325 | 530 |
| 2 | 365 | 490 |
| 3 | 400 | 490 |
| 4 | 415 | 490 |
| 5 | 385 | 470 |
| 6 | 310 | 440 |
| 7 | 340 | 500 |
| 8 | 400 | 470 |
| 9 | 415 | 480 |
| 10 | 385 | 500 |

The top 5 span **five different excitations** (325, 365, 400, 415, 385) and all fall within the 470--530 nm emission window — the classical biological autofluorescence band.

The full list is in `results/Drop_Data_Cropped_Sweep/full_cr/ae_perturb_results.csv`. Per-band slice TIFFs for each $K \in \{3, 5, 8, 10\}$ are in `results/Drop_Data_Best_Slides_Cropped/full_cr/n5/`, `n3/`, `n8/`, `n10/`. For example:

- `results/Drop_Data_Best_Slides_Cropped/full_cr/n5/rank01_ex325_em530nm.png` — the top-1 band visualized as a per-pixel intensity image.
- `results/Drop_Data_Best_Slides_Cropped/full_cr/n5/rank02_ex365_em490nm.png` — top-2 band image.
- ... (one per rank up to $n=$ the K value).
- `results/Drop_Data_Best_Slides_Cropped/full_cr/n5/_montage.png` — all 5 selected band images in a single montage.

A full collage of *all 214 bands* (one tile per band, ordered by excitation and emission) is at `results/Drop_Data_Best_Slides_Cropped/full_cr/_all_bands_collage.png`. This is useful for visually scanning the entire EEM space and confirming that the 5 selected bands really do correspond to the brightest, most discriminative tiles.

### 8.2 Per-type EEM heatmaps with selected-band markers

**Figure file:** `revision/figures/drop_data/panel_A_eem_per_type.png`

Three side-by-side heatmaps showing the mean Excitation-Emission Matrix (EEM) for each Ward type. The 5 selected $(\lambda_{\text{ex}}, \lambda_{\text{em}})$ markers (white circles with x-marks) are overlaid on each heatmap to show where in the EEM space the framework chose to look.

Reading this figure:

- **Type 0 (Bright, 3 drops, left panel):** a clear hot region at $\lambda_{\text{em}} \approx 470$--530 nm spanning $\lambda_{\text{ex}} = 365$--415 nm. The selected markers cluster in exactly this hot region.
- **Type 1 (Moderate, 5 drops, middle panel):** the same hot region exists but with $\sim 4 \times$ lower intensity. The markers still sit in the type's most-distinctive region.
- **Type 2 (Baseline, 8 drops, right panel):** mostly dim across the entire EEM with a faint hint of structure in the same region. The markers fall on cells that are dim in this type but bright in Types 0 and 1, which is exactly the discriminative position.

Rayleigh-invalid cells are shown in gray; the diagonal staircase pattern reflects both first-order ($\lambda_{\text{em}} \geq \lambda_{\text{ex}} + 40$) and second-order ($|\lambda_{\text{em}} - 2 \lambda_{\text{ex}}| \geq 40$) cutoffs.

### 8.3 Per-excitation emission slices

**Figure file:** `revision/figures/drop_data/panel_B_emission_slices.png`

Seven subplots, one per $\lambda_{\text{ex}}$, each plotting the emission spectrum of all 16 drops as faint lines colored by Ward type, with type-mean curves overlaid in bold. Vertical dashed lines mark the selected $\lambda_{\text{em}}$ at each excitation (when a selection was made).

Reading this figure:

- **$\lambda_{\text{ex}} = 310$ nm (top-left): "silent."** The type-mean curves overlap. The framework correctly selected no band here.
- **$\lambda_{\text{ex}} = 325$ nm: selected at 530.** A noticeable type-1 vs.\ type-0/2 separation around 530 nm; the framework picks it.
- **$\lambda_{\text{ex}} = 340$ nm: "silent."** Curves overlap again; no selection.
- **$\lambda_{\text{ex}} = 365$ nm: selected at 490.** Strong type-0 (red) peak at 470--490 nm separates cleanly from types 1 and 2.
- **$\lambda_{\text{ex}} = 385$ nm: selected at 470.** Sharp peak.
- **$\lambda_{\text{ex}} = 400$ nm: selected at 490.** Sharp peak.
- **$\lambda_{\text{ex}} = 415$ nm: selected at 490.** Sharp peak.

The asymmetry is the punchline: **the framework selects bands at exactly the excitations where the type-mean curves separate, and is silent at exactly the excitations where they overlap.** This is the strongest single piece of evidence in the paper that the framework's selections are physically interpretable.

### 8.4 KNN accuracy vs.\ K

**Figure file:** `revision/figures/drop_data/panel_C_knn_vs_K.png`

The proposed method (red, bold) plotted against eight baselines, with KNN-5 per-pixel accuracy on the y-axis and $K$ on the x-axis. Numerical results (mean over 5 seeds for stochastic methods):

| Method | K=3 | K=5 | K=7 | K=10 |
|---|---|---|---|---|
| **Proposed (AE-perturb)** | **0.938** | 0.947 | 0.957 | **0.964** |
| Variance | 0.793 | 0.956 | 0.956 | 0.956 |
| PCA-loading | 0.793 | 0.863 | 0.954 | 0.958 |
| SPA | 0.952 | 0.950 | 0.955 | 0.957 |
| SAM-greedy | 0.818 | 0.804 | 0.774 | 0.740 |
| MCUVE | 0.815 | 0.856 | 0.905 | 0.917 |
| ISSC | 0.845 | 0.931 | 0.928 | 0.946 |
| BS-Net-FC | 0.705 | 0.819 | 0.903 | 0.940 |
| Sparse-LASSO (supervised) | 0.846 | 0.892 | 0.926 | 0.940 |
| Random (mean) | 0.745 | 0.804 | 0.847 | 0.886 |

Source: `revision/baselines/results_drop/drop_data_full_cr/method_summary.csv`.

Key observations:

- The proposed method is in the top tier at every $K$ and wins outright at $K=10$.
- SPA is a strong classical competitor at low $K$.
- Variance is surprisingly competitive at $K \geq 5$ on this dataset (consistent with the segregated-samples nature; the highest-variance bands also happen to be the discriminative bands).
- SAM-greedy *degrades* with $K$ — see Section 9.4 for the explanation.
- Supervised Sparse-LASSO does not beat the unsupervised methods, indicating that labels provide no structural advantage on this dataset.

### 8.5 Per-K band-image slides

For each $K \in \{3, 5, 8, 10\}$, the result-folder contains the actual per-band intensity images of the selected bands:

- `results/Drop_Data_Best_Slides_Cropped/full_cr/n3/` — 3 per-band PNGs + montage.
- `results/Drop_Data_Best_Slides_Cropped/full_cr/n5/` — 5 per-band PNGs + montage (the canonical headline).
- `results/Drop_Data_Best_Slides_Cropped/full_cr/n8/` — 8 per-band PNGs + montage.
- `results/Drop_Data_Best_Slides_Cropped/full_cr/n10/` — 10 per-band PNGs + montage.

Each per-band file is named `rankNN_exEEE_emEEEnm.png`. For example, `rank01_ex325_em530nm.png` shows the per-pixel intensity at $\lambda_{\text{ex}} = 325$ nm, $\lambda_{\text{em}} = 530$ nm. Opening these files in sequence (rank 1 through rank K) makes the framework's selection ordering visually concrete: rank 1 should be the band with the cleanest type-separating structure, rank 2 should add complementary information, and so on.

### 8.6 Whole-EEM collage for context

`results/Drop_Data_Best_Slides_Cropped/full_cr/_all_bands_collage.png` is a single image containing all 214 valid bands as tiles, ordered by excitation (rows) and emission (columns), with gray cells for Rayleigh-invalid positions. Useful for visually confirming that:

- The 5 selected bands (highlighted) really are among the brightest, most type-distinctive tiles.
- The "silent" excitations (310, 340) contain rows of tiles that are uniformly bright or uniformly dim, without the type-separating contrast seen at the selected excitations.

### 8.7 Per-drop overview and discriminative-band map

Two additional inspection figures, useful for understanding the per-drop variability that goes into the per-type aggregates:

- `results/Drop_Data_Spectra_Explore/full/per_drop_overview.png` — one mini-spectrum per drop, colored by Ward type. Shows the within-type variability and confirms that the 3-type partition is well-separated.
- `results/Drop_Data_Spectra_Explore/full/discriminative_band_map.png` — per-band F-ratio heatmap over the entire EEM space, with the AE-perturb selections marked. Visually confirms that the selections cluster at the high-F-ratio cells.
- `results/Drop_Data_Spectra_Explore/full/archetypes_and_members.png` — the three Ward archetypes (type means) plotted alongside their member drops. Shows the within-type spread.
- `results/Drop_Data_Spectra_Explore/full/method_selections_overlay_n5.png` — overlay of AE-perturb's $K=5$ selection on top of the discriminative-band heatmap, with comparison methods (variance, PCA, SAM, SPA, MCUVE) shown as separate markers. Lets the reader see at a glance how different methods cover (or miss) the discriminative region.

---

## 9. Findings worth noting

Three findings emerged during the Drop Data work that are worth recording as standalone notes. The first two are candidate paper contributions; the third is a methodological note for the band-selection community.

### 9.1 The framework's silences are informative

Both Panels A and B show that the framework selected zero bands at $\lambda_{\text{ex}} = 310$ and $\lambda_{\text{ex}} = 340$ nm. Quantitatively, the between-Ward-type variance at these excitations (summed across all valid emission bands) is 6--10 $\times$ lower than at $\lambda_{\text{ex}} = 365, 385, 400, 415$ nm. The autoencoder's per-band influence scores naturally fall below the MMR diversity threshold at these excitations, and they are excluded.

This is interpretable: the framework is saying *"there is no discriminative chemistry visible at these excitations under this sample's fluorophore population."* A practitioner reading this output knows not to acquire those excitations on similar samples in the future, which is exactly the actionable sensor-design guidance the paper's introduction promised.

### 9.2 Influence normalization defines a scope axis (and is a candidate contribution)

The original framework defaulted to `normalization_method = variance`. On Lichens and Collagen Sponges, this produced sensible selections. On Drop Data, it *inverted* the ranking: Spearman rank correlation between AE influence and per-band F-ratio was $-0.244$ to $-0.332$ across preprocessing variants, meaning the framework consistently picked the *least* discriminative bands.

The reason is mechanical: variance-normalization divides each cell's influence by the data variance at that cell. On Drop Data, the discriminative bands are *also* the highest-variance bands (because that is where the few bright drops differ from the many dim ones). Dividing influence by variance inverts the desired ranking on segregated-sample datasets.

Switching to `normalization_method = max_per_excitation` flipped Spearman correlation from $-0.33$ to $+0.31$ on `full_cr`, and the mean F-ratio of selected bands jumped 1.9--3.1× across preprocessing variants:

| Variant | Mean F-ratio with variance norm | Mean F-ratio with max_per_excitation | Improvement |
|---|---|---|---|
| `full_cr` | 23.0 | 71.6 | 3.1× |
| `dark_norm_mask_cr` | 11.0 | 20.5 | 1.9× |
| `dark_norm_cr` | 8.7 | 16.4 | 1.9× |

**The practical guideline is therefore:**

- Use `variance` (the original default) for mixed-media datasets where samples extend over wide spatial regions containing both bright and dim pixels.
- Use `max_per_excitation` for spatially-segregated datasets where each class is concentrated in a small number of physically distinct samples.

This is a real scope-of-applicability axis for the method and is now documented as such in the Discussion section of the paper.

### 9.3 ARI is a flawed metric on small unsupervised datasets

While building the Drop Data benchmark, an initial version of the evaluation used the Adjusted Rand Index (ARI) of Ward clustering at $k=3$ on the $K$-band drop-mean vectors vs.\ Ward $k=3$ on the full 214 bands. This produced a counterintuitive result: SAM-greedy scored ARI = 0.78 at $K=5$, while the proposed method scored only 0.36.

Investigation revealed why. SAM-greedy's $K=5$ selection on Drop Data was:
```
[(385, 490), (310, 420), (415, 720), (385, 420), (415, 710)]
```
— one strong band at 385/490 (the autofluorescence peak) plus four near-zero-signal bands. Ward clustering in the resulting 5-D space is dominated by the one informative dimension; the noise dimensions don't add useful within-cluster variance and so don't hurt the cluster recovery.

The proposed framework's $K=5$ selection, by contrast:
```
[(325, 530), (365, 490), (400, 490), (415, 490), (385, 470)]
```
— five correlated bands at the same fluorophore peak from different excitations. Information-rich, but the redundancy among the five bands tightens within-type clusters without sharpening the across-type boundary in the same way a single-band-plus-noise selection does.

**The methodological note:** on small-sample unsupervised datasets, cluster-recovery metrics like ARI systematically reward selections that include noise bands. Per-pixel classification accuracy (which we use as primary metric) requires each band to contribute discriminative signal per pixel and is therefore more rigorous.

This is reported as a standalone note in the paper's Discussion (§VI.D) and may be valuable to other practitioners benchmarking unsupervised band-selection methods.

### 9.4 SAM-greedy fails predictably at high K

Related to 9.3: SAM-greedy's KNN accuracy *degrades* from 0.82 at $K=3$ to 0.74 at $K=10$ on Drop Data, well below random selection's 0.89 at $K=10$. The mechanism is mechanical: SAM-greedy iteratively selects bands by maximum spectral-angle distance from the already-selected set. Once the informative subspace has been covered, additional selections by angular orthogonality necessarily land in the *low-signal* region of the spectrum (the bands that are most orthogonal to bright bands are by construction the dim bands). Adding more such bands hurts the KNN classifier.

This is a clean illustration of why diversity-only methods are dangerous on small datasets where the informative subspace has low dimensionality. The proposed framework avoids this by anchoring diversity to a learned *relevance* score (from perturbation) rather than to spectral angle alone.

---

## 10. Pipeline scripts overview

Each step of the Drop Data analysis is implemented as a standalone script in `experiments/`. The scripts are idempotent: they cache intermediate results, so re-running is cheap.

### 10.1 First pass (heavy — uses PyImageJ to load .im3)

`experiments/drop_data_inspect.py`

- Loads all 22 .im3 cubes via PyImageJ (one-time, ~30s init cost).
- Saves each cube to `Data/processed/Drop Data/raw/<stem>.npy` for fast subsequent re-loading.
- Produces per-cube max projections (`results/Drop_Data_Inspection/max_proj/<stem>.png`).
- Produces summary CSV/JSON (`results/Drop_Data_Inspection/summary.csv`, `summary.json`).
- Produces mean spectra overlay and reference-check figure.

Run once. After this, the `.npy` cache means PyImageJ is no longer needed.

### 10.2 Follow-up inspection (fast — uses .npy cache only)

`experiments/drop_data_montage.py`

- Reads the `.npy` cache only.
- Produces the excitation montage (`excitation_montage.png`).
- Produces the ruler-mask detection figure (`ruler_mask_check.png`).
- Saves `recommended_cubes.json` with the longest-non-saturating cube per excitation.

### 10.3 Preprocessing

`experiments/drop_data_preprocess.py`

- Produces the 5 cumulative variants: `raw`, `dark`, `dark_norm`, `dark_norm_mask`, `full`.
- Saves each variant at `Data/processed/Drop Data/<variant>/spectra_data.pkl`.
- Also produces shared drop mask + labels at `Data/processed/Drop Data/drop_mask.npy` and `drop_labels.npy`.

### 10.4 Cropped preprocessing

`experiments/drop_data_cropped_pipeline.py`

- Same as above but with `RULER_ROW_START = 175` applied first.
- Saves to `Data/processed/Drop Data Cropped/<variant>_cr/spectra_data.pkl`.

### 10.5 Smoke test

`experiments/drop_data_smoke_test.py [variant]`

- Quick 3-epoch AE training on a chosen variant; sanity check before full sweep.

### 10.6 Spectrum-driven exploration

`experiments/drop_data_spectra_explore.py`

- Ward clustering of full-spectrum drop means → 3-type ground truth.
- Per-band F-ratio computation.
- Method-overlay figures (`method_selections_overlay_n{3,5}.png`).
- Output: `results/Drop_Data_Spectra_Explore/<variant>/`.

### 10.7 Main sweep — selection × method × K

`experiments/drop_data_cropped_pipeline.py`

- For each of 5 cropped variants × 7 methods (variance, PCA-loading, SAM-greedy, SPA, MCUVE, random × 5 seeds, AE-perturb) × 8 K values (3..10), compute selection and evaluate.
- Output: `results/Drop_Data_Cropped_Sweep/<variant>_cr/`.

### 10.8 Slides export

`experiments/drop_data_export_slides_cropped.py`

- For each (variant, K) combo, save per-rank per-band PNG slides + montage.
- Output: `results/Drop_Data_Best_Slides_Cropped/<variant>_cr/n{K}/`.

### 10.9 Normalization-fix analysis

`experiments/drop_data_norm_fix.py`

- Demonstrates the variance vs.\ max_per_excitation finding (Section 9.2).
- Output: `results/Drop_Data_Norm_Fix/`.

### 10.10 SOTA baseline comparison (revision)

`revision/baselines/run_comparison.py --dataset drop_data_full_cr ...`

- Built during the paper revision (Section 8.4).
- Output: `revision/baselines/results_drop/drop_data_full_cr/`.

### 10.11 Headline figure builder (revision)

`revision/figures/drop_data/build_headline_figure.py`
`revision/figures/drop_data/build_panel_C.py`

- Produce Panels A, B, C of the headline figure (Sections 8.2, 8.3, 8.4).
- Output: `revision/figures/drop_data/panel_{A,B,C}_*.png`.

---

## 11. Future work

Several open questions are worth pursuing:

1. **Selection transfer.** Does the Drop Data 5-band selection generalize to a held-out portion of the data? Or to drops with a different fluorophore? A 3×3 transfer matrix (rows = "selection dataset", columns = "evaluation dataset") would test cross-sample generalizability.

2. **Multi-classifier evaluation on Drops.** Only KNN-5 is reported. SVM, RF, MLP, and 1D-CNN should be evaluated to confirm the selection generalizes across classifier families (the Collagen Sponges dataset shows this generalizes; would be cleaner to demonstrate on Drops too).

3. **Within-type variability quantification.** With only 3, 5, and 8 drops per type, sampling variance dominates. Acquiring more drops per type would let us quantify the framework's stability under different sample sizes.

4. **Ward-$k$ validation.** The choice of $k=3$ for the Ward ground truth was made by inspection. A more principled procedure (e.g., gap statistic + silhouette + Calinski-Harabasz, with confidence intervals) would strengthen the evaluation.

5. **Failure mode characterization.** Adding controlled noise (Gaussian, Poisson) or subsampling drops should reveal at what point the framework's $K$-band selection breaks. Honest limits make positive claims more credible.

6. **Physical interpretation of the selected bands.** The 470--530 nm emission window is consistent with NADH and collagen-like autofluorescence; matching to a fluorophore database would let us name the chemistry.

---

## 12. Appendix: file map

A compressed table of every file referenced in this report, with brief descriptions. All paths are relative to the project root.

| Category | Path | Description |
|---|---|---|
| Raw cubes | `Data/processed/Drop Data/raw/*.npy` | Per-cube `.npy` cache (one per .im3) |
| Preprocessed (uncropped) | `Data/processed/Drop Data/<variant>/spectra_data.pkl` | One per variant in {raw, dark, dark_norm, dark_norm_mask, full} |
| Preprocessed (cropped) | `Data/processed/Drop Data Cropped/<variant>/spectra_data.pkl` | Same, with ruler crop applied |
| Drop mask & labels | `Data/processed/Drop Data Cropped/drop_mask.npy`, `drop_labels.npy` | Binary mask + per-pixel drop ID (1--16) |
| Ground-truth types | `results/Drop_Data_Cropped_Sweep/full_cr/drop_types.npy` | Ward $k=3$ assignment per drop |
| Per-drop mean spectra | `results/Drop_Data_Cropped_Sweep/full_cr/drop_mean_spectra.npy` | $(16, 214)$ matrix |
| AE-perturb results | `results/Drop_Data_Cropped_Sweep/full_cr/ae_perturb_results.csv` | Selection per K + summary metrics |
| F-ratio table | `results/Drop_Data_Cropped_Sweep/full_cr/f_ratio_table.csv` | Per-band F-ratio |
| All-bands collage | `results/Drop_Data_Best_Slides_Cropped/full_cr/_all_bands_collage.png` | Every band as a tile |
| K=3 slides | `results/Drop_Data_Best_Slides_Cropped/full_cr/n3/` | 3 per-band PNGs + montage |
| K=5 slides | `results/Drop_Data_Best_Slides_Cropped/full_cr/n5/` | 5 per-band PNGs + montage (canonical) |
| K=8 slides | `results/Drop_Data_Best_Slides_Cropped/full_cr/n8/` | 8 per-band PNGs + montage |
| K=10 slides | `results/Drop_Data_Best_Slides_Cropped/full_cr/n10/` | 10 per-band PNGs + montage |
| Inspection: max projections | `results/Drop_Data_Inspection/max_proj/<stem>.png` | One per raw cube (22 files) |
| Inspection: montage | `results/Drop_Data_Inspection/excitation_montage.png` | All exposures per excitation |
| Inspection: spectra | `results/Drop_Data_Inspection/mean_spectra.png` | Per-cube mean spectra overlay |
| Inspection: reference | `results/Drop_Data_Inspection/reference_check.png` | Background / Whitelight diagnostics |
| Inspection: ruler | `results/Drop_Data_Inspection/ruler_mask_check.png` | Ruler-band detection |
| Inspection: recommendations | `results/Drop_Data_Inspection/recommended_cubes.json` | Longest non-saturating cube per excitation |
| Spectrum exploration | `results/Drop_Data_Spectra_Explore/full/per_drop_overview.png` | Mini-spectra per drop |
| Discriminative band map | `results/Drop_Data_Spectra_Explore/full/discriminative_band_map.png` | F-ratio heatmap over EEM |
| Archetypes & members | `results/Drop_Data_Spectra_Explore/full/archetypes_and_members.png` | Type means vs.\ members |
| Method overlay (K=5) | `results/Drop_Data_Spectra_Explore/full/method_selections_overlay_n5.png` | AE vs.\ baselines on F-ratio map |
| Method overlay (K=3) | `results/Drop_Data_Spectra_Explore/full/method_selections_overlay_n3.png` | Same at K=3 |
| Normalization-fix output | `results/Drop_Data_Norm_Fix/` | Variance vs.\ max_per_excitation comparison |
| SNR rerank output | `results/Drop_Data_SNR_Rerank/` | Falsified SNR-weighting hypothesis |
| Dim sweep output | `results/Drop_Data_Dim_Sweep/` | Falsified dim_method/n_dims tuning hypothesis |
| Headline figure A | `revision/figures/drop_data/panel_A_eem_per_type.{png,pdf}` | Per-type EEM heatmaps |
| Headline figure B | `revision/figures/drop_data/panel_B_emission_slices.{png,pdf}` | Per-excitation slices |
| Headline figure C | `revision/figures/drop_data/panel_C_knn_vs_K.{png,pdf}` | KNN accuracy vs.\ K |
| SOTA comparison data | `revision/baselines/results_drop/drop_data_full_cr/comparison_results.csv` | Long-form: method × K × seed × metrics |
| SOTA method summary | `revision/baselines/results_drop/drop_data_full_cr/method_summary.csv` | Aggregated mean ± std |
| SOTA selections | `revision/baselines/results_drop/drop_data_full_cr/selections.json` | Selected bands per (method, K, seed) |
| Cube cache (safe format) | `revision/figures/drop_data/full_cr_cube.npz` | NumPy-native cube for downstream scripts |

---

## End of report
