# Poster — paste-ready texts (matched to DraftPosterPresentation2026.pdf)

Each block below corresponds to a labelled region on the draft. Texts are
sized to fill the existing block without overflow at 11 pt body / 9 pt
captions. Headings stay the green pill labels you already have.

---

## TITLE BAR  (top, full width)

```
Autoencoder-Based Perturbation Analysis Framework for
Generalizable Dimensionality Reduction in ME-HSI

Narek Meloyan(a,c) · Narine Sarvazyan(a,b,c)

(a) American University of Armenia, Yerevan, Armenia
(b) George Washington University, Washington, DC, USA
(c) L. A. Orbeli Institute of Physiology NAS RA, Yerevan, Armenia
```

---

## BLOCK 1 · Introduction

```
Hyperspectral imaging (HSI) records a full reflectance, emission, or
excitation spectrum at every pixel. Adding a varying excitation
wavelength turns HSI into a four-dimensional modality — multi-excitation
HSI (ME-HSI) — that resolves mixed fluorophores ordinary HSI cannot
separate.

A single ME-HSI scan stores two spatial axes coupled to two spectral
axes (excitation × emission), typically tens of excitations against
tens of emission bands per pixel. The resulting cube is information-
rich but heavy: data volumes climb into the gigabyte range and only
a small subset of (excitation, emission) pairs actually carries
sample-distinguishing signal.

The challenge addressed here is identifying that subset
automatically, without per-sample expert tuning, and in a way that
generalises across biologies.
```

---

## BLOCK 2 · Problem Statement

```
ME-HSI cubes are simultaneously large and redundant. Many
(excitation × emission) layers carry near-duplicate information; many
others carry only noise.

Classical band-selection methods — variance ranking, PCA loadings,
MCUVE, SPA, SAM-greedy — operate on marginal or projection-based
statistics. They treat bands independently and never see the joint
4-D coupling that fluorescence physics actually imposes, so they
miss diagnostic (excitation, emission) pairs whose individual bands
look unremarkable.

A useful selector must (i) be data-driven, (ii) transfer across
samples, and (iii) reason about the full excitation × emission
plane jointly.
```

---

## BLOCK 3 · Novelty and Innovation

```
We introduce an autoencoder-based perturbation analysis framework
that selects bands by causal influence rather than marginal
statistics.

A 3-D convolutional autoencoder with one encoder/decoder branch per
excitation learns a 20-dimensional shared latent representation of
the full ME-HSI cube. The top-k variance-ranked latent dimensions
are then perturbed (z'_d = z ± ε σ_d e_d) and the resulting
reconstruction change at every (λex, λem) is recorded as that band's
causal sensitivity.

Aggregating sensitivities across dimensions yields an influence
matrix that Maximum-Marginal-Relevance (λ = 0.5) ranks for relevance
and diversity, returning a compact, sample-agnostic band subset.
Clustering directly in the latent space (e.g., KMeans) provides an
additional unsupervised analysis path.
```

---

## MODELING (centre — most important block)

This region uses one **section heading** + a **left bullet panel** + two
**figure captions** (Fig 1 and Fig 2). Keep the existing heading pill.

### Section heading
```
Modeling — 3D Convolutional Autoencoder + Latent Perturbation
```

### Left bullet panel (replaces the existing left-side body)

```
3D Convolutional Autoencoder for Hyperspectral Analysis

The pipeline (Fig. 1) runs in three stages. Stage 1: a parallel-
branch 3-D CAE encodes the preprocessed cube (Rayleigh cutoff +
normalization) into a shared 20-dim latent z ∈ ℝ²⁰. Stage 2:
top-k latent dimensions are perturbed and the reconstruction change
yields per-band influence (Fig. 2). Stage 3: MMR (λ = 0.5) ranks
the influence matrix into a diverse, ordered band list.

Architecture design choices
• Convolutional layers preserve spatial structure with far fewer
  parameters than fully connected baselines.
• 3-D convolutions capture spatial × spectral couplings in one
  operation; 2-D conv treats spectral bands as independent.
• Autoencoder framework learns a non-linear mapping optimised for
  reconstruction — the bottleneck forces it to keep what is
  diagnostic and discard what is redundant.
• Shared latent space: averaging excitation-branch features into one
  latent surfaces correlations between excitation responses that a
  per-excitation latent would hide.

Optimised parameters
• 20 encoding filters — fewer underfit, more produced noisy
  reconstructions.
• Sigmoid activations preserve subtle intensity variations better
  than ReLU on this data.
• 5 × 5 kernels balance spatial-pattern capture against boundary
  smoothing.

Pipeline notes
• Training proceeds until any of the configured loss / patience /
  epoch thresholds is met.
• Reconstructions are manually inspected before any downstream
  analysis (clustering, perturbation, MMR).
```

### Fig 1 caption (under the U-shape pipeline)
```
Fig. 1 — End-to-end methodology. Top row (left → right): 4D ME-HSI
cube → preprocessing → parallel encoder branches → 20-dim shared
latent → parallel decoder branches. Bottom row (right → left):
latent perturbation → wavelength influence scores → MMR diversity
selection → final selected wavelengths. The brace marks the
autoencoder portion that is trained end-to-end; everything below
the latent is a frozen analysis loop.
```

### Fig 2 caption (under the perturbation triptych)
```
Fig. 2 — Latent perturbation produces a wavelength-influence map.
(1) The top-k variance-ranked latent dimensions z₀, z₁, …, z_{k}
are perturbed by z'_d = z ± ε σ_d e_d. (2) Each perturbation is
decoded; the per-band absolute reconstruction change |ΔX̂(λex, λem)|
is the dimension's sensitivity. (3) Aggregating across the top-k
dimensions yields the influence matrix fed to MMR; white circles
mark the bands MMR ultimately selects.
```

---

## ROBUSTNESS ANALYSIS (right of Modeling, under the perturbation figure)

The histogram is already in place. Add a short body underneath:

```
Robustness Analysis

We compare the learned band selection against 1,000 random
selections of the same K under an identical kNN classifier.
The random distribution centres at 46.1 % (median 44.9 %),
while the learned bands score 90.2 % — well above the random
tail. The selector therefore does not simply benefit from any
"good enough" small subset; the perturbation-driven ranking is
load-bearing.
```

### Histogram caption
```
Fig. 3 — Classification accuracy under 1,000 random K-band
selections (blue) versus the perturbation-driven selection
(magenta arrow). Mean / median of the random distribution are
shown for reference.
```

---

## LICHENS DATASET (full-width row, left → right: ROI · heatmap · envelope)

### Section heading
```
Case 1 — Lichens (TPAMI)
```

### Section body (one-line strap above the figures, optional)
```
Eight-class substrate identification on a 192-band ME-HSI cube.
spectral-select retrieves a compact band subset that matches —
and at K ≥ 9 exceeds — the full-cube baseline.
```

### Caption — ROI overlay (left)
```
Fig. 4a — Class layout. Four substrate types × four instances
each; the white square marks the per-class training ROI. The
remaining 12 thalli are held-out test material.
```

### Caption — Wavelength heatmap (centre)
```
Fig. 4b — Selection frequency across (excitation, emission).
Hotter cells are the (λex, λem) pairs picked most often across
runs. Energy concentrates near 365 / 490 and 400 / 530 nm,
consistent with chlorophyll and secondary-metabolite emission.
```

### Caption — Accuracy envelope (right)
```
Fig. 4c — kNN accuracy vs. number of selected bands. Shaded
band shows min–max across seeds; black is the mean; green star
marks the peak (95.23 % at K = 80). Dashed red is the full-cube
baseline (88.15 %). Selection matches baseline at K = 4 and
exceeds it from K = 9 onward.
```

> **NOTE — verify before print:** the draft currently shows
> *Peak 85.59 %, Baseline 79.78 %* on this row, which are
> Pepsin–Collagen numbers. Replace with the Lichens envelope
> (Peak 95.23 %, Baseline 88.15 %).

---

## COLLAGEN-PEPSIN DATASET (full-width row, same three figures)

### Section heading
```
Case 2 — Collagen–Pepsin (IASIM 2026)
```

### Section body
```
Treatment-detection task on a 158-band cube (three classes,
three instances each). The same pipeline, with no per-sample
re-tuning, finds a 5-band subset that matches the full cube
and a 30-band subset that beats it by +5.8 percentage points.
```

### Caption — ROI overlay (left)
```
Fig. 5a — Class layout for the Pepsin dataset. Three classes
(red, blue, green) with three instances each; white squares
mark per-class training ROIs.
```

### Caption — Wavelength heatmap (centre)
```
Fig. 5b — Selection frequency across (excitation, emission)
for the Pepsin dataset. Hatched columns are the Rayleigh-
cutoff regions excluded from analysis. Selected pairs cluster
around 325 / 530 and 365 / 490 nm — the spectral fingerprints
of collagen autofluorescence and pepsin-induced cleavage
products.
```

### Caption — Accuracy envelope (right)
```
Fig. 5c — kNN accuracy vs. K. Shaded band: min–max across
seeds. Green star: peak 85.59 % at K = 30. Dashed red: full-
cube baseline 79.78 %. Selection beats baseline at K = 5 and
peaks well below the full cube's 158 bands.
```

> **NOTE — verify before print:** the draft currently shows
> *Peak 95.23 %, Baseline 88.15 %* on this row, which are
> Lichens numbers. Replace with the Pepsin envelope
> (Peak 85.59 %, Baseline 79.78 %).

---

## NEW · MERGED RESULTS TABLE (proposed strip between the two dataset rows)

A 2-row × 7-column strip. Same column structure as the XLSX file
(`Showcase_Poster/POSTER_v2_results_table.xlsx`) but trimmed for
poster width. Recommended copy:

### Heading
```
Headline results
```

### Table

| Dataset | Total bands | K | Compression | Acc. (full) | Acc. (selected) | Δ (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Lichens (TPAMI)         | 192 | 13 | 93.2 % | 0.882 | 0.902 | **+2.0** |
| Collagen–Pepsin (IASIM) | 158 |  5 | 96.8 % | 0.798 | 0.808 | **+1.1** |

### One-line caption under the table
```
Same network, same selection rule, two unrelated samples. Selection
matches or beats the full cube on both — at < 7 % of the bands.
```

> If you'd rather keep the strip thin, drop columns *Total bands* and
> *Compression* and lead with *K*, *Δ accuracy*, and a one-line note.

---

## CONCLUSION AND NEXT STEPS

```
We presented an autoencoder-based perturbation framework for
high-resolution unsupervised analysis of 4-D ME-HSI cubes. A
3-D CAE with parallel excitation branches learns a shared
20-dim latent; perturbation of the top-k latent dimensions
yields a per-band influence matrix that MMR ranks into a
diverse, ordered band list. On both Lichens (TPAMI, 192
bands) and Collagen–Pepsin (IASIM 2026, 158 bands), the same
pipeline matches the full-cube baseline at single-digit K
and exceeds it at K ≈ 13 (Lichens) and K ≈ 30 (Pepsin).

Next steps. (i) Blind unsupervised validation on a new "Drop
Data" scan with no annotations. (ii) Cross-instrument
transfer: train on Maestro, deploy on a custom 5-band filter
wheel. (iii) Open-source release of the spectral-select
package and reference notebooks.
```

---

## ACKNOWLEDGMENTS  *(unchanged)*

```
Financial support of the European Union (ERA Chair NAS-SAR award)
is gratefully acknowledged.
```

---

## REFERENCES  *(unchanged + recommended additions)*

```
[1] Bank D., Koenigstein N., & Giryes R. (2020). Autoencoders.
    arXiv:2003.05991.
[2] Jiang Z. et al. (2023). Autoencoders and their applications
    in machine learning: a survey. Artificial Intelligence Review.
[3] Jiang Y., Sha H., Liu S., Qin P., & Zhang Y. (2023). AutoUnmix:
    an autoencoder-based ...
[4] Carbonell J. & Goldstein J. (1998). The use of MMR, diversity-
    based reranking for reordering documents and producing
    summaries. SIGIR.
[5] Plaza A. et al. (2009). Recent advances in techniques for
    hyperspectral image processing. Remote Sensing of Environment.
```

---

# Layout-level recommendations

1. **Swap the two right-column envelope figures** between the Lichens
   and Collagen–Pepsin rows (numbers in the draft are crossed).
2. **Add a thin 2-row table strip** between the two dataset rows
   (text + table above). Compact and high-impact.
3. **Remove the green section-heading pills around figures** if
   you can — the pills duplicate the row heading and eat vertical
   space. The figure captions (Fig. 4a–c, 5a–c) carry the same
   information without taking a full bar.
4. **Tighten the modelling text** — current bullets repeat between
   "Architecture Design Choices" and "Novelty and Innovation". The
   bullet panel above is already trimmed; if space stays tight,
   drop the *Pipeline notes* group and keep only *Architecture* +
   *Optimised parameters*.
5. **Title-bar affiliations** read awkwardly with the inline a/b/c
   superscripts. The reformatted version above (one affiliation per
   line) is easier to parse from a meter.
