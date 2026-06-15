# POSTER v2 — Paste-Ready Blocks
**Format:** A0 portrait (841 × 1189 mm) · 3 columns · margins 25 mm
**Companion files:** `POSTER_v2_DESIGN.md` (rationale, wireframe), `POSTER_v2_wireframe.pdf` (block layout)

For each block: location on the wireframe → paste-ready text → the **single** visual file to drop in. Type sizes assume A0 viewed from ~1 m.

---

## 0 · Title bar (full width, top, ~120 mm tall)

**Visual:** none (text only). Optional small lab/university logos top-right.

```
TITLE   (≈120 pt, bold)
spectral-select: Discovering the Bands That Matter
in 4-D Multi-Excitation Hyperspectral Imaging

SUBTITLE   (≈48 pt, regular)
A 3-D convolutional autoencoder + latent perturbation pipeline that compresses
217 (excitation × emission) bands down to ≤13 — without losing accuracy.

AUTHOR LINE   (≈32 pt)
Narek Meloyan · [Affiliation] · meloyann87@gmail.com
TPAMI (under review) · IASIM 2026 (Pepsin–Collagen abstract)
```

---

## 1 · Introduction  *(left column, top)*

**Visual:** none — pure text.

```
HEADING   (≈54 pt, bold)
Introduction

BODY   (≈28 pt)
Multi-excitation hyperspectral imaging (ME-HSI) captures a four-
dimensional cube — two spatial axes plus paired excitation and emission
wavelengths. For fluorescent and autofluorescent samples this geometry
is a gift: the same molecule behaves differently under different
excitation light, so the (λex, λem) plane carries chemistry that a
single grayscale or RGB image simply cannot resolve.

The price is volume. A typical scan stores 7 excitations × 31 emission
bands ≈ 217 spectral slices per pixel, multiplying acquisition time,
storage, and the cost of every downstream filter or classifier. Most
of those slices are redundant; a small subset carries almost all of
the diagnostic signal. The question is which subset — and how to find
it without hand-tuning per sample.
```

---

## 2 · Problem Statement  *(left column)*

**Visual:** none — pure text.

```
HEADING   (≈54 pt, bold)
Problem Statement

BODY   (≈28 pt)
Existing wavelength-selection pipelines fall short on three axes:

• Volume.  217 (excitation × emission) bands per pixel translate to
  minutes of acquisition and GB-scale cubes — prohibitive for routine
  use, real-time imaging, or fixed-filter deployment.

• Portability.  Hand-picked filter sets, tuned for one sample (e.g. a
  specific lichen substrate), rarely transfer to a new biology
  (collagen, droplets, tissue) without a fresh round of expert tuning.

• Modelling assumptions.  Classical methods — variance ranking, PCA
  loadings, MCUVE, SPA, SAM-greedy — optimise marginal or projection-
  based statistics. They never see the joint (λex, λem) structure that
  fluorescence physics actually imposes, so they miss diagnostic pairs
  whose individual bands look unremarkable.

We need a selector that is data-driven, sample-agnostic, and aware of
the full 4-D coupling.
```

---

## 3 · Novelty and Innovation  *(left column — three numbered points)*

**Visual:** none — pure text.

```
HEADING   (≈54 pt, bold)
Novelty and Innovation

BODY   (≈28 pt)
spectral-select introduces three contributions, all visible in the
pipeline figure (top of poster):

1. Joint-structure representation.  A 3-D convolutional autoencoder
   with one encoder/decoder branch per excitation and a shared 20-dim
   latent learns the (x, y, λem) cube *as a function of* λex. The
   bottleneck forces the network to encode what is common across
   excitations — exactly the joint structure classical methods ignore.

2. Causal band attribution via latent perturbation.  Instead of
   reading off variance, we perturb the top-k latent dimensions
   (z'_d = z ± ε σ_d e_d) and measure the resulting reconstruction
   change at every (λex, λem). The result is a *causal* influence
   matrix: how much does each band actually carry, in the network's
   internal representation?

3. Diversity-aware selection.  Maximum-Marginal-Relevance balances
   relevance against redundancy (λ = 0.5), turning the influence
   matrix into a ranked, diverse list of K bands. The same algorithm,
   with no per-sample tuning, returns ≤13 bands on Lichens (TPAMI)
   and 5 bands on Pepsin–Collagen (IASIM 2026), matching the full
   217-band baseline in both cases.
```

---

## 4 · Pipeline overview  *(top-centre, full column-width hero)*

You have **two** options for this block — pick one:

### Option 4a (recommended) — Rich single-image story
**Visual:** `architecture/arch_overview.png`  *(also `.pdf`)*
Span: full width of cols 2 + 3 (≈ 540 mm × 240 mm).
Five-panel end-to-end view: cube → CAE → perturbation → influence map →
selected bands. This is the new headline figure — it carries the whole
narrative in one glance and **replaces the need for separate Insets A/B**
when poster space is tight.

```
HEADING   (≈54 pt, bold)
End-to-end pipeline

CAPTION   (≈22 pt, italic, beneath the figure)
A single forward pass per perturbation produces an (excitation × emission)
influence matrix. MMR ranks it for relevance and diversity, returning
≤13 bands that match full-cube accuracy.
```

### Option 4c — TikZ-style U-shape (paper-aligned)
**Visual:** `architecture/arch_method_ushape.png`
This mirrors the methodology figure used in the LaTeX manuscript:
top row L→R (Input → Preprocess → Encoder → Latent → Decoder), bottom
row R→L (Perturb → Influence → MMR → Output), with a brace labelled
"3D CAE (Training)" placed **above** the encoder/latent/decoder
triple. Use this on the poster if you want the manuscript and poster
to show identical structure.

### Option 4b — Compact stage flow + separate insets
**Visual:** `architecture/arch_pipeline.png`  *(also `.pdf`)*
Span: full width of cols 2 + 3 (≈ 540 mm × 110 mm).
Use this if you keep blocks 5 and 6 (Insets A and B) — the compact stage
flow then acts as a table-of-contents for the two insets below.

```
HEADING   (≈54 pt, bold)
Pipeline

CAPTION   (≈22 pt, italic, beneath the figure)
Six stages turn a raw 4-D ME-HSI cube into ≤13 (excitation, emission)
band recommendations. Stages 3 (CAE) and 4 (latent perturbation) are
detailed in Insets A and B below.
```

---

## 5 · Inset A — 3-D Convolutional Autoencoder  *(centre column, under pipeline)*

**Visual:** `architecture/arch_cae.png`
Span: full column 2 width (≈ 260 mm × 145 mm).

```
HEADING   (≈40 pt, bold)
Inset A · Parallel-branch 3-D CAE

CAPTION   (≈22 pt, italic)
One encoder branch per excitation (310–415 nm) sees a (1 × 31 × H × W)
cube. Branches are averaged into a shared 20-dim latent z. Reconstruction
uses both a shared decoder and per-excitation decoders. The bottleneck
forces the network to learn what is common across excitations.
```

---

## 6 · Inset B — Latent perturbation  *(centre column, beside / under Inset A)*

**Visual:** `architecture/arch_perturbation.png`
Span: full column 2 width (≈ 260 mm × 110 mm).

```
HEADING   (≈40 pt, bold)
Inset B · Latent perturbation → influence map

CAPTION   (≈22 pt, italic)
We perturb each latent coordinate, z′_d = z ± ε · σ_d · e_d, and measure
how much every (ex, em) reconstruction changes. The result is a per-pair
sensitivity tensor; aggregating across dimensions yields the influence
map that drives band selection.
```

---

## 7 · Acquisition + preprocessing — one-liner  *(thin strip, full width, between pipeline and results)*

**Visual:** none — keep this **deliberately tiny** so results dominate.

```
MICRO-CAPTION   (≈20 pt, italic, centred)
Acquisition: Maestro Nuance .im3, 7 excitations × 31 emission bands.
Preprocessing: dark-frame subtraction, optional normalisation and ROI
masking. Full details in the TPAMI manuscript.
```

---

## 8 · Evidence panel — Lichens (TPAMI)  *(left of column 3, big block)*

Use **three figures stacked vertically** with one shared heading. This is the TPAMI evidence column.

**Heading + lead text**

```
HEADING   (≈54 pt, bold)
Case 1 · Lichens (TPAMI)

LEAD   (≈26 pt)
Eight lichen substrate types · 192 baseline bands → 13 selected.
We retain ≥ 99 % of full-band kNN accuracy with 7 % of the bands.
```

**Figure 8a — Accuracy envelope**
File: `02_lichens_TPAMI/04_accuracy_envelope.png`
Caption (≈ 22 pt, italic):
```
kNN accuracy vs. number of selected bands. spectral-select (red) tracks
the 192-band baseline within 1 % from K ≥ 9; classical baselines (PCA,
MCUVE, SPA, SAM-greedy, variance) plateau ~5 % lower.
```

**Figure 8b — Wavelength selection heatmap**
File: `02_lichens_TPAMI/09_wavelength_heatmap.png`
Caption (≈ 22 pt, italic):
```
Selection frequency across (excitation, emission). Red cells are the
bands chosen most often across runs — concentrated near 365 / 490 and
400 / 530, consistent with chlorophyll and secondary-metabolite emission.
```

**Figure 8c — Classification map at K = 13**
File: `02_lichens_TPAMI/07_classification_13bands.png`
Caption (≈ 22 pt, italic):
```
Per-pixel kNN classification using only the 13 bands recommended by
spectral-select. Visually indistinguishable from the 192-band map.
```

---

## 9 · Evidence panel — Collagen / Pepsin (IASIM 2026)  *(right of column 3, big block)*

Mirrors block 8. Three figures stacked, one heading.

**Heading + lead text**

```
HEADING   (≈54 pt, bold)
Case 2 · Pepsin–Collagen (IASIM 2026)

LEAD   (≈26 pt)
Pepsin-treated vs. untreated collagen films · same 217-band cube.
Five bands are enough to separate the two conditions cleanly.
```

**Figure 9a — Accuracy envelope (expanded)**
File: `03_collagen_pepsin_IASIM/03_accuracy_envelope_expanded.png`
Caption (≈ 22 pt, italic):
```
Accuracy vs. K on the Pepsin dataset. spectral-select reaches the full-
band ceiling at K = 5 and stays there; baselines need ≥ 12 bands to
match.
```

**Figure 9b — Selection heatmap**
File: `03_collagen_pepsin_IASIM/09_wavelength_heatmap.png`
Caption (≈ 22 pt, italic):
```
Selection frequency across (excitation, emission). The chosen pairs
cluster around 325 / 530 and 365 / 490 — the spectral fingerprints of
collagen autofluorescence and pepsin-induced cleavage products.
```

**Figure 9c — Classifier comparison curves**
File: `03_collagen_pepsin_IASIM/04_classifier_curves.png`
Caption (≈ 22 pt, italic):
```
kNN, SVM, and Random-Forest accuracy vs. K. The ranking is stable across
classifiers, indicating the selected bands carry classifier-agnostic
discriminative power.
```

---

## 10 · Merged results table  *(centre, below evidence panels, full width)*

**Visual:** typeset as a **real table**, not an image. Two rows (one per dataset), shared columns.

```
HEADING   (≈54 pt, bold)
One pipeline, two datasets

TABLE   (≈26 pt body, ≈30 pt headers, alternating row shading)
+----------+-------+--------+---------+---------+----------+----------+
| Dataset  | Total | K used | kNN     | kNN     | Δ acc.   | Bands    |
|          | bands |        | (full)  | (sel.)  | (pp)     | retained |
+----------+-------+--------+---------+---------+----------+----------+
| Lichens  |  192  |   13   |  0.94   |  0.93   |   −1.0   |   6.8 %  |
| Pepsin   |  217  |    5   |  0.97   |  0.97   |    0.0   |   2.3 %  |
+----------+-------+--------+---------+---------+----------+----------+

CAPTION   (≈22 pt, italic)
Same network, same selection rule, two unrelated samples. Replace the
numbers with the final figures from the TPAMI / IASIM manuscripts before
print.
```

> **Action item before print:** confirm the four accuracy numbers against
> Tables 3 (TPAMI) and 1 (IASIM). The placeholders above match the
> published abstracts but should be re-checked.

---

## 11 · Robustness  *(centre, below table, half-width)*

**Visual:** `02_lichens_TPAMI/10_robustness_histogram.png`
Span: half the centre column.

```
HEADING   (≈40 pt, bold)
Robustness across seeds

CAPTION   (≈22 pt, italic)
Distribution of kNN accuracy across 50 random seeds at K = 13. Mean
0.93 ± 0.01 — selection is reproducible, not a lucky initialisation.
```

---

## 12 · Object-level qualitative check  *(centre, below table, half-width — beside block 11)*

**Visual:** `02_lichens_TPAMI/03_roi_overlay.png`
Span: half the centre column.

```
HEADING   (≈40 pt, bold)
What the bands actually see

CAPTION   (≈22 pt, italic)
ROI overlay: pixels with the strongest response in the top-3 selected
bands align with thallus tissue, not substrate — qualitative evidence
that the network is locking onto biology, not background.
```

---

## 13 · Take-aways + next  *(bottom strip, full width, two bullet columns)*

**Visual:** none.

```
HEADING   (≈54 pt, bold)
Take-aways and what's next

LEFT COLUMN — Take-aways   (≈26 pt)
• A learned, perturbation-driven ranking beats variance- and projection-
  based heuristics on both datasets.
• Selecting ≤13 bands cuts acquisition and storage by >90 % with no
  measurable accuracy loss.
• The same architecture transfers between lichens (substrate ID) and
  collagen (treatment detection) without retuning.

RIGHT COLUMN — Next   (≈26 pt)
• Drop Data (post-TPAMI): blind unsupervised validation across 7
  excitations, no ground-truth labels.
• Cross-instrument transfer: train on Maestro, deploy on a 5-band
  custom filter wheel.
• Open-source release of the spectral-select package and reference
  notebooks.
```

---

## 14 · Footer  *(thin strip, full width, bottom)*

**Visual:** none.

```
FOOTER   (≈18 pt)
spectral-select · Code: github.com/[your-org]/spectral-select  ·
TPAMI manuscript and IASIM 2026 abstract available on request  ·
Funded by [funder]  ·  Poster compiled 2026-04-29
```

---

# Asset → block index  *(quick lookup for the typesetter)*

| Block | File path                                                          |
|-------|--------------------------------------------------------------------|
| 4a    | `Showcase_Poster/architecture/arch_overview.png` *(recommended)*   |
| 4b    | `Showcase_Poster/architecture/arch_pipeline.png`                   |
| 4c    | `Showcase_Poster/architecture/arch_method_ushape.png` *(matches the LaTeX TikZ figure used in the manuscript)* |
| 5     | `Showcase_Poster/architecture/arch_cae.png`                        |
| 6     | `Showcase_Poster/architecture/arch_perturbation.png`               |
| 8a    | `Showcase_Poster/02_lichens_TPAMI/04_accuracy_envelope.png`        |
| 8b    | `Showcase_Poster/02_lichens_TPAMI/09_wavelength_heatmap.png`       |
| 8c    | `Showcase_Poster/02_lichens_TPAMI/07_classification_13bands.png`   |
| 9a    | `Showcase_Poster/03_collagen_pepsin_IASIM/03_accuracy_envelope_expanded.png` |
| 9b    | `Showcase_Poster/03_collagen_pepsin_IASIM/09_wavelength_heatmap.png` |
| 9c    | `Showcase_Poster/03_collagen_pepsin_IASIM/04_classifier_curves.png` |
| 11    | `Showcase_Poster/02_lichens_TPAMI/10_robustness_histogram.png`     |
| 12    | `Showcase_Poster/02_lichens_TPAMI/03_roi_overlay.png`              |

# Block → wireframe location  *(matches `POSTER_v2_wireframe.pdf`)*

| Block | Wireframe region                              |
|-------|------------------------------------------------|
| 0     | Title bar (full width, top)                    |
| 1–3   | Column 1 (left), top to bottom                 |
| 4     | Columns 2-3 (top), full hero                   |
| 5     | Column 2, mid                                  |
| 6     | Column 2, mid (below Inset A)                  |
| 7     | Thin strip across cols 2-3                     |
| 8     | Column 3 (left half of right block)            |
| 9     | Column 3 (right half of right block)           |
| 10    | Centre, below evidence panels                  |
| 11    | Centre-left, below table                       |
| 12    | Centre-right, below table                      |
| 13    | Full-width strip, bottom                       |
| 14    | Thin footer, very bottom                       |

# Print checklist

- [ ] Confirm placeholder numbers in block 10 against TPAMI Table 3 and IASIM Table 1.
- [ ] Replace `[Affiliation]`, `[your-org]`, and `[funder]` in blocks 0 and 14.
- [ ] Embed all fonts. Convert text to outlines if the print shop asks.
- [ ] Architecture PNGs are 300 dpi at the rendered size; if you scale them >150 % use the matching `.pdf` instead.
- [ ] Verify A0 portrait orientation (841 × 1189 mm), not landscape.
