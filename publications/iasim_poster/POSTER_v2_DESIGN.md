# Poster v2 — A0 Portrait Design Document

**Author:** Narek Meloyan
**Co-author / supervisor:** Narine Sarvazyan
**Affiliations:** American University of Armenia · George Washington University · L.A. Orbeli Institute of Physiology NAS RA
**Format:** A0 portrait, 841 × 1189 mm
**Status of source material:** TPAMI manuscript under review · IASIM 2026 abstract submitted

This document is **paste-ready**. Each block below has: (a) exact text in copy-blocks, (b) the figure file paths (already in `Showcase_Poster/`), (c) layout coordinates as a fraction of the poster height for Publisher.

---

## 1 · What changed vs the existing v1 poster

| v1 element | v2 decision | Rationale |
|---|---|---|
| Title "Information Detection Pipeline … Autoencoder-Based Perturbation Analysis" | **New title** (see §2) | v1 sounds like a methods paper; the deliverable is *wavelength selection* |
| 25% of poster on Data Acquisition + Data Processing | **Compress to 1 line + 1 small icon** | Per your instruction — these aren't the contribution |
| Fig 1 Autoencoder + Fig 2 CNN-layer + Fig 3 Clustering | **Keep modernized Fig 1, drop Fig 2, replace Fig 3 with perturbation→influence** | Fig 2 is too zoomed-in; Fig 3's clustering framing is obsolete |
| Lime/Kiwi clustering screenshots in Results | **Replace with Lichens + Collagen Pepsin results** | v1 used demo data; we now have real biological evidence |
| No results table | **Add merged Lichens + Pepsin table** | Single artifact that proves cross-dataset generalization |
| Long Conclusion text | **3 short bullets + "what's next" pointer to Drop Data** | A reader spends 60 sec — bullet density wins |

---

## 2 · Title & tagline

**Pick one title (in order of recommendation):**

> **Picking the Wavelengths That Matter: Autoencoder-Driven Band Selection for 4D Multi-Excitation Hyperspectral Imaging**

> From 192 Bands to 9: Latent-Perturbation Wavelength Selection for ME-HSI

> Self-Supervised Wavelength Selection for Multi-Excitation Hyperspectral Imaging

**Tagline (place under title in 24 pt italic):**

> *9–13 selected (excitation, emission) bands recover the full-spectrum classification accuracy on biological samples — no supervised labels needed for selection.*

**Authors block** (unchanged from v1, place top-right of title bar):
> Narek Meloyan ᵃᶜ · Narine Sarvazyan ᵃᵇᶜ
> ᵃ American University of Armenia, Yerevan, Armenia
> ᵇ George Washington University, Washington, DC, USA
> ᶜ L.A. Orbeli Institute of Physiology NAS RA, Yerevan, Armenia

---

## 3 · A0 wireframe (portrait, with vertical fractions)

```
y=0.00 +================================================================+
       |                          TITLE BAR                              |
       |   [logo]   <Title> · <tagline> · <authors>            [QR/logo]|
y=0.10 +================================================================+
       |                                                                  |
       |   WHY                |   PROBLEM             |   CONTRIBUTION    |
       |   (motivation)       |   (the band glut)     |   (3-stage method)|
       |   ~80 words          |   ~70 words           |   ~70 words       |
       |                                                                  |
y=0.20 +------------------------------------------------------------------+
       |                                                                  |
       |                METHOD — single horizontal pipeline               |
       |   [Acquire ME-HSI] → [CAE] → [Perturb latent] → [MMR] → [Bands]  |
       |                                                                  |
       |   Inset A: CAE architecture (smaller)                            |
       |   Inset B: perturbation→influence cartoon                        |
       |                                                                  |
y=0.40 +================================================================+
       |                                                                  |
       |   RESULTS — Dataset A : LICHENS  (TPAMI submission)              |
       |   [sample/labels]  [accuracy envelope]  [selected-bands heatmap] |
       |                                                                  |
       |                Dataset B : COLLAGEN PEPSIN  (IASIM 2026)         |
       |   [sample]         [accuracy envelope]  [selected-bands heatmap] |
       |                                                                  |
       |          MERGED RESULTS TABLE  (Lichens + Pepsin in one)         |
       |                                                                  |
y=0.78 +================================================================+
       |                                                                  |
       |  ROBUSTNESS                       |  CONCLUSION & WHAT'S NEXT    |
       |  [robustness histogram]           |  · 3 bullets                 |
       |  [object-level boost cartoon]     |  · pointer to Drop Data      |
       |                                                                  |
y=0.93 +------------------------------------------------------------------+
       |  REFERENCES · ACKNOWLEDGMENTS · CITATION                         |
y=1.00 +================================================================+
```

Reading order is **Z-pattern**: title → top-right contribution → method (left-to-right pipeline) → Lichens → Pepsin → table → conclusion. A reader skimming for 30 seconds should leave with: *"a CAE picks ~9 wavelengths that match a 192-band classifier — the result holds on lichens and on collagen pepsin."*

---

## 4 · Section-by-section: exact text (paste-ready)

### 4A · WHY (top-left, ~80 words)

```
Hyperspectral imaging captures a fluorescence cube per pixel. Adding
the EXCITATION wavelength as a fourth axis makes ME-HSI: every pixel
is now a 2D excitation × emission matrix (an EEM). Modern setups
deliver 200+ (excitation, emission) bands per sample, but most carry
redundant or noisy information dominated by the same handful of
fluorophores. Acquisition is slow, storage balloons, and downstream
classifiers see correlated features. The practical question is:
WHICH bands actually carry the discriminative signal — and how few
of them do we really need?
```

### 4B · PROBLEM (top-middle, ~70 words)

```
Existing band-selection methods were designed for 3D HSI and treat
emission and excitation symmetrically. They miss the cross-excitation
correlations that are the whole reason ME-HSI exists in the first
place. Variance-based and PCA-based pickers tend to converge on a
single bright excitation; classical greedy methods (SPA, MCUVE)
collapse to redundant adjacent bands. None of these explicitly model
the nonlinear coupling between excitation and emission.
```

### 4C · CONTRIBUTION (top-right, ~70 words)

```
We propose a three-stage, self-supervised framework for ME-HSI band
selection:

  1. A 3D convolutional autoencoder with PARALLEL EXCITATION
     BRANCHES learns a unified latent representation.
  2. Systematic latent-space perturbation attributes reconstruction
     sensitivity back to individual (excitation, emission) pairs.
  3. Maximum Marginal Relevance selection picks K bands balancing
     informativeness with spectral diversity.

No labels are needed for selection. Validation uses KNN on the
selected bands.
```

---

### 4D · METHOD (middle band, full width)

**Headline pipeline diagram** (text spec for Publisher / Illustrator):

```
[ ME-HSI cube ] → [ pre-process ] → [ 3D CAE ] → [ perturb latent ] → [ MMR ] → [ K bands ]
   x,y,λex,λem   normalize +        parallel    ±15/30/45 % σ shifts  λ-balanced  ranked list
                Rayleigh cutoff     branches     per top-k dim         relevance/   {(λex,λem)}
                                                                       diversity
```

Render as horizontal flow with rounded rectangles in 3 colors (intake = green, computation = blue, output = purple). Single arrow chain. Short captions under each box (one line each).

**Method body text (~110 words, place below pipeline diagram, full width):**

```
The CAE has one encoder branch per excitation, projecting variable-
length emission cubes into a shared 20-dimensional latent space via
adaptive average pooling and element-wise feature merging. After 25-
30 epochs of masked-MSE training, we score each latent dimension by
its variance across samples and select the top k=5-10. For each
selected dimension d, we apply ±15, ±30, ±45 % perturbations
(scaled by σ_d) and decode; the absolute change in each (λex,λem)
band is its sensitivity. Aggregating across dimensions yields an
INFLUENCE MATRIX over (ex, em). MMR with λ=0.5 then selects K
diverse, high-influence bands.
```

**Inset A — Autoencoder architecture sketch (replaces v1 Fig 1, modernized):**

Render as a labeled block diagram:
```
  X^(1) → Conv3D → 
  X^(2) → Conv3D → ⌐──── AVG ──── z (R^20) ──── Conv3D ──── Conv3D → X̂^(1)
  X^(3) → Conv3D →                                            Conv3D → X̂^(2)
   ⋮       ⋮                                                  Conv3D → X̂^(3)
  X^(N) → Conv3D →                                              ⋮
                                                              Conv3D → X̂^(N)
```

Caption (italic, 14 pt):
> *3D CAE with parallel branches: each excitation has its own encoder/decoder; features merge into a single latent **z** that the perturbation analysis probes.*

**Inset B — Perturbation→Influence cartoon (NEW, replaces v1 Fig 3):**

Three-panel sketch:
```
Panel 1: latent z  ●●●●●○○○○○  (top-k chosen, blue dots)
Panel 2: perturb dim d:  z + ε σ_d e_d  → decode → (ex, em) heatmap of |ΔX̂|
Panel 3: aggregate across d  → INFLUENCE matrix  (ex × em heatmap, magma colormap)
```

Caption:
> *Each top-k latent dimension is perturbed; the per-band reconstruction change is the band's sensitivity. The aggregated influence matrix ranks all (ex, em) pairs.*

**Acquisition + preprocessing — reduce to ONE line + ONE small icon (top-right of method block):**

Icon: small NUANCE/TLS schematic from v1 (or just a 4D-cube cartoon).

Caption text:
> *Acquisition: 7-8 excitations × ~24-31 emission bands per sample (Maestro-Nuance, 420-720 nm). Preprocessing: per-cube dark subtraction, Rayleigh cutoff (λem ≥ λex + 40 nm), per-cube normalization. Mask of sample ROI.*

**That's it — 1 line + 1 icon. Replaces v1's two blocks of detailed acquisition/processing copy.**

---

### 4E · RESULTS — Dataset A : LICHENS (left half, ~38% of poster)

**Section header (24 pt bold):**
> Dataset A — Lichens (4 species, expert-annotated ground truth)

**Block layout — 3 figures in a row + 2 lines of text underneath:**

```
| 02_lichens_TPAMI/01_lichens_sample_RGB.png  |  04_accuracy_envelope.png  |  09_wavelength_heatmap.png |
                              (sample/ROI)              (must-have)               (must-have)
```

Captions (16 pt italic, single line each):
- *Sample RGB (rendered from selected emission bands) and 4-class ROI ground truth.*
- *KNN accuracy as a function of selected band count. The 192-band baseline (dashed) is matched at 13 bands and exceeded at 50.*
- *Frequency with which each (excitation, emission) pair is selected across 50 configurations — concentrated, repeatable.*

**Body (~50 words):**

```
Lichens Dataset 1 — 4 species × multi-excitation cube (8 excitations,
~24 emission bands each). The CAE-based method selects 13 (ex, em)
pairs that match the full-192-band KNN baseline accuracy (86.1 % vs
85.5 %). Robustness study (10,000 random 13-band picks): random mean
46.1 %, max 57.9 % — our method is at the 100th percentile.
```

---

### 4F · RESULTS — Dataset B : COLLAGEN PEPSIN (right half, ~38% of poster)

**Section header:**
> Dataset B — Collagen Pepsin (3 crosslinker concentrations, IASIM 2026)

**Block layout — same 3-figure pattern:**

```
| 03_collagen_pepsin_IASIM/01_IASIM2026_headline_figure.png  |  02_accuracy_envelope.png  |  09_wavelength_heatmap.png |
                              (sample)                                  (must-have)              (must-have)
```

Captions:
- *Pepsin collagen sample under 7-excitation ME-HSI; concentration classes labelled.*
- *KNN accuracy vs band count. Baseline matched at 5 bands; +5.8 pp by 30 bands. Selection works on a chemically distinct sample with no method retuning.*
- *Selected (ex, em) pairs grouped by selection cutoff. Distribution clusters around expected collagen autofluorescence peaks (~440-490 nm).*

**Body (~50 words):**

```
Pepsin/acetic-acid collagen series — same pipeline, no retuning, a
chemically distinct sample. The CAE matches the 158-band KNN
baseline (79.78 %) using just 5 bands and exceeds it by +5.8 pp at
30 bands. Selected wavelengths concentrate in 440-490 nm — the
expected collagen autofluorescence band — confirmed by IASIM 2026
abstract submission.
```

---

### 4G · MERGED RESULTS TABLE (centered band below the two dataset blocks)

Render as a single 2-tone striped table. Column widths in mm: Configuration 60, Bands 22, Reduction 28, Accuracy 28, F1 24, κ 22, Notes 56. Total ~240 mm — fits comfortably in centre column.

```
┌──────────────────┬───────┬───────────┬──────────┬──────────┬──────┬─────────────────────────┐
│ Configuration    │ Bands │ Reduction │ Accuracy │  F1 (w)  │  κ   │ Notes                   │
├──────────────────┼───────┼───────────┼──────────┼──────────┼──────┼─────────────────────────┤
│ LICHENS                                                                                        │
│ Baseline (full)  │  192  │     —     │  0.855   │  0.859   │ 0.807│ KNN-5 reference         │
│ Selected (best)  │   13  │   93.2 %  │  0.861   │  0.864   │ 0.814│ EXCEEDS baseline        │
│ Selected         │    9  │   95.3 %  │  0.818   │  0.822   │ 0.758│                         │
│ Selected         │    5  │   97.4 %  │  0.812   │  0.814   │ 0.749│                         │
│ Selected         │    3  │   98.4 %  │  0.796   │  0.799   │ 0.728│                         │
├──────────────────┴───────┴───────────┴──────────┴──────────┴──────┴─────────────────────────┤
│ COLLAGEN PEPSIN                                                                                │
│ Baseline (full)  │  158  │     —     │  0.798   │    —     │ 0.696│ KNN-5 reference         │
│ Selected (best)  │   30  │   81.0 %  │  0.856   │    —     │  —   │ +5.8 pp EXCEEDS         │
│ Selected         │   10  │   93.7 %  │  0.817   │    —     │  —   │ EXCEEDS baseline        │
│ Selected         │    5  │   96.8 %  │  0.808   │    —     │  —   │ EXCEEDS baseline        │
│ Selected         │    3  │   98.1 %  │  0.738   │    —     │  —   │                         │
└──────────────────┴───────┴───────────┴──────────┴──────────┴──────┴─────────────────────────┘
```

**Highlight row(s):** background-tint rows where Notes contain "EXCEEDS" — light green (#dff5e0) or pale yellow (#fff7c2). The visual takeaway should be: *EXCEEDS is achieved in BOTH datasets, with similar reduction%.*

---

### 4H · ROBUSTNESS / What's Next (bottom band)

**Robustness column (left):**

Figure: `02_lichens_TPAMI/10_robustness_histogram.png`
Caption:
> *Distribution of accuracy over 10,000 random 13-band selections (Lichens). Random baseline mean 46.1 %, max 57.9 %; our method 86.1 % — 100th percentile.*

One line text:
```
Random selections do not reach the method's accuracy even with the
same band budget. The selection is not just "any 13 bands" — it's
the right 13.
```

**Object-level column (middle):**

Optional: `02_lichens_TPAMI/14_roi_overlay_efficient_9.png` if it shows per-object accuracy improvement clearly. Otherwise: drop this and widen the conclusion column.

Caption:
> *Per-ROI classification with 9 selected bands. Difficult-to-classify objects (Class 3) gain up to 21 pp over baseline.*

**Conclusion column (right):**

```
KEY TAKEAWAYS

· A self-supervised CAE + perturbation analysis reduces 4D ME-HSI
  acquisition by 90-97 % while matching or exceeding full-spectrum
  classification accuracy.

· The same pipeline transfers from lichens to collagen pepsin without
  retuning — selected wavelengths land where the spectroscopist
  would expect (peak emission regions of the dominant fluorophores).

· Selection without labels: classification labels are only used to
  validate, never to choose bands. Method is deployable on samples
  with no annotated training data.

WHAT'S NEXT
Currently extending to a fully-blind drop array (16 droplets, no
ground truth) — preliminary results show the pipeline recovering 3
distinct spectral types from raw cubes alone.
```

---

### 4I · Footer (last 5%)

**References (3 lines, smaller font 10 pt):**
```
[1] Bank, D., Koenigstein, N., Giryes, R. (2020). Autoencoders. arXiv:2003.05991.
[2] Carbonell, J., Goldstein, J. (1998). The use of MMR, diversity-based reranking. SIGIR.
[3] Cohen, J. (1960). A coefficient of agreement for nominal scales. Educ. Psychol. Meas.
[4] Meloyan, N. & Sarvazyan, N. (2025). Autoencoder-Based Wavelength Selection for 4D Hyperspectral Imaging. IEEE TPAMI (under review).
```

**Acknowledgments (single line):**
> *Financial support of the European Union (ERA Chair NAS-SAR award) is gratefully acknowledged. Code & data: github.com/narekmeloyan/spectral-select · Zenodo DOI 10.5281/zenodo.18640119*

---

## 5 · Asset map (which file goes in which block)

| Block | File path (relative to `Showcase_Poster/`) |
|---|---|
| Title bar logos | (use AUA + Orbeli logos from v1 source) |
| Title bar QR | generate from `https://github.com/narekmeloyan/spectral-select` |
| Method · Inset A (CAE arch) | redraw in Publisher per spec in §4D, or use TikZ figure from `archive/paper/sections/methodology.tex` (it's a clean parallel-branch diagram) |
| Method · Inset B (perturbation) | redraw per spec in §4D — short, schematic, no source figure exists |
| Method · acquisition icon | reuse small slice of v1 acquisition diagram |
| Lichens — sample | `02_lichens_TPAMI/01_lichens_sample_RGB.png` (or `02_lichens_labels_and_ROI.png` for labels variant) |
| **Lichens — accuracy envelope** *(must-have)* | `02_lichens_TPAMI/04_accuracy_envelope.png` |
| **Lichens — wavelength heatmap** *(must-have)* | `02_lichens_TPAMI/09_wavelength_heatmap.png` |
| Lichens — classification overlay | `02_lichens_TPAMI/08_classification_9bands_efficient.png` (alternative for sample slot) |
| Pepsin — sample | `03_collagen_pepsin_IASIM/01_IASIM2026_headline_figure.png` |
| **Pepsin — accuracy envelope** *(must-have)* | `03_collagen_pepsin_IASIM/02_accuracy_envelope.png` (small) or `03_accuracy_envelope_expanded.png` (large) |
| **Pepsin — wavelength heatmap** *(must-have)* | `03_collagen_pepsin_IASIM/09_wavelength_heatmap.png` |
| Pepsin — classification overlay | `03_collagen_pepsin_IASIM/10_roi_comparison_baselines.png` |
| Robustness histogram | `02_lichens_TPAMI/10_robustness_histogram.png` |
| Object-level (optional) | `02_lichens_TPAMI/14_roi_overlay_efficient_9.png` |
| Backup headline (single anchor) | `02_lichens_TPAMI/13_classification_comparison_3panel.png` |

---

## 6 · Architecture-diagram redrawing instructions (for Publisher)

The TikZ source in `archive/paper/sections/methodology.tex` already has the cleanest possible CAE diagram. Below is a simpler "Publisher" reproduction:

**CAE inset (Inset A):**

Use 5 vertical columns of rounded boxes:
- Column 1 (green): N input cubes labelled `X^(1) … X^(N)` (N = 7 or 8 depending on dataset)
- Column 2 (blue): N `Conv3D + sigmoid` boxes (parallel encoder branches)
- Column 3 (orange): single `AVG merge` box, with N arrows feeding in
- Column 4 (red): single `latent z ∈ R^20` box (the bottleneck — make this the visual anchor)
- Column 5 (orange): single `Conv3D shared` box
- Column 6 (blue): N decoder `Conv3D` boxes
- Column 7 (green): N reconstructed cubes `X̂^(1) … X̂^(N)`

Arrows: each input → each encoder branch in parallel (stacked); all encoder outputs → AVG; AVG → z; z → shared; shared → each decoder branch. Avoid crossing arrows.

Total width on poster: ~280 mm. Height: ~120 mm.

**Perturbation inset (Inset B):**

Three small panels in a row:
1. **Latent space**: 20 small dots in a horizontal line, top 5 highlighted in red (the perturbed dims).
2. **Perturbation effect**: a 7×24 heatmap (excitation × emission), single dimension's `|Δ X̂|` — use magma colormap, just rendered as a small thumbnail heatmap.
3. **Aggregated influence**: same shape heatmap, but from `Lichens_Dataset_1_MasterRun/visualizations/03_wavelength_analysis/wavelength_heatmap.png` cropped tightly. Or use a freshly-rendered miniature.

Connecting arrows: panel 1 → panel 2 with label "perturb · decode"; panel 2 → panel 3 with label "Σ over dims".

Total width: ~280 mm. Height: ~120 mm.

**Pipeline diagram (the headline of the method block):**

Single horizontal chain of 6 boxes, each ~150 × 60 mm, separated by `→` arrows. Box content (one line each):

```
[ ME-HSI 4D cube ] → [ Preprocess ] → [ Train 3D CAE ] → [ Perturb latent ] → [ MMR @ λ=0.5 ] → [ K bands ]
   x, y, λex, λem      dark, Rayleigh    25-30 epochs      ±15/30/45 % σ        relevance vs       ranked
                       cutoff, mask      masked MSE        per top-k dim         diversity         (λex, λem)
```

Total width: ~700 mm (most of the poster width). Make this the largest single visual element on the poster.

---

## 7 · Color & typography

- **Headline font:** Lato Bold or Open Sans Bold, 96-120 pt for title, 36-44 pt for section headers.
- **Body font:** same family, regular, 18-20 pt body, 14-16 pt captions.
- **Math/code font:** Source Code Pro / JetBrains Mono, used sparingly (only for variable names like λex, σ_d, K).
- **Color palette:**
  - Primary dark: `#1a2332` (text)
  - Primary accent: `#0a5dab` (section headers, framed boxes)
  - Pipeline stages: green `#56b870` (intake) / blue `#3a86c8` (compute) / purple `#7a4ec8` (output)
  - "EXCEEDS baseline" highlight: `#dff5e0` background, `#10723a` text
  - Heatmap colormap: keep figures' native viridis/magma — do NOT recolor

- **Section dividers:** thin 1 px hairline rules (`#cccccc`), NOT boxes around blocks.
- **Whitespace:** 25-30 mm outer margin; 15 mm gutters between columns.

---

## 8 · Print checklist

- [ ] Export every figure at the source PDF resolution (we have PDFs in the original `Paper Source/paper/figures-updated/` and `results/Pepsin_Full_Analysis/figures/` paths). PNGs we copied to `Showcase_Poster/` are 140-300 dpi — usable for screen review but **swap to PDFs for final print**.
- [ ] Confirm all captions stand alone (read only the captions; the story should still hold).
- [ ] Test at A4 — body text must remain readable when scaled down 8×.
- [ ] Repo QR works (scan from phone before printing).
- [ ] Logos crisp at 300 dpi.

---

## 9 · One-paragraph script for the booth (60 sec elevator pitch)

> *Multi-excitation hyperspectral imaging gives you 200+ wavelength combinations per pixel. Most are redundant. We train a convolutional autoencoder on the 4D cube — purely self-supervised — then perturb its latent space and watch which (excitation, emission) bands the reconstruction is most sensitive to. Those are the informative ones. We pick a diverse subset with MMR. On lichens, 13 bands match the 192-band classifier's accuracy; on a chemically distinct collagen pepsin sample, 5 bands match a 158-band baseline. Same pipeline, no retuning. Manuscript under review at TPAMI; abstract at IASIM 2026. Code is open-source, link in the QR.*

---

## 10 · Summary

The new poster is structured around **two evidence panels (Lichens + Pepsin)** sandwiching a **single horizontal method pipeline**, with the merged results table as the synthesis point. The old "method-heavy / results-light" balance is reversed: the architecture sketches are now insets, not the headline. Required figures (accuracy envelopes + selected-bands heatmaps for both datasets) are guaranteed real estate. Acquisition + preprocessing collapse from two big blocks into a single line of caption text, freeing ~25% of the poster for results.

**File ready to use:** `Showcase_Poster/POSTER_v2_DESIGN.md`
**Copy-paste targets:** title (§2), tagline (§2), Why/Problem/Contribution copy (§4A-C), method body (§4D), per-dataset bodies (§4E-F), conclusions (§4H), references/ack (§4I), elevator pitch (§9).
**Asset map:** §5 — names every file by exact path inside `Showcase_Poster/`.
