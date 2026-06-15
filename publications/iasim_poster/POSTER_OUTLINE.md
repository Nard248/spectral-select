# Research Showcase Poster — Outline, Text & Layout

**Author:** Narek Meloyan
**Project:** `spectral-select` — Autoencoder-based wavelength selection for 4D hyperspectral imaging
**Status:** TPAMI submission (under review) + IASIM 2026 abstract (submitted)

---

## 1 · Recommended layout (A0 portrait, 3 columns × 4-5 rows)

```
+================================================================+
|  TITLE BAR  (Author · Affiliation · Contact · QR to repo)      |
+================+===============+===============================+
|                |               |                                |
|  WHY           |  HOW          |  KEY RESULT                    |
|  (Motivation)  |  (Method)     |  (One headline figure)         |
|                |               |                                |
+----------------+---------------+--------------------------------+
|                                                                |
|  DATASET 1 — LICHENS  (paper validation, ground truth)         |
|  [sample RGB] [labels/ROI] [accuracy envelope] [13-band map]   |
|                                                                |
+----------------------------------------------------------------+
|                                                                |
|  DATASET 2 — COLLAGEN PEPSIN  (IASIM 2026)                     |
|  [sample] [accuracy envelope]  [efficiency]  [bands heatmap]   |
|                                                                |
+----------------+---------------+-------------------------------+
|  ROBUSTNESS    |  EFFICIENCY   |  WHAT'S NEXT                   |
|  (histogram)   |  (curves)     |  (Drop Data, blind validation) |
+----------------+---------------+-------------------------------+
|  REFERENCES · ACKNOWLEDGMENTS · CITATION (BibTeX QR)            |
+================================================================+
```

Reading order is **Z-pattern**: top-left → top-right (KEY RESULT magnet) → middle-left → bottom-right.
A reader who only spends 20 seconds should still walk away with: *"a CAE picks ~9 wavelengths that match a 192-band classifier — saves >95% of acquisition time."*

---

## 2 · Suggested title (pick one)

- **"Picking the Wavelengths That Matter: An Autoencoder-Driven Approach to 4D Hyperspectral Imaging"**
- **"From 192 to 9: Latent-Perturbation Wavelength Selection for Multi-Excitation Fluorescence"**
- **"Spectral-Select: Compressing 4D Hyperspectral Acquisition by 20× Without Losing Class Information"**

---

## 3 · 30-second pitch (for the abstract block)

> Multi-excitation hyperspectral imaging captures 100s of wavelength combinations per sample — a tax on acquisition time, storage, and analysis. We train a convolutional autoencoder on 4D fluorescence cubes and read out the bands its latent space is most sensitive to. Across **lichen** species and **collagen pepsin** preparations, ~9-13 selected (excitation, emission) bands recover the classification accuracy of the full 192-band acquisition. The method needs no labels — selection is driven entirely by the autoencoder's reconstruction sensitivity.

---

## 4 · Section-by-section content & figure picks

### 4A · WHY (motivation block)
**Body text (≤80 words):**
> Multi-excitation fluorescence cubes are 4D: (height × width × emission × excitation). A typical scan covers 7+ excitations × ~30 emission bands = 200+ images per sample. Most bands are redundant — driven by the same handful of fluorophores. The challenge is identifying *which* bands carry the discriminative information, **without supervised labels**.

**No figure required — keep this block dense text + one icon.** Optional: a small sample-cube cartoon (3D box with sliced bands).

---

### 4B · HOW (method block)
**Body text (≤100 words):**
> We train a convolutional autoencoder with a sparse latent (k=12 dimensions) on the full 4D cube. The encoder learns a compact representation; the decoder reconstructs the input from the latent. We then perturb each latent dimension by ±10/20/30 percentile shifts and measure the resulting change in reconstruction at every (excitation, emission) band. Bands whose reconstruction is most sensitive to latent perturbations are deemed most informative. Wavelengths are ranked by aggregated influence across important latent dimensions.

**Optional figure (you may need to sketch this):** Encoder → latent → decoder → per-band sensitivity. If no figure exists, use a short ASCII / textual schematic instead — this section is fine as text-only on a poster.

---

### 4C · KEY RESULT (anchor figure block)
**Pick ONE and make it large (~25% of poster area):**

| Asset | Why it's the headline |
|---|---|
| `02_lichens_TPAMI/04_accuracy_envelope.png` | The single most quotable figure — shows accuracy vs n_bands, highlights the "knee" at 9-13 bands matching the 192-band ceiling |
| `02_lichens_TPAMI/11_publication_figure.png` | Already designed as a publication anchor — multi-panel summary |
| `02_lichens_TPAMI/13_classification_comparison_3panel.png` | Side-by-side: 192-band vs 80-band vs 9-band classifications — extremely visual |

**Caption suggestion:**
> *Classification accuracy of K-NN trained on autoencoder-selected wavelengths from the Lichens dataset. The dashed line marks the 192-band baseline (full acquisition); the shaded envelope is the 95% range across 50+ configurations. Selecting 9 bands recovers >97% of baseline accuracy — a 20× compression of the acquisition matrix.*

---

### 4D · DATASET 1 — Lichens (the canonical validation)
**Recommended layout:** 2 rows × 2 figs.

| Position | Asset | Caption |
|---|---|---|
| Top-left | `02_lichens_TPAMI/01_lichens_sample_RGB.png` | *Lichens Dataset 1 — RGB rendering from selected emission bands.* |
| Top-right | `02_lichens_TPAMI/02_lichens_labels_and_ROI.png` | *Expert-annotated ROI ground truth (4 lichen species + substrate).* |
| Bottom-left | `02_lichens_TPAMI/08_classification_9bands_efficient.png` | *Pixel classification map using only 9 selected bands; substantially preserves species boundaries.* |
| Bottom-right | `02_lichens_TPAMI/09_wavelength_heatmap.png` | *Frequency with which each (excitation, emission) was selected across 50 runs — the autoencoder is consistent.* |

**Section blurb (~50 words):**
> Lichens Dataset 1 is a 4-species multi-excitation cube (7 excitations × ~28 emission bands). With ground-truth ROIs from expert annotation, we trained the CAE and ranked all (ex, em) pairs. K-NN on 9 selected bands reaches **0.93 ARI** vs 0.96 with the full 192 bands.

---

### 4E · DATASET 2 — Collagen Pepsin (IASIM 2026)
**Recommended layout:** 2 rows × 2 figs.

| Position | Asset | Caption |
|---|---|---|
| Top-left | `03_collagen_pepsin_IASIM/01_IASIM2026_headline_figure.png` | *Collagen pepsin sample — IASIM 2026 headline figure.* |
| Top-right | `03_collagen_pepsin_IASIM/02_accuracy_envelope.png` | *K-NN accuracy envelope — knee at 7-10 bands.* |
| Bottom-left | `03_collagen_pepsin_IASIM/07_efficiency.png` | *Bands-vs-accuracy frontier; selected configurations dominate variance/random baselines.* |
| Bottom-right | `03_collagen_pepsin_IASIM/09_wavelength_heatmap.png` | *Selected bands across runs, grouped by selection cutoff.* |

**Section blurb (~50 words):**
> The Pepsin / Acetic-Acid collagen series tests transferability: a chemically distinct sample with different fluorophore composition. Same pipeline, no retuning. The CAE again reaches the full-band accuracy with **<10 selected bands**, with the chosen wavelengths concentrated in expected collagen autofluorescence regions (~440-490 nm).

---

### 4F · ROBUSTNESS / EFFICIENCY (lower band)
| Position | Asset | Caption |
|---|---|---|
| Robustness | `02_lichens_TPAMI/10_robustness_histogram.png` | *50-run robustness: distribution of selected bands per run; concentration around the discriminative peaks confirms stability.* |
| Efficiency | `03_collagen_pepsin_IASIM/06_classifier_bars.png` | *Per-classifier accuracy across band budgets — the method is classifier-agnostic.* |

---

### 4G · WHAT'S NEXT (optional, if room)
> Currently extending to a **fully blind** drop-array dataset (no ground truth, 7 UV excitations, 16 droplets of unknown composition). Preliminary results show the same selection pattern emerging from latent perturbation alone — three different excitations × peak emission band — when paired with the corrected `max_per_excitation` normalization (a fix discovered while validating on this new dataset).

---

## 5 · Color & typography

- **Headline font:** sans-serif, weight 700+, dark slate (#1a2332) — keep titles short.
- **Body font:** the same family at weight 400, size ≥18 pt at A0.
- **Figure captions:** weight 500, italic, ≥16 pt.
- **Accent color** for "selected bands" callouts: a punchy magenta (#c5008c) or the magma viridis-yellow (#f9c74f) — both read well on UV/spectroscopy posters.
- **Section dividers:** thin 1px rule between sections, NOT boxes — avoids the "cluttered slide deck" look.

---

## 6 · Mandatory poster elements

- **Affiliations / contact** in title bar bottom-left.
- **Repository QR** linking to `https://github.com/narekmeloyan/spectral-select` in title bar bottom-right.
- **Citation block** in footer:
  ```
  Meloyan, N. (2025). Autoencoder-Based Wavelength Selection for 4D Hyperspectral Imaging.
  IEEE TPAMI (under review). Code: spectral-select [Zenodo: 10.5281/zenodo.18640119]
  ```
- **Funding / acknowledgments** if any, in footer right.

---

## 7 · Asset map (printable cheat-sheet)

```
Showcase_Poster/
├── POSTER_OUTLINE.md                       <- this file
├── 01_method/                              <- (empty; method block is text-only or sketched)
│
├── 02_lichens_TPAMI/                       <- canonical validation dataset
│   ├── 01_lichens_sample_RGB.png
│   ├── 02_lichens_labels_and_ROI.png
│   ├── 03_roi_overlay.png
│   ├── 04_accuracy_envelope.png            <- HEADLINE candidate
│   ├── 05_classification_192bands_baseline.png
│   ├── 06_classification_80bands.png
│   ├── 07_classification_13bands.png
│   ├── 08_classification_9bands_efficient.png   <- USE in DATASET 1 block
│   ├── 09_wavelength_heatmap.png            <- USE in DATASET 1 block
│   ├── 10_robustness_histogram.png          <- USE in ROBUSTNESS block
│   ├── 11_publication_figure.png            <- alt HEADLINE
│   ├── 12_executive_summary.png             <- backup
│   ├── 13_classification_comparison_3panel.png  <- alt HEADLINE
│   └── 14_roi_overlay_efficient_9.png
│
├── 03_collagen_pepsin_IASIM/               <- IASIM 2026 dataset
│   ├── 01_IASIM2026_headline_figure.png
│   ├── 02_accuracy_envelope.png
│   ├── 03_accuracy_envelope_expanded.png   <- bigger version
│   ├── 04_classifier_curves.png
│   ├── 05_classifier_curves_expanded.png   <- bigger version
│   ├── 06_classifier_bars.png              <- USE in EFFICIENCY block
│   ├── 07_efficiency.png                   <- USE in DATASET 2 block
│   ├── 08_gap_heatmap.png
│   ├── 09_wavelength_heatmap.png           <- USE in DATASET 2 block
│   ├── 10_roi_comparison_baselines.png
│   ├── _full_summary_report.pdf            <- reference
│   └── _IASIM2026_abstract.docx            <- reference
│
└── 04_supplementary/                       <- if you have extra wall space
    ├── lichens_accuracy_envelope_full.png
    ├── lichens_top10_configs.png
    ├── lichens_ex_em_distributions.png
    ├── lichens_wavelength_heatmap_extended.png
    ├── lichens_wavelength_scatter.png
    └── lichens_top_wavelengths_bar.png
```

---

## 8 · Final checklist before printing

- [ ] All figures exported at ≥300 dpi (the PNGs we copied are 140-300 dpi — re-export from PDFs in the source tree if you need higher resolution; PDF originals live in the same source folders we copied from).
- [ ] Captions stand alone (someone scanning only captions should still get the story).
- [ ] One sentence answers "what's new?" — put it directly under the title.
- [ ] Test the poster at A4 size — if you can't read it, the body font is too small.
- [ ] Repo QR works.

---

## 9 · A few drop-in caption alternatives

If you want shorter, more punchy captions for the headline figure:

> *9 bands. 97% of the accuracy. No labels needed.*

Or:

> *The autoencoder agrees with the spectroscopist: ~9 bands carry most of the discriminative signal in 192-band cubes.*

Or:

> *Selection knee: where adding more bands stops helping. Lichens reaches the knee at 9; collagen at 7-10.*
