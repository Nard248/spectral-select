# Master's Thesis — Narek Meloyan

**Title:** Deep Learning for Dimensionality Reduction in Multi-Excitation Hyperspectral Imaging
**Author:** Narek Meloyan
**Institution:** American University of Armenia
**Supervisor:** Prof. Narine Sarvazyan
**Year:** 2026

---

## What is this directory?

This directory contains the Master's Thesis manuscript and defense presentation, derived from the spectral-select journal paper.

**Scope decision (2026-05-14):** the thesis is restricted to the experiments and datasets included in the IASIM 2026 poster — namely the **Lichens** primary dataset and the **Collagen Sponges** secondary dataset. The Drop Data dataset and all blind-validation material (which were added during the TPAMI revision) are intentionally **out of scope** for the thesis to align with the poster's published scope.

**Scope refinement (2026-05-14, second pass):** the SOTA (eight-baseline head-to-head) comparison was also removed from the thesis. The poster does not include a SOTA shootout — it validates each dataset using two complementary, dataset-appropriate procedures: *robustness vs.\ random* on Lichens, and *classifier-family generalization* on Collagen Sponges. The thesis now mirrors this structure. The SOTA experiments remain available for the journal paper revision but are out of scope here.

---

## Files

| File | Description |
|---|---|
| `main.tex` | Thesis manuscript root (compile with `pdflatex`) |
| `sections/*.tex` | Section files: abstract, introduction, related_work, methodology, experimental_setup, results, discussion, conclusion |
| `references.bib` | Bibliography (kept full — all SOTA citations still needed) |
| `figures/` | All thesis figures (Lichens RGB, ROI overlay, classification maps, robustness histogram, wavelength heatmap, SOTA comparisons) |
| `Makefile` | Build helper |
| `build_defense_pptx.py` | Builder for the defense slide deck |
| `MasterThesis_Narek_Meloyan_Defense.pptx` | 31-slide PowerPoint deck for the thesis defense, with full speaker notes |
| `images_third_party/` | CC-licensed Wikimedia images used in the narrative introduction slides (with `CREDITS.md`) |

---

## Derivation from the journal paper

The thesis was derived by copying `paper/` → `MasterThesis_Narek_Meloyan/` and removing all content related to the Drop Data dataset.

### Sections that were edited

| File | Change |
|---|---|
| `sections/abstract.tex` | Three-dataset framing → two-dataset framing. Removed Drop Data sentence; added SOTA comparison sentence. |
| `sections/introduction.tex` | Contribution list reduced from 5 → 4 bullets. Removed the methodological-note bullet about ARI-gameability (only relevant to Drop Data). Three datasets → two datasets in the empirical contribution. |
| `sections/experimental_setup.tex` | Removed §IV.A.3 *Drop Data Dataset* subsubsection (≈40 lines). Edited preamble paragraph (3 datasets → 2). |
| `sections/results.tex` | Removed §V.F *Blind Validation on Drop Data* (≈60 lines). Removed Drop Data SOTA table (`tab:sota_drops`) and its discussion paragraph. Edited §V.G intro paragraph (3 datasets → 2). Final summary paragraph rewritten. |
| `sections/discussion.tex` | Removed §VI.D *Evaluation Metric Choice for Unsupervised Datasets* (≈12 lines — was Drop-Data-specific). Removed §VI.E *Influence Normalization as a Scope Axis* (≈14 lines — was Drop-Data-specific). Edited §VI.B Cross-Dataset Generalization (3 datasets → 2). Edited §VI.C Comparison-to-Existing-Methods to drop Drop-Data-specific reasoning. |
| `sections/conclusion.tex` | Replaced 3-dataset empirical-results paragraph with 2-dataset framing including SOTA comparison summary. |
| `main.tex` | Title block now says "Master's Thesis", supervisor noted explicitly, Acknowledgments rewritten in first person (single-author), supervisor biography removed (kept author biography only). PDF metadata updated. |

### Sections that were NOT changed (still match the journal paper)

- `sections/methodology.tex` — the framework is identical regardless of which datasets it's applied to
- `sections/related_work.tex` — covers the same literature

### Figures removed

- `figures/drop_data/` (Panels A, B, C from the Drop Data headline figure) was deleted from the thesis figures directory.
- `figures/pepsin/sota_comparison.png` and `figures/lichens/sota_comparison.png` were also removed in the second-pass scope refinement.

### Figures added (second pass)

Three poster-derived Collagen Sponges visuals were added under `figures/collagen_sponges/`:
- `CollagenLabels_and_ROI.png` — the 3×3 classification grid (matches poster)
- `accuracy_envelope.png` — KNN accuracy envelope across band counts (peak: 85.59% @ K=30)
- `wavelength_heatmap.png` — per-(λex, λem) mean importance map
- `classifier_curves.png` — ten-classifier validation curves

All four are direct copies from `Showcase_Poster/03_collagen_pepsin_IASIM/`, so the thesis matches the poster figure-for-figure.

---

## Building the manuscript

From the project root:

```bash
cd MasterThesis_Narek_Meloyan
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

(or use the `Makefile`)

## Building the defense slides

From the project root:

```bash
.venv/bin/python MasterThesis_Narek_Meloyan/build_defense_pptx.py
```

This regenerates `MasterThesis_Narek_Meloyan_Defense.pptx` from the figures currently on disk. Edit the slide content directly in `build_defense_pptx.py`.

## Slide structure (25 slides, ~25 min talk)

1. Title
2. Outline
3. Hyperspectral Imaging (motivation)
4. Multi-Excitation HSI (4D structure)
5. The Wavelength Selection Problem
6. Related Work (4 method families)
7. Contributions of this thesis
8. Method overview (3-stage pipeline)
9. Stage 1: 3D CAE
10. Stage 2: Latent-space perturbation
11. Stage 3: MMR selection
12. Dataset 1: Lichens (specs)
13. Lichens: ground truth and ROI training set
14. Dataset 2: Collagen Sponges (specs)
15. Lichens results: accuracy envelope
16. Lichens: spatial classification maps
17. Lichens: robustness vs 10,000 random
18. Lichens: wavelength importance heatmap
19. Collagen Sponges: dataset layout and ground truth (3×3 grid)
20. Collagen Sponges: accuracy envelope
21. Collagen Sponges: wavelength importance heatmap
22. Collagen Sponges: validation via classifier-family generalization
23. Discussion / Key findings
24. Limitations
25. Conclusion & Future directions

Every slide has speaker notes (visible in PowerPoint's notes view) with the exact talking points.

---

## Relationship to the journal paper

The journal paper (`paper/`) remains the full version with all three datasets including Drop Data, and incorporates the TPAMI reviewer-feedback revisions. After thesis defense, journal submissions should target that version, not this one.
