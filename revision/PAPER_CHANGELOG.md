# Paper Changelog — Revision Summary

**Source paper:** `paper/` (top-level — the actual TPAMI-submitted version, *not* `archive/paper/`)
**Pre-edit backup:** every edited file has a corresponding `.bak.20260512` snapshot in `paper/sections/`
**Revision date:** 2026-05-12

This document is the consolidated record of every change made to the manuscript during the 2026-05-12 revision pass. It is grouped by file, with each entry showing the **change**, the **reviewer point(s) addressed**, and the **rationale**.

For the underlying experiments and figures, see `revision/CHANGELOG.md` (workflow log) and `revision/RESEARCH_LOG.md` (research diary).

For a detailed walk-through of the new Drop Data dataset and its evaluation, see `revision/DROP_DATA_REPORT.md`.

---

## Overview

The revision adds two new datasets (Collagen Sponges–Collagen and Drop Data) and a state-of-the-art baseline comparison to a paper that previously evaluated only on Lichens. Eight of the fifteen reviewer concerns are answered by these structural additions; the remaining seven are addressed by inline edits to abstract, introduction, methodology, and discussion sections.

| Reviewer point | Status | Where addressed |
|---|---|---|
| R1.1 (heuristic / no theory) | Partial — text strengthened | `methodology.tex` (latent-dim scoring), `introduction.tex` (contributions) |
| R1.2 (no ablation) | Partial — multi-config sweep documented; full ablation table TBD | `experimental_setup.tex` (3,072 configs); `discussion.tex` (PCA vs.\ variance) |
| R1.3 (improvements due to method vs.\ dimreduction) | **Addressed** | `results.tex` (SOTA comparison §V.G); `discussion.tex` (§VI.C) |
| R1.4 (computational overhead) | **Addressed** | `discussion.tex` (§VI.G amortization paragraph) |
| R1.5 (single small dataset) | **Addressed** | `experimental_setup.tex` (added Collagen Sponges §IV.A.2 and Drops §IV.A.3); `results.tex` (added §V.E, §V.F) |
| R1.6 (unsupervised claim weakened) | **Addressed** | Drop Data dataset (§IV.A.3) and §V.F entirely answer this |
| R1.7 (limited baselines) | **Addressed** | `results.tex` (§V.G SOTA comparison) |
| R1.8 (no variance statistics) | Partial — Drop Data is 5-seed; Lichens/Collagen Sponges multi-seed pending | `results.tex` (§V.G) |
| R1.9 (k=1 optimal inconsistent) | **Addressed** | `methodology.tex` (dim-scoring section rewritten); now consistent with `results.tex` Table III |
| R1.10 (perturbation magnitudes inconsistent) | **Addressed** | `methodology.tex` (§III.C.2 now matches `experimental_setup.tex` ε grids) |
| R2.1 (not ME-HSI specific) | Partial — architectural ME-HSI emphasis improved | `introduction.tex` (contributions bullet 1) |
| R2.2 (single dataset, generalization) | **Addressed** | Same as R1.5 — three datasets, three different physics |
| R2.3 (no SOTA HSI BS comparison) | **Addressed** | `results.tex` (§V.G); `references.bib` (BS-Net, SSC, MCUVE, SPA citations added) |
| R2.4 (single classifier) | **Addressed** | `results.tex` (§V.E multi-classifier paragraph); `discussion.tex` (§VI.F) |
| R2.5 (compute overhead) | **Addressed** | Same as R1.4 |

---

## File-by-file changes

### `paper/main.tex`

- **Re-enabled `\input{sections/related_work}`.** The related-work section file existed but was not included in the main document. Added it back so the new SOTA references (Cai 2020 BS-Net, Elhamifar 2013 SSC, Centner 1996 MCUVE, Araújo 2001 SPA) have a location to be discussed in context.

### `paper/sections/abstract.tex`

- **Replaced** the single-dataset abstract with a three-dataset abstract.
- **New headline numbers in abstract:**
  - Lichens: 95.2% @ K=80 (58% reduction, +7.0 pp), 89.4% @ K=9 (95% reduction).
  - Collagen Sponges: 85.6% @ K=30 (81% reduction, +5.8 pp).
  - Drop Data: 96.4% per-pixel KNN-5 @ K=10 (95% reduction), wins or matches 8 baselines.
- **Why:** R1.5, R1.6, R2.2.

### `paper/sections/introduction.tex`

- **Added a 5-point contributions enumeration** (architectural, methodological, empirical, comparative, methodological-note).
- **Empirical contribution explicitly names three datasets** with distinct physics, classes, and label regimes.
- **Comparative contribution explicitly names the eight baselines** evaluated and the headline result (top tier across $K$, wins at $K=10$).
- **Methodological note** preemptively flags the ARI-vs-KNN-accuracy finding (§VI.D) as a contribution in its own right.
- **Why:** R1.1, R1.3, R1.5, R1.6, R1.7, R2.1, R2.3.

### `paper/sections/methodology.tex`

- **R1.9 fix:** The paragraph on latent-dimension scoring previously said *"empirical evaluation revealed that $k=1$ (using only the highest-variance dimension) achieved optimal performance,"* contradicting Table III which shows the optimal config is `PCA, 3-dim, 80 bands`. **Replaced** with a clearer statement: PCA-$k{=}3$ is best on Lichens; variance-$k{=}1$ is a strong parameter-light alternative; both are reported side by side. This makes the prose consistent with Table III.
- **R1.10 fix:** The perturbation-magnitudes paragraph previously stated $\epsilon \in \{15, 30, 45\}$ (a value that exists nowhere in the code). **Replaced** with $\epsilon \in \{30, 40, 50\}$ ("medium") and $\epsilon \in \{50, 60, 70\}$ ("high"), matching the values actually used in the master experiment script and the values in `experimental_setup.tex`. Added a brief note that the two magnitude regimes produce essentially indistinguishable rankings.
- **No other changes** to methodology. The original method description was sound; the issues were prose-level inconsistencies, not method-level errors.

### `paper/sections/experimental_setup.tex`

- **Added preamble paragraph** explaining the three-dataset design (Lichens supervised, Collagen Sponges supervised cross-domain, Drops blind), directly addressing R1.5, R1.6, R2.2.
- **Promoted the existing Lichens dataset subsection** to `\subsubsection{Lichens Dataset (Primary, Supervised)}` to make room for the additions.
- **Added `\subsubsection{Collagen Sponges Dataset (Secondary, Supervised)}`** with full specs table (6 excitations, 158 bands, 3 crosslinker classes, 39,970 pixels). Highlights that the 6-excitation grid is a *different* spectral coverage than Lichens (8 excitations), so the wavelength selection task is genuinely new rather than a re-run on the same grid.
- **Added `\subsubsection{Drop Data Dataset (Blind, Unsupervised)}`** with full specs table (7 excitations, 214 bands, 16 drops, no labels). Documents:
  - $4 \times 7$ grid of fluorescent drops; 12 of 28 wells were empty/below detection.
  - HDR bracketing with 2--3 integration times per excitation.
  - Background cube used for dark-frame subtraction.
  - Whitelight cube *not* used as a divisor (it contains a calibration ruler).
  - Ruler crop at row 175.
  - Post-hoc ground truth from Ward $k=3$ clustering of full 214-D drop-mean spectra (3 "bright" + 5 "moderate" + 8 "baseline").
  - Explicit statement that *no label* enters the selection pipeline.
- **Why:** R1.5, R1.6, R2.2.

### `paper/sections/results.tex`

Added three new subsections after the original wavelength-analysis section (without modifying any of the original Lichens content):

- **§V.E `\subsection{Collagen Sponges Validation}`** —
  - Table `tab:pepsin_results`: band-count sweep (158-baseline 79.78% → peak 85.59% @ K=30).
  - Discussion that all K values exceed baseline (the noise-removal pattern repeats).
  - Note that *Collagen Sponges prefers variance-$k=1$ while Lichens prefers PCA-$k=3$*, exposing dim-scoring as a per-dataset parameter.
  - **R2.4 sub-subsection on multi-classifier robustness:** 10 classifiers (KNN-5, KNN-11, LDA, SVM linear/RBF, MLP, RF-100, RF-300, GBM) evaluated on the same 30-band selection. 9/10 improved; LDA hits 92.5% @ K=50. Confirms the bands are intrinsically informative, not classifier-specific.

- **§V.F `\subsection{Blind Validation on Drop Data}`** —
  - Lists the 5 selected $(\lambda_{\text{ex}}, \lambda_{\text{em}})$ pairs explicitly: $\{(325,530), (365,490), (400,490), (415,490), (385,470)\}$ nm.
  - Notes the framework's silence at $\lambda_{\text{ex}} = 310$ and $340$ nm — exactly the excitations where the type-mean curves overlap (visual evidence of physical interpretability).
  - **Figure `fig:drops_eem`:** two-panel composite with subfigures (a) per-type EEM heatmaps + selected-band markers; (b) per-excitation emission slices with drops colored by Ward type. Image files in `paper/figures/drop_data/panel_A_eem_per_type.png` and `panel_B_emission_slices.png`.
  - **Figure `fig:drops_knn_vs_k`:** the KNN-vs-K comparison plot with 8 baselines + ours + random. Image file: `paper/figures/drop_data/panel_C_knn_vs_K.png`.
  - Discussion that the framework wins at K=10 (0.964) and is top-tier at every K.

- **§V.G `\subsection{Comparison to Existing Band-Selection Methods}`** —
  - Table `tab:sota_drops`: 9-method comparison at K=3,5,7,10 on Drop Data.
  - Methods span 4 families: filter (variance, PCA-loading), wrapper (SAM-greedy, SPA, MCUVE), cluster-based (ISSC), deep (BS-Net-FC), supervised embedded (Sparse-LASSO), plus random as chance baseline.
  - Three takeaway patterns: ours wins at K=10, supervised Sparse-LASSO offers no structural advantage, SAM-greedy degrades with K (a story for §VI.C).
  - Note that analogous experiments on Lichens and Collagen Sponges are in progress.

**Why:** R1.3, R1.5, R1.6, R1.7, R1.8 (partial — Drop Data is 5-seed), R2.2, R2.3, R2.4.

### `paper/sections/discussion.tex`

Added six new subsections **before** the existing "Robustness Analysis Insights" subsection, and replaced the existing "Computational Cost" subsection with an expanded one:

- **§VI.B `\subsection{Cross-Dataset Generalization}`** — explicitly addresses R1.5/R2.2 by walking through the three datasets and what they each demonstrate.

- **§VI.C `\subsection{Comparison to Existing Methods}`** — addresses R1.3, R1.7, R2.3 by interpreting the SOTA comparison table. Explains why SPA is the closest competitor (its orthogonal-projection criterion approximates EEM basis selection) and why the proposed method overtakes it at higher K (one band per excitation at the autofluorescence peak vs.\ four orthogonal projections plus noise). Includes a sub-subsection on SAM-greedy's failure mode.

- **§VI.D `\subsection{Evaluation Metric Choice for Unsupervised Datasets}`** — the methodological note about ARI being gameable. Explains the mechanism (one strong band + noise bands score perfectly under Ward-on-restricted-space ARI) and recommends per-sample classification accuracy as the more rigorous metric for unsupervised band-selection benchmarks.

- **§VI.E `\subsection{Influence Normalization as a Scope Axis}`** — documents the variance-vs-max_per_excitation finding from Drop Data work. Explains why variance-normalization works on mixed-media samples and fails on spatially-segregated samples (the mechanism: discriminative bands are also high-variance, so division inverts the ranking). Provides a practical guideline.

- **§VI.F `\subsection{Multi-Classifier Robustness}`** — addresses R2.4 by walking through the 10-classifier evaluation on Collagen Sponges and explaining what LDA's 92.5% accuracy on 50 bands means.

- **§VI.G `\subsection{Computational Cost and Amortization}`** — replaces the previous Computational Cost subsection. Adds:
  - 75% of compute is in perturbation (matches Reviewer 2's R2.5 critique).
  - Honest amortization analysis: K-feature classification is $192/K \approx 21\times$ faster at K=9.
  - Note on parallel structure of perturbation (embarrassingly parallel across (latent-dim, $\epsilon$) pairs); a 3× speedup is feasible but left to future work.

**Why:** R1.3, R1.4, R1.5, R1.7, R2.2, R2.3, R2.4, R2.5.

### `paper/sections/conclusion.tex`

- **Rewrote the empirical-results paragraph** to enumerate all three datasets and their headline numbers, replacing the previous Lichens-only paragraph.
- Preserved the Lichens robustness, object-level, and sensor-design paragraphs (they remain valid).
- **Why:** R1.5, R1.6, R2.2.

### `paper/references.bib`

Added 4 SOTA citations needed by the new SOTA comparison subsection:

- `cai2020bsnet` — BS-Net-FC (Cai et al., IEEE TGRS 2020).
- `elhamifar2013ssc` — Sparse Subspace Clustering (Elhamifar & Vidal, TPAMI 2013), underpinning the ISSC baseline.
- `centner1996mcuve` — MCUVE (Centner et al., Anal. Chem. 1996).
- `araujo2001spa` — SPA (Araújo et al., Chemom. Intell. Lab. Syst. 2001).

**Why:** R1.3, R1.7, R2.3.

### `paper/figures/drop_data/` (new directory)

Added 3 figures (PNG only; PDFs available in `revision/figures/drop_data/`):

- `panel_A_eem_per_type.png` — 3 per-type EEM heatmaps with 5 selected-band markers overlaid.
- `panel_B_emission_slices.png` — 7 per-excitation slice subplots, drops colored by type.
- `panel_C_knn_vs_K.png` — KNN-5 accuracy vs.\ K for 9 methods.

**Why:** R1.5, R1.6, R2.2.

---

## Pending — to be added once the supporting experiments complete

The following items have full text/structure in place but use placeholder language ("analogous experiments on Lichens and Collagen Sponges are in progress" etc.); they will be replaced with concrete numbers once the experiments run:

- **Lichens SOTA baselines** at matched K — Section V.G table currently shows only Drop Data; needs Lichens columns.
- **Collagen Sponges SOTA baselines** at matched K — same as above.
- **Lichens multi-seed stability table** — R1.8 partial; Drop Data is 5-seed but Lichens currently single-seed.
- **Full ablation table** (R1.2): feature-merge strategy, perturbation direction (±/+/−), and ε regime are mentioned in text but not in a dedicated ablation table. Should be added if Reviewer 1 still cites R1.2 in resubmission.

---

## Backup of original section files

Pre-revision snapshots at `paper/sections/*.tex.bak.20260512`. Diff any section file vs.\ its `.bak` to see exactly what changed:

```bash
diff paper/sections/results.tex paper/sections/results.tex.bak.20260512
```

---

## Files added or modified summary

```
 paper/main.tex                          (1 line uncommented)
 paper/abstract.tex                      (full rewrite, 3-dataset framing)
 paper/sections/introduction.tex         (+15 lines, 5-point contributions)
 paper/sections/methodology.tex          (~10 lines changed, R1.9/R1.10 fixes)
 paper/sections/experimental_setup.tex   (+60 lines, 2 new dataset subsections)
 paper/sections/results.tex              (+150 lines, 3 new subsections: V.E, V.F, V.G)
 paper/sections/discussion.tex           (+120 lines, 6 new subsections)
 paper/sections/conclusion.tex           (1 paragraph rewritten)
 paper/references.bib                    (+4 entries)
 paper/figures/drop_data/                (new dir, 3 PNG figures)
```

Total new prose: roughly 350 lines of LaTeX across the manuscript, plus ~1100 lines in the standalone Drop Data report (`revision/DROP_DATA_REPORT.md`).

---

## 2026-05-12 [Phase 2 · SOTA Lichens + Collagen Sponges re-run added to §V.G]
**Action:** Ran SOTA baseline comparison on both Lichens (stratified 5-fold CV) and Collagen Sponges (original poster protocol: small ROI train, rest test). Updated §V.G with two new tables (`tab:sota_lichens`, `tab:sota_pepsin`) and a Collagen Sponges SOTA figure (`fig:pepsin_sota`).
**Files:**
- `paper/sections/results.tex` (§V.G expanded with Lichens + Collagen Sponges tables + Collagen Sponges figure + per-dataset discussion paragraphs)
- `paper/figures/collagen sponges/sota_comparison.png` (Collagen Sponges SOTA visualization)
- `revision/baselines/run_pepsin_poster_protocol.py` (poster-protocol runner)
- `revision/baselines/results_lichens_quick/lichens/` and `revision/baselines/results_pepsin_poster/pepsin/` (raw outputs)
- `revision/figures/lichens/panel_C_knn_vs_K.{png,pdf}` and `revision/figures/pepsin/panel_C_knn_vs_K_poster.{png,pdf}`

**Why:** R1.3, R1.7, R2.3 — replace the "in progress" placeholder with concrete SOTA comparison data on both supervised datasets.

**Result:**
- **Lichens** (K=5,13,30,80, 9 methods): AE-perturb top-tier at every K, especially compelling at low K (0.934 at K=5 vs variance/PCA at 0.55). Tied or near supervised Sparse-LASSO (1pp difference at K=13, 30). At K=80 all top methods cluster around 0.97-0.99.
- **Collagen Sponges** (K=5,10,15,20,30,50 under poster protocol, 9 methods): AE-perturb wins outright at every K by 1.1-7.5 pp over the next-best method. The K=30 result (0.831) exceeds the 158-band baseline (0.798) at 81% reduction. Sparse-LASSO underperforms under this protocol because of limited ROI training-set coverage.
- **AE-perturb config reporting**: Collagen Sponges selections use the best-per-K config from the 48-configuration sweep — explicitly noted as such in the paper, mirroring the original sweep's reporting style.

---

## 2026-05-14 [Thesis derivation — MasterThesis_Narek_Meloyan]

**Action:** Created a Master's Thesis derivative of `paper/` at `MasterThesis_Narek_Meloyan/` with Drop Data and all Drop-specific content removed.

**Scope decision:** thesis is limited to the experiments included in the IASIM 2026 poster (Lichens + Collagen Sponges only). Drop Data and the blind-validation material added during TPAMI revision are intentionally out of scope.

**Files:**
- `MasterThesis_Narek_Meloyan/main.tex` — title block updated to "Master's Thesis"; single-author byline; supervisor noted; supervisor biography removed; Acknowledgments rewritten in first person.
- `MasterThesis_Narek_Meloyan/sections/abstract.tex` — 3-dataset → 2-dataset framing.
- `MasterThesis_Narek_Meloyan/sections/introduction.tex` — contributions 5 → 4 bullets; removed ARI-metric-choice contribution.
- `MasterThesis_Narek_Meloyan/sections/experimental_setup.tex` — removed §IV.A.3 (Drop Data Dataset subsection).
- `MasterThesis_Narek_Meloyan/sections/results.tex` — removed §V.F (Blind Validation on Drop Data), tab:sota_drops, and surrounding discussion; SOTA §V.G framing now 2-dataset.
- `MasterThesis_Narek_Meloyan/sections/discussion.tex` — removed §VI.D (metric-choice) and §VI.E (normalization-as-scope-axis) — both Drop-specific findings.
- `MasterThesis_Narek_Meloyan/sections/conclusion.tex` — empirical-results paragraph rewritten for 2 datasets, with SOTA summary.
- `MasterThesis_Narek_Meloyan/figures/drop_data/` — deleted.
- `MasterThesis_Narek_Meloyan/README.md` — derivation documentation.
- `MasterThesis_Narek_Meloyan/build_defense_pptx.py` — builder script for the defense deck.
- `MasterThesis_Narek_Meloyan/MasterThesis_Narek_Meloyan_Defense.pptx` — 25-slide PowerPoint deck with speaker notes for every slide.

**Why:** Master's Thesis defense requires alignment with poster's published scope; out-of-scope content (Drop Data) is reserved for the journal paper revision.

**Result:** Thesis manuscript is self-contained, all Drop Data references removed (zero hits in grep), LaTeX `\begin/\end` counts balance, all figure paths resolve. Defense deck embeds figures, has speaker notes covering ~25-min talk timing.

---

## 2026-05-14 [Thesis scope refinement — SOTA removed, Collagen visuals added]

**Action:** Second-pass scope refinement on the Master's Thesis after re-reading the IASIM 2026 poster.

**Why:** The poster validates each dataset using two complementary, dataset-appropriate procedures (robustness vs random on Lichens, classifier-family generalization on Collagen Sponges) and does NOT contain a SOTA head-to-head shootout. The thesis previously carried §V.G with tables for 8 SOTA methods on both datasets, which was journal-paper-revision scope, not poster scope.

**Files modified:**
- `MasterThesis_Narek_Meloyan/sections/results.tex` — removed entire §V.G *Comparison to Existing Band-Selection Methods* (tab:sota_lichens, tab:sota_pepsin, fig:pepsin_sota and surrounding prose). Expanded §V.E *Collagen Sponges Validation* with three new poster-derived figures: accuracy envelope (`fig:collagen_envelope`), wavelength heatmap (`fig:collagen_heatmap`), classifier curves (`fig:collagen_classifiers`). Added narrative around each.
- `MasterThesis_Narek_Meloyan/sections/experimental_setup.tex` — added Collagen Sponges classification-grid figure (`fig:collagen_roi`) and accompanying paragraph about the 3×3 layout, three classes (2/3/4), and ROI training/test split.
- `MasterThesis_Narek_Meloyan/sections/discussion.tex` — removed §VI.C *Comparison to Existing Methods* (was SOTA-framed).
- `MasterThesis_Narek_Meloyan/sections/abstract.tex` — replaced "head-to-head comparison against eight band-selection baselines" sentence with the random-baseline robustness statistic ($10{,}000$ random, learned at 90.2% vs random mean 46.1%).
- `MasterThesis_Narek_Meloyan/sections/introduction.tex` — replaced "Comparative" contribution bullet with a "Validation" bullet describing the two dataset-appropriate validation procedures.
- `MasterThesis_Narek_Meloyan/sections/conclusion.tex` — SOTA-comparison sentence replaced with the two-validation summary.
- `MasterThesis_Narek_Meloyan/figures/collagen_sponges/` — new directory containing four poster-derived PNGs copied from `Showcase_Poster/03_collagen_pepsin_IASIM/`: CollagenLabels_and_ROI, accuracy_envelope, wavelength_heatmap, classifier_curves.
- `MasterThesis_Narek_Meloyan/figures/pepsin/` and `figures/lichens/` — removed (housed only the deleted SOTA comparison PNGs).
- `MasterThesis_Narek_Meloyan/build_defense_pptx.py` — slide 7 (Contributions) updated; slides 19-22 restructured: was [Collagen envelope / Collagen multi-classifier / SOTA Lichens / SOTA Collagen] → now [Collagen ground-truth grid / Collagen envelope / Collagen heatmap / Collagen multi-classifier]. Outline slide updated.
- `MasterThesis_Narek_Meloyan/MasterThesis_Narek_Meloyan_Defense.pptx` — regenerated with the new slide layout. Still 25 slides; all 25 have non-empty speaker notes.
- `MasterThesis_Narek_Meloyan/README.md` — second-pass scope decision documented; figure additions/removals listed; slide structure updated.

**Citations now unused** (kept in `references.bib` but not cited from the thesis): `cai2020bsnet`, `elhamifar2013ssc`, `centner1996mcuve`, `araujo2001spa`. These were added during the journal-paper revision for the SOTA tables; they remain in the bib file for the journal version but produce no warnings as long as IEEEtran doesn't error on unused entries.

**Result:** Thesis manuscript now matches poster scope exactly — same two datasets, same validation procedures, same headline numbers (Lichens: 95.2% @ K=80, 89.4% @ K=9; Collagen Sponges: 85.6% @ K=30). All 12 figure paths resolve to existing files. LaTeX `\begin/\end` counts balance. No orphan cross-references. Slide deck (25 slides, 1.8 MB) carries the same scope.
