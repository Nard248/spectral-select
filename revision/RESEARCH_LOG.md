# Research Log — Recursive Iteration Diary

Unlike `CHANGELOG.md` (which records *what* was done), this log records *what was thought, tried, learned, and what came next*. Every meaningful experimental branch — including dead ends — gets an entry. The goal is that the user can read this end-to-end and understand the *reasoning trace* of the revision, not just the outcomes.

## Entry template

```
## YYYY-MM-DD · [Topic]

### Idea
What we were curious about. One paragraph.

### Why it matters
Why this could improve the paper or the method. What reviewer concern it speaks to.

### Methodology
What we ran. Datasets, K values, seeds, parameters. Enough to reproduce.

### Result
Numbers, plots (referenced), observations. Plain language.

### Why this result
Mechanistic explanation if possible. If unclear, say "unclear" and what would clarify it.

### What this opens up
Next questions, hypotheses to test, branches to pursue or kill.
```

---

## 2026-05-12 · Initialization

### Idea
TPAMI returned the paper with two negative reviews focused on (i) experimental scope, (ii) baseline strength, (iii) ablation depth, and (iv) the question of whether the framework genuinely uses ME-HSI structure or is just a generic dimensionality-reduction trick. The user has scoped this session as: address the feedback substantively so any future submission survives the same critique, not just this one.

### Why it matters
A paper-level pivot. Three datasets instead of one; SOTA baselines instead of variance/random; theoretical scaffolding instead of pipeline-as-recipe; ablations that justify each design choice individually.

### Methodology
Set up `revision/` with five planning documents (MASTER_PLAN, MITIGATION_TABLE, CHANGELOG, RESEARCH_LOG, DROP_DATA_STRATEGY). No experiments yet — Phase 0 begins after.

### Result
Scaffold ready. Mitigation table maps every reviewer point to a phase + paper section + status (Accept/Partial/Rebut). All 15 points addressed; only 4 partial rebuttals (R1.4, R1.6, R2.1, R2.4) — the rest are full accepts.

### Why this result
The reviewer concerns collapse into three orthogonal themes (datasets, baselines, ablations) plus one rebuttable framing point (R2.1 — physical mechanism). Each can be addressed with concrete additions to the paper rather than rhetorical defense.

### What this opens up
Immediate next branch: **Phase 0 audit**.
1. Cross-check perturbation magnitudes between methodology.tex and the codebase (R1.10).
2. Verify the k=1 optimality claim multi-seed (R1.9).
3. Confirm numerical claims in Abstract vs `results/Lichens_Dataset_1_MasterRun/`.

These three should take less than an hour and surface any embarrassments before we build on top of frozen-but-possibly-wrong numbers.

---

## 2026-05-12 · Phase 0 Audit — three discrepancies found

### Idea
Cross-check every numerical claim in the paper against the data on disk in `results/Lichens_Dataset_1_MasterRun/`. Use `results.csv` (3072 configurations × accuracy/F1/kappa/ARI) and `paper_figures/paper_metrics.json` as ground truth.

### Why it matters
Phase 0 must be done before any other phase. If the paper's frozen numbers don't match reality, every downstream claim (relative performance, baseline comparison, the "13 bands exceeds baseline" headline) is built on sand.

### Methodology
1. `grep` perturbation magnitudes in `spectral_select/` and `experiments/`.
2. Load `results/Lichens_Dataset_1_MasterRun/results.csv` (3073 rows).
3. Compare to paper §V tables row-by-row.
4. Verify k=1 optimality by computing `groupby(['dimension_selection_method','n_important_dimensions','normalization_method'])['accuracy'].agg(['mean','max'])` at K=13.

### Result
Three major discrepancies, all detailed in `PHASE0_AUDIT_FINDINGS.md`:

1. **Perturbation magnitudes** (R1.10 confirmed): Paper says `{15,30,45}`. No file in the codebase uses these values. Actual experiments use `{30,40,50}` or `{50,60,70}`.

2. **Baseline accuracy off by 2.7pp**: Paper Table III says baseline = 85.5%. `results.csv` and `paper_metrics.json` both say baseline = 88.2%. The "Rel. Perf." column in Table IV is computed against the wrong baseline.

3. **k=1 optimality wrong** (R1.9 confirmed at paper level): Paper Table V claims variance + k=1 + 13 bands = 86.1%. Actual: that exact config scores **69.4%**. The 86.1% (and better) only appears for **PCA + k=3 + max_per_excitation** which scores **90.2%** at K=13.

4. **Headline understated**: Paper says optimal is 13 bands at 86.1% (+0.6pp above 85.5% baseline). Reality is K=80 at 95.2% (+7.0pp above 88.2% baseline), with a strong sweet spot at K=13 at 90.2% (+2.1pp).

### Why this result
The paper appears to have been written from an earlier, partial experimental run (Jan 26-27, 2026 per the pipeline directories) that was superseded by the MasterRun (Jan 29) but never re-anchored. The MasterRun has BETTER results across the board; the paper's numbers and configuration claims are stale. This is consistent with the user-side note that TPAMI numbers were "frozen" — but frozen against an earlier, suboptimal snapshot.

Reviewer R1.9's hunch ("k=1 is not fully consistent with experimental results") was actually correct, but for a different reason than they suggested: the paper's claim is internally inconsistent with the codebase's own data, not just with multi-seed variation.

### What this opens up
**Paused for user input** — three options laid out in `PHASE0_AUDIT_FINDINGS.md`:
1. Rebuild paper from MasterRun data (recommended).
2. Investigate which older run produced the paper's stated numbers.
3. Re-run Lichens with exact paper protocol to check reproducibility.

Cannot proceed to Phase 1 (paper edits) until the user decides which anchor to use. All other phases (Collagen Sponges/Drop integration, SOTA baselines, ablations) are unaffected and can proceed in parallel if desired.

**User decision (2026-05-12):** (1) Rebuild paper from MasterRun data as canonical. (2) Pause for paper-text changes; push forward on code/figures/experiments.

---

## 2026-05-12 - Drop Data Panels A + B built

### Idea
Render the agreed-upon EEM heatmap + per-excitation slice panels from existing `full_cr` data. Validate that the figure actually tells the story we want before investing in Panel C (ARI vs K, which requires SOTA baselines).

### Why it matters
This is the visceral payoff for R1.6 (the unsupervised claim). If these panels don't visually separate the drop types at the selected bands, the whole Drop Data integration thesis weakens.

### Methodology
1. One-time conversion of `Data/processed/Drop Data Cropped/full_cr/spectra_data.pkl` -> `revision/figures/drop_data/full_cr_cube.npz` (safe `.npz` format, allows downstream scripts to use only numpy native I/O).
2. `revision/figures/drop_data/build_headline_figure.py`:
   - Loads cubes from npz, drop_labels (cropped), drop_types (Ward k=3 on full-217 means).
   - Computes per-drop EEM means (16, 7, 31).
   - Panel A: aggregates to per-type EEMs, plots 3 heatmaps with Rayleigh mask grayed and 5 selected (lambda_ex, lambda_em) markers overlaid as white circles+x.
   - Panel B: 7 per-excitation subplots, 16 individual drop curves + 3 type-mean bold lines, dashed vertical at selected lambda_em, in-plot label.

### Result
Both panels generated as PNG + PDF in `revision/figures/drop_data/`.

**Panel A** confirms: the Bright drop type (n=3) has a strong peak at lambda_em 470-530 spanning lambda_ex 325-415; Moderate (n=5) shows the same structure attenuated; Baseline (n=8) is dim throughout. The 5 selected markers cluster exactly in the Bright type's peak region.

**Panel B** confirms: red (Bright) curves stand out from orange (Moderate) and blue (Baseline) precisely at the bands the AE picked. At lambda_ex = 310 nm and 340 nm — the two excitations where the AE selected NO band — the type-mean curves are visually closer, validating the AE's silence.

### Why this result
The AE has learned that bands at lambda_em near 470-530 carry the most discriminative information about drop chemistry, and that lambda_ex 310 and 340 add no marginal information given the others. This is consistent with biological autofluorescence physics (NADH/collagen-like peaks in that emission range, broad excitation continuum requiring only sparse sampling).

The drop type label semantics in this run differ from the conversation-history memory: Type 0 = Bright (3 drops, most intense) rather than baseline. The 3-type structure is preserved; the integer labels just got reassigned by Ward in a different ordering. The figure labels this clearly.

### What this opens up
1. **Panel C (ARI vs K)** requires SOTA baselines - that becomes the next deliverable.
2. **B.1 (normalization ablation)** can run now because the data and figure scaffolding are in place.
3. **B.8 (silent excitations)** has its quantitative evidence already visible in Panel B; could be promoted to a numerical claim.
4. **B.9 (Ward dendrogram from K bands)** could be the next supplement panel.

Priority next: SOTA baseline scaffolding (unlocks Panel C and answers R1.3/R1.7/R2.3 - the most-cited critique).

---

## 2026-05-12 - SOTA baselines suite + Drop Panel C + a metric-choice finding

### Idea
Build a unified baselines package (`revision/baselines/`) implementing 9 methods (variance, pca_loading, sam_greedy, spa, mcuve, issc, bsnet_fc, sparse_lasso, random) with one signature, a data adapter that loads each dataset into common (features, labels, band_catalog) form, and a runner that sweeps method x K x seed x metric. Use it to render Panel C of the Drop Data headline figure.

### Why it matters
R1.3 / R1.7 / R2.3 (cited by both reviewers) demand SOTA HSI band-selection comparison. The codebase already had variance/PCA/SAM/SPA/MCUVE/random in `experiments/drop_data_selection_sweep.py` but no ISSC, BS-Net, or Sparse-LASSO. Without this scaffolding, the revision cannot run any comparison-against-SOTA experiment.

### Methodology
1. Refactored the 6 existing rankers (variance, pca_loading, sam_greedy, spa, mcuve, random) into `revision/baselines/classical.py` with the unified signature `select(features, K, *, labels=None, seed=0, **kw) -> ndarray`.
2. Implemented three new methods:
   - **ISSC** (`issc.py`): L1-penalized self-expression per band, spectral clustering on |C|+|C.T|, one representative per cluster (highest variance).
   - **BS-Net-FC** (`bsnet.py`): Cai et al. 2020 attention-based reconstruction net, runs on MPS, top-K by average attention weight.
   - **Sparse-LASSO** (`sparse_lasso.py`): L1-penalized multinomial logistic regression, top-K by max-over-classes |coef|. Supervised; falls back to variance on unlabeled data.
3. Wrote `data_adapter.py` with `load_drop_data_full_cr()` returning (features=(6580,214), labels=per-pixel-type, band_catalog=list of (ex,em), drop_id, drop_types, drop_mean_spectra_full).
4. Added `ae_perturb_cached.py` adapter that loads the existing AE-perturb selections from `results/Drop_Data_Cropped_Sweep/full_cr/ae_perturb_results.csv` and remaps them to the unified band catalog.
5. Wrote `run_comparison.py` runner that sweeps method x K x seed and saves long-form CSV + selections.json + summary.csv.

### Result

**First run with ARI(Ward-k=3) as the metric surfaced a methodological problem.** AE-perturb scored only 0.36; SAM-greedy hit 0.78. Investigation showed why:
- SAM-greedy's K=5 selection: `[(385, 490), (310, 420), (415, 720), (385, 420), (415, 710)]` — *one* discriminative band (385,490) plus four near-zero-signal bands.
- Ward at k=3 in this 5-D space is dominated by the one informative dimension, and the noise dimensions don't hurt.
- AE-perturb's K=5: `[(325, 530), (365, 490), (400, 490), (415, 490), (385, 470)]` — five correlated bands all sampling the autofluorescence peak from different excitations. Information-rich, but Ward in this redundant subspace can't separate Type 1 from Type 2 cleanly.

**This means ARI(Ward on K-band drop-means) is a bad evaluation metric** — it rewards adding *noise* bands alongside one informative band. Replaced with **per-pixel KNN-5 5-fold CV accuracy** as the primary metric: this requires each band to actually contribute discriminative info per pixel; noise bands hurt rather than help.

**Re-evaluation with KNN accuracy (R1.6 / R2.4-aligned):**

| Method | K=3 | K=5 | K=7 | K=10 |
|---|---|---|---|---|
| **AE-perturb (ours)** | **0.938** | 0.947 | 0.957 | **0.964** |
| Variance | 0.793 | 0.956 | 0.956 | 0.956 |
| PCA-loading | 0.793 | 0.863 | 0.954 | 0.958 |
| SPA | **0.952** | **0.950** | 0.955 | 0.957 |
| SAM-greedy | 0.818 | 0.804 | 0.774 | 0.740 (degrades) |
| MCUVE | 0.815 | 0.856 | 0.905 | 0.917 |
| ISSC | 0.845 | 0.931 | 0.928 | 0.946 |
| BS-Net-FC | 0.705 | 0.819 | 0.903 | 0.940 |
| Random (mean) | 0.745 | 0.804 | 0.847 | 0.886 |

**AE-perturb is in the top tier across all K and wins outright at K=10.** SPA is strongest at K=3. Variance is competitive from K=5 onward — a story worth telling honestly.

### Why this result

Three observations of interest, each potentially paper-relevant:

1. **The metric question is the methodology question.** ARI on Ward(K-band) measures cluster-recovery; KNN accuracy measures pixel-level discriminability. They reward different selection behaviors. On Drop Data — a small-sample, internally-redundant dataset — they can disagree by 0.4. This is a method note worth adding to the experimental-setup section: *for unsupervised band selection evaluation, use multiple metrics; per-pixel classification accuracy is the most demanding.*

2. **Variance is hard to beat on segregated-sample datasets.** With only 3 spectral types and an obvious bright peak, the high-variance bands are exactly the discriminative ones. AE-perturb's edge here is small (~0.01-0.02). The paper should *show* this rather than hide it: argue that the AE-perturb advantage grows on harder mixed-media datasets (Lichens, Collagen Sponges), and on Drop Data it ties with the strongest classical baselines while picking a more *interpretable* set (5 different excitations sampling the peak, vs. variance's tight cluster around λem=470-490).

3. **SAM-greedy degrades with K.** At K=10 it's at 0.74, *below random*. This is because its "maximize spectral angle" criterion ends up picking maximally-different (i.e., maximally-noisy) bands once the informative subspace is saturated.

### What this opens up
1. **Run the same comparison on Lichens** — needs a Lichens data adapter + new run. AE-perturb's advantage should be larger there (mixed-media dataset, 4 classes).
2. **Run on Collagen Sponges** — same.
3. **Multi-classifier evaluation on Drops** — currently only KNN. R2.4 asks for SVM/RF/MLP/CNN. Switch should be cheap given the runner architecture.
4. **Compute resource:** the Drop Data run took ~3 minutes total; Lichens will be slower (much more pixels per fold) but still feasible on local Mac.
5. **Rebuild AE-perturb for K=13, 18, 25** to extend Panel C if a longer-K argument is wanted.
6. **Combine Panels A+B+C** into a single tile for the paper.

The "metric-choice finding" itself is a separate sub-paper worthy note — *most published HSI band-selection comparisons use cluster-recovery metrics (silhouette, ARI). On small-sample datasets these can systematically reward selections with noise bands. We recommend per-sample classification accuracy as the more rigorous benchmark.*
