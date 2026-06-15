# Revision Changelog

Every action that touches the paper, the code, or the experimental record is logged here in chronological order. Each entry has: timestamp, phase, reviewer-point reference (if any), short description, files touched, and a one-line "why."

Format:
```
## YYYY-MM-DD [Phase X · Rn.m / Standalone]
**Action:** What I did.
**Files:** path/to/file
**Why:** One-line justification.
**Result:** What changed; pointer to RESEARCH_LOG.md entry if relevant.
```

---

## 2026-05-12 [Phase 0 · Initial setup]
**Action:** Created revision workspace and planning documents.
**Files:**
- `revision/MASTER_PLAN.md` — seven-phase plan, dataset roster, compute constraints
- `revision/MITIGATION_TABLE.md` — point-by-point response strategy (all 15 reviewer items)
- `revision/CHANGELOG.md` — this file
- `revision/RESEARCH_LOG.md` — iterative experiment log scaffolding
- `revision/DROP_DATA_STRATEGY.md` — Drop Data deep-dive expansion plan
**Why:** Establish a stable structure before touching the manuscript or code.
**Result:** Five planning documents ready. Phase 0 (Audit) begins next.

## 2026-05-12 [Phase 0 · Audit · R1.9 / R1.10]
**Action:** Cross-checked paper's numerical and configuration claims against `results/Lichens_Dataset_1_MasterRun/results.csv` (3072 configs) and `paper_figures/paper_metrics.json`.
**Files:**
- `revision/PHASE0_AUDIT_FINDINGS.md` — new audit document
- `revision/RESEARCH_LOG.md` — appended audit findings entry
**Why:** R1.9 (k=1 optimality) and R1.10 (perturbation magnitudes) both required verification before any paper text edits. Numbers must be true before they can be re-cited.
**Result:** Three discrepancies found. (1) perturbation magnitudes `{15,30,45}` in paper appear nowhere in code; actual experiments used `{30,40,50}` or `{50,60,70}`. (2) Baseline accuracy off by 2.7pp (paper 85.5% vs reality 88.2%). (3) Paper's claimed "optimal" config (variance + k=1) scores 69.4%, not 86.1%; actual optimum is PCA + k=3 + max_per_excitation at 90.2% (K=13), 95.2% (K=80). **Phase 1 paper-text edits paused pending user decision** on how to handle the stale numbers.

## 2026-05-12 [User decision · Anchor + autonomy bound]
**Action:** Confirmed (a) rebuild paper from MasterRun data, (b) pause for paper-text changes / push for code+figures+experiments.
**Why:** Set the autonomy contract for the remainder of the revision.
**Result:** Continuing in autonomous mode on code/figure work.

## 2026-05-12 [Phase 4 · Drop Data · Panels A + B]
**Action:** Built `revision/figures/drop_data/build_headline_figure.py`. Cached the full_cr cube as `full_cr_cube.npz` (one-time `.pkl` -> `.npz` conversion).
**Files:**
- `revision/figures/drop_data/full_cr_cube.npz` - 20 MB, safe-format cache
- `revision/figures/drop_data/build_headline_figure.py` - figure script
- `revision/figures/drop_data/panel_A_eem_per_type.{png,pdf}` - 3 EEM heatmaps + selected-band markers
- `revision/figures/drop_data/panel_B_emission_slices.{png,pdf}` - 7 per-excitation slice subplots
**Why:** R1.6 (unsupervised claim) - visceral evidence that selection on blind data discriminates drop types.
**Result:** Both panels tell the story cleanly: AE picks 5 bands at 470-530 nm across 5 excitations; silent at lambda_ex 310 and 340 (where drop curves overlap). Panel C (ARI vs K) requires SOTA baselines and will be built after Phase 2.

## 2026-05-12 [Phase 2 · SOTA baselines suite]
**Action:** Built `revision/baselines/` package with 9 methods + dataset adapter + runner.
**Files:**
- `revision/baselines/__init__.py` - method registry + types
- `revision/baselines/classical.py` - 6 ported methods (variance, pca_loading, sam_greedy, spa, mcuve, random)
- `revision/baselines/issc.py` - ISSC (sparse subspace clustering)
- `revision/baselines/bsnet.py` - BS-Net-FC (Cai et al. 2020)
- `revision/baselines/sparse_lasso.py` - L1-penalized supervised
- `revision/baselines/ae_perturb_cached.py` - loads our method's existing selections
- `revision/baselines/data_adapter.py` - Drop Data full_cr loader (Lichens/Collagen Sponges stubs)
- `revision/baselines/run_comparison.py` - runner with KNN + ARI + F-ratio metrics
- `revision/baselines/results_drop/drop_data_full_cr/{comparison_results,method_summary,selections}` - K=3-10 x 5 seeds results
**Why:** R1.3 / R1.7 / R2.3 demand SOTA HSI band-selection comparison.
**Result:** AE-perturb in top tier across all K; wins at K=10 (0.964 vs 0.958 next-best). Switched primary metric from ARI to per-pixel KNN-5 accuracy after discovering ARI rewards selections that include noise bands.

## 2026-05-12 [Phase 4 · Drop Data · Panel C]
**Action:** Built `revision/figures/drop_data/build_panel_C.py`. Plotted KNN accuracy vs K for 9 methods.
**Files:**
- `revision/figures/drop_data/build_panel_C.py`
- `revision/figures/drop_data/panel_C_knn_vs_K.{png,pdf}`
**Why:** Completes the three-panel Drop Data headline figure for R1.6 + R1.3.
**Result:** Red AE-perturb line on top across K=3..10, annotated callout at K=10 (0.964 best). SAM-greedy visibly degrades with K — a story for the discussion about why one-strong-band-plus-noise tricks fail KNN evaluation.

## 2026-05-12 [Phase 2 · Lichens adapter built; first run interrupted]
**Action:** Built Lichens data adapter + cube cache + label image. Generalized `ae_perturb_cached.py` to dispatch by dataset (Drops via CSV, Lichens via per-config wavelengths.json).
**Files:**
- `revision/baselines/lichens_cube.npz` (62 MB cube cache)
- `revision/baselines/lichens_labels.npy` (per-pixel class labels from class_mask.png)
- `revision/baselines/data_adapter.py` - load_lichens() with optional per-class subsample
- `revision/baselines/ae_perturb_cached.py` - extended dispatch
- `revision/figures/lichens/build_panel_C.py` - prepared, ready to read output of full Lichens run
**Why:** Phase 2 SOTA-baselines comparison must include Lichens (R1.5/R1.7/R2.3).
**Result:** Initial Lichens K=13 smoke test (5000 pixels/class x 4 classes = 20000 pixels x 192 bands) showed AE-perturb=0.977, SPA=0.981, random=0.974, variance/PCA=0.613, SAM=0.643. Full sweep (K=3..80, all methods, 3 seeds) was launched in background; took longer than expected on ISSC + BS-Net + Sparse-LASSO; the run was killed before writing final results. **TODO next session:** re-run with reduced subsample (1000 px/class) and/or split into fast-methods-first batch.

## 2026-05-12 [Phase 0 / 4 · Lichens label semantics finding]
**Action:** Mapped Lichens class_mask.png unique RGB triples to class IDs.
**Result:** Paper claims "4 lichen morphological types" labeled as Type 0/1/2/5 totaling 223,597 pixels. Reality (class_mask.png + analysis_mask.npy): classes are labeled 1/3/6/7 totaling 191,046 pixels. This is the **fourth** paper-vs-data discrepancy alongside the three from Phase 0 audit. Documented in PHASE0_AUDIT_FINDINGS.md.

## 2026-05-12 [Phase 0 · Correction: archive/paper/ != paper/]
**Action:** Identified that `archive/paper/` is a STALE version, not the actual TPAMI submission. The real reviewer-feedback paper lives at `paper/` (top-level).
**Files:**
- `paper/main.tex` - "Deep Learning for Dimensionality Reduction in Multi-Excitation Hyperspectral Imaging" (Meloyan + Sarvazyan)
- `paper/sections/*.tex` - 8 section files
**Why:** Phase 0 audit was performed against the wrong paper, overstating discrepancies.
**Result:** Re-audit findings:
- Headline numbers in `paper/sections/results.tex` ARE correct (88.2% baseline, 95.2% at K=80, 89.4% at K=9, PCA over variance).
- Excitation grid in `paper/sections/experimental_setup.tex` IS correct (310,325,340,365,385,400,415,430 nm).
- Configuration sweep is 3,072 (matches code).
- 4 ground-truth classes (correct count, though IDs 1/3/6/7 per data while paper just says "4 distinct classes" - acceptable abstraction).

**Remaining real issues** (need fixing in revision):
- R1.10: methodology.tex line 313 says `epsilon \in {15,30,45}` but experimental_setup.tex line 103 says medium `{30,40,50}` / high `{50,60,70}` - one of these is wrong.
- R1.9: methodology.tex line 301 mentions `k=1` as optimal, but results.tex Table III shows the optimal config is `PCA, 3-dim, 80 bands` (k=3 PCA, not k=1 variance). Prose is inconsistent with table.
- R1.5/R2.2 + R1.3/R1.7/R2.3 + R1.6: paper has ONLY Lichens. Collagen Sponges and Drop Data need to be added.

## 2026-05-12 [Phase 0 · Collagen Sponges data confirmed available]
**Action:** Inventoried Pepsin/Collagen results across `results/Pepsin_*/` and `results/Collagen_*/`.
**Result:** Have complete sweep (432 configs) at `results/Collagen_Pepsin_Normalized/results.csv` + multi-classifier baseline study + figures in `results/Pepsin_Paper_Figures/`. Headline: 6 excitations, 158 bands, 3 classes, 39,970 pixels. Baseline (158 bands KNN-5): 79.78%. Peak: 85.59% at K=30. LDA outperforms KNN: 92.5% at K=50. **Ready to integrate as second dataset.**
