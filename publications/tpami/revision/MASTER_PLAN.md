# Master Revision Plan — spectral-select

**Started:** 2026-05-12
**Author:** Narek Meloyan (with Claude as research collaborator)
**Status:** Active revision after TPAMI desk-review-rejection-with-feedback

---

## Goal

Address reviewer feedback with substantive evidence, not rebuttal-only rhetoric, so the next submission to any venue does not invite the same critique. Venue is secondary; defensibility of every claim is primary.

## Scope of the final revised paper

Three labeled-or-unlabeled datasets, each playing a distinct role in the rebuttal:

| Dataset | Status | Role in revised paper |
|---|---|---|
| **Lichens Dataset 1** | TPAMI-submitted, frozen pending re-run | Primary supervised validation. Re-run with both `variance` and `max_per_excitation` normalizations to verify robustness (R1.2, R1.8). |
| **Collagen Collagen Sponges** | IASIM 2026 submitted | Second supervised dataset. Demonstrates cross-domain generalization. Includes classifier-comparison and hyperparameter-tuning artifacts (R1.5, R2.2, R2.4). |
| **Drop Data** | Post-TPAMI, blind | Third dataset, fully unsupervised. Headline "blind validation" case study. Directly answers R1.6 (unsupervised claim). |

## Computational constraints

Local M-series Mac with MPS (and `PYTORCH_ENABLE_MPS_FALLBACK=1` because `aten::_adaptive_avg_pool3d` is not MPS-native).

- Drop Data: ~17 s/epoch × 25 epochs ≈ 7 min per training run.
- Lichens patches (64×64 with overlap) are heavier; expect ~hour-scale per full run.
- Multi-seed × multi-method × multi-dataset must be batched carefully. Single autonomous overnight sessions, not parallel fleets.

Disk + memory: cache cubes as `.npy`. Never re-load `.im3` (PyImageJ init ~30s).

## Phase plan

The revision proceeds in seven phases. Each phase produces deliverables logged in `CHANGELOG.md`; every experimental decision is recorded in `RESEARCH_LOG.md`.

### Phase 0 — Audit & verify (consistency between paper and code)

**Goal:** Eliminate inconsistencies the reviewers can pin us on (R1.9, R1.10).

- [ ] Cross-check perturbation magnitudes: paper says ε ∈ {15, 30, 45}; verify code uses identical values; resolve discrepancy.
- [ ] Verify k=1 optimality claim on Lichens against full sweep data (and check whether it holds under multi-seed).
- [ ] Audit all numerical claims in the abstract / introduction against `results/Lichens_Dataset_1_MasterRun/`.
- [ ] Confirm 192-band baseline number (Reviewer 2 wonders if 192 vs 217 inconsistency matters — Collagen Sponges uses 6×31 = 186 and the abstract/methodology vary). Document precisely.

### Phase 1 — Method strengthening (text + theory)

**Goal:** Address R1.1 (heuristic), R2.1 (not ME-HSI-specific), R1.2 (no ablation).

- [ ] Add theoretical justification subsection. Frame the three-stage pipeline via the **information-bottleneck** view: AE compression maximizes I(z; X), perturbation attribution recovers gradients of I(z; X) with respect to (λex, λem) cells, MMR enforces submodular diversity. Pipeline becomes principled, not heuristic.
- [ ] Emphasize ME-HSI-specific architectural commitments: parallel encoder branches that handle variable emission bands per excitation, Rayleigh masking as a physical (not data-driven) preprocessing step, feature-merging as cross-excitation pooling.
- [ ] Run ablation suite (Phase 2) and integrate into a new Section V.D *Ablation Study*.

### Phase 2 — Ablations, baselines, classifiers, seeds

**Goal:** Address R1.2 (ablations), R1.3/R1.7/R2.3 (SOTA baselines), R1.8 (multi-seed), R2.4 (multi-classifier).

Ablations to run (each on Lichens, Collagen Sponges, Drops where applicable):

1. **Feature-merge strategy** (R1.2): average vs concat vs sum vs max.
2. **Perturbation direction**: ± symmetric vs +only vs −only.
3. **Perturbation magnitude ε**: {15, 30, 45} individually + the aggregated default.
4. **Dimension-selection method**: variance vs activation vs reconstruction vs PCA.
5. **Number of dimensions k**: 1, 3, 5, 7 — including multi-seed for the "k=1 optimal" claim.
6. **MMR λ**: 0.3, 0.45, 0.5, 0.7.
7. **Normalization method** (NEW, from Drop Data finding): variance vs max_per_excitation vs none — run on **all three** datasets.

SOTA baselines to add (sized for local compute):

| Method | Type | Implementation |
|---|---|---|
| **MCUVE** | Classical | Already in `drop_data_selection_sweep.py` — port to Lichens/Collagen Sponges |
| **SPA (Successive Projections Algorithm)** | Classical | Already in Drops sweep — port |
| **ISSC** (Improved Sparse Subspace Clustering) | Classical | Implement |
| **BS-Net-FC** (Cai et al. 2020) | Deep | Implement using their published architecture |
| **DARecNet-BS** | Deep | Implement if time allows |
| **Sparse-LASSO band selection** | Classical | Easy baseline; uses labels for Lichens/Collagen Sponges |

Multi-seed: 5 seeds per (method × dataset × K). Report mean ± std.

Multi-classifier: KNN (existing), SVM (RBF + linear), Random Forest, MLP, 1D-CNN. Demonstrate selection holds across classifier capacity (R2.4).

### Phase 3 — Collagen Sponges & Drop dataset integration into the paper

**Goal:** Address R1.5/R2.2 (single dataset, single classifier) and R1.6 (unsupervised claim).

- [ ] Add a new Section IV.A.3 *Collagen Collagen Sponges Dataset* (specifications, classes, ground truth).
- [ ] Add a new Section IV.A.4 *Drop Data (Blind Validation)* (specifications, **no labels**, 3-type Ward-clustering ground truth derived from full-spectrum unsupervised analysis).
- [ ] Add a new Section V.C *Results: Collagen Sponges* (envelope, heatmap, classifier comparison).
- [ ] Add a new Section V.D *Results: Blind Validation on Drop Data* (the three-panel argument).
- [ ] Update abstract + introduction + conclusion accordingly.

### Phase 4 — Drop Data deep dive

**Goal:** Build the agreed three-panel evidence (EEM, slices, ARI), then expand recursively. See `DROP_DATA_STRATEGY.md` for the full expansion tree.

Headline figure (per prior conversation):
- Panel A: 3 per-type EEM heatmaps with 5 selected `(λex, λem)` markers overlaid.
- Panel B: 7 per-excitation emission-slice subplots, drop curves colored by type, selected emission bands marked.
- Panel C: ARI(K-band → Ward-full) vs K for our method vs SOTA baselines, dashed line at ARI=1.

Secondary supplement:
- Mahalanobis-pairwise-distance heatmap on selected K-D space.
- Parallel-coordinates view of the 5-D representation.

### Phase 5 — Computational profiling

**Goal:** Address R1.4/R2.5 (cost).

- [ ] Profile component-by-component (AE training vs perturbation vs MMR).
- [ ] Identify why perturbation is 75% of total cost (R2.5 claim).
- [ ] Propose `fast` variant: e.g., fewer ε values, batch-vectorized perturbation, top-k pruning before perturbation.
- [ ] Benchmark `fast` vs `full` — accuracy delta and speedup.
- [ ] Honest discussion: the cost is one-time per dataset, downstream inference is `K×` faster than full spectrum.

### Phase 6 — Manuscript revision

**Goal:** Pull everything together.

- [ ] Rewrite Abstract to lead with three datasets + unsupervised emphasis.
- [ ] Expand Introduction to motivate from physics (ME-HSI specificity) and information theory (perturbation = attribution).
- [ ] Replace Methodology subsections per Phase 1 plan; add theoretical bridge.
- [ ] New Results section per Phase 3.
- [ ] New Ablation Study section per Phase 2.
- [ ] Expanded Related Work — add SOTA HSI BS references the reviewers feel are missing.
- [ ] Update Discussion: address each reviewer concern in passing within the text (not just rebuttal letter).
- [ ] Conclusion: stronger generalizability claim from three datasets.

### Phase 7 — Recursive iteration

**Goal:** After each completed phase or experiment, **ask "what else?"** and pursue interesting branches. Record every branch in `RESEARCH_LOG.md` with motivation and result, including dead ends.

Examples of questions that may emerge:
- Does selection from Lichens transfer to Collagen Sponges? (Cross-dataset transfer)
- What if we use a **labeled** loss alongside reconstruction? (Semi-supervised variant)
- Is the AE actually picking physically meaningful fluorophore peaks? (Match to Lakowicz spectra tables)
- Does the K=1 optimality break down on larger class vocabularies? (Synthetic test)

---

## File layout

```
revision/
├── MASTER_PLAN.md           ← this document
├── MITIGATION_TABLE.md      ← point-by-point reviewer response
├── CHANGELOG.md             ← chronological log of every change
├── RESEARCH_LOG.md          ← iterative experiment log (idea → result → next)
├── DROP_DATA_STRATEGY.md    ← Drop Data deep-dive expansion
├── REBUTTAL_LETTER.md       ← (to be written near end) formal cover letter
└── figures/                 ← any revision-specific figures
```

Manuscript edits land in `archive/paper/sections/*.tex` (existing location). Original `.tex` files are preserved via git; pre-edit snapshots saved as `*.tex.bak.YYYYMMDD` only when an edit is large enough to make the diff hard to follow.

---

## Definition of done

The revision is "ready to send" when:

1. Every reviewer point in `MITIGATION_TABLE.md` has status either **Resolved** (with evidence pointer) or **Rebutted** (with paper-text reference to the rebuttal in Discussion).
2. The paper builds with three datasets, an ablation table, and SOTA baselines.
3. Drop Data section has the three-panel figure and a discussion of the normalization finding.
4. `RESEARCH_LOG.md` has a coherent narrative — not just a chronological dump.
5. The user has reviewed and approved the rebuttal-letter draft.
