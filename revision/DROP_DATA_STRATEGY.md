# Drop Data — Deep-Dive Strategy

**Role in the revised paper:** Third dataset, fully blind, headline answer to R1.6 (unsupervised claim weakened). Provides the strongest possible evidence that the framework selects meaningful bands without labels.

This document expands the agreed three-panel headline figure (EEM heatmaps + per-excitation slices + ARI-vs-K) into the full research tree. Each branch is rated by priority (★) and feasibility on local Mac compute.

---

## A. Foundation: the headline three-panel figure

Already designed in conversation 2026-05-11. Restated here so this file is self-contained.

### Panel A — EEM context (the 2-D truth)
- Three small `7 × 31` heatmaps, one per Ward(full-217) drop type.
- Each cell colored by mean intensity over drops of that type, with the canonical Rayleigh mask grayed (em < ex+40 OR |em − 2·ex| < 40).
- Overlay the 5 selected `(λex, λem)` markers (circles) — same on all three panels.
- Reader takeaway: "Types have distinct EEM landscapes. Markers land where the landscapes differ most."

### Panel B — Per-excitation emission slices (the familiar payoff)
- 7 subplots (one per λex), x = λem, y = ROI-mean intensity.
- One curve per drop (16 curves), colored by Ward(full) type.
- At excitations where the AE selected a band, mark the chosen λem with a filled marker on each curve.
- Reader takeaway: "The curves separate at exactly the bands we picked, and the AE was silent at excitations where curves overlap (λex = 340)."

### Panel C — ARI vs K (quantitative ceiling check)
- Line plot, x = K (3..10), y = ARI(K-band → Ward-full).
- One line per method: ours (`ae_perturb` + max_per_excitation), variance, PCA-loading, SAM-greedy, SPA, MCUVE, BS-Net-FC, random (mean ± std over 5 seeds).
- Dashed horizontal at ARI = 1.0 (full-cube ceiling, by construction).
- Reader takeaway: "Ours saturates near 1 at low K; baselines lag."

**Status:** Panel A & B require new code (Phase 4 deliverable). Panel C requires running the SOTA baselines (Phase 2 deliverable).

**ROI definition:** Hand-placed circular ROIs from the broadband max-projection (per user's preference for the load-bearing figure). Watershed-mask version included as supplement-only sensitivity check.

---

## B. Expansion dimensions (recursive iteration tree)

### B.1 ★★★ Normalization ablation — "the bug becomes a contribution"
**Idea:** Variance-normalization (TPAMI default) inverts ranking on Drop Data because discriminative bands are also high-variance. Document this as a method-scope finding.

**Method:**
- Run AE-perturbation with `normalization_method ∈ {variance, max_per_excitation, none}` on all three datasets.
- Report Spearman ρ between AE-influence and the discriminative ground truth (F-ratio for Lichens/Collagen Sponges labeled-class separability; per-band F-ratio for Drops Ward-types).
- Report ARI / accuracy of K-band selections under each normalization.

**Why it matters:**
- Falsifies "the method just picks high-variance bands."
- Demonstrates a real scope-of-applicability axis: *mixed-media* (lichens, collagen sponges) tolerates variance normalization; *spatially-segregated* (drops) requires max_per_excitation.
- Addresses R1.2 (ablation) AND R1.6 (unsupervised), AND becomes a paper contribution in its own right.

**Output:** Paper §V.D.2 *Normalization Method as a Scope Axis*, plus a figure showing Spearman ρ flipping sign as we change normalization on Drops but stays positive on Lichens.

---

### B.2 ★★★ Preprocessing-variant ablation — which variant is the "honest" one?
**Idea:** Drop Data has 5 cumulative preprocessing variants: raw, dark, dark_norm, dark_norm_mask, full. After the ruler crop (row 175): raw_cr, dark_cr, dark_norm_cr, dark_norm_mask_cr, full_cr.

Memory says `full_cr` is the champion (mean F = 78.74). But: are we cherry-picking the variant?

**Method:**
- Report all 5 variants × all methods × K = 3..10.
- Make explicit which preprocessing step contributes how much. Background subtraction (dark) does X; per-pixel normalization (dark_norm) does Y; drop-mask (dark_norm_mask) does Z; full pipeline does W.
- Identify the minimum preprocessing needed for the AE to recover the 3-type structure.

**Why it matters:** Pre-empts the reviewer who asks "you cherry-picked the variant where your method works." Shows that the gain comes from selection, not preprocessing.

**Output:** Supplement table; one sentence in main paper.

---

### B.3 ★★★ Multi-seed stability — does the AE pick the same bands every time?
**Idea:** Train AE with 5–10 different random seeds. Compute Jaccard overlap of top-K selections across seeds.

**Why it matters:** R1.8 directly. Also addresses the implicit "is this just lucky initialization?" concern.

**Method:**
- 5 seeds per (dataset × K).
- For each pair of seeds, compute |S_i ∩ S_j| / |S_i ∪ S_j|. Average across pairs.
- Compare to multi-seed Jaccard of variance/PCA/etc (which should be near 1 since they're deterministic).
- Report a table: method × dataset × mean-Jaccard × std.

**Output:** Paper §V.B sub-table + RESEARCH_LOG entry.

---

### B.4 ★★ Within-type variability — is the figure honest?
**Idea:** Panel B (per-excitation slices) currently shows one curve per drop, colored by type. But there are only 2–3 drops per type — is that enough to call "the curves separate"?

**Method:**
- Compute per-type mean ± SD bands at each excitation.
- Overlay individual drops as fine lines and type-mean as bold lines with shaded SD.
- Test: at what λex/λem does mean(Type 1) ± SD overlap mean(Type 2) ± SD? Those are the *non-discriminative* cells, and the AE should not have selected them.

**Why it matters:** Defensiveness against "your 2 Type-2 drops are not statistically separable from your 13 Type-0 drops" objection. Also makes the figure more honest.

**Output:** Refined Panel B with shaded SD bands.

---

### B.5 ★★★ Selection-transfer test — do Drop-selected bands work on Collagen Sponges/Lichens?
**Idea:** Take the 5 bands picked on Drops. Apply them as the band set for Collagen Sponges classification (without re-selecting on Collagen Sponges). See if accuracy is comparable to Collagen Sponges-selected bands. Likewise reverse.

**Why it matters:**
- Addresses generalizability across datasets — a deeper claim than within-dataset robustness.
- If transfer is poor: the method is dataset-specific (which is *fine* — bands carry chemistry, and chemistry varies).
- If transfer is good: surprising and a major contribution.
- Either way, an interesting story for the Discussion.

**Method:**
- 3×3 transfer matrix: rows = "selection dataset," columns = "evaluation dataset." Diagonal = within-dataset. Off-diagonal = cross-dataset.
- Metric: accuracy (Lichens, Collagen Sponges) or ARI (Drops) of the transferred selection.

**Output:** New paragraph in Discussion §VI; transfer table in supplement.

---

### B.6 ★★ ε-magnitude ablation
**Idea:** Paper claims ε ∈ {15, 30, 45} aggregated. Reviewer R1.10 questions consistency. Re-test: ε ∈ {0.5σ, 1σ, 1.5σ, 2σ, 3σ, 5σ} and see whether the selection changes.

**Why it matters:** Robustness of the perturbation step. Direct R1.2 and R1.10.

**Method:**
- For each ε individually, compute the influence matrix and the K=5 selection.
- Report Jaccard overlap of selections across ε values.
- If overlap is high, ε is not load-bearing — the method is robust. If overlap is low, we need to recalibrate or aggregate.

**Output:** Supplement table; one paragraph in §V.D Ablation Study.

---

### B.7 ★★ Per-pixel vs ROI-mean spectra
**Idea:** ROI-mean is the load-bearing choice for the headline figure. But individual pixels are noisier. Test: does the selection still discriminate when computed on individual pixels rather than ROI means?

**Method:**
- For each drop ROI, pick the brightest single pixel (interior of the well, not edge).
- Compute its 217-D spectrum. Apply our 5 selected bands.
- Plot Panel B's emission slices as individual-pixel curves instead of ROI means.
- Compare discriminability (Mahalanobis distance between type-centroids in 5-D).

**Why it matters:** Pre-empts "your means are smoothed; real per-pixel data is too noisy."

**Output:** Supplement panel; one paragraph.

---

### B.8 ★★ "Where the AE was silent" — interpret unselected excitations
**Idea:** The AE picked 0 bands at λex = 340 in the K=5 selection. That's interesting — it means the AE thinks λex=340 carries no marginal information beyond what's at other excitations.

**Method:**
- Quantify: at λex=340, what is the average between-type variance? Should be low.
- Compare to λex=325 (selected) — should be high.
- Plot a per-excitation "discriminative budget" bar chart.

**Why it matters:** Demonstrates the AE's selections are *consistent with the physics*, not just statistical artifacts. Strong storytelling for the Discussion.

**Output:** Discussion paragraph; small bar chart figure.

---

### B.9 ★★ Hierarchical re-clustering from K bands
**Idea:** The 3-type ground truth was Ward-clustered on full 217-D drop-means. Can Ward find the same 3 types from only the 5 selected bands?

**Method:**
- Re-run Ward on 5-D drop-mean vectors (using our 5 picks).
- Plot the resulting dendrogram side-by-side with the full-217 dendrogram.
- Compute ARI between the partitions at k=3.

**Why it matters:** This is the most visceral "the bands preserve the structure" demonstration — a dendrogram-equivalence picture.

**Output:** Side-by-side dendrogram figure; ARI score caption.

---

### B.10 ★ Failure-mode characterization — when does the method break?
**Idea:** Reviewer-proof the method by showing what we *can't* do.

**Method:**
- Synthetic: generate a Drop-Data-like cube where types differ only at one specific (λex, λem) cell. Does the AE find it?
- Adversarial: add Gaussian noise at increasing scales to the cube. At what SNR does ARI drop below 0.9?
- Sample size: subsample drops (use 10/16, 8/16, 6/16, 4/16). At what point does the AE fail to recover the 3 types?

**Why it matters:** Honest limits make the positive claims more credible.

**Output:** §VI Limitations subsection expanded.

---

### B.11 ★ Cross-validation of "ground truth" — is the 3-type structure real?
**Idea:** Ward clustering at k=3 was chosen. What if the true structure is k=2 (just type 0 vs not-type-0) or k=4?

**Method:**
- Run Ward at k=2, 3, 4, 5 on full-217 drop spectra.
- Compute silhouette, Calinski-Harabasz, gap statistic for each k.
- Report which k is most defensible.
- Re-run ARI evaluation at the most-defensible k.

**Why it matters:** Strengthens the "ground truth" foundation that ARI is computed against.

**Output:** Methodology footnote; supplement table.

---

### B.12 ★ Physical interpretation — match selected bands to known fluorophores
**Idea:** Our picks for K=5: `325/530, 365/490, 400/490, 415/490, 385/470`. Look up which fluorophores are known to emit in those (λex, λem) cells.

**Method:**
- Consult Lakowicz fluorophore tables.
- 470–490 nm emission from UV excitation is classic NADH / pyridoxine / collagen autofluorescence.
- 530 nm with 325 nm excitation is unusual — flag for follow-up.

**Why it matters:** Shows the method is doing chemistry, not statistics. Strong Discussion paragraph.

**Output:** New Discussion §VI.E *Physical interpretation of selected bands*.

---

## C. Priority order (Phase 4 execution sequence)

1. **B.1** (normalization ablation) — must precede everything else, because Drop Data results depend on the right normalization being used.
2. Headline three-panel figure (Panels A, B, C).
3. **B.3** (multi-seed stability).
4. **B.5** (selection transfer).
5. **B.9** (Ward dendrogram equivalence).
6. **B.2** (preprocessing variants), **B.4** (within-type SD bands), **B.6** (ε ablation) — bundle as one ablation table.
7. **B.8** (where the AE was silent), **B.12** (physical interpretation) — Discussion enrichments.
8. **B.7** (per-pixel vs ROI), **B.10** (failure modes), **B.11** (ground-truth k validation) — supplement.

## D. Resource estimate (local Mac compute)

| Item | Per-run cost | Runs needed | Wall-clock |
|---|---|---|---|
| AE training (Drop) | 7 min | 5 variants × 5 seeds = 25 | ~3 hr |
| AE training (Collagen Sponges) | ~15 min | 5 seeds | ~1.5 hr |
| AE training (Lichens) | ~1 hr | 5 seeds × 2 normalizations = 10 | ~10 hr |
| Perturbation analysis | ~5 min | per AE checkpoint | trivial relative |
| SOTA baselines on Drops | varies | 6 methods × 8 K × 5 seeds | ~2 hr |
| SOTA baselines on Lichens/Collagen Sponges | varies | similar | ~6 hr |
| **Total compute** | | | **~22 hr wall-clock** |

Acceptable as overnight + a day on local Mac, *if* batched into discrete jobs.
