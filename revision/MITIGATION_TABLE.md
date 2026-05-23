# Mitigation Table — Reviewer Feedback → Response

**Started:** 2026-05-12
**Format:** Each row covers one reviewer point. Status is one of:
- **Accept** — concrete fix in this revision.
- **Partial** — partially accept (do most of what they ask), partially rebut (one part is wrong/misframed).
- **Rebut** — respectfully decline with reasoning; the criticism mis-reads the contribution.

The "Resolved by" column points to a phase in `MASTER_PLAN.md` and/or a section in the revised paper.

---

## Editor's overall framing

> *"Both reviewers show negative opinion on this paper. They all concern on the experimental verification for the proposed method. For example, the adopted datasets and baselines are not ideal, and the contribution of each of the proposed key parts is not clearly justified."*

**Triangulated meta-critique:** dataset diversity, baseline strength, ablation depth. Every other point is a sub-instance of one of these three. Our revision must clearly answer all three at once.

---

## Reviewer 1

### R1.1 — Heuristic design, no unified objective, no theoretical justification

> *"The proposed method relies on a sequence of heuristic components ... without a unified objective or clear theoretical justification. ... the connection between reconstruction-based representation learning and the final goal of discriminative wavelength selection is not formally established."*

| Field | Detail |
|---|---|
| **Status** | **Partial — accept (most), rebut (a little)** |
| **What's right** | The original methodology section does not explicitly justify *why* reconstruction-based representation learning yields discriminative bands. That is a real gap. |
| **What's overstated** | "Loosely coupled pipeline" is unfair — the three stages have an information-theoretic interpretation that we simply did not surface. |
| **Response strategy** | Add a *Theoretical Motivation* subsection (Methodology §III.A.1 or a new §III.A.0). Frame the pipeline as: (i) AE compression approximately maximizes the mutual information I(z; X) subject to the bottleneck constraint; (ii) latent perturbation reads out per-band contributions to I(z; X) via finite-difference Jacobian of the reconstruction; (iii) MMR is a submodular diversity term that prevents redundant selection. Cite Tishby & Zaslavsky (information bottleneck), Achille & Soatto (information dropout), Carbonell & Goldstein (MMR submodularity). |
| **Resolved by** | Phase 1 — Methodology rewrite |
| **Paper section** | New §III.A.0 *Theoretical Motivation*; brief mention in Introduction §I |

---

### R1.2 — Critical design choices not justified by ablation

> *"Several critical design choices (e.g., averaging across excitation branches, perturbation strategy, and latent dimension selection) are not sufficiently justified through comprehensive ablation studies."*

| Field | Detail |
|---|---|
| **Status** | **Accept (fully).** This is the most actionable critique. |
| **Response strategy** | Build a full ablation table covering: feature-merge (avg/concat/sum/max), perturbation direction (±/+/−), ε magnitude, dim-method (variance/activation/reconstruction/PCA), k (1/3/5/7), MMR λ (0.3/0.45/0.5/0.7), **normalization (NEW)**. Run on Lichens + Collagen Sponges + Drops where applicable. Report mean ± std over seeds. |
| **Resolved by** | Phase 2 — Ablations |
| **Paper section** | New §V.D *Ablation Study* (3-panel table at minimum: design-choice × dataset × accuracy/F1/ARI) |

---

### R1.3 — Are improvements due to method or just dimensionality reduction / noise suppression?

> *"It is unclear whether the reported improvements are due to the proposed method or simply the effect of reducing dimensionality and suppressing noise. Stronger baselines with comparable dimensionality are needed to isolate this factor."*

| Field | Detail |
|---|---|
| **Status** | **Accept.** Genuinely valid — random/variance/PCA aren't enough. |
| **Response strategy** | Add SOTA HSI band-selection baselines at matched K: MCUVE, SPA, ISSC, BS-Net-FC, sparse-LASSO (supervised). Report side-by-side accuracy at K = {3, 5, 7, 13, 30, 50}. If our method still wins, the noise-reduction-alone hypothesis is refuted: any K-band reduction would suffice, but it doesn't. |
| **Resolved by** | Phase 2 — Baselines |
| **Paper section** | Expand §V.B; new comparison table in §V.B-bis |

---

### R1.4 — Computational overhead, scalability

> *"The proposed method involves both autoencoder training and extensive perturbation analysis, which results in non-negligible computational overhead and may limit scalability to larger datasets."*

| Field | Detail |
|---|---|
| **Status** | **Partial — accept (cost is real), rebut (one-time amortization)** |
| **What's right** | Perturbation is expensive. Reviewer 2 quantifies: ~75% of pipeline time. |
| **What's overstated** | "Scalability" — the cost is one-time per dataset; downstream inference uses only K bands and is K×/N× faster than full-spectrum classification. The trade is paying once upstream for permanent downstream savings. |
| **Response strategy** | (1) Profile and report per-stage breakdown. (2) Implement a `fast` variant: vectorize perturbation across dimensions and ε values, drop low-variance dims earlier. (3) Add a *Computational Considerations* subsection that honestly reports cost AND amortization analysis. (4) Show scalability via Collagen Sponges (smaller) and Drops (different shape). |
| **Resolved by** | Phase 5 — Profiling |
| **Paper section** | New §V.E *Computational Considerations*; brief in §VI Discussion |

---

### R1.5 — Single small-scale dataset, only four classes

> *"The experimental validation is conducted on a relatively small dataset with only four classes, which raises concerns about the generalizability of the proposed approach."*

| Field | Detail |
|---|---|
| **Status** | **Accept.** Both reviewers raise this; the most important structural fix. |
| **Response strategy** | Add Collagen Sponges–Collagen (≥4 classes) and Drop Data (3 unsupervised types) as full datasets. The three together cover: lichens (mixed-media biological), collagen sponges sponges (industrial collagen), and droplets (spatially segregated, blind). Three independent stories converging on the same method. |
| **Resolved by** | Phase 3 — Dataset integration |
| **Paper section** | §IV.A expanded with three datasets; §V.B/C/D one results subsection per dataset |

---

### R1.6 — "Unsupervised" claim weakened by supervised model selection

> *"Although the selection module does not directly use labels, the final configuration appears to rely on supervised evaluation for model selection, making the 'unsupervised' claim less convincing in practice."*

| Field | Detail |
|---|---|
| **Status** | **Partial — accept (Lichens uses labels for the final K), rebut (the selection itself is unsupervised)** |
| **What's right** | Choosing "K=13" as the headline number used the labeled accuracy. |
| **What's overstated** | The selection process — autoencoder training, perturbation, MMR — uses zero labels. Labels only enter at the end for accuracy reporting, not for picking which bands. |
| **Response strategy** | (1) Add **Drop Data** as the cleanest counter-evidence: zero labels, zero supervised K choice, evaluation is internal (ARI vs Ward-on-full-spectrum). Same method, same code path, no labels touched. (2) For Lichens/Collagen Sponges, distinguish *selection* (unsupervised) from *K-choice* (which can be set by an unsupervised criterion — reconstruction-knee or stability — and we'll demonstrate this option). |
| **Resolved by** | Phase 3 (Drop integration) + Phase 4 (Drop figure) |
| **Paper section** | New §V.D *Blind Validation on Drop Data*; expanded §VI.A *Why the framework is truly unsupervised* |

---

### R1.7 — Limited comparison to baselines

> *"The comparisons are mainly limited to random selection and internal variants, without including strong or widely-used band selection baselines, which makes it difficult to evaluate the true advantage of the method."*

| Field | Detail |
|---|---|
| **Status** | **Accept.** Same answer as R1.3. |
| **Response strategy** | See R1.3. The same baseline expansion answers both. |
| **Resolved by** | Phase 2 — Baselines |
| **Paper section** | Expanded §V.B with comparison table |

---

### R1.8 — No multiple runs / variance statistics

> *"The main results are reported without multiple runs or variance statistics, which makes it difficult to evaluate the stability and reproducibility of the proposed method."*

| Field | Detail |
|---|---|
| **Status** | **Accept.** Straightforward fix. |
| **Response strategy** | 5 seeds per (method × dataset × K). Report mean ± std on accuracy, F1, κ, and ARI. Add a selection-overlap metric: Jaccard similarity of top-K selections across seeds. |
| **Resolved by** | Phase 2 — Multi-seed |
| **Paper section** | All result tables updated to "mean ± std (n=5)" format |

---

### R1.9 — k=1 optimality claim possibly inconsistent

> *"The claim that 'k=1 is optimal' is not fully consistent with experimental results, where configurations with multiple dimensions may perform better."*

| Field | Detail |
|---|---|
| **Status** | **Partial — verify before accepting.** Current paper Table V shows: Variance(1)=0.8606 > Variance(3)=0.8547 > Variance(5)=0.8489. So in the *reported* numbers k=1 *is* optimal. The reviewer may be reacting to: (a) absence of multi-seed (a wider distribution might erase the gap), or (b) the prose "fewer dimensions yield better selection" sounds too strong as a universal claim. |
| **Action** | Audit the dim-sweep data multi-seed. If k=1 holds robustly → keep but soften prose to "k=1 is optimal in our experiments; this may reflect that the top variance dimension carries the most discriminative pattern." If k>1 sometimes wins → report the full distribution, drop the universal claim. |
| **Resolved by** | Phase 0 (audit) + Phase 2 (multi-seed) |
| **Paper section** | §V.F (formerly §V.E) prose softened; full distribution in Ablation Table |

---

### R1.10 — Perturbation magnitudes inconsistent between methodology and experiments

> *"The perturbation magnitudes described in the methodology and those used in the experiments are inconsistent, which may cause confusion and affect reproducibility."*

| Field | Detail |
|---|---|
| **Status** | **Verify, then accept or rebut.** The paper currently says ε ∈ {15, 30, 45} in both methodology and setup sections. Need to grep the codebase for actual values used. |
| **Action** | Cross-check `spectral_select/Analyzer` and `Config` defaults against the paper. Document the canonical set. If experiments used a subset (e.g., only ε=30), say so explicitly. |
| **Resolved by** | Phase 0 — Audit |
| **Paper section** | Methodology §III.C.2 and Setup §IV.B.2 made consistent |

---

## Reviewer 2

### R2.1 — Method doesn't reflect ME-HSI physics; applies to any HSI

> *"The method does not incorporate or reflect the underlying physical mechanisms of multi-excitation spectroscopy. Because it is purely data-driven, the framework is essentially applicable to any standard hyperspectral image, rather than being uniquely tailored or necessary for ME-HSI data."*

| Field | Detail |
|---|---|
| **Status** | **Rebut (mostly), with strengthened text** |
| **Why rebut** | The architectural commitments **are** ME-HSI-specific: (i) parallel encoder branches that handle **variable** emission band counts per excitation (a direct consequence of Rayleigh cutoff, which is ME-HSI physics); (ii) Rayleigh masking as a physics-based preprocessing step (excluding `em < ex + 40` and the 2nd-order `|em - 2*ex| < 40`); (iii) feature-merging via cross-excitation averaging, which is meaningless for single-excitation HSI. A 3D HSI dataset would have one branch, no Rayleigh mask, no cross-excitation merge — i.e., would degenerate to a much simpler design. |
| **Response strategy** | Add an explicit subsection *§III.B.0 ME-HSI-Specific Design Commitments* that names each architectural choice and the physics it reflects. Add a remark in §III.B.1 that the architecture **degenerates** to standard 3D-HSI design when N_ex = 1, which validates that the extra structure is ME-HSI-specific rather than incidental. |
| **Resolved by** | Phase 1 — Method strengthening |
| **Paper section** | New §III.B.0 *ME-HSI-Specific Design*; tightened prose throughout §III.B |

---

### R2.2 — Single dataset, limited sample size, insufficient generalization

> *"The experimental validation relies on a single dataset with a limited sample size. ... fails to prove that the proposed method can serve as a universal framework for ME-HSI applications."*

| Field | Detail |
|---|---|
| **Status** | **Accept.** Same as R1.5. |
| **Response strategy** | See R1.5. Three datasets total. |
| **Resolved by** | Phase 3 — Dataset integration |
| **Paper section** | §IV.A, §V.B/C/D |

---

### R2.3 — No SOTA HSI band-selection comparison

> *"There is a severe lack of comparative analysis. The paper does not compare the proposed approach with existing state-of-the-art hyperspectral band selection methods to demonstrate its superiority."*

| Field | Detail |
|---|---|
| **Status** | **Accept.** Same as R1.3. |
| **Response strategy** | See R1.3. SOTA baselines at matched K. |
| **Resolved by** | Phase 2 — Baselines |
| **Paper section** | Expanded §V.B + comparison table |

---

### R2.4 — Single classifier (KNN); no NN classifiers

> *"The selection of the classification method appears arbitrary. The evaluation relies exclusively on a single classifier without testing various machine learning algorithms to verify the robustness of the selected bands. Notably, neural network classifiers are also not considered or evaluated."*

| Field | Detail |
|---|---|
| **Status** | **Partial — accept (multi-classifier is good practice), but explain the original KNN choice** |
| **Original rationale** | KNN was chosen because it is *non-parametric* and makes minimal assumptions, so accuracy depends primarily on feature quality rather than classifier capacity. This is a defensible choice **for isolating the effect of band selection** — but the reviewer's broader point (does the selection generalize across classifier choice?) is legitimate. |
| **Response strategy** | Keep KNN as the controlled baseline but add SVM (linear + RBF), Random Forest, MLP, 1D-CNN. Report a table showing accuracy across (method × classifier × K). If our selection wins across all classifiers, the band-quality claim is generalized. We *already have* `experiments/collagen_classifier_comparison.py` — port and expand. |
| **Resolved by** | Phase 2 — Multi-classifier |
| **Paper section** | New §V.B sub-table *Classifier Robustness*; rationale paragraph in §IV.D |

---

### R2.5 — Compute cost (2+ hours, 75% in perturbation)

> *"The computational overhead of the proposed method is excessively high. The full pipeline requires over two hours to complete, with approximately 75% of the computational time consumed solely by the perturbation process."*

| Field | Detail |
|---|---|
| **Status** | **Partial.** Same as R1.4. |
| **Response strategy** | See R1.4. |
| **Resolved by** | Phase 5 — Profiling + fast variant |
| **Paper section** | New §V.E *Computational Considerations* |

---

## Cross-cutting summary

| Theme | Reviewer points | Phase | Effort |
|---|---|---|---|
| **More datasets** | R1.5, R2.2 | 3 | Medium (Collagen Sponges & Drop integration) |
| **More baselines** | R1.3, R1.7, R2.3 | 2 | High (SOTA implementations) |
| **More classifiers** | R2.4 | 2 | Low–Medium |
| **More seeds & variance stats** | R1.8 | 2 | Low (rerun) |
| **More ablations** | R1.2 | 2 | High (combinatorial) |
| **Theoretical grounding** | R1.1 | 1 | Medium (writing) |
| **ME-HSI specificity** | R2.1 | 1 | Low (text emphasis) |
| **Unsupervised emphasis** | R1.6 | 3, 4 | Medium (Drop integration) |
| **Computational profile** | R1.4, R2.5 | 5 | Medium |
| **Internal consistency** | R1.9, R1.10 | 0 | Low (audit) |

Total fixes: **15 of 15 reviewer points** addressed; **0 pure rebuttals** (all involve some substantive change), **3 partial rebuttals** (R1.4, R1.6, R2.1, R2.4) where the criticism is partially overstated.

---

## Reading order for the reviewer

When the rebuttal letter is written, address points in the order: R2.1 (physics rebuttal), then dataset/baseline points together (R1.5+R2.2+R1.3+R1.7+R2.3), then unsupervised (R1.6 via Drops), then ablation/theory/cost. The goal is that the first three responses reframe the entire critique before they read the rest.
