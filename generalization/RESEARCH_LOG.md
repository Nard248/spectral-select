# Generalization Research Log

Idea -> result -> next-questions diary for the general channel-selection effort.

---

## 2026-05-23 — Literature landscape (P0)

### Idea
Before committing to the "unsupervised + conditional + discrete" novelty framing, map the
prior art so we position correctly and pick the right baselines.

### Result
The framing is **rhetorically clean but the cell is not genuinely empty**:
- **Concrete Autoencoder** (Balın, Abid & Zou, ICML 2019) — unsupervised, differentiable,
  discrete feature selection via a Concrete/Gumbel selector + reconstruction. **Closest
  relative; mandatory baseline to beat or match.**
- **AEFS** (ICASSP 2018), **DUFS** (2021/22), **Graph-AE UFS** (2018) — deep unsupervised
  feature selection, some redundancy-aware.
- **Unsupervised MRMR band selection** (Geng 2015; MIMR-DGSA) — already unsupervised +
  redundancy-aware + discrete on hyperspectral.
- HAR sensor/channel selection is an established subfield, but **almost all supervised**
  (mRMR, Relief-F, GA wrappers, channel-attention CNNs/transformers).

### Why it matters
Our real, defensible contribution is the **combination** (group-AE + latent-perturbation
reconstruction-sensitivity as the relevance signal + MMR redundancy) and **cross-domain
transfer** (HSI -> IMU HAR) with an identical engine. Perturbation/sensitivity attribution
is a known interpretability primitive, so frame ours as a novel *application/combination*,
not a new attribution paradigm.

### Decisions
1. Add **Concrete Autoencoder** as the headline baseline (P3).
2. Report **LOSO macro-F1** as the primary metric (k-fold inflates via subject leakage on
   PAMAP2/Opportunity, which are class-imbalanced).
3. Baseline set: Concrete AE, Laplacian Score, MCFS, unsupervised-MRMR, ISSC, variance/PCA
   floors (unsupervised head-to-head); mRMR + GA-wrapper (supervised reference upper bounds);
   random x seeds.
4. Venue: lean **Information Fusion** or **Pattern Recognition** (cross-domain method fit).

### Next questions
- Does AE-perturb beat random-K on PAMAP2 under LOSO? (P2 gate, running.)
- How close does label-free AE-perturb get to supervised mRMR/GA on macro-F1?
- Does Concrete AE (also unsupervised) outperform us, and if so on which K?

---

## 2026-05-23 — P1 package built + P2 slice launched

### Result
`channel_select/` package complete and tested (20 tests): protocols/config,
GroupedChannelDataset (+LOSO), shared engine (perturbation/influence/normalize/MMR
generalized over arbitrary group/channel/axis layout), Conv1d grouped temporal AE,
PAMAP2 (raw + MONSTER) adapters. `spectral_select/` core untouched.

MONSTER PAMAP2 downloaded (ungated HF mirror, UCI was unreachable): (38856, 52, 100)
float64, 8-9 subjects, 12 classes. Vertical-slice script runs unsupervised AE -> selection
-> LOSO KNN acc-vs-K + random control on held-out subject 5.

### Next
Record slice numbers here when the run completes; decide P2 gate (beats random?).
