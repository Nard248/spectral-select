# ME-Attention robustness study — pre-registration (frozen 2026-10-28)

**Status: FROZEN.** Written before any robustness result exists. Do not edit the conditions, levels, hypotheses,
model list or decision rule below. If something cannot be run as written, record it in the *Deviations* section at the
bottom (what, why, when) and run the closest faithful variant. Every condition is reported for every model, including
the ones where attention loses.

## Motivation

On the clean benchmark the attention models are statistically tied with strong classical baselines (Lichens 2 3-class
subset: RBF-SVM 96.6, ex_tokens_v2 95.7, kNN 95.3 balanced accuracy; specimen vote 100% for all models). The clean
task is near ceiling, so it cannot separate the models. The question here is whether attention models degrade *less*
under realistic perturbations of the instrument and of the annotation. That is a robustness claim, not a clean-accuracy
claim.

## Datasets (in run order)

| Key | Env | Repeats × folds | Priority |
|---|---|---|---|
| `lichens2_sub3` | none (recorded wavelengths built in) | 5 × 5 = 25 | primary |
| `lichens2` | `ME_ATTENTION_RECORDED_WL=1` | 5 × k | primary |
| `collagen_pepsin` | `ME_ATTENTION_RECORDED_WL=1` | 5 × 3 = 15 | secondary |
| `lichens1` | `ME_ATTENTION_RECORDED_WL=1` | 1 × 4 = 4 | secondary, last (largest) |

Same outer whole-specimen folds and seeds as the clean sweeps (`sub3`, `final_recwl`).

## Models

Spectral track only. Baselines: `lr, rbfsvm, rf, hgb, knn, mlp, cnn1d, spectralformer`. Attention:
`smartattention_v0, ex_tokens_v2, hybrid_conv_attn, meas_tokens, ex_tokens_lin`.

Pre-specified attention candidates for the confirmatory test: **`ex_tokens_v2` and `ex_tokens_lin`**. Comparator:
the **best clean baseline** of each dataset, i.e. the baseline with the highest clean mean balanced accuracy in the
existing clean results (`sub3` for `lichens2_sub3`, `final_recwl` for the others). Fix it from the clean results before
running anything below and write it in the report.

## Training protocol (no re-tuning)

For each (dataset, model, repeat, fold), take `best_config` from the clean record (it already contains the tuned
`epochs`). Retrain on the same outer training split with the record's `seed`. No inner validation and no early stopping:
train for exactly `best_config["epochs"]`. Hyperparameters are never re-tuned on perturbed data.

## Perturbations

All perturbations act on the **raw** feature matrix `X` (before the model's own preprocessing), so every model receives
byte-identical inputs. The RNG seed for every perturbation is derived from `(dataset, repeat, fold, family, level)`
only, never from the model. Excitation block = all features of one excitation (`MEData.excitations`).

### Test-time families (one fit per unit, evaluated on every level; level 0 = clean)

- **T1 — missing excitations.** Remove m ∈ {1, 2, 3} excitation blocks from the test set. For each m, draw 10 random
  subsets of excitations (the same subsets for all models) and average over them. A removed block is set to the
  training-set value that maps to 0 after the shared log1p + standardise preprocessing, i.e. per feature
  `expm1(mean(log1p(X_train)))`. This equals the zeroing used by the `ex_drop` augmentation, so models trained with
  `ex_drop` see the same kind of input they were trained on.
- **T2 — per-excitation gain drift.** Multiply each excitation block of each test *specimen* by
  `g = exp(N(0, σ²))`, σ ∈ {0.1, 0.25, 0.5}. One factor per (specimen, excitation), the same for all pixels of that
  piece. This mimics LED power or exposure error between acquisitions (cf. the 10× exposure error found in Lichens 2).
- **T3 — measurement noise.** Add Gaussian noise with std `s · std_train(feature)` (raw space), s ∈ {0.1, 0.25, 0.5,
  1.0}, then clip at 0.

### Training-time families (refit per level; test set clean)

- **L — label noise.** Flip a fraction p ∈ {0.1, 0.2, 0.3, 0.4} of training pixels uniformly to one of the other
  classes (symmetric noise).
- **C — learning curve.** (a) Pixels per training specimen n ∈ {25, 100, 400, all}, drawn uniformly. (b) Training
  specimens per class s ∈ {1, 2, all} (only values smaller than the available count; `collagen_pepsin` has 2 per class,
  so s ∈ {1, all}). Where a hyperparameter becomes invalid (for example kNN `n_neighbors` above the training-set size),
  clip it to the nearest valid value and log that.

## Endpoint and statistics

- Metric: test balanced accuracy, paired by (repeat, fold).
- Per family, the **robustness score** of a model is its mean balanced accuracy over that family's non-clean levels
  (T1 also averaged over its 10 subsets). Also report the full degradation curve per level.
- Confirmatory comparisons: Δ = candidate − best clean baseline, per (dataset, family, candidate), for the two
  primary datasets and five families: 2 × 5 × 2 = 20 tests. Nadeau–Bengio corrected paired t-test, one family of
  Holm correction across all 20. 95% bootstrap CI over folds.
- **Decision rule:** attention has a robustness advantage on (dataset, family) if Δ > 0 and the Holm-adjusted p < 0.05.
  Everything else (secondary datasets, other attention models, comparisons against every baseline, the best baseline
  *per condition*) is exploratory and labelled as such.

## Predictions written before running

- H1 (T1, missing excitations): attention degrades less. **Most likely genuine advantage.**
- H2 (T2, gain drift): attention degrades less than kNN / RBF-SVM. Plausible.
- H3 (T3, noise): no clear prediction.
- H4 (L, label noise): kNN degrades the most; attention ≈ other neural models.
- H5 (C, learning curve): **attention degrades more** than kNN / RBF-SVM. Reported regardless.

## Deviations

(append below: date, what changed, why)
