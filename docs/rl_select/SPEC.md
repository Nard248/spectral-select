# Reversible RL for excitation–emission pair selection — study spec (lead-owned)

Branch `research/rl-band-selection`, worktree `/Users/narekmeloyan/PycharmProjects/spectral-select-rl`.
Run everything with `PYTHONPATH=src python ...` from the worktree root. Package: `src/rl_select/`.
Contracts: `src/rl_select/types.py` (Task, Objective, SelectionResult) — do not change without the lead.
Student formulation this study starts from (not ground truth): add/remove MDP, J = grouped balanced
accuracy − λ·R_corr − μ·C_hw, REINFORCE, oracle priors, MC look-ahead.

## 1. Research questions (the paper)
- **Q1 Search.** On one labelled dataset, under a matched budget of objective evaluations, does an RL
  policy (add-only, add/remove) find better subsets than classic search (greedy, swap local search, beam,
  SA, GA, random search)? How close to the exact optimum (n = 2, 3)? Expectation: classic search ≥ RL
  per-instance; we report it honestly.
- **Q2 Transfer to new data (headline).** A policy trained on labelled datasets is deployed **without
  labels** on an unseen dataset (state = label-free features only, no reward at test time). Does it beat
  label-free baselines? Does the add/remove (reversible) policy beat a matched add-only policy here?
- **Q3 Few-shot.** With a small labelled evaluation budget on the target (25–400 evals), does the
  transferred policy (+ MC look-ahead for removals) beat search methods at the same budget?
- **Q4 Ablations.** Student's list: add-only; add-only+oracle prior; add/remove; add/remove+MC look-ahead;
  add/remove+oracle+look-ahead; reward shaping (per-step ΔJ vs terminal); PPO vs REINFORCE; redundancy
  penalty λ; hardware cost μ (#distinct excitations) Pareto front.

## 2. Data
Caches `results/rl_select/cache/<key>.npz` (copied from the ME-attention study; keys X, ex_nm, em_nm, y,
groups, coords, specimen_classes, excitations, emissions, grid_index, specimen_map, meta). Raw counts
float32. Features are already physically filtered (first- and second-order Rayleigh rejection: verified,
no pair has |λem − 2λex| ≤ 40 nm), so `valid` = all True by default.

| key | N px | F pairs | classes | specimens (per class) | outer folds k |
|---|---|---|---|---|---|
| lichens1 | 191046 | 192 | 4 | 16 (4,4,4,4) | 4 |
| lichens2 | 22562 | 210 | 5 | 28 (4,10,7,5,2) | 2 |
| collagen_acetic | 46670 | 149 | 3 | 9 (3,3,3) | 3 |
| collagen_pepsin | 39970 | 158 | 3 | 9 (3,3,3) | 3 |

Lichens1 and Lichens2 are SEPARATE acquisitions (Lichens1 = Zenodo 18640119 palette, 2025-07-07; Lichens2 =
2026-01-09, unpublished, class taxa not recorded; corrected 2026-10-28, see docs/datasets/audit_2026-10-22/). Same
sample family and instrument, so Lichens1↔Lichens2 transfer is near-domain; the strict test is cross-family
Lichens↔Collagen.

**Preprocessing (per task, fit on training pixels only):** `L = log1p(clip(X, 0))`; `raw_logmean/std` =
per-feature mean/std of L on training pixels; X = (L − mean)/std (std floored at 1e-6) for train and test.

## 3. Protocol
- **Outer folds** (copy logic from ME-attention `protocol.specimen_folds`): k = min specimens per class;
  per repeat r, seeded per-class shuffle of specimen ids, deal round-robin into k folds; each fold holds
  ≥1 specimen of every class. Test = pixels of the fold's specimens; train = the rest. Repeats r = 0,1,2.
- **J subsample `sub_idx`:** per training specimen, random cap of `max(150, 3000 // n_train_specimens)`
  pixels (seed = repeat*100+fold). All J evaluations use only `Xtr[sub_idx]`.
- **Inner folds for J_sup:** same class-covering deal over the *training* specimens (k_in = min training
  specimens per class, capped at 4). If a class has only one training specimen (lichens2 class 4), split
  that specimen into two pseudo-specimens by its median coordinate along the longer axis.
- Selection only ever sees Xtr/ytr/gtr; Xte/yte only in `evaluate.py`.

## 4. Objectives (`objective.py`)
All cache by `frozenset(S)`, count uncached evaluations in `n_evals`, return float.
- **`JSup` (uses labels):** mean over inner folds of balanced accuracy of shrinkage LDA on `Xtr[sub_idx][:, S]`.
  *Speed trick (required):* per inner fold precompute class sums `[C, F]`, class counts and the pooled
  within-class scatter `[F, F]` on the fold-train part, once. For any S, Σ_w = scatter[S][:, S]/(n−C)
  + shrinkage, means = sums[:, S]/counts; solve; score fold-val rows `Xval[:, S]`; equal class priors.
  Shrinkage: Σ_w ← (1−α)Σ_w + α·(tr Σ_w/|S|)·I with α = 0.1. Target ≤ 0.5 ms per uncached eval at |S| ≤ 20.
  Optional terms: `lam_corr * R_corr(S)` (student Eq. 5, mean |Pearson| over pairs in S, on training
  pixels) and `mu_hw * n_distinct_excitations(S)`; J = BA − λR − μC. Defaults λ = μ = 0.
- **`JRecon` (label-free):** linear reconstruction R² of all F standardized features from S, closed form from
  the training correlation matrix Σ: `R2(S) = tr(Σ[:,S] Σ[S,S]^{-1} Σ[S,:]) / F` (ridge 1e-6 on Σ[S,S]).
- **`JReconPC` (label-free):** same but the targets are the top q = min(20, F) PCA scores of the training data,
  each whitened to unit variance, so small-variance components count equally. R2 = mean over q targets.
- Helper (shared with RL features): `residual_stats(Sigma, S)` → for every candidate c: residual variance
  `Σ_cc − Σ_cS Σ_SS^{-1} Σ_Sc`, the gain in R2 if c is added (`||r_c||² / (F·r_cc)`, r = residual covariance
  column), and for c ∈ S the R2 loss if c is removed. Vectorised, O(F·|S|² + F²|S|).

## 5. Evaluation (`evaluate.py`) — outer test only
`evaluate(task, S, classifiers=("knn","lda","rf","svm"), seed=0) -> dict[clf -> metrics]`. Train on
Xtr[:, S] (cap per training specimen: knn/lda/rf 1000 px, svm 300 px; seeded), predict all of Xte[:, S].
kNN: k=5 (primary metric — continuity with our prior papers' KNN protocol); LDA: sklearn lsqr+auto
shrinkage; RF: 200 trees, n_jobs=1, seed; SVM: RBF, C=10, gamma="scale". Metrics: balanced_acc, macro_f1,
pixel_acc, specimen_vote_acc (copy `metrics()` from ME-attention protocol). Also an "all F pairs" reference
per task. Set-level metrics: n_distinct_excitations, mean |corr| (R_corr), JRecon, JReconPC.
Stability across the outer folds of a repeat: mean pairwise Jaccard and Kuncheva index.

## 6. Baselines (`baselines.py`), each returns `SelectionResult`
Label-free (usable on a new dataset): `random` (seeded), `variance` (top-n raw_logstd), `uniform_grid`
(spread n pairs evenly over excitations, then emission quantiles within excitation), `spa` (successive
projections algorithm on z-scored Xtr[sub_idx]), `css_greedy` (forward greedy on JRecon), `css_greedy_pc`
(on JReconPC), `mmr_var` (relevance = normalised raw_logstd, λ=0.5 · max |cos| to selected; mimics our prior
rule label-free). Label-based filters: `fisher` (top-n Fisher score), `mrmr` (F-statistic relevance −
mean |corr| redundancy). Wrapper search on any Objective with an eval budget B and `trace`:
`random_search`, `sfs` (greedy forward), `sfs_swap` (greedy then best-improvement 1-swap until no gain or
budget), `beam_swap` (width 5 then swap), `sa` (simulated annealing over 1-swaps, geometric cooling),
`ga` (steady-state GA, fixed-size sets, uniform crossover + repair, mutation = swap), `exact` (enumerate all
n-subsets for n ≤ 3 with joblib; record time). All wrappers stop at budget B (uncached evals) and return
best-so-far.

## 7. RL (`features.py`, `env.py`, `policy.py`, `ppo.py`, `rl_search.py`)
**MDP.** State: current set S (mask), phase q ∈ {add, remove}, step counters, tabu list, task features.
Action masking: no duplicate adds, no removing absent pairs, `valid` only, tabu (a removed pair cannot be
re-added for the next L = 3 cycles; the pair just added cannot be removed in the next cycle).
- **Mode `add`:** S0 = ∅ (or init set), add until |S| = n, terminate.
- **Mode `addremove`:** construct to n, then up to T refinement cycles (default T = n). Each cycle =
  REMOVE(c ∈ S) **or STOP**, then ADD(c ∉ S). STOP ends the episode. Repeated subset → −0.01 penalty.
- **Init option:** `empty` | `css_greedy` (start the refinement from the label-free greedy set: tests
  whether learned removal improves a strong label-free start).
- **Reward:** construction r_t = J(S_{t+1}) − J(S_t) with J(∅) = 1/C (chance BA); refinement r = 0 after
  REMOVE, J(S') − J(S_before) after the ADD. Option `terminal`: only J(S_final) at the end. Training reward
  objective = JSup on the training task (labels exist for training datasets); option JRecon/JReconPC.
- **Returned set:** if the objective is observable at test time (Q1, Q3) → best-so-far n-set; label-free
  deployment (Q2) → the final set (best-so-far would need labels).
**Candidate features φ(c, S)** — all label-free unless marked, z-scored/rank-normalised *within the task*
so the policy is scale-free across datasets (F differs: 149–210): ex/1000, em/1000, Stokes shift
(em−ex)/1000, raw_logmean, raw_logstd (rank-normalised), in_S, JRecon gain-if-added / loss-if-removed,
JReconPC gain/loss, residual variance given S, max and mean |corr| with S, #selected pairs sharing c's
excitation, "adds a new excitation" flag, |λem − nearest selected λem| on the same excitation.
Supervised (flag `sup_features`, only when labels are available on the task, i.e. Q1/Q3): Fisher score rank,
oracle priors from exact n=2 enumeration on the task (P_band = frequency of c in the top-1% pairs;
P_pair(c,S) = mean over s ∈ S of normalised pair score J({c,s})).
Global context: |S|/n, n/20, phase one-hot, cycles left/T, current JRecon, JReconPC.
**Policy.** Shared per-candidate MLP φ → h (64); set context = [mean, max] of h over S ⊕ global; add head
and remove head = MLP([h_c, ctx]) → logit; STOP logit = MLP(ctx); value = MLP(ctx). Masked categorical.
**Training.** PPO (clip 0.2, GAE λ 0.95, γ 1, entropy 0.01 → 0.001, lr 3e-4, minibatch updates), B ≈ 64
vectorised episodes per rollout; episodes sample (task, n ∈ {3,5,8,13}, candidate dropout 20% of pairs
masked as invalid as augmentation). Ablation: REINFORCE with batch-mean baseline (student Eq. 14).
torch CPU, `torch.set_num_threads(1)` per process; parallelise runs across processes (≤ 12).
**MC look-ahead (test time, objective observable):** at a REMOVE decision take the top-m (m = 4)
removals under π_remove; for each, sample K = 4 adds from π_add; evaluate J; apply the best (r, a) if it
improves J, else STOP. Every J call counts toward the budget.

## 8. Settings / run grid (drivers in `experiments/rl_select/`)
- **S1 search (Q1, Q4):** all 4 datasets × n ∈ {3,5,8,13} × outer folds of repeat 0 (12 tasks) ×
  budget B = 2000 JSup evals (trace logged) × seeds {0,1,2} for stochastic methods. Methods: random_search,
  sfs, sfs_swap, beam_swap, sa, ga, rl_add, rl_addremove (+oracle, +MC variants), exact n = 2 (and n = 3 if
  time). Plus label-free and filter baselines (no evals). Evaluate every returned set on the outer test.
- **S2 transfer (Q2, headline):** leave-one-dataset-out (train on the other 3) and cross-family (Lichens{1,2}
  → Collagen{AA,Pepsin} and back). Train policies (rl_add, rl_addremove, rl_addremove init=css_greedy) on
  all outer-fold training tasks of the source datasets (repeat 0), 3 training seeds. Deploy label-free on
  the target's outer folds, repeats 0–2, n ∈ {3,5,8,13}. Compare to label-free baselines on the same folds;
  supervised `sfs_swap` with target labels is the upper reference.
- **S3 few-shot (Q3):** S2 policies on the target with JSup observable and budgets B ∈ {25,50,100,200,400}:
  rl_add best-of-k sampling, rl_addremove + MC look-ahead, vs random_search, sa, ga, sfs_swap truncated.
- **S4 cost:** μ ∈ {0, 0.005, 0.01, 0.02, 0.05} in JSup for rl_addremove and sa on lichens1 (Pareto: BA vs
  #excitations).

## 9. Outputs
One JSON line per (setting, dataset, repeat, fold, n, method, seed): SelectionResult fields + `eval`
(per-classifier metrics) + set metrics → `results/rl_select/<setting>/<dataset>.jsonl`. Aggregators write
CSVs + figures to `results/rl_select/summary/`. `results/` is git-ignored: the paper folder
`publications/rl_band_selection/` will hold copied summary CSVs and figures.

## 10. Ownership
- **Agent A (core):** `data.py` (Task construction + folds + inner folds), `objective.py`, `evaluate.py`,
  `baselines.py`, `tests/rl_select/test_core.py`, `experiments/rl_select/bench_core.py` (speed report).
- **Agent B (RL):** `features.py`, `env.py`, `policy.py`, `ppo.py`, `rl_search.py`,
  `tests/rl_select/test_rl.py`. Until Agent A's objective lands, test against a synthetic Task + toy
  objective; then integrate with `objective.JSup`/`JRecon` (import, do not duplicate).
- Drivers/aggregation/figures: later agents. Paper: lead.
Coding: numpy-vectorised, type-hinted, docstrings short, no silent fallbacks (raise). Commit to the branch
with clear messages; never push.
