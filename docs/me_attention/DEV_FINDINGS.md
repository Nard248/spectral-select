# Dev findings (engineer E) -- collagen_acetic only, 6 units (2 repeats x 3 outer folds)

Full table: `results/me_attention/dev/LEADERBOARD.md` (71 configs; raw: `dev_results.jsonl`). Code: `src/me_attention/models/attention_v2.py`,
`experiments/me_attention/dev_explore.py` + `dev_stages.py`.  Std over 6 units is 4-10 points, so differences < ~3 points are NOT resolved.

Two modes: **inner** = fit on inner-train (1 specimen/class), early-stop on inner-val, test on outer fold (the spec'd dev protocol;
max 40 epochs, patience 8, 1000 samples/specimen).  **[full]** = fit on the whole outer-train (2 specimens/class), fixed 25 epochs, no ES.
[full] has 2x the training specimens, so it is NOT comparable to inner numbers (it is what the final retrain does). Cross-specimen
objectives (mixup, SupCon positives, adversary) only make sense in [full]: in inner mode every class has one specimen.

## Top 15 (balanced accuracy %, mean +- std)
| # | config | bAcc |
|---|---|---|
| 1 | mlp [full] base (area-gl-log) | 86.7 +- 4.1 |
| 2 | hybrid_conv_attn [full] + scale+noise | 86.5 +- 5.4 |
| 3 | mlp [full] + scale+noise | 86.5 +- 4.4 |
| 4 | ex_tokens, mean pool (inner) | 86.3 +- 5.0 |
| 5 | hybrid_conv_attn [full] base | 86.0 +- 5.9 |
| 6 | hybrid [full] + SupCon 0.3 | 85.4 +- 6.8 |
| 7 | axial (dim32, depth1) (inner) | 85.3 +- 4.3 |
| 8 | hybrid + scale+noise (inner) | 85.3 +- 7.0 |
| 9 | hybrid [full] + adversary | 85.1 +- 5.2 |
| 10 | ex_tokens + scale+noise (inner) | 84.0 +- 6.7 |
| 11-12 | rbfsvm (area, global, +-log; max_train 10000) (inner) | 83.9 +- 6.2 |
| 13 | ex_tokens sub2 conv (inner) | 83.8 +- 5.6 |
| 14 | ex_tokens (inner) | 83.7 +- 4.9 |
| 15 | hybrid + SupCon 0.3 (inner) | 83.5 +- 8.2 |
Inner-mode references: v0 82.2 +- 7.6, LR(area-gl-log) 81.9 +- 8.3, hgb 81.2, mlp 81.0, cnn1d 79.5, knn 78.4. Not finished in the time box
(killed, no result): spectralformer(dim32,depth3), bandvit, meas_tokens variants (3.4 s/epoch on mps with contention; reduce for final).

## Step 1 -- preprocessing (the one thing that clearly mattered)
* `norm_scope=per_ex` is much worse than `global` (mlp 73 vs 81; lr 77 vs 81; hgb 73 vs 81). Per-excitation normalisation destroys
  the cross-excitation intensity ratios, which are class-informative and stable across specimens.
* `norm=none` is the worst for neural nets (mlp 73.7 lin / 79.0 log); area or l2 (global) fixes it (81). LR/SVM/hgb are insensitive.
* log1p: neutral (identical for trees, +-0.5 elsewhere). Best overall dev preprocessing: **area, global, log1p** (used in all later runs).

## Step 2 -- existing attention (inner mode)
v0 82.2, ex_tokens 83.7 (mean-pool 86.3), ex_tokens sub2conv 83.8, axial small 85.3.  None is separated from rbfsvm 83.9 or mlp 81-83 given
+-5-7 std.  Augmentation (scale+noise) gives +0.3 on v0/mlp, +0.3 on ex_tokens, +3.3 on hybrid (5/6 wins) -- the only consistent aug gain.

## Step 3 -- new mechanisms (paired delta vs the same model's base, 6 units)
| mechanism | hybrid inner | hybrid [full] | mlp inner | mlp [full] |
|---|---|---|---|---|
| scale+noise aug | +3.3 (5/6) | +0.5 (5/6) | +0.3 | -0.3 |
| per-excitation scale aug (a) 0.4 / 0.8 | -7.0 | -- | -8.7 / -9.6 | -4.4 / -6.7 |
| + spectral smoothing jitter | -- | -- | -8.1 (with exscale) | -- |
| log-intensity pixel/ex centring (b) | -8.2 (pixel) | -3.3 (pixel) | -3.2 / -2.6 | -- |
| conv-stem + attention (c) vs best MLP | 82.0 vs 81.0 | 86.0 vs 86.7 | | |
| SupCon 0.3, cross-specimen positives (d) | +1.5 | -0.6 | +0.8 (5/6) | not run |
| conditional specimen-adversary (e) | n/a (degenerate, 1 spec/class) | -0.9 (1/6) | n/a | not run |
| excitation dropout 0.25 (g) | +0.4 | not run | **+2.3 (5/6)** | not run |
| same-class cross-specimen mixup (f) | n/a | not run (killed) | n/a | not run |
Honest reading:
* **Invariance tricks hurt.** Per-excitation scale augmentation and brightness centring make things 3-10 points worse in every model. On this
  dataset the absolute excitation-to-excitation intensity ratios are acquired under fixed settings and are *discriminative*; the "drift"
  they simulate does not exist between collagen specimens (it may on lichens -- test via inner-val there, not here).
* SupCon / adversary / mixup: no gain beyond noise (+1.5 / -0.9 / not run). Adversary and mixup not tested on the MLP (time box).
* Excitation dropout helped the MLP (+2.3, 5/6 wins) but not the hybrid (+0.4): not attributable to attention; cheap enough to keep as an option.
* **Attention does not beat tuned baselines on dev.** Best attention (ex_tokens mean 86.3, hybrid[full] 86.0-86.5) ~ MLP[full] 86.7 and
  rbfsvm 83.9 (different mode). The ablation says the aug gain and the best numbers are not specific to attention.
* Not done: mixup runs, adversary/SupCon on MLP, temperature/depth studies, meas_tokens/SF/bandvit refs, and any tuning of baselines per fold.

## Step 4 -- recommendation for the final sweep
Frozen spaces are in `ATTENTION_V2_MODELS` (attention_v2.py), track spectral, family ours, built with `TorchEstimatorV2`.
Default = dev-best preprocessing (area, global, log1p), aug scale+noise, samples_per_specimen 1000.
| model | space (model keys) | shared keys | params (F=192, E=8) | mps s/epoch (16 spec x 1000, aug scale+noise, uncontended) |
|---|---|---|---|---|
| `hybrid_conv_attn` (default dim64 depth2 chunks6) | dim {32,64}, depth {1,2,3}, n_chunks {4,6,8}, dropout {0.1,0.3} | aug {none, scale+noise}, lr {3e-4,1e-3,3e-3}, norm {none,l2,area}, log1p {F,T} | 75.5k | 3.0 |
| `ex_tokens_v2` (default dim64 depth3, pool=cls) | dim {32,64}, depth {2,3}, sub_tokens {1,2}, dropout {0.1,0.3} | same | 112.5k | 2.5 |
(timings: file `/tmp/tm.py` pattern, measured after killing the sweep; the s/epoch is dominated by CPU numpy augmentation + MPS kernel launch, not model size -- port aug to torch on-device in train.py to speed up.)
Add `pool` {cls, mean} to ex_tokens_v2 if budget allows: mean-pool was the best existing-attention run (86.3 vs 83.7), an unreplicated single-config observation.
A third variant is not recommended; `meas_tokens` is too slow (150 tokens) and unevaluated here.

* **Fairness / all models:** keep `norm in {none,l2,area}`, `log1p`, but **drop `per_ex` scope** (or keep it only if the lead insists on spec fidelity; it lost 4-8 points for all model families, so it wastes search budget). Remove `max` unless needed (untested).
* **Port into shared `train.py`:** (1) `ex_drop` (zero whole excitation block, ex-level dropout) -- the only objective trick with a consistent gain on a non-attention net;
  (2) torch-side augmentation to avoid per-batch numpy preprocessing. Do NOT add per-excitation scale, centring, adversary, SupCon or mixup as defaults.
  `TorchEstimatorV2` (fit loop with all of these, marked `PORT`) is the reference implementation; `degenerate_` flags configs where a
  cross-specimen objective cannot act (<2 specimens per class).
* Spatial-capable variant: none built (hybrid/ex_tokens encoders can be plugged as `spec_encoder` of `spatial_attn` unchanged -- `ExTokens` already is).
* Code issues found: `adaptive_avg_pool1d` fails on mps for non-divisible sizes (used pooling matrix instead); `TorchEstimator` caches preprocessed
  data only when `aug=none`; with `inner_split` each class keeps 1 train specimen on 3-specimen datasets, so inner-val tuning cannot assess cross-specimen objectives.
