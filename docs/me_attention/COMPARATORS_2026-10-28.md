# Frozen comparators for the robustness study (fixed 2026-10-28, before any robustness run)

`PREREG_ROBUSTNESS_2026-10-28.md` requires the confirmatory comparator to be "the best clean baseline of each
dataset, i.e. the baseline with the highest clean mean balanced accuracy in the existing clean results", fixed
*before* any robustness run. This file records that choice. It is frozen; do not change it after seeing
robustness results.

All 8 pre-registered baselines (`lr, rbfsvm, rf, hgb, knn, mlp, cnn1d, spectralformer`) were already complete in
the clean sweeps when this was fixed, so the choice does not depend on Phase A (Phase A only adds the attention
models `meas_tokens` and `ex_tokens_lin`).

| Dataset | Priority | Clean source | **Best clean baseline** | Clean mean bacc % | n units |
|---|---|---|---|---|---|
| `lichens2_sub3` | primary | `results/me_attention/sub3/lichens2_sub3` | **`rbfsvm`** | 96.61 | 25 |
| `lichens2` | primary | `results/me_attention/final_recwl/lichens2` | **`rbfsvm`** | 86.57 | 10 |
| `collagen_pepsin` | secondary | `results/me_attention/final_recwl/collagen_pepsin` | **`lr`** | 90.34 | 15 |
| `lichens1` | secondary | `results/me_attention/final_recwl/lichens1` | **`spectralformer`** | 99.47 | 4 |

Full clean baseline ranking (mean balanced accuracy %, from which the above was taken):

- `lichens2_sub3`: rbfsvm 96.61, spectralformer 95.94, hgb 95.84, mlp 95.63, knn 95.26, rf 94.84, cnn1d 93.47, lr 93.44
- `lichens2`: rbfsvm 86.57, spectralformer 86.08, mlp 85.58, hgb 84.62, knn 83.53, lr 83.34, rf 83.25, cnn1d 81.56
- `collagen_pepsin`: lr 90.34, rbfsvm 87.30, hgb 84.67, spectralformer 83.94, rf 83.68, mlp 83.21, knn 81.32, cnn1d 79.20
- `lichens1`: spectralformer 99.47, rbfsvm 99.31, lr 99.17, mlp 98.83, rf 98.62, cnn1d 98.01, hgb 97.50, knn 91.81

The 20 confirmatory tests are therefore: {`lichens2_sub3`, `lichens2`} x {T1, T2, T3, L, C} x
{`ex_tokens_v2`, `ex_tokens_lin`}, each as Delta = candidate - `rbfsvm`, paired by (repeat, fold), one Holm family
of 20.
