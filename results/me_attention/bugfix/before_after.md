# Bug-fix before/after on Collagen_Acetic_Acid (whole-specimen CV, 3 folds)

Notebook config {'dim': 16, 'depth': 3, 'heads': 2, 'dim_head': 8, 'mlp_dim': 4, 'dropout': 0.25, 'embedding_dropout': 0.1, 'epochs': 30, 'batch_size': 256, 'samples_per_specimen': 2500, 'lr': 0.0005, 'weight_decay': 0.0, 'scheduler_gamma': 0.9, 'scheduler_steps': 10}; device cpu. All metrics computed by one function from stored predictions (macro-F1 with fixed labels). `spec vote` = specimens correct by majority vote / total, as a fraction.

## Alignment (B1)

- before: emission axis has 30 positions = indices 0..29 (index fallback), 149 valid cells.
- after: 31 real wavelengths, 420-720 nm, 149 valid cells.

## Test-class composition per fold, seed 42 (B2)

| fold | before (test specimens -> classes) | after |
|---|---|---|
| 1 | [0, 3, 6] -> [0, 1, 2] | [2, 3, 7] -> [0, 1, 2] |
| 2 | [4, 5, 7] -> [1, 1, 2] | [1, 5, 8] -> [0, 1, 2] |
| 3 | [1, 2, 8] -> [0, 0, 2] | [0, 4, 6] -> [0, 1, 2] |

## Results, split seed 42, model seed 42 (pooled over folds; per-fold in JSON)

| code | model | pixel acc | balanced acc | macro-F1 | spec vote | per-fold bal. acc | time s |
|---|---|---|---|---|---|---|---|
| before | attn_circular | 79.8 | 79.8 | 80.4 | 100.0 | 83.9 / 90.2 / 67.0 | 465 |
| before | attn_zero | 79.0 | 79.0 | 79.6 | 88.9 | 84.0 / 91.1 / 63.6 | 332 |
| before | attn_discard | 81.7 | 81.5 | 82.1 | 88.9 | 89.8 / 92.4 / 64.9 | 510 |
| before | lr | 63.6 | 63.7 | 64.3 | 66.7 | 77.2 / 74.3 / 47.1 | 0 |
| after | attn_zero | 81.8 | 81.9 | 82.2 | 100.0 | 94.1 / 76.9 / 76.2 | 275 |
| after | attn_discard | 82.6 | 82.7 | 83.1 | 100.0 | 94.3 / 73.8 / 81.5 | 199 |
| after | attn_circular | 81.6 | 81.7 | 82.0 | 100.0 | 93.8 / 76.6 / 76.2 | 210 |
| after | lr | 73.5 | 73.8 | 74.0 | 88.9 | 84.4 / 73.2 / 64.9 | 0 |

## Variance (after code, edge_mode=zero, pre-registered default, not selected on test results)

Model seeds 42-46 (split seed 42), pooled balanced accuracy %: 81.9, 81.1, 82.6, 82.6, 82.6 -> mean 82.2 +/- 0.7

| split seed | test-fold classes (all folds covered) | attention bal. acc | LR bal. acc | attention spec vote | LR spec vote |
|---|---|---|---|---|---|
| 42 | [[0, 1, 2], [0, 1, 2], [0, 1, 2]] | 81.9 | 73.8 | 100.0 | 88.9 |
| 43 | [[0, 1, 2], [0, 1, 2], [0, 1, 2]] | 81.8 | 79.2 | 100.0 | 100.0 |
| 44 | [[0, 1, 2], [0, 1, 2], [0, 1, 2]] | 81.0 | 75.3 | 100.0 | 77.8 |
| 45 | [[0, 1, 2], [0, 1, 2], [0, 1, 2]] | 80.1 | 75.7 | 100.0 | 88.9 |
| 46 | [[0, 1, 2], [0, 1, 2], [0, 1, 2]] | 80.7 | 77.2 | 100.0 | 77.8 |

Over 5 fold assignments: attention 81.1 +/- 0.8, LR 76.2 +/- 2.1 (balanced acc %).
