# E3 findings: cover confounding of the optical redox ratio (re-analysed with the centred Sobol' estimator)

Re-analysis of the stored runs (no re-simulation): `E3_sensitivity_redox.py --reanalyse`. Pre-fix numbers are kept in `results_pre_centering.json`.

## What changed
`sensitivity._sobol_point` computed S1 = mean(fB (fAB - fA)) / Var on uncentred outputs. The estimator is unbiased, but its variance scales with mean(Y)^2, so for band signals (mean >> spread) S1 was noise (peak wavelength: sum S1 = -13.0; redox ratio: -0.04). Outputs are now centred on mean(fA, fB) (recomputed inside each bootstrap resample). ST (Jansen) is shift-invariant and unchanged. Morris and the GP-surrogate group indices (own standardisation) are unaffected.

## Design / noise
N = 64 base rows, k = 8, 640 runs (0 failed), 1000 bootstrap resamples; Morris 8 trajectories (72 runs); noise study 6 points x 4 replicates. MC noise floor (noise variance / output variance): NADH 0.011, FAD 0.011, redox ratio 0.007, rr_F 0.006, rr_N 0.020, rr_FN 0.022, peak 0.447. All ratios are far above noise except the peak wavelength.

## Before / after S1 (new S1 [95 % CI]; ST unchanged, with CI)
| input | NADH S1 old->new | FAD | redox ratio | peak (nm) |
|---|---|---|---|---|
| cover thickness | -0.00 -> 0.02 [-0.07,0.12] | 0.35 -> 0.38 [0.11,0.70] | 0.40 -> 0.49 [0.23,0.81] | -5.51 -> 0.13 [-0.10,0.35] |
| mu_s'(500) | 0.06 -> 0.01 | 0.05 -> 0.11 | -0.14 -> 0.03 [-0.12,0.18] | 7.73 -> 0.19 |
| b_Mie | -0.03 -> 0.01 | 0.03 -> 0.08 | -0.09 -> 0.03 | -5.84 -> 0.04 |
| blood | 0.69 -> 0.74 [0.43,1.10] | 0.11 -> 0.17 | -0.15 -> -0.01 | -0.97 -> -0.06 |
| StO2 | -0.02 -> -0.02 | 0.04 -> 0.12 [0.04,0.23] | -0.02 -> 0.10 [-0.02,0.22] | 2.24 -> -0.17 |
| collagen AF | -0.02 -> -0.03 | -0.10 -> 0.01 | -0.13 -> 0.03 | -8.17 -> 0.18 |
| NADH conc. | -0.00 -> 0.02 | -0.00 -> -0.06 | -0.06 -> -0.06 [-0.18,0.06] | -2.74 -> 0.03 |
| FAD conc. | -0.08 -> -0.05 | 0.24 -> 0.29 [0.09,0.57] | 0.17 -> 0.22 [0.09,0.37] | 0.26 -> 0.18 |
| **sum S1** | 0.59 -> 0.70 | 0.70 -> 1.09 | **-0.04 -> 0.82** | **-13.0 -> 0.52** |

ST (unchanged), redox ratio: thickness 1.04 [0.68,1.57], mu_s' 0.31, b_Mie 0.27, blood 0.24, StO2 0.16, collagen 0.19, NADH 0.13 [0.08,0.21], FAD 0.27; sum ST 2.61. NADH band: blood ST 1.13; FAD band: thickness ST 0.95.

Cover-vs-metabolic share (sum over the group of the per-input index):
| output | cover S1 old->new | metabolic S1 old->new | cover ST | metabolic ST |
|---|---|---|---|---|
| NADH band | 0.67 -> 0.73 | -0.08 -> -0.03 | 1.40 | 0.04 |
| FAD band | 0.47 -> 0.87 | 0.23 -> 0.22 | 2.09 | 0.52 |
| redox ratio | -0.14 -> 0.66 | 0.10 -> 0.16 | 2.21 | 0.40 |
| peak | -10.5 -> 0.31 | -2.47 -> 0.21 | 3.51 | 1.23 |
Group (GP surrogate, unaffected; R2_cv 0.73): cover ST 0.91, metabolic ST 0.16.

## Conclusion
The scientific conclusion does not change. Before the fix the per-input S1 of the redox ratio was uninterpretable (sum -0.04); now sum S1 = 0.82, a coherent picture: cover thickness alone explains about half of the redox-ratio variance (S1 0.49), the cover group about two thirds (0.66) versus 0.16 for NADH and FAD (and FAD alone carries it; NADH S1 is about 0). This agrees with the grouped GP result (cover 0.91 vs metabolism 0.16), with the scatter (Spearman rho = -0.67 against thickness, +0.04 NADH), and with Morris (rho_s(mu*, ST) = 0.83; thickness then FAD then mu_s' by mu*). The corrections (FAD/elastic, NADH/cover-AF, both) do not reduce the cover share (group ST 0.93-0.95 vs 0.91); this was already the group-index result and is unchanged.

## Remaining reliability caveats
- ST estimates are noisy at N = 64: single-input ST > 1 occurs (thickness on redox ratio 1.04, blood on NADH 1.13), sum ST 2.5-4.7. Interpret ST only as ranks with CIs.
- Sum S1 outside [-0.2, 1.3]: redox ratio via NADH/cover-AF (rr_N) = 1.85 (cover S1 sum 1.45); this output is unreliable for absolute shares. rr_FN 1.17 and FAD 1.09 are inside the range.
- S1 CIs wider than 0.3: all 8 inputs for peak wavelength (also noise floor 0.45, treat as inconclusive); NADH band (blood); FAD band (thickness, b_Mie, blood, FAD); redox ratio (thickness, mu_s', b_Mie, blood, collagen); rr_F, rr_N, rr_FN similarly. Many S1 values near zero are not distinguishable from zero.
- GP group-index bootstrap CIs are one-sided/degenerate (lower bound of the metabolic ST CI 0.17 above the 0.16 estimate); the point estimates are the usable part. Larger N would be needed for tight CIs.
