# E9 findings: label-free tissue segmentation under a collagen cover

Setup: 14 captures (LHS design, seed 9) of 3 side-by-side tissue types (NADH-rich / FAD-rich / stroma buried layers, 1 mm) under a
dense-collagen cover wedge, t0 in [0,0.3] -> t1 in [1.0,1.5] mm, no inclusions; 7 excitations (310/325/340/365/400/415/430 nm) x 31 bands =
217 channels; `fast` mode, 87x64 px, 2e6 x 0.15 = 3e5 MC photons per step; ~1900 tissue px/capture; K=3. Dataset via `dataset.py`
(cached in ~/.cache/spectralforge/e9_dataset{,_b}, not committed; part B = same design/seeds with the 3 extra excitations).
Metrics: ARI/NMI vs tissue type per local-thickness bin; mean +- SE over the 14 captures. Figures `figures/E9_{scene,ari_vs_thickness,channels}`.
Methods: k-means / GMM on z-scored PCA(10) features (raw or 3x3 masked spatial binning), k-means on the latent of the repo's
`spectral_select` CAE (HyperspectralCAEWithMasking, 30 epochs per capture, relu hidden activation; workaround for an even-band-count kernel
shape bug in `enc_conv1`, applied in the script, src untouched). Supervised shrinkage-LDA (4-fold CV accuracy per bin) is a non-label-free reference.

Noise check (E9_noise_check.py, `noise_check.json`): same scene, two seeds, relative pixel noise (std(a-b)/sqrt2 / channel mean, median over channels with
signal) = 8.2 % at the dataset photon budget and 7.3 % at 4x photons -> MC share ~4.3 %, sensor/other ~7.0 %. This is a channel-mean-normalised
metric, so it differs from the C3 per-pixel tally figure (~39 %); in our metric MC noise is not the dominant term. Still, noise matters a lot for pixel-wise
segmentation: 3x3 binning (9x variance reduction) raises per-bin ARI and supervised accuracy strongly at thick covers (below), and was run as the control.

Findings
- Clustering fitted separately in each thickness bin (isolates burial from the thickness gradient), all channels, 3x3 binning: ARI 0.92+-0.05 (0-0.05 mm), 0.91+-0.03 (0.05-0.1), 0.58+-0.04 (0.1-0.2), 0.43+-0.01 (0.2-0.35), 0.33+-0.02 (0.35-0.6), 0.19+-0.02 (0.6-1.0), 0.08+-0.01 (1-1.5 mm). Raw pixels: 0.72, 0.60, 0.43, 0.30, 0.19, 0.12, 0.05. Burial degrades label-free separability monotonically; this is the depth trend at a noise level that includes sensor noise but with MC noise averaged down.
- Binning lifts ARI most at intermediate depth (0.1-0.6 mm: +0.1-0.15) and is irrelevant at the extremes.
- Supervised LDA stays 0.96-1.00 at all depths with 3x3 binning (raw: 0.98 -> 0.43): the tissue information is still in the spectra at 1.5 mm when binned; it is the unsupervised structure (dominant variance) that is lost, not the information itself. At raw noise the accuracy decays 0.98 -> 0.43, so single-pixel noise is part of the raw-pixel degradation.
- Global (whole-capture) clustering fails badly: ARI vs tissue 0.04+-0.01 (k-means raw), 0.07+-0.01 (k-means 3x3), 0.15+-0.02 (GMM 3x3), 0.07+-0.00 (CAE + k-means); NMI between clusters and thickness bin 0.08-0.19. The clusters follow the thickness gradient / illumination edges (see E9_scene), not tissue type. Per-bin ARI of the global fits peaks at 0.2-0.6 mm (GMM 0.40+-0.03) and is ~0 at the thinnest bins (where the global clusters group by thickness).
- The repo's CAE latent does not beat PCA+GMM here (0.07 vs 0.15 whole-capture ARI); it is comparable to k-means. This is a null result for the CAE on this benchmark; no tuning was done (repo default sparsity weight, 30 epochs).
- Channel reduction (per-bin k-means, 3x3 bin, mean over bins 0-0.35 mm; all 217 ch. = 0.71): k=5: E5-optimal 0.18, uniform 0.26, random 0.29; k=15: E5 0.23, uniform 0.48, random 0.48; k=26: E5 0.55, uniform 0.63, random 0.55. The E5 information-optimal sets (designed for cover-thickness estimation) are NOT better, and are worse at k = 5-15, than uniform/random for tissue-type segmentation: sets optimised for one estimand (thickness) discard the emission/excitation diversity that discriminates tissue types. Supervised accuracy shows the same ordering (E5 15 ch. 0.91 -> 0.66 vs uniform 0.97 -> 0.64).
- Reduced sets lose most of the 217-channel performance (k=15 uniform 0.48 vs 0.71), i.e. channel count matters more than selection rule for this task.

Caveats
- One scene family, 3 tissue types differing strongly in fluorophore mix; thickness is perfectly correlated with image column (wedge), so the global fits are confounded by design (the per-bin fits remove this). The 3x3 binning uses the tissue mask and slightly mixes class borders (1 px gap between blocks).
- E5 sets are for eps = 0 photon-limited thickness (`thick`, `thick+amp`, `joint_D`); results.json of E5 was read at run time (full run). Per-pixel noise metric definitions differ from C3's.
- CAE clustering used K = 3 and latent k-means; CAE only ran for all/E5-15/E5-joint-5/uniform15/uniform5 sets.
- Bins with few pixels (thinnest, ~40-90 px per capture) are noisy (GMM global 0-0.1 mm SE up to 0.18).
