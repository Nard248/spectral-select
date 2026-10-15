# E10: performance and convergence of the 3D layered Monte Carlo

Scene: E2 collagen-covered block (0.5 mm cover, 40x40 mm), excitation 400 nm, preview grid 87x64, `timings_s['mc3d']` (one excitation pass + 9 emission nodes).
N = packets per pass. Script `experiments/E10_performance.py`, figures `figures/E10_performance.*`, `figures/E10_estimators_inclusions.*`, data `results.json`, `runs.parquet`.
Hardware: Apple M4 Pro, 14 cores (10P+4E), 48 GB, macOS 26.7.1, Python 3.11.0, numba 0.63.1 (workqueue-policy layer), numpy 2.1.3. Run 2026-10-15, 7939 s total.

## Key numbers

| Quantity | Result |
|---|---|
| Per-pixel MC noise vs N (peel-off, N = 1e4 ... 6.4e5, 1.8 decades, 56 runs) | log-log slope -0.467, 95 % CI [-0.476, -0.459] |
| Per-pixel MC noise vs N (cone, smoothing off) | slope -0.496, CI [-0.503, -0.488] |
| ROI-mean relative SE vs N (peel, 8 seeds, bootstrap) | slope -0.32, CI [-0.45, -0.16]: NOT -1/2, floor ~1 % (see caveats) |
| Runtime (peel) | t = 0.96 s + 12.3 s per 1e5 packets (log-log slope 0.79 only because of the ~1 s fixed overhead); cone: 0.78 s + 5.1 s per 1e5 |
| Thread speed-up (N = 8e4 peel, 5 interleaved reps, paired, median [IQR]) | 2: 1.88 [1.86,1.88] (eff 0.94); 4: 3.60 [3.56,3.62] (0.90); 8: 6.44 [6.26,6.67] (0.80); 14: 7.10 [7.02,7.35] (0.51) |
| 1-thread / 14-thread time | 75.7 s / 10.7 s |
| Bit-identical output across 1/2/4/8/14 threads | YES, all 5 reps, max abs difference 0.0 (full radiance cube) |
| Peel vs raw cone at equal packets | peel 2.1-2.4x slower per packet; per-pixel noise 2.9-3.4x lower; efficiency 1/(sigma^2 T) 3.6-5.3x higher for peel (peak 5.3 at N = 4e4, 3.6 at 6.4e5) |
| Equal-noise comparison (interpolated) | peel reaches the cone's noise at N = 6.4e5 with N ~ 5.7e4 packets (11x fewer) in ~7 s vs 33 s (4.7x less time) |
| Estimator bias (ROI-mean radiance, cone minus peel, 8 seeds/N) | raw cone: mean -0.09 %, |z| < 1.3 at every N; engine-smoothed cone: mean -1.0 %, z -0.7 ... -2.1 |
| Preview (10 % photons, 87x64) vs full (100 %, 348x256, 4x4-binned), 0-2 mm wedge | RMS rel. diff 0.324; full/full (two seeds) 0.132; preview/preview 0.431; sum ratio 0.998; mean rel. diff +1.6 %; corr 0.69 (log 0.81); 2160 px |
| Preview / full cost | mc3d 5.5 s / 49.6 s (single runs) |
| Inclusion overhead (N = 1e5/pass, 9 interleaved rounds, paired median [IQR]) | sphere r = 1 mm: 1.99x [1.99,2.00]; vessel r = 1 mm: 1.74x [1.71,1.75]; host-filled null sphere: 1.59x [1.58,1.60]. Design target <= 1.6x: MISSED for real inclusions (met only by the optically empty one) |

Preview-vs-full interpretation: with pair-normalised relative differences, full/full = sqrt(2) sigma_f gives sigma_f = 0.093 (binned full); preview/preview = sqrt(2) sigma_p gives sigma_p = 0.305; the predicted preview/full difference sqrt(sigma_p^2 + sigma_f^2) = 0.319 vs observed 0.324. The disagreement is therefore explained by MC noise alone; no systematic preview-vs-full bias is detectable (+1.6 % mean, sum ratio 0.998).

## Inclusion overhead profile (no engine change)
Mean steps per photon are unchanged (ratio 1.00 sphere, 0.94-0.98 vessel), so the extra cost is per step, not more steps. The inclusion-map build is negligible (14 ms sphere, 24 ms vessel). Emission passes dominate (6.1 of 6.5 s without inclusion) and slow by 2.0x (sphere) / 1.73x (vessel); the excitation pass by 1.87x / 1.59x. The host-filled sphere (same map, same host optics everywhere, so no boundary physics) already costs 1.58x, so most of the overhead is the per-step voxel region lookup/ray-marching through the inclusion map; boundary handling for the real inclusion optics adds a further ~25 % (sphere). Not profiled at instruction level (numba kernels); to reach 1.6x the voxel lookup would need to be cheapened (e.g. skip it outside the inclusion bounding box / coarse occupancy grid).

## Caveats (honest)
- The workstation was heavily loaded throughout: 1-min load never fell below 4 before the first two blocks (waited the full 20 min for warm-up, 690 s for convergence, loads at wait end 9.5 / 3.9); recorded run loads were 10-99 (partly self-induced by the 14 numba threads). Absolute times are therefore inflated and noisy; ratios come from interleaved A/B reps with tight IQRs (inclusion ratios within +-1.5 %, thread speed-ups within +-3 %). No rep exceeded 3x its condition median, so no sleep artefacts were discarded.
- The 14-thread point (0.51 efficiency) mixes 10 P and 4 E cores and a loaded machine; 8 threads is the cleaner scaling statement.
- ROI-mean SE does not follow N^-1/2: it flattens at ~0.9-1.1 % above N ~ 1.6e5 for all estimators, so the 1/sqrt(N) law is demonstrated for per-pixel noise only. The floor is seed-to-seed variation that does not shrink with the packet count (hypothesis, not tested here: the per-seed random sampling of launch points / fixed-size launch sample set). With 8 seeds each SE value itself carries ~25 % error, and the ROI-based peel/cone efficiency ratios are too noisy to interpret (ranging 0.55-4.9), which is why the figure shows per-pixel noise only.
- Per-pixel noise is the spatial SD of a plane-detrended 24x24 ROI; it includes any real small-scale structure, and the peel-off image is spatially correlated (smoother), so per-pixel efficiency gains are an upper-bound style statement about pixel noise, not about ROI-mean noise.
- "cone_prod" (the engine's N-dependent Gaussian smoothing, sigma 0.25-3 mm) suppresses pixel noise (but with resolution loss and a -1 % ROI-mean bias, z ~ -2) and its noise-vs-N slope (-0.14) is smoothing-limited; it is reported in results.json but not a like-for-like noise measure.
- Photon-count fairness: peel and cone use the same packets per pass; production uses 0.2x packets with peel-off. Timings are CPU/wall time of the whole block run including its ~1 s fixed overhead.
- Preview vs full uses one wedge scene, two seeds, 400 nm only; full-capture time is a single, non-interleaved run.
- Inclusion overhead measured on the E4 scene (16x16 mm block, voxel 0.1 mm, cover 0.5 mm, 400 nm), one N; the ratio could differ for other voxel sizes or geometries.
