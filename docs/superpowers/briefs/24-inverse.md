# W4 brief: inverse engineer (unmixing, parameter retrieval, Fisher information and optimal acquisition design)

Read `20-v12-common.md` first, then `analysis.py`, `capture.py` (the public capture entry points and
`explain_pixel`), `sensor.py` (the noise chain, for the noise model), `block_capture.py`, and `sweeps.py`.
Goal: turn the simulator into a tool for **quantitative inference and experiment design**. It should
answer questions like "how well can I estimate the collagen cover thickness from this camera, and which
excitation/emission bands should I acquire?"

## 1. `unmix.py`: spectral unmixing on captures
- Endmembers from (a) the forward model: each dye's emission through the camera's bands, built from
  the capture's band responses (pure-component captures through the same instrument via toggles or
  single-dye materials; document the approach), (b) user ROIs, and (c) unsupervised NMF (sklearn) or VCA.
- Solvers: per-pixel NNLS (scipy), fully constrained least squares (FCLS: non-negative, sum to one), and
  multi-excitation **joint EEM unmixing** (concatenate the excitations).
- Outputs: abundance maps, residual/RMSE map, and comparison with GT (`origin_layer_fraction`, `dye`
  components from `explain_pixel` on a subsample).
- Validate: perfect recovery on a synthetic linear mixture; on a 2-dye capture, the abundance correlation with
  GT is above 0.95 at high SNR.

## 2. `surrogate.py` + `inverse.py`: parameter retrieval
- `build_surrogate(base_exp, params, grid, metric_fn | "roi_eem", mode, seed)` samples the simulator on a
  grid (or Sobol points) of 1–3 parameters (JSON pointers, e.g. cover thickness, dye concentration, cover
  μs'). It stores the noise-free mean signal: shot/read/dark noise OFF, with a fixed MC seed for **common
  random numbers**. The interpolator is cubic for 1-D, RBF or multilinear otherwise. Cache it to disk.
- `fit(observation, surrogate, noise_model, x0)` performs maximum-likelihood fitting under the
  **Poisson–Gaussian** sensor noise. Variance in ADU² = (signal_e + dark_e)/gain² + read_e²/gain² +
  quantisation 1/12 (derive it from `sensor.py`). Use weighted least squares / LM (scipy
  `least_squares`). Return the estimate, the covariance (J^T W J)^-1, and profile-likelihood CIs.
- `retrieve_map(...)` does per-pixel or per-ROI-tile retrieval to give a **thickness map** against the GT
  `layer_thickness:cover`. Report the bias and RMSE.

## 3. `information.py`: Fisher information, CRLB, and design
- `fisher(base_exp, params, *, bands="all", excitations="all", roi, step)` computes finite-difference
  Jacobians of the noise-free ROI mean signal per (excitation, band) channel, using common random
  numbers (identical MC seed) so the differences are smooth. The Poisson–Gaussian variance is per channel,
  and the ROI averaging of n pixels reduces the variance by n. It returns the Fisher matrix and the CRLB per
  parameter, and per-channel information contributions.
- `optimal_design(..., k, criterion="D"|"A"|"target:<param>", budget="channels"|"exposure_time")`:
  - Greedy forward selection of k (excitation, band) channels maximising log det F (D-opt), or minimising
    the target CRLB.
  - An exhaustive search for small k as a check.
  - Optionally, exposure-time allocation across the selected channels under a total-time budget (convex:
    maximise log det Σ t_i F_i).
  - It returns the ranked channels and the CRLB as a function of k.
- Link to the repo's research (unsupervised channel selection, `src/spectral_select`): provide
  `compare_with_channel_selection(...)`, which runs one of the repo's channel-selection methods on a
  simulated dataset (read `src/spectral_select`, find a method that takes a SpectraData object, and wrap
  it) and reports the CRLB achieved by its selected channels against the information-optimal ones. Skip it
  gracefully if it is too heavy, and document why.
- Validate:
  - The CRLB is checked against the empirical variance of `fit` over 200 noisy realisations of a cheap
    case. They should agree within 20 %, which shows the estimator is efficient.
  - The Fisher matrix of a linear-Gaussian toy model matches the analytic value exactly.
  - The greedy D-optimal choice equals the exhaustive optimum on a small case.

## 4. Router `routes_inverse.py`
An `APIRouter` named `router`, mounted by W3 in `server.py`:
- `POST /api/unmix` (capture_id, method, endmembers spec).
- `POST /api/fisher`.
- `POST /api/design`.
- `POST /api/surrogates` and `POST /api/fit` (long jobs: reuse the job/worker pattern of `sweeps.py`, or
  synchronous with a size limit; document which).
JSON only. Document them in `docs/studio/INVERSE_API.md` for the web engineer.

## 5. Figures (`scripts/studio_inverse_figures.py` → `docs/studio/figures/`)
- `fig24_crlb_vs_thickness.png`: the CRLB of the cover thickness against the true thickness, for the
  lab-replica camera, compared with the empirical estimator SD.
- `fig25_optimal_channels.png`: the information heat map over (excitation × emission band), with the
  greedy picks and CRLB(k).
- `fig26_unmixing.png`: abundance maps against GT.
Use the `tissue_under_collagen` and `layered_wedge_phantom` presets, in preview-scale runs, so it
completes in under 15 minutes total. Look at the figures and make them publication quality.
