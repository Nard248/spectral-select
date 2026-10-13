# Phase 2: experiment assignments (read 30-campaign-common.md first)

## C1: light transport in covered tissue (ids E2, E4, E10)
**E2, cover-thickness response with literature tissue optics.** Use the `tissue_under_collagen` scene, rebuilt
on the W1 tissue materials (literature collagen cover over a dermis-like NADH/FAD/collagen-autofluorescent
layer), and sweep the collagen thickness from 0 to 2 mm (≥ 9 values × 3 replicate seeds, full mode or a
justified preview scale). For each excitation, report:
- (a) the buried-layer signal against thickness, fitting I = I0·exp(−t/δ_eff) per excitation, and comparing δ_eff
  with the diffusion-theory penetration depth δ = 1/sqrt(3μa(μa+μs')) of collagen at λex and λem;
- (b) the collagen autofluorescence fraction from `explain_pixel` against thickness, and the crossover thickness where
  the cover dominates;
- (c) **spectral distortion**: the emission peak shift and the blue/red band ratio against thickness. Repeat with
  blood in the cover (B = 0, 1, 3 %) to show the hemoglobin (Soret/Q-band) inner-filter dips;
- (d) the wedge phantom: the line-profile response against the analytic local thickness, comparing the lateral-transport
  blur with uniform blocks (3D against 1D effects).

**E4, inclusion detectability.**
- A buried spherical fluorescent nodule (r = 0.5, 1, 2 mm) at depth 0.25–3 mm under collagen; a blood vessel (cylinder
  r = 0.25–1 mm) at depth 0.25–2 mm.
- Compute the CNR (`study` metric `cnr`, nodule ROI against a background ROI, using the sensor noise) against depth
  for each excitation band, and the **maximum detectable depth** (CNR = 3) against excitation wavelength and size.
- Show image panels.
- Interpret the result with the effective attenuation.

**E10, performance and convergence.**
- Runtime against photon count; MC relative SE against photons (expect a 1/sqrt(N) slope; fit the slope).
- Thread scaling (1, 2, 4, 8, 14 threads, if numba allows via `numba.set_num_threads`).
- The peel-off against cone estimator noise; preview against full agreement.
- Report the hardware (Apple M4 Pro, 14 cores).

## C2: global sensitivity and confounders (id E3)
Using `study.py` + `sensitivity.py` (Saltelli design, N ≥ 256 base samples in preview mode or whatever
completes in ≤ 25 min; report N and the MC noise floor), on a tissue-under-collagen scene, use these 6–8 inputs:
- collagen thickness;
- collagen μs'(500) and b_Mie;
- the cover blood fraction;
- StO₂;
- NADH and FAD concentrations in the buried layer;
- the collagen fluorophore concentration.

The outputs are:
- the buried NADH band signal (ex ≈ 365, em ≈ 460);
- the FAD signal (ex ≈ 450, em ≈ 530);
- the **optical redox ratio** FAD/(FAD+NADH);
- the peak emission wavelength.

Report the Sobol S1/ST with bootstrap CIs, and Morris μ*/σ as a cross-check. The key scientific question
is: **how much of the variance in the measured redox ratio is caused by the cover (thickness, scattering,
blood) rather than the true metabolic state (the NADH/FAD concentrations)?** Then propose and test a
correction: ratio the signals against the elastic channel or a cover-autofluorescence channel, and show
whether it reduces the confounder indices. Figures: an S1/ST bar chart with CIs, a scatter matrix, and
before/after correction.

## C3: inference and optimal acquisition (ids E5, E6, E7)
**E5, information-optimal acquisition.**
- On the lab-replica instrument (`mehsi_lab_replica` excitations × bands) imaging collagen-covered tissue,
  compute the per-channel Fisher information and the CRLB for (a) cover thickness alone and (b) the joint
  (thickness, buried NADH conc, cover μs').
- Run greedy D-optimal and target-CRLB channel selection for k = 1..30. How many channels reach within 10 % of
  the full 8×31 protocol's CRLB? Report the acquisition-time saving.
- Compare with uniform/random channel subsets (≥ 50 random draws, median and IQR) and with the repo's
  unsupervised channel-selection method (`information.compare_with_channel_selection`), which ties this
  work to the author's channel-selection research.
- Validate the CRLB against the empirical SD of the ML fit on noisy realisations (efficiency).

**E6, thickness mapping.**
- Retrieve the per-tile cover thickness on the wedge phantom and on a tissue-optics wedge, with the
  surrogate fit. Report the bias, RMSE and CI coverage (the fraction of 95 % CIs containing the truth)
  against exposure/SNR (3 levels), using all channels and the k-optimal channels.

**E7, depth-dependent unmixing failure.**
- Linear unmixing (NNLS) of buried NADH/FAD/collagen abundances with pure-component endmembers, as a
  function of cover thickness. Quantify the abundance bias caused by the spectral distortion of the cover.
- Show whether joint multi-excitation EEM unmixing, or endmembers simulated at the right depth, fixes it.

## C4: machine learning and sim-to-real (ids E8, E9)
**E8, sim-to-real.**
- Consolidate the W6 results into a paper-ready figure and table: the calibrated collagen twin, hold-out
  prediction metrics, and drop-dye peak recovery.
- Then use the calibrated collagen fluorophore in a simulated collagen-cover scene and compare a
  simulated EEM shape with the measured one (spectral angle per excitation).

**E9, synthetic benchmark for unsupervised clustering.**
- With `dataset.py`, generate a labelled synthetic dataset: at least 12 captures of a layered tissue scene
  with 3–4 materials or inclusions, randomised cover thickness of 0–1.5 mm, in preview or full mode
  within budget.
- Run unsupervised clustering on the per-pixel EEM features: k-means and GMM on PCA features, plus one
  method from the repo (`src/spectral_select` or `src/channel_select`, e.g. its autoencoder or
  channel-selection pipeline, if it runs in under 10 min; read the repo docs and code to choose).
- Report the ARI/NMI against GT as a function of the local cover thickness (binned), showing how burial
  degrades label-free tissue segmentation, and whether channel selection (information-optimal channels
  from E5 against all channels) changes it.
- This is the bridge to the author's research line (4D hyperspectral unsupervised clustering).
