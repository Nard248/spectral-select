# Sim-to-real: calibrating the digital twin of the ME-HSI lab

Code: `src/spectraforge/studio/realdata.py`, `calibration.py`; study: `scripts/studio_sim2real.py` (about 6 min);
numbers: `docs/studio/figures/sim2real_numbers.json`; figures `fig27`-`fig29`; result library
`src/spectraforge/studio/data/calibrated_fluorophores.json` (`calibration.calibrated_fluorophores()`).
Tests: `tests/studio/test_realdata.py`, `test_calibration.py` (real-data tests skip without `Data/`).

## 1. Method

**Real data.** `load_real_cube(path)` reads the lab's processed `SpectraData` pickles into a `RealCube`
(per-excitation cubes with their own band lists, mask, ROIs from `roi_regions.json` or a label image).
`real_roi_eem(cube, roi)` returns an `analysis.EemResult` identical in structure to `analysis.eem` (same 2 nm union grid,
NaN outside each excitation's band span), so measured and simulated EEMs can be subtracted directly.

**Fast forward model** (thin, clear, optically dilute layer). From `capture.fluorescence_spectrum`:

    S[ex,b] = sum_k a_k A_k(ex) M_k(ex,b)
    A_k(ex)  = sum_lam line_ex(lam) exc_k(lam) dlam        (source.line_shape x Gaussian excitation)
    M_k(ex,b)= sum_lam R_b(lam) e_k(lam) dlam               (sensor.band_responses: passband x long-pass x LCTF peak T x QE x photon energy)

`a_k = c_k eps_k Phi_k ln10/10` is not separable from relative data, so calibrated specs carry nominal
`quantum_yield=0.5`, `extinction=1`. Gains (lamp power, exposure, geometry) are profiled out in closed form
(variable projection, Golub and Pereyra 1973): `gain="global"` (one scale; excitation parameters identifiable) or
`"per_excitation"` (excitation position of one component not identifiable, pinned at the prior).
Objective: per-excitation relative residual (or log), soft-L1 robust loss, deterministic multi-start, covariance from the
Jacobian `s^2 (J^T J)^+` (statistical only). Optional shared instrument terms: smooth log-throughput polynomial
(`response_degree`, Gaussian prior) and ripple orthogonal to cubic polynomials.

## 2. Validation of the machinery (twin to twin)

| check | result |
|---|---|
| fast model vs full `capture` (3 spectra, 5 excitations, 17 bins each) | max relative deviation 4.1e-4 (target 3 %) |
| noisy full captures (8 seeds, detector chain on, dark-subtracted), global gain, truth em 470/80 nm, ex 350/70 nm, high signal | em peak bias +0.02 nm (sd 0.06), ex peak bias -0.001 nm, CI95 coverage 100 % (4 of 5 parameters; ex FWHM 25 %, bias 0.06 nm) |
| same, low signal (4 ms) | em peak bias +0.31 nm (sd 0.63), FWHM +0.7 nm, coverage 100 % |

Without dark subtraction the low-signal em peak was biased by -1.5 nm (dark pedestal), so real data must be dark corrected.

## 3. Collagen (fig27)

Pepsin ROI 2 and five more ROIs (Pepsin 2-4, Acetic Acid 1-3), per-excitation gain, 1 component, 16 starts.

| model | R2 (6 ROIs) | mean SAM |
|---|---|---|
| nominal twin instrument | 0.21-0.78 | 11.0-16.9 deg |
| + smooth throughput correction (degree 2) | 0.80-0.92 | 6.2-7.8 deg |
| 2 components | 0.87-0.93 (no better than 1, higher BIC) | |

Pepsin ROI 2: R2 0.919, SAM 7.0 deg (nominal 0.638, 14.9 deg); per-pixel SAM to the model median 9.0 deg.
Identifiable emission descriptors (the Nuance window starts at 420 nm, so `em_peak_nm` sits on its 380 nm lower bound
for all collagen ROIs and is NOT identifiable): centroid above 420 nm 498-505 nm, red half-maximum 516-526 nm, consistent across both
datasets. Pixel bootstrap interval is under 0.1 nm: sampling error is negligible, systematic error dominates.

Residual structure (fig27 residual panel): (i) first band (420 nm) is over-predicted (long-pass edge / LCTF width),
(ii) an excitation-dependent tilt (385 nm: blue deficit, red excess) that a single fluorophore cannot produce,
(iii) common band-to-band ripple (about 5 %) at all excitations (LCTF transmission).

**Radiometric finding.** The per-excitation gains imply that the processed collagen units are not consistent across
lines: relative to 340 nm the gains are 1.3 (310), 1.0 (325), 1.0 (340), 1620 (365), 3660 (385), 10060 (400) (Pepsin), the same
pattern for Acetic Acid. The `(ADU - dark)/(exposure x lamp power)` normalisation has a roughly 1000x step between 340 and
365 nm, so the global-gain mode is not valid for collagen; it is valid for the droplet `rad_merged` data.

**Systematic uncertainty from the instrument.** Refitting under alternative twin instruments (LCTF FWHM 10/14 nm, long-pass 405/420 nm, UV-enhanced
CCD): collagen centroid 497-499 nm, red half-max 514-516 nm; bright droplets centroid 470-480 nm, red half-max 495-509 nm.
A flat LCTF throughput (unphysical) moves them by 20-50 nm, so the calibrated emission is conditional on the twin's LCTF
transmission curve. A smooth throughput error is degenerate with the emission peak/skew (a tilt biases the peak instead of being absorbed).

## 4. Hold-out validation (fig28; mean SAM, deg; lower is better)

* B1, calibrate on 3 excitations, predict the other 3 (6 ROIs x 4 splits): corrected twin 8.4, twin + ripple 8.6, 2-comp 12.3, nominal 14.2;
  baselines: empirical pooled spectrum 7.7, flat 23.7; in-sample refit 6.4.
* B2, calibrate on one ROI, predict another (30 pairs): corrected twin 7.8 (same dataset) / 7.5 (other dataset); empirical 6.9 / 6.7;
  flat 23.7; in-sample 6.8.
* B3, droplets, global gain, train 310/340/385/415 nm, predict 325/365/400 nm absolutely (no refit of any gain), 16 drops: median R2 0.956
  (nominal 0.951), median SAM 6.0 deg, median |ln(predicted/measured peak)| 0.17 (about 17 % amplitude error).

Honest reading: the twin predicts held-out excitations, samples and across datasets with SAM near its in-sample value (about 1-2 deg above), removes about
90 % of the flat-baseline error, and predicts excitation-line amplitudes of the droplets to about 17 %. It does NOT beat a non-parametric copy of another
measured spectrum, and the excitation-dependent shape changes of collagen are not predicted by 1 or 2 components. Its advantage is that it
transfers to instrument settings the empirical copy cannot represent.

## 5. Droplets (fig29)

Per-drop global-gain calibration (R2 0.93-0.97, SAM 4.4-8.3 deg). Ward clustering gives 3 groups: bright (drops 12, 13, 16; the
TPAMI type 2), mid (8), dim (5). Bright: excitation 373 nm (FWHM 70), emission peak 441 nm, centroid 480 nm, skew at its bound 1.5 (steep
blue edge, partly absorbing the instrument edge); fitted drops agree to within 7 nm in excitation peak. Mid/dim: broad blue emission, emission peak
at bound 380 nm, excitation 328-351 nm, the same spectral family as collagen (centroid about 500 nm).
**Dye identities are not recorded anywhere in the repo** (the droplets are "fluorescent drops of varied composition", an unlabeled
dataset), so the literature check requested cannot be made. Nearest library entries (EBFP2 383/448, optical brightener, collagen) are hypotheses only.

## 6. Scattering

`calibrate_scattering` returns not identifiable: all bands start at least 25-30 nm above the excitation line (Rayleigh cut-off) and there is no flat-field
white reference (the droplet `Whitelight` is a scene), so no elastic channel exists.

## 7. Limitations

Gaussian excitation and split-Gaussian emission only; em peak below 420 nm unobservable; relative amplitudes only (no concentration or quantum yield);
instrument curves are datasheet approximations; samples assumed thin and dilute (collagen sponges and drops are not, so the "effective" parameters absorb
scattering and thickness).
