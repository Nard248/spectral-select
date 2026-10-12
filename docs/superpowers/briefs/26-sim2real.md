# W6 brief: sim-to-real engineer (digital twin of the ME-HSI lab, fluorophore calibration from real data)

Read `20-v12-common.md` first, then `presets.py` (the `mehsi_lab_replica` preset, which encodes the real
instrument: a tunable lamp 310–430 nm ×8, a Nuance 348×256 sensor, 31 bands from 420 nm, and
band start = max(420, ex + 20)), `optics.py`, `spectra.py`, `analysis.py` (normalisation modes), and the
repo's real-data loaders (`src/spectral_select/types.py` `SpectraData`, and loaders under
`src/mehsi_preprocessor/` / `src/spectral_select/`). Real data is in `Data/processed/`:
- `Collagen Pepsin/spectra_masked.pkl` + `roi_regions.json` + `class_mask.png`
- `Collagen_Acetic_Acid/`
- `Drop Data Radiometric/rad_merged`: drops of known fluorophores, radiometrically normalised
  (exposure + lamp power).
- `Lichens_2_reprocessed/`

You OWN the NEW files `src/spectraforge/studio/realdata.py`, `calibration.py`, `tests/studio/test_realdata.py`,
`tests/studio/test_calibration.py`, `scripts/studio_sim2real.py`, and `docs/studio/SIM2REAL.md`. Data files are
large, so never copy them into the repo. Tests that need real data must `pytest.skip` if `Data/` is absent,
and a small synthetic fixture must cover the logic.

## Build
1. `realdata.py`: `load_real_cube(path) -> RealCube`, with excitations, per-excitation band wavelengths,
   cube (H, W, B) per excitation, exposure/power metadata if present, the mask, and ROIs. It must also
   adapt from `SpectraData`. Add `real_roi_eem(cube, roi_or_class)` → the same `EemResult`-like structure as
   `analysis.eem`, so real and simulated EEMs are directly comparable (same grids, NaN outside the band spans).
2. `calibration.py`: **data-driven fluorophore calibration**. Fit `FluorophoreSpec` parameters (ex/em peak,
   FWHM, em_skew, vibronic, relative brightness per component; 1–3 components) so that the SIMULATED EEM
   of a thin, optically dilute layer seen through the digital-twin instrument (filter passbands, QE, LCTF
   transmission, longpass, lamp spectrum, line FWHM: use the engine's instrument-response pieces from
   `sensor.py`/`spectra.py` directly as a fast forward model rather than full captures, and verify the fast
   model against a full `capture` at one point within 3 %) matches the measured ROI EEM. Use robust least
   squares on log or normalised intensities, multi-start, and parameter CIs from the Jacobian.
   - Output: a list of calibrated `FluorophoreSpec`s, plus fit diagnostics (R², spectral angle per
     excitation, and residual maps).
   - Optionally, `calibrate_scattering`: estimate the effective μs' of the sample from the elastic-leak
     channel, if the data allows. If not, explain why.
3. **Sim-to-real validation study** (`scripts/studio_sim2real.py`):
   - (a) For Collagen Pepsin (and Collagen Acetic Acid if quick), calibrate the collagen fluorophore.
     Compare the measured and simulated EEM (`fig27_sim2real_collagen.png`: measured, simulated, residual
     heat maps and emission slices at each excitation).
   - (b) Hold-out validation. Calibrate on half the excitations and predict the others, or calibrate on one
     sample and predict another. Report the spectral angle and R². This shows the twin predicts, rather than
     merely fits.
   - (c) For the Drop data (several known fluorophores), check that the calibrated peaks match the dyes'
     literature peaks. Find the dye identities in the repo's docs/memory/code (e.g. grep "Drop Data" in
     `docs/` and `publications/`).
   - Write the calibrated specs to `src/spectraforge/studio/data/calibrated_fluorophores.json` with
     provenance (dataset, ROI, fit metrics), and add a function `calibrated_fluorophores()` that loads it.
     The lead may add them to the builtins.
4. `docs/studio/SIM2REAL.md`: the method, equations, results tables, and figures, written so the lead can
   lift them into a paper.
Report all numbers and keep the whole script under 20 minutes.
