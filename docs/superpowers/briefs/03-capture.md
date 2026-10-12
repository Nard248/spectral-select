# Brief — capture engineer (material optics, spectra, sensor, capture orchestration)

Read `00-common.md` first. Also read the legacy engine `src/spectraforge/{fluorophore,forward,physics,artifacts}.py`
(reuse `Fluorophore` excitation/emission shapes, `raman_center_nm`, FRET semantics) and
`src/spectral_select/types.py` (SpectraData/ExcitationData), and `apps/studio-web/src/api/types.ts`
(`PixelExplain` component names).

Concurrent engineers: tracer (`trace_scene(experiment, width, height, *, pixels=None, toggles=None) -> HitBuffer`;
`source.py` with line shapes, leakage spectrum, `xenon_relative_spectrum(nm)`, `drifted_powers(...)`), transport
(`get_transport_model(n_photons, seed, cache_dir) -> TransportModel`, `clear_medium_kernel(...)`). Code against
contracts, import them lazily inside `capture()`, unit-test with synthetic HitBuffers + a local clear-medium stub.
If `source.py` isn't ready, use private fallbacks for line shapes / drift. Near the end poll for their files and
run an integration test on the presets; adapt your side if needed.

## Files
`src/spectraforge/studio/spectra.py`, `optics.py`, `sensor.py`, `capture.py` + tests
`tests/studio/test_spectra.py`, `test_optics.py`, `test_sensor.py`, `test_capture.py`.

## spectra.py — tabulated, cited, smooth (≤10 nm tables, interpolate 250–900 nm)
Pure water absorption (Pope & Fry 1997 / Segelstein: tiny in blue, ~0.0006/mm at 700, ~0.002/mm at 800);
haemoglobin oxy/deoxy shapes (Prahl), normalised 1 at 500 nm (`scale` = μa@500); melanin ∝ λ^−3.46 (norm 500);
carbon black ≈ λ^−1; QE curves `ccd_bv`, `ccd_uv`, `scmos`, `flat`; LCTF peak transmission `lctf_vis` (Nuance VIS
class: a few % at 420 rising to ~40–50 % at 650–720) and `flat`; room lights: fluorescent tube (Hg 405/436/546/578 +
tri-phosphor incl. 611), white LED, D65 approx, incandescent (Planck 2856 K); lux → spectral irradiance via CIE V(λ).

## optics.py
- `merged_fluorophores(experiment) -> dict[name, FluorophoreSpec]` (`presets.builtin_fluorophores()` overridden/
  extended by `experiment.fluorophores`).
- `material_optics(material, fluorophores, grid=SPECTRAL_GRID_NM, toggles=None, concentration_scale=None) -> MaterialOptics`:
  μa_k(λ) = ln(10)·ε_k(λ)·c_k[M]/10 per mm, ε_k(λ) = extinction × `EXTINCTION_REFERENCE_M_CM` × `Fluorophore.excitation(λ)`;
  background absorber; μs = μs′/(1−g), μs′(λ) = a(λ/500)^−b; `clear` same rule as tracer; Φ_eff = Φ/(1+Ksv·Σc);
  emission density ∫=1 per nm (`Fluorophore.emission` on the fine grid / grid step). Respect toggles (document
  what "absorption off" means for dye excitation). `average_over_line(arr, line_shape)`.

## sensor.py — the HSI camera
- `band_centres(filter_spec, ex_nm)` with `band_start_policy` (Nuance: start = max(band_start, ex+offset)).
- Band transmission T_b(λ): Gaussian passband FWHM × peak-transmission curve × emission long-pass (smooth edge,
  out-of-band 10^−OD). **Angle blue shift** λ(θ) = λ₀√(1 − sin²θ/n_eff²) per pixel field angle (bin into rings).
- Radiometric chain spectral radiance (P, W) → band photo-electrons: photon conversion λ/(hc), QE(λ), pixel
  area (pitch×binning), image-plane irradiance E_img = π L T_lens / (4N²(1+m)²)·cos⁴θ, exposure. Document and
  unit-test against a hand calculation.
- Optical blur before the detector: thin-lens defocus CoC from per-pixel depth vs focus distance, axial
  chromatic focus shift per band (`axial_chromatic_mm_per_100nm` re 550 nm), diffraction σ ≈ 0.42 λN; spatially
  varying via quantised CoC levels + mask compositing, per band.
- Detector: Poisson shot, dark current × exposure, Gaussian read, PRNU/DSNU fixed patterns deterministic per
  sensor seed, hot pixels, full-well clip, gain e⁻/ADU + offset, ADC quantisation and bit-depth clip; each toggleable.

## capture.py — orchestration
- `capture(experiment, *, mode=None, scale=1.0, excitations=None, progress=None, cancel=None, transport=None, tracer=None) -> CaptureResult`
- `preview(experiment, excitation_nm=None, scale=None) -> CaptureResult`
- `explain_pixel(experiment, x, y, *, scale=1.0) -> dict` — per excitation → component → per-band noise-free
  radiance, plus ground-truth scalars; recompute one pixel exactly via `trace_scene(pixels=[idx])` (no blur; document).
  Component names exactly: `dye:<name>`, `secondary:<name>`, `elastic`, `glint`, `raman`, `leak_source`,
  `background:<material>`, `room_light`.
Per enabled step, in sequence order: drifted power; line shape + leakage; for each HitBuffer layer/material:
E = P·irradiance_per_mW·T_light·Beer(Σ_m μa_m(λ)·light_clear_path_mm) over the line shape; kernels from the
transport model (cos_light, cos_view, thickness_mm, n_outside, path_mm); fluorescence per dye
L_k(λ) = E·⟨μa_k⟩_line·Φ_k·e_k(λ)·G(λ)·b_k(pixel); elastic E·ρ·line(λ); glint glint_per_mW·P·line(λ); leakage
through the same paths; **Raman** for aqueous media (water line at `raman_center_nm(ex)`, FWHM 15–20 nm,
cross-section ~8e-30 cm²/sr/molecule at 488 nm with λ⁻⁴ scaling, through G); **secondary emission**
(documented one-generation approximation: dye-j light reabsorbed by dye k re-emits with Φ_k e_k); **FRET** per
material pairs (donor ×(1−E), sensitised acceptor), toggle; attenuation of deeper layers by the clear layers in
front (Beer on μa(λ) along their `path_mm`) × `T_view`. **Room light** (if enabled): ambient spectral irradiance
→ elastic remission + dye excitation (hemispherical; representative angles). **Subsurface spread**: convolve
each turbid material's fluorescence image with its radial profile scaled by `pixel_footprint_mm`, within the
material's region. **Photobleaching**: per-pixel absorbed photons per molecule (σ_k = ln10·ε/N_A cm² × photon
fluence near the surface, turbid build-up if available, × exposure) → b_k ← b_k·exp(−Φ_bleach,k·dose); order
matters. Then sensor → ADU.
- Modes: `dark` (source off; room light + detector terms remain); `white_reference` (objects replaced by a
  Spectralon plane filling the field of view, illuminated by the source's broadband continuum or a flat white
  source — document the frame keys; the lab records Background.im3 and Whitelight.im3 for normalisation);
  `preview` = one excitation at `preview_scale`.
- Determinism: seed from `acquisition.seed`, mode, step index → bit-identical repeats.
- `GROUND_TRUTH_KEYS` (document): material/object index maps, depth_mm, cos_light, cos_view, path_mm,
  thickness_mm, shadow, `irradiance@<ex>`, `conc:<dye>` (µM at layer 0), `bleached:<dye>` (final survival),
  optional `absorbed_photons:<dye>@<ex>`.
- Warnings: saturated fraction per excitation, beam misses every object, glint into camera, objects outside DOF,
  very low SNR. `progress(fraction, message)`; `cancel()` True → abort.

## Tests
Etendue hand calculation; shot+read noise mean/variance vs theory; saturation; determinism; bleaching order
effect (reversed order differs; with bleaching off identical); **dilute-limit equivalence with legacy
`spectraforge.render`** (flat slab, normal incidence, tiny concentrations: per-dye emission shape and
cross-excitation amplitude ratios match up to one global geometry constant, on noise-free fine-grid radiance —
expose an internal function for it); FRET quench direction; Raman position (488 → ~584 nm); Nuance band-start
(ex 430 → first band 450); off-axis blue-shift sign. Report timings of full-res `mehsi_lab_replica` capture
and preview if the other modules exist.
