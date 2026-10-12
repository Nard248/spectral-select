# SpectralForge Studio — 3D experiment designer & physical HSI simulator

Status: approved in conversation 2026-10-09/10 (sections 1–4). Branch `studio/3d-editor`,
worktree `/Users/narekmeloyan/PycharmProjects/spectral-select-studio`.

## Purpose
Replace the 2D Qt workbench with a 3D experiment designer: place uniform objects on an optical
bench, configure a light source (tunable lamp / LED / laser) and a hyperspectral camera as
attractive 3D hardware, then **Preview** / **Capture** a physically simulated multi-excitation
4D cube (excitation × emission × H × W) from the camera's perspective with perfect ground truth.
Demo-ready for WHISPERS 2026 (Glasgow, Nov 17–19; target Nov 10), then a research instrument.
**Physics comprehensiveness is the top priority of this iteration.**

## Architecture
```
apps/studio-web/      Vite · React 19 · TS · three r186 · @react-three/fiber 9 · drei 10 ·
                      @react-three/postprocessing · zustand
src/spectraforge/studio/
  schema.py      Experiment contract (Pydantic) — DO NOT edit without the lead
  presets.py     builtin fluorophores, materials, 4 experiment presets
  contracts.py   engine-internal data contracts + radiometric definitions — read this first
  tracer.py      camera rays, ray–shape intersection, shadows, Fresnel, layers  → HitBuffer
  source.py      beam profile, line shapes, lamp curve, excitation-filter leakage
  surface.py     Fresnel (unpolarised, TIR), Snell, GGX microfacet BRDF
  geometry.py    analytic primitives (box, sphere, cylinder, cap, dish, cuvette)
  optics.py      MaterialSpec → MaterialOptics (dye μa via ε, absorbers, scattering, quenching)
  spectra.py     tabulated spectra: water, Hb, melanin, carbon; QE curves; LCTF T(λ); room lights
  transport.py   Monte Carlo (Numba) bake + clear closed form → TransportKernels
  sensor.py      lens (DOF, diffraction, chromatic), vignetting, filter bands & angle shift, detector
  capture.py     orchestration → CaptureResult; bleaching, FRET, secondary emission, drift, modes
  exports.py     SpectraData pickle / npz / TIFF / ground-truth folder (lead)
  server.py      FastAPI: presets, validate, preview, capture jobs (WebSocket progress), pixel, export
```
Generated contract for the web: `scripts/export_studio_contract.py` → `apps/studio-web/src/generated/`
(`experiment.schema.json`, `presets.json`, and `npm run gen:types` → `experiment.ts`).

## Conventions
Y-up right-handed world (three.js), table top `y = 0`, mm / nm / deg / mW / ms / µM / 1/mm.
Euler XYZ rotations. Sources & cameras posed by position + target (+ roll). Image row 0 = top.
Molar extinction: `ε(λ) = extinction × 55 000 × excitation_profile(λ)` M⁻¹cm⁻¹ and
`μa_dye(λ) = ln(10) · ε(λ) · c[M] / 10` 1/mm.

## Physics (v1 catalogue)
- **Absorption**: dye spectra (+ line-shape-averaged at excitation), background absorbers
  (water, haemoglobin, melanin, carbon black, custom). Inner filter & reabsorption EMERGE from
  transport (no ad-hoc knobs).
- **Scattering**: μs′(λ) = a(λ/500)^-b, HG anisotropy g; Monte Carlo transport baked per material
  (φ(z)·η(z) depth formulation, angle/thickness tables, cached by optics hash); diffusion-limit and
  MCML-validated; elastic diffuse remission; radial subsurface-spread kernel.
- **Fluorescence**: Φ, emission shape (skew, vibronic), Stern-Volmer self-quenching, FRET (reuses
  `physics.apply_fret` semantics), secondary emission (re-emission of reabsorbed fluorescence),
  water Raman (3400 cm⁻¹) for aqueous media.
- **Photobleaching**: per-pixel photon dose accumulates over the sequence in acquisition order,
  `c_k ← c_k · exp(−Φ_b,k · absorbed photons per molecule)`.
- **Surfaces**: Fresnel/Snell/TIR, GGX glints, one-bounce specular, optional one-bounce diffuse.
- **Source**: tunable lamp (Xe curve × Gaussian monochromator pass), LED (asymmetric), laser line;
  excitation clean-up filter OD (out-of-band leakage); Gaussian / flat-top beam, divergence,
  inverse-square, shadows; step-to-step power drift; room light (fluorescent tube lines etc.).
- **Camera**: thin-lens DOF, diffraction PSF (Airy ≈ Gaussian σ = 0.42 λN), axial chromatic focus
  shift, cos⁴ vignetting, LCTF / interference passbands (Gaussian FWHM, T(λ) curve, angle blue
  shift λ(θ) = λ₀√(1 − sin²θ/n_eff²)), emission long-pass OD, band-start policy (Nuance protocol).
- **Detector**: QE(λ), shot (Poisson), read (Gaussian), dark current, PRNU/DSNU, hot pixels, full
  well + saturation, ADC quantisation + offset, binning; modes sample / dark / white reference.
- **Ground truth**: material/object id, depth, angles, irradiance, path, per-dye concentrations,
  absorbed photons per dye, bleached fraction, noise-free radiance cube, per-component explain.
- Out of scope v1: polarisation, lifetime/time-resolved, spectrograph/push-broom, >1 bounce.

## Verification
MCML slab benchmark (μa=1/mm, μs=9/mm, g=0.75, n=1, d=0.2 mm: Rd≈0.0973, Tt≈0.661 — Wang &
Jacques 1995 / Prahl), diffusion limit, Beer–Lambert, clear closed form ≡ MC as μs→0, dilute
limit ≡ legacy `spectraforge.render` up to a geometry constant, Fresnel/TIR identities, energy
conservation, noise statistics (mean/variance), bleaching dose, determinism (seed ⇒ bit-identical).

## Performance targets (M4 Pro)
Preview (one excitation, ¼ res) < 300 ms after bakes are cached; full lab-replica capture < 30 s;
MC bake < 10 s/material at default quality, cached on disk (`~/.cache/spectralforge/mc`).

## UX
Modes Design → Acquire → Explore. Outliner + library (left), R3F viewport with gizmos, drag-to-aim,
look-through-camera, PiP camera view (center), inspector with live spectral plots (right), sequence
bar with Preview / Capture / Dark / White ref, physics warnings, progress (bottom). Explore: cube
scrubber, false-colour/RGB/log, click-pixel EEM + stacked component decomposition, GT overlays,
ROI, compare, export. Procedural PBR hardware (HSI camera with LCTF stack and rotating lens rings,
Xe tunable lamp + fibre + collimator, LED array, laser), volumetric wavelength-coloured beam (UV
violet shimmer), optical breadboard with M6 grid, studio Lightformers (no network assets).
