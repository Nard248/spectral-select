# SpectraForge — Synthetic ME-HSI Generator (v1 engine) — Design Spec

- **Date:** 2026-06-19
- **Author:** Narek Meloyan
- **Status:** Approved design — pending spec review → implementation plan
- **Package:** `src/spectraforge/` (new package in the spectral-select monorepo)

## 1. Goal

A pure-Python forward-model engine that turns user-defined **fluorophores → materials → a painted
spatial scene** into a physically-grounded **multi-excitation hyperspectral (ME-HSI)** cube **plus
the ground truth** needed to validate unsupervised wavelength/band-selection methods. Renders
directly to spectral-select's `SpectraData` so synthetic datasets feed the existing `Analyzer` and
GUI with zero glue. A painter GUI is the next milestone, built on this engine.

Non-goal (v1): the GUI; nonlinear/high-concentration physics; live online databases.

## 2. Decisions (locked with owner)

| # | Decision | Choice |
|---|---|---|
| 1 | Build order | **Engine first** (library core, fully tested); GUI is the next milestone |
| 2 | Location | **`src/spectraforge/`** in this monorepo (binds to `spectral_select.types`) |
| 3 | v1 physics | **Lean linear model + key artifacts** (scatter lines, noise, exposure/power) |
| 4 | Name | **SpectraForge** |

## 3. The physics (what we implement)

Dilute-regime fluorescence is a **linear superposition of fluorophore excitation–emission matrices**.
For excitation wavelength λex, emission wavelength λem, pixel (x,y):

```
clean[x,y,λem | λex] = Σ_k  c_k(x,y) · ε_k · Φ_k · exc_k(λex) · em_k(λem)
measured             = clean · g(λex) · exposure(λex) · power(λex)  + scatter + noise
```

- `exc_k(λ)` — excitation profile, Gaussian(peak=`ex_peak_k`, FWHM=`ex_fwhm_k`), **normalized to peak 1**.
- `em_k(λ)` — emission profile, Gaussian(peak=`em_peak_k`, FWHM=`em_fwhm_k`), **normalized to unit area** on the emission grid.
- `c_k(x,y)` — concentration map for fluorophore k (from the painted scene).
- `ε_k` — relative extinction (unitless in v1), `Φ_k` — quantum yield. Amplitude lives in `c·ε·Φ`.
- `g(λex)` — relative lamp/laser intensity per excitation; `exposure`, `power` — per-excitation scalars.
  Signal **includes** exposure×power scaling so it mimics raw instrument output that spectral-select's
  Step 3 normalization divides back out. These scalars are stored in `ExcitationData`.

**Key invariant (correctness backbone):** for the **clean** render (artifacts off — no scatter, no
noise), the model is exactly linear in concentration, so
`render(sceneA + sceneB) == render(sceneA) + render(sceneB)`. (Additive scatter is
concentration-independent, so it intentionally breaks this; the invariant is tested on the clean
path.) This is the primary property test.

Sigma from FWHM: `σ = FWHM / 2.3548`.

## 4. Artifacts (v1)

- **Rayleigh line:** additive Gaussian bump centered at `λem = λex` (width `rayleigh_fwhm`, amplitude
  `rayleigh_strength·g·exposure·power`), scaled by a per-pixel reflectance map (default 1).
- **2nd-order line:** same bump at `λem = 2·λex` (only if inside the emission grid).
- **Noise (seedable):** Poisson shot noise (`poisson(signal·photon_scale)/photon_scale`) + additive
  Gaussian read noise `N(0, read_sigma)`. A single `numpy.random.Generator(seed)` drives all noise →
  fully reproducible.

These are exactly what spectral-select removes (Step 5 scatter lines) / normalizes (Step 3) — so
synthetic data exercises the whole pipeline.

## 5. Domain model & module layout

```
src/spectraforge/
  __init__.py        public API (Fluorophore, Material, Scene, AcquisitionConfig, render, ...)
  fluorophore.py     @dataclass Fluorophore + excitation(λ)/emission(λ) profile methods (vectorized)
  library.py         load_builtin_library() from data/fluorophores.json; add/save user fluorophores
  material.py        @dataclass Material = list[(fluorophore_name, concentration)]; the "brush"
  scene.py           Scene: H×W canvas; paint_rect/paint_circle/paint_polygon(material, conc, layer);
                     resolve() -> {fluorophore_name: concentration_map(H,W)}  (additive mixing)
  acquisition.py     @dataclass AcquisitionConfig: excitations[], emission grid (min,max,step),
                     per-excitation lamp g/exposure/power, optional detector QE
  artifacts.py       ArtifactConfig + add_scatter_lines(...), add_noise(..., rng)
  forward.py         render(scene, library, acquisition, artifacts=None, seed=None)
                       -> (SpectraData, GroundTruth)
  groundtruth.py     @dataclass GroundTruth: concentration_maps, material_label_map, class_defs,
                     clean cube(s); save(dir) sidecar (.npz + .json); informative_bands(threshold)
  data/fluorophores.json   starter library
```

Each module has one responsibility, no GUI, no I/O beyond `library`/`groundtruth`. Boundaries:
`forward.render` is the only place spectra are assembled; `scene.resolve` is the only place shapes
become concentration maps; `fluorophore` is the only place spectra shapes are defined.

## 6. Output & binding

- `render(...) -> (SpectraData, GroundTruth)`.
- `SpectraData.to_pickle(path)` — loads in `spectral_select.Analyzer` / `mehsi_preprocessor`.
- `GroundTruth.save(dir)` — sidecar: `groundtruth.npz` (per-fluorophore concentration maps, dominant
  material label map, clean cube per excitation) + `groundtruth.json` (fluorophores, materials,
  acquisition, seed). Optional `class_mask.png` (dominant material) for the validation/annotation flow.
- This makes the validation loop concrete: render → `Analyzer.select()` → compare selected bands to
  `GroundTruth.informative_bands()`.

## 7. Starter fluorophore library (~12)

Parametric entries (approx ex/em peaks, nm), spanning autofluorescence + common dyes, overlapping
the project's real biology: tryptophan 280/350, collagen 330/390, elastin 350/420, NADH 340/460,
FAD 450/535, DAPI 358/461, EGFP 488/507, fluorescein/FITC 495/519, rhodamine 540/565, Texas Red
595/615, mCherry 587/610, Cy5 649/670. Each: ex/em FWHM (~40–80 nm), Φ (0.1–0.9), relative ε.
Stored in `data/fluorophores.json`; users can add their own parametrically.

## 8. Testing (TDD)

- `fluorophore`: excitation peak=1 at `ex_peak`; emission unit-area; σ from FWHM correct.
- `material`/`scene`: painted shapes produce expected concentration maps; layering is additive;
  `resolve()` returns one map per fluorophore.
- `forward` (core): single fluorophore @ one excitation → emission Gaussian centered at `em_peak`,
  amplitude `∝ c·ε·Φ·exc(λex)`; **linearity invariant** `render(A+B)==render(A)+render(B)` (clean path, artifacts off);
  exposure/power scaling applied & recorded; emission-grid length == cube bands.
- `artifacts`: scatter bump lands at `λem≈λex` (and `2λex`); noise is seed-deterministic, raises
  variance, preserves mean approximately.
- `groundtruth`: concentration maps match scene; sidecar save/load round-trips.
- **binding**: render a 2-fluorophore scene → `SpectraData.to_pickle`/`from_pickle` round-trip →
  `Analyzer.prepare()/select()` runs and selects bands near the planted fluorophores' emission peaks.

## 9. Phasing (engine v1 — each phase ships working, tested code)

1. **Fluorophore + library** — dataclass, profiles, starter JSON.
2. **Material** — recipe + mixing.
3. **Scene** — shapes/layers → concentration maps.
4. **Acquisition + forward (clean)** — `render()` → `SpectraData` (no artifacts).
5. **Artifacts** — scatter lines + seedable noise.
6. **GroundTruth + binding** — sidecar export; end-to-end test feeding `Analyzer`.
7. **Demo** — `spectraforge-demo` console script that generates a ready dataset + ground truth.

## 10. Packaging

Register `spectraforge*` in `pyproject.toml` `[tool.setuptools.packages.find].include`; add a
`spectraforge-demo` console script. Extend CI `--cov` to include `spectraforge`. `data/*.json` shipped
as package data.

## 11. Risks & mitigations

- **Physics bug** → the linearity invariant + single-fluorophore analytic checks catch most.
- **Unit ambiguity (ε molar vs relative)** → v1 uses relative ε, documented; realistic-units mode later.
- **Scope creep** → strict v1 boundary (no inner-filter/PSF/Raman/FRET/GUI); those are §12.
- **Binding drift** → the end-to-end `Analyzer` test guards the `SpectraData` contract.

## 12. Out of scope (future milestones)

Painter **GUI** (next milestone, on `Scene`/`render`); Beer-Lambert inner-filter nonlinearity; PSF
blur; water Raman; FRET; bi-Gaussian/imported measured spectra; live FPbase/PhotochemCAD API; FLIM;
in-app endmember-recovery (N-FINDR/VCA) scoring.
