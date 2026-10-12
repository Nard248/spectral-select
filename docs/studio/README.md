# SpectralForge Studio

A 3D experiment designer and a physically comprehensive multi-excitation fluorescence hyperspectral simulator, in the browser.

You place samples on a virtual optical bench, set up a light source (tunable xenon lamp, LED array or laser) and a hyperspectral camera (LCTF or filter wheel), and press **Preview** or **Capture**. The engine ray-traces the scene from the camera's point of view and simulates the whole acquisition, producing a 4D cube (excitation × emission × height × width) with the raw detector counts **and perfect ground truth**: which material, how deep, how much of each dye, how much of the signal is fluorescence, scatter, Raman or glint, how much each dye was bleached.

![SpectralForge Studio, Design mode: the ME-HSI lab-replica preset](images/01-design.png)

## Why it exists

* **It replaces the 2D Qt workbench.** SpectraForge's painter GUI (`spectraforge-gui`) paints materials on a flat canvas and applies a handful of effects (inner filter, blur, noise) to a 2D forward model. It cannot represent geometry, illumination, optics or the instrument. Studio simulates the experiment itself: where the lamp and the camera are, what the beam does inside a scattering sample, what the filters pass, what the detector does. The 2D engine (`spectraforge.render`) stays in the repository as the validated reference; Studio agrees with it in the dilute limit (validation figure 4).
* **A test bench for band selection.** The lab's research question is which wavelength bands of a multi-excitation cube carry information. With a simulator that knows the true composition of every pixel and every physical contribution to every band, selection methods can be tested against ground truth instead of against another measurement.
* **Teaching and design.** Every physical effect has a switch. New lab members can see what the inner-filter effect, a scattering sample, an out-of-band lamp leak or a bleaching sequence does to a cube; and a new instrument configuration can be tried before it is built.
* **A demo.** Studio is shown at IEEE WHISPERS 2026 (Glasgow, 17 to 19 November). The guided tour (`/?tour=1`) is built for that.

## 60-second quickstart

You need the project environment (Python 3.11+, the repository's `.venv`) and Node 20+. Once per machine:

```bash
pip install -e ".[studio]"            # FastAPI + uvicorn + pydantic
pip install numba                      # the engine's numerical kernels (not yet listed in the extras, see DEVELOPER_GUIDE)
cd apps/studio-web && npm install && npm run build && cd ../..    # builds the web app into apps/studio-web/dist
```

Then, every time, **one command**:

```bash
python -m spectraforge.studio --open          # or the console script: spectralforge-studio --open
```

(From a git worktree that is not the venv's editable install, prefix with `PYTHONPATH=src`.) The engine starts on `http://127.0.0.1:8765`, serves the web app and opens your browser. For the guided tour open:

```
http://127.0.0.1:8765/?tour=1
```

The very first start spends about a minute in the background preparing numerical kernels and Monte Carlo tables (the app is usable meanwhile, and subsequent starts take seconds). Without the engine the app still opens, in an offline mock mode with synthetic data. Details, development setup and troubleshooting: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

## Feature overview

**Design**
* Seven presets: ME-HSI lab replica (8-line tunable source and a Nuance-class LCTF camera with lichen thalli), generic macro bench, LED multispectral, laser line (90° cuvette row), and three layered-block presets (v1.1): layered wedge phantom, tissue under collagen, pencil-beam penetration.
* **Layered blocks (v1.1)**: a substrate / fluorescent layer / cover stack with uniform or linear-wedge layer thickness, a layer-stack editor, a cutaway with the 3D light field painted on the cut plane.
* Lab-ware library: petri dish, cuvette, slide, droplet, bead, block, puck; drag-and-drop, gizmos, snapping, drag-to-aim, look-through-camera, picture-in-picture camera view, frustum, focus plane and depth-of-field visualisation.
* Materials as optical media (dyes in µM, refractive index, scattering power law, background absorbers, roughness, quenching, FRET, Raman); 17 builtin materials and 14 builtin fluorophores.
* Procedural PBR hardware: xenon tunable lamp with fibre and collimator, LED array, laser, HSI camera with LCTF stack; volumetric, wavelength-coloured beams.

**Simulate** (see [HOW_IT_WORKS.md](HOW_IT_WORKS.md))
* Exact ray tracing of analytic solids with refraction, Fresnel and total internal reflection, shadows, GGX glints, clear-media chains (glass, liquid, glass) and side-lit volumes (laser through a cuvette).
* Absorption, inner-filter and re-absorption effects that emerge from transport; scattering by Monte Carlo transport (validated against MCML and the diffusion limit); quenching, FRET, secondary emission, water Raman, elastic scatter, excitation-filter leakage, room light.
* Photobleaching accumulated over the excitation sequence (order matters).
* Instrument model: monochromator or LED or laser line shapes, beam profile and divergence, LCTF and interference-filter bands with angle-dependent blue shift, thin-lens depth of field, diffraction, chromatic aberration, vignetting, quantum efficiency, shot, read and dark noise, fixed-pattern noise, full well and ADC.
* Acquisition modes: sample, dark, white reference. Deterministic given a seed.
* 26 physics toggles for ablation and teaching.

**Simulate, v1.1**
* 3D Monte Carlo of layered blocks (MCML on tilted planar interfaces, peel-off estimator, Fresnel bottom boundary, layer-conforming voxel grid): scattering and absorption of a cover acting on a buried fluorophore, with the 3D fluence, absorbed power and fluorescence source as outputs.

**Explore**
* Band scrubbing, band / true colour / false colour views, ground-truth overlays, ROI spectra, side-by-side compare.
* Click any pixel for a decomposition of its spectrum into physical components, its excitation-emission matrix and its ground truth.
* **Analysis workspace (v1.1)**: ROI tools (point, disk, rectangle, ellipse, polygon, line), EEM of any ROI (overlay, A-B, A/B), line profiles, response curves (signal versus a ground-truth map such as cover thickness), seven normalisation modes including *irradiance-normalised*, the 3D light field viewer, and parameter sweeps (for example cover thickness 0 to 2 mm). Offline use: `analysis.load_npz_capture` (USER_GUIDE section 14).
* Exports: `SpectraData` pickle (loads with the existing pipeline unchanged), npz, 16-bit TIFF stacks, ground-truth arrays, the experiment JSON.

## Documents

| Document | For | Contents |
|---|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | everyone | UI walkthrough: layout, Design → Acquire → Explore, building an experiment, physics toggles, the Explorer, exports and loading them in `spectral_select`, shortcuts, the tour, demo tips; v1.1: layered blocks, the analysis workspace, the notebook API |
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | scientists, reviewers | the simulation end to end with the equations as implemented, units and conventions, approximations and limits; section 14: the layered-block 3D Monte Carlo |
| [PHYSICS_VALIDATION.md](PHYSICS_VALIDATION.md) | scientists, reviewers | evidence: 17 figures and numbers (section 12: the 3D layered-block engine) (MCML, diffusion limit, closed form versus quadrature, inner filter, reabsorption, bleaching, photon-transfer curve, depth of field, filter angle shift), regenerated by a script |
| [ARCHITECTURE.md](ARCHITECTURE.md) | developers | components, HTTP/WebSocket API, jobs and warm-up, Python module map, web app map, contract-first design, caching, determinism, tests |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | developers | setup, dev workflow, tests, contract regeneration, recipes (fluorophore, material, preset, shape, physics toggle, endpoint), troubleshooting |
| [`docs/onboarding/components/07-spectralforge-studio.md`](../onboarding/components/07-spectralforge-studio.md) | new lab members | one-page component guide pointing here |
| [`apps/studio-web/README.md`](../../apps/studio-web/README.md) | web developers | the web app's own short README |
| [`docs/superpowers/specs/2026-10-10-spectralforge-studio-design.md`](../superpowers/specs/2026-10-10-spectralforge-studio-design.md) | everyone | the approved design and scope |

## Where things are

```
src/spectraforge/studio/     the engine (schema, tracer, transport, sensor, capture, server, exports, ...)
apps/studio-web/             the web app (Vite, React 19, TypeScript, three.js / react-three-fiber)
scripts/export_studio_contract.py      schema + presets -> web app (JSON schema, TypeScript types)
scripts/studio_validation_figures.py   regenerates the validation figures
tests/studio/                Python tests (slow ones marked `slow`); the web app has its own vitest suite
docs/studio/                 this documentation, images/, figures/ (validation)
```
