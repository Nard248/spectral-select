# SpectralForge Studio: 3D experiment designer and physical HSI simulator

## Purpose and status

Studio is the successor of the 2D painter GUI of [SpectraForge](01-spectraforge.md). Instead of painting materials on a flat canvas and applying a few effects to a 2D forward model, you build a virtual **optical bench** in the browser (samples, a tunable lamp / LED array / laser, an LCTF or filter-wheel camera), and a Python engine simulates the whole multi-excitation acquisition physically: ray tracing from the camera, Monte Carlo light transport in scattering samples, fluorescence with inner filter, FRET, secondary emission and Raman, photobleaching over the excitation sequence, and a detector model. The output is a 4D cube (excitation × emission × H × W) in the lab's usual data format **plus perfect ground truth** (material, depth, per-dye concentration, and a per-pixel split of the signal into fluorescence, scatter, Raman, glint, ...).

Status in October 2026: built on branch `studio/3d-editor` for the IEEE WHISPERS 2026 demo (17 to 19 November), then to become a research instrument. The engine has 203 tests (11 slow) and is validated against MCML, the diffusion limit and an independent closed-form implementation (see `docs/studio/PHYSICS_VALIDATION.md`). The 2D engine (`spectraforge.render`) and its GUI remain in the repository; Studio agrees with the 2D engine in the dilute limit.

**The full documentation lives in [`docs/studio/`](../../studio/README.md)**; this page is the map.

## Where to read what

| You want to | Read |
|---|---|
| run it and click around | [`docs/studio/README.md`](../../studio/README.md) (60-second quickstart) and [`USER_GUIDE.md`](../../studio/USER_GUIDE.md) |
| understand the physics, equations and limits | [`HOW_IT_WORKS.md`](../../studio/HOW_IT_WORKS.md), evidence in [`PHYSICS_VALIDATION.md`](../../studio/PHYSICS_VALIDATION.md) |
| change the code | [`ARCHITECTURE.md`](../../studio/ARCHITECTURE.md) then [`DEVELOPER_GUIDE.md`](../../studio/DEVELOPER_GUIDE.md) (setup, tests, recipes, troubleshooting) |
| use a simulated cube in the existing pipeline | the export section of the user guide: `SpectraData.from_pickle("<capture>.pkl")` |

## Files

```
src/spectraforge/studio/
├── schema.py        the Experiment contract (Pydantic): fluorophores, materials, objects, sources, cameras, physics toggles
├── presets.py       builtin fluorophores, 17 materials, 4 experiment presets
├── contracts.py     engine-internal data (HitBuffer, TransportKernels, CaptureResult) + the radiometric definitions
├── geometry*.py, tracer.py, surface.py     analytic solids, camera rays, layers, shadow rays, Fresnel, GGX
├── source.py, optics.py, spectra.py        beam and line shapes, dye / medium optics, tabulated spectra
├── transport.py, transport_mc.py           closed-form clear media and baked white Monte Carlo for turbid media
├── sensor.py        bands, angle shift, étendue, blur, detector noise chain, ADC
├── capture.py       orchestration: capture(), preview(), explain_pixel()
└── server.py, jobs.py, exports.py, render_png.py     FastAPI service, job queue, SpectraData / npz / TIFF exports
apps/studio-web/     the web app (Vite, React 19, TypeScript, three.js / react-three-fiber, zustand)
scripts/export_studio_contract.py      schema + presets -> apps/studio-web/src/generated/
tests/studio/        Python tests;  apps/studio-web/src/**/*.test.ts  vitest tests
```

## Run it

```bash
pip install -e ".[studio]" numba
(cd apps/studio-web && npm install && npm run build)          # once
python -m spectraforge.studio --open                          # http://127.0.0.1:8765 ; tour: /?tour=1
```

The first start prepares numerical kernels and Monte Carlo tables for about a minute (cached afterwards in `~/.cache/spectralforge/mc`).

## Use a capture from Python

```python
from spectral_select import SpectraData
sd = SpectraData.from_pickle("capture.pkl")      # exported from Explore > Export > spectradata pkl
cube = sd.get_excitation(365.0).cube              # (H, W, bands), ADC offset removed
experiment_json = sd.metadata["experiment_json"]  # the exact experiment that produced it
```

Headless (no browser): `from spectraforge.studio.capture import capture`, `from spectraforge.studio.schema import Experiment`; load a saved `.sfx.json` with `Experiment.model_validate_json(...)` and call `capture(exp)` (details in the user guide).

## Known issues and gaps (found while writing the docs)

* `pyproject.toml`'s `studio` extra lacks `numba` (required by the engine) and the test-only packages `hypothesis` and `httpx`; install them explicitly until the extra is fixed.
* The physics toggle `indirect_bounce` exists in the schema and the UI but is not implemented in the engine; the inspector shows it disabled (not yet implemented, v1).
* Spectra are parametric Gaussians (no S0→S2 bands, no solvent shifts), FRET has no spectral-overlap integral, and the other simplifications listed in `HOW_IT_WORKS.md` section 13: read it before quoting any simulated number as quantitative.
* The web app's README says `scripts/check-framing.mjs` must report under 1 px; measured worst error was 1.11 px.
