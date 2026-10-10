# SpectralForge Studio: developer guide

How to set up, run, test and extend Studio. Read [ARCHITECTURE.md](ARCHITECTURE.md) first for the map; the physics is in [HOW_IT_WORKS.md](HOW_IT_WORKS.md). Every command in this guide was run while writing it (macOS, Python 3.11, Node 22, npm 11); where something did not behave as hoped it says so.

## Contents

1. [Setup](#1-setup)
2. [Running in development](#2-running-in-development)
3. [Running the tests](#3-running-the-tests)
4. [Regenerating the contract](#4-regenerating-the-contract)
5. [Recipes](#5-recipes)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Setup

You need Python 3.11+, Node 20+ (22 was used) and npm. The repository-wide environment setup (venv, Java for `.im3` files, GPU) is in [`docs/onboarding/SETUP.md`](../onboarding/SETUP.md); Studio needs only the Python and Node parts below.

```bash
# from the repository root, inside the project venv
pip install -e ".[studio]"          # fastapi, uvicorn[standard], pydantic
pip install numba hypothesis httpx   # see the note below
cd apps/studio-web && npm install    # web app dependencies (three, R3F, zustand, vitest, Playwright, ...)
```

> **Dependency note (as of this branch).** The engine imports `numba` (all geometry, Monte Carlo and tracer kernels), and the test suite uses `hypothesis` and, through FastAPI's `TestClient`, `httpx`. At the time of writing `pyproject.toml` lists none of the three: the `studio` extra contains only `fastapi`, `uvicorn[standard]` and `pydantic`. The lab's `.venv` already has them (numba 0.63, hypothesis 6.168, httpx 0.28), so nothing is missing there, but on a fresh machine install them explicitly as shown, or add them to the extras.

Optional, for the screenshot and smoke scripts: Playwright's browser, `cd apps/studio-web && npx playwright install chromium`.

**Running from a git worktree.** The venv's editable install points at *one* checkout. If you work in a different worktree (as the Studio branch did), put that checkout's `src/` first: `PYTHONPATH=src python ...` from the worktree root. Every Python command below uses that form; with a plain `pip install -e .` in your own checkout you can drop it and use the `spectralforge-studio` console script instead of `python -m spectraforge.studio`.

## 2. Running in development

Two processes: the engine (Python, port 8765) and the Vite dev server (port 5173, hot reload, proxies `/api` and the job WebSocket to the engine).

```bash
# terminal 1 — engine
PYTHONPATH=src python -m spectraforge.studio              # http://127.0.0.1:8765
# options: --host 127.0.0.1 --port 8765 --open   (--open launches the browser)

# terminal 2 — web app with hot reload
cd apps/studio-web && npm run dev                          # http://localhost:5173
```

Open `http://localhost:5173/`. The top bar shows a green "engine online" badge; if the engine is not running the app falls back to the in-browser mock engine and shows "engine offline — mock data" (click it to retry). Add `?tour=1` to the URL for the guided tour.

**Without the dev server** (what a user runs): build once and let the engine serve the app.

```bash
cd apps/studio-web && npm install && npm run build         # writes apps/studio-web/dist (git-ignored)
cd ../.. && PYTHONPATH=src python -m spectraforge.studio --open     # one process, one port
```

The first start after an empty cache spends about a minute in the background warm-up (numba compilation and Monte Carlo table bakes, see [Troubleshooting](#cold-start-is-slow)); `curl localhost:8765/api/health` shows its progress in the `warmup` field. The app is usable meanwhile.

Other useful commands:

```bash
npm run lint                  # eslint (0 errors, a few react-refresh warnings at the time of writing)
npm run build                 # tsc -b && vite build  (type-check + production bundle)
node scripts/smoke.mjs        # Playwright UI smoke test, needs the dev server (and optionally the engine) -> "SMOKE OK"
node scripts/check-framing.mjs   # engine image vs 3D projection, needs the engine on 8765 (see below)
```

## 3. Running the tests

### Python

```bash
# everything except the long Monte Carlo benchmarks
PYTHONPATH=src python -m pytest tests/studio -q -o addopts="" -m "not slow"
# only the slow ones (MCML benchmark, diffusion limit, reciprocity vs forward MC, cache/determinism)
PYTHONPATH=src python -m pytest tests/studio -q -o addopts="" -m slow
```

`-o addopts=""` neutralises any `addopts` of the repository's pytest configuration. Measured here: 203 tests, 11 of them `slow`; the fast run gives `191 passed, 1 xfailed` in about 30 s, the slow run `11 passed` in about 2 minutes. The one `xfail` (`tests/studio/test_qa_findings.py`) documents a known, optional tracer enhancement (see HOW_IT_WORKS §13).

Layout: one test module per engine module (`test_geometry.py`, `test_surface.py`, `test_source.py`, `test_optics.py`, `test_spectra.py`, `test_tracer.py`, `test_transport.py`, `test_sensor.py`, `test_capture.py`, `test_server.py`, `test_exports.py`) plus `test_qa_findings.py` for minimal reproductions of past bugs. Geometry and surface tests use `hypothesis`. The server tests inject fake engine callables into `create_app(...)`, so they run in seconds.

### Web

```bash
cd apps/studio-web
npm test                      # vitest run: 6 files, 40 tests (store, spectral maths, progress parser, mock engine, explorer rendering)
```

### Playwright scripts (need the dev server; most need the engine)

| Script | Output | Notes |
|---|---|---|
| `node scripts/smoke.mjs [url]` | prints `SMOKE OK` | exercises library, typing, physics toggles, capture, explore, back; fails on console errors |
| `node scripts/screenshots.mjs [url]` | `screenshots/01-design.png` ... `04-explore.png` | overwrites the reference screenshots; 1440×900 |
| `node scripts/tour-shots.mjs [url] [WxH]` | `screenshots/tour/step-N.png` (or `tour-WxH/`) | walks all 8 tour steps; overwrites |
| `node scripts/shoot-three.mjs [view ...]` | `screenshots/three/<view>.png` | screenshots of `showcase.html` (standalone 3D models); expects a dev server on port **5199** |
| `node scripts/check-framing.mjs [preset]` | per-object pixel errors | compares the engine's object centroids with the browser's projection; the engine must be on 8765 |

Notes from running them: `smoke.mjs` and `check-framing.mjs` pass. `check-framing.mjs` printed `worst error 1.11 px` for the four presets, while the README of the web app says it "must report < 1 px"; the 3D-versus-engine agreement is good but that claim is slightly optimistic. The screenshot scripts were deliberately **not** re-run while writing this guide because they overwrite the committed reference images; they use `--use-angle=swiftshader` (software WebGL), so they work headless without a GPU but a full capture screenshot takes tens of seconds.

## 4. Regenerating the contract

The browser's TypeScript types and the bundled presets are generated from the Python models. After editing `schema.py` **or** `presets.py` (materials, fluorophores, presets):

```bash
PYTHONPATH=src python scripts/export_studio_contract.py && (cd apps/studio-web && npm run gen:types)
```

This rewrites `apps/studio-web/src/generated/{experiment.schema.json, presets.json, experiment.ts}`. Never edit them by hand. Then `npm run build` (the type-check shows what the schema change broke in the UI) and `npm test`. On an unchanged tree the command reproduces the committed files exactly (checked), so a diff in `generated/` after you touched only the UI means something is wrong. `tests/studio/test_server.py::test_presets_json_in_sync` fails if `presets.json` and `GET /api/presets` disagree, which catches a forgotten regeneration.

## 5. Recipes

### 5.1 Add a fluorophore

A fluorophore is a `FluorophoreSpec` (`schema.py`): excitation and emission peak and FWHM, quantum yield, relative `extinction` (1.0 = 55 000 M⁻¹cm⁻¹), `em_skew`, `vibronic`, `bleach_quantum_yield`, `display_color`.

*For one experiment only* (no code): add it to the experiment's `fluorophores` list in the `.sfx.json` (it adds to, or overrides by name, the builtin library; the web UI does not yet have an editor for this list), then reference its name in a material's `dyes`.

*As a builtin* (appears in the **Dyes** tab and in every export):

1. In `src/spectraforge/studio/presets.py` add a `FluorophoreSpec(...)` to `_EXTRA_FLUOROPHORES` (as `optical_brightener` and `chlorophyll_a` are). Dyes that belong to the 2-D library live in `src/spectraforge/data/fluorophores.json` instead; entries there are picked up automatically by `builtin_fluorophores()`.
2. Optionally add a colour to `_DYE_COLORS`, a bleaching quantum yield to `_BLEACH` (default 1e-6) and, if you know the literature value, the peak extinction in M⁻¹cm⁻¹ to `_EXTINCTION_MCM`. Spectra are Gaussians by construction; say in the commit message what they were fitted to.
3. Regenerate the contract (section 4) so `presets.json` includes it.
4. Add or extend a test (for example in `tests/studio/test_optics.py`: μa at the peak equals `ln10 · ε · c / 10`).

### 5.2 Add a material

Materials are `MaterialSpec` entries built by `_mat(id, name, color, dyes=[(fluorophore, µM), ...], **kwargs)` in `builtin_materials()` (`presets.py`). Fields you will usually set: `refractive_index`, `scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=..., power_b=..., g=...)` (omit for a clear medium), `background_absorption=AbsorberSpec(preset="water"|"hemoglobin"|"melanin"|"carbon_black"|"custom", scale=...)`, `surface_roughness`, `aqueous=True` (for the water Raman line), `quenching_ksv_per_uM`, `fret=[{...}]`. Regenerate the contract; the material appears under **Library > Materials** automatically. Remember: even surfaces are volumes (white paper = scattering volume + brightener). If the material is turbid, its `(g, n, n_outside)` combination needs a Monte Carlo table: the first capture bakes it (several seconds) and caches it; if the material belongs to a builtin preset, the warm-up bakes it at start.

### 5.3 Add a preset

1. Write a function in `presets.py` returning an `Experiment` (see `generic_macro_bench()` and `laser_line()`; `nuance_camera()` and `tunable_lamp()` are reusable builders). Sequence powers and exposures should be chosen so the brightest sample reaches roughly 20–30 k electrons (full well 60 k) without saturation.
2. Register it in the `PRESETS` dict.
3. Add `{ key, label, blurb }` to `PRESET_ORDER` in `apps/studio-web/src/state/presets.ts` (the Presets menu).
4. Regenerate the contract. The warm-up reads `PRESETS` (`transport.builtin_bake_plan`), so its Monte Carlo tables are prepared at engine start.
5. Run `node scripts/check-framing.mjs <preset>` to confirm the 3D view and the engine image agree, and run the preset through `capture()` once to look at the warnings.

### 5.4 Add a shape

Example: a new primitive "wedge".

1. **Schema** (`schema.py`): add a `WedgeShape(_Model)` with `kind: Literal["wedge"] = "wedge"` and its parameters; add it to the `Shape` union. Document the local frame (origin, orientation) in the docstring as the other shapes do.
2. **Geometry** (`geometry.py`): add a branch in `build_scene()` that builds the solid from existing primitives (`Box`, `Sphere`, `Cylinder`, `HalfSpace`, combined with `Intersection` or `Difference`). If a new object is a CSG of those, **no numba code is needed**: `geometry_nb.compile_solids()` flattens solids of up to two leaf primitives in single, intersection or difference mode, and silently falls back to the (slower) NumPy path for anything else.
3. **A genuinely new primitive** needs a `Primitive` subclass in `geometry.py` (`interval`, `contains`, `bound`), a numba port in `geometry_nb.py` (a new `PT_*` code, a case in `_prim` and `_prim_contains`, and a branch in `_flat_prim`), and a parity test against the NumPy reference (the existing tests check agreement to 1e-10).
4. **Tests**: add chord/normal tests like those in `tests/studio/test_geometry.py` (a few rays with known intersections; a hypothesis property that rays inside report exit points on the surface).
5. **Contract**: regenerate (section 4). `experiment.ts` now has the new type in the `Shape` union, and `tsc` will point at every place that switches on `shape.kind`.
6. **Web**: add the shape to `LIBRARY_SHAPES` in `state/library.ts` (the factory), to the `KIND` map in `panels/Library.tsx`, to `KIND_LABEL`, the defaults and the geometry fields in `panels/inspector/ObjectInspector.tsx`, an icon in `ui/icons.tsx`, a mesh in `three/SampleMesh.tsx` (a `case` in the switch plus the thickness estimate near its end), and a branch in `api/mockScene.ts` so the offline mock draws it.
7. Check "what you see is what you capture" with `check-framing.mjs`: place the new object in a preset and compare the 3D and engine images. Shapes with tall vessels are skipped by that script (it compares object tops).

### 5.5 Add a physics toggle

Example: `lens_distortion`.

1. `PhysicsToggles` in `schema.py`: add `lens_distortion: bool = True` (or `False` for costly or experimental effects, as `indirect_bounce` is). Keep it a plain bool.
2. Implement the effect where it belongs (`optics.py`, `sensor.py`, `capture.py`, `tracer.py`) and gate it with `tg.lens_distortion`, where `tg` is `experiment.acquisition.physics` (the tracer receives `toggles` explicitly). Look at how an existing toggle is gated, for example `tg.vignetting` in `sensor.vignetting`.
3. Regenerate the contract.
4. Web: add `['lens_distortion', 'Lens distortion', 'one-sentence tooltip']` to the right group in `GROUPS` in `panels/inspector/SceneInspectors.tsx`. If the new toggle defaults to **off**, also exclude it in the *Reset* button of the same file, which turns every toggle on except `indirect_bounce`.
5. Tests: an ablation test (effect on versus off changes the result in the expected direction and off reproduces the previous behaviour) in `tests/studio/test_capture.py` or the module's own test file.
6. Document it in HOW_IT_WORKS.md.

> **Known gap.** `indirect_bounce` is declared in the schema and shown in the UI, but nothing in the engine reads it: switching it on changes nothing. Implementing it is a natural first physics contribution.

### 5.6 Add an API endpoint

1. In `server.py`, add the route inside `create_app()` (that is where the capture store, the job manager and the injected callables live). If it needs new engine functionality, add an injectable callable parameter like `explain_fn` and a lazy default adapter (`_default_*`), so the tests can pass a fake.
2. Describe it in the header comment of `apps/studio-web/src/api/types.ts` and add request/response interfaces; add the method to the `Engine` interface and `realEngine` in `api/client.ts`, **and** implement it in `api/mock.ts`, otherwise the offline mode breaks.
3. Test it in `tests/studio/test_server.py` with a fake engine (see `fake_result()` and the `client` fixture).
4. Update the endpoint table in [ARCHITECTURE.md](ARCHITECTURE.md#2-http-and-websocket-api).

## 6. Troubleshooting

### Cold start is slow

Symptoms: the first preview or capture after installing takes a minute or more, `/api/health` shows `warmup.state: "running"`. Two one-time costs: (1) numba compiles the kernels (about 20 s; cached next to the sources in `__pycache__/*.nbi,*.nbc`, then ~1 s), (2) the Monte Carlo tables for each `(g, n, n_out)` of the builtin presets are baked and written to `~/.cache/spectralforge/mc` (7 combinations, ~52 s measured; ~270 MB). Measured total: ~60 s from an empty cache, ~4 s with warm caches. To pre-pay it deliberately, start the engine once and wait for `curl localhost:8765/api/health` to report `"state":"done"`. To bake into a different place use `SPECTRALFORGE_MC_CACHE=/path`. Changing `mc_photons` or the seed changes the cache key and triggers a new bake.

### The "engine offline — mock data" badge

The browser could not reach `/api/health` within 1.2 s. Check that the engine is running (`curl localhost:8765/api/health`); that you are using the Vite dev server on 5173 (which proxies `/api`) or the engine's own URL (not `file://`); and that nothing else listens on 8765. Click the badge to retry. Data produced in this state is synthetic; exports are not available in mock mode.

### Port conflicts

The engine defaults to 8765 and the Vite proxy is hard-wired to it (`apps/studio-web/vite.config.ts`). If 8765 is busy (`lsof -i :8765`) you can start the engine elsewhere, `python -m spectraforge.studio --port 8799`, and open it directly at `http://127.0.0.1:8799/` (it serves the built app if `dist/` exists). For the dev server, change the proxy target in `vite.config.ts`. Note that `check-framing.mjs` also assumes 8765. If the browser shows an older UI than your code, you are probably looking at a stale `dist/` served by the engine: rebuild with `npm run build`.

### numba cache problems

Odd `numba` errors after pulling changes, or a crash in a cached kernel: delete the cache and recompile (about 20 s once): `find src/spectraforge/studio -name "*.nbi" -delete -o -name "*.nbc" -delete`. Set `NUMBA_CACHE_DIR` to put the cache somewhere writable if the source tree is read-only. If the machine is shared and slow, `NUMBA_NUM_THREADS` limits the parallel kernels. The MC table cache is separate; delete files in `~/.cache/spectralforge/mc` to force rebakes (a corrupt file is also detected and rebaked automatically).

### Headless WebGL (CI, containers, screenshots)

The 3D view needs WebGL. Playwright's chromium is launched with `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist` in the repository's scripts (software rendering, no GPU needed). The app has a *Render quality* toggle (the gauge button in the viewport toolbar, or `window.studio.getState().setUi({ quality: 'low' })`) that turns off ambient occlusion and anti-aliasing: use it on slow or software GPUs and in `smoke.mjs`. A blank viewport in a normal browser usually means WebGL is disabled or blocklisted (check `chrome://gpu`). `window.studio` (the zustand store) and `window.sfTour` are exposed **in dev builds only** (`npm run dev`); this is why the Playwright scripts need the dev server rather than the engine-served `dist/`.

### A capture "does nothing"

Check the sequence bar: a source with no *enabled* steps gives the warning "The excitation sequence has no enabled steps" and `capture()` raises. Open the warning list (click the warning count at the bottom left): the validation runs on every edit and reports excitation inside the camera's band range, beams that miss every object, objects outside the depth of field, and specular glints before you spend time on a capture. Failed jobs show the exception as `ExceptionType: message` in the toast and in `GET /api/jobs/{id}`.

### One possible transient

While verifying the export snippet for the user guide, one download of an 88 MB `spectradata_pkl` (taken while the test suite and the engine warm-up were running concurrently) turned out to be corrupt in its last 8 MB (`pickle.UnpicklingError: invalid load key`); several later downloads of the same and of new captures were byte-identical and loaded. It could not be reproduced. If an export fails to load, download it again and compare checksums before suspecting the simulation.
