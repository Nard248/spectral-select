# Brief — server engineer (FastAPI engine service, jobs, exports, CLI)

Read `00-common.md` first. Implement `apps/studio-web/src/api/types.ts` EXACTLY (it's the contract the web app
is built against). Read `src/spectral_select/types.py` (SpectraData / ExcitationData and how they are saved and
loaded — find the existing pickle save/load path used by `spectral_select` and `mehsi_preprocessor` loaders so
exported captures load with the existing tools unchanged) and `docs/onboarding/DATA_GUIDE.md` (real data formats).

Concurrent engineers write `capture.py` (`capture(experiment, *, mode=None, scale=1.0, excitations=None,
progress=None, cancel=None, ...) -> CaptureResult`, `preview(experiment, excitation_nm=None, scale=None)`,
`explain_pixel(experiment, x, y, *, scale=1.0) -> dict`, `GROUND_TRUTH_KEYS`) and `tracer.py`
(`trace_scene(...) -> HitBuffer`). `CaptureResult` is defined in contracts.py. Use dependency injection: the app
factory takes capture/preview/explain callables (defaulting to lazy imports of the real ones) so your tests run
with fakes that return synthetic `CaptureResult`s. Near the end, poll for the real modules and run an end-to-end
smoke test; adapt your side if needed.

## Files
`src/spectraforge/studio/server.py` (`create_app(...)`, `main()`), `jobs.py` (job manager), `exports.py`,
`render_png.py` (colour rendering), `src/spectraforge/studio/__main__.py`, tests `tests/studio/test_server.py`,
`test_exports.py`; in `pyproject.toml` add script `spectralforge-studio = "spectraforge.studio.server:main"` and
optional extra `studio = ["fastapi>=0.110", "uvicorn[standard]>=0.29", "pydantic>=2.6"]` (touch nothing else).

## Endpoints (see types.ts for shapes)
- `GET /api/health`, `GET /api/presets` (built from presets.py, same content as `src/generated/presets.json`).
- `POST /api/validate`: Pydantic errors → `{loc, msg}` list; plus physics warnings computed cheaply (e.g. trace at
  1/8 resolution if the tracer exists: beam misses all objects, object outside depth of field, glint enters the
  camera, camera sees nothing, excitation inside the camera's band list without long-pass…). Never crash on bad input.
- `POST /api/preview` (synchronous; run in a threadpool) → `CaptureMeta` with mode `preview`; stored like captures.
- `POST /api/captures` → job; single worker thread (captures are CPU-heavy), queue the rest; progress events via
  `WS /api/jobs/{id}` (send history then live events; close after terminal event) and `GET /api/jobs/{id}`;
  cancel via `POST /api/jobs/{id}/cancel` (cooperative `cancel()` callable).
- Capture store: in-memory LRU (keep last ~10) + `GET /api/captures` newest first, `GET /api/captures/{id}`.
  `stats` per excitation per band (ADU min/max/p01/p99/mean), `saturated_fraction`, `ground_truth_layers`.
- Binary: `/band`, `/cube` (band fastest), `/gt/{name}` as float32 little-endian with `X-Width`, `X-Height`,
  `X-Bands` headers (and `Access-Control-Expose-Headers`).
- `/image.png`: `true_color` (integrate band radiance against CIE 1931 colour-matching functions → XYZ → sRGB,
  auto exposure at p99, gamma), `false_color` (3 chosen bands or auto), `band` (single band, viridis), `gain` param.
- `/pixel?x=&y=` → `PixelExplain` (wraps `explain_pixel`; scale coordinates if the capture was a scaled preview).
- `/download?format=`: `spectradata_pkl` (a `SpectraData` with ExcitationData per excitation: cube float32 ADU
  minus offset, emission wavelengths, exposure_time, laser_power; metadata incl. the experiment JSON, engine
  version, seed, ground-truth layer names; prove it reloads with the repo's existing loader in a test),
  `npz` (adu/radiance/emission/ground truth), `tiff_zip` (one multi-page 16-bit TIFF per excitation, tifffile is
  installed), `ground_truth_zip` (npy per layer + json), `experiment_json`. Attachment filenames from the
  experiment name + capture id.
- Serve the built web app from `apps/studio-web/dist` at `/` if it exists (SPA fallback), CORS for
  http://localhost:5173 (Vite dev).
- `main()`: `spectralforge-studio [--host 127.0.0.1] [--port 8765] [--open]` (uvicorn; `--open` opens the browser).
  `python -m spectraforge.studio` works too.

## Tests
FastAPI `TestClient` for every endpoint with fake capture callables (incl. WebSocket progress stream, cancel,
binary shapes/headers, PNG decodes, pixel explain, each download format reloads); validation errors for a broken
experiment; presets round-trip through `Experiment.model_validate`.
