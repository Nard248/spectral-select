# Brief — analysis engineer (analysis library, server endpoints, sweeps, exports)

Read `00-common.md`, `10-v11-common.md`, `apps/studio-web/src/api/types.ts` (the v1.1 section at the end is
your contract — implement it exactly), `contracts.py` (`CaptureResult` incl. new `blocks` / `volumes`), and
the current `server.py`, `jobs.py`, `exports.py` (security middleware must keep working: every new POST is
JSON, same-origin rules apply to the new WebSocket too). The block engineer fills `CaptureResult.blocks`,
`.volumes` and the new ground-truth keys concurrently — test with fake `CaptureResult`s that include them.

## Files
NEW `src/spectraforge/studio/analysis.py`, NEW `sweeps.py`; `server.py`, `jobs.py`, `exports.py` (+ tests
`tests/studio/test_analysis.py`, `test_sweeps.py`, existing server/exports tests).

## analysis.py (pure functions on CaptureResult — also usable from notebooks)
- `normalize(result, norm, *, dark=None, white=None) -> dict[ex, (H,W,B) float32]` for every NormSpec mode:
  adu, radiance, electrons, dark_subtracted (ADU − dark ADU), exposure_power (ADU−offset)/(exposure_s ×
  power_mW) — matching the lab's normalisation; white_normalized ((S−D)/(W−D), with safe division);
  return the unit string. Validate that dark/white captures match the sample geometry (size, excitations,
  bands) and raise a clear error otherwise.
- `roi_mask(spec, H, W)` for point, disk, rect, ellipse, polygon (even-odd rule, pixel centres).
- `eem(result, roi, norm, stat)` → EemResult (per excitation per band mean/median + std; plus the common
  2 nm emission grid resampling with NaN outside each excitation's band span).
- `roi_stats(... components=False)` → RoiStats incl. ground-truth stats (numeric layers only, NaN-aware)
  and, if requested, the decomposition averaged over ≤ 64 subsampled ROI pixels via the capture module's
  `explain_pixel` (inject it; cache).
- `line_profile(...)`: bilinear sampling along p0→p1 with perpendicular width averaging; GT sampled along the
  line; `distance_mm` from depth × pixel angular size where depth is finite.
- `response_curve(...)`: per-pixel metric (band nearest to band_nm; band_ratio; band_sum over a wavelength
  span; peak_wavelength with parabolic interpolation) vs a GT map inside the ROI (or all finite pixels),
  binned mean/std/count + ≤ 4000-point scatter subsample.
- `volume(result, object_id, kind, ex, channel)` + `volume_infos(result)`.
- `load_npz_capture(path)` → a CaptureResult-like object from the npz export, so all functions work offline
  in Jupyter. Document a 15-line notebook example in the module docstring and test it.

## Server (`server.py`)
All v1.1 endpoints from types.ts; `CaptureMeta` gains `volumes` and `blocks`. Errors → 4xx with clear
messages (bad ROI, mismatched dark/white, unknown GT layer). Binary volume response float32 LE with
`X-Dims`. Keep all v1 endpoints unchanged.

## Sweeps (`sweeps.py`)
`SweepManager` (one worker, shares the capture worker or queues behind it — do not run two heavy MC jobs
at once): for each value, deep-copy the experiment, set the JSON pointer (RFC 6901) value, re-validate
with `Experiment.model_validate` (report the bad value as an error event, continue or stop — document), run
preview or sample capture, store the capture (pin sweep captures so the LRU doesn't evict them — e.g. a
separate store sized to the sweep, max 64 points), emit `point` events. Cancel cooperatively. WebSocket
history replay like jobs.

## Exports
npz export includes block volumes (keys `volume/<object>/<kind>/<ex>[/<channel>]`, sanitised with the
existing `safe_name`) and the new GT layers; `ground_truth_zip` includes the new layers.

## Tests
Every NormSpec mode against hand-computed values; every ROI shape mask; EEM grid resampling; line profile
on a synthetic gradient; response curve recovers a known synthetic dependence; sweeps end-to-end with a fake
capture fn (incl. cancel, invalid value); server endpoints via TestClient (JSON-only, origin rules);
npz round-trip through `load_npz_capture`.
