# Study API (SpectralForge Studio v1.2)

Endpoints added by `routes_study.py` for computational studies (design of experiments, Sobol'/Morris sensitivity),
Methods text and provenance. All requests are JSON (`Content-Type: application/json`); the v1 security rules (loopback
Host, same-origin, JSON-only POST) apply. Studies run on the **same single heavy worker** as captures and sweeps, so a
study queues behind a running capture. Results are written to `$SPECTRAFORGE_STUDY_DIR` (default
`<tmp>/spectraforge_studies/<study_id>/`). The Python equivalent of everything here is `spectraforge.studio.study`,
`sensitivity`, `methods_text` and `provenance`; the CLI is `spectralforge-studio study|sensitivity|methods ...`.

Typed schema: `StudySpec` in `src/spectraforge/studio/study_schema.py` (the web engineer can generate JSON Schema with
`StudySpec.model_json_schema()`).

## StudySpec

```json
{
  "name": "cover vs fluorescent layer",
  "base_experiment": "layered_wedge_phantom",
  "parameters": [
    {"pointer": "/objects/0/shape/layers/2/thickness/thickness_mm", "label": "cover_mm",
     "kind": "continuous", "low": 0.1, "high": 1.5, "scale": "linear",
     "also_set": ["/objects/0/shape/layers/2/thickness/end_thickness_mm"]},
    {"pointer": "/acquisition/preview_scale", "label": "scale", "kind": "choice", "values": [0.2, 0.3]}
  ],
  "design": {"kind": "saltelli", "n": 64, "levels": 4, "seed": 0},
  "replicates": 2,
  "mode": "preview",
  "excitations": null,
  "metrics": [
    {"name": "signal", "kind": "band_sum", "roi": {"shape": "rect", "x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7}},
    {"name": "peak", "kind": "peak_wavelength", "ex": 340},
    {"name": "cnr", "kind": "cnr", "roi": {"shape": "rect", "x0": 0, "y0": 0, "x1": 0.2, "y1": 0.2},
     "roi_b": {"shape": "rect", "x0": 0.5, "y0": 0.5, "x1": 0.9, "y1": 0.9}}
  ],
  "max_runs": 20000
}
```

* `base_experiment`: a full `Experiment` object **or** a preset name.
* `parameters[].kind`: `continuous` (low/high, `scale` linear|log), `integer` (low/high, uniform over the integers),
  `choice` (`values`). `also_set` pointers receive the same value (linked parameters).
* `design.kind`: `grid` (`levels` per continuous parameter), `random`/`lhs`/`sobol` (`n` points), `saltelli`
  (`n` base samples, rows = n(k+2); required for Sobol' indices), `morris` (`n` trajectories, `levels` even, rows = n(k+1)).
* `replicates`: Monte Carlo seeds `base_seed + r` per design point (gives the noise floor).
* `mode`: `preview` (first excitation, `preview_scale` resolution) or `full` (all excitations, native resolution).
* `metrics[].kind`: `roi_band`, `band_ratio`, `band_sum`, `peak_wavelength`, `eem_integral`, `gt_stat`, `contrast`
  (Michelson or Weber), `cnr` (between `roi` and `roi_b`). ROI coordinates are image **fractions** (0..1) unless
  `"roi_units": "pixels"`. `norm` is an `analysis.NormSpec` (`adu` default). Full field list: `MetricSpec`.

## Endpoints

### `POST /api/studies`  (optional `?keep_captures=1`)
Body: a `StudySpec`. Validates (pointers apply, ranges produce valid experiments, `n_runs <= max_runs`) before queuing.

```json
200 {"study_id": "9f2c1a7b3d40", "n_runs": 256, "n_points": 128}
422 [{"loc": ["design", "kind"], "msg": "Input should be 'grid', ..."}]      // schema errors
422 "path '/objects/0/x': no key 'x' (...)"                                   // pointer / value errors
```

### `GET /api/studies/{id}`
```json
{"study_id": "9f2c1a7b3d40", "name": "cover vs fluorescent layer", "state": "running",
 "fraction": 0.31, "message": "run 80/256", "n_runs": 256, "error": null}
```
`state`: `queued | running | done | error | cancelled`. When terminal, `manifest` (the contents of `manifest.json`:
spec, spec hash, provenance, counts, timings) is included.

### `WS /api/studies/{id}/events`
Server pushes `{"type":"progress","fraction":0.31,"message":"run 80/256"}` and finally one of
`{"type":"done","study_id":"...","n_failed":0}`, `{"type":"error","message":"..."}`, `{"type":"cancelled"}`, then closes.
Unknown id: close code 4404.

### `POST /api/studies/{id}/cancel`
Cooperative cancel between runs; returns the status. Finished runs are kept (resume by re-submitting the same spec
to a script / CLI with the same output directory).

### `GET /api/studies`
List of status objects, newest first.

### `GET /api/studies/{id}/runs?offset=0&limit=1000`
One row per (design point, replicate); NaN is `null`; `limit` <= 5000. 409 until the first results are written.
```json
{"total": 256, "columns": ["run_id","point","replicate","seed","exp_hash","block","row","traj","step","step_param",
  "direction","p:cover_mm","u:cover_mm","p:scale","u:scale","m:signal","m:peak","status","error","t_capture_s",
  "t_metrics_s","warnings"],
 "rows": [{"run_id": "…", "point": 0, "replicate": 0, "seed": 0, "p:cover_mm": 0.62, "u:cover_mm": 0.37,
           "m:signal": 3239.4, "status": "ok", "t_capture_s": 1.7, "...": "..."}]}
```
`p:*` are parameter values, `u:*` the unit-cube coordinates, `m:*` the metrics; `block`/`row` identify Saltelli
blocks (`A`, `B`, `AB:i`) and `traj`/`step`/`step_param`/`direction` Morris trajectories.

### `GET /api/studies/{id}/sensitivity?boot=300`
Only for `saltelli` (Sobol') and `morris` designs (422 otherwise); computed on demand and cached.
```json
{"design": "saltelli", "parameters": ["cover_mm", "scale"],
 "metrics": {"signal": {"method": "sobol", "n_rows": 64, "variance": 412.3, "sum_S1": 0.93,
   "noise": {"replicates": 2, "within_var": 0.8, "noise_floor": 0.004},
   "indices": {"cover_mm": {"S1": 0.71, "S1_ci": [0.64, 0.78], "ST": 0.75, "ST_ci": [0.69, 0.81]},
               "scale": {"S1": 0.2, "S1_ci": [0.12, 0.27], "ST": 0.25, "ST_ci": [0.2, 0.3]}}}}}
```
Morris entries have `method: "morris"` and per parameter `mu`, `mu_star`, `mu_star_ci`, `sigma`, `n_effects`.
`noise_floor` is the fraction of output variance due to Monte Carlo noise (NaN -> `null` with one replicate).

### `POST /api/methods_text`
Body `{"experiment": <Experiment>, "study": <StudySpec>?, "style": "markdown"|"latex"}`.
```json
200 {"style": "markdown", "text": "**Simulation.** Multi-excitation ... "}
```

### `GET /api/provenance`
```json
{"package": "spectral-select", "package_version": "0.1.0", "engine_version": "studio-1.0",
 "git": {"commit": "…", "branch": "studio/v1.2-research", "dirty": true},
 "python": "3.11.0", "numpy": "2.1.3", "scipy": "1.15.2",
 "numba": {"version": "0.63.1", "threading_layer": "workqueue", "num_threads": 14},
 "platform": "macOS-…", "machine": "arm64", "cpu_count": 14, "timestamp_utc": "2026-10-13T08:00:00+00:00"}
```

## Files written per study
`runs.parquet`, `runs.csv`, `manifest.json`, `runs.jsonl` (journal; makes a study resumable), optional
`captures/<run_id>.npz` (each embeds its provenance and reloads with `api.load_capture`), and after an analysis
`sensitivity.json` / `sensitivity.csv`.

## Provenance in exports
`exports.to_npz` stores `result_meta_json` (incl. `provenance`: versions, git commit/dirty flag, seed, `experiment_hash`
= SHA-256 of the canonical experiment JSON); `analysis.load_npz_capture` restores it as `result.meta`. The
SpectraData pickle export carries a top-level `provenance` key (lands in `SpectraData.metadata`).
