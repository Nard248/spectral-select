# W3 brief: study engineer (headless API, design of experiments, global sensitivity, provenance, datasets, CLI)

Read `20-v12-common.md` first, then `sweeps.py`, `analysis.py`, `server.py`, `exports.py`, `jobs.py`,
`__main__.py`, and `src/spectral_select/types.py` (`SpectraData`). Goal: scientists can run whole
**computational studies** reproducibly from Python, the CLI, or the web, and cite them.

## 1. `api.py`: headless Python API (a notebook-first, documented facade)
```python
from spectraforge.studio import api
exp = api.preset("tissue_under_collagen")          # or api.load("exp.json")
exp = api.set(exp, "/objects/0/shape/layers/2/thickness/thickness_mm", 0.3)   # RFC 6901, returns a validated copy
res = api.capture(exp, mode="preview"|"full", seed=0)   # CaptureResult
api.eem(res, roi=..., norm=...) ; api.response(...) ; api.save(res, "out.npz")
api.to_spectradata(res, ...)  # -> spectral_select.types.SpectraData (the repo's ML pipeline input)
```
Reuse `capture.py` and `analysis.py`; do not duplicate them. Include a module docstring with a complete example.

## 2. `study_schema.py` + `study.py`: design of experiments
Pydantic `StudySpec`:
- `name`, `base_experiment` (Experiment or preset name).
- `parameters`: list of `{pointer, label, kind: continuous|integer|choice, low, high, scale: linear|log, values}`.
  Linked parameters: `also_set: [pointers]`, which set the same value (for example a wedge's start and end).
- `design`: `{kind: grid|random|lhs|sobol|saltelli|morris, n, levels, seed}`. Write your own scrambled
  Sobol sequence via `scipy.stats.qmc` (allowed), LHS, a Saltelli scheme for Sobol indices, and Morris
  trajectories.
- `replicates`: MC seeds per design point (for the MC-noise standard error).
- `mode`: preview or full.
- `metrics`: list of `{name, kind: roi_band|band_ratio|band_sum|peak_wavelength|eem_integral|gt_stat|
  contrast|cnr, roi, norm, ...}`. Contrast and CNR are between two ROIs, for inclusion detectability.
  Reuse `analysis.py`.

`run_study(spec, out_dir, *, workers=1, resume=True, progress=cb)` writes:
- `runs.parquet` (and CSV): one row per (point, replicate), with parameter values, metrics, timings, seed,
  and experiment hash.
- `manifest.json`: provenance.
- Optionally the per-run `.npz` (flag `keep_captures`).

It must be resumable (skip run hashes already present), cancellable, and deterministic.

## 3. `provenance.py`
`experiment_hash(exp)` is a canonical-JSON SHA-256. `provenance()` returns:
- package version and git commit (with a dirty flag);
- Python, numpy, numba, and platform versions;
- CPU count, timestamps, and the numba threading layer.
Embed provenance in every export (npz, study manifest) and in a `CaptureResult.meta` entry (additive).

## 4. `sensitivity.py`: global sensitivity analysis
- Sobol first-order and total-order indices from a Saltelli design. Use the Saltelli 2010 / Jansen 1999
  estimators, with bootstrap 95 % CIs.
- Morris μ*, σ (Campolongo 2007).
- Validate both on the **Ishigami function** (analytic S1 = 0.3139, 0.4424, 0; ST = 0.5576, 0.4424, 0.2437
  for a=7, b=0.1) and on Sobol' G. Tolerance ±0.02 at N = 2^13.
- Report indices for study metrics. Account for MC noise: the replicate variance gives a "noise floor" index.

## 5. `dataset.py`: synthetic dataset generator for ML
`generate_dataset(spec, out_dir)`: a randomized domain over the study parameters produces N captures. It
saves cubes + GT + labels (the per-pixel class map from GT, e.g. material/layer/inclusion ids, plus
continuous targets such as cover thickness), with a `datasets.json` index, and exports to `SpectraData`
pickles that the repo's `spectral_select` pipeline loads. Test the round trip.

## 6. `methods_text.py`
`methods_text(exp, study=None, *, style="markdown"|"latex")` generates a publication-ready **Methods
paragraph**: the geometry, every material's optical properties at the excitation wavelengths (μa, μs', g, n,
dye concentrations), the source, camera, filter and sensor, the physics toggles that are OFF, the MC photon
counts, the seeds, the software version and commit, and a citation line. Test it with a snapshot.

## 7. CLI `cli.py` (wired into `__main__.py`; keep `spectralforge-studio` / `python -m spectraforge.studio` serving by default)
- `spectralforge-studio serve [--port] [--open]` (default).
- `run EXP.json|preset --mode full --out DIR [--png]`.
- `study STUDY.json --out DIR [--workers]`.
- `sensitivity DIR`, which prints and saves Sobol/Morris tables.
- `dataset SPEC.json --out DIR`.
- `methods EXP.json [--latex]`.
- `presets`.

## 8. Server
`routes_study.py` (an APIRouter) handles:
- `POST /api/studies` (StudySpec), WS `/api/studies/{id}/events`, `GET /api/studies/{id}`,
  `GET /api/studies/{id}/runs` (JSON rows), and `GET /api/studies/{id}/sensitivity`.
- `POST /api/methods_text`.
- `GET /api/provenance`.

It runs through the same single heavy worker as captures and sweeps. Include it and the W4 router
`routes_inverse.router` in `server.py` (lazy try-import, so the server starts even if the W4 file is
missing). Keep the security middleware rules. Write a `docs/studio/STUDY_API.md` that documents every
endpoint (with JSON examples) for the web engineer.

## 9. Tests + demo
Unit tests for every module, including a tiny end-to-end study on `layered_wedge_phantom` in preview mode
with a 2-parameter LHS of 6 points and 2 replicates, which must finish in < 60 s. Include the resume test,
the Ishigami validation, and the SpectraData round trip. Also provide `examples/studio/notebook_study.py`
(a py:percent script) that runs a small Sobol study and prints the indices.
