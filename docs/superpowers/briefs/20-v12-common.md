# SpectralForge Studio v1.2 "Research grade": common rules (read before your own brief)

You are one of several parallel engineers taking **SpectralForge Studio** (a 3D experiment designer and a
physically comprehensive multi-excitation fluorescence hyperspectral simulator) from a demo to a **research
instrument** that scientists use to design experiments, quantify effects, and write papers. The user
(sole author, researcher in hyperspectral fluorescence imaging; the lab studies collagen, tissue, and
lichens with an ME-HSI setup: a tunable excitation lamp and a Nuance LCTF camera) gave us full autonomy. **The
priorities are physical correctness and quantitative rigor, then flexibility and usability.** A paper will be written
from your work, so every claim needs a number, a test, or a reference.

## Workspace (CHANGED from v1/v1.1)
- Single central checkout: `/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering`,
  branch **`studio/v1.2-research`**. There is NO worktree any more. The venv's editable install points
  here, so no PYTHONPATH is needed.
- Python: `cd /Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering && QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/studio -q -o addopts="" -m "not slow" -p no:warnings`
  (numpy 2.1, scipy 1.15, numba 0.63, sklearn 1.6, pandas 2.2, pyarrow 19, matplotlib 3.10, torch 2.6;
  NO SALib or statsmodels: implement what you need). Do not install Python packages.
- Web: `apps/studio-web` (`npm test`, `npm run build`, `npx eslint .`). Only the web engineer may `npm install`.
- Do NOT git commit, push, switch branches, stash, or reset. Other engineers are editing the same checkout
  at the same time, so touch ONLY your own files. Never run `git checkout -- <file>` on files you don't own.
- Read `docs/superpowers/briefs/00-common.md` (quality bar) and `10-v11-common.md`. Their workspace paths
  are outdated: use the paths above. Then read `docs/studio/HOW_IT_WORKS.md`, `ARCHITECTURE.md`, and
  `PHYSICS_VALIDATION.md` to understand the current engine.

## New contracts (lead-owned, read-only for you)
`src/spectraforge/studio/schema.py`:
- `ScatteringSpec.model = "mie_rayleigh"` with `f_rayleigh` (Jacques 2013 eq. 2).
- `TissueComposition` with `MaterialSpec.tissue` (chromophore absorption: blood fraction and StO2,
  water, lipid, melanosome, bilirubin, β-carotene).
- `Inclusion` with `LayeredBlockShape.inclusions` (sphere, ellipsoid, cylinder, box; index-matched and
  voxelised at `acquisition.volume_mc.inclusion_voxel_mm`).
Regenerate the web contract only via `.venv/bin/python scripts/export_studio_contract.py && (cd apps/studio-web && npm run gen:types)`.
If you need a schema change, describe it in your report (the lead applies it). Adding NEW schema files of
your own (for example `study_schema.py`) is fine when your brief says so.

## v1.2 ownership map
| engineer | owns |
|---|---|
| tissue (W1) | `optics.py`, `spectra.py`, NEW `tissue_optics.py` + NEW data files under `src/spectraforge/studio/data/`, NEW tissue presets via NEW `presets_tissue.py` (the lead wires it into `presets.py`), tests `test_optics.py`, `test_spectra.py`, NEW `test_tissue_optics.py` |
| inclusions (W2) | `transport3d.py`, `transport3d_mc.py`, `block_capture.py`, the block section of `contracts.py`, `geometry.py`/`geometry_nb.py`/`tracer.py` (block parts only), tests `test_transport3d*.py`, `test_block_capture.py`, NEW `test_inclusions*.py` |
| study (W3) | NEW `study_schema.py`, `study.py`, `provenance.py`, `sensitivity.py`, `dataset.py`, `methods_text.py`, `api.py` (headless Python API), `cli.py`, NEW `routes_study.py`; `server.py`, `__main__.py`, `sweeps.py`, `exports.py`, `jobs.py`; `pyproject.toml` (scripts/extras only); tests for all of these |
| inverse (W4) | NEW `unmix.py`, `inverse.py`, `information.py` (CRLB / Fisher / design), `surrogate.py`, NEW `routes_inverse.py` (FastAPI `APIRouter` named `router`; W3 includes it in `server.py`), tests `test_unmix.py`, `test_inverse.py`, `test_information.py` |
| web (W5) | `apps/studio-web/**` except `src/generated/**` |

Concurrent modules may not exist yet. Code against the schema, import other engineers' modules lazily, and use
fakes in unit tests. Near the end, integrate with the real modules and adapt YOUR side. `analysis.py`,
`capture.py`, and `sensor.py` are shared read-mostly: make only small, additive changes, and list them in your report.

## Quality bar (stricter for v1.2)
- Physics: literature references (author, year, journal, DOI) in docstrings next to every equation and table.
  Tabulated literature data goes in plain CSV/NPZ under `src/spectraforge/studio/data/` with a header citing the source.
- Every quantitative feature needs a validation test against an analytic result or independent reference. Report the
  numbers (value against reference, relative error) in your final report.
- Determinism: everything is seedable and the same seed gives the same result.
- Keep all tests fast (< 30 s each) unless they are marked `@pytest.mark.slow`.
- At the end, the full suite (`tests/studio` and `tests/spectraforge`, not slow) must pass, along with your own tests.
- Final report: what you built, public API, decisions and approximations, validation numbers, timings, and
  needed contract changes. It should be good enough for the lead to copy into a paper's Methods and Results.
