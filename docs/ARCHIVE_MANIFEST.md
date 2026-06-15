# Archive Manifest

Everything under the gitignored `archive/` directory, with its original path, date moved, and
reason. **Items here are NOT deleted** — they remain on disk and are recoverable. This file is
tracked so the provenance lives in git even though the archived content does not.

## Pre-existing legacy archive (predates the 2026-06-15 cleanup)

These directories were already under `archive/` before this reorganization (~7.1 GB total):

- `archive/wavelength_analysis/` — early experiment outputs (.ipynb, .pkl, .xlsx, .npy) from the
  pre-public-API era.
- `archive/MCR-Analysis/`, `archive/manual-utils/`, `archive/setupScripts/`,
  `archive/visualizations/`, `archive/paper/` — assorted pre-cleanup exploratory work.

## Moved during the 2026-06-15 structural cleanup

| Original path | Archived to | Date | Reason |
|---|---|---|---|
| `TM.py` | `archive/misc/TM.py` | 2026-06-15 | Unrelated Turing-machine class artifact |
| `Paper Source/` | `archive/misc/Paper Source/` | 2026-06-15 | Legacy paper wrapper, superseded by `publications/tpami/paper/` |
| `model_output/` | `archive/misc/model_output/` | 2026-06-15 | Old `.pth`/`.npy` training artifacts, superseded by `results/` |
| `visualizations/` | `archive/misc/visualizations/` | 2026-06-15 | Old viz outputs, superseded by `results/` |
| `run_preprocessor.py` | `archive/misc/run_preprocessor.py` | 2026-06-15 | Redundant launcher; use `python -m mehsi_preprocessor` |
