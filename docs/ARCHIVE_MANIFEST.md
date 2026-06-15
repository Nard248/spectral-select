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
| `Data/Raw/Lime/` | `archive/redundant_data/Raw/Lime/` | 2026-06-15 | Orphan raw set, 0 code refs |
| `Data/processed/Sample Processed/` | `archive/redundant_data/processed/Sample Processed/` | 2026-06-15 | Orphan, 0 code refs |
| `Data/processed/Sample Export 2/` | `archive/redundant_data/processed/Sample Export 2/` | 2026-06-15 | Orphan, 0 code refs |
| `Data/processed/LichensProcessed/` | `archive/redundant_data/processed/LichensProcessed/` | 2026-06-15 | Redundant Lichens (60nm), 0 refs; canonical = Lichens Dataset 1 |
| `Data/processed/Lichhens 2 - Processed/` | `archive/redundant_data/processed/Lichhens 2 - Processed/` | 2026-06-15 | Typo dup of Lichens_2, 0 refs |
| `Data/processed/lichens_data_cropped/` | `archive/redundant_data/processed/lichens_data_cropped/` | 2026-06-15 | Redundant Lichens (3.9G), 0 refs |
| `Data/processed/{spectra_masked,spectra_unmasked}.pkl`, `class_mask.png`, `roi_regions.json` | `archive/redundant_data/processed/_loose_root/` | 2026-06-15 | Stray exports at processed/ root, 0 refs |
| `Data/Raw/Lichens/` (empty), `Data/processed/YourSample/` (empty) | removed (rmdir) | 2026-06-15 | Empty placeholder dirs |
