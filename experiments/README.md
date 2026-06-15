# Experiments

Three tiers, by lifecycle:

- **Top level** — maintained, reusable pipeline drivers (the canonical Lichens run + analysis).
- **`pamap2/`** — the generalization domain (wearables / time-series) using `channel_select`.
- **`_archive/2026_paper_runs/`** — one-off, dataset-specific paper & figure scripts kept for
  provenance. Runnable, but tied to specific data paths and 2026 submissions; not maintained.

## Prerequisites

```bash
pip install -e ".[dev]"
```
Place the preprocessed `Lichens Dataset 1` (`.pkl`) under `Data/processed/Lichens Dataset 1/`
and ground-truth labels as documented in `docs/DATA.md`.

## Reusable drivers — run order (canonical Lichens pipeline)

| Step | Script | Description | Approx. Time |
|------|--------|-------------|--------------|
| 1 | `run_master_experiment.py` | Run all wavelength-selection configurations | ~4–8 h |
| 2 | `analyze_results.py` | Generate Excel summary and grouped results | ~1 min |
| 3 | `extract_wavelengths.py` | Extract top wavelength combinations per config | ~1 min |
| 4 | `export_wavelength_combinations.py` | Export selected band combinations | ~1 min |
| 5 | `generate_figures.py` | Create all paper figures | ~2 min |
| 6 | `rerun_knn.py` | Re-run KNN for publication-quality maps | ~10 min |
| 7 | `export_tiffs.py` | Export wavelength slices as multi-page TIFFs | ~1 min |

Outputs are written to `results/Lichens_Dataset_1_MasterRun/` (`results.csv`,
`comprehensive_analysis.xlsx`, `visualizations/`, `wavelength_analysis/`, `knn_paper_results/`).

## `pamap2/` — generalization domain

Domain-agnostic channel selection on the PAMAP2 wearables dataset, validating that the
perturbation method transfers beyond hyperspectral imaging. Uses `channel_select` (engine +
temporal grouped autoencoder + PAMAP2 adapters). Scripts: `general_pamap2_slice.py`,
`general_pamap2_loso.py`, `general_pamap2_baseline_diag.py`, `general_pamap2_richfeat_diag.py`,
`general_make_figures.py`. Requires `Data/Raw/PAMAP2_MONSTER/`.

## `_archive/2026_paper_runs/` — one-off paper scripts

Dataset-specific, run-once scripts for the 2026 submissions, grouped by topic:
`collagen_*` (3), `drop_data_*` (16, incl. radiometric reruns), `pepsin_*` (6), `poster_*` (6),
plus `collagen_tuning.py` / `reprocess_with_metadata.py` recovered from the
`collagen-experiments` branch. Kept for reproducibility; not part of the maintained surface.

## Notes
- Step 1 benefits from a CUDA-capable GPU; the rest run quickly on CPU.
- Scripts use paths relative to the repository root.
