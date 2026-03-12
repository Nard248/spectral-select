# Reproducing Paper Results

Scripts in this directory reproduce all experiments and figures from the paper.

## Prerequisites

1. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
2. Place the preprocessed Lichens Dataset 1 (`.pkl`) in `data/processed/Lichens Dataset 1/`.
3. Place the ground-truth label image in `data/ground_truth/`.

## Run Order

| Step | Script | Description | Approx. Time |
|------|--------|-------------|--------------|
| 1 | `run_master_experiment.py` | Run all wavelength selection configurations | ~4-8 hours |
| 2 | `analyze_results.py` | Generate Excel summary and grouped results | ~1 min |
| 3 | `extract_wavelengths.py` | Extract top wavelength combinations per config | ~1 min |
| 4 | `generate_figures.py` | Create all paper figures | ~2 min |
| 5 | `rerun_knn.py` | Re-run KNN for publication-quality maps | ~10 min |
| 6 | `export_tiffs.py` | Export wavelength slices as multi-page TIFFs | ~1 min |

## Expected Outputs

All outputs are written to `results/Lichens_Dataset_1_MasterRun/`:

- `results.csv` — per-experiment metrics
- `comprehensive_analysis.xlsx` — Excel workbook with grouped analysis
- `visualizations/` — all paper figures (PNG)
- `wavelength_analysis/` — top wavelength tables
- `knn_paper_results/` — classification maps and confusion matrices

## Notes

- Step 1 is computationally intensive and benefits from a CUDA-capable GPU.
- Steps 2-6 are post-processing and run quickly on CPU.
- All scripts use relative paths from the repository root.
