# Component guides

One technical guide per component of the repository, written for someone who has finished [`../SETUP.md`](../SETUP.md) and the walkthrough notebook and now needs to work *inside* a component. Each guide covers purpose and status, the files, the data model and API, how to run it, the internals, what it writes, its tests, and the known issues found when the guides were checked against the code (September 2026).

| # | Guide | Package / folder | One line |
|---|---|---|---|
| 1 | [SpectraForge](01-spectraforge.md) | `src/spectraforge/` | synthetic ME-HSI generator with ground truth, validation harness, painter GUI |
| 2 | [MEHSI preprocessor (the Image Reader)](02-mehsi-preprocessor.md) | `src/mehsi_preprocessor/` | `.im3` loader through PyImageJ, pure preprocessing functions, the 10-step wizard |
| 3 | [spectral_select](03-spectral-select.md) | `src/spectral_select/` | the HSI library: data types, `Analyzer` (train + select), autoencoder, visualiser, validator, viewers |
| 4 | [selection_core](04-selection-core.md) | `src/selection_core/` | the shared perturbation-influence algorithm, step by step |
| 5 | [channel_select](05-channel-select.md) | `src/channel_select/` | domain-agnostic DL selection: grouped-channel data, bottleneck autoencoders, adapters, synthetic benchmark |
| 6 | [Experiments, results and publications](06-experiments-and-results.md) | `experiments/`, `results/`, `reports/`, `publications/` | how the paper numbers are produced, which scripts are canonical, where outputs go |
| 7 | [SpectralForge Studio](07-spectralforge-studio.md) | `src/spectraforge/studio/`, `apps/studio-web/`, `docs/studio/` | the 3D experiment designer and physical HSI simulator (successor of the 2D painter GUI); this page is a map into `docs/studio/` |

Suggested order for a new developer: 3 (the data types everything else uses), 2, 4, 5, 1, 6, then 7 if you will touch the simulator. For a new data analyst: 2, 3, 6, and 7 to understand where simulated cubes come from.
