# GUI: In-App Training + Perturbation Selection — Design Spec

- **Date:** 2026-06-16
- **Author:** Narek Meloyan
- **Status:** Approved design — pending spec review → implementation plan
- **Repo:** `spectral-select`

## 1. Goal

Let a non-coder run the entire Perturbation-Based Autoencoder wavelength-selection workflow
from a GUI: load and preprocess hyperspectral data (already supported), then **train the
autoencoder** and **run band selection** — without writing or running any code. Deliver this by
extending the existing `mehsi_preprocessor` wizard, not by building a separate app.

Non-goal: changing the selection science or `selection_core` math.

## 2. Decisions (locked with owner)

| # | Decision | Choice |
|---|---|---|
| 1 | Architecture | **Extend `mehsi_preprocessor`** — add Train + Select as wizard steps 9–10 |
| 2 | Training | **Train in-app** (background thread, progress, cancel) **+ load pretrained `.pth`** |
| 3 | Packaging | **`pip install spectral-select[gui]` + console command now**; bundled `.app/.exe` later |

## 3. Architecture

The wizard gains two steps after Export:

```
1 Load → 2 Metadata → 3 Normalize → 4 Spatial Crop → 5 Spectral Crop
  → 6 Draw Classes → 7 ROI Regions → 8 Export
  → 9 Train Autoencoder        (NEW)
  → 10 Select Bands             (NEW)
```

Steps 9–10 read `PipelineState.current_spectra` (the most-processed `SpectraData`) and
`PipelineState.class_mask`, and drive `spectral_select.Analyzer` (which sits on `selection_core`).
No export/import handoff — it is one continuous session.

## 4. Enabling library changes (`spectral_select`)

The GUI needs **train once, tune selection many times**. Two small, reusable additions:

### 4.1 Split `Analyzer` train vs. select

- `Analyzer.prepare(data: SpectraData) -> Analyzer` — the slow, one-time part:
  `_load_data` + `_load_or_train_model` + `_setup_baseline`. Sets `is_prepared`.
- `Analyzer.select(config: Config | None = None) -> WavelengthResult` — the fast, re-runnable part:
  `_select_important_dimensions` + `_compute_influence_scores` + (`_normalize_influences` unless
  `none`) + `_select_top_bands`, reusing the prepared model + baseline. Uses only the
  analysis/perturbation/normalization/selection/diversity fields of `config` (defaults to
  `self._config`); does **not** retrain.
- `fit(data)` becomes exactly `self.prepare(data); self.select(self._config); return self`
  (backward-compatible — experiments and existing API unaffected).

**Guard:** characterization test that `prepare()+select()` produces a `WavelengthResult`
byte-identical to today's `fit()` on a deterministic injected baseline (same fake-model pattern as
`tests/test_analyzer_core_equivalence.py`), so the split changes nothing numerically.

### 4.2 Training progress callback

`spectral_select.models.training.train_with_masking` gains an optional
`progress_callback: Callable[[int, int, float], None] | None = None` invoked once per epoch with
`(epoch, total_epochs, loss)`. Default `None` = current behavior (CLI/experiments unaffected). The
GUI worker passes a callback that emits a Qt signal.

## 5. GUI components

### 5.1 Step 9 — Train Autoencoder (`steps/step9_train.py`)

- Mode toggle: **Train new** (epochs, learning rate, `▸Advanced` for k1/k3/filter_size/sparsity/
  dropout — defaults from `Config`) vs **Load pretrained** (file picker for `.pth`).
- **Train** button starts a `TrainWorker` (QThread): runs `Analyzer.prepare(data)` with a model
  trained via `train_with_masking(..., progress_callback=…)`. Live: epoch progress bar + loss
  curve (reuse `Visualizer`/matplotlib canvas); **Cancel** stops the thread.
- **Load pretrained**: sets `Config.model_path`, runs `prepare(data)` (which loads instead of
  trains), still sets up the baseline.
- On success: stores the prepared `Analyzer` (and `training_losses`) in `PipelineState`, marks the
  model ready, enables Step 10.

### 5.2 Step 10 — Select Bands (`steps/step10_select.py`)

- Selection form (sensible defaults + tooltips): `n_bands_to_select`, `dimension_selection_method`,
  `perturbation_method` + magnitudes + directions, `normalization_method`,
  `use_diversity_constraint` + `diversity_method` + `lambda_diversity` / `min_distance_nm`.
- **Run selection** starts a `SelectWorker` (QThread) calling `analyzer.select(updated_config)`.
  Fast and **re-runnable** with new params without retraining.
- Results: a sorted bands table (rank · excitation nm · emission nm · influence) + an influence
  heatmap (`Visualizer`), and **Export results** → `ResultsManager` (CSV / JSON / TIFF layers).

### 5.3 Threading

A small `workers.py` with `TrainWorker(QThread)` and `SelectWorker(QThread)` emitting
`progress(…)`, `finished(result)`, `error(str)` signals. The window disables sidebar navigation and
the run buttons while a worker is active; Cancel requests cooperative stop.

## 6. State & invalidation (`state.py`)

Add `STEP_TRAIN = 9`, `STEP_SELECT = 10`; extend `_STEP_ATTRIBUTES`:

- step 9 owns: `analyzer` (prepared), `training_losses`, `model_source` ("trained"|"loaded").
- step 10 owns: `selection_result` (`WavelengthResult`), `selection_config`.

`invalidate_from(STEP_EXPORT)` etc. extends to 10, so re-preprocessing (steps 1–7) invalidates the
trained model, and retraining (step 9) invalidates the selection result.

## 7. Packaging

- `pyproject.toml`: keep `[project.optional-dependencies] gui = ["PyQt6>=6.0"]` (matplotlib is
  already a core dependency). Add `[project.scripts] spectral-select-gui = "mehsi_preprocessor.app:main"`.
  Keep `python -m mehsi_preprocessor`.
- README + `docs/`: a short "Run the GUI" section.
- **Later milestone (not now):** a PyInstaller spec producing a double-click `.app/.exe`
  (bundling torch is the hard part; documented, deferred).

## 8. Testing

- **Library:** characterization test `prepare()+select()` == `fit()` (byte-identical); test
  `train_with_masking` invokes `progress_callback` the right number of times.
- **Workers:** test the worker bodies' non-Qt logic (the prepare/select calls + result shape)
  headless, without a display.
- **GUI smoke:** construct `Step9Train`/`Step10Select` under `QT_QPA_PLATFORM=offscreen` and assert
  they build and react to a stubbed analyzer (mark `@pytest.mark.gui`, opt-in in CI).

## 9. Phasing (for the implementation plan)

1. **Library API** — `prepare()`/`select()` split + `fit()` rewrite + `progress_callback`, with
   characterization tests. (No GUI yet; fully testable.)
2. **State + packaging** — `PipelineState` steps 9–10, console entry point, `[gui]` extra.
3. **Step 9 Train** — widget + `TrainWorker` + progress/loss/cancel + load-pretrained.
4. **Step 10 Select** — widget + `SelectWorker` + results table/heatmap + export.
5. **Wire-up + smoke tests + docs** — register steps in `app.py`, offscreen smoke tests, README.

Each phase produces working, testable software; phase 1 is independently valuable.

## 10. Risks & mitigations

- **UI freeze during training** → all heavy work in `QThread`; only signals touch the UI.
- **`prepare/select` split changes results** → byte-identical characterization test before refactor.
- **Cancel mid-training leaves partial state** → worker sets model only on clean completion;
  Cancel restores "no model" state.
- **No-GPU users / long training** → progress + cancel + the load-pretrained path; document that
  training is slow on CPU.
- **Qt in headless CI** → GUI tests use `offscreen` and are opt-in (`-m gui`).

## 11. Out of scope (YAGNI / future)

- Validation/clustering-metrics step (future **Step 11**, reusing `Validator` + `class_mask`).
- Bundled standalone installer (later milestone).
- The non-HSI `channel_select` domain (this GUI is HSI-focused; `channel_select` stays code-only).
