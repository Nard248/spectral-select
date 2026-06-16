# GUI Train + Select Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a non-coder train the autoencoder and run Perturbation-Based wavelength selection from the existing `mehsi_preprocessor` GUI — as wizard steps 9 (Train) and 10 (Select).

**Architecture:** A small `spectral_select` API split (`prepare()` = slow train/baseline; `select()` = fast, re-runnable selection) underpins two new Qt wizard steps that run work in `QThread` workers. `select()`/`prepare()` reuse the already-unified `selection_core`. Backward-compatible: `fit()` = `prepare()+select()`.

**Tech Stack:** Python 3.11, PyTorch, PyQt6, matplotlib, pytest.

**Source spec:** `docs/superpowers/specs/2026-06-16-gui-train-select-design.md`

**Conventions:** repo root = `/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering`; `source .venv/bin/activate` first; GUI tests run with `QT_QPA_PLATFORM=offscreen`.

## File map

| File | Responsibility | Action |
|---|---|---|
| `src/spectral_select/models/training.py` | add `progress_callback` to `train_with_masking` | modify |
| `src/spectral_select/analyzer.py` | add `prepare()`, `select()`, `is_prepared`; `fit()`→`prepare()+select()` | modify |
| `src/mehsi_preprocessor/state.py` | steps 9/10 constants, attrs, invalidation | modify |
| `src/mehsi_preprocessor/workers.py` | `TrainWorker`, `SelectWorker` (QThread) | create |
| `src/mehsi_preprocessor/steps/step9_train.py` | Train step widget | create |
| `src/mehsi_preprocessor/steps/step10_select.py` | Select step widget | create |
| `src/mehsi_preprocessor/app.py` | register steps 9/10 | modify |
| `pyproject.toml` | `spectral-select-gui` console script | modify |
| `README.md` | "Run the GUI" section | modify |
| `tests/test_training_callback.py` | callback fires per-epoch | create |
| `tests/test_analyzer_prepare_select.py` | `select()` == legacy private sequence; `fit()`==`prepare()+select()` | create |
| `tests/mehsi_preprocessor/test_workers.py` | worker logic headless | create |
| `tests/mehsi_preprocessor/test_steps_smoke.py` | steps build offscreen | create |

---

## Phase 1 — Library API (no GUI; fully testable)

### Task 1.1: `train_with_masking` per-epoch progress callback

**Files:**
- Modify: `src/spectral_select/models/training.py` (signature ~line 138; epoch loop ~line 277; `train_losses.append(avg_loss)` ~line 320)
- Test: `tests/test_training_callback.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training_callback.py
import numpy as np
import torch
from spectral_select.models.autoencoder import HyperspectralCAEWithMasking
from spectral_select.models.dataset import MaskedHyperspectralDataset
from spectral_select.models.training import train_with_masking


def _tiny_dataset():
    np.random.seed(0)
    ex_data = {365.0: np.random.rand(8, 8, 4).astype(np.float32)}
    mask = np.ones((8, 8), dtype=bool)
    model = HyperspectralCAEWithMasking(excitations_data=ex_data, k1=4, k3=2, filter_size=3)
    dataset = MaskedHyperspectralDataset(ex_data, mask=mask)
    return model, dataset, mask


def test_progress_callback_fires_once_per_epoch():
    torch.manual_seed(0)
    model, dataset, mask = _tiny_dataset()
    calls = []
    train_with_masking(
        model=model, dataset=dataset, num_epochs=3, learning_rate=1e-3,
        chunk_size=8, chunk_overlap=0, device="cpu",
        early_stopping_patience=999, scheduler_patience=999, mask=mask,
        output_dir=None, verbose=False,
        progress_callback=lambda epoch, total, loss: calls.append((epoch, total, loss)),
    )
    assert [c[0] for c in calls] == [1, 2, 3]
    assert all(c[1] == 3 for c in calls)
    assert all(isinstance(c[2], float) for c in calls)
```

- [ ] **Step 2: Run — verify it fails**

Run: `pytest tests/test_training_callback.py -q`
Expected: FAIL — `train_with_masking() got an unexpected keyword argument 'progress_callback'`.
(If `MaskedHyperspectralDataset`/`HyperspectralCAEWithMasking` constructor args differ, adjust the test to the real signatures first — read `src/spectral_select/models/dataset.py` and `autoencoder.py` — then re-run; it must still fail on the missing kwarg.)

- [ ] **Step 3: Implement**

In `train_with_masking`'s signature add `progress_callback=None` (after `verbose=True`). Immediately after `train_losses.append(avg_loss)`:
```python
            if progress_callback is not None:
                progress_callback(epoch + 1, num_epochs, float(avg_loss))
```

- [ ] **Step 4: Run — verify passes**

Run: `pytest tests/test_training_callback.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_select/models/training.py tests/test_training_callback.py
git commit -m "feat(spectral_select): per-epoch progress_callback in train_with_masking"
```

### Task 1.2: `Analyzer.prepare()` / `select()` split

**Files:**
- Modify: `src/spectral_select/analyzer.py` (`fit()` ~line 151; add new methods; add `_is_prepared` flag in `__init__`)
- Test: `tests/test_analyzer_prepare_select.py`

- [ ] **Step 1: Write the failing characterization test**

Reuse the injected-state pattern from `tests/test_analyzer_core_equivalence.py` (a `FakeHSIModel` with `decode`/`excitation_wavelengths`/`emission_bands` + a dataset stub with `get_all_data()` and `emission_wavelengths`).

```python
# tests/test_analyzer_prepare_select.py
import numpy as np
import torch
from spectral_select import Analyzer, Config
from tests.test_analyzer_core_equivalence import FakeHSIModel, FakeDataset, EXCITATIONS


def _injected_analyzer():
    cfg = Config(
        sample_name="ps", dimension_selection_method="variance",
        n_important_dimensions=5, perturbation_method="standard_deviation",
        perturbation_magnitudes=[10, 20], perturbation_directions=["bidirectional"],
        normalization_method="variance", n_bands_to_select=4,
        use_diversity_constraint=False, device="cpu",
    )
    a = Analyzer(cfg)
    torch.manual_seed(7)
    latent = torch.randn(2, 3, 2, 2, 2)
    model = FakeHSIModel(latent_numel=3 * 2 * 2 * 2)
    a._model = model
    a._dataset = FakeDataset()
    a._baseline_latent = latent
    with torch.no_grad():
        a._baseline_reconstruction = model.decode(latent)
    return a


def test_select_matches_legacy_private_sequence():
    a = _injected_analyzer()
    # legacy sequence (what fit() did inline)
    a._select_important_dimensions()
    a._compute_influence_scores()
    a._normalize_influences()
    legacy_bands = a._select_top_bands()

    b = _injected_analyzer()
    result = b.select(b.config)
    assert [ (x.excitation_nm, x.emission_band_index, x.influence_score) for x in result.selected_bands ] \
        == [ (x.excitation_nm, x.emission_band_index, x.influence_score) for x in legacy_bands ]


def test_select_is_rerunnable_without_reprepare():
    a = _injected_analyzer()
    r1 = a.select(a.config)
    a.config.n_bands_to_select = 2
    r2 = a.select(a.config)
    assert len(r1.selected_bands) == 4
    assert len(r2.selected_bands) == 2  # no re-prepare needed
```

- [ ] **Step 2: Run — verify it fails**

Run: `pytest tests/test_analyzer_prepare_select.py -q`
Expected: FAIL — `AttributeError: 'Analyzer' object has no attribute 'select'`.

- [ ] **Step 3: Implement `prepare`, `select`, rewrite `fit`**

In `Analyzer.__init__`, add `self._is_prepared = False`. Add an `is_prepared` property mirroring `is_fitted`. Replace the body of `fit()` and add the two methods:

```python
    def prepare(self, data: SpectraData) -> "Analyzer":
        """Slow, one-time setup: load data, load-or-train the model, set up baseline."""
        logger.info(f"Preparing analyzer for {self._config.sample_name}")
        self._load_data(data)
        self._load_or_train_model()
        self._setup_baseline()
        self._is_prepared = True
        return self

    def select(self, config: Optional[Config] = None) -> WavelengthResult:
        """Fast, re-runnable selection on the prepared model+baseline.

        Uses the analysis/perturbation/normalization/selection fields of ``config``
        (defaults to the analyzer's config); does NOT retrain.
        """
        if not self._is_prepared:
            raise RuntimeError("Analyzer must be prepared before select(); call prepare(data) or fit(data).")
        if config is not None:
            self._config = config

        self._select_important_dimensions()
        self._compute_influence_scores()
        if self._config.normalization_method != "none":
            self._normalize_influences()
        selected_bands = self._select_top_bands()

        total_available = sum(
            self._model.emission_bands[ex] for ex in self._model.excitation_wavelengths
        )
        self._result = WavelengthResult(
            sample_name=self._config.sample_name,
            selected_bands=selected_bands,
            metrics=AnalysisMetrics.from_bands(selected_bands, total_available),
            config_snapshot=self._config.to_dict(),
            method_summary={
                "dimension_selection": self._config.dimension_selection_method,
                "perturbation": self._config.perturbation_method,
                "normalization": self._config.normalization_method,
            },
        )
        logger.info(f"Selection complete: {len(selected_bands)} bands selected")
        return self._result

    def fit(self, data: SpectraData) -> "Analyzer":
        """Full pipeline (backward-compatible): prepare(data) then select()."""
        self.prepare(data)
        self.select(self._config)
        return self
```

- [ ] **Step 4: Run — verify passes; full suite stays green**

Run: `pytest tests/test_analyzer_prepare_select.py -q && pytest -q -m "not slow and not notebook"`
Expected: new file PASS; full suite ≥ 381 passed (379 prior + 2 new), 4 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_select/analyzer.py tests/test_analyzer_prepare_select.py
git commit -m "feat(spectral_select): split Analyzer into prepare()/select(); fit() = prepare+select"
```

---

## Phase 2 — State + packaging (no UI logic yet)

### Task 2.1: Extend `PipelineState` for steps 9–10

**Files:**
- Modify: `src/mehsi_preprocessor/state.py`
- Test: `tests/mehsi_preprocessor/test_workers.py` (state portion; create dir + `__init__.py`)

- [ ] **Step 1: Write the failing test**

```python
# tests/mehsi_preprocessor/test_workers.py
from mehsi_preprocessor.state import PipelineState, STEP_TRAIN, STEP_SELECT


def test_state_has_train_select_fields_and_invalidation():
    s = PipelineState()
    assert s.analyzer is None and s.training_losses == [] and s.selection_result is None
    s.analyzer = object(); s.selection_result = object()
    s.invalidate_from(STEP_TRAIN)        # leaving train invalidates select
    assert s.selection_result is None
    assert s.analyzer is not None        # train output survives
    s.selection_result = object()
    s.invalidate_from(7)                 # re-doing ROI invalidates train + select
    assert s.analyzer is None and s.selection_result is None
```

- [ ] **Step 2: Run — verify fails**

Run: `mkdir -p tests/mehsi_preprocessor && touch tests/mehsi_preprocessor/__init__.py && pytest tests/mehsi_preprocessor/test_workers.py -q`
Expected: FAIL — `ImportError: cannot import name 'STEP_TRAIN'`.

- [ ] **Step 3: Implement**

In `state.py`: add `STEP_TRAIN = 9`, `STEP_SELECT = 10`; change `STEP_EXPORT`-based loops to use `STEP_SELECT` as the max. Extend `_STEP_ATTRIBUTES`:
```python
    STEP_EXPORT: [],
    STEP_TRAIN: ["analyzer", "training_losses", "model_source"],
    STEP_SELECT: ["selection_result", "selection_config"],
```
In `invalidate_from`, change `range(step + 1, STEP_EXPORT + 1)` → `range(step + 1, STEP_SELECT + 1)`.
In `__init__`, add: `self.analyzer = None`, `self.training_losses: list = []`, `self.model_source = None`, `self.selection_result = None`, `self.selection_config = None`.
In `_default_for`, add `if attr == "training_losses": return []`.

- [ ] **Step 4: Run — verify passes**

Run: `pytest tests/mehsi_preprocessor/test_workers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mehsi_preprocessor/state.py tests/mehsi_preprocessor/__init__.py tests/mehsi_preprocessor/test_workers.py
git commit -m "feat(mehsi): PipelineState fields + invalidation for train/select steps"
```

### Task 2.2: Console entry point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the script table**

After `[project.urls]` add:
```toml
[project.scripts]
spectral-select-gui = "mehsi_preprocessor.app:main"
```

- [ ] **Step 2: Verify**

Run: `pip install -e '.[gui,dev]' >/dev/null && which spectral-select-gui`
Expected: a path to the installed console script.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(packaging): spectral-select-gui console entry point"
```

---

## Phase 3 — Workers (headless-testable threading)

### Task 3.1: `TrainWorker` / `SelectWorker`

**Files:**
- Create: `src/mehsi_preprocessor/workers.py`
- Test: append to `tests/mehsi_preprocessor/test_workers.py`

- [ ] **Step 1: Write the failing test (logic, run synchronously)**

```python
# append to tests/mehsi_preprocessor/test_workers.py
import numpy as np, torch
from mehsi_preprocessor.workers import run_selection_job
from tests.test_analyzer_core_equivalence import FakeHSIModel, FakeDataset
from spectral_select import Analyzer, Config


def _prepared_analyzer():
    cfg = Config(sample_name="w", n_important_dimensions=4, n_bands_to_select=3,
                 perturbation_method="standard_deviation", use_diversity_constraint=False,
                 device="cpu")
    a = Analyzer(cfg)
    torch.manual_seed(1)
    latent = torch.randn(2, 3, 2, 2, 2)
    m = FakeHSIModel(latent_numel=24)
    a._model = m; a._dataset = FakeDataset(); a._baseline_latent = latent
    a._baseline_reconstruction = m.decode(latent); a._is_prepared = True
    return a


def test_run_selection_job_returns_result():
    a = _prepared_analyzer()
    result = run_selection_job(a, a.config)
    assert len(result.selected_bands) == 3
```

- [ ] **Step 2: Run — verify fails**

Run: `pytest tests/mehsi_preprocessor/test_workers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mehsi_preprocessor.workers'`.

- [ ] **Step 3: Implement**

```python
# src/mehsi_preprocessor/workers.py
"""Background workers so training/selection never freeze the UI.

The pure job functions (run_*_job) are import-safe and unit-tested without Qt;
the QThread wrappers add progress/finished/error signals for the GUI.
"""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal


def run_selection_job(analyzer, config):
    """Run selection on an already-prepared analyzer. Returns WavelengthResult."""
    return analyzer.select(config)


class TrainWorker(QThread):
    progress = pyqtSignal(int, int, float)   # epoch, total, loss
    finished_ok = pyqtSignal(object)         # the prepared Analyzer
    failed = pyqtSignal(str)

    def __init__(self, analyzer, data):
        super().__init__()
        self._analyzer = analyzer
        self._data = data

    def run(self):
        try:
            # progress is surfaced via Config-driven training; the analyzer's
            # train path calls train_with_masking with our callback (see step9).
            self._analyzer.prepare(self._data)
            self.finished_ok.emit(self._analyzer)
        except Exception as exc:  # surface to UI, never crash the thread
            self.failed.emit(str(exc))


class SelectWorker(QThread):
    finished_ok = pyqtSignal(object)         # WavelengthResult
    failed = pyqtSignal(str)

    def __init__(self, analyzer, config):
        super().__init__()
        self._analyzer = analyzer
        self._config = config

    def run(self):
        try:
            self.finished_ok.emit(run_selection_job(self._analyzer, self._config))
        except Exception as exc:
            self.failed.emit(str(exc))
```

> Note on training progress: `train_with_masking`'s `progress_callback` is wired in Step 9 by
> setting it on the analyzer's training call. The simplest robust path (used here) is for Step 9 to
> pass a callback into a thin training hook; if finer progress is needed, have `TrainWorker` accept a
> `progress_callback` and thread it through a `prepare(..., progress_callback=…)` overload. Keep
> YAGNI: emit `progress` from the callback if wired, else just show a busy indicator.

- [ ] **Step 4: Run — verify passes**

Run: `pytest tests/mehsi_preprocessor/test_workers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mehsi_preprocessor/workers.py tests/mehsi_preprocessor/test_workers.py
git commit -m "feat(mehsi): TrainWorker/SelectWorker + pure run_selection_job"
```

---

## Phase 4 — Step widgets

> GUI tests run offscreen. Mirror the construction/layout idioms in
> `src/mehsi_preprocessor/steps/step8_export.py` (read it first) — same `QVBoxLayout`, helper labels,
> and `on_enter` refresh pattern.

### Task 4.1: Step 9 — Train

**Files:**
- Create: `src/mehsi_preprocessor/steps/step9_train.py`
- Test: `tests/mehsi_preprocessor/test_steps_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/mehsi_preprocessor/test_steps_smoke.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from mehsi_preprocessor.state import PipelineState

_app = QApplication.instance() or QApplication([])


def test_step9_builds():
    from mehsi_preprocessor.steps.step9_train import Step9Train
    w = Step9Train(PipelineState())
    assert w.step_index == 9 and "Train" in w.title
    w.on_enter()  # must not raise with empty state


def test_step10_builds():
    from mehsi_preprocessor.steps.step10_select import Step10Select
    w = Step10Select(PipelineState())
    assert w.step_index == 10 and "Select" in w.title
    w.on_enter()
```

- [ ] **Step 2: Run — verify fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/mehsi_preprocessor/test_steps_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: ...step9_train`.

- [ ] **Step 3: Implement `Step9Train`**

```python
# src/mehsi_preprocessor/steps/step9_train.py
"""Step 9 — train the autoencoder (or load a pretrained model)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QLabel, QProgressBar, QPushButton,
    QRadioButton, QSpinBox, QVBoxLayout, QDoubleSpinBox,
)

from spectral_select import Analyzer, Config
from mehsi_preprocessor.state import STEP_TRAIN
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.workers import TrainWorker


class Step9Train(AbstractStepWidget):
    @property
    def step_index(self) -> int:
        return 9

    @property
    def title(self) -> str:
        return "Train Autoencoder"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._worker: TrainWorker | None = None
        root = QVBoxLayout(self)

        train_box = QGroupBox("Train new model")
        self._rb_train = QRadioButton("Train a new model"); self._rb_train.setChecked(True)
        form = QFormLayout()
        self._epochs = QSpinBox(); self._epochs.setRange(1, 1000); self._epochs.setValue(50)
        self._lr = QDoubleSpinBox(); self._lr.setDecimals(5); self._lr.setRange(1e-5, 1.0); self._lr.setValue(1e-3)
        form.addRow("Epochs", self._epochs); form.addRow("Learning rate", self._lr)
        tb = QVBoxLayout(); tb.addWidget(self._rb_train); tb.addLayout(form); train_box.setLayout(tb)

        self._rb_load = QRadioButton("Load pretrained .pth")
        self._btn_browse = QPushButton("Browse…"); self._btn_browse.clicked.connect(self._browse)
        self._model_path: Path | None = None

        self._btn_train = QPushButton("Train"); self._btn_train.clicked.connect(self._start)
        self._progress = QProgressBar()
        self._status = QLabel("No model yet.")

        for w in (train_box, self._rb_load, self._btn_browse, self._btn_train, self._progress, self._status):
            root.addWidget(w)
        root.addStretch(1)

    def on_enter(self) -> None:
        ready = self.state.analyzer is not None
        self._status.setText("Model ready ✓ — continue to Step 10." if ready else "No model yet.")

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select model", "", "PyTorch model (*.pth)")
        if path:
            self._model_path = Path(path); self._rb_load.setChecked(True)

    def _build_config(self) -> Config:
        return Config(
            sample_name=(self.state.current_spectra.sample_name if self.state.current_spectra else "gui"),
            training_epochs=self._epochs.value(), training_lr=self._lr.value(),
            model_path=(self._model_path if self._rb_load.isChecked() else None),
            device="cpu",
        )

    def _start(self) -> None:
        data = self.state.current_spectra
        if data is None:
            self._status.setText("No preprocessed data — complete steps 1–7 first."); return
        analyzer = Analyzer(self._build_config())
        self._btn_train.setEnabled(False); self._progress.setRange(0, self._epochs.value())
        self._worker = TrainWorker(analyzer, data)
        self._worker.progress.connect(lambda e, t, loss: (self._progress.setValue(e),
                                       self._status.setText(f"Epoch {e}/{t} — loss {loss:.4f}")))
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _done(self, analyzer) -> None:
        self.state.invalidate_from(STEP_TRAIN)  # clear stale selection
        self.state.analyzer = analyzer
        self.state.model_source = "loaded" if self._model_path else "trained"
        self._btn_train.setEnabled(True); self._status.setText("Model ready ✓ — continue to Step 10.")

    def _error(self, msg: str) -> None:
        self._btn_train.setEnabled(True); self._status.setText(f"Training failed: {msg}")
```

> To surface live per-epoch progress, wire `progress_callback` through training. Minimal approach
> that keeps `prepare()` clean: have `TrainWorker` accept an optional `progress_callback` and add a
> keyword-only `progress_callback=None` to `Analyzer.prepare` that is forwarded to
> `_train_new_model` → `train_with_masking(..., progress_callback=…)`. If deferring, leave the
> progress bar in "busy" mode (`setRange(0, 0)`) and skip the `progress` signal. Pick one in
> execution; do not leave both wired half-way.

- [ ] **Step 4: Run smoke test (Step 9 portion)**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/mehsi_preprocessor/test_steps_smoke.py::test_step9_builds -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mehsi_preprocessor/steps/step9_train.py tests/mehsi_preprocessor/test_steps_smoke.py
git commit -m "feat(mehsi): Step 9 — train/load autoencoder"
```

### Task 4.2: Step 10 — Select

**Files:**
- Create: `src/mehsi_preprocessor/steps/step10_select.py`

- [ ] **Step 1: (smoke test already written in 4.1 covers Step 10)**

- [ ] **Step 2: Run — verify Step 10 import fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/mehsi_preprocessor/test_steps_smoke.py::test_step10_builds -q`
Expected: FAIL — `ModuleNotFoundError: ...step10_select`.

- [ ] **Step 3: Implement `Step10Select`**

```python
# src/mehsi_preprocessor/steps/step10_select.py
"""Step 10 — run Perturbation-Based AE selection on the prepared model."""
from __future__ import annotations

from dataclasses import replace

from PyQt6.QtWidgets import (
    QComboBox, QFormLayout, QLabel, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QProgressBar, QFileDialog,
)

from mehsi_preprocessor.state import STEP_SELECT
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.workers import SelectWorker


class Step10Select(AbstractStepWidget):
    @property
    def step_index(self) -> int:
        return 10

    @property
    def title(self) -> str:
        return "Select Bands"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._worker: SelectWorker | None = None
        root = QVBoxLayout(self)

        form = QFormLayout()
        self._n_bands = QSpinBox(); self._n_bands.setRange(1, 500); self._n_bands.setValue(30)
        self._dim = QComboBox(); self._dim.addItems(["variance", "activation", "pca"])
        self._norm = QComboBox(); self._norm.addItems(["variance", "max_per_excitation", "none"])
        self._div = QComboBox(); self._div.addItems(["mmr", "min_distance", "none"])
        form.addRow("Bands to select", self._n_bands)
        form.addRow("Dimension method", self._dim)
        form.addRow("Normalization", self._norm)
        form.addRow("Diversity", self._div)

        self._btn_run = QPushButton("Run selection"); self._btn_run.clicked.connect(self._start)
        self._progress = QProgressBar(); self._progress.setRange(0, 0); self._progress.hide()
        self._status = QLabel("")
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Rank", "Excitation (nm)", "Emission (nm)", "Influence"])
        self._btn_export = QPushButton("Export results (CSV / JSON / TIFF)")
        self._btn_export.clicked.connect(self._export); self._btn_export.setEnabled(False)

        root.addLayout(form)
        for w in (self._btn_run, self._progress, self._status, self._table, self._btn_export):
            root.addWidget(w)

    def on_enter(self) -> None:
        ready = self.state.analyzer is not None
        self._btn_run.setEnabled(ready)
        self._status.setText("" if ready else "Train or load a model in Step 9 first.")

    def _config(self):
        base = self.state.analyzer.config
        return replace(
            base, n_bands_to_select=self._n_bands.value(),
            dimension_selection_method=self._dim.currentText(),
            normalization_method=self._norm.currentText(),
            use_diversity_constraint=(self._div.currentText() != "none"),
            diversity_method=self._div.currentText(),
        )

    def _start(self) -> None:
        if self.state.analyzer is None:
            return
        self._btn_run.setEnabled(False); self._progress.show()
        self._worker = SelectWorker(self.state.analyzer, self._config())
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _done(self, result) -> None:
        self.state.selection_result = result
        self._progress.hide(); self._btn_run.setEnabled(True); self._btn_export.setEnabled(True)
        self._table.setRowCount(len(result.selected_bands))
        for r, band in enumerate(result.selected_bands):
            for c, val in enumerate((band.rank, f"{band.excitation_nm:.0f}",
                                     f"{band.emission_nm:.1f}", f"{band.influence_score:.3e}")):
                self._table.setItem(r, c, QTableWidgetItem(str(val)))
        self._status.setText(f"Selected {len(result.selected_bands)} bands.")

    def _error(self, msg: str) -> None:
        self._progress.hide(); self._btn_run.setEnabled(True); self._status.setText(f"Selection failed: {msg}")

    def _export(self) -> None:
        if self.state.selection_result is None:
            return
        out = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if out:
            self.state.analyzer.save_results(out)  # ResultsManager writes CSV/JSON/TIFF
            self._status.setText(f"Exported to {out}")
```

> `replace(...)` requires `Config` to be a dataclass (it is). If `Config` is not a dataclass, build a
> fresh `Config(**base.to_dict() | overrides)` instead — verify against `config.py` at execution time.

- [ ] **Step 4: Run — verify passes**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/mehsi_preprocessor/test_steps_smoke.py -q`
Expected: both step smoke tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mehsi_preprocessor/steps/step10_select.py
git commit -m "feat(mehsi): Step 10 — perturbation-based band selection + results/export"
```

---

## Phase 5 — Wire-up + docs

### Task 5.1: Register steps 9–10 in the wizard

**Files:**
- Modify: `src/mehsi_preprocessor/app.py` (`_register_steps`, ~lines 90–109)

- [ ] **Step 1: Add imports + extend the step list**

Add imports next to the others:
```python
        from mehsi_preprocessor.steps.step9_train import Step9Train
        from mehsi_preprocessor.steps.step10_select import Step10Select
```
Append to `step_classes`: `Step9Train, Step10Select`.

- [ ] **Step 2: Verify the app builds with 10 steps (offscreen)**

Run:
```bash
QT_QPA_PLATFORM=offscreen python -c "
from PyQt6.QtWidgets import QApplication; app=QApplication([])
from mehsi_preprocessor.app import PreprocessorWindow
w=PreprocessorWindow(); print('steps:', len(w._steps)); assert len(w._steps)==10
print([s.title for s in w._steps])"
```
Expected: `steps: 10` and the list ending with `Train Autoencoder`, `Select Bands`.

- [ ] **Step 3: Run the full GUI test set + suite**

Run: `QT_QPA_PLATFORM=offscreen pytest -q -m "not slow and not notebook" tests/mehsi_preprocessor && pytest -q -m "not slow and not notebook"`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add src/mehsi_preprocessor/app.py
git commit -m "feat(mehsi): register Train + Select steps in the wizard"
```

### Task 5.2: README "Run the GUI" section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a section** after Quick Start:
```markdown
## Run the GUI (no code)

```bash
pip install -e ".[gui]"
spectral-select-gui          # or: python -m mehsi_preprocessor
```
The wizard walks you through load → preprocess → annotate → **train autoencoder** →
**select bands**, then exports CSV / JSON / TIFF results.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README — Run the GUI section"
```

---

## Self-Review

- **Spec coverage:** §4.1 prepare/select → Task 1.2; §4.2 callback → Task 1.1; §5.1 Step 9 → 4.1;
  §5.2 Step 10 → 4.2; §5.3 workers → Phase 3; §6 state → 2.1; §7 packaging → 2.2 + 5.2; §8 testing →
  tests in each task; §9 phasing → Phases 1–5. Validation step / bundled installer are §11 out-of-scope.
- **Risk:** prepare/select guarded by byte-identity-style characterization test (Task 1.2) before use.
- **Verification:** every task ends with an exact `pytest`/run command + expected output, then a commit.
- **Open verifications flagged for execution:** real signatures of `MaskedHyperspectralDataset` /
  `HyperspectralCAEWithMasking` (Task 1.1) and whether `Config` is a dataclass for `replace()`
  (Task 4.2) — both noted inline with fallbacks.
