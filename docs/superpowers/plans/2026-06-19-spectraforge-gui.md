# SpectraForge GUI ("the Forge") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PyQt6 workbench that defines fluorophores/materials, paints them onto layered canvases, and renders/exports synthetic ME-HSI datasets — a thin shell over the SpectraForge engine.

**Architecture:** All real logic in pure, tested functions (`Scene.paint_map`, `build_scene`, render/export helpers); Qt panels are a thin shell that mutate `ForgeState` and call them. `RenderWorker(QThread)` keeps rendering off the UI thread. Reuses `mehsi_preprocessor` canvas widgets.

**Tech Stack:** Python 3.11, NumPy, PyQt6, matplotlib, pytest. Builds on `spectraforge` (engine) + `spectral_select`.

**Source spec:** `docs/superpowers/specs/2026-06-19-spectraforge-gui-design.md`

**Conventions:** repo root `/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering`; `source .venv/bin/activate`; logic tests `pytest -q tests/spectraforge`; GUI tests `QT_QPA_PLATFORM=offscreen pytest -q tests/spectraforge/gui`.

## File map

| File | Responsibility |
|---|---|
| `src/spectraforge/scene.py` | + `paint_map(material, amount_map)` (engine addition) |
| `src/spectraforge/gui/__init__.py` | gui subpackage marker |
| `src/spectraforge/gui/layer.py` | `Layer` + `build_scene(state)` |
| `src/spectraforge/gui/state.py` | `ForgeState` working document |
| `src/spectraforge/gui/render_ops.py` | `render_state()`, `export_dataset()` (pure helpers) |
| `src/spectraforge/gui/workers.py` | `RenderWorker(QThread)` |
| `src/spectraforge/gui/widgets/spectrum_plot.py` | matplotlib ex/em curve widget |
| `src/spectraforge/gui/panels/*.py` | library / material / canvas / layers / acquire_render panels |
| `src/spectraforge/gui/app.py` | main window + `main()` |
| `tests/spectraforge/gui/test_*.py` | logic + offscreen smoke tests |
| `pyproject.toml` | `spectraforge-gui` console script |

---

## Phase 1 — Pure foundation (engine + state, no Qt)

### Task 1.1: `Scene.paint_map` (per-pixel concentration)

**Files:** Modify `src/spectraforge/scene.py`; Test `tests/spectraforge/test_scene.py`

- [ ] **Step 1: Add failing test** (append to `tests/spectraforge/test_scene.py`)
```python
def test_paint_map_accumulates_per_pixel():
    s = Scene(4, 4)
    amt = np.zeros((4, 4)); amt[0, 0] = 2.0; amt[1, 1] = 0.5
    s.paint_map(Material("m", {"collagen": 3.0}), amt)
    maps = s.resolve()
    assert maps["collagen"][0, 0] == 6.0   # 3.0 * 2.0
    assert maps["collagen"][1, 1] == 1.5
    assert maps["collagen"][2, 2] == 0.0
```
- [ ] **Step 2: Run — FAIL** (`pytest -q tests/spectraforge/test_scene.py::test_paint_map_accumulates_per_pixel`).
- [ ] **Step 3: Implement** (add method to `Scene`)
```python
    def paint_map(self, material: Material, amount_map) -> None:
        """Add per-pixel concentration: concentration[f] += conc * amount_map."""
        amount_map = np.asarray(amount_map, dtype=float)
        for fname, conc in material.recipe.items():
            self._ensure(fname)[:] += conc * amount_map
```
- [ ] **Step 4: Run — PASS** + full `pytest -q tests/spectraforge`.
- [ ] **Step 5: Commit** (`feat(spectraforge): Scene.paint_map per-pixel concentration painting`).

### Task 1.2: `Layer` + `build_scene`

**Files:** Create `src/spectraforge/gui/__init__.py`, `src/spectraforge/gui/layer.py`, `tests/spectraforge/gui/__init__.py`, `tests/spectraforge/gui/test_layer.py`

- [ ] **Step 1: Failing test**
```python
# tests/spectraforge/gui/test_layer.py
import numpy as np
from spectraforge.material import Material
from spectraforge.gui.layer import Layer, build_scene


class _State:  # minimal duck-typed state for the unit test
    def __init__(self, h, w, layers):
        self.height, self.width, self.layers = h, w, layers


def _layer(name, fmap_val, mat, visible=True):
    amt = np.zeros((6, 6)); amt[:3, :3] = fmap_val
    return Layer(name=name, material=mat, amount_map=amt, visible=visible)


def test_build_scene_sums_visible_layers():
    s = _State(6, 6, [
        _layer("a", 1.0, Material("a", {"collagen": 1.0})),
        _layer("b", 2.0, Material("b", {"collagen": 0.5, "NADH": 1.0})),
        _layer("hidden", 5.0, Material("h", {"collagen": 9.0}), visible=False),
    ])
    maps = build_scene(s).resolve()
    assert maps["collagen"][0, 0] == 1.0 * 1.0 + 0.5 * 2.0   # visible a + b, hidden excluded
    assert maps["NADH"][0, 0] == 1.0 * 2.0
    assert maps["collagen"][5, 5] == 0.0
```
- [ ] **Step 2: Run — FAIL** (`QT_QPA_PLATFORM=offscreen` not needed; layer.py has no Qt). `mkdir -p tests/spectraforge/gui && touch tests/spectraforge/gui/__init__.py`.
- [ ] **Step 3: Implement**
```python
# src/spectraforge/gui/__init__.py
"""SpectraForge GUI (the Forge) — workbench painter app."""
```
```python
# src/spectraforge/gui/layer.py
"""Layer model and scene assembly for the Forge GUI (pure, no Qt)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from spectraforge.material import Material
from spectraforge.scene import Scene


@dataclass
class Layer:
    name: str
    material: Material
    amount_map: np.ndarray   # (H, W) per-pixel painted amount
    visible: bool = True


def build_scene(state) -> Scene:
    """Assemble the engine Scene from a state's visible layers (in order)."""
    scene = Scene(state.height, state.width)
    for layer in state.layers:
        if layer.visible:
            scene.paint_map(layer.material, layer.amount_map)
    return scene
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): Layer + build_scene`).

### Task 1.3: `ForgeState`

**Files:** Create `src/spectraforge/gui/state.py`, `tests/spectraforge/gui/test_state.py`

- [ ] **Step 1: Failing test**
```python
# tests/spectraforge/gui/test_state.py
import numpy as np
from spectraforge.material import Material
from spectraforge.gui.state import ForgeState
from spectraforge.gui.layer import build_scene


def test_state_defaults_and_add_layer():
    st = ForgeState(height=8, width=8)
    assert "collagen" in st.library            # built-in library loaded
    assert st.layers == []
    layer = st.add_layer("patch", Material("m", {"collagen": 1.0}))
    assert layer.amount_map.shape == (8, 8)
    assert st.layers[-1] is layer
    assert st.active_layer == 0
    # painting into the layer flows through build_scene
    layer.amount_map[0, 0] = 1.0
    assert build_scene(st).resolve()["collagen"][0, 0] == 1.0


def test_state_has_default_acquisition_and_artifacts():
    st = ForgeState(height=4, width=4)
    assert len(st.acquisition.excitations) >= 1
    assert st.artifacts is not None
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
```python
# src/spectraforge/gui/state.py
"""ForgeState: the working document for the Forge GUI."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from spectraforge.acquisition import AcquisitionConfig
from spectraforge.artifacts import ArtifactConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.library import load_builtin_library
from spectraforge.material import Material
from spectraforge.gui.layer import Layer


def _default_acquisition() -> AcquisitionConfig:
    return AcquisitionConfig(excitations=[340.0, 450.0, 488.0], em_min=360, em_max=700, em_step=5)


@dataclass
class ForgeState:
    height: int
    width: int
    library: dict = field(default_factory=load_builtin_library)
    materials: dict = field(default_factory=dict)          # name -> Material
    layers: list = field(default_factory=list)             # list[Layer]
    acquisition: AcquisitionConfig = field(default_factory=_default_acquisition)
    artifacts: ArtifactConfig = field(default_factory=lambda: ArtifactConfig(
        rayleigh_strength=0.15, rayleigh_fwhm=12, photon_scale=400.0, read_sigma=0.005))
    seed: int = 0
    active_layer: int = -1
    last_render: object = None                             # (SpectraData, GroundTruth) | None

    def add_layer(self, name: str, material: Material) -> Layer:
        layer = Layer(name=name, material=material,
                      amount_map=np.zeros((self.height, self.width), dtype=float))
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        return layer
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): ForgeState working document`).

---

## Phase 2 — Render/export helpers + worker

### Task 2.1: pure render/export helpers

**Files:** Create `src/spectraforge/gui/render_ops.py`, `tests/spectraforge/gui/test_render_ops.py`

- [ ] **Step 1: Failing test**
```python
# tests/spectraforge/gui/test_render_ops.py
from spectral_select.types import SpectraData
from spectraforge.material import Material
from spectraforge.gui.state import ForgeState
from spectraforge.gui.render_ops import render_state, export_dataset


def _state():
    st = ForgeState(height=12, width=12)
    layer = st.add_layer("g", Material("g", {"EGFP": 1.0}))
    layer.amount_map[:, :6] = 1.0
    return st


def test_render_state_returns_spectradata():
    spectra, gt = render_state(_state())
    assert isinstance(spectra, SpectraData)
    assert spectra.n_excitations == 3
    assert "EGFP" in gt.concentration_maps


def test_export_dataset_writes_files(tmp_path):
    spectra, gt = render_state(_state())
    export_dataset(spectra, gt, tmp_path)
    assert (tmp_path / "spectra_unmasked.pkl").exists()
    assert (tmp_path / "groundtruth.npz").exists()
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
```python
# src/spectraforge/gui/render_ops.py
"""Pure render/export helpers shared by the worker and tests (no Qt)."""
from __future__ import annotations
from pathlib import Path
from spectraforge.forward import render
from spectraforge.gui.layer import build_scene


def render_state(state):
    """Build the scene from state and render -> (SpectraData, GroundTruth)."""
    scene = build_scene(state)
    return render(scene, state.library, state.acquisition,
                  artifacts=state.artifacts, seed=state.seed, sample_name="forge")


def export_dataset(spectra, ground_truth, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spectra.to_pickle(out / "spectra_unmasked.pkl")
    ground_truth.save(out)
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): render_state + export_dataset helpers`).

### Task 2.2: `RenderWorker`

**Files:** Create `src/spectraforge/gui/workers.py`, `tests/spectraforge/gui/test_workers.py`

- [ ] **Step 1: Failing test** (drive the pure body synchronously)
```python
# tests/spectraforge/gui/test_workers.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from spectraforge.material import Material
from spectraforge.gui.state import ForgeState
from spectraforge.gui.workers import run_render_job


def test_run_render_job_returns_pair():
    st = ForgeState(height=10, width=10)
    layer = st.add_layer("g", Material("g", {"EGFP": 1.0})); layer.amount_map[:] = 1.0
    spectra, gt = run_render_job(st)
    assert spectra.n_excitations == 3
```
- [ ] **Step 2: Run — FAIL** (`QT_QPA_PLATFORM=offscreen pytest -q tests/spectraforge/gui/test_workers.py`).
- [ ] **Step 3: Implement**
```python
# src/spectraforge/gui/workers.py
"""Background render worker for the Forge GUI."""
from __future__ import annotations
from PyQt6.QtCore import QThread, pyqtSignal
from spectraforge.gui.render_ops import render_state


def run_render_job(state):
    """Pure render entry point (unit-tested without Qt)."""
    return render_state(state)


class RenderWorker(QThread):
    finished_ok = pyqtSignal(object)   # (SpectraData, GroundTruth)
    failed = pyqtSignal(str)

    def __init__(self, state):
        super().__init__()
        self._state = state

    def run(self):
        try:
            self.finished_ok.emit(run_render_job(self._state))
        except Exception as exc:
            self.failed.emit(str(exc))
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): RenderWorker`).

---

## Phase 3 — Spectrum plot widget + Library panel

### Task 3.1: `spectrum_plot` widget
**Files:** Create `src/spectraforge/gui/widgets/__init__.py`, `src/spectraforge/gui/widgets/spectrum_plot.py`, `tests/spectraforge/gui/test_widgets.py`

- [ ] **Step 1: Offscreen smoke test**
```python
# tests/spectraforge/gui/test_widgets.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from spectraforge.fluorophore import Fluorophore

_app = QApplication.instance() or QApplication([])


def test_spectrum_plot_builds_and_plots():
    from spectraforge.gui.widgets.spectrum_plot import SpectrumPlot
    w = SpectrumPlot()
    w.plot_fluorophore(Fluorophore("X", 480, 40, 520, 40))  # must not raise
```
- [ ] **Step 2: Run — FAIL** (module missing).
- [ ] **Step 3: Implement** (matplotlib FigureCanvas; plot ex/em curves over a fixed grid)
```python
# src/spectraforge/gui/widgets/__init__.py
"""Forge GUI widgets."""
```
```python
# src/spectraforge/gui/widgets/spectrum_plot.py
"""Matplotlib widget that plots fluorophore excitation/emission curves."""
from __future__ import annotations
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class SpectrumPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(4, 2.4))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        lay = QVBoxLayout(self); lay.addWidget(self._canvas)
        self._grid = np.arange(250, 750, 2.0)

    def plot_fluorophore(self, fluor) -> None:
        self._ax.clear()
        self._ax.plot(self._grid, fluor.excitation(self._grid), label="excitation")
        em = fluor.emission(self._grid); em = em / em.max() if em.max() > 0 else em
        self._ax.plot(self._grid, em, label="emission")
        self._ax.set_xlabel("wavelength (nm)"); self._ax.legend(fontsize=7)
        self._ax.set_title(fluor.name, fontsize=9)
        self._canvas.draw_idle()

    def plot_material(self, material, library, excitation: float) -> None:
        self._ax.clear()
        total = None
        for fname, conc in material.recipe.items():
            f = library[fname]
            contrib = conc * f.extinction * f.quantum_yield * float(f.excitation(excitation)) * f.emission(self._grid)
            total = contrib if total is None else total + contrib
        if total is not None and total.max() > 0:
            self._ax.plot(self._grid, total / total.max(), label=f"mix @ {excitation:.0f}nm")
            self._ax.legend(fontsize=7)
        self._ax.set_xlabel("wavelength (nm)"); self._ax.set_title(material.name, fontsize=9)
        self._canvas.draw_idle()
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): SpectrumPlot widget`).

### Task 3.2: Library panel
**Files:** Create `src/spectraforge/gui/panels/__init__.py`, `src/spectraforge/gui/panels/library_panel.py`; add test to `tests/spectraforge/gui/test_panels.py`

- [ ] **Step 1: Smoke test**
```python
# tests/spectraforge/gui/test_panels.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from spectraforge.gui.state import ForgeState

_app = QApplication.instance() or QApplication([])


def test_library_panel_builds_and_defines():
    from spectraforge.gui.panels.library_panel import LibraryPanel
    st = ForgeState(height=8, width=8)
    p = LibraryPanel(st)
    n0 = len(st.library)
    p.define_fluorophore("MyDye", 500, 40, 560, 50, 0.7, 0.9)
    assert len(st.library) == n0 + 1 and "MyDye" in st.library
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `LibraryPanel(QWidget)`: a `QListWidget` of `state.library` names; on selection, `SpectrumPlot.plot_fluorophore`; a form (name + 6 spin boxes) whose "Add" calls `define_fluorophore(...)` which constructs a `Fluorophore`, inserts into `state.library`, refreshes the list. Keep `panels/__init__.py` a one-line docstring module.
```python
# src/spectraforge/gui/panels/library_panel.py  (essential logic; layout mirrors mehsi panels)
from __future__ import annotations
from PyQt6.QtWidgets import (QDoubleSpinBox, QFormLayout, QHBoxLayout, QLineEdit,
                             QListWidget, QPushButton, QVBoxLayout, QWidget)
from spectraforge.fluorophore import Fluorophore
from spectraforge.gui.widgets.spectrum_plot import SpectrumPlot


class LibraryPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._list = QListWidget(); self._plot = SpectrumPlot()
        self._list.currentTextChanged.connect(self._on_select)
        self._name = QLineEdit()
        self._spins = {k: QDoubleSpinBox() for k in
                       ("ex_peak", "ex_fwhm", "em_peak", "em_fwhm", "qy", "eps")}
        for k, sp in self._spins.items():
            sp.setRange(0, 2000); sp.setDecimals(2)
        self._spins["qy"].setRange(0, 1); self._spins["eps"].setRange(0, 100)
        form = QFormLayout()
        form.addRow("Name", self._name)
        for label, key in (("Ex peak", "ex_peak"), ("Ex FWHM", "ex_fwhm"),
                           ("Em peak", "em_peak"), ("Em FWHM", "em_fwhm"),
                           ("Quantum yield", "qy"), ("Extinction", "eps")):
            form.addRow(label, self._spins[key])
        add = QPushButton("Add fluorophore"); add.clicked.connect(self._on_add)
        root = QVBoxLayout(self)
        root.addWidget(self._list); root.addWidget(self._plot); root.addLayout(form); root.addWidget(add)
        self._refresh()

    def _refresh(self):
        self._list.clear(); self._list.addItems(sorted(self.state.library))

    def _on_select(self, name):
        if name in self.state.library:
            self._plot.plot_fluorophore(self.state.library[name])

    def _on_add(self):
        s = self._spins
        self.define_fluorophore(self._name.text() or "dye",
                                s["ex_peak"].value(), s["ex_fwhm"].value(),
                                s["em_peak"].value(), s["em_fwhm"].value(),
                                s["qy"].value(), s["eps"].value())

    def define_fluorophore(self, name, ex_peak, ex_fwhm, em_peak, em_fwhm, qy, eps):
        self.state.library[name] = Fluorophore(name, ex_peak, ex_fwhm, em_peak, em_fwhm, qy, eps)
        self._refresh()
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): Library panel`).

---

## Phase 4 — Material composer panel

### Task 4.1: Material panel
**Files:** Create `src/spectraforge/gui/panels/material_panel.py`; add to `tests/spectraforge/gui/test_panels.py`

- [ ] **Step 1: Smoke test**
```python
def test_material_panel_composes():
    from spectraforge.gui.panels.material_panel import MaterialPanel
    st = ForgeState(height=8, width=8)
    p = MaterialPanel(st)
    p.compose_material("tissue", {"collagen": 1.0, "NADH": 0.3})
    assert "tissue" in st.materials
    assert st.materials["tissue"].recipe["collagen"] == 1.0
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `MaterialPanel(QWidget)`: a name field + a table/rows of (fluorophore combo, concentration spin) with add-row; "Compose" calls `compose_material(name, recipe)` → builds `Material`, stores in `state.materials`, plots the mixed spectrum via `SpectrumPlot.plot_material`; a list of composed materials.
```python
# src/spectraforge/gui/panels/material_panel.py  (essential logic)
from __future__ import annotations
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit,
                             QListWidget, QPushButton, QVBoxLayout, QWidget)
from spectraforge.material import Material
from spectraforge.gui.widgets.spectrum_plot import SpectrumPlot


class MaterialPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._name = QLineEdit()
        self._rows: list[tuple[QComboBox, QDoubleSpinBox]] = []
        self._rows_box = QVBoxLayout()
        self._plot = SpectrumPlot()
        self._materials_list = QListWidget()
        add_row = QPushButton("+ fluorophore"); add_row.clicked.connect(self._add_row)
        compose = QPushButton("Compose material"); compose.clicked.connect(self._on_compose)
        root = QVBoxLayout(self)
        root.addWidget(self._name); root.addLayout(self._rows_box)
        root.addWidget(add_row); root.addWidget(compose)
        root.addWidget(self._plot); root.addWidget(self._materials_list)
        self._add_row()

    def _add_row(self):
        combo = QComboBox(); combo.addItems(sorted(self.state.library))
        spin = QDoubleSpinBox(); spin.setRange(0, 100); spin.setValue(1.0)
        self._rows.append((combo, spin))
        h = QHBoxLayout(); h.addWidget(combo); h.addWidget(spin)
        self._rows_box.addLayout(h)

    def _on_compose(self):
        recipe = {c.currentText(): s.value() for c, s in self._rows if s.value() > 0}
        self.compose_material(self._name.text() or "material", recipe)

    def compose_material(self, name, recipe):
        mat = Material(name, dict(recipe))
        self.state.materials[name] = mat
        self._materials_list.clear(); self._materials_list.addItems(sorted(self.state.materials))
        ex = self.state.acquisition.excitations[0]
        self._plot.plot_material(mat, self.state.library, ex)
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): Material composer panel`).

---

## Phase 5 — Canvas + Layers panels

### Task 5.1: Layers panel
**Files:** Create `src/spectraforge/gui/panels/layers_panel.py`; add to `tests/spectraforge/gui/test_panels.py`

- [ ] **Step 1: Smoke test**
```python
def test_layers_panel_add_toggle():
    from spectraforge.gui.panels.layers_panel import LayersPanel
    from spectraforge.material import Material
    st = ForgeState(height=8, width=8); st.materials["m"] = Material("m", {"collagen": 1.0})
    p = LayersPanel(st)
    p.add_layer_with_material("m")
    assert len(st.layers) == 1 and st.layers[0].material.name == "m"
    p.set_visible(0, False)
    assert st.layers[0].visible is False
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `LayersPanel(QWidget)`: a `QListWidget` of layers (name + checkbox visibility); buttons add (material combo → `state.add_layer`), remove, up/down; `set_visible(i, bool)`; selection sets `state.active_layer`. Emits a `changed` signal so the canvas repaints. Provide `add_layer_with_material(material_name)` and `set_visible(i, v)` as the tested methods.
```python
# src/spectraforge/gui/panels/layers_panel.py  (essential logic)
from __future__ import annotations
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QListWidget, QPushButton,
                             QVBoxLayout, QWidget)


class LayersPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_select)
        self._combo = QComboBox()
        add = QPushButton("Add layer"); add.clicked.connect(self._on_add)
        h = QHBoxLayout(); h.addWidget(self._combo); h.addWidget(add)
        root = QVBoxLayout(self); root.addWidget(self._list); root.addLayout(h)
        self._refresh_combo(); self._refresh_list()

    def _refresh_combo(self):
        self._combo.clear(); self._combo.addItems(sorted(self.state.materials))

    def _refresh_list(self):
        self._list.clear()
        self._list.addItems([f"{'☑' if l.visible else '☐'} {l.name}" for l in self.state.layers])

    def _on_add(self):
        if self._combo.currentText():
            self.add_layer_with_material(self._combo.currentText())

    def _on_select(self, row):
        if 0 <= row < len(self.state.layers):
            self.state.active_layer = row

    def add_layer_with_material(self, material_name):
        mat = self.state.materials[material_name]
        self.state.add_layer(material_name, mat)
        self._refresh_list(); self.changed.emit()

    def set_visible(self, i, visible):
        self.state.layers[i].visible = visible
        self._refresh_list(); self.changed.emit()
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): Layers panel`).

### Task 5.2: Canvas panel
**Files:** Create `src/spectraforge/gui/panels/canvas_panel.py`; add to `tests/spectraforge/gui/test_panels.py`

- [ ] **Step 1: Smoke test** (verify painting writes into the active layer's amount_map)
```python
def test_canvas_panel_paint_rect_writes_active_layer():
    from spectraforge.gui.panels.canvas_panel import CanvasPanel
    from spectraforge.material import Material
    st = ForgeState(height=10, width=10); st.materials["m"] = Material("m", {"collagen": 1.0})
    st.add_layer("L", st.materials["m"])
    p = CanvasPanel(st)
    p.paint_rect(2, 6, 2, 6, amount=1.0)        # programmatic paint into active layer
    assert st.layers[0].amount_map[3, 3] == 1.0
    assert st.layers[0].amount_map[0, 0] == 0.0
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `CanvasPanel(QWidget)`: a tool selector (rect/circle/brush) + brush radius; an `ImageCanvas`-style preview showing a false-color composite of visible layers (each material → a stable color, weighted by amount). Mouse drag (via a `RectSelector` for rect; brush via the canvas) writes into `state.layers[state.active_layer].amount_map`. Provide programmatic `paint_rect(r0,r1,c0,c1,amount)` / `paint_circle(cy,cx,radius,amount)` (used by the test and the mouse handlers) and `refresh()` to redraw the composite.
```python
# src/spectraforge/gui/panels/canvas_panel.py  (essential logic; reuse mehsi ImageCanvas for display)
from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from mehsi_preprocessor.widgets.image_canvas import ImageCanvas


class CanvasPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._canvas = ImageCanvas(self)
        root = QVBoxLayout(self); root.addWidget(self._canvas)
        self.refresh()

    def _active(self):
        i = self.state.active_layer
        return self.state.layers[i] if 0 <= i < len(self.state.layers) else None

    def paint_rect(self, r0, r1, c0, c1, amount=1.0):
        layer = self._active()
        if layer is not None:
            layer.amount_map[r0:r1, c0:c1] += amount
            self.refresh()

    def paint_circle(self, cy, cx, radius, amount=1.0):
        layer = self._active()
        if layer is None:
            return
        yy, xx = np.ogrid[: self.state.height, : self.state.width]
        layer.amount_map[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] += amount
        self.refresh()

    def composite(self):
        """False-color RGB composite of visible layers (stable per-material hue)."""
        rgb = np.zeros((self.state.height, self.state.width, 3))
        for n, layer in enumerate(l for l in self.state.layers if l.visible):
            color = np.array([(n * 53) % 255, (n * 101) % 255, (n * 151) % 255]) / 255.0
            a = layer.amount_map
            norm = a / a.max() if a.max() > 0 else a
            rgb += norm[:, :, None] * color[None, None, :]
        return np.clip(rgb, 0, 1)

    def refresh(self):
        self._canvas.show_image(self.composite())
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): Canvas panel`).

---

## Phase 6 — Acquire/Render/Export panel

### Task 6.1: acquire-render panel
**Files:** Create `src/spectraforge/gui/panels/acquire_render_panel.py`; add to `tests/spectraforge/gui/test_panels.py`

- [ ] **Step 1: Smoke test** (render via the worker logic + export)
```python
def test_acquire_render_panel_render_and_export(tmp_path):
    from spectraforge.gui.panels.acquire_render_panel import AcquireRenderPanel
    from spectraforge.material import Material
    st = ForgeState(height=10, width=10); st.materials["m"] = Material("m", {"EGFP": 1.0})
    layer = st.add_layer("L", st.materials["m"]); layer.amount_map[:] = 1.0
    p = AcquireRenderPanel(st)
    p.render_now()                       # synchronous render (no thread) for the test
    assert st.last_render is not None
    p.export_to(tmp_path)
    assert (tmp_path / "spectra_unmasked.pkl").exists()
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `AcquireRenderPanel(QWidget)`: excitation list editor (line edit "340,450,488"), emission grid spin boxes, artifact toggles bound to `state.acquisition`/`state.artifacts`; a **Render** button starting `RenderWorker` (and a synchronous `render_now()` used by the test/headless); a slice preview (`ImageCanvas` + excitation/emission sliders reading `state.last_render`); an **Export** button → `QFileDialog` → `export_to(dir)`.
```python
# src/spectraforge/gui/panels/acquire_render_panel.py  (essential logic)
from __future__ import annotations
from pathlib import Path
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from spectraforge.gui.render_ops import export_dataset, render_state
from spectraforge.gui.workers import RenderWorker


class AcquireRenderPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker = None
        self._ex = QLineEdit(",".join(f"{e:.0f}" for e in state.acquisition.excitations))
        self._status = QLabel("")
        render_btn = QPushButton("Render"); render_btn.clicked.connect(self._on_render)
        export_btn = QPushButton("Export pkl + ground truth"); export_btn.clicked.connect(self._on_export)
        root = QVBoxLayout(self)
        h = QHBoxLayout(); h.addWidget(QLabel("Excitations")); h.addWidget(self._ex)
        root.addLayout(h)
        for b in (render_btn, export_btn, self._status):
            root.addWidget(b)

    def _apply_excitations(self):
        try:
            self.state.acquisition.excitations = [float(x) for x in self._ex.text().split(",") if x.strip()]
        except ValueError:
            pass

    def render_now(self):
        self._apply_excitations()
        self.state.last_render = render_state(self.state)
        self._status.setText("Rendered.")

    def _on_render(self):
        self._apply_excitations()
        self._worker = RenderWorker(self.state)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(lambda m: self._status.setText(f"Render failed: {m}"))
        self._status.setText("Rendering…"); self._worker.start()

    def _on_done(self, result):
        self.state.last_render = result
        self._status.setText(f"Rendered {result[0].n_excitations} excitations.")

    def export_to(self, out_dir):
        if self.state.last_render is None:
            return
        spectra, gt = self.state.last_render
        export_dataset(spectra, gt, out_dir)
        self._status.setText(f"Exported to {out_dir}")

    def _on_export(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.export_to(Path(d))
```
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge.gui): Acquire/Render/Export panel`).

---

## Phase 7 — Main window + packaging

### Task 7.1: `ForgeWindow` + entry point
**Files:** Create `src/spectraforge/gui/app.py`; Modify `pyproject.toml`, `README.md`; add to `tests/spectraforge/gui/test_panels.py`

- [ ] **Step 1: Smoke test**
```python
def test_forge_window_builds():
    from spectraforge.gui.app import ForgeWindow
    w = ForgeWindow()
    assert w._state is not None
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `ForgeWindow(QMainWindow)`: create `ForgeState(height=64, width=64)`; instantiate the panels; place Library+Material in a left `QDockWidget`, Layers in a right dock, Canvas central, Acquire/Render/Export in a bottom dock; wire `LayersPanel.changed` → `CanvasPanel.refresh`. `main()` runs `QApplication`.
```python
# src/spectraforge/gui/app.py  (assembly)
from __future__ import annotations
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QMainWindow, QTabWidget
from spectraforge.gui.state import ForgeState
from spectraforge.gui.panels.library_panel import LibraryPanel
from spectraforge.gui.panels.material_panel import MaterialPanel
from spectraforge.gui.panels.canvas_panel import CanvasPanel
from spectraforge.gui.panels.layers_panel import LayersPanel
from spectraforge.gui.panels.acquire_render_panel import AcquireRenderPanel


class ForgeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraForge")
        self.resize(1500, 950)
        self._state = ForgeState(height=64, width=64)
        self._canvas = CanvasPanel(self._state)
        self.setCentralWidget(self._canvas)

        left = QTabWidget()
        left.addTab(LibraryPanel(self._state), "Library")
        self._material = MaterialPanel(self._state)
        left.addTab(self._material, "Materials")
        self._add_dock("Library / Materials", left, Qt.DockWidgetArea.LeftDockWidgetArea)

        self._layers = LayersPanel(self._state)
        self._layers.changed.connect(self._canvas.refresh)
        self._add_dock("Layers", self._layers, Qt.DockWidgetArea.RightDockWidgetArea)

        self._add_dock("Acquire / Render / Export", AcquireRenderPanel(self._state),
                       Qt.DockWidgetArea.BottomDockWidgetArea)

    def _add_dock(self, title, widget, area):
        dock = QDockWidget(title, self); dock.setWidget(widget)
        self.addDockWidget(area, dock)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpectraForge")
    win = ForgeWindow(); win.show()
    app.exec()


if __name__ == "__main__":
    main()
```
- [ ] **Step 4: Run smoke** (`QT_QPA_PLATFORM=offscreen pytest -q tests/spectraforge/gui`). All panel + window smoke tests PASS.
- [ ] **Step 5: Packaging + README + commit**
In `pyproject.toml` `[project.scripts]` add `spectraforge-gui = "spectraforge.gui.app:main"`; `pip install -e '.[gui,dev]'`; confirm `QT_QPA_PLATFORM=offscreen python -c "from spectraforge.gui.app import ForgeWindow; from PyQt6.QtWidgets import QApplication; QApplication([]); ForgeWindow(); print('ok')"`. Add a "Run the Forge" line to README. Commit (`feat(spectraforge.gui): main window + spectraforge-gui entry`).

---

## Self-Review

- **Spec coverage:** §3 architecture/modules → all tasks (one file per task); `Scene.paint_map` → 1.1; layer model/`build_scene` → 1.2; `ForgeState` → 1.3; render/export → 2.1; worker → 2.2; spectrum plot → 3.1; Library → 3.2; Materials → 4.1; Layers → 5.1; Canvas → 5.2; Acquire/Render/Export → 6.1; main window/packaging → 7.1. §6 testing → logic tests (1.x, 2.x) + offscreen smoke (3–7). §8 packaging → 7.1.
- **Type consistency:** `Scene.paint_map(material, amount_map)`; `Layer{name,material,amount_map,visible}`; `build_scene(state)`; `ForgeState.{height,width,library,materials,layers,acquisition,artifacts,seed,active_layer,last_render,add_layer}`; `render_state(state)`/`export_dataset(spectra,gt,out)`; `run_render_job(state)`/`RenderWorker`; panels expose tested methods (`define_fluorophore`, `compose_material`, `add_layer_with_material`/`set_visible`, `paint_rect`/`paint_circle`, `render_now`/`export_to`). Consistent across tasks.
- **Verification flagged for execution:** confirm `mehsi_preprocessor.widgets.image_canvas.ImageCanvas.show_image` accepts an RGB float array (Task 5.2) — adjust composite dtype/format to its real signature if needed.
