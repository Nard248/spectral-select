# SpectraForge Engine (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SpectraForge forward-model engine — fluorophores → materials → painted scene → physically-grounded ME-HSI `SpectraData` + ground truth — fully tested, no GUI.

**Architecture:** Pure NumPy. Dilute-regime linear superposition of fluorophore excitation–emission matrices. Each module has one responsibility; `forward.render()` is the only assembler. Renders to `spectral_select.types.SpectraData` for zero-glue binding.

**Tech Stack:** Python 3.11, NumPy, matplotlib (polygon paths), pytest. Depends on `spectral_select` (same repo).

**Source spec:** `docs/superpowers/specs/2026-06-19-spectraforge-design.md`

**Conventions:** repo root `/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering`; `source .venv/bin/activate`; run tests `pytest -q tests/spectraforge`.

## File map

| File | Responsibility |
|---|---|
| `src/spectraforge/__init__.py` | public API exports |
| `src/spectraforge/fluorophore.py` | `Fluorophore` + excitation/emission profiles |
| `src/spectraforge/data/fluorophores.json` | starter library (~12) |
| `src/spectraforge/library.py` | `load_builtin_library()` |
| `src/spectraforge/material.py` | `Material` (fluorophore recipe) |
| `src/spectraforge/scene.py` | `Scene` painting → concentration maps |
| `src/spectraforge/acquisition.py` | `AcquisitionConfig` |
| `src/spectraforge/artifacts.py` | `ArtifactConfig` + scatter/noise |
| `src/spectraforge/groundtruth.py` | `GroundTruth` container + save |
| `src/spectraforge/forward.py` | `render()` → `(SpectraData, GroundTruth)` |
| `src/spectraforge/demo.py` | `main()` demo dataset generator (console script) |
| `tests/spectraforge/test_*.py` | one test module per source module |
| `pyproject.toml` | register `spectraforge*`, `spectraforge-demo` script, CI cov |

---

## Phase 1 — Fluorophore + library

### Task 1.1: `Fluorophore` spectral profiles

**Files:** Create `src/spectraforge/__init__.py`, `src/spectraforge/fluorophore.py`, `tests/spectraforge/__init__.py`, `tests/spectraforge/test_fluorophore.py`

- [ ] **Step 1: Write failing test**
```python
# tests/spectraforge/test_fluorophore.py
import numpy as np
from spectraforge.fluorophore import Fluorophore

def test_excitation_peaks_at_one():
    f = Fluorophore("X", ex_peak_nm=480, ex_fwhm_nm=40, em_peak_nm=520, em_fwhm_nm=40)
    assert np.isclose(f.excitation(480.0), 1.0)
    assert f.excitation(480.0) > f.excitation(520.0)

def test_emission_unit_area_on_grid():
    f = Fluorophore("X", ex_peak_nm=480, ex_fwhm_nm=40, em_peak_nm=520, em_fwhm_nm=40)
    grid = np.arange(400, 700, 2.0)
    em = f.emission(grid)
    assert np.isclose(em.sum(), 1.0)
    assert grid[int(np.argmax(em))] == 520.0

def test_fwhm_to_sigma_width():
    f = Fluorophore("X", ex_peak_nm=500, ex_fwhm_nm=50, em_peak_nm=560, em_fwhm_nm=50)
    half = 500 + 25  # at +FWHM/2 the excitation should be ~0.5
    assert np.isclose(f.excitation(half), 0.5, atol=1e-3)
```

- [ ] **Step 2: Run — expect ModuleNotFoundError**
Run: `pytest -q tests/spectraforge/test_fluorophore.py`  Expected: FAIL (no module `spectraforge`).

- [ ] **Step 3: Implement**
```python
# src/spectraforge/fluorophore.py
"""Fluorophore: parametric Gaussian excitation/emission spectra (dilute-regime model)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493  # 1/(2*sqrt(2*ln2))


@dataclass(frozen=True)
class Fluorophore:
    name: str
    ex_peak_nm: float
    ex_fwhm_nm: float
    em_peak_nm: float
    em_fwhm_nm: float
    quantum_yield: float = 0.5      # Phi
    extinction: float = 1.0         # relative epsilon (unitless in v1)

    def excitation(self, wl):
        """Relative absorption probability at wl, normalized to peak 1."""
        wl = np.asarray(wl, dtype=float)
        sigma = self.ex_fwhm_nm * _FWHM_TO_SIGMA
        return np.exp(-0.5 * ((wl - self.ex_peak_nm) / sigma) ** 2)

    def emission(self, wl):
        """Emission shape over grid wl, normalized to unit sum (area) on that grid."""
        wl = np.asarray(wl, dtype=float)
        sigma = self.em_fwhm_nm * _FWHM_TO_SIGMA
        g = np.exp(-0.5 * ((wl - self.em_peak_nm) / sigma) ** 2)
        total = g.sum()
        return g / total if total > 0 else g
```
```python
# src/spectraforge/__init__.py
"""SpectraForge — synthetic ME-HSI dataset generator."""
from spectraforge.fluorophore import Fluorophore

__all__ = ["Fluorophore"]
```
Create empty `tests/spectraforge/__init__.py`.

- [ ] **Step 4: Run — expect PASS** (`pytest -q tests/spectraforge/test_fluorophore.py`). Then register package: in `pyproject.toml` `[tool.setuptools.packages.find].include` add `"spectraforge*"`; run `pip install -e '.[dev]'`.

- [ ] **Step 5: Commit**
```bash
git add src/spectraforge/__init__.py src/spectraforge/fluorophore.py tests/spectraforge pyproject.toml
git commit -m "feat(spectraforge): Fluorophore parametric excitation/emission profiles"
```

### Task 1.2: starter library

**Files:** Create `src/spectraforge/data/fluorophores.json`, `src/spectraforge/library.py`, `tests/spectraforge/test_library.py`

- [ ] **Step 1: Write failing test**
```python
# tests/spectraforge/test_library.py
from spectraforge.library import load_builtin_library
from spectraforge.fluorophore import Fluorophore

def test_library_loads_known_fluorophores():
    lib = load_builtin_library()
    assert "collagen" in lib and "EGFP" in lib and "fluorescein" in lib
    assert isinstance(lib["collagen"], Fluorophore)
    assert lib["collagen"].em_peak_nm == 390
    assert len(lib) >= 10
```

- [ ] **Step 2: Run — expect FAIL** (no `library`).

- [ ] **Step 3: Implement**
Create `src/spectraforge/data/fluorophores.json`:
```json
[
  {"name": "tryptophan", "ex_peak_nm": 280, "ex_fwhm_nm": 35, "em_peak_nm": 350, "em_fwhm_nm": 60, "quantum_yield": 0.13, "extinction": 0.6},
  {"name": "collagen", "ex_peak_nm": 330, "ex_fwhm_nm": 50, "em_peak_nm": 390, "em_fwhm_nm": 70, "quantum_yield": 0.3, "extinction": 0.7},
  {"name": "elastin", "ex_peak_nm": 350, "ex_fwhm_nm": 55, "em_peak_nm": 420, "em_fwhm_nm": 75, "quantum_yield": 0.3, "extinction": 0.7},
  {"name": "NADH", "ex_peak_nm": 340, "ex_fwhm_nm": 50, "em_peak_nm": 460, "em_fwhm_nm": 90, "quantum_yield": 0.4, "extinction": 0.6},
  {"name": "FAD", "ex_peak_nm": 450, "ex_fwhm_nm": 50, "em_peak_nm": 535, "em_fwhm_nm": 60, "quantum_yield": 0.3, "extinction": 0.7},
  {"name": "DAPI", "ex_peak_nm": 358, "ex_fwhm_nm": 45, "em_peak_nm": 461, "em_fwhm_nm": 60, "quantum_yield": 0.5, "extinction": 0.8},
  {"name": "EGFP", "ex_peak_nm": 488, "ex_fwhm_nm": 40, "em_peak_nm": 507, "em_fwhm_nm": 40, "quantum_yield": 0.6, "extinction": 0.9},
  {"name": "fluorescein", "ex_peak_nm": 495, "ex_fwhm_nm": 40, "em_peak_nm": 519, "em_fwhm_nm": 40, "quantum_yield": 0.92, "extinction": 1.0},
  {"name": "rhodamine", "ex_peak_nm": 540, "ex_fwhm_nm": 40, "em_peak_nm": 565, "em_fwhm_nm": 45, "quantum_yield": 0.7, "extinction": 0.9},
  {"name": "TexasRed", "ex_peak_nm": 595, "ex_fwhm_nm": 40, "em_peak_nm": 615, "em_fwhm_nm": 45, "quantum_yield": 0.6, "extinction": 0.85},
  {"name": "mCherry", "ex_peak_nm": 587, "ex_fwhm_nm": 45, "em_peak_nm": 610, "em_fwhm_nm": 50, "quantum_yield": 0.22, "extinction": 0.8},
  {"name": "Cy5", "ex_peak_nm": 649, "ex_fwhm_nm": 40, "em_peak_nm": 670, "em_fwhm_nm": 45, "quantum_yield": 0.28, "extinction": 0.9}
]
```
```python
# src/spectraforge/library.py
"""Built-in fluorophore starter library."""
from __future__ import annotations
import json
from pathlib import Path
from spectraforge.fluorophore import Fluorophore

_DATA = Path(__file__).parent / "data" / "fluorophores.json"


def load_builtin_library() -> dict[str, Fluorophore]:
    records = json.loads(_DATA.read_text())
    return {r["name"]: Fluorophore(**r) for r in records}
```
Add `load_builtin_library` to `__init__.py` exports.

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** (`feat(spectraforge): starter fluorophore library`).

---

## Phase 2 — Material

### Task 2.1: `Material` recipe
**Files:** Create `src/spectraforge/material.py`, `tests/spectraforge/test_material.py`

- [ ] **Step 1: Failing test**
```python
# tests/spectraforge/test_material.py
from spectraforge.material import Material

def test_material_recipe():
    m = Material("tissue", {"collagen": 1.0, "NADH": 0.4})
    assert m.recipe["collagen"] == 1.0
    assert set(m.fluorophores()) == {"collagen", "NADH"}
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
```python
# src/spectraforge/material.py
"""Material = a named recipe of fluorophores with relative concentrations (the 'brush')."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Material:
    name: str
    recipe: dict[str, float] = field(default_factory=dict)  # fluorophore_name -> concentration

    def fluorophores(self) -> list[str]:
        return list(self.recipe.keys())
```
Export `Material`.
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge): Material recipe`).

---

## Phase 3 — Scene

### Task 3.1: `Scene` painting → concentration maps
**Files:** Create `src/spectraforge/scene.py`, `tests/spectraforge/test_scene.py`

- [ ] **Step 1: Failing test**
```python
# tests/spectraforge/test_scene.py
import numpy as np
from spectraforge.material import Material
from spectraforge.scene import Scene

def test_paint_rect_sets_concentration():
    s = Scene(20, 20)
    s.paint_rect(Material("m", {"collagen": 2.0}), 5, 10, 5, 10)
    maps = s.resolve()
    assert maps["collagen"][7, 7] == 2.0
    assert maps["collagen"][0, 0] == 0.0

def test_painting_is_additive_and_mixes():
    s = Scene(10, 10)
    s.paint_rect(Material("a", {"collagen": 1.0}), 0, 10, 0, 10)
    s.paint_rect(Material("b", {"collagen": 0.5, "NADH": 1.0}), 0, 5, 0, 5)
    maps = s.resolve()
    assert maps["collagen"][2, 2] == 1.5   # overlap accumulates
    assert maps["collagen"][7, 7] == 1.0
    assert maps["NADH"][2, 2] == 1.0

def test_scene_addition_sums_maps():
    a = Scene(8, 8); a.paint_rect(Material("m", {"collagen": 1.0}), 0, 4, 0, 4)
    b = Scene(8, 8); b.paint_rect(Material("m", {"collagen": 1.0}), 0, 8, 0, 8)
    c = a + b
    assert c.resolve()["collagen"][1, 1] == 2.0
    assert c.resolve()["collagen"][6, 6] == 1.0

def test_paint_circle():
    s = Scene(21, 21)
    s.paint_circle(Material("m", {"EGFP": 1.0}), cy=10, cx=10, radius=5)
    maps = s.resolve()
    assert maps["EGFP"][10, 10] == 1.0
    assert maps["EGFP"][0, 0] == 0.0
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
```python
# src/spectraforge/scene.py
"""Scene: paint materials onto an H x W canvas -> per-fluorophore concentration maps."""
from __future__ import annotations
import numpy as np
from spectraforge.material import Material

class Scene:
    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)
        self._maps: dict[str, np.ndarray] = {}

    def _ensure(self, name: str) -> np.ndarray:
        if name not in self._maps:
            self._maps[name] = np.zeros((self.height, self.width), dtype=float)
        return self._maps[name]

    def _add_region(self, mask: np.ndarray, material: Material, amount: float) -> None:
        for fname, conc in material.recipe.items():
            self._ensure(fname)[mask] += conc * amount

    def paint_rect(self, material, r0, r1, c0, c1, amount=1.0):
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[r0:r1, c0:c1] = True
        self._add_region(mask, material, amount)

    def paint_circle(self, material, cy, cx, radius, amount=1.0):
        yy, xx = np.ogrid[: self.height, : self.width]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        self._add_region(mask, material, amount)

    def paint_polygon(self, material, vertices, amount=1.0):
        from matplotlib.path import Path as MplPath
        yy, xx = np.mgrid[: self.height, : self.width]
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = MplPath(vertices).contains_points(pts).reshape(self.height, self.width)
        self._add_region(inside, material, amount)

    def resolve(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._maps.items()}

    def __add__(self, other: "Scene") -> "Scene":
        if (self.height, self.width) != (other.height, other.width):
            raise ValueError("scenes must have the same shape to add")
        out = Scene(self.height, self.width)
        for src in (self, other):
            for name, m in src._maps.items():
                out._ensure(name)[:] += m
        return out
```
Export `Scene`.
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge): Scene painting to concentration maps`).

---

## Phase 4 — Acquisition + forward (clean)

### Task 4.1: `AcquisitionConfig`
**Files:** Create `src/spectraforge/acquisition.py`, `tests/spectraforge/test_acquisition.py`

- [ ] **Step 1: Failing test**
```python
# tests/spectraforge/test_acquisition.py
import numpy as np
from spectraforge.acquisition import AcquisitionConfig

def test_emission_grid_and_scalars():
    a = AcquisitionConfig(excitations=[340, 488], em_min=400, em_max=700, em_step=5,
                          exposure={340: 2.0}, power={488: 0.5})
    grid = a.emission_grid()
    assert grid[0] == 400 and grid[-1] == 700 and grid[1] - grid[0] == 5
    assert a.exposure_for(340) == 2.0
    assert a.exposure_for(488) == 1.0  # default
    assert a.power_for(488) == 0.5
    assert a.lamp_for(340) == 1.0
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
```python
# src/spectraforge/acquisition.py
"""Instrument acquisition configuration for a synthetic ME-HSI scan."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AcquisitionConfig:
    excitations: list[float]
    em_min: float
    em_max: float
    em_step: float
    lamp: dict[float, float] = field(default_factory=dict)
    exposure: dict[float, float] = field(default_factory=dict)
    power: dict[float, float] = field(default_factory=dict)

    def emission_grid(self) -> np.ndarray:
        return np.arange(self.em_min, self.em_max + 1e-9, self.em_step)

    def lamp_for(self, ex: float) -> float:
        return float(self.lamp.get(ex, 1.0))

    def exposure_for(self, ex: float) -> float:
        return float(self.exposure.get(ex, 1.0))

    def power_for(self, ex: float) -> float:
        return float(self.power.get(ex, 1.0))
```
Export `AcquisitionConfig`.
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge): AcquisitionConfig`).

### Task 4.2: `GroundTruth` container (needed by render)
**Files:** Create `src/spectraforge/groundtruth.py`, `tests/spectraforge/test_groundtruth.py`

- [ ] **Step 1: Failing test**
```python
# tests/spectraforge/test_groundtruth.py
import numpy as np
from spectraforge.groundtruth import GroundTruth

def test_groundtruth_save_roundtrip(tmp_path):
    gt = GroundTruth(
        concentration_maps={"collagen": np.ones((4, 4)), "NADH": np.zeros((4, 4))},
        clean_cubes={340.0: np.ones((4, 4, 3))},
        emission_grid=np.array([400.0, 450.0, 500.0]),
        excitations=[340.0],
    )
    gt.save(tmp_path)
    assert (tmp_path / "groundtruth.npz").exists()
    assert (tmp_path / "groundtruth.json").exists()
    loaded = np.load(tmp_path / "groundtruth.npz")
    assert "conc__collagen" in loaded
    assert np.array_equal(loaded["conc__collagen"], np.ones((4, 4)))
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
```python
# src/spectraforge/groundtruth.py
"""Ground-truth maps that accompany a synthetic dataset (the validation oracle)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

@dataclass
class GroundTruth:
    concentration_maps: dict[str, np.ndarray]   # fluorophore_name -> (H, W)
    clean_cubes: dict[float, np.ndarray]         # excitation -> (H, W, n_em) noise/scatter-free
    emission_grid: np.ndarray
    excitations: list[float]
    materials: dict = field(default_factory=dict)
    seed: int | None = None

    def save(self, out_dir) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {f"conc__{k}": v for k, v in self.concentration_maps.items()}
        arrays.update({f"clean__{ex}": c for ex, c in self.clean_cubes.items()})
        arrays["emission_grid"] = self.emission_grid
        np.savez_compressed(out / "groundtruth.npz", **arrays)
        meta = {
            "fluorophores": list(self.concentration_maps.keys()),
            "excitations": [float(e) for e in self.excitations],
            "emission_grid": [float(x) for x in self.emission_grid],
            "materials": self.materials,
            "seed": self.seed,
        }
        (out / "groundtruth.json").write_text(json.dumps(meta, indent=2))

    def informative_bands(self, threshold: float = 0.01) -> dict[float, np.ndarray]:
        """For each excitation, the boolean mask of emission bands whose max clean signal
        exceeds threshold * global max — i.e., bands that actually carry fluorophore signal."""
        gmax = max((c.max() for c in self.clean_cubes.values()), default=0.0) or 1.0
        out = {}
        for ex, cube in self.clean_cubes.items():
            band_max = cube.reshape(-1, cube.shape[-1]).max(axis=0)
            out[ex] = band_max > (threshold * gmax)
        return out
```
Export `GroundTruth`.
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge): GroundTruth container + sidecar save`).

### Task 4.3: `render()` — clean forward model
**Files:** Create `src/spectraforge/forward.py`, `tests/spectraforge/test_forward.py`

- [ ] **Step 1: Failing test** (the physics core + linearity invariant)
```python
# tests/spectraforge/test_forward.py
import numpy as np
from spectral_select.types import SpectraData
from spectraforge.fluorophore import Fluorophore
from spectraforge.material import Material
from spectraforge.scene import Scene
from spectraforge.acquisition import AcquisitionConfig
from spectraforge.forward import render

LIB = {"A": Fluorophore("A", 480, 40, 520, 30, quantum_yield=0.5, extinction=1.0)}
ACQ = AcquisitionConfig(excitations=[480.0], em_min=400, em_max=700, em_step=5)

def _scene(conc):
    s = Scene(6, 6); s.paint_rect(Material("m", {"A": conc}), 0, 6, 0, 6); return s

def test_render_emits_spectradata_with_right_shape():
    spectra, gt = render(_scene(1.0), LIB, ACQ)
    assert isinstance(spectra, SpectraData)
    ex = spectra.get_excitation(480.0)
    assert ex.cube.shape == (6, 6, len(ACQ.emission_grid()))
    assert len(ex.emission_wavelengths) == ex.cube.shape[2]

def test_emission_peak_at_fluorophore_em_peak():
    spectra, _ = render(_scene(1.0), LIB, ACQ)
    cube = spectra.get_excitation(480.0).cube
    grid = ACQ.emission_grid()
    peak_band = int(np.argmax(cube[0, 0]))
    assert grid[peak_band] == 520.0

def test_amplitude_scales_with_concentration():
    s1, _ = render(_scene(1.0), LIB, ACQ)
    s2, _ = render(_scene(2.0), LIB, ACQ)
    assert np.allclose(2 * s1.get_excitation(480.0).cube, s2.get_excitation(480.0).cube)

def test_linearity_invariant_clean():
    a, b = _scene(1.0), _scene(0.7)
    ra, _ = render(a, LIB, ACQ)
    rb, _ = render(b, LIB, ACQ)
    rab, _ = render(a + b, LIB, ACQ)
    assert np.allclose(rab.get_excitation(480.0).cube,
                       ra.get_excitation(480.0).cube + rb.get_excitation(480.0).cube)

def test_exposure_power_scaling_applied_and_recorded():
    acq = AcquisitionConfig(excitations=[480.0], em_min=400, em_max=700, em_step=5,
                            exposure={480.0: 3.0}, power={480.0: 2.0})
    base, _ = render(_scene(1.0), LIB, ACQ)
    scaled, _ = render(_scene(1.0), LIB, acq)
    assert np.allclose(scaled.get_excitation(480.0).cube, 6.0 * base.get_excitation(480.0).cube)
    assert scaled.get_excitation(480.0).exposure_time == 3.0
    assert scaled.get_excitation(480.0).laser_power == 2.0
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
```python
# src/spectraforge/forward.py
"""The forward model: scene + fluorophores + acquisition -> SpectraData + GroundTruth."""
from __future__ import annotations
import numpy as np
from spectral_select.types import ExcitationData, SpectraData
from spectraforge.groundtruth import GroundTruth

def render(scene, library, acquisition, artifacts=None, seed=None, sample_name="synthetic"):
    conc = scene.resolve()                       # {fname: (H, W)}
    h, w = scene.height, scene.width
    em = acquisition.emission_grid()
    rng = np.random.default_rng(seed)

    excitations = {}
    clean_cubes = {}
    for ex in acquisition.excitations:
        cube = np.zeros((h, w, len(em)), dtype=float)
        for fname, cmap in conc.items():
            f = library[fname]
            amp = f.extinction * f.quantum_yield * float(f.excitation(ex))  # scalar
            em_profile = f.emission(em)                                     # (n_em,)
            cube += (cmap * amp)[:, :, None] * em_profile[None, None, :]
        scale = acquisition.lamp_for(ex) * acquisition.exposure_for(ex) * acquisition.power_for(ex)
        cube *= scale
        clean_cubes[float(ex)] = cube.copy()
        if artifacts is not None:
            from spectraforge.artifacts import add_scatter_lines, add_noise
            add_scatter_lines(cube, ex, em, artifacts, scale)
            cube = add_noise(cube, artifacts, rng)
        excitations[float(ex)] = ExcitationData(
            cube=cube,
            excitation_nm=float(ex),
            emission_wavelengths=[float(x) for x in em],
            exposure_time=acquisition.exposure_for(ex),
            laser_power=acquisition.power_for(ex),
        )
    spectra = SpectraData(excitations=excitations, sample_name=sample_name)
    gt = GroundTruth(
        concentration_maps=conc,
        clean_cubes=clean_cubes,
        emission_grid=em,
        excitations=[float(e) for e in acquisition.excitations],
        seed=seed,
    )
    return spectra, gt
```
Export `render`.
- [ ] **Step 4: Run — PASS.**  **Step 5: Commit** (`feat(spectraforge): clean forward render -> SpectraData + GroundTruth`).

---

## Phase 5 — Artifacts

### Task 5.1: scatter lines + noise
**Files:** Create `src/spectraforge/artifacts.py`, `tests/spectraforge/test_artifacts.py`

- [ ] **Step 1: Failing test**
```python
# tests/spectraforge/test_artifacts.py
import numpy as np
from spectraforge.artifacts import ArtifactConfig, add_scatter_lines, add_noise

def test_rayleigh_adds_energy_at_excitation():
    em = np.arange(400, 700, 5.0)
    cube = np.zeros((3, 3, len(em)))
    cfg = ArtifactConfig(rayleigh_strength=1.0, rayleigh_fwhm=10, second_order=False)
    add_scatter_lines(cube, ex=500.0, em_grid=em, cfg=cfg, scale=1.0)
    peak = int(np.argmax(cube[0, 0]))
    assert em[peak] == 500.0
    assert cube[0, 0].max() > 0

def test_second_order_line_at_double_excitation():
    em = np.arange(400, 900, 5.0)
    cube = np.zeros((2, 2, len(em)))
    cfg = ArtifactConfig(rayleigh_strength=1.0, rayleigh_fwhm=10, second_order=True)
    add_scatter_lines(cube, ex=320.0, em_grid=em, cfg=cfg, scale=1.0)
    # peak near 640 = 2*320
    assert abs(em[int(np.argmax(cube[0, 0]))] - 640.0) <= 5.0

def test_noise_is_seed_deterministic():
    cube = np.full((4, 4, 5), 10.0)
    cfg = ArtifactConfig(photon_scale=1.0, read_sigma=0.5)
    a = add_noise(cube.copy(), cfg, np.random.default_rng(0))
    b = add_noise(cube.copy(), cfg, np.random.default_rng(0))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, cube)          # noise actually applied
    assert abs(a.mean() - 10.0) < 1.0           # mean approximately preserved
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
```python
# src/spectraforge/artifacts.py
"""Instrument artifacts: Rayleigh / 2nd-order scatter lines and detector noise."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493

@dataclass
class ArtifactConfig:
    rayleigh_strength: float = 0.0   # 0 disables scatter
    rayleigh_fwhm: float = 10.0
    second_order: bool = True
    photon_scale: float = 0.0        # 0 disables shot noise; else Poisson(signal*scale)/scale
    read_sigma: float = 0.0          # additive Gaussian read noise

def _line(em_grid, center, fwhm):
    sigma = fwhm * _FWHM_TO_SIGMA
    return np.exp(-0.5 * ((em_grid - center) / sigma) ** 2)

def add_scatter_lines(cube, ex, em_grid, cfg, scale, reflectance=None):
    """Add Rayleigh (em=ex) and optional 2nd-order (em=2*ex) lines, in place."""
    if cfg.rayleigh_strength <= 0:
        return
    h, w, _ = cube.shape
    refl = np.ones((h, w)) if reflectance is None else reflectance
    bump = cfg.rayleigh_strength * scale * _line(em_grid, ex, cfg.rayleigh_fwhm)
    cube += refl[:, :, None] * bump[None, None, :]
    if cfg.second_order and (2 * ex) <= em_grid[-1]:
        bump2 = cfg.rayleigh_strength * scale * _line(em_grid, 2 * ex, cfg.rayleigh_fwhm)
        cube += refl[:, :, None] * bump2[None, None, :]

def add_noise(cube, cfg, rng):
    """Return cube with Poisson shot noise + Gaussian read noise (seeded by rng)."""
    out = cube
    if cfg.photon_scale > 0:
        out = rng.poisson(np.clip(out, 0, None) * cfg.photon_scale) / cfg.photon_scale
    if cfg.read_sigma > 0:
        out = out + rng.normal(0.0, cfg.read_sigma, size=out.shape)
    return out
```
Export `ArtifactConfig`.
- [ ] **Step 4: Run — PASS** + run full `pytest -q tests/spectraforge` (clean render tests still green).  **Step 5: Commit** (`feat(spectraforge): scatter lines + seedable noise artifacts`).

---

## Phase 6 — Binding to spectral-select

### Task 6.1: end-to-end binding test
**Files:** Create `tests/spectraforge/test_binding.py`

- [ ] **Step 1: Write test** (renders, round-trips, runs the Analyzer, checks recovery)
```python
# tests/spectraforge/test_binding.py
import numpy as np
from spectraforge.fluorophore import Fluorophore
from spectraforge.material import Material
from spectraforge.scene import Scene
from spectraforge.acquisition import AcquisitionConfig
from spectraforge.artifacts import ArtifactConfig
from spectraforge.forward import render
from spectral_select.types import SpectraData

LIB = {
    "G": Fluorophore("G", 488, 40, 520, 30, quantum_yield=0.6, extinction=1.0),
    "R": Fluorophore("R", 560, 40, 610, 35, quantum_yield=0.4, extinction=0.9),
}
ACQ = AcquisitionConfig(excitations=[488.0, 560.0], em_min=400, em_max=700, em_step=5)

def _two_material_scene():
    s = Scene(16, 16)
    s.paint_rect(Material("g", {"G": 1.0}), 0, 16, 0, 8)
    s.paint_rect(Material("r", {"R": 1.0}), 0, 16, 8, 16)
    return s

def test_render_roundtrips_through_pickle(tmp_path):
    spectra, gt = render(_two_material_scene(), LIB, ACQ,
                         artifacts=ArtifactConfig(rayleigh_strength=0.2, photon_scale=200, read_sigma=0.01),
                         seed=0, sample_name="synth")
    p = tmp_path / "synth.pkl"
    spectra.to_pickle(p)
    reloaded = SpectraData.from_pickle(p)
    assert reloaded.n_excitations == 2
    assert reloaded.get_excitation(488.0).cube.shape == spectra.get_excitation(488.0).cube.shape

def test_groundtruth_informative_bands_are_near_emission_peaks():
    spectra, gt = render(_two_material_scene(), LIB, ACQ, seed=0)
    info = gt.informative_bands(threshold=0.05)
    grid = ACQ.emission_grid()
    # at 488 excitation, the G emission (~520) band should be flagged informative
    band_520 = int(np.argmin(np.abs(grid - 520)))
    assert info[488.0][band_520]

def test_analyzer_runs_on_synthetic(tmp_path):
    from spectral_select import Analyzer, Config
    spectra, _ = render(_two_material_scene(), LIB, ACQ,
                        artifacts=ArtifactConfig(photon_scale=500, read_sigma=0.01), seed=1)
    cfg = Config(sample_name="synth", n_important_dimensions=4, n_bands_to_select=4,
                 perturbation_method="standard_deviation", use_diversity_constraint=False,
                 training_epochs=2, device="cpu",
                 output_dir=tmp_path)
    analyzer = Analyzer(cfg)
    analyzer.fit(spectra)              # prepare (train) + select on synthetic data
    bands = analyzer.get_wavelengths()
    assert len(bands) == 4
```
- [ ] **Step 2: Run — expect PASS** (all engine modules already exist; this is the integration check). If `Analyzer.fit` needs a mask or different config keys, adjust the config to the real `Config` fields (verify against `src/spectral_select/config.py`) — the binding (render → SpectraData → fit) is what must hold.
- [ ] **Step 3: Commit** (`test(spectraforge): end-to-end binding to spectral_select.Analyzer`).

---

## Phase 7 — Demo + packaging

### Task 7.1: demo generator + console script
**Files:** Create `src/spectraforge/demo.py`; Modify `pyproject.toml`, `.github/workflows/test.yml`, `README.md`

- [ ] **Step 1: Implement `demo.py`**
```python
# src/spectraforge/demo.py
"""Generate a demo synthetic ME-HSI dataset + ground truth, then print where it went."""
from __future__ import annotations
import argparse
from pathlib import Path
from spectraforge.library import load_builtin_library
from spectraforge.material import Material
from spectraforge.scene import Scene
from spectraforge.acquisition import AcquisitionConfig
from spectraforge.artifacts import ArtifactConfig
from spectraforge.forward import render

def build_demo():
    lib = load_builtin_library()
    scene = Scene(64, 64)
    scene.paint_rect(Material("collagen_patch", {"collagen": 1.0, "NADH": 0.3}), 5, 40, 5, 40)
    scene.paint_circle(Material("fad_spot", {"FAD": 1.0}), cy=45, cx=45, radius=12)
    scene.paint_rect(Material("egfp_stripe", {"EGFP": 0.8}), 50, 60, 5, 60)
    acq = AcquisitionConfig(excitations=[340.0, 450.0, 488.0], em_min=360, em_max=700, em_step=5,
                            exposure={340.0: 2.0}, power={488.0: 0.8})
    artifacts = ArtifactConfig(rayleigh_strength=0.15, rayleigh_fwhm=12,
                               photon_scale=400.0, read_sigma=0.005)
    return render(scene, lib, acq, artifacts=artifacts, seed=42, sample_name="spectraforge_demo")

def main():
    ap = argparse.ArgumentParser(description="Generate a SpectraForge demo dataset.")
    ap.add_argument("-o", "--out", default="spectraforge_demo_out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spectra, gt = build_demo()
    spectra.to_pickle(out / "spectra_unmasked.pkl")
    gt.save(out)
    print(f"Wrote {spectra.n_excitations} excitations to {out}/spectra_unmasked.pkl")
    print(f"Ground truth (concentration maps + clean cubes) in {out}/groundtruth.npz")

if __name__ == "__main__":
    main()
```
- [ ] **Step 2: Test `build_demo()`**
```python
# tests/spectraforge/test_demo.py
from spectraforge.demo import build_demo
from spectral_select.types import SpectraData

def test_build_demo_produces_valid_dataset():
    spectra, gt = build_demo()
    assert isinstance(spectra, SpectraData)
    assert spectra.n_excitations == 3
    assert set(gt.concentration_maps) >= {"collagen", "NADH", "FAD", "EGFP"}
```
Run: `pytest -q tests/spectraforge/test_demo.py` → PASS.

- [ ] **Step 3: Packaging**
In `pyproject.toml`: `[tool.setuptools.packages.find].include` already has `spectraforge*` (Task 1.1); under `[project.scripts]` add `spectraforge-demo = "spectraforge.demo:main"`; add `[tool.setuptools.package-data]` `spectraforge = ["data/*.json"]`. In `.github/workflows/test.yml` add `--cov=spectraforge`. Run `pip install -e '.[dev]'` then `spectraforge-demo -o /tmp/sf_demo` and confirm files written.

- [ ] **Step 4: README** — add a "SpectraForge (synthetic data)" section: `spectraforge-demo` usage + that output loads in `spectral-select-gui`/`Analyzer`.

- [ ] **Step 5: Commit** (`feat(spectraforge): demo generator + console script + packaging`).

---

## Self-Review

- **Spec coverage:** §3 physics → Task 4.3; §4 artifacts → 5.1; §5 modules → all tasks (one per module); §6 output/binding → 4.2 (GroundTruth.save), 6.1 (pickle + Analyzer); §7 starter library → 1.2; §8 testing → tests in every task incl. linearity invariant (4.3) + binding (6.1); §9 phasing → Phases 1–7; §10 packaging → 7.1. No gaps.
- **Type consistency:** `Fluorophore.excitation/emission`, `Material.recipe`, `Scene.resolve()/paint_*/__add__`, `AcquisitionConfig.{emission_grid,lamp_for,exposure_for,power_for}`, `render(scene, library, acquisition, artifacts, seed, sample_name)`, `ArtifactConfig.{rayleigh_strength,rayleigh_fwhm,second_order,photon_scale,read_sigma}`, `add_scatter_lines(cube, ex, em_grid, cfg, scale, reflectance)`, `add_noise(cube, cfg, rng)`, `GroundTruth.{concentration_maps,clean_cubes,emission_grid,excitations,save,informative_bands}` — used consistently across tasks.
- **Verification flagged for execution:** confirm `Config` field names in Task 6.1 against `src/spectral_select/config.py` (training_epochs/n_important_dimensions/etc.) and that `Analyzer.fit` trains on the small synthetic cube quickly.
