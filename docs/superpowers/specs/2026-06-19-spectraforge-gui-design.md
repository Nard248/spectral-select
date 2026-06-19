# SpectraForge GUI ("the Forge") — Design Spec

- **Date:** 2026-06-19
- **Author:** Narek Meloyan
- **Status:** Approved design — pending spec review → implementation plan
- **Package:** `src/spectraforge/gui/` (GUI subpackage of the existing `spectraforge` engine)

## 1. Goal

An interactive desktop **workbench** that lets a user define fluorophores, mix them into materials
("brushes"), paint those materials onto a layered canvas, configure a multi-excitation acquisition,
render a physically-grounded ME-HSI dataset, preview it, and export it (with ground truth). It is a
thin, well-tested shell over the already-built SpectraForge engine. Separate app from the real-data
`mehsi_preprocessor`; launched via `spectraforge-gui`.

Non-goal: new physics (engine is fixed); 3D/volumetric painting; the deferred engine features
(inner-filter, PSF, FRET).

## 2. Decisions (locked with owner)

| # | Decision | Choice |
|---|---|---|
| 1 | Structure | **Workbench** — docked panels around a persistent canvas |
| 2 | v1 scope | **Fuller** — library/spectra + define-fluorophore + material composer + canvas/layers + render/export |
| 3 | App identity | **Separate app** `spectraforge-gui` (own window), reuses `mehsi_preprocessor` canvas widgets |

## 3. Architecture

`QMainWindow` with dock widgets:

```
left dock:  Library/Materials   center: Canvas    right dock: Layers
bottom:     Acquisition · Render · slice preview · Export
```

```
src/spectraforge/gui/
  __init__.py
  state.py            ForgeState: library{name->Fluorophore}, materials{name->Material},
                      layers[Layer], acquisition (AcquisitionConfig), artifacts (ArtifactConfig),
                      last_render: tuple|None
  layer.py            Layer{name, material, amount_map(H,W float), visible}; build_scene(state)->Scene
  workers.py          RenderWorker(QThread): runs forward.render off the UI thread
  app.py              main window assembly + main() (console entry)
  widgets/
    spectrum_plot.py  matplotlib widget plotting excitation/emission curves (+ mixed material spectrum)
  panels/
    library_panel.py        fluorophore list + spectrum plot + "Define fluorophore" form
    material_panel.py        compose Material from fluorophores + concentrations; mixed-spectrum preview
    canvas_panel.py          paint active material onto active layer (rect/circle/freehand/polygon)
    layers_panel.py          layer stack: add/remove/reorder/toggle-visibility/select-active
    acquire_render_panel.py  excitation list + emission grid + artifact toggles; Render; slice preview; Export
```

### Engine addition (one small, pure function)
`Scene.paint_map(material, amount_map)` — per-pixel concentration painting (the existing
`paint_rect`/`paint_circle` apply a uniform amount over a region; brush strokes need per-pixel
amounts). Adds `concentration[fname] += conc * amount_map` per fluorophore. Tested in the engine
suite.

### The layer model (the one piece of real logic)
A `Layer` carries a `material` and a float `amount_map` (H×W) accumulated by brush/shape strokes.
`build_scene(state) -> Scene` creates a `Scene(H, W)` and, for each **visible** layer in order,
calls `scene.paint_map(layer.material, layer.amount_map)`. The engine's additive mixing then yields
the per-fluorophore concentration maps. This is the only non-trivial GUI logic and is pure/tested.

## 4. Data flow

Define fluorophores → compose materials → paint materials onto layers (amount maps) →
`build_scene(state)` → `RenderWorker` → `forward.render()` → `(SpectraData, GroundTruth)` → preview a
cube slice (excitation/emission sliders) → `SpectraData.to_pickle()` + `GroundTruth.save()`. Output
loads directly in `spectral-select-gui` / `Analyzer`.

## 5. Panels (behavior)

- **Library:** list built-in + user fluorophores; selecting one plots its ex/em curves; a "Define
  fluorophore" form (name, ex/em peak & FWHM, Φ, ε) adds a new `Fluorophore` with a live preview.
- **Materials:** create a named material; add fluorophore rows with concentration spin boxes; preview
  the *mixed* emission spectrum at a chosen excitation; the material becomes a selectable brush.
- **Canvas:** shows a false-color composite of the visible layers (each material gets a display color,
  blended by amount). The active material + tool (rect / circle / freehand brush / polygon) paints into
  the **active layer's** amount_map. Reuses `BrushCanvas`/`RectSelector` mechanics.
- **Layers:** ordered stack; add (with a chosen material), remove, reorder, toggle visibility, set
  active. Active layer receives paint.
- **Acquire/Render/Export:** edit excitation list, emission grid (min/max/step), artifact toggles
  (Rayleigh strength, noise); **Render** (background `RenderWorker`, progress); a slice preview with
  excitation + emission sliders; **Export** writes `spectra_unmasked.pkl` + `groundtruth.npz/json`.

## 6. Testing

- **Pure/logic (headless):** `Scene.paint_map` (per-pixel accumulation, additive mixing);
  `Layer`/`build_scene` (visible-layers-only, order, correct concentration maps); render/export glue
  (renders a 2-layer state → `SpectraData` with expected shape; export writes files).
- **GUI smoke (offscreen):** each panel + the main window builds under `QT_QPA_PLATFORM=offscreen` and
  reacts to a stub `ForgeState` without raising (mirrors the mehsi GUI tests).

## 7. Phasing

1. **`Scene.paint_map` (engine) + `ForgeState` + `Layer` + `build_scene`** — pure, tested.
2. **`RenderWorker` + render/export helpers** — headless-testable job functions.
3. **`spectrum_plot` widget + Library panel** — view spectra + define fluorophore.
4. **Material composer panel** — compose materials + mixed-spectrum preview.
5. **Canvas + Layers panels** — paint materials onto layers; layer stack.
6. **Acquire/Render/Export bar + slice preview.**
7. **Main-window assembly + `spectraforge-gui` entry + offscreen smoke tests + README.**

Each phase ships working, tested code; Phases 1–2 are valuable headless even before any panel exists.

## 8. Packaging

`[project.scripts]` add `spectraforge-gui = "spectraforge.gui.app:main"`; GUI uses the existing
`gui` extra (PyQt6) + matplotlib (core dep). Smoke tests marked so they run offscreen.

## 9. Risks & mitigations

- **Painting interaction complexity** → reuse `mehsi_preprocessor` canvas widgets; keep brush logic
  writing into a float `amount_map` rather than reinventing.
- **UI freeze on render** → all rendering in `RenderWorker`; only signals touch the UI.
- **Correctness hidden in Qt** → all real logic in pure `build_scene`/`paint_map`/render helpers,
  fully unit-tested; Qt code is a thin shell.
- **Cross-package widget reuse** → import `mehsi_preprocessor.widgets` (same monorepo); if coupling
  grows, extract a shared `hsi_widgets` package later (out of scope now).

## 10. Out of scope (future)

Custom display-color editing per material; undo/redo; saving/loading a Forge "project" file;
freehand spline shapes; the deferred engine physics; multi-sample batch generation.
