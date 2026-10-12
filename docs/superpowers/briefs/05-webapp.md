# Brief — web-app engineer (app shell, state, panels, API client, data explorer)

Read `00-common.md` first. The user disliked the old 2D PyQt app ("not user friendly"); this must feel like a
polished professional product (Blender / Figma / Linear quality, dark scientific-instrument aesthetic) and will be
demoed live at a conference. You own `apps/studio-web/**` EXCEPT `src/three/**` (3D engineer), `src/generated/**`
and `src/api/types.ts` (contracts). You MAY `npm install` packages that clearly help (e.g. @fontsource fonts,
lucide-react icons); prefer lightweight custom canvas/SVG charts.

Read: `src/generated/experiment.ts`, `src/generated/presets.json`, `src/api/types.ts`, `src/three/types.ts`,
`src/three/pose.ts`, `src/lib/color.ts`. The 3D engineer replaces the placeholder components in `src/three/`
(`CameraModel`, `LightSourceModel`, `SampleMesh`, `Stage`, `StudioEnvironment`, and will add e.g. `BeamVolume`,
frustum, `Effects`, `Showcase`, barrel `src/three/index.ts`) in place with the same names/props — consume them
only through props, poll occasionally and integrate new ones. Your viewport assembly goes under `src/scene/`.

The Python engine (FastAPI, http://127.0.0.1:8765) is written concurrently. Vite dev proxy `/api` →
`http://127.0.0.1:8765` (incl. WebSocket). `src/api/client.ts` exactly per `src/api/types.ts`, plus a **mock
engine** used automatically when `/api/health` is unreachable: generates plausible fake captures client-side
(gaussian dye spectra over material regions) so every UI path (preview, capture progress, explorer, pixel
explain) is demoable offline; show a subtle "engine offline — mock data" badge.

## Build everything (polish is a requirement)
**Layout**: top bar (wordmark "SpectralForge Studio", editable experiment name, preset menu, undo/redo,
save/open `.sfx.json`, Design | Acquire | Explore switch, engine status); left outliner (source, camera,
objects, background; visibility/select/delete/duplicate) + **library** (shapes; lab-ware: Petri dish, cuvette,
glass slide = thin glass box, droplet, bead, block, puck; materials from presets.json; dyes); center R3F
viewport; right inspector; bottom acquisition sequence bar. Works down to a 1280×720 projector; collapsible
panels; shortcuts (G/R/S gizmo, F frame, Del, Ctrl/Cmd+Z / Shift+Z, Ctrl/Cmd+D, C look-through-camera, Space preview).

**State**: zustand + immer store with the full `Experiment` (from presets.json), selection, UI mode, captures,
jobs; undo/redo (patches or snapshots, debounced for slider drags); autosave to localStorage (try/catch);
load/save `.sfx.json` (file input + Blob download); unique ids.

**Viewport (`src/scene/`)**: Stage, SampleMesh objects, LightSourceModel(s) with beam, CameraModel(s) with
frustum, StudioEnvironment (+ Effects when available). OrbitControls with damping; drei `TransformControls` for
objects (translate/rotate, optional 25 mm grid snap); **drag-to-aim** target handles for source and camera
(update `target_mm`) and hardware position dragging; click select, hover highlight; drei `Html` readouts (beam
incidence angle on the selected object, camera–object distance). **Look-through-camera**: viewport camera =
HSI camera (same pose as pose.ts; vertical FOV = 2·atan(sensor_h/(2v)), v = f·s/(s−f), sensor_h =
height_px·pixel_pitch_um·binning/1000 mm; sensor aspect; letterbox frame + thirds overlay). **PiP "Camera view"**:
live GPU preview from the HSI camera intrinsics (drei `View` or FBO), shaded as approximate fluorescence: per
material under the active excitation, Σ_dyes ε·c·exc(λex)·Φ × emission colour (emission spectrum integrated
against approximate RGB), modulated by the Gaussian beam footprint and shadows → pass `glowColor`/`glowIntensity`
to SampleMesh. Label "Preview (approx.)" vs "Captured".

**Inspector**: context-sensitive; every schema field editable, grouped/collapsible, units, numeric inputs with
slider-drag, dropdowns. Object (shape params, transform, material + vessel material). Material (dye recipe chips
with µM, n, scattering, absorber, roughness, aqueous, quenching, FRET pairs; **live plots** μa(λ), μs′(λ), mini EEM
heatmap). Source (kind, pose, beam, line FWHM, lamp curve, filter OD, drift, **excitation sequence table**
add/remove/reorder/edit; plot line shapes vs dye absorption). Camera (lens, sensor, filter + band list preview,
detector; plot passbands + QE + long-pass). Background + room light. Acquisition: seed, MC quality, preview scale,
**physics toggles grid** (all of them, with explanatory tooltips — a teaching feature). Debounced
`POST /api/validate`: field errors inline, physics warnings in the bottom bar.

**Acquire**: sequence bar (wavelength-coloured excitation chips, UV violet glow, estimated total time,
"order matters — photobleaching" hint); [Preview] (POST /api/preview → PiP + explorer), [Capture]
(POST /api/captures + WS progress per excitation, cancel), [Dark], [White ref]; while acquiring, camera
`acquiring` prop true and the beam colour cycles through the active excitation.

**Explore** (full-screen): captures list; image canvas rendering float32 bands (`/band`) with colormaps
(viridis, magma, gray, 3-band false colour, true colour via `/image.png?mode=true_color`), percentile contrast,
log; excitation tabs + band scrubber (arrows) + playback; **click pixel →** EEM heatmap (excitation × emission)
from `/pixel`, per-excitation spectra, **stacked area chart of `components`** (the hero teaching view),
ground-truth scalars; ground-truth overlay selector (`/gt/<name>`); ROI rectangle with mean spectrum; compare two
captures side by side; export buttons (`/download?format=…`); warnings + timings.

**Visual design**: dark graphite (#0d0f12-ish), one accent (electric cyan or similar) + wavelength colours for
spectral content; Inter + JetBrains Mono (numbers) via @fontsource with fallbacks; subtle glass panels, 8 px
grid, micro-interactions, focus rings, empty states, tooltips. First screen must impress: load "ME-HSI lab
replica" or "Generic macro bench" by default with a beautiful camera angle.

## Engineering
Strict TS, no `any` leaks; small focused files (`src/state/`, `src/api/`, `src/ui/`, `src/panels/`,
`src/explorer/`, `src/scene/`, `src/lib/spectral.ts` — client spectral math consistent with Python:
Gaussian profiles, ε = extinction·55000·exc(λ), μa = ln10·ε·c/10). Vitest for the store (undo/redo, add/delete,
preset load) and spectral math. `npm run build` passes with zero TS errors. When the UI is in place run
`npm run dev` (background) and take Playwright screenshots at 1440×900 of Design, Acquire (mock capture in
progress) and Explore (mock capture with a pixel selected) into `apps/studio-web/screenshots/`; LOOK at them with
your image-reading tool and iterate on visual polish; stop the dev server afterwards. Final report: structure,
features done/not, how to run, screenshot paths, tests, anything missing from the API contract.
