# SpectralForge Studio — web app

3D experiment designer + HSI explorer for the SpectralForge engine (Vite 6, React 19, TypeScript, three / R3F, zustand + immer).

```bash
npm install
npm run dev                   # http://localhost:5173 ; /api (HTTP + WS) is proxied to http://127.0.0.1:8765
npm test                      # vitest: store, spectral math, mock engine
npm run build                 # tsc -b && vite build
node scripts/screenshots.mjs  # Playwright screenshots (dev server running) -> screenshots/
node scripts/smoke.mjs        # UI smoke test against the mock engine
node scripts/tour-shots.mjs [url] [WxH]  # walks the guided tour, screenshots every step -> screenshots/tour/
node scripts/check-framing.mjs           # engine preview vs viewport/PiP projection (needs the engine) - measured worst case 1.11 px (about 1 px)
```

Engine: `PYTHONPATH=src python -m spectraforge.studio` (port 8765). If `/api/health` is unreachable the app falls back
to an in-browser mock engine (`src/api/mock.ts`) and shows an "engine offline — mock data" badge; click it to retry.

## Layout

| path | contents |
|---|---|
| `src/state/` | zustand store (experiment, undo/redo, selection, jobs), presets, library, `.sfx.json` io + autosave, engine actions, hotkeys |
| `src/api/` | `client.ts` (HTTP/WS client per `types.ts`), `mock.ts` + `mockScene.ts` (offline engine) |
| `src/scene/` | viewport assembly: nodes, gizmos, aim handles, look-through camera, PiP "Camera view" |
| `src/panels/` | top bar, outliner, library, inspector (+ `inspector/*`), acquisition bar |
| `src/explorer/` | cube image canvas, colormaps, pixel / ROI analysis |
| `src/ui/` | controls, charts (SVG/canvas), colormaps, icons, toast |
| `src/lib/` | client spectral math, camera intrinsics, beam model, glow shading |
| `src/three/` | presentational 3D models (separate work stream; consumed only through props) |

Shortcuts: `G`/`R` gizmo, `S` snap, `F` frame, `C` look through camera, `Space` preview, `Del`, `Cmd/Ctrl+Z` / `Shift+Z`, `Cmd/Ctrl+D`, `Cmd/Ctrl+S`, `1/2/3` modes, `[` / `]` panels.

## Guided tour

Press **Tour** in the top bar (or open `http://localhost:5173/?tour=1`, `?tour=auto` for auto-advance, `&step=N` to start at step N+1).
Eight captioned steps with eased camera fly-tos: bench, camera optics (frustum, focus plane, DOF), source and beam, samples and
materials, engine preview, live capture with progress, pixel decomposition in the Explorer, and the physics toggles (scattering
off vs on, side by side). `→` / `←` navigate, `Esc` ends the tour, the dots jump to a step, **Auto-play** advances on a timer.
It works with the real engine and in mock mode; the experiment is restored exactly when the tour ends.
Code: `src/tour/` (`steps.ts` holds the script, `Tour.tsx` the card + controller).

## Explorer analysis workspace (v1.1)

Explore mode has three workspaces (switch at the top of the centre column) and a four-tab analysis panel on the right.

- **Image** — ROI tools on the cube image: `V` select / pick pixel, `P` point, `D` disk, `R` rectangle, `E` ellipse, `G` polygon (Enter / double-click closes, Backspace removes a vertex, Esc cancels), `L` line. Regions are named, colour-coded, listed in the left column (rename, hide `H`, delete `Del`) and editable by dragging the body or the handles.
- **Panel tabs** — *Pixel* (component decomposition), *EEM* (ROI heatmap with contours / log / hover, emission spectra per excitation, excitation spectrum per band, ±std, Single / Overlay / A−B / A÷B incl. the same ROI in two captures, per-band and ground-truth statistics, optional averaged components), *Profile* (bands of one excitation along a line + a ground-truth trace on a second axis, mm axis when depth is known), *Response* (metric vs a ground-truth map, e.g. fluorescence vs cover thickness: binned mean ± std over a faint scatter). The **Units** bar selects the normalisation (ADU, radiance, electrons, dark-subtracted, exposure × power, white-normalised) and validates the dark / white captures; every axis shows its unit. Each result exports to CSV (self-describing `#` header: units, ROI, normalisation, capture, experiment) and JSON.
- **Light field** — for captures of a layered block: fluence / absorbed power / emission source per layer and dye, XZ section and XY plan drawn in physical coordinates (wedges look like wedges, layer interfaces overlaid, log colour scale) and the lateral-average depth profile per layer.
- **Sweep** — pick any numeric experiment parameter (searchable: layer thicknesses, dye concentrations, scattering, source power, camera…), a range or list, preview / full; results (ROI-averaged metric vs parameter) plot as each point arrives and every point opens its capture.

Code: `src/analysis/` (ROI geometry, store, hooks, panels, light-field geometry, export), `src/sweep/` (sweep store, parameter enumeration, panel), `src/ui/plots.tsx` (XY chart, EEM heatmap). State lives in its own zustand stores (`useAnalysis`, `useSweep`). The mock engine implements every endpoint (`src/api/mockAnalysis.ts`, `mockBlock.ts`); `?engine=mock` forces it. The engine only accepts the dev origin `http://localhost:5173`.
`node scripts/shoot-analysis.mjs [eem|line|response|lightfield|sweep|draw|all] [url] [--mock]` writes the QA screenshots to `screenshots/v11/`.

## Tissue optics + inclusions (v1.2, phase 1)

- **Material inspector** — *Optics* now has a scattering selector (none / power law / **Mie + Rayleigh**, Jacques 2013 eq. 2, with the Rayleigh fraction `f`); *Tissue composition* (switch on/off ↔ `tissue` object or `null`) has blood volume, StO₂, [Hb], water, lipid, melanosome, bilirubin, β-carotene with units, tooltips and a "water + lipid > 100 %" warning; *Optical properties* plots μa(λ) (log, per chromophore + total) and μs′(λ) over 350–900 nm and tabulates μa, μs′ and the penetration depth δ = 1/√(3 μa (μa+μs′)) at the sequence excitations and a free probe wavelength. All client-side: `src/lib/tissueSpectra.ts` is a coarse-grained copy of the chromophore tables (≈ 10–15 % of the engine's `tissue_optics`; the engine is authoritative). `materialMua` adds the tissue term, so the mock engine and the charts see it.
- **Inclusion editor** (layered-block inspector, `panels/inspector/InclusionEditor.tsx`) — add from presets (*Blood vessel*, *Tumour nodule*, *Fluorescent bead*) or custom (sphere / ellipsoid / cylinder / box), duplicate, delete; material picker; centre x/y/z plus a **depth-below-top** helper; per-kind size fields; cylinder axis and length. Instant validation (outside the block, missing material, thinner than 2 voxels). Hovering a card highlights it in 3D. Maths in `src/lib/inclusions.ts`.
- **3D** — `three/layers/InclusionMeshes.tsx`: translucent coloured meshes inside `LayeredBlockMesh` (x-ray over the turbid layers), clipped by the cutaway plane with an analytic capped cross-section + outline and a label; the cutaway card lists the inclusions cut by the plane.
- **Library** — lab-ware *Vessel under collagen*, *Buried tumour nodule*, *Skin (3-layer)*; *Materials → Tissue materials* (`state/tissueLibrary.ts`: the engine bundle's 18 tissue materials, plus the client-side `fluorescent_bead`).
- **Mock engine** — `api/mockBlock.ts` `columnSegments` overlays every inclusion's vertical chord on the layer column: a buried absorber darkens everything below it, a buried fluorophore glows attenuated by exp(−μ_eff·depth).
- `node scripts/shoot-v12.mjs [tissue|inclusions|cutaway|all]` writes the QA screenshots to `screenshots/v12/` (needs `npx vite --port 5192`; software GL makes it slow).
