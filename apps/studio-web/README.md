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
