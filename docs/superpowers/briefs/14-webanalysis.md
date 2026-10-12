# Brief — webanalysis engineer (Explorer analysis workspace, light-field viewer, sweeps UI)

Read `00-common.md`, `10-v11-common.md`, `05-webapp.md` (v1 web conventions), `apps/studio-web/README.md`,
the v1.1 section at the end of `src/api/types.ts` (your contract), and the current `src/explorer/**`.
You own `src/explorer/**`, NEW `src/analysis/**`, NEW `src/sweep/**`, `src/api/client.ts`, `src/api/mock.ts`,
`src/api/mockScene.ts` (+ tests). Keep your state in your OWN zustand store under `src/analysis/` — don't
modify the main experiment store (the weblayers engineer owns `src/state/**`). You may ADD new files under
`src/ui/`. The engine side (analysis endpoints, sweeps, volumes) is built concurrently by the analysis
engineer: implement `client.ts` methods exactly per types.ts AND matching mock implementations in
`mock.ts` so everything works offline (the mock must also render `layered_block` objects plausibly in
`mockScene.ts` — e.g. buried-layer emission attenuated by exp(−k·cover_thickness) and the cover's own
autofluorescence — and provide fake volumes; fix the current TS error in `mockScene.ts`).

## Build (scientist-grade, polished)
1. **ROI toolbar** on the explorer image: point, disk, rectangle, ellipse, polygon, line. Multiple named,
   colour-coded ROIs (list with rename/hide/delete), drag-to-edit handles, keyboard shortcuts.
2. **EEM workspace**: for the selected ROI — heatmap (excitation × emission, log/linear, contour lines,
   hover readout), emission spectra per excitation and excitation spectra per emission band (slices),
   ±std shading, **ROI overlay mode** (several ROIs on one chart), **difference / ratio mode** (ROI A vs ROI B,
   or same ROI across two captures), stat mean/median.
3. **Normalisation selector** (adu, radiance, electrons, dark-subtracted, exposure×power, white-normalised)
   with dark/white capture pickers that validate compatibility; units shown on every axis.
4. **ROI stats** table: n pixels, per-band mean±std, ground-truth stats (e.g. mean cover thickness), optional
   averaged component decomposition (stacked area, names `dye:<dye>@<layer>` rendered as "NADH · tissue").
5. **Line profile**: draw a line (e.g. across the wedge) → bands of a chosen excitation vs distance (mm when
   available), with a ground-truth trace on a secondary axis (e.g. `layer_thickness:cover`).
6. **Response plot**: metric (band / band ratio / band sum / peak wavelength) vs a ground-truth variable
   (dropdown of numeric GT layers, e.g. cover thickness), binned mean±std + faint scatter; optional ROI.
   This is the headline "signal vs cover thickness" plot.
7. **Light-field viewer** (for captures with `volumes`): object / kind (fluence, absorbed, source+channel) /
   excitation pickers; slice views XZ-plane (depth vs x) and XY-plane at a chosen depth, drawn in PHYSICAL
   coordinates (layer-conforming cells mapped to true depth using the layer thicknesses, so wedges look
   like wedges), layer boundaries overlaid, log colour scale; lateral-average depth profile chart per layer.
8. **Sweep panel**: choose a numeric parameter (a picker listing numeric fields of the current experiment
   with readable labels — must include layered-block layer thicknesses/end thicknesses, material dye
   concentrations, scattering, source power, camera settings), range + steps (or explicit list), preview or
   full mode; run via `/api/sweeps` with live progress; results: chosen metric (ROI + metric spec) vs the
   parameter, plotted as each point arrives; click a point to open that capture.
9. **Export** every analysis result as CSV and JSON (Blob download) with headers including units, ROI
   definition, normalisation, capture id and experiment name.
Make it look as polished as v1 (same design system). Vitest tests for ROI geometry, CSV formatting, mock
endpoint contracts and the store. `npm run build` 0 TS errors, `npm test` green, eslint 0 errors.
Screenshots (Playwright, 1440×900, real engine if it supports the endpoints by then, else mock): EEM
workspace with 2 ROIs, line profile across the wedge, response plot vs cover thickness, light-field viewer,
sweep results → `apps/studio-web/screenshots/v11/`; look at them and iterate.
