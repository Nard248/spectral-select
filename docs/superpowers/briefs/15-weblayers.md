# Brief — weblayers engineer (3D layered-block modelling & editing in the editor)

Read `00-common.md`, `10-v11-common.md`, `05-webapp.md`, `06-three.md`, `apps/studio-web/README.md`, the new
`LayeredBlockShape` / `BlockLayer` / `ThicknessProfile` / `VolumeMCSettings` types in
`src/generated/experiment.ts` (schema docstrings in `src/spectraforge/studio/schema.py`), and the new presets
in `src/generated/presets.json` (`layered_wedge_phantom`, `tissue_under_collagen`). You own
`src/three/**` (except pose.ts), `src/panels/**`, `src/scene/**`, `src/state/**`, `src/tour/**`, `src/ui/**`.
The webanalysis engineer concurrently owns explorer/analysis/sweep/api — don't touch those.

## Build
1. **LayeredBlockMesh** (`src/three/`): exact geometry matching the engine — footprint (x,z) centred, bottom on
   local y=0, each layer between planar interfaces y = Σ t_j(x,z) (uniform or linear wedge along x or z;
   layers may thin to zero at an edge — no degenerate-triangle artefacts). Each layer a distinct translucent
   physical material from its `display_color` and optical character (turbid → milky sheen; clear → glassy;
   fluorescent → subtle emissive glow under the active excitation using the existing glow model), thin
   bright interface lines. Integrate into `SampleMesh` dispatch for `shape.kind === 'layered_block'`.
2. **Cutaway / section view**: toggle in the viewport toolbar + inspector: a clipping plane (x or z, with a
   slider) that cuts the block so the layer stack and the wedge are visible in cross-section, with a capped
   section face showing coloured layer bands and dimension annotations (thickness labels at both ends of the
   wedge, drei `Html`).
3. **Light-field overlay**: when the latest capture has `volumes` for a block, offer "Show light field" in the
   inspector/viewport: render a fluence (or source) slice on the cutaway plane as a texture in physical
   coordinates (fetch `/api/captures/{id}/volume` yourself with a small typed fetch helper in `src/scene/` —
   don't edit `client.ts`), with log colour mapping, legend, excitation picker. This is the "see scattering
   and absorption in 3D" money shot — make it beautiful (e.g. glowing banana-shaped photon paths under an
   oblique beam).
4. **Inspector: layer-stack editor** for layered blocks: footprint, an ordered list of layers (add / remove /
   duplicate / drag-to-reorder; name; material picker incl. "new material from…"; thickness profile: uniform
   or linear with start/end thickness and axis, live mini cross-section diagram of the profile), validation
   messages (zero total thickness, duplicate names). Fix the TS errors in `ObjectInspector.tsx` caused by the
   new shape kind. Acquisition inspector: a "3D Monte Carlo" group for `acquisition.volume_mc` with
   explanatory tooltips (quality ↔ time trade-off and an estimated runtime).
5. **Library**: "Layered block" lab-ware items (e.g. "Covered fluorophore (wedge)", "Tissue under collagen",
   "Two-layer slab") that insert a pre-configured block; presets menu shows the two new presets.
6. **Tour**: add a step (or two) for the layered block: cutaway → wedge → capture → light field.
Polish to v1 standard. Vitest for the profile math (thickness at x,z; interface heights) matching the Python
`ThicknessProfile.at`, and for the layer-stack editing reducers. `npm run build` 0 TS errors (whole app —
coordinate: if the build breaks only in webanalysis-owned files, report it), `npm test` green, eslint 0
errors. Screenshots 1440×900 → `apps/studio-web/screenshots/v11/` (block in design view, cutaway with wedge
annotations, layer-stack inspector, light-field overlay after a real-engine capture if available); look at
them and iterate.
