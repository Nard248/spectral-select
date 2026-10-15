# W5 brief (phase 1): web engineer, editors for the v1.2 schema

Read `20-v12-common.md` first, then `apps/studio-web/README.md`, `src/generated/experiment.ts` (new:
`TissueComposition`, `MaterialSpec.tissue`, `ScatteringSpec.model "mie_rayleigh"` + `f_rayleigh`,
`Inclusion`, `LayeredBlockShape.inclusions`, `VolumeMCSettings.inclusion_voxel_mm`) and the current
`src/panels/**`, `src/three/**` (LayeredBlockMesh, layers/*), `src/state/**`. Phase 2 (study/inverse UI)
comes later in a separate brief, so don't build it now.

## Build (to v1 polish standard)
1. **Fix the 9 TS errors** caused by the new required fields (library items, store defaults, mocks:
   `tissue: null`, `inclusions: []`, `f_rayleigh: 0`, `inclusion_voxel_mm: 0.2`).
2. **Material inspector: tissue optics.**
   - A "Tissue composition" group (toggle on/off → `tissue` null or object): sliders with units and
     tooltips for blood fraction (%), StO₂ (%), water/lipid/melanosome fractions, bilirubin, β-carotene,
     with sanity validation (water + lipid ≤ 100 % warning).
   - A scattering model selector (none / power law / Mie+Rayleigh with f_Ray).
   - A **live optical-properties chart**: μa(λ) (log) and μs'(λ) 350–900 nm, computed client-side. Port
     the chromophore spectra coarse-grained into a small TS table in `src/lib/tissueSpectra.ts`, or better,
     fetch `GET /api/materials/{id}/optics` if W3/W1 expose it (check `docs/studio/STUDY_API.md` later; the
     client-side fallback is fine). Also show the penetration depth δ = 1/sqrt(3μa(μa+μs')).
3. **Inclusion editor** in the layered-block inspector:
   - A list with add / duplicate / delete; kind (sphere/ellipsoid/cylinder/box); material picker; centre
     x/y/z with "depth below top" helper input; size fields per kind; cylinder axis.
   - Presets "blood vessel", "tumour nodule", "fluorescent bead".
   - Validation: the inclusion lies at least partly inside the block.
4. **3D rendering of inclusions** inside `LayeredBlockMesh`: translucent coloured meshes (sphere,
   ellipsoid, cylinder clipped to the footprint, box) visible through the semi-transparent layers, clipped by
   the cutaway plane with a capped section, and labelled in the section card. Drag-gizmo editing of an
   inclusion's position is optional (bonus).
5. **Library**: "Vessel under collagen", "Buried tumour nodule", "Skin (3-layer)" lab-ware items (use the
   W1 tissue material ids if they exist in `presets.json` by then, else generic materials). Also a
   "Tissue materials" section in the material library listing W1's tissue materials once they are
   exported.
6. Mock engine (`src/api/mockScene.ts`/`mockBlock.ts`): render inclusions plausibly: a buried absorber
   darkens, a buried fluorophore brightens attenuated by exp(−μ_eff·depth).
Vitest: inclusion geometry helpers (depth-below-top, inside-block test), the tissue μa math against two
anchor values, and store reducers. `npm run build` 0 TS errors, `npm test` green, eslint 0 errors.
Screenshots (Playwright, 1440×900) → `apps/studio-web/screenshots/v12/`: the tissue material inspector with
the chart, the inclusion editor, and a vessel under collagen in cutaway. Look at them and iterate until
they are beautiful.
