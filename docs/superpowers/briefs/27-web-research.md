# W5b brief (phase 2): web engineer, research workspace UI (studies, sensitivity, inverse, design, methods)

Read `20-v12-common.md` first, then `docs/studio/STUDY_API.md`, `docs/studio/INVERSE_API.md` and
`docs/studio/SIM2REAL.md` (endpoints implemented by W3, W4 and W6), and the existing web app: the
`src/analysis/**` workspace, `src/sweep/**`, `src/api/client.ts`, `src/ui/plots.tsx`. You own
`apps/studio-web/**` except `src/generated/**`. Add TS types for the new endpoints in a NEW file
`src/api/researchTypes.ts`, mirroring the Pydantic models exactly. Read `study_schema.py` and the route
modules.

## Build: a "Research" workspace (a new top-level tab next to Design / Explorer)
1. **Study builder**:
   - Parameter picker: reuse the sweep panel's numeric-field picker, with ranges, log scale, linked
     pointers and readable labels.
   - Design: grid / random / LHS / Sobol / Saltelli / Morris, with N and an estimated runtime from the
     preview timing.
   - Replicates; metric builder: ROI band / ratio / contrast / CNR / GT stat, with an ROI picker from the
     analysis store.
   - Run with live progress over WS; resume and cancel.
2. **Study results**:
   - A sortable runs table, a parallel-coordinates plot, and scatter (metric against parameter) with replicate
     error bars.
   - A Sobol S1/ST bar chart with CIs and the MC-noise-floor line, and a Morris μ*–σ plot.
   - Click a run to open its capture, if it was kept.
   - CSV/JSON export.
3. **Inverse panel**:
   - Unmixing: method, endmember source, abundance maps with GT comparison, RMSE map.
   - Fisher/CRLB: choose the parameters and ROI, then show the information heat map (excitation ×
     band), the CRLB per parameter and the per-channel contributions.
   - Optimal design: k slider, greedy picks highlighted on the heat map, and a CRLB(k) curve against
     random/uniform baselines.
   - Fit: build the surrogate, then fit the ROI with the estimate ± CI, or produce a thickness map against GT.
4. **Methods text**: a button that shows the generated Methods paragraph (Markdown/LaTeX toggle) with copy,
   and a provenance panel (version, commit, experiment hash).
5. **Calibrated fluorophores** (W6): show them in the fluorophore library with a "calibrated from real data"
   badge and fit metrics.
6. Mocks: offline mock implementations of every new endpoint, so the UI works without the engine.
Same quality bar as v1/v1.1: beautiful, consistent design system, keyboard-accessible, and 0 TS errors,
eslint 0 errors, vitest for the new pure logic (parsers, CSV, design-size estimates). Playwright screenshots
(1440×900) against the REAL engine: start it with `.venv/bin/python -m spectraforge.studio --port 8765`,
serving the dist or using the dev server per the README. Take them of the study results with Sobol bars,
the optimal-design view and the methods text, save them to `apps/studio-web/screenshots/v12/`, look at
them and iterate.

## Addendum (lead)
- **Engine-backed optics**: the phase-1 client-side spectra in `src/lib/tissueSpectra.ts` were typed from memory
  (~10–15 % off). Add a small endpoint `POST /api/material_optics` (body `{material: MaterialSpec, nm_min, nm_max,
  step}` → `{nm[], mua_per_mm[], mus_prime_per_mm[], g, n, components: {name: mua[]}}`) using
  `optics.material_optics` + `tissue_optics` per-chromophore terms, in a NEW `src/spectraforge/studio/routes_materials.py`
  (APIRouter `router`, JSON only, pydantic validation, size limits), include it in `server.py` like the other routers,
  with a pytest in NEW `tests/studio/test_routes_materials.py`. The material inspector charts use it (debounced),
  falling back to the client tables offline (mock). You may touch exactly these Python files.
- Study modes are `preview` (1st excitation only) | `fast` (all excitations, preview res) | `full`; default the UI to
  `fast` and explain the modes in a tooltip.
- `docs/studio/INVERSE_API.md` / `SIM2REAL.md` may not exist yet (W4/W6 still finishing): build the study UI,
  methods text, provenance and optics endpoint first; then poll for those docs (check every ~10 min of work) and build
  the inverse panel and calibrated-fluorophore badges when they appear.
