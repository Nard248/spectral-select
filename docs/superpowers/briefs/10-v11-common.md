# SpectralForge Studio v1.1 — layered 3D structures + analysis tools (common rules)

Read `00-common.md` first (workspace, venv command, quality bar), then this file, then your own brief.
Branch is now **`studio/layered-mc`** in the same worktree `/Users/narekmeloyan/PycharmProjects/spectral-select-studio`.
v1 (the whole current app) is DONE and committed — read `docs/studio/HOW_IT_WORKS.md` and
`docs/studio/ARCHITECTURE.md` to understand it before changing anything.

## What v1.1 adds (user's request)
"Define a 3D structure and simulate the light in the 3D field": a **layered block** — e.g. a substrate,
a uniform **fluorescent layer** in the middle, and a **cover layer on top** whose thickness can be uniform
or **decrease linearly across the object** (a wedge) — and see the effects of scattering and absorption in
the cover on the buried fluorophore. The end target is *tissue covered by collagen*; the tool must stay
generic (any materials). Then **scientist analysis tools**: pixel / disk / ROI **EEM retrieval** with
averaging, line profiles, signal-vs-ground-truth (e.g. vs local cover thickness) response curves,
normalisation modes matching the real lab pipeline (dark subtraction, exposure × power, white reference),
parameter sweeps (e.g. cover thickness 0 → 2 mm), and visualisation of the 3D light field inside the block.

## New contracts (lead-owned, read-only for you)
- `schema.py`: `ThicknessProfile` (uniform | linear wedge along local x or z), `BlockLayer`,
  `LayeredBlockShape` (shape kind `layered_block`, `footprint_mm=(x,z)`, layers bottom→top, every interface
  is an exact plane), `VolumeMCSettings` at `acquisition.volume_mc`.
- `presets.py`: new layer materials (`tissue_substrate`, `tissue_fluor_layer`, `fluorescein_agarose_layer`,
  `collagen_cover`, `intralipid_cover`, `absorbing_cover`, `clear_cover`) and presets
  `layered_wedge_phantom`, `tissue_under_collagen`.
- `contracts.py` (end of file): `BlockGeometry`, `VoxelGrid` (layer-conforming), `FaceGrid`, `BlockLaunch`,
  `RegionOptics`, `ExcitationTallies`, `EmissionTallies`; `CaptureResult.blocks` and `.volumes`.
- `apps/studio-web/src/api/types.ts` (end of file): NormSpec, RoiSpec, Eem*/Roi*/Line*/Response*, BlockInfo,
  VolumeInfo, Sweep* + the endpoint list. `src/generated/*` regenerated (new shape in the TS `Shape` union).
- Regenerate the web contract only via `PYTHONPATH=src <venv python> scripts/export_studio_contract.py && (cd apps/studio-web && npm run gen:types)` — and only the lead edits schema/presets.

## v1.1 ownership map
| engineer | owns |
|---|---|
| mc3d | NEW `src/spectraforge/studio/transport3d.py`, `transport3d_mc.py`, `tests/studio/test_transport3d*.py` |
| block | `geometry.py`, `geometry_nb.py`, `tracer.py`, `capture.py`, `optics.py`, `sensor.py`, `spectra.py`, `source.py`, `surface.py` + their tests, NEW `tests/studio/test_block_capture.py` |
| analysis | NEW `src/spectraforge/studio/analysis.py`, `sweeps.py`; `server.py`, `jobs.py`, `exports.py`, `render_png.py` + tests (`test_analysis.py`, `test_sweeps.py`, server/exports tests) |
| webanalysis | `apps/studio-web/src/explorer/**`, NEW `src/analysis/**`, NEW `src/sweep/**`, `src/api/client.ts`, `src/api/mock.ts`, `src/api/mockScene.ts` (+ tests), its OWN zustand store file(s) under `src/analysis/` (do not modify the main experiment store) |
| weblayers | `apps/studio-web/src/three/**` (except `pose.ts`), `src/panels/**`, `src/scene/**`, `src/state/**`, `src/tour/**`, `src/ui/**` (may add components; webanalysis may only ADD new files under `src/ui/`) |

Unchanged rules: no commits/pushes, no pip installs, only webanalysis/weblayers may `npm install` (coordinate
by only adding, never removing deps), never edit another engineer's files, import their not-yet-written
modules lazily and stub them in tests, poll near the end and integrate. Existing tests must stay green
(the preset-tracing test for `layered_wedge_phantom` currently FAILS until the block engineer teaches the
tracer the new shape — that is expected).

## Physics notes everyone must respect
- Units/conventions as in v1 (mm, nm, mW, ms, µM, 1/mm, Y-up, row 0 = image top).
- Radiance definitions from `contracts.py` module docstring. Block pixels get their radiance from the 3D MC
  exit tallies (mW/(mm²·sr·nm) after multiplying by emission density), NOT from the per-pixel kernel G.
- The tracer's specular glint at the block's top surface stays the tracer's job; the MC excitation pass must
  NOT also tally specularly reflected entry light (no double counting) — it tallies only light that entered.
- Component naming for block pixels in explain/decomposition: `dye:<fluorophore>@<layer name>`,
  `raman@<layer name>`, `elastic`, `glint`, `leak_source`.
- Ground-truth keys added for block pixels (NaN elsewhere): `layer_thickness:<layer>`, `layer_depth:<layer>`
  (depth of that layer's TOP interface below the block's top surface at that x,z), `block_face`,
  `origin_depth_mm`, `origin_layer_fraction:<layer>`.
