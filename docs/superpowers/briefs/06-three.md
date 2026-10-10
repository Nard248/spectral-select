# Brief — 3D engineer (hardware models, beams, samples, stage, rendered look)

Read `00-common.md` first. The user explicitly asked for **eye-catching, beautiful 3D models of each hardware
component**. You own `apps/studio-web/src/three/**`. **Do NOT npm install** (use installed packages only). No
network assets: no HDRI downloads, no remote GLTF, no drei `Environment preset=` (CDN). Build all geometry
procedurally; use drei `Lightformer` inside `<Environment>` for studio reflections; generate textures
(breadboard hole pattern, labels, brushed-metal roughness) with CanvasTexture/DataTexture.

Replace IN PLACE the placeholders `src/three/hardware/CameraModel.tsx`, `src/three/hardware/LightSourceModel.tsx`,
`src/three/SampleMesh.tsx`, `src/three/Stage.tsx`, `src/three/StudioEnvironment.tsx` — keep export names and the
props in `src/three/types.ts` (you may ADD optional props only). `pose.ts`: optical axis = local −Z, local +Y =
device up — keep it. Add a barrel `src/three/index.ts`. Components must be **pure functions of props** (no store
access) — the web-app engineer (under `src/scene/` etc.) consumes them via props only.

Read: `src/generated/experiment.ts` (CameraSpec, LightSourceSpec, SceneObject shapes box|sphere|cylinder|droplet|
petri_dish|cuvette in mm, MaterialSpec, Background), `src/generated/presets.json` (real scenes),
`src/lib/color.ts` (`wavelengthToRGB`, UV → violet), `src/spectraforge/studio/schema.py` shape docstrings.
**1 unit = 1 mm**, Y up, table top y = 0; camera ~330 mm above table, source ~250 mm away, samples 5–90 mm.
Meshes MUST match the schema dimensions exactly (box/sphere/cylinder centred; droplet = spherical cap with base on
local y=0; petri dish / cuvette bottom on local y=0) — the physics engine ray-traces the same shapes.

## Deliverables (product-render beautiful, plausible PBR, 60 fps)
1. **CameraModel** — detailed Nuance/LCTF-class HSI camera (~90×90×140 mm body + lens): dark anodized body with
   bevels (rounded boxes / ExtrudeGeometry bevel), cooling fins, subtle brushed finish, label plate
   "SpectralForge HSI" (canvas texture), status LED pulsing when `acquiring`, connector ports + cable stub; a
   visible **LCTF filter stack** (thin glass plates with iridescent thin-film MeshPhysicalMaterial, shimmering when
   acquiring); C-mount lens barrel with **knurled focus and aperture rings rotating with `focus_distance_mm` /
   `f_number`**, barrel length from focal length, coated front element; post + clamp mount to the table (auto
   length); selected → rim highlight; `showFrustum` → frustum from sensor (`width_px·pixel_pitch_um·binning`) and
   v = f·s/(s−f), translucent **focus plane**, depth-of-field slab (CoC = 2 px). Optical axis local −Z; group at
   `position_mm` with `lookAtQuaternion(...)`.
2. **LightSourceModel** per `source.kind`: `tunable_lamp` — light exits at `position_mm` from a **collimator head**
   on an articulated post/clamp, connected by a smooth **fibre-optic cable** (TubeGeometry on CatmullRom) to a
   **benchtop xenon lamp + monochromator unit** on the table (vents, glowing LCD showing the current wavelength
   e.g. "365 nm", knobs, UV warning sticker); `led_array` — ring/array of LED dies on a heat-sinked PCB disc glowing
   in the active colour; `laser` — cylindrical head, aperture, warning triangle, key switch. Plus exported
   **BeamVolume** (rendered inside LightSourceModel when `showBeam` and `activeWavelengthNm != null`): volumetric
   cone with the real beam geometry (radius D/2 + z·tan α toward `target_mm`, extended past it), custom
   ShaderMaterial (additive, no depth write, gaussian/flat-top radial falloff per `beam.profile`, distance falloff,
   soft edges, view-angle shading), colour from `wavelengthToRGB`, animated violet shimmer for UV (< 400 nm),
   glowing **footprint** ellipse where the cone meets y = 0.
3. **SampleMesh** for every shape kind: glass vessels as MeshPhysicalMaterial (transmission, thickness, ior 1.52);
   Petri dish base+wall at correct thickness, liquid fill (ior 1.333, attenuation colour from `display_color`, flat
   top); cuvette open-top shell + fill; droplet spherical cap (LatheGeometry) with clear-coat sparkle; turbid solids
   (lichen, tissue, paper) subsurface-ish (roughness from `surface_roughness`, sheen, slight transmission when low
   scattering, `display_color`); `glowColor` + `glowIntensity` → emissive fluorescence glow (bloom-friendly);
   hover/selected highlight.
4. **Stage**: `plane` → optical **breadboard** (`extent_mm` square, ~12 mm thick, black anodized) with **M6 holes at
   25 mm pitch** (instancing or generated normal/alpha/roughness texture; good up close and far, mipmapped, no moiré),
   chamfered edges, rails/legs optional; look follows background material id (`black_anodized`, `matte_grey`,
   `white_paper`, `spectralon`, `glass`). `none` → subtle infinite grid in a void with a dark gradient backdrop.
   Export `Post` / `Clamp` helpers used by hardware.
5. **StudioEnvironment**: Lightformer studio (key/fill/rim softboxes), soft shadows (AccumulativeShadows +
   RandomizedLight, or ContactShadows if cheaper), background gradient, depth fog. Separate **Effects**
   (`@react-three/postprocessing`): bloom (beams/glows), SMAA, subtle vignette, optional N8AO/SSAO, ACES/AgX tone
   mapping; `quality: 'high'|'low'` prop for weak projectors.
6. **Showcase.tsx** (+ export): self-contained scene with every model (all three source kinds with beams at
   365/488/640 nm, camera with frustum while acquiring, every sample shape on the breadboard) + OrbitControls. For
   QA add `src/three/showcase-main.tsx` + `apps/studio-web/showcase.html` (Vite serves extra html).

## Process
Build, then LOOK: `npx vite --port 5199` in the background; Playwright (chromium installed; a small node script in
`apps/studio-web/scripts/`) screenshots of `showcase.html` at 1440×900 from 3–4 angles incl. close-ups of the
camera and the tunable lamp → `apps/studio-web/screenshots/three/`; view them with your image tool and iterate
until genuinely premium. Headless WebGL: launch chromium with `--use-angle=swiftshader --enable-unsafe-swiftshader`
if needed; wait a few seconds for shadows to settle. `npx tsc --noEmit -p tsconfig.app.json` clean for your files.
Kill the dev server when done. Final report: components/props (incl. new optional props), performance, screenshot
paths, integration notes (BeamVolume, frustum, Effects, Showcase).
