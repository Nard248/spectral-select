# Brief — tracer engineer (geometry, ray optics, light source)

Read `00-common.md` first. Also read `apps/studio-web/src/three/pose.ts` — the browser's orientation
convention; your camera model MUST match it so the live browser preview and the capture show identical framing.

## Files
`src/spectraforge/studio/geometry.py`, `surface.py`, `source.py`, `tracer.py` + tests
`tests/studio/test_geometry.py`, `test_surface.py`, `test_source.py`, `test_tracer.py`.

## geometry.py — analytic solids, exact vectorised intersections (N rays at once)
- Object world transform from `Transform`; rotation must equal three.js
  `new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(x,y,z,'XYZ'))` (R = Rx·Ry·Rz). Test with a
  hard-coded expected matrix for a non-trivial angle triple.
- Primitives (local frames per schema.py docstrings): box (centred), sphere (centred), cylinder (axis +Y,
  centred), droplet = spherical cap (flat base on local y=0, apex y=height; also handle height > base_radius).
- Composite vessels: `petri_dish` = glass base disc + glass annular wall + liquid fill cylinder (fill =
  object's `material_id`, vessel = `vessel_material_id`, defaulting to material id `"glass"` if present);
  `cuvette` = open-top rectangular glass shell + fill box. Use CSG intervals for non-convex parts.
- Background slab `y ∈ [-thickness, 0]`, square `extent_mm` centred at origin, when `background.kind=="plane"`.
- API (refine as needed): `build_solids(experiment) -> list[Solid]` (material index into experiment.materials,
  object index, −2 for background) and a scene query `first_hit(origins, dirs, t_min, exclude=None)` →
  t, outward normal, solid index, exit distance/chord for that solid. Rays may start inside a solid.
  Robust epsilons / grazing rays.

## surface.py
Unpolarised Fresnel R(cos_i, n1, n2) with TIR; Snell (cosines and refracted direction vectors); GGX
microfacet specular BRDF with Smith masking + Fresnel (alpha = `surface_roughness`, clamp ≥ 0.005).
Tests: normal-incidence R=((n1−n2)/(n1+n2))², R+T=1, TIR above critical angle, reciprocity, GGX normalisation sanity.

## source.py — excitation light
- **Beam irradiance per 1 mW** at world points (mW/mm²): source at `position_mm`, axis → `target_mm`; beam
  radius w(z) = D/2 + z·tan α (D = `aperture_diameter_mm`, α = `divergence_half_angle_deg`) — this gives
  inverse-square for a diverging beam. Gaussian (1/e² radius w): I = 2/(πw²)·exp(−2r²/w²) on the
  perpendicular plane; flat-top: 1/(πw²) inside with a few-% soft edge. Multiply by the incidence cosine
  between the local propagation direction (from the cone's virtual apex through the point) and the surface
  normal. Behind the source plane → 0.
- **Line shapes** normalised to ∫=1 (1/nm) on any grid, analytic per-bin (erf) so 1 nm lasers on a 1 nm grid
  are exact: tunable_lamp Gaussian FWHM `line_fwhm_nm`; led_array asymmetric (longer red tail); laser narrow Gaussian.
- **Out-of-band leakage** spectrum per mW in-band, suppressed by 10^−`excitation_filter_od`: tunable lamp →
  a realistic **xenon arc** relative spectrum (tabulated smooth approximation: UV roll-off < ~300 nm, broad
  continuum, NIR lines ~820–1000 nm), expose `xenon_relative_spectrum(nm)`; LED → its own broad tail; laser ≈ 0.
- `drifted_powers(source, steps, rng, enabled) -> list[float]` (Gaussian, sigma `power_drift_pct` %).

## tracer.py — `trace_scene(experiment, width, height, *, pixels=None, toggles=None) -> HitBuffer`
Exact field semantics are in `contracts.py` — satisfy every field.
- Camera rays: pinhole at the lens with thin-lens image distance v = f·s/(s−f) (s = `focus_distance_mm`);
  sensor size = `width_px·pixel_pitch_um·binning` × `height_px·…` µm; the requested width/height may be a
  scaled-down resolution of the same sensor area. Orientation = three.js `Matrix4.lookAt(eye, target, up)`
  with up = +Y, or (0,0,−1) when |dir.y| > 0.999, then roll about the optical axis (identical to pose.ts).
  Row 0 = image top, column 0 = image left as seen by the camera. Test: camera above origin looking down sees
  world +X on image right and world −Z at image top.
- `pixels`: optional flat pixel-index subset (server's single-pixel "explain"); document width/height semantics.
- Layers (K = `MAX_LAYERS`): layer 0 = first surface hit. If its material is CLEAR (rule:
  `scattering.model=="none"` or `mus_prime_500_per_mm==0` or `toggles.scattering` False) refract in (Snell;
  on TIR follow the reflection at most once or terminate — document), traverse the exact chord, refract out,
  continue → next layer. Turbid materials and the background terminate the chain. Fill every per-layer
  field: material, object_index, point_mm, normal, path_mm, thickness_mm, cos_view, cos_view_internal,
  cos_light, cos_light_internal, n_outside, T_view, T_light, irradiance_per_mW, light_clear_path_mm (K,P,M).
- **Fresnel bookkeeping**: `T_view`/`T_light` include ONLY interfaces crossed BEFORE this layer's entry
  interface; the entry interface is applied by the transport kernel. No double counting.
- Shadow rays from each layer's entry point toward the source position: clear occluders multiply Fresnel T
  (two interfaces each) and add path into `light_clear_path_mm`; non-clear occluders → T_light = 0;
  `cos_light <= 0` → irradiance 0. Respect `toggles.shadows`, `toggles.fresnel` (off → T=1, no refraction).
- Per pixel: depth_mm, view_dir, ray_dir, field_angle_rad, pixel_footprint_mm (= depth × pixel angular size /
  |cos_view|), glint_per_mW (GGX specular radiance factor of the source at the first interface, effective
  roughness widened by the source aperture's angular radius to avoid aliasing; `toggles.specular_glints`),
  focus_depth_mm, diagnostics in `info`.
- Performance: lab-replica preset 348×256 with all layers + shadow rays in < 0.5 s on an M4 Pro. Report it.

## Tests
Hypothesis property tests (hit points on surface, unit outward normals); chord lengths vs analytic for
sphere/box/cylinder; CSG annulus; Euler/three.js equivalence; image orientation; beam power conservation
(∫ Gaussian irradiance over a perpendicular plane = 1 mW per mW) and inverse-square at large z; line shapes
integrate to 1; shadowing; camera → glass dish → liquid → stage gives 3 layers with sensible T; every preset
in `presets.PRESETS` traces at ¼ resolution without NaN.
