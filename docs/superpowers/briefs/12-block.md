# Brief — block engineer (layered blocks in geometry/tracer + capture orchestration of the 3D MC)

Read `00-common.md`, `10-v11-common.md`, `contracts.py` (whole file; the block section at the end), and
the current `geometry.py`, `geometry_nb.py`, `tracer.py`, `capture.py`, `optics.py`. The mc3d engineer is
writing `transport3d.py` concurrently with this API (code against it, stub it in your unit tests, integrate
the real one at the end):
```python
build_block_geometry(experiment, object_index) -> BlockGeometry
make_voxel_grid(geom, settings) -> VoxelGrid ; make_face_grids(geom, settings) -> dict[str, FaceGrid]
excitation_pass(geom, optics: RegionOptics, launch: BlockLaunch, camera_dir_local, settings, *, n_photons, seed, toggles=None) -> ExcitationTallies
emission_pass(geom, optics_nodes, sources (C,nx,nz,L,S), channels, camera_dir_local, settings, *, n_photons, seed, toggles=None) -> EmissionTallies
```

## 1. Geometry & tracer
- `layered_block` as a solid in `build_scene` (numpy AND numba paths): a convex prism — bottom y=0, four
  vertical sides, top = the top interface plane of the last layer (tilted for wedges); use its
  `world_from_local` transform. It is an opaque/turbid occluder for shadow rays and terminates the layer
  chain. Camera hits on it must expose: block object index, face id ('top', '+x', …) and local (u,v) —
  put them in HitBuffer.info under documented keys (e.g. `info['block_hits'] = {object_index: (pixel_idx,
  face_id, u, v)}`) since the HitBuffer contract is read-only; `material` for those pixels = the top
  layer's material index.
- The existing preset-tracing test must pass for the two new presets.

## 2. Capture orchestration for blocks (`capture.py`)
For every layered block in the scene:
1. `BlockGeometry` / grids from transport3d.
2. Per enabled excitation step (in order): build a `BlockLaunch` by sampling points on the faces facing
   the source (top + sides), weight = beam irradiance at that point (reuse `source.py`) × face area ×
   shadowing by OTHER objects (shadow ray test excluding the block) × drifted power; direction from the
   source (cone geometry as in `source.beam_samples`). Optics per layer at λex (`optics.material_optics`
   on each layer material, toggles respected; line-shape-averaged μa for dyes). Run `excitation_pass`.
3. **Photobleaching per voxel per dye** using the fluence tally × exposure × σ (same model as v1),
   sequentially over the steps → survival volumes; **quenching** Φ_eff per layer; **FRET** per layer from
   the layer's acceptor concentration (constant per layer): donor ×(1−E), sensitised acceptor channel.
4. Sources: channel per (excitation, layer, dye) = absorbed power × (μa_dye/μa_total at λex in that
   layer) × Φ_eff × survival (+FRET); plus a **Raman** channel per (excitation, aqueous layer) from the
   fluence × the water Raman coefficient used in v1, with its own spectral line at `raman_center_nm(ex)`.
5. Emission nodes: every `emission_node_step_nm` covering all camera bands (± passband FWHM) — run
   `emission_pass` once per node with all channels.
6. Block pixels: bilinear sample of the face radiance (nu,nv,Ne,C) at the pixel's (face,u,v); interpolate
   the per-channel transport factor along the nodes onto the fine grid (it is smooth), × the channel's
   emission density e_k(λ) (Raman: its line shape) → spectral radiance; + elastic radiance × excitation
   line shape (+ leakage spectrum through the same elastic factor); the tracer's glint stays. Feed into the
   existing sensor chain unchanged (blur, noise, ADC). Clear objects in front of a block: attenuate by their
   Beer/Fresnel factors as v1 does for layers (document if approximated).
7. Ground truth (NaN off-block): `layer_thickness:<layer>`, `layer_depth:<layer>`, `block_face`,
   `origin_depth_mm`, `origin_layer_fraction:<layer>`; fill `CaptureResult.blocks` and `.volumes`
   (fluence, absorbed, per-channel source volumes as specified in contracts.py).
8. `explain_pixel` for block pixels: components `dye:<dye>@<layer>`, `raman@<layer>`, `elastic`, `glint`,
   `leak_source` from the cached MC tallies — cache the block MC results per (experiment hash, mode,
   scale) in a small LRU so explain/preview/capture don't re-run the MC.
9. Preview uses `volume_mc.preview_fraction` of the photon counts; progress messages keep the v1 format
   and add `simulating <ex> nm (<i>/<n>) · 3D excitation` and `emission node <λ> nm (<j>/<m>) · 3D`.
10. Toggles: fluorescence, absorption, scattering (μs→0), fresnel, photobleaching, quenching, fret, raman,
    elastic_leak, shot/read/… unchanged. `subsurface_spread` is inherent in 3D (ignored, document);
    `secondary_emission` not modelled for blocks in v1.1 (document in the module docstring).

## Tests (`tests/studio/test_block_capture.py`)
Both new presets capture without NaN (preview + small full-res); along the `layered_wedge_phantom` wedge the
buried fluorescein signal increases monotonically as the cover thins (ROI averages at several x), while the
`tissue_under_collagen` collagen autofluorescence component increases with collagen thickness; the
blue/red band ratio of the buried tissue emission changes with cover thickness (spectral distortion);
bleaching order matters for blocks; dark mode gives no block fluorescence; explain_pixel components sum to
the noise-free radiance at an interior block pixel (within 1 %); GT `layer_thickness:cover` matches the
analytic wedge at sampled pixels. Report preview and full-capture timings for both presets.
