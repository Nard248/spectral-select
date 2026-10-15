# W2 brief: inclusions + 3D MC physics engineer (physics-critical)

Read `20-v12-common.md` first, then `transport3d.py`, `transport3d_mc.py`, `block_capture.py`, the block
section of `contracts.py`, and `docs/studio/PHYSICS_VALIDATION.md` (block section). The new contract is
`schema.Inclusion` with `LayeredBlockShape.inclusions` and `VolumeMCSettings.inclusion_voxel_mm`.

## 1. Heterogeneous 3D Monte Carlo
- Keep the exact planar layer interfaces (wedges stay exact). Add a regular Cartesian **inclusion map** over
  the block's bounding box, with voxel size `inclusion_voxel_mm`, built by sampling each voxel analytically
  with k³ supersampling (k=2 is fine). It stores a uint8 region id: 0 means "the layer material at this point",
  and r ≥ 1 means inclusion r. Overlaps: the later inclusion wins (document this).
- Optics arrays become (L + R): layers followed by inclusion regions. Extend `RegionOptics` in a
  backward-compatible way, for example `incl_mua/mus/g` arrays with default empty.
- Photon stepping: the step is limited by the layer planes AND, inside or next to inclusion voxels, by voxel
  boundaries (mcxyz-style). Use dimensionless step bookkeeping, so changing μt across a voxel boundary is
  exact. Inclusions are index-matched: there is no Fresnel at their boundaries. Mixed-voxel optics are
  NOT averaged; the voxel takes its sampled majority id. Document this.
- Blocks without inclusions must give **bit-identical** results to before. Gate the new path, test it, and
  keep the peel-off/MIS estimator working.
- Tallies: the existing layer-conforming voxel tallies stay. Add per-inclusion totals (absorbed power,
  emitted power) and an inclusion-source channel for every dye in the inclusion material
  (`"<inclusion>|<dye>"`), as for layers. Emission sources inside inclusions must be sampled from the
  inclusion voxels themselves. A source volume on the inclusion map (nx', ny', nz') per channel is
  acceptable. Document the choice in contracts.py.
- Performance target: with 4 inclusions at 0.2 mm, at most 1.6× the time of the same block without
  inclusions. Report the timings.

## 2. Capture, ground truth, explain
In `block_capture.py`:
- Inclusion materials take part in bleaching, quenching, FRET, and Raman like layers.
- New GT layers (NaN off-block): `inclusion_mask:<name>` (1 if the camera ray's top-surface hit point
  projects vertically onto the inclusion's footprint), `inclusion_depth_mm:<name>` (depth of the
  inclusion's top below the local top surface), and `origin_fraction:<name>` (the fraction of the escaping
  fluorescence that originated in the inclusion, from the emission tallies).
- `explain_pixel` components gain `dye:<dye>@<inclusion>`.
- Add `BlockInfo`/volume metadata for inclusions so the web can draw them. Extend `CaptureResult.blocks`
  entries additively.

## 3. Validation (`tests/studio/test_inclusions*.py`; report every number)
1. **Null inclusion**: an inclusion whose material equals its host layer gives the host-only result within
   MC noise (same seed, and ideally exactly, if the stepping is equivalent).
2. **Filling inclusion**: a box inclusion covering a whole layer equals replacing that layer's material
   (within MC noise, < 1 %).
3. **Energy conservation** with inclusions: R + T + A + specular = 1 to 1e-9.
4. **Perturbation theory check**: a small absorbing sphere (Δμa) in a homogeneous semi-infinite turbid
   medium under wide illumination. The total change in diffuse reflectance should match the first-order
   perturbation prediction ΔR ≈ −Δμa ∫ Φ(r)Φ*(r) dV, using the MC's own fluence (from the excitation tally)
   and the adjoint, by reciprocity. Within about 10 %, or explain why not.
5. **Spatially resolved reflectance validation (lateral transport)**: for a pencil beam on a semi-infinite
   homogeneous block (no inclusions; μa=0.01, μs'=1 /mm, n=1.4), compare the MC radial diffuse reflectance
   R(ρ) for ρ = 1…10 mm with Farrell, Patterson & Wilson 1992 (Med. Phys. 19, 879) diffusion theory.
   Expect agreement within ~10 % for ρ > 2 mfp'. Produce `docs/studio/figures/fig22_radial_reflectance.png`.
   This validates the lateral transport of the 3D engine, which the 1D checks cannot. Keep it as a slow test
   plus a figure script `scripts/studio_inclusion_figures.py`.
6. **Buried fluorescent sphere**: fluorescence contrast drops monotonically with depth. Produce
   `fig23_inclusion_depth.png` (images and contrast against depth for 3 cover scatterings).
7. Determinism and thread invariance with inclusions; an inclusion straddling the block edge or crossing
   a wedge interface gives no NaN.
