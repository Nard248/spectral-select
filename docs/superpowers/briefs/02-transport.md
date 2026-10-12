# Brief — transport engineer (light transport inside uniform media; the most physics-critical module)

Read `00-common.md` first. `contracts.py` defines `MaterialOptics`, `TransportKernels`, the
`TransportModel.kernels(...)` signature and the radiometric definitions of the fluorescence kernel **G** (mm/sr)
and elastic remission **ρ** (1/sr) — satisfy them exactly. Realistic values in `presets.py`: tissue phantom
μs′≈1.2/mm; lichen 6–10/mm; paper 25/mm; Spectralon 120/mm; black anodized μa≈40/mm.

## Files
`src/spectraforge/studio/transport.py` (+ optional `transport_mc.py` for Numba kernels), tests
`tests/studio/test_transport.py`. Implement your own small Fresnel helper (don't depend on surface.py).

## Requirements
1. **`clear_medium_kernel(optics, ex_nm, cos_in, cos_out, thickness_mm, n_outside=1.0, path_mm=None) -> TransportKernels`**
   — the exact closed form in contracts.py: refract external cosines (`n_outside` → `optics.n`), entry-interface
   Fresnel in (excitation angle) and out (view angle, inside→outside; TIR → 0), 1/n² radiance factor (relative
   to air), `T_in·T_out/(4π n² cosθi')·(1−e^{−κL})/κ` with `L = path_mm` or `thickness/cosθo'`, κ from μt at
   excitation and emission. G on the fine grid `optics.wavelengths_nm` (shape (P, W)); ρ = 0. Stable for κL→0
   (expm1). Tests: dilute limit G → T_in T_out L/(4π n² cosθi'), Beer–Lambert behaviour.
2. **Monte Carlo for turbid media** (Numba `@njit(cache=True)`, parallel where useful): photon packets in a
   plane-parallel slab (thickness d or semi-infinite), Henyey-Greenstein phase function, index-mismatched
   boundaries (`optics.n` vs `n_outside` on top; assume `n_outside` below a finite slab too — document),
   Fresnel at both boundaries with TIR, Russian roulette, implicit capture. Produce depth-resolved
   **excitation fluence φ(z)** for collimated incidence at internal angle θi′ and the **escape function
   η(z, θo)** (radiance leaving toward the camera per unit isotropic source at depth z). Recommended: use
   **optical reciprocity** so one family of collimated-incidence fluence runs gives both legs — derive the
   normalisation carefully and VERIFY numerically against a brute-force forward MC that launches isotropic
   emitters at depth z and tallies escaping radiance into a cone around θo. G(λem) = ∫ φ_ex(z) η_em(z) dz with
   units such that the turbid kernel → `clear_medium_kernel` as μs → 0. Also: **elastic diffuse remission
   ρ(θi, θo)** (1/sr) at the excitation wavelength, and a **radial subsurface-spread profile** (diffuse
   reflectance vs radius, normal incidence, normalised ∫2πr p dr = 1) at a few wavelength nodes.
3. **Speed via bake + cache**: `kernels()` for P ≈ 90k pixels < ~100 ms per (material, excitation) after
   baking; a bake < ~10 s per material at default quality (`n_photons` ≈ 2e5; scale sensibly). Strong
   suggestion: bake **dimensionless universal tables** in similarity variables (transport albedo
   a′ = μs′/(μa+μs′), optical thickness τ, relative index n, cosines) once per relative index, so materials /
   wavelengths only interpolate and geometry/concentration edits never re-bake — or bake per material on
   wavelength nodes; justify the choice. Disk cache `~/.cache/spectralforge/mc/` (overridable; tests use tmp)
   keyed by optics/n/quality/engine version; deterministic seeds; vectorised interpolation over
   (cos_in, cos_out, thickness incl. ∞).
4. **Factory** `get_transport_model(n_photons=200_000, seed=0, cache_dir=None) -> TransportModel`, whose
   `kernels(...)` dispatches on `optics.clear`; fill `mean_depth_mm`.
5. Optional if cheap: `reabsorbed_fraction(optics, em_nm, ...)` = fraction of emitted light absorbed before
   escape (helps the capture engineer's secondary-emission model). Document it.

## Verification (physics credibility IS the product)
- **MCML benchmark** (Wang & Jacques 1995 / Prahl): μa=1/mm, μs=9/mm, g=0.75, n=1, d=0.2 mm, normal
  incidence → Rd ≈ 0.09734, Tt ≈ 0.66096; within ~1% (slow version `@pytest.mark.slow`, quick looser one too).
  Mismatched-boundary case vs known reference values if you know them (van de Hulst / adding-doubling);
  otherwise an independent semi-analytic check — say which.
- **Diffusion limit**: high albedo semi-infinite → fluence ∝ exp(−μ_eff z), μ_eff = √(3μa(μa+μs′)) beyond a few
  mfp; diffuse reflectance vs Farrell/Patterson dipole or Kubelka–Munk.
- Energy conservation R+T+A=1; turbid → clear as μs→0; reciprocity normalisation vs brute force;
  monotonicity (more absorption → less G; thicker → saturating G = inner filter); determinism.
Report every verification as number-vs-reference.
