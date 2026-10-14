# E2 findings: cover-thickness response with literature tissue optics

Setup: `tissue_under_collagen`-type scene on the W1 tissue materials (dense-collagen cover, Jacques 2013 optics, over a
papillary-dermis NADH/FAD/collagen/elastin layer, reticular-dermis substrate); excitations 340/365/400/430 nm; 13 thicknesses
0-2 mm x 3 replicate seeds; preview scale 0.25, 2e6 excitation + 2e6 emission photons, 10 % preview MC fraction; central 3x3 px.
Blood study: B = 0/1/3 % at t = 0.1/0.25/0.5/1 mm, 400 and 430 nm. Wedges: gentle (40 mm, 2 seeds) and steep (8 mm).
Figures: `figures/E2_{penetration,fraction_distortion,blood,wedge}.{png,pdf}`; script `experiments/E2_cover_thickness.py`
(+`E2_figures.py`, which also writes `wedge_binned.json`). Runtime of the full run 51 min.

(a) Penetration depth
- delta_eff (MC exponential fit, 95 % bootstrap CI): 340 nm 0.073 [0.057, 0.084] mm; 365 nm 0.105 [0.070, 0.132]; 400 nm 0.199 [0.164, 0.199]; 430 nm 0.358 [0.219, 0.436].
- Diffusion prediction 1/sqrt(3 mua (mua+mus')), two-way (excitation and emission path): 0.068, 0.096, 0.300, 0.496 mm. Ratio MC/diffusion = 1.07, 1.10 at 340/365 nm (inside the CIs), 0.66 and 0.72 at 400/430 nm (CIs exclude 1).
- So the simple diffusion estimate is good where absorption dominates (mua/mus' ~ 0.5) and overestimates delta by 1.4-1.5x where scattering dominates (mua/mus' ~ 0.04-0.1), where diffusion theory is not valid (anisotropic scattering, mua << mus', finite beam/exit geometry in a layered 3D MC, collagen cover thickness comparable to the transport mean free path). The fit window and R^2 (0.90-0.97) show the decay is not perfectly mono-exponential at 400/430 nm; delta_eff is a window-dependent summary there.
- The buried-layer signal falls ~5 decades over 2 mm at 430 nm and ~3 decades over 0.5 mm at 340 nm (MC noise floor ~1e-5 of S(0), clipped in the figure).

(b) Autofluorescence fraction / crossover
- Collagen's own autofluorescence is half of the total (50 % crossover) already at t = 0.027 / 0.034 / 0.045 mm for 340 / 365 / 400 nm excitation (10 % crossover at 5-8 um): at UV excitation any cover thicker than ~50 um is dominated by its own emission. At 430 nm the crossover is 0.55 mm (10 %: 0.15 mm) because collagen barely absorbs there.
- The cover fraction is ~1 for t >= 0.3 mm (UV) / >= 1.5 mm (430 nm).

(c) Spectral distortion
- Emission centroid shifts blue by 12-13 nm (340/365), 29 nm (400) once the cover dominates (this is the collagen emission replacing the NADH/FAD emission, not a transport artefact); at 430 nm the shift reaches -65 nm after the crossover. Plateau values are an estimate of the cover-vs-buried spectral difference.
- Blue/red ratio relative to t = 0 grows 1.5-2.5x already at 25-50 um and to 2e2-3e2 (UV) because the buried layer's red/green emission is lost first; at 430 nm the ratio is flat until ~0.3 mm, then diverges (red band extinguished; values at t >= 1.25 mm are noisy, bars = SE).
- Blood in the cover (B = 1 %, 3 %) at t = 0.25 mm: total buried fluorescence drops to 0.30 / 0.09 (400 nm) and 0.32 / 0.057 (430 nm) of the B = 0 value; the S_B/S_0 ratio shows the HbO2 Soret/Q-band dips at ~542 and 577 nm (figure panel b) and the red side is far less attenuated than the blue-green side; at 430 nm (in the Soret band) 3 % blood removes 97 % of the signal. Emission-centroid changes with blood are small at 400 nm (+6/+7 nm) and variable at 430 nm.
- Caveat: panel (c) of the blood figure is a ratio of two small bands with SEs of up to 100 % at t >= 0.5 mm (3 seeds); trend (monotone increase with B and t) is robust, magnitudes are not.

(d) Wedge phantom (3D lateral transport vs uniform blocks)
- Binned (0.2 mm), seed-pooled 3D-wedge / 1D-uniform-block ratio of the buried-layer fluorescence: mean log-ratio gentle wedge -0.03 +- 0.03 (400 nm) and -0.10 +- 0.04 (430 nm); steep wedge +0.12 +- 0.03 and +0.13 +- 0.05 (n = 188 / 36 pixels). The ratio stays within a factor ~0.6-1.6 over the whole 0-2 mm range with no systematic trend with local thickness.
- The 1D uniform-block response therefore predicts the wedge response to within ~10-15 % (systematic, bins) in the gentle wedge (slope 0.05 mm/mm); the steep wedge (slope 0.25) shows a small positive bias (+12-14 %; sign consistent with lateral transport, but not attributed because the effect is small), and both are comparable to the per-pixel MC noise (rms log-ratio 0.2-0.5 per pixel). The fitted lateral blur sigma was at the bounds of the search grid (0 or 3 mm) for most cases, i.e. not identifiable above the MC noise at 2 seeds x 2e6 photons: this is a null result for lateral blur, not a measurement of it.

Caveats
- explain_pixel vs capture consistency: relative difference of band sums 1e-9 (exact).
- Preview-scale MC (10 % photon fraction) and 3 seeds: 95 % CIs on delta_eff are 20-60 % wide; wedge per-pixel noise is large.
- The cover fraction uses the engine's own component bookkeeping (collagen cover dyes vs buried dyes), so the "autofluorescence fraction" is model-defined, not measurable in real data without the cover alone.
