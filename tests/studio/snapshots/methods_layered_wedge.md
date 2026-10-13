**Simulation.** Multi-excitation fluorescence hyperspectral images were simulated with SpectralForge Studio (version 0.0.0-test, commit 0123456789), a physics-based forward model that combines ray tracing of the scene, Monte Carlo light transport, dye absorption, fluorescence emission and a detector model. The experiment 'Layered wedge phantom' (SHA-256 `5f7c425400c774b6`) was simulated in sample mode.

**Scene.** The scene consisted of 1 object(s) on a 600 mm background plane of black_anodized. Layered block (cover wedge): a layered block of 40 × 40 mm footprint with layers (bottom to top) substrate (tissue_substrate, 4 mm), fluorescent layer (fluorescein_agarose_layer, 0.5 mm) then cover (intralipid_cover, 2 mm to 0 mm along x), centred at (0.0, 0.0, 0.0) mm.

**Optical properties.** Optical properties at the excitation wavelengths (340 nm, 380 nm, 420 nm, 450 nm, 470 nm and 490 nm) are listed below: refractive index n, scattering anisotropy g, absorption coefficient μa and reduced scattering coefficient μs' = μs(1 − g) in mm⁻¹, and dye concentrations. Non-scattering (clear) materials have no g or μs'.

| material | n | g | μa / μs' at 340 nm (mm⁻¹) | μa / μs' at 380 nm (mm⁻¹) | μa / μs' at 420 nm (mm⁻¹) | μa / μs' at 450 nm (mm⁻¹) | μa / μs' at 470 nm (mm⁻¹) | μa / μs' at 490 nm (mm⁻¹) | dyes |
|---|---|---|---|---|---|---|---|---|---|
| tissue_substrate | 1.38 | 0.85 | 0.436 / 2.38 | 0.358 / 2.09 | 1.15 / 1.85 | 0.161 / 1.7 | 0.0858 / 1.62 | 0.0583 / 1.54 | NADH 8 µM and collagen 80 µM |
| fluorescein_agarose_layer | 1.33 | 0.9 | 2.5e-05 / 0.441 | 1.14e-05 / 0.395 | 4.87e-06 / 0.357 | 0.00111 / 0.333 | 0.0241 / 0.319 | 0.0854 / 0.306 | fluorescein 5 µM |
| intralipid_cover | 1.36 | 0.8 | 2.5e-05 / 2.21 | 1.14e-05 / 1.97 | 4.5e-06 / 1.79 | 7.86e-06 / 1.67 | 1.21e-05 / 1.6 | 1.75e-05 / 1.53 | none |
| black_anodized | 1.5 | 0.6 | 58.8 / 2.43 | 52.6 / 2.29 | 47.6 / 2.18 | 44.4 / 2.11 | 42.6 / 2.06 | 40.8 / 2.02 | none |

**Source.** Excitation was provided by Tunable light source (Xe + monochromator) (tunable lamp; line width 10 nm FWHM; gaussian beam of 10 mm diameter, 7° half-angle divergence; clean-up filter OD 6, step-to-step power drift 0.8 % (1σ)) at (-200.0, 240.0, 60.0) mm aimed at (0.0, 3.0, 0.0) mm. The excitation sequence (wavelength, delivered power, exposure) was 340 nm (1.3 mW, 400 ms), 380 nm (3.6 mW, 120 ms), 420 nm (4.4 mW, 80 ms), 450 nm (4.6 mW, 60 ms), 470 nm (4.7 mW, 60 ms) and 490 nm (4.8 mW, 60 ms).

**Detection.** Images were recorded with Nuance-class HSI camera: a 35 mm f/4 lens focused at 330 mm, a 348×256 px sensor (6.45 µm pitch, 4×4 binning, effective 25.8 µm; ccd_bv quantum-efficiency curve, 16-bit, gain 1.5 e⁻/ADU, offset 100 ADU, read noise 8 e⁻, dark current 0.5 e⁻/s, full well 60000 e⁻, PRNU 1 %, DSNU 2 e⁻) and a lctf spectral filter (31 bands in 10 nm steps starting 20 nm above each excitation line; 20 nm FWHM, 415 nm long-pass (OD 5)).

**Physics and Monte Carlo.** The following physics modules were switched OFF: `indirect_bounce`; all others were enabled. The per-material transport Monte Carlo used 200,000 photons per bake. Layered blocks were solved with a 3D layered Monte Carlo (2,000,000 excitation and 2,000,000 emission photons per step, 8 sublayers per layer, lateral cell 0.5 mm, exit cell 0.5 mm, emission nodes every 20 nm, inclusion voxel 0.2 mm). Random numbers were seeded with 0 (the identical seed reproduces the identical result).

**Software.** Simulations ran with Python 3.11.0, NumPy 2.1.3 and Numba 0.63.1 on test-platform. If you use this simulator please cite: N. Meloyan, "SpectralForge: A Physics-Based Simulator for Fluorescence Hyperspectral Imaging," 16th IEEE GRSS Workshop on Hyperspectral Imaging and Signal Processing (WHISPERS), Glasgow, 2026.
