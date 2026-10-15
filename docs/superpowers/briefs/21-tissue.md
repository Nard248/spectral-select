# W1 brief: tissue optics engineer (chromophore absorption, Mie/Rayleigh scattering, tissue library)

Read `20-v12-common.md` first. Goal: make materials **literature-grade tissue optics**, so that "tissue
covered by collagen" (the user's end target), skin, and mucosa are simulated with real optical properties
instead of hand-tuned numbers.

## 1. `tissue_optics.py` + `data/` tables
Use high-resolution tabulated spectra (at least 250–1000 nm where available, at ≤ 2 nm resolution, interpolated
in log space where appropriate), each in a CSV with a citation header:
- Oxy- and deoxy-hemoglobin molar extinction (Prahl, omlc.org/spectra/hemoglobin/summary.html, 250–1000 nm).
- Water absorption (Hale & Querry 1973 for 200–200 µm, with Segelstein 1981 or Pope & Fry 1997 for 380–700 nm;
  omlc.org/spectra/water).
- Lipid (van Veen et al. 2005, J. Biomed. Opt. 10(5) 054004; omlc.org/spectra/fat) and the melanosome
  ``mua = 519 (λ/500)^-3.5`` 1/cm (Jacques 2013).
- Bilirubin and β-carotene molar extinction (omlc.org/spectra/PhotochemCAD).

Try `WebFetch` on omlc.org first. If you can't fetch a table, reconstruct it from published anchor values
you are sure of. Mark the file `# APPROXIMATE (reconstructed)` and say so in your report. Never present
invented numbers as tabulated data. Validate against known anchor values, for example:
- HbO2 at 415 nm ≈ 524 280 cm⁻¹/M, at 542 ≈ 53 236, at 576 ≈ 55 540.
- Hb at 555 ≈ 54 540.
- Isosbestic points near 500, 529, 545, 570, and 584 nm.
- Water at 980 nm ≈ 0.45 /cm.
Provide `chromophore_mua_per_mm(name, nm, amount)` and `tissue_mua_per_mm(comp: TissueComposition, nm)`
following Jacques 2013 eq. 1. Hemoglobin: mua = ln10·ε·C, with C = 150 g/L / 64 500 g/mol, scaled by blood
fraction B and saturation S. Document the units.

## 2. `optics.py` / `spectra.py`
- `material_optics` adds `tissue_mua_per_mm(material.tissue)` to the background absorption.
- Implement `ScatteringSpec.model == "mie_rayleigh"`.
- Keep every existing behaviour and test unchanged. The current coarse `hemoglobin_shape` preset should
  switch to the new high-resolution table if it stays consistent (document any numeric changes to presets).
- Add a refractive-index dispersion helper: Cauchy for water and tissue, `n(λ)`. It is a utility only;
  the schema keeps a scalar n.

## 3. Tissue material library `presets_tissue.py`
Write `tissue_materials() -> list[MaterialSpec]` with literature-parameterised tissues, citing the source for each:
- Epidermis (melanosome fractions for light, medium, and dark skin).
- Papillary dermis and reticular dermis (Jacques 2013 average-tissue μs'(500)=… with b_Mie and f_Ray;
  blood ~0.2–4 %; water ~65 %).
- Skeletal muscle.
- Adipose (fat).
- Oral and colon mucosa.
- Liver.
- Collagen gel (acellular, low blood) and dense collagen (tendon-like).
- Breast.
- Brain grey matter.
- An Intralipid 1 % phantom with the Michels 2008 / van Staveren 1991 μs(λ) and g(λ).
- An India-ink absorber phantom.
Use the autofluorophores already in `builtin_fluorophores()` (read `presets.py`), at realistic µM
concentrations: collagen, elastin, NADH, FAD, porphyrin, keratin where relevant.
Also write `tissue_presets() -> list[Experiment]` with 3 or more new experiments built on the existing
`tissue_under_collagen` scene structure (copy its source/camera setup):
- `skin_three_layer` (epidermis, dermis, subcutis).
- `mucosa_dysplasia_inclusion` (mucosa with a buried spherical NADH-rich nodule; uses `Inclusion`).
- `vessel_under_collagen` (cylindrical blood vessel at depth under collagen; uses `Inclusion`).
The lead wires them into `presets.py`. Each must validate with `Experiment.model_validate`, and you add a
test that `build_scene` and the preview capture work for them at a small `preview_scale`, once W2's
inclusions land. Until then, test validation only, and run an integration check at the end.

## 4. Validation tests (`test_tissue_optics.py`)
- The anchor values above.
- Unit consistency: mua of whole blood at 415 nm.
- `mie_rayleigh` reduces to `power_law` at f_rayleigh=0.
- Jacques 2013 "generic tissue" μs'(500 nm) and spectral shape.
- The Intralipid μs' at 633 nm against the van Staveren formula.
- A figure script `scripts/studio_tissue_figures.py` writing PNGs to `docs/studio/figures/` (`fig20_chromophores.png`:
  log-scale extinction spectra; `fig21_tissue_optics.png`: mua and μs' of the tissue library). Look at the
  figures and make them publication quality (300 dpi, labelled axes with units, legends).
