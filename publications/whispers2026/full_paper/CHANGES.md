# Camera-ready changes (accepted source preserved unchanged as main_accepted.tex)

Final: main.pdf = 5 pages including references (checked with pdfinfo). No font, margin or template change.

## R1 (real acquisition; "how is the evaluation valid")
- New Sec. 2.6 "Validation against real acquisitions" (text from validation/realdata_paragraph.tex, Fig. 5 = figures/fig_realdata): drops and collagen fitted/held-out against measured data. The collagen result is reported as is (shape only, slightly worse than copying another sample).
- Two-level validity paragraph at the start of Sec. 3: (a) simulator fidelity = linearity tests + real-data check; (b) benchmark validity = ground truth + random / full-spectrum controls.
- Abstract: one clause noting the fit to measured drop and collagen spectra.

## R2.1 (setup summary)
- New Table 1 (single column): selectors, regimes, 5 noise levels, classifiers, controls, seeds, scene size and grid, each with its reason. Numbers from the audit file and reports/ scripts.

## R2.2 (bands / range in the simulator)
- New Sec. 3.1 "Spectral grid": 4 excitations (405/470/488/506 nm) x 141 emission bands (420-700 nm, 2 nm) = 564 for Fig. 6 ("full-564" explained, selection up to 100 is a subset); Table 2 runs use 5 nm step = 57 bands, 228 features; simulator accepts any grid.
- Fig. 6 caption states k is out of 564 bands.

## R2.3 (6 -> 5 pages)
- Removed old Fig. 4 (library dialog) and old workbench screenshot; replaced by one full-width Studio figure (see author request).
- Tightened Introduction, Secs. 2.3, 2.5, 3, 4, Discussion first paragraph (limitations paragraph untouched except one appended clause).
- Shrunk Figs. 1, 2, 3, 5, 6 widths; abbreviated venue names in references.

## Audit corrections
- Eq. (2): now F_obs = F exp(-s_ie A_ex) exp(-s_re A_em(lambda)), with A_ex, A_em defined and s as dimensionless strengths (physics.py:81-85); inner-filter/reabsorption text updated.
- Nonlinear-only information: Table 2 six-seed run values (AE-MLP +0.014, conv-AE -0.001, PCA -0.016) replace +0.030/-0.021.
- Spatial texture: 3-seed values (per-pixel 0.52, random 0.71, oracle bands 0.75); removed claim that a spatial model reaches the oracle; learned selectors (texture 0.66-0.69, CNN 0.53 from a separate finer config, random 0.58) do not beat random; spatial selection stated as open.
- Fig. 6 right panel regenerated from the 3-seed log only (spatial-CNN bar dropped) via reports/paper_figures.py::fig_results (that script was edited accordingly; it writes to repo paper/figures, I restored that tracked copy and copied the output into figures/).
- "Variance ranker barely beats random" replaced: it does no better than, usually below, random (0.40 vs 0.45 realistic).
- Table 2 (battery) caption: "bootstrap 95% CI excludes zero", Wilcoxon p=0.16 (reabsorption, marginal), p=0.03 (clean); verdict "marginal AE gain"; text says "marginally overtakes".
- build_dataset: stated as a study script in reports/, not library API; library call is render().
- FPbase: library importer from a downloaded payload, not exposed in the workbench.
- References [1]-[8] now cited: Lakowicz, Valeur (photophysics), Jablonski (excited-state picture), Kasha (rule), Bro (PARAFAC), Jolliffe (PCA), Hinton (autoencoder), BS-Nets (band selection). None deleted.
- fig_cube4d.png was missing from figures/ (accepted source would not compile); regenerated with reports/paper_figures_4d.py::fig_cube4d.

## Author request (Studio figure)
- Old Figs. 4-5 replaced by one figure* (Fig. 4) with panels (a) 01-design.png, (b) 04-explore.png (copied to figures/fig_studio_design.png, fig_studio_explore.png). Caption states it is the in-development workbench and that the study used the core generator, not Studio physics.
- Sec. 2.5 workbench sentences rewritten for the 3D workflow; one clause appended to Discussion limitations noting Studio's 3D Monte Carlo transport and acquisition-order photobleaching.

## Kept as is
Title, authors, affiliations, acknowledgment/AI disclosure, early-access wording, limitations content.

## Open items for the author
- Setup table lists LR among classifiers (used in the Table 2 suite; Fig. 6 scripts use k-NN, RF, MLP only; flagged in the table).
- Fig. 6 panel text is small; Studio panels are screenshots at half width.
- Collagen real-data result is weaker than the drops; wording kept from the evidence agent.
