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

## Author request (acknowledgment)
- Removed the United States R44 HL120511 award. Text now reads exactly: "Financial support from the European Union NAS SAR-101087403 award is gratefully acknowledged." AI-disclosure sentence unchanged.

## Author request (collagen error figure)
- Fig. (real-data validation) redesigned as three panels: (a) drop at held-out excitations; (b) real collagen ROI (pepsin ROI 4, split D, the median-SAM case of 24) measured vs twin-predicted spectra at held-out 310/325/340 nm with a residual strip (measured - predicted, per-peak normalised); (c) compact SAM bars (twin / empirical copy / nominal / flat).
- New error numbers over all 24 collagen cases (2 datasets, 6 ROIs, 4 held-out splits), mean +- SD: SAM 8.4 +- 1.8 deg, shape R2 0.83 +- 0.13, relative RMS 0.097 +- 0.023 of the peak. Computed by validation/run_collagen_error.py (same fits/seeds as run_validation.py; per-excitation gain refit; pooled R2 and rms_rel from calibration._diagnostics); saved under collagen.error_twin_corrected in validation/validation_numbers.json (no keys removed; representative-case curves in validation/collagen_rep_curves.json). Figure by validation/make_fig_realdata.py.
- Sec. 2.6 text and caption updated; caveats kept (throughput correction, gain step, emission shape only, empirical copy slightly better).
- To stay at 5 pages: Figs. 1, 2, 3 and the benchmark figure shrunk; the benchmark figure moved to before Sec. 2.6 in the source (so it is Fig. 5 and the real-data figure is Fig. 6); a few redundant sentences removed in Secs. 2.2, 2.4, 2.5, 3, 4 and Discussion (no content, citation or limitation removed).

## Kept as is
Title, authors, affiliations, acknowledgment/AI disclosure, early-access wording, limitations content.

## Open items for the author
- Setup table lists LR among classifiers (used in the Table 2 suite; Fig. 6 scripts use k-NN, RF, MLP only; flagged in the table).
- Fig. 6 panel text is small; Studio panels are screenshots at half width.
- Collagen real-data result is weaker than the drops; wording kept from the evidence agent.

## Layout / readability pass (2026-10-14)
- Revision text cut by about 36 % (804 -> 512 words over Sec. 2.6, Sec. 3 intro, Table 1, Sec. 3.1, Fig. 4/5/6 and Table 2 captions); every number and caveat kept; the workbench clause in the Discussion shortened; the supervisor-written limitation paragraphs untouched.
- Figures enlarged: Fig. 1 to 0.96 of text width; Fig. 2 and Fig. 3 regenerated for full column width with larger fonts (Fig. 3 reduced to the two panels the text uses: class labels and variance-vs-informativeness scatter); Fig. 4 uses cropped screenshots (fig_studio_*_crop.png; originals kept) at about 2x the content size; Fig. 6 regenerated (validation/make_fig_realdata.py) as a full-width 3-panel figure* with 8 pt fonts.
- Fonts: body text kept at the template's \ninept (9 pt), as in the accepted version; a 10 pt body does not fit 5 pages with the larger figures. Captions are 9 pt (\small) via a local \@makecaption redefinition in main.tex; Table 1 uses \small (9 pt). stywhispers.sty untouched.
- Float spacing (\textfloatsep, \floatsep, \intextsep, \dbltextfloatsep, \dblfloatsep) reduced to about 8-9 pt to remove stretched gaps; paper stays at exactly 5 pages.
- Regenerated main_diff.tex/.pdf and the camera_ready_upload files (PDF, changes-marked PDF, Overleaf zip).
- Final polish: Fig. 5 regenerated as figures/fig_results_v2.png/.pdf (validation/regen_fig5_results.py; 7.0x2.0 in, serif, same data, full text width; original kept); Fig. 4 slightly narrowed; fixed spacing everywhere (aggedbottom, parskip 0, fixed float/section skips); bibliography: [15] now 'under review, 2026', added Zenodo dataset refs [16,17] cited in Sec. 4; still exactly 5 pages; diff, upload PDFs and Overleaf zip regenerated.
