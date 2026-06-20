# Review response changelog — Communications AI & Computing manuscript

**Reviewer:** Narine Sarvazyan
**Source PDFs:** `Unsupervised_DL_ME-HSI_NAS June 11.pdf` (Review 1, 31 marks) and
`Unsupervised_DL_ME-HSI_NAS June 11 2pm.pdf` (Review 2, 40 marks — superset).
**Paper edited:** `CommsAIComputing_Overleaf/` (sections + main.tex + references.bib).
**Date:** 2026-06-13. Compiles clean, 16 pp, 0 undefined citations/references.

Every mark from both reviews is listed below with: what the reviewer wrote, what
was changed, where, and why. Review 2 contains all of Review 1's marks plus 9 new
reference-related ones, so the numbering follows Review 2 (#1–#40).

---

## Abstract (page 1)

| # | Reviewer mark | Change made | Where | Why |
|---|---|---|---|---|
| 1 | highlight + note "datasets" | "Processing ME-HSI is challenging" → "Processing ME-HSI **datasets** is challenging" | `abstract.tex` | author-confirmed wording |
| 2 | note "data" | "correlations present in ME-HSI." → "…present in ME-HSI **data**." | `abstract.tex` | author-confirmed wording |
| 3 | highlight last sentence + replacement note | final sentence replaced with reviewer's: *"The proposed approach allows to mitigate the computational and hardware challenges of ME-HSI by dramatically decreasing acquisition complexity while boosting its discriminative power."* (the collagen-result sentence was kept) | `abstract.tex` | reviewer's preferred closing sentence. **Note:** reintroduces "dramatically/boosting" — overrides the earlier grounded-tone pass, per your approval |

## Introduction (page 1)

| # | Reviewer mark | Change made | Where | Why |
|---|---|---|---|---|
| 4 + 9 | note "data" + strikeout "cameras" | "RGB or multispectral **cameras**" → "RGB or multispectral **data**" | `introduction.tex` l.7 | reviewer struck "cameras", noted "data" |
| 5 | note "photons" | "re-emit at longer wavelengths" → "re-emit **photons** at longer wavelengths" | `introduction.tex` l.9 | reviewer addition. *Interpretation:* placed "photons" after "re-emit"; flag if she meant "light photons" |
| 7 + 8 | note "data points" + strikeout "spectral layers" | "yields 192 **spectral layers** per pixel" → "yields 192 **data points** per pixel" | `introduction.tex` l.19 | reviewer terminology |
| 11 + 12 | strikeout "frame." + note "ME-HSI dataset." | "184 million data points per **frame**." → "…per **ME-HSI dataset**." | `introduction.tex` l.19 | reviewer terminology |
| 10 | strikeout "e" | (subsumed by the l.19 rewrite; single stray character, no separate action) | — | ambiguous single char |
| 6, 17 | empty Text annotations (no note) | no action — empty insertion markers | — | nothing to apply |
| 13 | "add Muselimyan 2016 + Gil 2017" | `\cite{muselimyan2016,gil2017}` added after "…appear identical under reflectance imaging but possess distinct fluorescent signatures." | `introduction.tex` l.10 | reviewer's group references |
| 14 | highlight "[3]" + "remove here, ok in next paragraph" | removed `\cite{lakowicz2006principles}` after "re-emit … wavelengths"; kept the later one (applications sentence) | `introduction.tex` l.9 | duplicate citation |
| 15 | "add Swift 2018" | `\cite{...,swift2018}` added after "…variety of different applications" | `introduction.tex` l.15 | reviewer's group reference |
| 16 | "add Asfour 2018 + Guan 2018" | `\cite{...,asfour2018,guan2018}` added to "…studied for conventional 3D HSI" | `introduction.tex` l.27 | reviewer's group references (wavelength selection / unsupervised HSI) |
| 18 | highlight "current" + "to the best of our knowledge" | "To the best of **current** knowledge" → "…of **our** knowledge" | `introduction.tex` l.32 | reviewer wording (also applied at l.42, discussion, conclusion for consistency). **Note:** reintroduces "our" — overrides earlier de-"we" pass, per your approval |
| 19 | "add space before (" | "autoencoder(CAE)" → "autoencoder (CAE)" | `introduction.tex` l.39 | typo |
| 20 | "(lichen samples and collagen sponges)" | "datasets, lichen samples and collagen sponges, that differ" → "datasets **(lichen samples and collagen sponges)** that differ" | `introduction.tex` l.41 | reviewer punctuation |

## Methods / Experimental Setup (page 6)

| # | Reviewer mark | Change made | Where | Why |
|---|---|---|---|---|
| 21 | note "data" (preprocessing line) | minor — covered by the "ME-HSI data/datasets" standardization above; no separate wording change made to avoid an unclear edit | — | ambiguous target; left as-is (flag) |
| 22 | "add Chilingaryan WHISPERS 2025 after Armenia" | `\cite{chilingaryan2025whispers}` added after "…National Academy of Sciences of Armenia." | `experimental_setup.tex` l.42 | reviewer's lichen reference |
| 23 + 24 | strikeout "prepared at three distinct crosslinker concentrations" + fabrication note | replaced with reviewer's exact text: *"Collagen sponges were fabricated from purified Type I collagen derived from bovine tendon followed by chemical crosslinking with glutaraldehyde at three different concentrations (0.002, 0.0075 and 0.013%)."* | `experimental_setup.tex` l.70 | reviewer's precise fabrication detail |

## Results (pages 10–12)

| # | Reviewer mark | Change made | Where | Why |
|---|---|---|---|---|
| 27 | "I will leave only 3 digits" | Table I accuracies/F1/κ rounded from 4 → **3 decimals** (0.9523 → 0.952, etc.) | `results.tex` (tab:main_results) | reviewer formatting |
| 30 | highlight "bands(Table" | "192 bands(Table V)" → "192 bands **(Table V)**" | `discussion.tex` l.11 | missing space |
| 31 | highlight "5.8%" + "better explain what this number means" | "an improvement of 5.8% points over the baseline" → "**5.8 percentage points higher than the 79.8% full-spectrum baseline (i.e., 85.6% versus 79.8% accuracy)**, while retaining only 30 of the 158 bands (81% data reduction)" | `results.tex` l.287 | clarity |
| 32 | highlight Fig.10 caption + "indicate this is lichen dataset" | caption → "Mean wavelength importance score **for the lichen dataset**, across the 16 above-baseline configuration groups…" | `results.tex` (fig:wavelength_heatmap) | clarity |
| 33 | highlight Fig.11 caption + "place next to Fig.12" | collagen envelope kept as a **standalone** column-width figure (same shape/size as the lichen envelope) and positioned immediately after the collagen heatmap, so it appears on the page right after / near the heatmap. *(Two earlier attempts were reverted per author feedback: (a) merging both into one two-panel figure looked weird; (b) a `\FloatBarrier` forced same-page placement but left an empty half-page — removed.)* Exact same-page placement is left to typesetting; full-width floats and the lichen heatmap compete for the page. | `results.tex` | envelope standalone, near the heatmap, no empty page |

## Discussion (page 13)

| # | Reviewer mark | Change made | Where | Why |
|---|---|---|---|---|
| 34 | highlight "such as chlorophyll or lignin)" + "leave specific fluorophores out" | removed "such as chlorophyll or lignin" → "…to specific fluorophores) requires…" | `discussion.tex` l.87 | reviewer request |
| 35 | "add Chilingaryan SPIE 2025 after [50]" | `\cite{meysurova2021optical}` → `\cite{meysurova2021optical,chilingaryan2025spie}` | `discussion.tex` l.87 | reviewer's group reference |
| 36 | highlight "current" + "to the best of our knowledge" | "of current knowledge" → "of our knowledge" | `discussion.tex` l.127 | reviewer wording |

## Back matter (pages 14–15)

| # | Reviewer mark | Change made | Where | Why |
|---|---|---|---|---|
| 37 | highlight Data Availability + "put on Zenodo + link" | "available from the corresponding author on reasonable request." → "**The collagen sponge dataset … is openly available on Zenodo at https://doi.org/10.5281/zenodo.20677504. The lichen dataset is available from the corresponding author on reasonable request.**" | `main.tex` | reviewer + you supplied the collagen DOI. **Note:** only the collagen DOI was provided; lichen kept as "on request" until a DOI exists |
| 38 | "and Kristina Ghahramanyan" | added "**Kristina Ghahramanyan for collagen sponge data acquisition**" to the Acknowledgment | `main.tex` | you specified the credit |
| 39 | "include several references from our group" | satisfied by the additions in #13, #15, #16, #22, #35 (7 group papers) | `references.bib` | reviewer request |
| 40 | "did you check all the references for being real ones?" | **Sanity-checked:** 59 cited refs all resolve (0 undefined), no duplicate keys, all entries have title+author. The 7 newly added are verified real Sarvazyan-group papers. **Recommendation:** a final external (database) verification of the less-common legacy entries is advised — see note below | `references.bib` | reviewer request |

## Figures / formatting

| # | Reviewer mark | Change made | Where | Why |
|---|---|---|---|---|
| 26 | "make all figure fonts IDENTICAL" | **Done** in the figure-unification pass — all plots use one sans-serif (DejaVu Sans / Helvetica) at consistent sizes | figures + `main.tex` | reviewer request |
| — | (author bug report) lichen heatmap showed a periodic black checkerboard | **Fixed** — `regenerate_figures.py` snapped emission wavelengths with `round`, which aliased the 5 nm-offset Ex=415 nm grid (455, 465, …) into every-other cell. Changed to `floor` (matching the original `generate_figures.py`), and dropped per-figure max-normalization to use raw mean importance. Heatmap is now smooth. | `regenerate_figures.py`, `figures/wavelength_heatmap.png` | author-reported artifact |
| 25, 28, 29 | "line should start on next page" / "free-hanging two-liner" / "shift figures so no free-hanging lines" | added `\clubpenalty=\widowpenalty=\displaywidowpenalty=\brokenpenalty=10000` so LaTeX no longer leaves single orphan/widow lines at column/page boundaries; the merged collagen figure also removed one floating small figure | `main.tex` preamble | reviewer request |

---

## New references added (`references.bib`)
All real Sarvazyan-group papers, formatted as BibTeX:
`muselimyan2016` (PLoS ONE), `gil2017` (J. Biophotonics), `swift2018` (Heart Rhythm),
`asfour2018` (Biomed. Opt. Express), `guan2018` (J. Med. Imaging),
`chilingaryan2025whispers` (WHISPERS), `chilingaryan2025spie` (SPIE Medical Imaging 13410).

## Production technical check (post-submission, 2026-06-15)
The journal's technical check flagged three floats that had captions/labels but no
in-text citation, so the typesetter had no anchor for placement:

| Flagged | Float | Fix |
|---|---|---|
| Table II | `tab:dataset_collagen` (Collagen Sponge Dataset Specifications) | added "Table~\ref{tab:dataset_collagen} summarizes the collagen sponge dataset specifications." before the camera/grid sentence (`experimental_setup.tex`) |
| Figure 3 | `fig:lichen_rgb` (visual appearance of lichens) — was only referenced inside Fig. 8's caption, which does not count | added "The lichen species used in this study are shown in Figure~\ref{fig:lichen_rgb}." to the lichen dataset intro |
| Figure 5 | `fig:collagen_roi` (collagen ground-truth / ROI layout) | added "(Figure~\ref{fig:collagen_roi})" to the $3\times3$ grid annotation sentence |

Verified: all 12 figures and all 7 tables now have ≥1 body-text `\ref`; recompiled
clean, 16 pp, 0 undefined references/citations.

## Open items / flags for the author
1. **Tone overrides (#3, #18, #36):** the reviewer's wording reintroduces "our knowledge"
   and promotional "dramatically/boosting" that the earlier pass had removed. Applied per
   your approval — revisit if the journal flags promotional language.
2. **Lichen data DOI (#37):** only the collagen Zenodo DOI was available; if/when the
   lichen cubes are deposited, update the Data Availability statement.
3. **Ambiguous micro-edits (#5, #21):** "photons" placement and the page-3 "data" note are
   best-effort interpretations — quick to flip if not what the reviewer intended.
4. **Reference realness (#40):** structural checks pass; a full external verification of
   legacy citations is still recommended before final submission.
