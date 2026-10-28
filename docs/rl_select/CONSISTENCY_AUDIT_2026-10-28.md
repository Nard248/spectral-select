# Dataset descriptions across our manuscripts: consistency audit (2026-10-28)

Manuscripts compared:
- **RL**: `publications/rl_band_selection/main.tex`
- **ATT**: `publications/me_attention/main.tex`
- **SR**: Scientific Reports autoencoder revision, `publications/scientific_reports/revision_2026_09/manuscript/revised/`.
  CommsAIComputing (`commsai_computing/submission/`) has the same dataset text.
- **WH**: WHISPERS 2026 camera-ready
- **GEN**: CODASSCA generalization paper

The reference facts are those established in the 2026-10-22 audit (`docs/datasets/audit_2026-10-22/`): `.im3` headers,
Zenodo records 18640119 and 20677504, and pixel registration of our masks against the deposited masks.

## Agreements (no action)

| Fact | Value | Where stated |
|---|---|---|
| Instrument | WheeLED (Mightex) LED source + Nuance FX tunable-filter camera, emission 420–720 nm in 10 nm steps | RL, ATT, SR |
| Lichens 1 | Zenodo 18640119 palette, columns 466–1390, 1040 × 925 px, 8 excitations 310–430 nm, 192 pairs, 4 classes × 4 pieces (*R. sinensis* lower/upper surface, *F. soredica*, *R. fraxinea*) | RL, ATT; SR has the same image and species |
| Collagen | Zenodo 20677504, 9 sponges in a 3 × 3 grid, columns = glutaraldehyde level, 6 excitations 310–400 nm | RL, ATT, SR |
| Collagen pair counts | Pepsin 158 (30 nm cutoff), Acetic 149 (40 nm cutoff). The SR/CommsAI collagen data = Pepsin (158 pairs, 39,970 px) | RL, ATT, SR |
| Lichens 2 | separate unpublished acquisition 2026-01-09, 256 × 348 px, 210 pairs, 5 classes, names not recorded | RL, ATT |
| Dataset citations | identical authors, titles and DOIs | RL, ATT, WH |

## Discrepancies

| # | Issue | Where | Fix |
|---|---|---|---|
| D1 | **Lichens 2 scope.** The ME-Attention study now uses the 3-class subset `lichens2_sub3` (rows 2–4 × columns 1–5; part of the full set is suspected mislabelled). The RL paper still uses all 5 classes. | RL vs ATT (after the sub3 update) | **Decision for the user.** Either rerun the RL study on the subset (remote GPU), or keep 5 classes in RL and say why. Both papers must state which version they use. |
| D2 | **Dataset names differ.** RL uses Collagen-AA / Collagen-P; ATT uses Collagen Acetic / Collagen Pepsin. | RL, ATT | Use "Collagen Pepsin" / "Collagen Acetic" in both. |
| D3 | RL lacks dates, image sizes, pixel counts, specimens per class and glutaraldehyde concentrations; ATT has them in its Table "Datasets". | RL | Give RL the same dataset table, generated from the data. |
| D4 | **Lichens 2 400 nm exposure error** (metadata 12.5 ms vs header 125 ms, so the block is scaled 10× too high). Stated in ATT, missing in RL, which used the same cache. Effect on RL is expected to be negligible: per-feature log1p + standardisation removes a constant factor except near zero counts. Not yet verified. | RL | Add one sentence; verify by rescoring the stored sets with the corrected block. |
| D5 | **The RL spec still says Lichens 1 and 2 "image the same lichen specimens".** This is false: they are separate acquisitions. The paper text was already corrected; the spec was not. | `docs/rl_select/SPEC.md` §2 | Correct the spec (done in this pass). |
| D6 | **Cutoff stated as 40 nm while reporting 158 collagen pairs.** 158 pairs implies the 30 nm Pepsin cutoff. | SR / CommsAI methodology ("Δλ = 40 nm … ±40 nm") | SR revision: say 30 nm for the collagen data, or give both. |
| D7 | **Spectral "resolution".** The lichen table says ~12.5 nm and the collagen table says 10 nm, for the same camera. | SR | Use "10 nm step" for both. Give the LCTF bandwidth only if it is known. |
| D8 | **Collagen source.** "Type I collagen derived from bovine tendon" vs "extracted from bovine trotters". | SR (internal) | Ask Kristina Ghahramanyan. |
| D9 | **Camera manufacturer.** SR says PerkinElmer; ATT says CRi. Nuance FX was made by CRi, later Caliper/PerkinElmer. | SR, ATT | Write "Nuance FX (CRi; now PerkinElmer)" in all papers. |
| D10 | **SR 9-band emission list** ("emission peaks spanning 500–680 nm") uses the mislabelled emission axis for the 385–430 nm cubes. | SR | Convert to recorded wavelengths (offsets in `src/me_attention/data.py`, `RECORDED_WL_OFFSET`). |
| D11 | **Collagen column → concentration mapping** (left to right = 0.002 / 0.0075 / 0.013 %) is not confirmed. | ATT (flagged), SR (asserted) | Confirm with Kristina; then remove the flag or fix SR. |
| D12 | **Collagen Acetic plays different roles.** It is a development set in ATT and a full benchmark target in RL. | RL, ATT | Fine, but each paper should say it explicitly (RL currently does not). |

| D13 | **Collagen 365–400 nm blocks are about 10³× brighter than 310–340 nm** (class-mean EEMs, `figures/proto/fig_datasets`); the lichens show no such step. This fits the LED powers for 365–400 nm being recorded in µW instead of mW (1.3–11 vs 4–15 mW for the same LEDs in the lichen data): power normalisation would then inflate exactly those blocks by ~10³. Pepsin is also ~10³× brighter overall than Acetic (365 nm: 5e6 vs 7e3). Classifiers are unaffected (per-feature log1p + standardisation removes per-feature scale). Absolute EEM plots, cross-excitation comparisons and unmixing on raw intensities are affected. | processed collagen files | Confirm the units with Kristina; re-normalise from the raw files if confirmed. Until then, plot collagen per excitation or say in the caption that the scale is not comparable. |
| D14 | **Collagen class names differ in the caches** ("Class 2/3/4" for Pepsin vs "Class 1/2/3" for Acetic). Both run left to right by column. | processed collagen masks | Cosmetic; papers use GA levels. |
| D15 | **The Lichens 1 mask covers only a central rectangle of each piece**, not the whole piece. | Lichens 1 mask | State it in the data section, so the maps aren't misread as segmenting whole pieces. |

## Resolved by this audit

- **Collagen 158 vs 149 pairs:** Pepsin was processed with a ±30 nm second-order window (310 nm: 590–650 removed) and Acetic with ±40 nm (580–660). The low-emission edge also differs by one bin at 385 and 400 nm. Together these explain the 9-pair gap. See `figures/proto/dataset_table.csv`.

## Still open (unchanged from the 2026-10-22 audit)

- Taxa of the Lichens 2 classes, who annotated them, and whether any pieces share thalli with Lichens 1.
- Units of the collagen LED powers (µW vs mW).
- How many thalli or collections the four pieces of each Lichens 1 class come from.
