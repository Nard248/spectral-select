# RL band-selection paper — redesign plan (2026-10-28)

## Why
The 6-page draft is numbers-only. A reader never sees the samples or the spectra, never sees the policy act, and
never sees what a selected set produces. The user's request: show the datasets, show what the RL pick yields under
a variety of downstream algorithms (classification, clustering, unmixing), and make all papers agree on the data.

## New story, one sentence per section
1. **Problem**: a cheap instrument can measure only a few (excitation, emission) pairs. Which ones should it measure
   on a new sample type, without labels?
2. **Data**: four datasets, what they look like, and where in the EEM their classes differ.
3. **Method**: a policy that builds a set and may undo choices, shown on a real episode.
4. **What a pick produces**: per-pixel maps from 4 classifiers, 2 clustering methods and NMF unmixing, for the RL picks
   vs. label-free, random and supervised references.
5. **Quantitative regimes**: search (S1), label-free transfer (S2), few-shot (S3), training cost, as now.
6. **Honest reading**: where reversibility helps and why. Construction vs. refinement: the gain comes mostly from how the
   reversibly trained policy builds the set, and the swaps matter only at small n. This is being measured on all
   datasets (`results/rl_select/summary/decompose_gain.md`).

## Figures (prototypes in `publications/rl_band_selection/figures/proto/`)

| # | Figure | Status |
|---|---|---|
| 1 | Concept: cube → per-pixel EEM → pick n of M pairs → few-LED instrument; MDP diagram (add / remove / stop) | to draw |
| 2 | Datasets: appearance, specimens, class-mean EEMs, 365 nm spectra (`fig_datasets`) | prototype ✓ |
| 3 | Where classes differ: per-pair Fisher ratio, overlaid with the pairs each method picks (`fig_eem_discriminability` + existing `fig_eem_pairs`) | prototype ✓; overlay to add |
| 4 | A real episode: construction, then swaps, J_rec and held-out BA per step, with add-only beside it (`fig_episode`) | prototype ✓ (Lichens 1, reproduces stored record) |
| 5 | What a pick produces: maps × selection methods × algorithms (`maps_<dataset>_n8`) | prototype ✓ all 4 datasets (BA identical to stored records) |
| 6 | Downstream score vs. n for every algorithm (kNN/LDA/SVM/RF/k-means/GMM/NMF) per selection method | to compute (cheap) |
| 7 | Label-free transfer (existing `fig_transfer`) | exists |
| 8 | Construction vs. refinement decomposition per n and dataset | running |
| 9 | Few-shot (existing) and training curves (existing) | exist |

## Things the prototypes revealed
- **Lichens 2 specimen 1** (top row, class A) is predicted as class B by every classifier, even with all pairs. The
  clustering agrees. This supports the suspected mislabelling of the top row.
- **Collagen clustering follows illumination**, not crosslinker level: k-means/GMM split the sponges by vertical
  position (vignetting visible in the false-colour image).
- **Collagen 365–400 nm blocks are ~10³× brighter than 310–340 nm**: a likely µW/mW unit error (audit D13).
- The **NMF composites are almost uniform grey**: abundances are near-uniform mixtures. Showing 2 abundance maps +
  endmember spectra for one selection would be more informative than a composite per panel.
- **A single repeat/seed is noisy.** On Collagen Pepsin, repeat 0, the transferred RL sets score below one random
  draw. Fig. 5 must either average over repeats in its labels or say which draw it shows. It must not be cherry-picked.

## Decisions (user, 2026-10-28)
- Unmixing = **both**: class/cluster maps as the main outcome figure, plus a dedicated unmixing figure (NMF endmembers
  + abundances, and FCLS with known class endmembers).
- **Journal length** (~10–12 pp); venue to be chosen later.
- **Lichens 2 → 3-class subset** (`lichens2_sub3`) in the RL paper too. Full RL rerun on the remote machine with
  recorded-wavelength caches and 3 policy seeds (`docs/rl_select/HANDOFF_REMOTE_RL_v2.md`, in preparation).

## Construction vs. refinement (measured on all 1152 S2 units; `results/rl_select/summary/decompose_gain.md`)
Gain of add/remove over add-only = construction (the reversibly trained policy builds a different set; Jaccard
overlap with the add-only set is only 0.21) + refinement (the swaps).

| | total | construction | refinement |
|---|---|---|---|
| pooled | +0.026 [0.023, 0.029] | +0.013 | +0.013 |
| n = 3 / 5 | +0.024 / +0.031 | +0.003 / +0.011 | **+0.021 / +0.021** |
| n = 8 / 13 | +0.025 / +0.022 | **+0.018 / +0.020** | +0.007 / +0.002 (n.s.) |
| cross-family | +0.034 | +0.019 | +0.015 |
| Collagen Pepsin | +0.009 | **−0.019** | +0.027 |
| Collagen Acetic | +0.021 | +0.026 | −0.005 |

Paper consequence: "undo actions fix mistakes" is true only at small n. At larger n the benefit is that training with
undo produces a different (better-spread, +0.5 excitations) constructive policy. The pooled total reproduces the stored
S2 gain exactly. The split depends on the policy seed (s0: 0.006 / 0.015; s1: 0.019 / 0.011), so the 3-seed rerun is
needed before final wording.

## Downstream algorithms and unmixing (prototypes `fig_score_vs_n`, `fig_unmixing_*`, current results)
- RL add/remove ≥ add-only for all 4 classifiers on all 4 datasets, but clearly outside the CIs only on Lichens 1.
  Clustering is a tie (k-means), except GMM on Lichens 1 (+0.10). NMF is mixed (loses on Lichens 2). The reversible
  advantage is a classifier-side result. Do not claim it for clustering or unmixing.
- SA (supervised search) is best or near best almost everywhere. CSS beats both RL policies for the supervised
  classifiers at n ≤ 8 on Lichens 1 and on collagen, but collapses for clustering/NMF on Lichens 1.
- Unsupervised NMF endmembers do not map to classes (purity 0.27–0.50). FCLS with training-class mean endmembers is
  clean and nearly selection-independent (Collagen Pepsin held-out BA 0.64–0.82, Lichens 1 0.85–0.95).
- All prototypes use policy seed 0 / repeat 0 for maps and the current (5-class Lichens 2) results. Regenerate everything
  from `results/rl_select_v2/` after the rerun (all scripts take `--results-root` and `--datasets`).

## Decisions needed from the user (answered above)
1. **What "unmixing" means** in your request: per-pixel class/cluster maps (built), linear spectral unmixing
   (abundance maps, NMF, built), or something else?
2. **Venue / length.** This many figures do not fit 6 IEEE conference pages. Options: a journal-length paper
   (~10–12 pages), or a conference paper plus a supplement.
3. **Lichens 2 in the RL paper.** Full 5 classes as now, or rerun on the 3-class subset like the Attention study.
   The rerun needs the GPU machine.
4. **Reruns** (GPU machine): recorded wavelengths for the wavelength-feature variants (S1, with-wl S2), and more
   policy seeds (currently 1–2).
