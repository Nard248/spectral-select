# 02 — Investigation & Findings

This records what we did and what we found, in order, with the evidence. The headline:
**on synthetic ME-HSI the published perturbation-autoencoder selects non-informative bands because
the convolutional autoencoder converges to a degenerate "predict the per-band mean" solution
(reconstruction R ≈ 0); its perturbation-influence is therefore noise.** The selection *principle*
is sound — a spectral autoencoder that actually learns the data recovers the informative bands.

> Important framing (from the project owner): **peak-hitting is not the goal.** What matters is
> whether the selected bands' *neighbourhood* supports **classification**. The findings below use
> classification (KNN macro-F1) as the real criterion; `peak_recovery` is used only as a sharp
> mechanistic probe.

---

## Phase A — the validation harness was initially misleading (and was corrected)

Early runs reported the method "recovers informative bands, especially with realistic spectra." An
**adversarial review caught this as an artifact** and it was withdrawn:

- The ground-truth "informative-band" mask covers **83–93 %** of the emission grid (broad spectral
  tails + the autofluorescence floor + a loose global threshold). So `precision`/`recall`/broad
  `recovered` **saturate**.
- A **uniformly-random** 12-band selector matches or beats the AE on those saturated metrics
  (random measured 1.00/0.98 vs AE 0.92/0.79). The "measured > Gaussian" gap was a **mask-footprint
  artifact**, not method skill.

**Corrections made (already on `main`):**
- Added a **chance baseline** (`sweep.make_random_selector`) — always report it.
- Added a **tight metric** `peak_recovery` (hit the true emission peak) and `mask_coverage` (exposes
  saturation) to `validate_selection`.
- Rewrote `reports/fpbase_validation.py` and `reports/spectraforge_validation_report.py` with the
  correction + baseline. **No claim was placed in any paper.**

Lesson encoded in the harness: never read precision/recovery without (a) a random baseline and
(b) `mask_coverage`.

---

## Phase B — reframing to classification, and the degenerate-AE discovery

We built a **labelled, balanced** 4D ME-HSI classification benchmark
(`reports/classification_experiment.py`): `scenegen.make_labeled_scene` (per-pixel class = argmax
material, balanced because the concentration fields are i.i.d.), then KNN macro-F1 (stratified
split, standardized features) on different band selections. Representative result (3 real FPbase
dyes, 12-band budget, mean over scenes):

| band selection | KNN macro-F1 | peak_recovery |
|----------------|-------------:|--------------:|
| peak-neighbourhood (oracle) | **0.55** | 1.00 |
| variance-ranking | 0.48 | 0.67 |
| all bands | 0.47 | – |
| random | 0.37 | 0.33 |
| **CAE (the method)** | **0.33** | 0.00 |

The CAE selection classifies **worse than random** and far below a trivial variance-ranking. The
information *is* in the peaks/their neighbourhood (the oracle is best). So the AE is failing, not
the validation.

---

## Phase C — root cause (systematic debugging: 5 hypotheses refuted)

Stable empirical fact across every intervention: **`corr(influence, band-variance) ≈ −0.8`** — the
AE places the *most* perturbation-influence where there is the *least* signal/variance.

Hypotheses tested and **refuted** (each left the inversion at ≈ −0.8 and F1 ≈ 0.33):

| # | Hypothesis | How tested | Result |
|---|------------|-----------|--------|
| H1 | Influence normalization (`variance`) divides out peaks | ablate `variance`/`none`/`max` | refuted |
| H5 | Underfit because a 64×64 image is one training chunk | image 64→128→192 (1→9→16 chunks) | refuted |
| H6 | Wrong latent dims perturbed | `dimension_selection` `activation`/`variance`/`pca` | refuted |
| H7 | Sigmoid output saturation | monkeypatch sigmoid → identity | refuted |
| H3 | Global "blind-intensity" normalization | per-pixel spectral-shape normalization | refuted |

**The decisive diagnostic:** per-band reconstruction correlation R between the model's output and
the input is **≈ 0.00 on every band** (mean 0.01) — yet training MSE drops to ~0.004. The model
minimizes MSE by predicting roughly the per-band mean: the cube is mostly ≈0 after global
normalization (sparse), so "predict ~0" wins without learning any spatial/spectral structure. With
R ≈ 0 the latent encodes nothing, so the perturbation-influence is noise (and happens to
anti-correlate with variance).

Contributing architectural facts: the encoder **collapses the emission-band axis**
(`adaptive_avg_pool3d(..., (1, H, W))`, `autoencoder.py:159`) and the data is globally min-max
normalized into a sparse, mostly-zero range. Real data avoids this: it is large (many chunks) and
**dense** (every band carries signal, so predicting the mean is *not* a cheap optimum).

---

## Phase D — the fix direction works (constructive proof)

A plain **per-pixel spectral MLP-autoencoder** (every pixel = one sample → batches of thousands; no
spatial pooling, no band collapse), trained with the *same* perturbation-selection principle:

| metric | CAE (current) | spectral MLP-AE |
|--------|--------------:|----------------:|
| reconstruction R | ≈ 0.00 | **+0.27 … +0.29** |
| corr(influence, signal-variance) | −0.80 | **+0.56** |
| peak_recovery | 0.00 | **0.67 … 1.00** |
| KNN macro-F1 | 0.33 | **0.43 … 0.49** |

It selects EGFP 511, EBFP2 448, mCherry 610 (the true peaks and their neighbourhoods), classifies
like variance-ranking, and its influence **positively** tracks signal. **Conclusion: the failure is
the spatial-CAE architecture/objective on this data, not the validation, normalization, or the
selection knobs.** (Reproduced over scenes + a noise sweep in `reports/cae_vs_spectral_ae.py`; under
very heavy noise *all* methods collapse to chance — the signal is simply gone.)

---

## Artifacts produced (all on `main`)

| File | What it does |
|------|--------------|
| `src/spectraforge/scenegen.py::make_labeled_scene` | balanced labelled scenes for classification |
| `src/spectraforge/sweep.py::make_random_selector` | the chance baseline |
| `src/spectraforge/validation.py` (`peak_recovery`, `mask_coverage`) | the corrected, honest metrics |
| `reports/classification_experiment.py` | KNN-F1 on selections vs baselines (the core benchmark) |
| `reports/cae_vs_spectral_ae.py` | consolidated CAE-vs-spectral-AE + noise sweep |
| `reports/fpbase_validation.py`, `reports/spectraforge_validation_report.py` | corrected validation reports (with baseline) |

## Reproduce the findings

```bash
source .venv/bin/activate
cd reports
QT_QPA_PLATFORM=offscreen python classification_experiment.py   # CAE worse than random; peaks best
QT_QPA_PLATFORM=offscreen python cae_vs_spectral_ae.py          # CAE R~0 vs spectral-AE R>0 (+ noise)
```

These are the **fitness functions** for the next phase: a good architecture must beat the CAE (and
the random baseline) on KNN-F1 and produce R > 0 with influence that positively tracks signal.
