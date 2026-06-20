# 04 — Training Runbook (handoff for the training machine's Claude Code agent)

**You are a coding agent on a separate, more powerful machine.** Your job is to implement the
**anti-cheat autoencoder ladder** from [`03-architecture-plan.md`](03-architecture-plan.md), run the
training/comparison (which this repo's owner deliberately did **not** run on the laptop), and report
results back. Read [`02-investigation-and-findings.md`](02-investigation-and-findings.md) first — it
tells you *why* this work exists (the CAE collapses to predicting the per-band mean on synthetic
ME-HSI; reconstruction R ≈ 0; influence is noise). Do **not** modify the published CAE or the
selection math; reuse `selection_core`.

Follow Test-Driven Development. Work on a branch. Commit frequently.

---

## 0. Environment setup & sanity check

```bash
# Python 3.11. From the repo root:
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'                      # editable install + dev deps (note the quotes for zsh)

# Verify the suite is green (~460 tests):
QT_QPA_PLATFORM=offscreen pytest -q -m "not slow and not notebook"

# Reproduce the baseline findings (these ARE your fitness functions):
cd reports
QT_QPA_PLATFORM=offscreen python classification_experiment.py   # expect: CAE F1 ~0.33 (worse than random ~0.37); peak-neighbourhood best ~0.55
QT_QPA_PLATFORM=offscreen python cae_vs_spectral_ae.py          # expect: CAE reconR ~0.00, spectral-AE reconR >0, beats CAE on F1
cd ..
```

If those numbers don't roughly reproduce, stop and investigate the environment before building
anything. A GPU is fine and welcome (set `device="cuda"` in the `Config`/candidate training); the
CAE and candidates are small.

---

## 1. The model interface contract

A candidate architecture must be a `torch.nn.Module` that behaves like
`spectral_select.models.autoencoder.HyperspectralCAEWithMasking`:

```python
class Candidate(nn.Module):
    def __init__(self, excitations_data: dict[float, np.ndarray], k1=20, k3=20, filter_size=5, **kw):
        # excitations_data: {excitation_nm: (H, W, n_bands) float array}  (already normalized by the dataset)
        ...
        self.excitation_wavelengths = sorted(excitations_data)         # required attr
        self.emission_bands = {ex: d.shape[2] for ex, d in excitations_data.items()}  # required attr

    def encode(self, data_dict: dict[float, Tensor]) -> Tensor:        # data_dict: {ex: (B, H, W, bands)}
        ...                                                            # returns the latent
    def decode(self, latent: Tensor) -> dict[float, Tensor]:           # returns {ex: (B, H, W, bands)}
        ...
    def reconstruction_loss(self, data_dict) -> Tensor:                # OPTIONAL hook (see below)
        ...
```

- **Constructor**: `_create_model` (analyzer.py:644) calls custom classes as
  `Cls(excitations_data=..., k1=..., k3=..., filter_size=...)`. Accept those and ignore what you
  don't use.
- **`reconstruction_loss`**: the standard trainer uses plain MSE. Candidates with a non-MSE objective
  (C3 masked, C4 VAE/ELBO) need their objective honored. **Cleanest path for this phase:** do *not*
  rely on `models/training.py`. Instead build candidates as **self-contained band selectors** (next
  section) that own their training loop. Only when a winner is chosen do you wire it into the
  production Analyzer (`BUILT_IN_AUTOENCODERS` + a `reconstruction_loss` hook in `train_with_masking`)
  — that is a *follow-up*, not part of this phase.

---

## 2. Where the code goes & the selector interface

Create a new package `src/spectral_select/architectures/` (one file per candidate + a base):

```
src/spectral_select/architectures/
  __init__.py
  base.py            # BandSelectorModel ABC: fit(spectra) -> self ; select(n_bands) -> [(ex_nm, em_nm)]
                     #   + diagnostics: reconstruction_r() ; influence_signal_corr()
  cae_baseline.py    # C0: thin wrapper over the existing Analyzer/standard CAE (reference)
  deep_cae.py        # C1
  spectral_ae.py     # C2  (port reports/cae_vs_spectral_ae.py::_SpectralAE — already proven)
  masked_spectral_ae.py  # C3
  variational_spectral_ae.py  # C4
```

`BandSelectorModel.select` must **reuse the perturbation principle**. Two acceptable routes:
- **Reuse `selection_core`** directly: provide `decode_fn`, baseline latent, baseline recon,
  `channels_per_group`, and `important_dims` to `selection_core.accumulate_influence`, then
  `normalize_influence`, then pick a diverse top-`n` (you may copy the Analyzer's MMR or use a simple
  ≥10 nm dedup within an excitation as in `reports/cae_vs_spectral_ae.py`).
- Or replicate the small perturbation loop already in `reports/cae_vs_spectral_ae.py::spectral_ae_select`
  (perturb each latent dim ±std, accumulate mean |Δrecon| per band). Either is fine; prefer reusing
  `selection_core` so the math stays identical across candidates.

Write **unit tests** in `tests/spectral_select/architectures/` for each candidate: shapes round-trip
(`encode`→`decode`), `select` returns `n_bands` valid `(ex, em)` pairs, and a **smoke `fit` at 2–3
epochs** runs without error. Keep these fast (tiny scene, `QT_QPA_PLATFORM=offscreen` not needed for
non-GUI tests).

---

## 3. Candidate build specs

Reuse the labelled-dataset + metric helpers in `reports/classification_experiment.py`
(`build_dataset`, `feature_matrix`, `cols_for_bands`, `knn_macro_f1`, `peak_neighbourhood_bands`)
and `spectraforge.sweep.make_random_selector`, `spectraforge.validation.validate_selection`.

- **C1 — Deeper / spectral-preserving CAE.** Start from `HyperspectralCAEWithMasking` but (a) add
  conv depth (2–3 more encoder/decoder conv blocks with nonlinearities) and (b) **remove the
  `adaptive_avg_pool3d(x, (1, H, W))` band collapse** (`autoencoder.py:159`) so the latent keeps
  emission-band resolution. Same MSE objective. Purpose: isolate "capacity + keep spectral info."
- **C2 — Spectral AE.** Per-pixel: input = each pixel's concatenated multi-excitation spectrum
  (D = Σ bands). MLP or 1D-conv encoder→latent(8–16)→decoder→D. ReLU, linear output, standardized
  features, Adam, ~300 epochs. **This is already implemented** as `_SpectralAE` + `spectral_ae_select`
  in `reports/cae_vs_spectral_ae.py` — port it into `spectral_ae.py` and make it a `BandSelectorModel`.
- **C3 — Masked spectral AE (the most promising).** C2's network, but each training step **randomly
  masks a fraction (e.g. 40–70 %) of input bands** (zero them or drop them) and the loss is computed
  on reconstructing the **full** spectrum (especially the masked bands). At selection time, encode the
  *unmasked* spectrum for the baseline, then perturb. Sweep mask ratio. This is the SS-MAE/SMAE idea;
  mean-prediction cannot satisfy it.
- **C4 — Variational spectral AE.** C2/C3 encoder outputs (μ, logσ²); reparameterize; loss =
  reconstruction + β·KL. **Must** include an anti-posterior-collapse guard: **free-bits**
  (clamp per-dim KL at a floor λ, e.g. 0.5 nats) and/or **KL annealing** (β: 0→1 over the first ~⅓ of
  epochs). Report the active-units count (dims with KL above the floor). For perturbation, perturb μ.

---

## 4. Comparison harness (the "pluggable options + comparison" deliverable)

Create `reports/architecture_comparison.py` that, over ≥3 scene seeds and ≥2 noise levels, prints one
table comparing **C0–C4 + the baselines** (random, variance-ranking, peak-neighbourhood, all-bands)
on: **reconstruction R**, **corr(influence, signal-variance)**, **peak_recovery**, **KNN macro-F1**.
Model it on `reports/cae_vs_spectral_ae.py`. Mute training logs
(`contextlib.redirect_stdout(os.devnull)`). Keep it deterministic (fixed seeds).

### Synthetic gate (a candidate must pass ALL):
1. reconstruction R > 0 (CAE ≈ 0)
2. corr(influence, band-variance) > 0 (CAE ≈ −0.8)
3. KNN macro-F1 ≥ variance-ranking, and clearly > random and > CAE
4. peak_recovery clearly > 0.33 (sanity)

---

## 5. Real-data acceptance (the gate that matters)

The published CAE achieves classification parity on real data; a replacement must not regress it.

```python
from spectral_select.types import SpectraData
spectra = SpectraData.from_pickle("Data/processed/Lichens Dataset 1/spectra_unmasked.pkl")  # 8 ex, 1040x925x22
# also: "Data/processed/Collagen Pepsin/spectra_unmasked.pkl" (6 ex, 256x348x24)
```

Steps:
1. **Locate the existing real-data evaluation** the publications used (search `experiments/`,
   `publications/`, and any `*_knn*.py` / classification scripts) — it defines the labels/masks and
   the KNN protocol on selected-vs-all bands. Reuse it; do **not** invent a new label source.
2. Train C0 (CAE) and each passing candidate on Lichens (and Collagen), select the same band budget,
   and compute the same classification metric.
3. **Accept** a candidate iff its real-data accuracy **matches or beats the CAE** at equal budget.
   Also confirm the candidate's reconstruction R > 0 on real data (it should — real data is dense).
4. If you cannot run the exact published evaluation, fall back to the KNN-on-selected protocol with
   whatever labels/masks ship with the processed datasets, and **clearly state which protocol you
   used**.

Until a candidate is accepted on real data, **leave `autoencoder_architecture="standard"` as the
default.** Productionizing the winner into `BUILT_IN_AUTOENCODERS` + a `reconstruction_loss` hook in
`train_with_masking` is a *follow-up* spec, not this phase.

---

## 6. Report back

Write `docs/spectraforge/05-training-results.md` containing:
- the full comparison table(s) (synthetic, all candidates + baselines, all metrics),
- which candidates passed the synthetic gate and the real-data acceptance, with the exact numbers,
- the real-data evaluation protocol you used (and the label source),
- mask-ratio / β / free-bits settings that worked (for C3/C4),
- a recommendation (which architecture to adopt, or what to try next),
- anything surprising or any place the findings in doc 02 did **not** hold.

Commit everything (code + tests + the new report + `05-training-results.md`) on a branch and open a
PR (or push and report the branch). Keep the published CAE and its results untouched.

---

## Quick reference

| Need | Where |
|------|-------|
| Why this exists / root cause | `docs/spectraforge/02-investigation-and-findings.md` |
| Candidate specs & rationale | `docs/spectraforge/03-architecture-plan.md` |
| Labelled dataset + metrics helpers | `reports/classification_experiment.py` |
| Proven spectral-AE to port (C2) | `reports/cae_vs_spectral_ae.py` |
| Shared perturbation engine | `src/selection_core/engine.py` |
| Custom-architecture seam | `src/spectral_select/analyzer.py:644`, `config.autoencoder_architecture` |
| Validation metrics | `src/spectraforge/validation.py::validate_selection` (`peak_recovery`, `mask_coverage`) |
| Chance baseline | `src/spectraforge/sweep.py::make_random_selector` |
| Real data | `Data/processed/{Lichens Dataset 1,Collagen Pepsin,Sponges Acid Group 1}/spectra_unmasked.pkl` |
