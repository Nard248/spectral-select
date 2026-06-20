# 03 — How to Proceed: anti-cheat architecture ladder

Goal: replace/augment the autoencoder so it **cannot minimize the reconstruction loss by predicting
the per-band mean**, and so its perturbation-influence tracks the informative bands. Deliver a set
of **pluggable, comparable** architectures, gate them on synthetic data, and **accept** only what
matches/beats the published CAE on **real** data.

## Research grounding (why these candidates)

The "cheat" is a loss-landscape failure: with sparse data, MSE rewards mean-prediction without
learning structure. The literature's anti-cheat mechanisms, strongest first:

- **Masked / denoising reconstruction** — corrupt/mask part of the input and reconstruct the
  *original*. Mean-prediction becomes impossible: masked bands must be inferred from the others, so
  the model is forced to learn inter-band structure. For HSI this is **SS-MAE** (spatial+spectral
  band masking) and **SMAE** (spectral masking); "masking is a form of denoising." Best aligned with
  perturbation-selection (the latent is forced to carry discriminative spectral structure).
- **Latent variance regularization / free-bits** — keep latent components above a variance floor so
  they can't go dead.
- **VAE is not automatically a fix** — it has its own **posterior collapse** (decoder ignores the
  latent). It must be paired with **KL-annealing / free-bits / β-VAE**.
- **HSI band-selection precedent** — BS-Nets (band-attention + reconstruct-from-subset),
  stochastic-gate AEs, sparse 1D-operational AEs — all reframe selection as reconstruction from a
  subset, which is inherently masking-like.

Sources: SS-MAE (arXiv 2505.05710), Spectral-MAE for Raman (arXiv 2504.16130), MAE (He et al. CVPR
2022), "Don't Blame the ELBO" posterior-collapse (NeurIPS 2019), KL-annealing dynamics
(arXiv 2310.15440), BS-Nets (arXiv 1904.08269), stochastic-gate AE (Pattern Recognition 2022),
variance regularization vs collapse (arXiv 2112.09214).

## The candidate ladder (C0 → C4)

Each candidate is one architecture/objective change; each reuses `selection_core` for the
perturbation→influence→selection so only the *model + loss* differ.

| # | Candidate | Anti-cheat lever | Notes / expectation |
|---|-----------|------------------|---------------------|
| **C0** | Standard CAE (baseline) | — | reference; degenerate on synthetic |
| **C1** | Deeper / spectral-preserving CAE | add conv depth; **remove the band-axis `adaptive_avg_pool`** so the latent keeps spectral resolution | tests capacity + keeping spectral info; objective unchanged, so may only partially help |
| **C2** | Per-pixel **spectral AE** (1D-conv or MLP) | per-pixel samples → real batches; dense per-pixel spectrum | **already shown to work** (R +0.28, recovers peaks); the simplest fix |
| **C3** | **Masked spectral AE** (denoising) | randomly mask input bands; reconstruct the **full** spectrum | most principled anti-cheat (SS-MAE/SMAE); best-aligned with perturbation; likely the winner |
| **C4** | **Variational spectral AE** | C2/C3 latent + KL **with free-bits / KL-annealing** | structured latent may yield cleaner influence; **must** guard posterior collapse |

Design principles: each candidate is a small, self-contained `nn.Module` with one clear
responsibility, an explicit `encode`/`decode`, and an encapsulated training objective
(`reconstruction_loss`). Keep them **Analyzer-interface-compatible** so the eventual winner can be
registered in `BUILT_IN_AUTOENCODERS` / passed as `config.autoencoder_architecture` with no Analyzer
changes. **Leave the published CAE (`"standard"`) untouched.**

## Fitness function (synthetic — the gate)

Run each candidate through `reports/classification_experiment.py`'s harness and compare to the
baselines already there (random, variance-ranking, peak-neighbourhood, all-bands). A candidate
**passes the synthetic gate** if, averaged over scenes:

1. **reconstruction R > 0** (clearly non-degenerate; CAE ≈ 0), and
2. **corr(influence, band-variance) > 0** (influence tracks signal; CAE ≈ −0.8), and
3. **KNN macro-F1 ≥ variance-ranking** and **clearly > the random baseline and > the CAE**, and
4. **peak_recovery clearly > random (≈0.33)** — sanity, not the target.

Report all candidates side-by-side (this is the "pluggable options + comparison" deliverable).

## Acceptance (real — the gate that matters)

The published CAE achieves classification parity on real data; a replacement must not regress it.
A candidate is **accepted** if, on the real datasets (Lichens, Collagen — see
`01-system-overview.md`), its selected-band **KNN/clustering accuracy matches or beats the CAE's** at
the same band budget. Only then is it eligible to become a default or recommended architecture.
Until accepted on real data, the default stays `"standard"`.

## Scope guardrails (YAGNI)

- Do **not** rewrite the Analyzer or the selection math — reuse `selection_core`.
- Do **not** change the published CAE or its results.
- Build C1–C4 as the plan, but it is fine to **stop early** if C2/C3 clearly pass the synthetic gate
  and win on real data — C4 (VAE) is exploratory.
- This phase is **build + smoke-test only** locally; real training/comparison runs on the training
  machine (see `04-training-runbook.md`).
