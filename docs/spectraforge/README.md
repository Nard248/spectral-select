# SpectraForge & the Band-Selection Investigation — Documentation

This directory documents (1) the **SpectraForge** synthetic ME-HSI generator and its validation
harness, (2) the **investigation** that found why the perturbation-autoencoder band selector
behaves degenerately on synthetic data, and (3) the **plan + runbook** for the next phase
(trying autoencoder architectures that don't "cheat"), which is intended to be executed and
**trained on a separate machine**.

## Read in this order

| Doc | What it covers | For whom |
|-----|----------------|----------|
| [`01-system-overview.md`](01-system-overview.md) | How every part works: SpectraForge engine, GUI, the validation harness, the band-selection pipeline (`spectral_select` + `selection_core`). | Anyone onboarding. |
| [`02-investigation-and-findings.md`](02-investigation-and-findings.md) | What we did and what we found: the corrected validation harness, and the **root cause** — the spatial CAE collapses to predicting the per-band mean on this data. The full evidence chain (5 refuted hypotheses). | Anyone who needs the *why*. |
| [`03-architecture-plan.md`](03-architecture-plan.md) | How to proceed: a research-grounded **ladder of candidate architectures/objectives (C0–C4)** designed so the AE cannot cheat, and the fitness/acceptance protocol. | Whoever designs/reviews the next phase. |
| [`04-training-runbook.md`](04-training-runbook.md) | **The executable handoff.** Exact environment, commands, the model interface contract, per-candidate build specs, smoke-test criteria, and the synthetic-gate / real-accept training+comparison protocol. Written so another machine's **Claude Code agent** can implement and run it. | The agent/operator on the training machine. |

## One-paragraph summary of where we are

SpectraForge generates chemically-grounded synthetic 4D ME-HSI cubes (excitation × emission ×
height × width) with perfect ground truth, to validate unsupervised band-selection methods. Using
it we discovered that the published perturbation-autoencoder selector, **on these synthetic cubes**,
selects non-informative (off-peak) bands and classifies worse than random. After ruling out the
validation metric, normalization, and several selection knobs, the **root cause is the spatial
convolutional autoencoder**: on this (sparse, after global normalization) data it converges to a
degenerate "predict the per-band mean" solution — reconstruction correlation R ≈ 0 — so its
perturbation-influence is noise. A simple per-pixel **spectral** autoencoder learns the data
(R ≈ +0.28) and the *same* selection principle then recovers the informative bands. The next phase
builds and compares a ladder of non-cheating architectures (deeper/spectral-preserving CAE →
spectral AE → masked/denoising spectral AE → variational spectral AE) on a separate machine.

## Status & branch

All code and docs are on **`main`**. Nothing is trained yet for the next phase; the runbook
produces code + smoke tests, and the actual training/comparison runs on the training machine.
Full local test suite: ~460 passing (`pytest -m "not slow and not notebook"`).
