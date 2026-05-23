# Generalization Master Plan — Dependency-Aware Channel Selection (general method)

Standalone-paper effort generalizing the `spectral_select` perturbation-autoencoder
method beyond ME-HSI to grouped multi-channel sensor data, validated on HAR.

- **Spec:** `docs/superpowers/specs/2026-05-23-general-channel-selection-design.md`
- **Plan (P1+P2):** `docs/superpowers/plans/2026-05-23-general-channel-selection.md`
- **Code package:** `channel_select/` (sibling of `spectral_select/`, which stays frozen)
- **Output target:** standalone paper.

## Contribution (honest framing — see RESEARCH_LOG 2026-05-23 lit scan)

The "unsupervised + conditional + discrete" cell is **not empty** (Concrete Autoencoder,
unsupervised-MRMR, DUFS live there). The defensible contribution is the **combination**:
group-structured AE + latent-perturbation reconstruction-sensitivity relevance + MMR
conditional-redundancy control, plus **cross-domain transfer** (hyperspectral -> HAR) with
an identical selection engine. **Do not claim "first unsupervised conditional discrete."**
Must-beat baseline: **Concrete Autoencoder (Balın et al., ICML 2019)**. Headline metric:
**LOSO macro-F1** (not k-fold).

## Architecture invariant vs per-domain

- Invariant: parallel per-group encoders + mean fusion + per-group decoders (the coupling prior).
- Per-domain: conv rank matches the data axis (HSI 2D->Conv3d, HAR 1D->Conv1d, tabular->MLP).
- Shared/identical: the selection engine (`channel_select/engine.py`), via the
  `GroupStructuredModel` protocol.

## Datasets (HAR-only v1)

- **PAMAP2** — MONSTER preprocessed (ungated HF: monster-monash/PAMAP2). 38,856 windows,
  52 channels -> 3 IMU groups x 9 channels, 8-9 subjects, 12 classes. LOSO-ready. Primary slice.
- **Opportunity** — hard/dramatic, 100+ channels (P4). Likely its own encoder.
- **ME-HSI anchor** — Lichens/Pepsin via `adapters/spectral.py` (P3 regression test).

## Phases

- **P0** Lit landscape — DONE (see RESEARCH_LOG 2026-05-23).
- **P1** Package + engine + temporal AE — DONE (20 tests).
- **P2** PAMAP2 vertical slice — IN PROGRESS. Gate: AE-perturb beats random-K under LOSO.
- **P3** Full baselines (incl. Concrete Autoencoder, mRMR, Laplacian, MCFS, unsup-MRMR, ISSC) +
  full LOSO harness + HSI engine-regression test.
- **P4** Opportunity adapter + run.
- **P5** Ablations: grouped-vs-ungrouped, normalization (variance vs max_per_group),
  fusion (mean/concat/attention), latent dim, window length.
- **P6** Figures (F1 architecture, F2 acc-vs-K, F3 channel-importance map, F4 stability,
  F5 interpretability) + write-up.

## Constraints

- `spectral_select/` core is FROZEN (TPAMI numbers). Engine is shared via a regression test,
  not by editing the HSI code, in v1.
- Local Mac MPS compute only. Run model code with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Strict LOSO from day one; no random splits (subject leakage).
