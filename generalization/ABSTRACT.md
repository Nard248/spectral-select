# Abstract Draft — Label-Free Dependency-Aware Channel Selection

**Working title:** A Label-Free, Dependency-Aware Method for Selecting Informative Channels
in Coupled Multi-Channel Sensor Data

**Target:** general-method short-paper / abstract submission (fits the CODASSCA 2026
de-branded framing in `CODASSCA2026_Submission/`; can feed that submission directly).

---

## Abstract (~210 words)

Modern sensing systems produce many channels that are highly correlated and organized into
coupled groups. Choosing a small, interpretable subset of the *original* channels — without
labels — reduces cost and complexity, but existing options force a trade-off: unsupervised
filters (variance, PCA) ignore inter-channel dependency, while dependency-aware selectors
(mRMR, JMI) require labels. We present an unsupervised method that captures channel dependency
directly. A group-structured autoencoder learns a joint representation across channel groups;
each channel's importance is measured by the sensitivity of the reconstruction to perturbations
of the learned latent factors; and a relevance–redundancy criterion selects a diverse,
informative subset of the actual channels. The method is discrete — it returns real channels,
not a projection — and interpretable. Originally developed for multi-excitation hyperspectral
imaging, where it identifies discriminative wavelength bands and outperforms classical
band-selection baselines, the same selection procedure transfers to a very different modality
with only a domain-appropriate encoder. On a standard wearable-sensor human-activity-recognition
benchmark, evaluated with leave-one-subject-out cross-validation, the label-free method matches
supervised mutual-information selection and is substantially more stable across subjects than
variance-based selection. A single dependency-aware, label-free selection principle thus
generalizes across markedly different sensor modalities.

---

## Framing of baseline results (internal — for the paper body / response, not the abstract)

Honest, defensible claims (all under leave-one-subject-out, macro-F1):
- **Matches supervised selection without labels.** Label-free AE-perturb ~ supervised
  mutual-information on the HAR benchmark (e.g. K=7: 0.627 vs 0.606; K=10: 0.679 vs 0.684).
- **Beats variance and is more stable.** AE-perturb > variance at all K (+~0.09 mid-K), with
  far lower fold-to-fold variance (std ~0.07 vs ~0.15) — variance over-selects redundant
  high-variance channels.
- **Cross-modality transfer with an unchanged selection engine** (only the encoder changes:
  3D-conv for imaging cubes, 1D-conv for sensor time series).

Claims to AVOID (not supported / would invite attack):
- Do NOT claim "beats random" on the HAR benchmark. On that benchmark **no method, including
  supervised MI, beats random** — the channel set is intrinsically redundant (confirmed with
  rich features + RandomForest). Frame the HAR result as *generalization + matches-supervised +
  stability*, not as a benchmark win.
- Do NOT claim "first unsupervised conditional discrete selector" — the cell is occupied
  (Concrete Autoencoder, unsupervised-MRMR, DUFS). Claim the *combination* + transfer.

## Why PAMAP2 is not centered in the abstract

The HAR benchmark is intrinsically redundant for channel selection (random ~ supervised),
which would require substantial dataset-specific exposition to explain to a reader without
adding a clean headline. We therefore keep it at a high level (one sentence) as
*generalization evidence*, and let the method + cross-domain transfer carry the abstract.

## Open follow-ups (future work line in the paper)
- A higher-redundancy HAR benchmark (Opportunity) where intelligent selection should separate
  from random — the discriminating test, deferred.
- Concrete Autoencoder head-to-head (the closest unsupervised relative).
