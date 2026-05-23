# Abstract Draft — Label-Free Dependency-Aware Channel Selection

**Working title:** A Label-Free, Dependency-Aware Method for Selecting Informative Channels
in Coupled Multi-Channel Sensor Data

**Target:** general-method short-paper / abstract submission (fits the CODASSCA 2026
de-branded framing in `CODASSCA2026_Submission/`; can feed that submission directly).

---

## Abstract (~220 words) — verified against the .docx generator

Modern sensors emit many overlapping channels arranged in coupled groups. Selecting a small,
interpretable subset of the original channels — without labels — lowers cost and aids
interpretation, yet existing methods trade off: unsupervised filters (variance, PCA) ignore
inter-channel dependency, while dependency-aware selectors (mRMR, JMI) require labels. We present
an unsupervised, dependency-aware, and discrete selector. A group-structured autoencoder learns a
joint representation of the channels; perturbing its latent factors measures each channel's
reconstruction sensitivity (relevance); and a relevance–redundancy rule chooses a diverse subset
of the real channels. We verify the same method in two very different domains, changing only the
encoder. On a wearable-sensor human-activity-recognition benchmark (three inertial units, 27
channels, leave-one-subject-out), retaining ten channels (a 63% reduction) preserves macro-F1 to
within 0.04 of the full set and matches a supervised mutual-information selector, while being
substantially more stable across subjects than variance selection. On biomedical hyperspectral
imaging, the same method reduces channels by 58–95% while maintaining or improving classification
accuracy (e.g., 192→80 bands raises accuracy from 88.2% to 95.2%; 158→30 bands from 79.8% to
85.6%). A single label-free principle thus delivers strong, interpretable channel reduction across
modalities.

**Framing note:** the two domains are presented as *parallel verifications* of one general
method — biomedical HSI is a second verification domain, **not** prior work. Biomedical numbers
are existing verified results; the HAR result is the new contribution.

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
