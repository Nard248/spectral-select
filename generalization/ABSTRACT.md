# Abstract Draft — Label-Free Dependency-Aware Channel Selection

**Working title:** A Label-Free, Dependency-Aware Method for Selecting Informative Channels
in Coupled Multi-Channel Sensor Data

**Target:** general-method short-paper / abstract submission (fits the CODASSCA 2026
de-branded framing in `CODASSCA2026_Submission/`; can feed that submission directly).

---

## Abstract — verified against the .docx generator (natural phrasing, no em-dashes, distinct from intro)

We present a method that picks a small, useful set of sensor channels without using any labels.
A group-structured autoencoder first learns a shared representation of all the channels. We then
perturb its latent factors and watch how each channel's reconstruction responds. A channel that
responds strongly is one the model relies on, so this gives a label-free measure of relevance. A
relevance and redundancy rule turns these scores into an ordered shortlist of the real channels,
avoiding picks that merely repeat information already kept. The same method applies to very
different sensors, and only the front-end encoder changes between them. On a wearable-sensor
activity benchmark with three inertial units, keeping ten of twenty-seven channels holds accuracy
to within 0.04 macro-F1 of the full set. It also equals a supervised selector that does use
labels, and it is far more stable across people than variance ranking. On biomedical hyperspectral
imaging the same method removes 58 to 95 percent of channels while holding or improving accuracy.
Accuracy rises from 88.2 to 95.2 percent on lichen tissue and from 79.8 to 85.6 percent on
collagen. One simple, label-free idea therefore reduces channels well across markedly different
sensing modalities.

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
