# Design Spec — Dependency-Aware Unsupervised Channel Selection (general method)

**Date:** 2026-05-23
**Author:** Narek Meloyan
**Status:** Approved (design); implementation starting
**Related:** `spectral_select/` (origin method, TPAMI), `revision/` (TPAMI revision), `CODASSCA2026_Submission/` (de-branded framing)

## 1. Motivation & contribution

`spectral_select` selects informative wavelength bands from multi-excitation hyperspectral
cubes by training a group-structured convolutional autoencoder, perturbing latent dimensions,
and ranking channels by reconstruction sensitivity. This spec generalizes that method beyond
bio-imaging / ME-HSI into a **domain-agnostic, dependency-aware, unsupervised, discrete,
interpretable channel-selection** method, and validates it on **multivariate human-activity-
recognition (HAR) sensor time series**.

The empty cell in the channel-selection design space:

|              | Marginal (ignores coupling) | Conditional (dependency-aware) |
|--------------|-----------------------------|--------------------------------|
| Supervised   | mutual-info, chi^2          | mRMR, JMI, CMIM, ReliefF       |
| Unsupervised | variance, PCA, Laplacian    | **<- this method**             |

Headline claim to aim for: **match or beat *supervised* conditional selectors (mRMR/JMI)
without using labels**, and demonstrate it with a method that is *identical* across two very
different sensor modalities (fluorescence cubes and IMU streams).

Output target: **standalone paper**.

## 2. Scope

**In scope (v1):**
- New domain-agnostic package `channel_select/`.
- Shared perturbation+selection engine, validated to reproduce `spectral_select.Analyzer`.
- Temporal (`Conv1d`-over-time) group-structured autoencoder.
- HAR datasets: **PAMAP2** (clean primary), **Opportunity** (hard/dramatic).
- One **ME-HSI anchor** (Lichens or Pepsin) via an adapter, for continuity + engine regression.
- Benchmark protocol with leave-one-subject-out (LOSO) CV, multi-classifier downstream eval.
- SOTA + supervised-selector baselines, ablations, paper figures.

**Out of scope (v1, YAGNI):**
- Tennessee Eastman / industrial domain (deferred to v2 if reviewers want more breadth).
- Modifying `spectral_select` core (TPAMI numbers stay frozen).
- Tabular/omics datasets (no regular axis -> conv prior does not apply).
- Attention fusion as the default (it is an ablation, not the baseline architecture).
- Any GUI.

## 3. Architecture

### 3.1 Layout
```
channel_select/                 # new domain-agnostic package (sibling of spectral_select)
  protocols.py                  # GroupedChannelData, GroupStructuredModel, SelectionConfig
  engine.py                     # SHARED perturbation + influence + normalization + MMR engine
  data.py                       # GroupedChannelDataset in-memory container
  models/temporal.py            # Conv1d-over-time grouped autoencoder
  models/factory.py             # build encoder/decoder by axis_type
  adapters/pamap2.py            # PAMAP2 -> GroupedChannelDataset
  adapters/opportunity.py       # Opportunity -> GroupedChannelDataset
  adapters/spectral.py          # SpectraData -> GroupedChannelData (anchor + regression test)
generalization/                 # paper workspace (NOT code), mirrors revision/
  MASTER_PLAN.md  RESEARCH_LOG.md  CHANGELOG.md
  baselines/  figures/  reports/
tests/channel_select/           # unit + regression tests
```

### 3.2 Shared engine (`engine.py`) — the "method"
Single source of truth for the selection algorithm. Generalizes the four `Analyzer` private
steps so they operate over an abstract `(group, channel, *latent_axes)` layout rather than the
HSI-specific `excitation`/`emission`/`(c,l,h,w)` vocabulary:

- `select_important_dimensions(latent, method, n)` — variance | activation | pca (unchanged math).
- `compute_influence(model, important_dims, magnitudes, directions, method)` — perturb each
  latent coord, decode, reduce `mean(|perturbed - baseline|)` **over all axes except the channel
  axis** to get per-channel influence per group.
- `normalize_influence(influence, data, method)` — `variance` | `max_per_group` | `none`.
- `select_channels(influence_catalog, K, diversity)` — MMR | min_distance over the
  `(group, channel)` catalog.

The engine consumes two protocols and never imports HSI types.

### 3.3 Protocols (`protocols.py`)
```python
class GroupStructuredModel(Protocol):
    groups: list[Hashable]                  # excitations | sensor locations
    channels_per_group: dict[Hashable, int] # emission bands | IMU axes
    def encode(self, batch) -> Tensor                  # -> latent (batch, lat_ch, *latent_axes)
    def decode(self, latent) -> dict[Hashable, Tensor] # per-group recon (batch, *axes, channel)

class GroupedChannelData(Protocol):
    groups: list[Hashable]
    channels_per_group: dict[Hashable, int]
    axis_type: str                          # "spatial2d" | "temporal1d"
    def get_all_data(self) -> dict[Hashable, Tensor]
```

### 3.4 Reproducibility guarantee
`spectral_select` core is **not modified** in v1. Instead, a regression test runs the shared
engine through `adapters/spectral.py` on a cached Lichens cube and asserts the engine reproduces
`spectral_select.Analyzer`'s selections (same `(excitation, band)` set, same order). This makes
"the same method runs on both domains" a *verified* claim, not an assertion. Migrating the HSI
core to import the shared engine is a later, opt-in step gated on this regression passing.

### 3.5 Data contract (`data.py`)
`GroupedChannelDataset` holds:
- `data: dict[group -> Tensor[window, *axis, channel]]`
- `axis_type: str`
- `labels: Optional[Tensor[window]]`
- `subject_ids: Optional[Tensor[window]]` (for LOSO)
Two `from_*` adapters fill it. Conforms to `GroupedChannelData`.

### 3.6 Temporal model (`models/temporal.py`)
Per-group `Conv1d` encoder stacks (group = sensor location), mean-fusion across groups, latent,
per-group `ConvTranspose1d` decoders. Satisfies `GroupStructuredModel`. Windowing ~1-2 s, 50%
overlap (standard HAR practice).

## 4. Datasets & adapters

- **PAMAP2** — first vertical slice. Groups = {hand, chest, ankle} IMUs; each group has acc(x,y,z)
  x2 scales, gyro(x,y,z), mag(x,y,z), temp; 18 activities; 9 subjects. Download from UCI.
- **Opportunity** — 100+ channels, on-body + ambient; locomotion/gesture labels; LOSO. UCI.
- **ME-HSI anchor** — cached Lichens (or Pepsin) cube via `adapters/spectral.py`.

If a download source is unavailable, the adapter must fail loudly with a clear message and a
documented manual-download path; no silent synthetic fallback.

## 5. Benchmark protocol

1. Train AE unsupervised on training split (no labels).
2. Engine selects top-K channels.
3. Train downstream classifiers (KNN, SVM, RF, 1D-CNN) on the K selected channels.
4. Report **accuracy/F1 vs K** envelope.
5. **Leave-one-subject-out CV** (non-negotiable for HAR; random splits leak subjects).
6. Secondary metrics: selection stability (Jaccard across folds), redundancy of selected set.

## 6. Baselines

- Unsupervised: variance, PCA-loading, Laplacian score, MCFS, ISSC (port from `revision/baselines/`).
- Supervised (the bar to match label-free): mRMR, JMI, CMIM, ReliefF, Lasso, RFE.
- Random x multiple seeds.

## 7. Ablations

1. **Grouped vs ungrouped encoder** — the single most important ablation; proves the coupling
   prior earns its place. If grouped ~= ungrouped, the novelty thins (verify in PAMAP2 slice).
2. Normalization re-audit: `variance` vs `max_per_group` (the HSI inversion finding will recur).
3. Latent dim, perturbation magnitude/method, fusion (mean/concat/attention).
4. Window length.

## 8. Figures

- F1 generalized architecture.
- F2 accuracy-vs-K per dataset (money figure).
- F3 channel-importance x activity heatmap + selected channels on a body diagram
  (wavelength-heatmap analog).
- F4 selection stability / redundancy comparison.
- F5 interpretability case study (which channels picked for which activities; do they make
  physical sense?).

## 9. Risks

- **Grouping must demonstrably matter** — checked early via the grouped-vs-ungrouped ablation.
- **"Beats supervised unsupervised-ly" is bold** — decide the headline only after PAMAP2
  numbers land; fallback framing ("competitive, label-free, interpretable, cross-domain") is
  still a paper.
- **HAR leakage** — strict LOSO from day one.
- **Normalization sensitivity** — new domain needs its own normalization audit before trusting
  numbers.

## 10. Execution phases

- **P0** — Lit landscape: map "sensor/channel selection for HAR" so positioning is correct.
- **P1** — Package skeleton + `protocols.py` + `data.py` + `models/temporal.py` + `engine.py`.
- **P2** — PAMAP2 adapter + vertical slice (load -> AE -> one selection -> one accuracy-vs-K
  curve + grouped-vs-ungrouped sanity check). De-risks the whole transfer.
- **P3** — Baselines + LOSO harness; engine regression test vs `spectral_select`.
- **P4** — Scale to Opportunity.
- **P5** — Ablations.
- **P6** — Figures + write-up.
