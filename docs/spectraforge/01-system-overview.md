# 01 — System Overview: how each part works

This is the architecture reference for the pieces involved in the band-selection investigation.
Three packages matter, all under `src/`:

- **`spectraforge`** — the synthetic ME-HSI data generator + validation harness (what we built).
- **`spectral_select`** — the band-selection method under test (the published perturbation-AE).
- **`selection_core`** — the shared, model-agnostic perturbation→influence→selection engine.

---

## 1. `spectraforge` — synthetic data + validation

The forward model is a dilute linear superposition of fluorophore excitation–emission matrices,
rendered to a `spectral_select.SpectraData` plus a `GroundTruth` sidecar.

```
F(x,y,λex,λem) = lamp·exposure·power · Σ_k  c_k(x,y) · ε_k · Φ_k · exc_k(λex) · em_k(λem)
```

| File | Responsibility |
|------|----------------|
| `fluorophore.py` | `Fluorophore` — parametric Gaussian excitation/emission. `excitation()` peak-normalized; `emission()` area-normalized on the query grid. |
| `measured.py` | `MeasuredFluorophore` — interpolates **real** measured curves (drop-in for `Fluorophore`). `from_fpbase_payload()` imports FPbase API JSON (handles the real `state` tag, e.g. `default_ex`/`default_em`). |
| `material.py` | `Material` — a fluorophore recipe (the "brush"): `{fluorophore_name: concentration}`. |
| `scene.py` | `Scene` — paint materials onto concentration maps (`paint_rect/circle/polygon/paint_map`), `resolve()` → `{fluorophore: (H,W)}`, `__add__` for linearity tests. |
| `scenegen.py` | `random_field` (smooth random concentration field), `random_scene` (rich-variance mixtures), **`make_labeled_scene`** (balanced, per-pixel class = argmax material → for classification experiments). |
| `acquisition.py` | `AcquisitionConfig` — excitations, emission grid, lamp/exposure/power. |
| `artifacts.py` | `ArtifactConfig` — Rayleigh + 2nd-order scatter lines, seedable Poisson + read noise. |
| `physics.py` | `PhysicsConfig` (all default OFF → exactly-linear invariant preserved): PSF blur, Beer–Lambert inner-filter (first nonlinearity), autofluorescence. |
| `forward.py` | `render(scene, library, acquisition, artifacts, physics, seed)` → `(SpectraData, GroundTruth)`. Also stores **per-fluorophore per-pixel-max spectra** in `GroundTruth` (for peak-based validation). |
| `groundtruth.py` | `GroundTruth` — concentration maps, clean cubes, `informative_bands()` (broad mask), `informative_bands_per_fluorophore()`. |
| `validation.py` | **`validate_selection(ground_truth, selected, tol_nm)`** → metrics dict (see below). |
| `sweep.py` | `run_validation_sweep`, `aggregate_metrics`, `make_random_selector` (chance baseline), `make_analyzer_selector`. |
| `library.py`, `demo.py` | 12-fluorophore starter library; CLI demo. |
| `gui/` | PyQt6 "Forge" workbench — see below. |

### `validate_selection` metrics (the harness output)
- `precision` / `recall` / `f1` — over the **broad** informative-band mask. **These saturate** (the
  mask covers 83–93 % of the grid), so a random selector matches them — read them only next to a
  baseline and next to `mask_coverage`.
- `mask_coverage` — fraction of the emission grid flagged "informative" (exposes the saturation).
- `peak_recovery` / `peak_hits` — **tight**: did a selected band land within `tol_nm` of a
  fluorophore's *true emission peak*? This is the discriminating metric (a random selector ≈ 0.33).
- `fluorophores_recovered` / `per_fluorophore` — broad-mask per-fluorophore recovery.

### The Forge GUI (`spectraforge/gui/`)
`ForgeWindow` (`app.py`) docks: Library/Materials (left), Canvas painter (center), Layers (right),
Acquire/Render/Export + a **"Validate selection vs ground truth"** button (bottom). Pure logic lives
in `render_ops.py` (`render_state`, `validate_state`), `workers.py` (`RenderWorker`,
`ValidateWorker` — run off the UI thread), `state.py`, `layer.py`, `project.py` (`.forge` save/load).
Launch: `spectraforge-gui` (needs a display; tests run with `QT_QPA_PLATFORM=offscreen`).

---

## 2. `spectral_select` — the band-selection method under test

`Analyzer(Config).fit(spectra)` then `get_wavelengths()` → list of `WavelengthBand(excitation_nm,
emission_nm, emission_band_index, influence_score, rank)`.

Pipeline inside `fit()`:
1. **`models/dataset.py`** — builds the dataset and **normalizes the data globally**:
   `(x - global_min) / (global_max - global_min)` over the whole 4D cube (`dataset.py:219`). This is
   the "blind intensity" normalization; combined with sparse data it is part of the root cause.
2. **`models/autoencoder.py`** — `HyperspectralCAEWithMasking`: per-excitation `Conv3d` encoder,
   **`adaptive_avg_pool3d(x, (1, H, W))` collapses the emission-band axis** (`autoencoder.py:159`),
   excitation features are averaged, a `Conv3d` bottleneck + **`sigmoid`** produces the latent; the
   decoder mirrors it with **`sigmoid` outputs**. Trains on spatial **chunks**
   (`training_chunk_size=64`, overlap 8 → a 64×64 image is **one chunk**).
3. **`models/training.py`** — `train_with_masking`: chunked MSE reconstruction training.
4. **Selection** — perturb the latent, measure per-band influence, normalize, pick a diverse subset
   (delegated to `selection_core`; see §3). The diversity step (MMR) lives in the Analyzer.

### Key `Config` knobs (defaults)
`autoencoder_architecture="standard"` (or a custom class — the seam for the next phase),
`model_k1=20`, `model_k3=20`, `model_filter_size=5`, `model_dropout_rate=0.5`,
`normalization_method="variance"` (this is the **influence** normalization, not the data one),
`perturbation_method="percentile"`, `dimension_selection_method="activation"`,
`n_important_dimensions=15`, `n_bands_to_select=30`, `training_epochs=30`,
`training_chunk_size=64`, `n_baseline_patches=50`, `patch_size=32`.

### Custom-architecture seam (used by the next phase)
`Analyzer._create_model` (`analyzer.py:644`) resolves `config.autoencoder_architecture`: the string
`"standard"`, or a **custom class** instantiated as `Cls(excitations_data=..., k1=..., k3=...,
filter_size=...)`. A custom model must behave like `HyperspectralCAEWithMasking`:
`encode(data_dict) -> latent`, `decode(latent) -> {ex: recon}`, be an `nn.Module`, and expose
`emission_bands` and `excitation_wavelengths`. (The loose `AutoencoderProtocol` in `protocols.py`
only sketches `encode`/`decode`; the *real* contract is the working CAE's interface.)

---

## 3. `selection_core` — the shared perturbation engine (`engine.py`)

Model-agnostic, so any architecture can reuse it. The pipeline:
- `select_important_dimensions(latent, method, n)` — rank latent coordinates (`variance`/`activation`/`pca`).
- `latent_statistics(flat)` — per-coordinate std/min/max/percentiles to scale perturbations
  (guards batch=1 → zero std).
- `accumulate_influence(decode_fn, groups, channels_per_group, latent, baseline_recon,
  important_dims, magnitudes, directions, perturbation_method)` — for each important latent dim and
  perturbation, decode and accumulate **per-channel influence = mean |perturbed − baseline|** over
  all axes except the last (channel/band).
- `normalize_influence(influence, data, method)` — `none` / `max_per_group` / `variance` (divide by
  per-band variance).

**Why this matters for the next phase:** the perturbation→influence→selection logic is sound and
reusable. Only the *model + training objective* need to change; a new architecture can feed its own
`encode`/`decode`/`baseline` into `accumulate_influence` unchanged.

---

## 4. Real datasets (for the acceptance gate)

Loaded with `SpectraData.from_pickle(path)`:

| Dataset | Path | Excitations | Spatial × bands |
|---------|------|-------------|-----------------|
| Lichens | `Data/processed/Lichens Dataset 1/spectra_unmasked.pkl` | 8 (310–430 nm) | 1040 × 925 × 22 |
| Collagen | `Data/processed/Collagen Pepsin/spectra_unmasked.pkl` | 6 (310–400 nm) | 256 × 348 × 24 |
| Sponges | `Data/processed/Sponges Acid Group 1/spectra_unmasked.pkl` | — | — |

Note the contrast with synthetic: real cubes are **large** (≈10⁶ pixels → hundreds of training
chunks) and **dense** (every one of ~22–24 bands carries signal). The synthetic cubes are small
(64×64) and **sparse** (a few bright peak bands, the rest ≈0 after normalization). That difference
is the heart of the next section.
