# Phase 7 — Engine Unification Implementation Plan

> **For agentic workers:** behavior-preserving refactor. TDD/characterization discipline:
> pin current numeric output, extract, prove identical. Steps use `- [ ]`.

**Goal:** Extract the shared, domain-agnostic perturbation→influence→normalize algorithm into
`src/selection_core/`, and have both `spectral_select.Analyzer` (HSI) and
`channel_select.engine` (general) delegate to it — with byte-identical output to today.

**Source spec:** `docs/superpowers/specs/2026-06-15-repo-cleanup-reorg-design.md` (§Phase 7).

## Verified equivalence (line-by-line read of both engines)

`spectral_select/analyzer.py` math methods are mathematically identical to
`channel_select/engine.py` functions:

| Analyzer (private) | Engine fn | Notes |
|---|---|---|
| `_select_important_dimensions` | `select_important_dimensions` | identical; coords = `latent.shape[1:]` (4-tuple for HSI) |
| `_calculate_latent_statistics` | `latent_statistics` | analyzer keeps unused `"mean"` key; safe to drop |
| `_calculate_perturbation_amount` | `perturbation_amount` | manual 4-D flat index == `np.ravel_multi_index` (C-order) |
| `_measure_band_influence` | `measure_channel_influence` | HSI `dim=(0,1,2)` == "all axes but last" |
| `_compute_influence_scores` (loop) | `run_selection` (loop) | same nesting + weights |
| `_normalize_influences` | `normalize_influence` | **two divergences below** |

**Divergences to preserve for byte-identity:**
1. Variance-normalization dtype: engine casts data to `float64`; Analyzer stays `float32`.
   → `selection_core.normalize_influence(..., variance_float64: bool = True)`; Analyzer passes `False`.
2. Method name: Analyzer config uses `"max_per_excitation"`; engine uses `"max_per_group"`.
   → Analyzer's delegating method translates the name before calling core.

**Domain-specific (NOT extracted):** diversity selection — HSI uses *nm* distance + `WavelengthBand`
objects (`_select_top_bands`/`_select_bands_mmr`/`_select_bands_min_distance`); the general engine
uses *channel-index* distance + `(group,channel)` tuples (`select_channels`). Each stays in its package.

## `selection_core` public API

```
selection_core/
  __init__.py        # re-exports the functions below
  engine.py
    select_important_dimensions(latent, method, n) -> [(score, coord), ...]
    latent_statistics(latent_flat) -> dict
    perturbation_amount(coord, latent_dims, magnitude, sign, stats, baseline_latent, method) -> float
    measure_influence(decode_fn, groups, perturbed_latent, baseline_recon, weight=1.0) -> {g: np.ndarray}
    accumulate_influence(decode_fn, groups, channels_per_group, latent, baseline_recon,
                         important_dims, *, magnitudes, directions, perturbation_method) -> {g: np.ndarray}
    normalize_influence(influence, data, method, *, variance_float64=True) -> {g: np.ndarray}
```

## Tasks (TDD)

- [ ] **T1 (RED→GREEN):** `tests/selection_core/test_imports.py` imports the API → fails (module
  missing) → create `src/selection_core/{__init__,engine}.py` (move primitives from
  `channel_select.engine`; add `measure_influence` decode_fn/groups form + `accumulate_influence`
  + `variance_float64` param) → green.
- [ ] **T2:** Repoint `channel_select/engine.py` — import primitives from `selection_core`;
  keep `measure_channel_influence(model,...)` as a thin wrapper over `measure_influence`;
  refactor `run_selection` to call `accumulate_influence`. Run `pytest tests/channel_select` → all green (unchanged behavior).
- [ ] **T3 (characterization):** `tests/test_analyzer_core_equivalence.py` — build a deterministic
  injected baseline state (5-D latent + fake HSI model with `decode`/`excitation_wavelengths`/
  `emission_bands` + dataset stub), run the **current** Analyzer math methods, and assert they equal
  `selection_core` outputs (dims, influence, variance-normalized influence with `variance_float64=False`).
  Run on current code → passes (proves faithful extraction). This guards the next step.
- [ ] **T4:** Refactor `Analyzer` math methods to delegate to `selection_core`
  (`_select_important_dimensions`, `_calculate_latent_statistics`, `_calculate_perturbation_amount`,
  `_measure_band_influence`, `_compute_influence_scores`, `_normalize_influences`). Keep
  diversity selection (`_select_top_bands`/mmr/min_distance) as-is. Run full suite + T3 → all green.
- [ ] **T5:** Register `selection_core` in `pyproject.toml` `include`; add `--cov=selection_core` to CI.
  Run full suite → 367 + new tests green. Commit.

## Acceptance
- `pytest -m "not slow and not notebook"` ≥ 367 passed (baseline) + new tests, all green.
- `channel_select` 13 tests unchanged-green (guards `selection_core`).
- Equivalence test proves Analyzer math == `selection_core`.
- One algorithm, two thin domain adapters.
