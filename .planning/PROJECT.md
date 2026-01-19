# spectral_select Library Refactor

## What This Is

A refactored, library-style Python package for 4D hyperspectral wavelength selection analysis. Transforms the existing research codebase into a clean, importable library (`spectral_select`) with a class-based API that enables the academic community to reproduce and extend this wavelength selection framework.

## Core Value

Clean, stable, reproducible wavelength selection analysis that anyone can `from spectral_select import Analyzer` and use immediately.

## Requirements

### Validated

- ✓ Wavelength analysis pipeline — existing (wavelengthselectionV2-2.py)
- ✓ Autoencoder-based dimensionality reduction — existing (scripts/models/autoencoder.py)
- ✓ Perturbation-based wavelength attribution — existing
- ✓ Ground truth validation metrics — existing (ground_truth_validation.py)
- ✓ Multiple sample support (Lime, Kiwi, Lichens) — existing

### Active

- [ ] Clean, readable module structure with proper separation of concerns
- [ ] Unified class-based API (`Analyzer`, `Config`, `Visualizer`)
- [ ] Config-driven behavior preserving all 5+ variant functionality
- [ ] Proper Python package structure (`spectral_select/`)
- [ ] Test suite verifying identical results before/after refactor
- [ ] Notebook migration — existing notebooks work with new imports
- [ ] Remove hardcoded paths — use config/environment variables
- [ ] Proper error handling — replace bare `except:` blocks
- [ ] Python logging instead of print statements

### Out of Scope

- New analysis features — focus on cleanup only, algorithms unchanged
- Web UI/dashboard — no Streamlit/Dash interface
- PyPI publishing — local `pip install -e .` only for now
- Support for Python < 3.11 — modern Python only

## Context

**Research Project:** This implements the ME-HSI (Multi-Excitation Hyperspectral Imaging) wavelength selection framework documented in the accompanying paper. The core algorithm uses autoencoder-based latent space perturbation to identify informative spectral bands.

**Current State:** Working but messy research code with:
- 5+ wavelengthSelection*.py variants with overlapping functionality
- Hardcoded Windows paths (breaks on macOS/Linux)
- 1,550+ print statements instead of logging
- 30+ sys.path manipulations instead of proper imports
- Bare exception handlers hiding errors

**Codebase Analysis:** See `.planning/codebase/` for detailed architecture, structure, and technical debt documentation.

**Branch Strategy:** All refactoring work happens in a separate branch, not main, until validated.

## Constraints

- **Preserve All Variants**: The 5+ wavelengthSelection*.py behaviors must all be available via configuration options in the unified implementation
- **Identical Results**: Given the same inputs, the refactored library must produce bit-identical outputs to the original code
- **Notebook Compatibility**: Existing Jupyter notebooks in `notebooks/` must work after migration to new imports

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Class-based API | Familiar pattern for scientific Python users (like scikit-learn) | — Pending |
| Unified via config | Simpler than inheritance hierarchy, easier to maintain | — Pending |
| Package name: spectral_select | Descriptive, available, easy to type | — Pending |
| Separate branch first | Safe refactoring without breaking working code | — Pending |

---
*Last updated: 2026-01-19 after initialization*
