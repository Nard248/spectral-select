# Roadmap: spectral_select Library Refactor

## Overview

Transform the existing research codebase into a clean, importable Python library. Starting with package structure and configuration, we'll systematically extract and organize the wavelength selection pipeline, visualization, and validation code into a proper `spectral_select` package that anyone can install and use.

## Domain Expertise

None

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Package Structure** - Create spectral_select/ package skeleton with proper imports (Complete)
- [ ] **Phase 2: Config System** - Unified configuration class supporting all variant behaviors
- [ ] **Phase 3: Core Data Types** - Data classes for spectra, wavelengths, and results
- [ ] **Phase 4: Analysis Engine** - Refactor core wavelength selection into Analyzer class
- [ ] **Phase 5: Visualization Module** - Clean Visualizer class from scattered plotting code
- [ ] **Phase 6: Ground Truth Validation** - Refactor validation metrics into dedicated module
- [ ] **Phase 7: Notebook Migration** - Update existing notebooks to use new imports
- [ ] **Phase 8: Testing & Validation** - Test suite ensuring bit-identical results

## Phase Details

### Phase 1: Package Structure
**Goal**: Create the `spectral_select/` package skeleton with `__init__.py`, submodules, and `pyproject.toml` for editable install
**Depends on**: Nothing (first phase)
**Research**: Unlikely (standard Python packaging)
**Plans**: TBD

Plans:
- [x] 01-01: Package skeleton and pyproject.toml ✓
- [x] 01-02: Module structure and public API exports ✓

### Phase 2: Config System
**Goal**: Unified `Config` class that captures all 5+ variant behaviors via configuration options instead of separate files
**Depends on**: Phase 1
**Research**: Unlikely (internal refactoring)
**Plans**: TBD

Plans:
- [ ] 02-01: Config dataclass with all variant options
- [ ] 02-02: Config loading from YAML/JSON and validation

### Phase 3: Core Data Types
**Goal**: Clean data classes (`SpectraData`, `WavelengthResult`, etc.) replacing scattered dict/array usage
**Depends on**: Phase 2
**Research**: Unlikely (internal code organization)
**Plans**: TBD

Plans:
- [ ] 03-01: Data classes for input spectra and metadata
- [ ] 03-02: Result classes for wavelength selection outputs

### Phase 4: Analysis Engine
**Goal**: Extract core wavelength selection algorithm into `Analyzer` class with clean `fit()`, `transform()`, `get_wavelengths()` API
**Depends on**: Phase 3
**Research**: Unlikely (algorithm extraction)
**Plans**: TBD

Plans:
- [ ] 04-01: Analyzer class skeleton with public API
- [ ] 04-02: Autoencoder integration and latent space handling
- [ ] 04-03: Perturbation-based wavelength attribution
- [ ] 04-04: Multi-excitation wavelength ranking

### Phase 5: Visualization Module
**Goal**: Clean `Visualizer` class consolidating scattered matplotlib code with consistent styling
**Depends on**: Phase 4
**Research**: Unlikely (matplotlib patterns)
**Plans**: TBD

Plans:
- [ ] 05-01: Visualizer class with core plotting methods
- [ ] 05-02: Wavelength heatmaps and spectral plots
- [ ] 05-03: Clustering and validation visualizations

### Phase 6: Ground Truth Validation
**Goal**: Refactor `ground_truth_validation.py` into clean `Validator` class with metrics API
**Depends on**: Phase 4
**Research**: Unlikely (internal refactoring)
**Plans**: TBD

Plans:
- [ ] 06-01: Validator class with metrics computation
- [ ] 06-02: Ground truth comparison and reporting

### Phase 7: Notebook Migration
**Goal**: Update all Jupyter notebooks in `notebooks/` to use new `spectral_select` imports
**Depends on**: Phase 5, Phase 6
**Research**: Unlikely (import changes only)
**Plans**: TBD

Plans:
- [ ] 07-01: Migrate analysis notebooks
- [ ] 07-02: Migrate visualization notebooks
- [ ] 07-03: Verify notebook outputs unchanged

### Phase 8: Testing & Validation
**Goal**: Pytest suite verifying bit-identical results between old and new implementations
**Depends on**: Phase 7
**Research**: Unlikely (standard pytest)
**Plans**: TBD

Plans:
- [ ] 08-01: Test fixtures with sample data
- [ ] 08-02: Unit tests for Config and data types
- [ ] 08-03: Integration tests comparing old vs new outputs
- [ ] 08-04: CI configuration

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Package Structure | 2/2 | Complete | 2026-01-19 |
| 2. Config System | 0/2 | Not started | - |
| 3. Core Data Types | 0/2 | Not started | - |
| 4. Analysis Engine | 0/4 | Not started | - |
| 5. Visualization Module | 0/3 | Not started | - |
| 6. Ground Truth Validation | 0/2 | Not started | - |
| 7. Notebook Migration | 0/3 | Not started | - |
| 8. Testing & Validation | 0/4 | Not started | - |
