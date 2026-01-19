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
- [x] **Phase 2: Config System** - Unified configuration class supporting all variant behaviors (Complete)
- [x] **Phase 3: Core Data Types** - Data classes for spectra, wavelengths, and results (Complete)
- [x] **Phase 4: Analysis Engine** - Refactor core wavelength selection into Analyzer class (Complete)
- [x] **Phase 5: Visualization Module** - Clean Visualizer class from scattered plotting code (Complete)
- [x] **Phase 6: Ground Truth Validation** - Refactor validation metrics into dedicated module (Complete)
- [x] **Phase 7: Notebook Migration** - Update existing notebooks to use new imports (Complete)
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
- [x] 02-01: Config dataclass with all variant options ✓
- [x] 02-02: Config loading from YAML/JSON and validation ✓

### Phase 3: Core Data Types
**Goal**: Clean data classes (`SpectraData`, `WavelengthResult`, etc.) replacing scattered dict/array usage
**Depends on**: Phase 2
**Research**: Unlikely (internal code organization)
**Plans**: TBD

Plans:
- [x] 03-01: Data classes for input spectra and metadata ✓
- [x] 03-02: Result classes for wavelength selection outputs ✓

### Phase 4: Analysis Engine
**Goal**: Extract core wavelength selection algorithm into `Analyzer` class with clean `fit()`, `transform()`, `get_wavelengths()` API
**Depends on**: Phase 3
**Research**: Unlikely (algorithm extraction)
**Plans**: TBD

Plans:
- [x] 04-01: Analyzer class skeleton with public API ✓
- [x] 04-02: Autoencoder integration and latent space handling ✓
- [x] 04-03: Perturbation-based wavelength attribution ✓
- [x] 04-04: Multi-excitation wavelength ranking ✓

### Phase 5: Visualization Module
**Goal**: Clean `Visualizer` class consolidating scattered matplotlib code with consistent styling
**Depends on**: Phase 4
**Research**: Unlikely (matplotlib patterns)
**Plans**: TBD

Plans:
- [x] 05-01: Visualizer class with core plotting methods ✓
- [x] 05-02: Wavelength heatmaps and spectral plots ✓
- [x] 05-03: Clustering and validation visualizations ✓

### Phase 6: Ground Truth Validation
**Goal**: Refactor `ground_truth_validation.py` into clean `Validator` class with metrics API
**Depends on**: Phase 4
**Research**: Unlikely (internal refactoring)
**Plans**: TBD

Plans:
- [x] 06-01: Validator class with metrics computation ✓
- [x] 06-02: Ground truth comparison and reporting ✓

### Phase 7: Notebook Migration
**Goal**: Create new example notebooks demonstrating the `spectral_select` API
**Depends on**: Phase 5, Phase 6
**Research**: Unlikely (documentation)
**Plans**: 1

Plans:
- [x] 07-01: Create example notebooks (quickstart + validation) ✓

### Phase 8: Testing & Validation
**Goal**: Pytest suite verifying bit-identical results between old and new implementations
**Depends on**: Phase 7
**Research**: Unlikely (standard pytest)
**Plans**: TBD

Plans:
- [x] 08-01: Test fixtures with sample data ✓
- [ ] 08-02: Unit tests for Config and data types
- [ ] 08-03: Integration tests comparing old vs new outputs
- [ ] 08-04: CI configuration

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Package Structure | 2/2 | Complete | 2026-01-19 |
| 2. Config System | 2/2 | Complete | 2026-01-19 |
| 3. Core Data Types | 2/2 | Complete | 2026-01-19 |
| 4. Analysis Engine | 4/4 | Complete | 2026-01-19 |
| 5. Visualization Module | 3/3 | Complete | 2026-01-19 |
| 6. Ground Truth Validation | 2/2 | Complete | 2026-01-19 |
| 7. Notebook Migration | 1/1 | Complete | 2026-01-19 |
| 8. Testing & Validation | 1/4 | In progress | - |
