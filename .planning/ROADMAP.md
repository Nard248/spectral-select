# Roadmap: spectral_select Library Refactor

## Overview

Transform the existing research codebase into a clean, importable Python library. Starting with package structure and configuration, we'll systematically extract and organize the wavelength selection pipeline, visualization, and validation code into a proper `spectral_select` package that anyone can install and use.

## Domain Expertise

None

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

### v1.0 Library Refactor (Complete)
- [x] **Phase 1: Package Structure** - Create spectral_select/ package skeleton with proper imports
- [x] **Phase 2: Config System** - Unified configuration class supporting all variant behaviors
- [x] **Phase 3: Core Data Types** - Data classes for spectra, wavelengths, and results
- [x] **Phase 4: Analysis Engine** - Refactor core wavelength selection into Analyzer class
- [x] **Phase 5: Visualization Module** - Clean Visualizer class from scattered plotting code
- [x] **Phase 6: Ground Truth Validation** - Refactor validation metrics into dedicated module
- [x] **Phase 7: Notebook Migration** - Update existing notebooks to use new imports
- [x] **Phase 8: Testing & Validation** - Test suite ensuring bit-identical results

### v1.1 Production Ready (In Progress)
- [x] **Phase 9: Flexible Model Config** - Parameterize autoencoder, support custom models
- [x] **Phase 10: Results Organization** - Structured results directories, checkpoint naming
- [x] **Phase 11: Excel Export & Reporting** - Wavelength pairs export, consistent reporting
- [x] **Phase 12: Data Pipeline Improvements** - Streamline raw→pkl workflow
- [x] **Phase 13: Masking GUI Tool** - Standalone mask creation app
- [x] **Phase 14: Jupyter ROI Widget** - ipywidgets-based ROI selection
- [ ] **Phase 15: End-to-End Testing** - Test notebooks and scripts runability
- [ ] **Phase 16: Coverage & Quality** - Increase coverage to 80%+

## Milestones

- ✅ **v1.0 Library Refactor** - Phases 1-8 (shipped 2026-01-20)
- 🚧 **v1.1 Production Ready** - Phases 9-16 (in progress)

---

## 🚧 v1.1 Production Ready (In Progress)

**Milestone Goal:** Transform the library into a production-ready tool with flexible configuration, organized outputs, data preparation tools, and comprehensive testing.

### Phase 9: Flexible Model Config ✓
**Goal**: Parameterize autoencoder (layers, channels, latent_dim), add architecture selection, support custom models via protocol
**Depends on**: Phase 8
**Research**: Unlikely (internal refactoring)
**Plans**: 2
**Completed**: 2026-01-20

Plans:
- [x] 09-01: Model and training config parameters ✓
- [x] 09-02: Analyzer model integration ✓

### Phase 10: Results Organization ✓
**Goal**: Structured results directories (sample/run/artifacts), model checkpoint naming, wavelength selection tracking
**Depends on**: Phase 9
**Research**: Unlikely (internal patterns)
**Plans**: 3
**Completed**: 2026-01-20

Plans:
- [x] 10-01: ResultsManager class and checkpoint naming ✓
- [x] 10-02: Analyzer-ResultsManager integration ✓
- [x] 10-03: Metadata tracking and tests ✓

### Phase 11: Excel Export & Reporting ✓
**Goal**: Simple flat table Excel export (Rank, Excitation_nm, Emission_nm, Score), consistent reporting format across experiments
**Depends on**: Phase 10
**Research**: Unlikely (openpyxl/xlsxwriter patterns)
**Plans**: 2
**Completed**: 2026-01-20

Plans:
- [x] 11-01: WavelengthResult.to_excel() method ✓
- [x] 11-02: ResultsManager integration and Excel tests ✓

### Phase 12: Data Pipeline Improvements ✓
**Goal**: Streamline raw→pkl workflow, better error handling, consistent file naming conventions
**Depends on**: Phase 9
**Research**: Unlikely (internal refactoring)
**Plans**: 3
**Completed**: 2026-01-21

Plans:
- [x] 12-01: DataLoader wrapper and SpectraData.from_raw() ✓
- [x] 12-02: SpectraData.to_pickle() and error handling improvements ✓
- [x] 12-03: Data pipeline tests ✓

### Phase 13: Masking GUI Tool ✓
**Goal**: Standalone tkinter/Qt app for mask creation with drawing tools (polygon, brush, flood fill)
**Depends on**: Phase 12
**Research**: Likely (GUI frameworks, drawing tools)
**Research topics**: tkinter canvas vs PyQt5/6, polygon drawing patterns, mask serialization
**Plans**: 3
**Completed**: 2026-01-21

Plans:
- [x] 13-01: ME-HSI Viewer core framework ✓
- [x] 13-02: Band browser and false color composer ✓
- [x] 13-03: Spectral profile and statistics panels ✓

### Phase 14: Jupyter ROI Widget ✓
**Goal**: ipywidgets-based ROI selection for notebooks, integration with SpectraData
**Depends on**: Phase 13
**Research**: Likely (ipywidgets patterns)
**Research topics**: ipywidgets canvas/drawing, ipympl, matplotlib.widgets selectors
**Plans**: 2
**Completed**: 2026-01-21

Plans:
- [x] 14-01: ROIWidget core with ipympl and LassoSelector ✓
- [x] 14-02: Multi-class ROI labeling and GroundTruth export ✓

### Phase 15: End-to-End Testing (In Progress)
**Goal**: Test all notebooks with reduced epochs, test data loading scripts, verify full pipeline runability
**Depends on**: Phase 14
**Research**: Unlikely (pytest patterns)
**Plans**: 3

Plans:
- [x] 15-01: Notebook test infrastructure ✓
- [ ] 15-02: TBD
- [ ] 15-03: TBD

### Phase 16: Coverage & Quality
**Goal**: Increase test coverage to 80%+, add integration tests for full workflow, property-based tests
**Depends on**: Phase 15
**Research**: Unlikely (pytest/hypothesis patterns)
**Plans**: TBD

Plans:
- [ ] 16-01: TBD

---

<details>
<summary>✅ v1.0 Library Refactor (Phases 1-8) - SHIPPED 2026-01-20</summary>

### Phase 1: Package Structure
**Goal**: Create the `spectral_select/` package skeleton with `__init__.py`, submodules, and `pyproject.toml` for editable install
**Depends on**: Nothing (first phase)
**Research**: Unlikely (standard Python packaging)
**Plans**: 2

Plans:
- [x] 01-01: Package skeleton and pyproject.toml ✓
- [x] 01-02: Module structure and public API exports ✓

### Phase 2: Config System
**Goal**: Unified `Config` class that captures all 5+ variant behaviors via configuration options instead of separate files
**Depends on**: Phase 1
**Research**: Unlikely (internal refactoring)
**Plans**: 2

Plans:
- [x] 02-01: Config dataclass with all variant options ✓
- [x] 02-02: Config loading from YAML/JSON and validation ✓

### Phase 3: Core Data Types
**Goal**: Clean data classes (`SpectraData`, `WavelengthResult`, etc.) replacing scattered dict/array usage
**Depends on**: Phase 2
**Research**: Unlikely (internal code organization)
**Plans**: 2

Plans:
- [x] 03-01: Data classes for input spectra and metadata ✓
- [x] 03-02: Result classes for wavelength selection outputs ✓

### Phase 4: Analysis Engine
**Goal**: Extract core wavelength selection algorithm into `Analyzer` class with clean `fit()`, `transform()`, `get_wavelengths()` API
**Depends on**: Phase 3
**Research**: Unlikely (algorithm extraction)
**Plans**: 4

Plans:
- [x] 04-01: Analyzer class skeleton with public API ✓
- [x] 04-02: Autoencoder integration and latent space handling ✓
- [x] 04-03: Perturbation-based wavelength attribution ✓
- [x] 04-04: Multi-excitation wavelength ranking ✓

### Phase 5: Visualization Module
**Goal**: Clean `Visualizer` class consolidating scattered matplotlib code with consistent styling
**Depends on**: Phase 4
**Research**: Unlikely (matplotlib patterns)
**Plans**: 3

Plans:
- [x] 05-01: Visualizer class with core plotting methods ✓
- [x] 05-02: Wavelength heatmaps and spectral plots ✓
- [x] 05-03: Clustering and validation visualizations ✓

### Phase 6: Ground Truth Validation
**Goal**: Refactor `ground_truth_validation.py` into clean `Validator` class with metrics API
**Depends on**: Phase 4
**Research**: Unlikely (internal refactoring)
**Plans**: 2

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
**Plans**: 4

Plans:
- [x] 08-01: Test fixtures with sample data ✓
- [x] 08-02: Unit tests for Config and data types ✓
- [x] 08-03: Integration tests comparing old vs new outputs ✓
- [x] 08-04: CI configuration ✓

</details>

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → ... → 8 → 9 → 10 → ... → 16

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Package Structure | v1.0 | 2/2 | Complete | 2026-01-19 |
| 2. Config System | v1.0 | 2/2 | Complete | 2026-01-19 |
| 3. Core Data Types | v1.0 | 2/2 | Complete | 2026-01-19 |
| 4. Analysis Engine | v1.0 | 4/4 | Complete | 2026-01-19 |
| 5. Visualization Module | v1.0 | 3/3 | Complete | 2026-01-19 |
| 6. Ground Truth Validation | v1.0 | 2/2 | Complete | 2026-01-19 |
| 7. Notebook Migration | v1.0 | 1/1 | Complete | 2026-01-19 |
| 8. Testing & Validation | v1.0 | 4/4 | Complete | 2026-01-20 |
| 9. Flexible Model Config | v1.1 | 2/2 | Complete | 2026-01-20 |
| 10. Results Organization | v1.1 | 3/3 | Complete | 2026-01-20 |
| 11. Excel Export & Reporting | v1.1 | 2/2 | Complete | 2026-01-20 |
| 12. Data Pipeline Improvements | v1.1 | 3/3 | Complete | 2026-01-21 |
| 13. Masking GUI Tool | v1.1 | 3/3 | Complete | 2026-01-21 |
| 14. Jupyter ROI Widget | v1.1 | 2/2 | Complete | 2026-01-21 |
| 15. End-to-End Testing | v1.1 | 1/3 | In progress | - |
| 16. Coverage & Quality | v1.1 | 0/? | Not started | - |
