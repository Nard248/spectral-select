# Architecture

**Analysis Date:** 2026-01-19

## Pattern Overview

**Overall:** Layered Monolith with Pipeline Pattern

**Key Characteristics:**
- Scientific computing pipeline for 4D hyperspectral analysis
- Single-machine execution (no distributed components)
- Config-driven analysis with dataclass configuration
- Autoencoder-based dimensionality reduction followed by perturbation analysis

## Layers

**Data Ingestion Layer:**
- Purpose: Load and normalize raw hyperspectral data
- Contains: Data loaders, preprocessors, normalization utilities
- Location: `scripts/data_processing/`
- Depends on: External libraries (numpy, PIL, h5py)
- Used by: Model and Analysis layers

**Model Layer:**
- Purpose: PyTorch-based deep learning infrastructure
- Contains: Autoencoders, datasets, training loops, clustering
- Location: `scripts/models/`
- Depends on: Data Ingestion layer, PyTorch
- Used by: Analysis layer

**Analysis Orchestration Layer:**
- Purpose: High-level wavelength analysis coordination
- Contains: WavelengthAnalyzer, AnalysisConfig, WavelengthVisualizer
- Location: `wavelength_analysis/core/`
- Depends on: Model layer, Data Ingestion layer
- Used by: CLI entry points

**CLI/Execution Layer:**
- Purpose: User-facing command-line interfaces
- Contains: Entry point scripts, batch processing utilities
- Location: `wavelength_analysis/run_*.py`, `scripts/run_*.py`
- Depends on: Analysis Orchestration layer
- Used by: End users

## Data Flow

**Wavelength Analysis Pipeline:**

1. User runs: `python run_analysis.py --sample Lime --config default`
2. Configuration loaded via `AnalysisConfig` dataclass
3. `WavelengthAnalyzer.__init__()` initializes PyTorch device
4. `load_data_and_model()`:
   - Load hyperspectral data (pickle) → numpy dict
   - Load mask (PNG) → numpy array
   - Create `MaskedHyperspectralDataset` → normalized tensors
   - Load pretrained `HyperspectralCAEWithMasking`
5. `setup_baseline()`: Extract patches, encode to latent space
6. `select_important_dimensions()`: Rank latent dimensions
7. `analyze_wavelength_sensitivity()`: Perturbation-based attribution
8. `select_bands_with_diversity()`: Final wavelength selection
9. `extract_layer_images()`: Save TIFF per selected band
10. Output saved: JSON results, text summaries, visualizations

**State Management:**
- File-based state (`.planning/`, `results/`)
- No persistent in-memory state between runs
- Each analysis execution is independent

## Key Abstractions

**WavelengthAnalyzer:**
- Purpose: Main analysis orchestrator
- Location: `wavelength_analysis/core/analyzer.py`
- Pattern: Orchestrator coordinating data loading, model setup, analysis, visualization
- Methods: `run_complete_analysis()`, `load_data_and_model()`, `analyze_wavelength_sensitivity()`

**AnalysisConfig:**
- Purpose: Configuration dataclass with sample-specific presets
- Location: `wavelength_analysis/core/config.py`
- Pattern: Dataclass with `save()` and `load()` for JSON persistence
- Presets: `LIME_CONFIG`, `KIWI_CONFIG`, `LICHENS_CONFIG`

**MaskedHyperspectralDataset:**
- Purpose: PyTorch Dataset wrapper with preprocessing
- Location: `scripts/models/dataset.py`
- Pattern: PyTorch Dataset implementing `__getitem__`, `__len__`
- Handles: Normalization, mask application, ROI extraction

**HyperspectralCAEWithMasking:**
- Purpose: Convolutional Autoencoder with mask-aware encoding
- Location: `scripts/models/autoencoder.py`
- Pattern: PyTorch nn.Module with encode/decode methods
- Architecture: 2-layer CNN encoder/decoder with sparsity regularization

**ExperimentFramework:**
- Purpose: Meta-orchestrator for parameter sweeps
- Location: `wavelength_analysis/core/experiments.py`
- Pattern: Framework running multiple configurations
- Methods: `run_parameter_sweep()`, `compare_configurations()`

## Entry Points

**Primary CLI:**
- Location: `wavelength_analysis/run_analysis.py`
- Triggers: User runs `python run_analysis.py --sample Lime`
- Options: `--sample`, `--all-samples`, `--comparison`, `--config`
- Responsibilities: Parse args, initialize analyzer, run analysis, save outputs

**Data Processing:**
- Location: `scripts/run_hyperspectral_processing.py`
- Triggers: User runs data processing pipeline
- Responsibilities: Load .im3 files, normalize, create processed datasets

**Notebooks:**
- Location: `notebooks/complete_*_analysis_notebook.ipynb`
- Purpose: Interactive exploration and manual analysis
- Samples: Lime, Kiwi, Lichens, Lichens_2

## Error Handling

**Strategy:** Exception catching at orchestrator level with logging

**Patterns:**
- Try/except around model loading and analysis steps
- Warnings module for non-fatal issues
- Print statements for progress (technical debt - should use logging)

**Issues (Technical Debt):**
- Many bare `except:` clauses that silently swallow errors
- Inconsistent error handling across modules

## Cross-Cutting Concerns

**Logging:**
- Mixed: Python logging module in some files, print statements in others
- No centralized logging configuration
- Technical debt: 1,550+ print statements

**Validation:**
- Minimal input validation
- Type hints present but not enforced
- No schema validation for configuration

**Configuration:**
- Dataclass-based via `AnalysisConfig`
- JSON serialization for persistence
- Hardcoded paths in some scripts (technical debt)

---

*Architecture analysis: 2026-01-19*
*Update when major patterns change*
