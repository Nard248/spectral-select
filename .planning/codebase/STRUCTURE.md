# Codebase Structure

**Analysis Date:** 2026-01-19

## Directory Layout

```
4D-Hyperspectral-Unsupervised-Clustering/
├── wavelength_analysis/      # MAIN ANALYSIS MODULE
│   ├── core/                 # Core analysis abstractions
│   ├── Results/              # Output TIFF layers
│   ├── validation_results_v2/# Validation outputs
│   ├── archive/              # Legacy scripts/notebooks
│   ├── run_analysis.py       # PRIMARY ENTRY POINT
│   └── *.py                  # Various analysis scripts
├── scripts/                  # MODELING & PROCESSING
│   ├── models/               # PyTorch model layer
│   ├── data_processing/      # Data loading layer
│   └── utils/                # Supporting utilities
├── notebooks/                # INTERACTIVE ANALYSIS
├── Data/                     # INPUT DATA
│   ├── Raw/                  # Raw .im3 files per sample
│   └── processed/            # Processed normalized data
├── results/                  # OUTPUT RESULTS
│   └── models/               # Trained model checkpoints
├── MCR-Analysis/             # Alternative analysis method
├── paper/                    # Publication artifacts
├── .planning/                # GSD planning documents
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## Directory Purposes

**wavelength_analysis/**
- Purpose: Main wavelength selection and analysis code
- Contains: Core modules, CLI entry points, test files, visualization scripts
- Key files: `run_analysis.py`, `ground_truth_validation.py`, `supervised_metrics.py`
- Subdirectories: `core/` (abstractions), `archive/` (legacy), `Results/` (outputs)

**wavelength_analysis/core/**
- Purpose: Core analysis abstractions and orchestration
- Contains: `analyzer.py`, `config.py`, `visualization.py`, `selector.py`, `experiments.py`
- Key files: `analyzer.py` (33KB - main engine), `config.py` (presets)
- Subdirectories: None

**scripts/models/**
- Purpose: PyTorch model infrastructure
- Contains: `autoencoder.py`, `dataset.py`, `training.py`, `clustering.py`, `visualization.py`
- Key files: `autoencoder.py` (HyperspectralCAEWithMasking), `dataset.py` (MaskedHyperspectralDataset)
- Subdirectories: None

**scripts/data_processing/**
- Purpose: Data loading and normalization
- Contains: `hyperspectral_loader.py`, `hyperspectral_processor.py`, `hyperspectral_utils.py`
- Key files: `hyperspectral_loader.py` (HyperspectralDataLoader class)
- Subdirectories: None

**scripts/utils/**
- Purpose: Supporting utilities and tools
- Contains: `masking_tool.py`, `masking_utils.py`, `process_trq_files.py`
- Key files: `masking_tool.py` (ROI masking interface)
- Subdirectories: None

**notebooks/**
- Purpose: Jupyter notebooks for interactive analysis
- Contains: `complete_*_analysis_notebook.ipynb` (4 sample notebooks)
- Key files: One notebook per sample (Lime, Kiwi, Lichens, Lichens_2)
- Subdirectories: None

**Data/**
- Purpose: Input data storage
- Contains: Raw hyperspectral scans, processed datasets
- Key locations: `Raw/Lime/`, `Raw/Kiwi/`, `Raw/Lichens_2/`, `processed/`
- Subdirectories: `Raw/` (per-sample), `processed/`

**results/**
- Purpose: Analysis outputs
- Contains: Wavelength selection results, visualizations, model checkpoints
- Key locations: `models/` (trained weights), `*_wavelength_selection/`
- Subdirectories: Per-sample result directories

## Key File Locations

**Entry Points:**
- `wavelength_analysis/run_analysis.py` - Primary CLI entry point
- `scripts/run_hyperspectral_processing.py` - Data processing entry
- `wavelength_analysis/run_visualizations.py` - Visualization runner

**Configuration:**
- `wavelength_analysis/core/config.py` - AnalysisConfig dataclass
- `requirements.txt` - Python dependencies
- `.gitignore` - Version control exclusions

**Core Logic:**
- `wavelength_analysis/core/analyzer.py` - WavelengthAnalyzer (main engine)
- `scripts/models/autoencoder.py` - HyperspectralCAEWithMasking
- `scripts/models/dataset.py` - MaskedHyperspectralDataset
- `wavelength_analysis/ground_truth_validation.py` - Validation metrics

**Testing:**
- `wavelength_analysis/test_v2_pipeline.py` - Pipeline tests
- `wavelength_analysis/test_object_wise_analysis.py` - Object analysis tests
- `wavelength_analysis/test_roi_overlay_v2.py` - ROI overlay tests

**Documentation:**
- `README.md` - Project documentation
- `.planning/` - GSD planning files

## Naming Conventions

**Files:**
- snake_case: `hyperspectral_loader.py`, `ground_truth_validation.py`
- PascalCase (legacy): `WavelengthSelectionFinal.py`, `DIAGNOSTIC_ROI_ISSUE.py`
- Test files: `test_*.py` (pytest convention)
- Entry points: `run_*.py`

**Directories:**
- snake_case: `data_processing`, `wavelength_analysis`
- Domain-specific: `core` (abstractions), `archive` (legacy), `Results` (outputs)
- Plural for collections: `scripts`, `notebooks`, `models`

**Special Patterns:**
- `__init__.py` - Package exports
- `*_old.py` - Deprecated versions (technical debt)
- `fix_*.py` - One-time fix scripts

## Where to Add New Code

**New Feature:**
- Primary code: `wavelength_analysis/core/` (if analysis-related)
- Primary code: `scripts/models/` (if ML-related)
- Tests: `wavelength_analysis/test_{feature}.py`
- Config: Extend `AnalysisConfig` in `wavelength_analysis/core/config.py`

**New Analysis Module:**
- Implementation: `wavelength_analysis/{module_name}.py`
- Core abstraction: `wavelength_analysis/core/{module_name}.py`
- Tests: `wavelength_analysis/test_{module_name}.py`

**New Model:**
- Implementation: `scripts/models/{model_name}.py`
- Training: Extend `scripts/models/training.py`
- Dataset: Extend or add to `scripts/models/dataset.py`

**Utilities:**
- Shared helpers: `scripts/utils/{utility_name}.py`
- Data processing: `scripts/data_processing/{processor_name}.py`

## Special Directories

**.planning/**
- Purpose: GSD planning documents
- Source: Created by /gsd:* commands
- Committed: Yes (planning artifacts)

**wavelength_analysis/archive/**
- Purpose: Deprecated scripts and old notebooks
- Source: Moved during refactoring
- Committed: Yes (historical reference)
- Note: Consider cleaning up

**results/models/**
- Purpose: Trained model checkpoints (.pt files)
- Source: Generated by training scripts
- Committed: Yes (via git)
- Note: Large files, consider Git LFS

---

*Structure analysis: 2026-01-19*
*Update when directory structure changes*
