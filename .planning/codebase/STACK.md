# Technology Stack

**Analysis Date:** 2026-01-19

## Languages

**Primary:**
- Python 3.11 - All application code (verified via `.venv/bin/python` symlink)

**Secondary:**
- None detected

## Runtime

**Environment:**
- Python 3.11.0 managed via virtual environment at `.venv/`
- No browser runtime (scientific computing/CLI tools only)

**Package Manager:**
- pip (via `requirements.txt`)
- Lockfile: Not present (only requirements.txt with pinned versions)

## Frameworks

**Core:**
- PyTorch 2.x - Deep learning (autoencoders) - `scripts/models/autoencoder.py`
- TensorFlow 2.19.0 - Installed but less utilized than PyTorch
- scikit-learn 1.6.1 - Machine learning algorithms - `wavelength_analysis/core/analyzer.py`

**Testing:**
- pytest (convention-based) - `wavelength_analysis/test_*.py`
- No formal test framework configuration

**Build/Dev:**
- No explicit build tools
- JPype 1.5.2 / scyjava 1.10.2 - Java bridge for ImageJ/Fiji integration

## Key Dependencies

**Critical:**
- numpy 2.1.3 - Array operations and numerical computations
- scipy 1.15.2 - Scientific algorithms (`scipy.optimize`, `scipy.ndimage`, `scipy.stats`)
- pandas 2.2.3 - Data manipulation and metrics analysis
- torch - Autoencoder architecture (`HyperspectralCAEWithMasking`)
- scikit-learn - Clustering (KMeans), decomposition (PCA, ICA, NMF), metrics

**Image Processing:**
- Pillow (PIL) 11.1.0 - Image file I/O - `wavelength_analysis/core/analyzer.py`
- tifffile 2025.3.30 - TIFF file handling for 16-bit layers
- opencv-python 4.11.0.86 - Image preprocessing - `scripts/prepare_lichens_2_for_wavelength_analysis.py`
- h5py 3.13.0 - HDF5 data format support

**Visualization:**
- matplotlib 3.10.1 - Primary plotting - `wavelength_analysis/core/visualization.py`
- seaborn 0.13.2 - Statistical visualization - `wavelength_analysis/ground_truth_validation.py`
- plotly 6.0.1 - Interactive visualizations (installed)

**ML Utilities:**
- kneed 0.14.0 - Elbow method for optimal clustering
- tqdm 4.67.1 - Progress bars

**Jupyter:**
- jupyterlab 4.3.6, notebook 7.3.3 - Interactive analysis
- IPython 9.1.0 - Interactive shell

## Configuration

**Environment:**
- No .env files detected
- Hardcoded paths used in scripts (technical debt)
- Configuration via `AnalysisConfig` dataclass - `wavelength_analysis/core/config.py`

**Build:**
- No tsconfig.json equivalent
- Virtual environment in `.venv/`

## Platform Requirements

**Development:**
- macOS/Linux/Windows with Python 3.11+
- Optional: Apache Maven 3.9.9, Java JDK (for ImageJ/Fiji integration)

**Production:**
- Local compute only (no cloud deployment)
- GPU optional (CUDA support for PyTorch)

---

*Stack analysis: 2026-01-19*
*Update after major dependency changes*
