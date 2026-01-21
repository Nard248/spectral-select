# External Integrations

**Analysis Date:** 2026-01-19

## APIs & External Services

**Payment Processing:**
- Not applicable

**Email/SMS:**
- Not applicable

**External APIs:**
- Not detected - Pure local compute with file-based data I/O

## Data Storage

**Databases:**
- Not applicable - All file-based storage

**File Storage:**
- Local filesystem only
- Data formats: `.pkl` (pickle), `.npy` (numpy), `.h5` (HDF5), `.im3` (proprietary hyperspectral)
- Processed data: `Data/processed/{sample_name}/`

**Caching:**
- Not applicable

## Authentication & Identity

**Auth Provider:**
- Not applicable

**OAuth Integrations:**
- Not applicable

## Monitoring & Observability

**Error Tracking:**
- Not detected (no Sentry, etc.)

**Analytics:**
- Not applicable

**Logs:**
- stdout only via print statements and logging module
- No centralized logging service

## CI/CD & Deployment

**Hosting:**
- Local execution only
- No cloud deployment configured

**CI Pipeline:**
- Not detected (no GitHub Actions, etc.)

## Environment Configuration

**Development:**
- Required: Python 3.11+, virtual environment
- Optional: Java JDK, Apache Maven (for ImageJ)
- No `.env` files - hardcoded paths (technical debt)

**Staging:**
- Not applicable

**Production:**
- Local compute only

## Webhooks & Callbacks

**Incoming:**
- Not applicable

**Outgoing:**
- Not applicable

## External Tool Integrations

### ImageJ/Fiji Integration (Java Bridge)

**Purpose:** Load proprietary .im3 hyperspectral files

**Integration:**
- Python Bridge: JPype 1.5.2 and scyjava 1.10.2 - `requirements.txt`
- Module: `scripts/data_processing/hyperspectral_loader.py`

**External Dependencies:**
- Apache Maven 3.9.9
- Java JDK (setup documented in `README.md`)

### PyTorch Ecosystem

**Autoencoder Implementation:**
- `scripts/models/autoencoder.py` - `HyperspectralCAEWithMasking` class
- `scripts/models/training.py` - Training loop with masking
- `scripts/models/dataset.py` - `MaskedHyperspectralDataset` PyTorch Dataset

**GPU Support:**
- Conditional CUDA via `torch.cuda.is_available()` - `wavelength_analysis/core/analyzer.py`

### scikit-learn Algorithms

**Clustering:**
- KMeans - `wavelength_analysis/wavelengthselectionV2-2.py`

**Dimensionality Reduction:**
- PCA, FastICA, NMF - `scripts/data_processing/hyperspectral_loader.py`

**Metrics:**
- silhouette_score, davies_bouldin_score, calinski_harabasz_score
- Location: `wavelength_analysis/supervised_metrics.py`

### Ground Truth Validation

**PNG Annotation Processing:**
- Module: `wavelength_analysis/ground_truth_validation.py`
- Libraries: Pillow (Image I/O), NumPy, SciPy (linear_sum_assignment)

**Metrics Calculated:**
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Fowlkes-Mallows Score
- V-Measure, Homogeneity, Completeness

### Data Serialization

**Pickle Format:**
- Primary persistence for hyperspectral data
- Utilities: `scripts/data_processing/hyperspectral_utils.py`
  - `load_data_from_pickle()`
  - `save_data_to_pickle()`

**Excel Integration:**
- openpyxl 3.1.5 - For metadata and laser power measurements
- Used in: `scripts/data_processing/hyperspectral_processor.py`

---

*Integration audit: 2026-01-19*
*Update when adding/removing external services*
