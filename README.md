# spectral-select

A Python library for reproducible wavelength selection in 4D hyperspectral imaging using autoencoder-based latent space perturbation analysis.

## Overview

`spectral-select` identifies the most informative wavelength combinations in multi-excitation hyperspectral datasets by:

1. Training or loading a convolutional autoencoder on the hyperspectral data
2. Analyzing the latent space to identify important dimensions
3. Perturbing latent dimensions and measuring reconstruction sensitivity
4. Ranking and selecting wavelength bands based on influence scores

This approach enables dimensionality reduction while preserving the most discriminative spectral information for downstream clustering and classification tasks.

## Installation

### From source (development)

```bash
git clone https://github.com/Nard248/4D-Hyperspectral-Unsupervised-Clustering.git
cd spectral-select
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 2.0
- scikit-learn >= 1.0
- matplotlib >= 3.7
- tifffile >= 2023.0

See `pyproject.toml` for the complete list of dependencies.

## Quick Start

```python
from spectral_select import Analyzer, Config, SpectraData

# 1. Create configuration
config = Config(
    sample_name="my_sample",
    n_bands_to_select=30,
    model_path="models/autoencoder.pth",
    output_dir="results/",
)

# 2. Load your hyperspectral data
data = SpectraData.from_pickle("processed_data.pkl")

# 3. Run wavelength selection analysis
analyzer = Analyzer(config)
analyzer.fit(data)

# 4. Get selected wavelengths
wavelengths = analyzer.get_wavelengths()
for band in wavelengths[:5]:
    print(f"Rank {band.rank}: Ex={band.excitation_nm}nm, Em={band.emission_nm}nm")

# 5. Save results
analyzer.save_results()
```

## Data Format

### Input Data Structure

The library expects multi-excitation hyperspectral data organized as follows:

```python
from spectral_select import SpectraData, ExcitationData
import numpy as np

# Create ExcitationData for each excitation wavelength
ex_365 = ExcitationData(
    excitation_nm=365.0,
    cube=np.zeros((512, 512, 50)),  # (height, width, n_emission_bands)
    emission_wavelengths=list(range(400, 700, 6)),  # 50 emission wavelengths
)

ex_405 = ExcitationData(
    excitation_nm=405.0,
    cube=np.zeros((512, 512, 45)),
    emission_wavelengths=list(range(420, 690, 6)),
)

# Combine into SpectraData
data = SpectraData(
    excitations={365.0: ex_365, 405.0: ex_405},
    mask=np.ones((512, 512)),  # Optional binary mask (1=valid, 0=masked)
    sample_name="my_sample",
)
```

### Loading from Pickle

For existing processed data in pickle format:

```python
# Expected pickle structure:
# {
#     'data': {
#         365.0: {'cube': np.array(...), 'wavelengths': [...]},
#         405.0: {'cube': np.array(...), 'wavelengths': [...]},
#     },
#     'excitation_wavelengths': [365.0, 405.0],
#     'mask': np.array(...),  # Optional
# }

data = SpectraData.from_pickle("processed_data.pkl")
```

## Configuration

### Basic Configuration

```python
from spectral_select import Config

config = Config(
    # Required
    sample_name="Lichens_2",

    # Paths (all optional)
    data_path="Data/processed/sample.pkl",
    model_path="models/autoencoder.pth",
    output_dir="results/analysis/",

    # Analysis parameters
    n_bands_to_select=30,           # How many wavelengths to select
    n_important_dimensions=15,       # Latent dimensions to analyze
    dimension_selection_method="activation",  # "variance", "activation", or "pca"
    perturbation_method="percentile",        # "percentile", "standard_deviation", "absolute_range"
    normalization_method="variance",          # "variance", "max_per_excitation", "none"

    # Device
    device="cuda",  # "cuda", "cpu", or "mps"
)
```

### Configuration from Files

```python
# Load from YAML
config = Config.from_yaml("config.yaml")

# Load from JSON
config = Config.from_json("config.json")

# Save configuration
config.to_yaml("config_backup.yaml")
config.to_json("config_backup.json")
```

### Full Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_name` | str | "sample" | Identifier for the sample |
| `data_path` | Path | None | Path to input data file |
| `mask_path` | Path | None | Path to mask file |
| `model_path` | Path | None | Path to autoencoder model |
| `output_dir` | Path | None | Directory for results |
| `dimension_selection_method` | str | "activation" | Method for selecting important latent dimensions ("variance", "activation", "pca") |
| `n_important_dimensions` | int | 15 | Number of latent dimensions to analyze |
| `perturbation_method` | str | "percentile" | How to perturb latent space ("percentile", "standard_deviation", "absolute_range") |
| `perturbation_magnitudes` | List[float] | [10, 20, 30] | Perturbation strengths |
| `perturbation_directions` | List[str] | ["bidirectional"] | Direction(s) of perturbation |
| `normalization_method` | str | "variance" | Score normalization ("variance", "max_per_excitation", "none") |
| `n_bands_to_select` | int | 30 | Target number of wavelengths |
| `n_layers_to_extract` | int | 10 | TIFF layers to extract |
| `use_diversity_constraint` | bool | False | Enable spectral diversity |
| `diversity_method` | str | "mmr" | Diversity method ("mmr", "min_distance", "none") |
| `lambda_diversity` | float | 0.5 | MMR diversity trade-off (0=relevance, 1=diversity) |
| `min_distance_nm` | float | 15.0 | Minimum spectral distance in nm |
| `save_tiff_layers` | bool | True | Save extracted TIFF layers |
| `save_visualizations` | bool | True | Generate plots |
| `save_detailed_results` | bool | True | Save detailed analysis files |
| `device` | str | "cuda" | Computation device |
| `n_baseline_patches` | int | 50 | Patches for baseline |
| `patch_size` | int | 32 | Patch size in pixels |
| `patch_stride` | int | 16 | Stride between patches |
| `random_seed` | int | 42 | Reproducibility seed |

## API Reference

### Analyzer

The main class for running wavelength selection analysis.

```python
from spectral_select import Analyzer, Config, SpectraData

# Initialize
config = Config(sample_name="sample", n_bands_to_select=30)
analyzer = Analyzer(config)

# Fit to data
analyzer.fit(data)

# Check if fitted
if analyzer.is_fitted:
    print("Analysis complete!")

# Get results
bands = analyzer.get_wavelengths()      # List[WavelengthBand]
result = analyzer.result                 # WavelengthResult

# Transform data to selected bands only
reduced_data = analyzer.transform(data)

# Or fit and transform in one step
reduced_data = analyzer.fit_transform(data)

# Save results
analyzer.save_results()  # Uses config.output_dir
analyzer.save_results(output_dir="custom/path/")
```

### WavelengthBand

Represents a single selected wavelength combination.

```python
from spectral_select import WavelengthBand

band = WavelengthBand(
    rank=1,
    excitation_nm=365.0,
    emission_nm=500.0,
    emission_band_index=8,
    influence_score=0.85,
)

print(band)
# WavelengthBand(rank=1, ex=365.0nm, em=500.0nm, score=0.8500)

# Serialize
band_dict = band.to_dict()
band_restored = WavelengthBand.from_dict(band_dict)
```

### WavelengthResult

Complete output container for analysis results.

```python
from spectral_select import WavelengthResult

# Access from analyzer
result = analyzer.result

# Properties
result.n_bands               # Number of selected bands
result.top_band              # Highest-ranked band
result.excitation_wavelengths  # Unique excitations in selection
result.selected_bands        # All bands (sorted by rank)
result.metrics               # AnalysisMetrics object
result.timestamp             # When analysis was run
result.config_snapshot       # Config at time of analysis

# Get bands for specific excitation
bands_365 = result.get_bands_for_excitation(365.0)

# Save/load
result.to_json("results.json")
loaded = WavelengthResult.from_json("results.json")
```

### Validator

Validates clustering results against ground truth.

```python
from spectral_select import Validator, GroundTruth, load_ground_truth_from_png

# Load ground truth from annotated PNG
ground_truth = load_ground_truth_from_png(
    "annotations/ground_truth.png",
    class_colors={
        "class_a": (255, 0, 0, 255),    # Red
        "class_b": (0, 255, 0, 255),    # Green
        "background": (0, 0, 0, 255),   # Black (will be -1)
    }
)

# Or create from array
ground_truth = GroundTruth.from_array(
    labels_2d,  # 2D int array, -1 for background
    class_names=["class_a", "class_b", "class_c"],
)

# Validate clustering
validator = Validator(ground_truth)
metrics = validator.evaluate(cluster_labels_2d)

# Print summary
print(metrics.summary())

# Access individual metrics
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score:.3f}")
print(f"Normalized Mutual Info: {metrics.normalized_mutual_info:.3f}")
print(f"Purity: {metrics.purity:.3f}")
```

### Visualizer

Generate visualizations of analysis results.

```python
from spectral_select import Visualizer

visualizer = Visualizer(config)

# Plot influence scores
visualizer.plot_influence_heatmap(analyzer.influence_matrix, save_path="influence.png")

# Plot selected bands distribution
visualizer.plot_band_distribution(analyzer.get_wavelengths(), save_path="distribution.png")

# Plot validation metrics
visualizer.plot_confusion_matrix(metrics, save_path="confusion.png")
```

## Data Types Reference

### ExcitationData

Single excitation wavelength data cube.

```python
from spectral_select import ExcitationData

ed = ExcitationData(
    excitation_nm=365.0,
    cube=np.zeros((100, 100, 50)),  # [H, W, bands]
    emission_wavelengths=[400 + i*6 for i in range(50)],
    exposure_time=0.5,   # Optional
    laser_power=100.0,   # Optional
)

# Properties
ed.height       # 100
ed.width        # 100
ed.n_bands      # 50
ed.shape        # (100, 100, 50)
```

### SpectraData

Multi-excitation container.

```python
from spectral_select import SpectraData

data = SpectraData(
    excitations={365.0: ex_365, 405.0: ex_405},
    mask=mask_array,
    sample_name="sample",
)

# Properties
data.excitation_wavelengths  # [365.0, 405.0]
data.n_excitations           # 2
data.spatial_shape           # (height, width)

# Access specific excitation
ex_data = data.get_excitation(365.0)
```

### LoadingOptions

Preprocessing configuration.

```python
from spectral_select import LoadingOptions

opts = LoadingOptions(
    cutoff_offset=30,
    apply_rayleigh_cutoff=True,
    apply_second_order_cutoff=True,
    normalize_exposure=True,
    normalize_laser_power=True,
    roi=(100, 400, 100, 400),  # (row_min, row_max, col_min, col_max)
    downscale_factor=2,
)
```

### ValidationMetrics

Clustering evaluation metrics.

```python
from spectral_select import ValidationMetrics

# Usually obtained from Validator.evaluate()
metrics = validator.evaluate(cluster_labels)

# Global metrics
metrics.adjusted_rand_score      # ARI (-1 to 1)
metrics.normalized_mutual_info   # NMI (0 to 1)
metrics.adjusted_mutual_info     # AMI
metrics.v_measure                # V-measure
metrics.homogeneity              # Homogeneity
metrics.completeness             # Completeness
metrics.fowlkes_mallows_score    # Fowlkes-Mallows
metrics.purity                   # Cluster purity

# Per-class metrics
metrics.per_class_precision      # {class_id: precision}
metrics.per_class_recall         # {class_id: recall}
metrics.per_class_f1             # {class_id: f1}

# Other
metrics.confusion_matrix         # np.ndarray
metrics.cluster_to_gt_mapping    # {cluster_id: gt_class}
metrics.n_ground_truth_classes   # int
metrics.n_predicted_clusters     # int

# Summary string
print(metrics.summary())
```

## Extensibility

The library supports custom components through protocols. Implement the protocol interface and pass your class to Config.

```python
from spectral_select import Config
from spectral_select.protocols import ClassifierProtocol

class MyCustomClassifier:
    """Must implement ClassifierProtocol interface."""

    def fit(self, X, y):
        # Your implementation
        pass

    def predict(self, X):
        # Your implementation
        pass

# Use custom classifier
config = Config(
    sample_name="sample",
    classifier=MyCustomClassifier,  # Pass class, not instance
)
```

Available protocols:
- `ClassifierProtocol` - For classification tasks
- `ClusteringProtocol` - For clustering algorithms
- `AutoencoderProtocol` - For custom autoencoder architectures
- `WavelengthRankerProtocol` - For custom ranking methods

## Output Files

When `save_results()` is called, the following files are created:

```
output_dir/
├── wavelength_result.json    # Main results (selected bands, metrics)
├── analysis_config.json      # Configuration snapshot
├── selected_bands.txt        # Human-readable summary
└── layers/                   # If save_tiff_layers=True
    ├── layer_01_ex365nm_em500nm_inf0.850000.tiff
    ├── layer_02_ex365nm_em520nm_inf0.820000.tiff
    ├── ...
    └── layer_metadata.json
```

## Example Workflow

### Complete Analysis Pipeline

```python
from spectral_select import (
    Analyzer, Config, SpectraData,
    Validator, load_ground_truth_from_png
)

# 1. Configuration
config = Config(
    sample_name="Lichens_2",
    model_path="models/lichens_autoencoder.pth",
    output_dir="results/lichens_analysis/",

    # Analysis settings
    n_bands_to_select=30,
    n_important_dimensions=15,
    dimension_selection_method="activation",
    perturbation_method="percentile",
    normalization_method="variance",

    # Diversity constraint
    use_diversity_constraint=True,
    diversity_method="mmr",
    lambda_diversity=0.3,

    # Output
    save_tiff_layers=True,
    save_detailed_results=True,
)

# 2. Load data
data = SpectraData.from_pickle("Data/processed/Lichens_2_processed.pkl")
print(f"Loaded: {data.n_excitations} excitations, shape {data.spatial_shape}")

# 3. Run analysis
analyzer = Analyzer(config)
analyzer.fit(data)

# 4. Review results
print(f"\nSelected {analyzer.result.n_bands} wavelength bands:")
print(f"Compression ratio: {analyzer.result.metrics.compression_ratio:.1f}x")

print("\nTop 10 bands:")
for band in analyzer.get_wavelengths()[:10]:
    print(f"  {band.rank}. Ex={band.excitation_nm:.0f}nm, "
          f"Em={band.emission_nm:.1f}nm (score={band.influence_score:.4f})")

# 5. Save results
output_path = analyzer.save_results()
print(f"\nResults saved to: {output_path}")

# 6. Optional: Validate with ground truth
ground_truth = load_ground_truth_from_png(
    "annotations/lichens_ground_truth.png",
    class_colors={
        "lichen_a": (255, 0, 0, 255),
        "lichen_b": (0, 255, 0, 255),
        "substrate": (0, 0, 255, 255),
    }
)

# ... run clustering on selected bands ...
# cluster_labels = your_clustering_method(reduced_data)

validator = Validator(ground_truth)
metrics = validator.evaluate(cluster_labels)
print(metrics.summary())
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=spectral_select --cov-report=html

# Specific test file
pytest tests/test_config.py -v
```
## Citation

If you use this library in your research, please cite:

```bibtex
@software{spectral_select,
  author = {Narek Meloyan},
  title = {spectral-select: Wavelength Selection for Hyperspectral Imaging},
  year = {2024},
  url = {https://github.com/narekmeloyan/spectral-select}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request
