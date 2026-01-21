# Configuration Reference

This document describes all configuration options available in Spectral-Select.

## Basic Usage

```python
from spectral_select import Config

config = Config(
    sample_name="MySample",
    n_bands_to_select=30,
)
```

## Complete Parameter Reference

### Identification

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_name` | str | "sample" | Name to identify your sample in outputs |

### File Paths

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | Path/None | None | Path to input data file |
| `mask_path` | Path/None | None | Path to mask file (binary) |
| `model_path` | Path/None | None | Path to save/load autoencoder model |
| `output_dir` | Path/None | None | Directory for all outputs |

### Wavelength Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bands_to_select` | int | 30 | How many wavelengths to select |
| `n_layers_to_extract` | int | 10 | Number of TIFF layers to save (top-ranked bands) |

### Dimension Selection Method

Controls how the software identifies important latent dimensions:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension_selection_method` | str | "activation" | Method for selecting important dimensions |
| `n_important_dimensions` | int | 15 | Number of latent dimensions to analyze |

**Methods:**
- `"activation"` - Selects dimensions with highest activation magnitudes (recommended)
- `"variance"` - Selects dimensions with highest variance across patches
- `"pca"` - Uses PCA to identify principal components in latent space

### Perturbation Method

Controls how latent dimensions are perturbed to measure influence:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `perturbation_method` | str | "percentile" | How to calculate perturbation magnitude |
| `perturbation_magnitudes` | List[float] | [10, 20, 30] | Perturbation strength values |
| `perturbation_directions` | List[str] | ["bidirectional"] | Direction(s) of perturbation |

**Methods:**
- `"percentile"` - Perturb to specified percentile values (e.g., 10th, 90th percentile)
- `"standard_deviation"` - Perturb by multiples of standard deviation
- `"absolute_range"` - Perturb by fraction of the value range

**Directions:**
- `"bidirectional"` - Perturb both positive and negative (recommended)
- `"positive"` - Only positive perturbations
- `"negative"` - Only negative perturbations

### Normalization Method

Controls how influence scores are normalized across excitations:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalization_method` | str | "variance" | How to normalize influence scores |

**Methods:**
- `"variance"` - Normalize by excitation wavelength variance (recommended)
- `"max_per_excitation"` - Scale each excitation's max to 1.0
- `"none"` - No normalization (raw scores)

### Diversity Constraints

Ensures selected wavelengths are spread across the spectral range:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_diversity_constraint` | bool | False | Enable diversity constraints |
| `diversity_method` | str | "mmr" | Diversity algorithm |
| `lambda_diversity` | float | 0.5 | Trade-off: 0=pure relevance, 1=pure diversity |
| `min_distance_nm` | float | 15.0 | Minimum nm distance between selections |

**Methods:**
- `"mmr"` - Maximal Marginal Relevance: balances relevance and diversity
- `"min_distance"` - Enforces minimum spectral distance between bands
- `"none"` - No diversity (simple ranking)

### Autoencoder Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_k1` | int | 20 | Number of filters in first conv layer |
| `model_k3` | int | 20 | Number of filters in bottleneck layer |
| `model_filter_size` | int | 5 | Convolutional filter size (must be odd) |
| `model_sparsity_target` | float | 0.1 | Target activation sparsity (0.1 = 10%) |
| `model_sparsity_weight` | float | 1.0 | Weight of sparsity loss term |
| `model_dropout_rate` | float | 0.5 | Dropout rate during training |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_epochs` | int | 100 | Number of training epochs |
| `training_lr` | float | 0.001 | Learning rate |
| `training_chunk_size` | int | 256 | Image chunk size for training |
| `training_chunk_overlap` | int | 64 | Overlap between chunks |
| `training_early_stopping_patience` | int | 10 | Epochs without improvement before stopping |

### Baseline Extraction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_baseline_patches` | int | 50 | Number of patches for baseline |
| `patch_size` | int | 32 | Patch size in pixels |
| `patch_stride` | int | 16 | Stride between patches |

### Output Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_tiff_layers` | bool | True | Save top bands as TIFF images |
| `save_visualizations` | bool | True | Generate visualization plots |
| `save_detailed_results` | bool | True | Save detailed JSON results |

### Computation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | "cuda" | Device: "cuda", "cpu", or "mps" |
| `random_seed` | int | 42 | Random seed for reproducibility |

## Configuration Presets

### Fast Exploration

```python
config = Config(
    sample_name="QuickTest",
    n_bands_to_select=10,
    training_epochs=20,
    n_baseline_patches=20,
    patch_size=32,
    save_tiff_layers=False,
    save_visualizations=False,
)
```

### Standard Analysis

```python
config = Config(
    sample_name="Standard",
    n_bands_to_select=30,
    training_epochs=100,
    dimension_selection_method="activation",
    perturbation_method="percentile",
    normalization_method="variance",
)
```

### High Quality (Publication)

```python
config = Config(
    sample_name="Publication",
    n_bands_to_select=30,
    training_epochs=500,
    n_baseline_patches=100,
    n_important_dimensions=20,
    use_diversity_constraint=True,
    diversity_method="mmr",
    lambda_diversity=0.3,
    save_tiff_layers=True,
    save_visualizations=True,
    save_detailed_results=True,
)
```

### Diverse Selections

```python
config = Config(
    sample_name="Diverse",
    n_bands_to_select=25,
    use_diversity_constraint=True,
    diversity_method="mmr",
    lambda_diversity=0.5,  # Balance relevance and diversity
)
```

### Minimum Distance Constraint

```python
config = Config(
    sample_name="Spaced",
    n_bands_to_select=20,
    use_diversity_constraint=True,
    diversity_method="min_distance",
    min_distance_nm=30.0,  # At least 30nm apart
)
```

## Loading and Saving Configurations

### Save to YAML

```python
config.to_yaml("my_config.yaml")
```

Example YAML file:
```yaml
sample_name: MySample
n_bands_to_select: 30
training_epochs: 100
dimension_selection_method: activation
perturbation_method: percentile
normalization_method: variance
device: cuda
```

### Load from YAML

```python
config = Config.from_yaml("my_config.yaml")
```

### Save to JSON

```python
config.to_json("my_config.json")
```

### Load from JSON

```python
config = Config.from_json("my_config.json")
```

## Method Comparison

### Dimension Selection Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| `activation` | General use | Fast, intuitive | May favor high-intensity regions |
| `variance` | Variable data | Captures variation | May include noise |
| `pca` | Correlated features | Statistically principled | Computationally heavier |

### Perturbation Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| `percentile` | General use | Robust to outliers | Fixed percentiles |
| `standard_deviation` | Normal distributions | Statistically meaningful | Sensitive to outliers |
| `absolute_range` | Bounded data | Predictable range | Less adaptive |

### Normalization Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| `variance` | Comparing excitations | Fair comparison | May reduce contrast |
| `max_per_excitation` | Balanced selection | Equal weight per excitation | May overweight weak excitations |
| `none` | Raw scores | No information loss | Biased toward strong signals |

### Diversity Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| `mmr` | Flexible trade-off | Tunable lambda | Requires tuning |
| `min_distance` | Physical separation | Guaranteed spacing | May miss important close bands |
| `none` | Pure ranking | Maximum relevance | May have redundant bands |

## Validation

All configuration values are validated when you create a Config object:

```python
# This will raise an error
config = Config(
    n_bands_to_select=-5,  # ValueError: must be positive
)
```

Common validation errors:
- `n_bands_to_select` must be positive
- `model_filter_size` must be odd
- `lambda_diversity` must be between 0 and 1
- `device` must be "cuda", "cpu", or "mps"
