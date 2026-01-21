# Wavelength Selection Guide

This guide provides detailed instructions for running wavelength selection analysis.

## Overview

The wavelength selection process identifies the most informative excitation/emission combinations in your hyperspectral data using an autoencoder-based approach.

## Step-by-Step Guide

### Step 1: Load Your Data

```python
from spectral_select import SpectraData

# Load from processed file
data = SpectraData.from_pickle("Data/processed/YourSample/data_processed.pkl")

# Verify the data
print(f"Sample: {data.sample_name}")
print(f"Spatial shape: {data.spatial_shape}")
print(f"Excitations: {data.excitation_wavelengths}")
```

### Step 2: Configure the Analysis

```python
from spectral_select import Config

config = Config(
    sample_name="YourSample",
    output_dir="results/YourSample_analysis/",
    n_bands_to_select=30,
    dimension_selection_method="activation",
    perturbation_method="percentile",
    normalization_method="variance",
    training_epochs=100,
    device="cpu",  # or "cuda" for GPU
)
```

### Step 3: Run the Analysis

```python
from spectral_select import Analyzer

analyzer = Analyzer(config)
analyzer.fit(data)
print("Analysis complete!")
```

### Step 4: Examine Results

```python
# Get selected wavelengths
wavelengths = analyzer.get_wavelengths()

# Show top 10
for band in wavelengths[:10]:
    print(f"Rank {band.rank}: Ex={band.excitation_nm}nm, Em={band.emission_nm}nm")
```

### Step 5: Save and Visualize

```python
# Save results
analyzer.save_results()

# Create visualizations
from spectral_select import Visualizer
viz = Visualizer.from_analyzer(analyzer)
viz.plot_all()
```

## Choosing Parameters

### How many wavelengths to select?

| Use Case | Recommended |
|----------|-------------|
| Quick visualization | 5-10 |
| Standard clustering | 15-30 |
| Detailed analysis | 30-50 |

### Method Selection Guide

**Dimension Selection:**
- `"activation"` - Default, works well for most cases
- `"variance"` - Better for high-variability data
- `"pca"` - Most statistically principled

**Normalization:**
- `"variance"` - Fair comparison across excitations
- `"max_per_excitation"` - Ensures representation from all excitations
- `"none"` - Preserves raw importance scores

### When to Use Diversity Constraints

Enable when all selections come from one excitation:

```python
config = Config(
    use_diversity_constraint=True,
    diversity_method="mmr",
    lambda_diversity=0.5,
)
```

## Tips for Best Results

1. **Start simple**: Use default parameters first
2. **Check training**: Make sure loss decreases
3. **Visualize**: Look at heatmaps to understand selection
4. **Validate**: If you have ground truth, measure performance
5. **Iterate**: Try different parameters and compare

## Next Steps

After wavelength selection:
1. Run clustering on selected bands
2. Validate against ground truth (see [VALIDATION.md](VALIDATION.md))
3. Iterate on parameters based on results
