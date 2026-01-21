# Quick Reference Card

## Essential Commands

### Load Data
```python
from spectral_select import SpectraData
data = SpectraData.from_pickle("Data/processed/sample.pkl")
```

### Configure Analysis
```python
from spectral_select import Config
config = Config(
    sample_name="MySample",
    n_bands_to_select=30,
    training_epochs=100,
    device="cpu",  # or "cuda"
)
```

### Run Analysis
```python
from spectral_select import Analyzer
analyzer = Analyzer(config)
analyzer.fit(data)
```

### Get Results
```python
# Selected wavelengths
bands = analyzer.get_wavelengths()
for b in bands[:5]:
    print(f"Ex={b.excitation_nm}nm, Em={b.emission_nm}nm, Score={b.influence_score:.4f}")

# Save results
analyzer.save_results("results/")
```

### Visualize
```python
from spectral_select import Visualizer
viz = Visualizer.from_analyzer(analyzer)
viz.plot_influence_heatmap()
viz.plot_wavelength_scatter()
viz.plot_influence_ranking()
```

### Validate
```python
from spectral_select import Validator, load_ground_truth_from_png
gt = load_ground_truth_from_png("ground_truth.png")
validator = Validator()
validator.fit(cluster_labels, gt)
print(f"ARI: {validator.score():.3f}")
```

## Configuration Presets

### Fast Test
```python
Config(sample_name="Test", n_bands_to_select=10, training_epochs=20, device="cpu")
```

### Standard
```python
Config(sample_name="Standard", n_bands_to_select=30, training_epochs=100)
```

### Publication Quality
```python
Config(sample_name="Pub", n_bands_to_select=30, training_epochs=500,
       use_diversity_constraint=True, diversity_method="mmr", lambda_diversity=0.3)
```

## Key Parameters

| Parameter | Description | Default | Quick Test | Publication |
|-----------|-------------|---------|------------|-------------|
| `n_bands_to_select` | Wavelengths to select | 30 | 10 | 30 |
| `training_epochs` | Training iterations | 100 | 20 | 500 |
| `n_baseline_patches` | Baseline samples | 50 | 20 | 100 |
| `use_diversity_constraint` | Spread selections | False | False | True |

## Method Options

### Dimension Selection
- `"activation"` - Highest activations (default, recommended)
- `"variance"` - Highest variance
- `"pca"` - Principal components

### Perturbation Method
- `"percentile"` - Percentile-based (default, recommended)
- `"standard_deviation"` - Std dev multiples
- `"absolute_range"` - Fraction of range

### Normalization
- `"variance"` - By variance (default, recommended)
- `"max_per_excitation"` - Scale each excitation to 1
- `"none"` - Raw scores

### Diversity
- `"mmr"` - Maximal Marginal Relevance (recommended)
- `"min_distance"` - Minimum nm distance
- `"none"` - No diversity constraint

## Output Files

```
output_dir/
├── wavelength_result.json    # Machine-readable results
├── analysis_config.json      # Configuration used
├── selected_bands.txt        # Human-readable summary
└── layers/                   # TIFF images (if enabled)
```

## Validation Metrics

| Metric | Range | Good Value | Description |
|--------|-------|------------|-------------|
| ARI | -1 to 1 | > 0.5 | Adjusted Rand Index |
| NMI | 0 to 1 | > 0.5 | Normalized Mutual Info |
| Purity | 0 to 1 | > 0.7 | Cluster purity |
| F1 | 0 to 1 | > 0.7 | Per-class F1 score |

## Common Tasks

### Check GPU availability
```python
import torch
print(torch.cuda.is_available())
```

### Verify data
```python
print(f"Excitations: {data.excitation_wavelengths}")
print(f"Shape: {data.spatial_shape}")
```

### Transform data to selected bands
```python
reduced_data = analyzer.transform(data)
```

### Generate all plots
```python
viz.plot_all()  # Saves all available plots
```

## Files to Know

| File | Purpose |
|------|---------|
| `notebooks/examples/01_quickstart.ipynb` | Basic tutorial |
| `notebooks/examples/02_validation.ipynb` | Validation tutorial |
| `docs/USER_GUIDE.md` | Complete user guide |
| `docs/CONFIGURATION.md` | All config options |
| `docs/TROUBLESHOOTING.md` | Problem solving |
