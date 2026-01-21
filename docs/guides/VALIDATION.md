# Validation Guide

This guide explains how to validate your wavelength selection results against ground truth data.

## Why Validate?

Wavelength selection identifies bands that the autoencoder deems important, but "important" doesn't automatically mean "useful for your specific task." Validation helps you:

1. **Measure quality**: Quantify how well selected bands distinguish different materials
2. **Compare methods**: See which configuration works best for your data
3. **Build confidence**: Verify results before publishing or making decisions

## What You Need

### Ground Truth Data

Ground truth is a labeled image where each pixel has a known class:
- **Class 0**: Background (usually ignored)
- **Class 1, 2, 3...**: Different materials/regions you want to distinguish

Ground truth can come from:
- Manual annotation (drawing regions in ImageJ/Fiji)
- Expert labeling
- Reference measurements
- Previous analyses

### Cluster Predictions

After wavelength selection, you run clustering on the selected bands:
- Each pixel gets assigned to a cluster (0, 1, 2, ...)
- The validation compares these cluster assignments to ground truth

## Step-by-Step Validation

### Step 1: Load Ground Truth

```python
from spectral_select import load_ground_truth_from_png
import numpy as np

# Load from PNG image
ground_truth = load_ground_truth_from_png("ground_truth.png")

# Check what you loaded
print(f"Shape: {ground_truth.labels.shape}")
print(f"Classes: {ground_truth.class_names}")
print(f"Unique labels: {np.unique(ground_truth.labels)}")
```

### Step 2: Get Cluster Predictions

After running wavelength selection and clustering:

```python
# Your clustering results (same shape as ground truth)
cluster_labels = your_clustering_result  # Shape: (height, width)

# Make sure shapes match
print(f"Predictions shape: {cluster_labels.shape}")
print(f"Ground truth shape: {ground_truth.labels.shape}")
```

### Step 3: Run Validation

```python
from spectral_select import Validator

# Create validator
validator = Validator()

# Fit to your data
validator.fit(cluster_labels, ground_truth)

# Get overall score
print(f"Adjusted Rand Index: {validator.score():.3f}")
```

### Step 4: Examine Detailed Metrics

```python
# Get all metrics
metrics = validator.get_metrics()

print(f"ARI: {metrics['ari']:.3f}")
print(f"NMI: {metrics['nmi']:.3f}")
print(f"Purity: {metrics['purity']:.3f}")
print(f"V-measure: {metrics['v_measure']:.3f}")

# Per-class metrics
for class_name, class_metrics in metrics['per_class'].items():
    print(f"\n{class_name}:")
    print(f"  Precision: {class_metrics['precision']:.3f}")
    print(f"  Recall: {class_metrics['recall']:.3f}")
    print(f"  F1: {class_metrics['f1']:.3f}")
```

## Understanding Metrics

### Adjusted Rand Index (ARI)

**Range**: -1 to 1
**Good value**: > 0.5

Measures how similar two clusterings are, adjusted for chance:
- **1.0**: Perfect match
- **0.0**: Random clustering (no better than chance)
- **< 0**: Worse than random

ARI doesn't require clusters to have the same labels as ground truth - it just measures if the same pixels are grouped together.

### Normalized Mutual Information (NMI)

**Range**: 0 to 1
**Good value**: > 0.5

Measures information shared between clusterings:
- **1.0**: Knowing one tells you everything about the other
- **0.0**: Completely independent

### Purity

**Range**: 0 to 1
**Good value**: > 0.7

For each cluster, what fraction belongs to the majority class?
- **1.0**: Each cluster contains only one class
- Lower values mean clusters mix multiple classes

**Caution**: High purity can be achieved by having many small clusters.

### V-Measure

**Range**: 0 to 1
**Good value**: > 0.5

Combines homogeneity (each cluster has one class) and completeness (each class is in one cluster).

### Per-Class Metrics

**Precision**: Of pixels assigned to this class, how many are correct?
**Recall**: Of actual pixels of this class, how many were found?
**F1**: Harmonic mean of precision and recall

## Creating Ground Truth

### From PNG Image

Create a PNG where pixel colors represent classes:

| Color | Class |
|-------|-------|
| Black (0,0,0) | Background (ignored) |
| Red (255,0,0) | Class 1 |
| Green (0,255,0) | Class 2 |
| Blue (0,0,255) | Class 3 |

```python
from spectral_select import load_ground_truth_from_png

ground_truth = load_ground_truth_from_png(
    "labeled_image.png",
    color_to_class={
        (255, 0, 0): 1,    # Red = Class 1
        (0, 255, 0): 2,    # Green = Class 2
        (0, 0, 255): 3,    # Blue = Class 3
    },
    class_names={
        1: "Lichen Type A",
        2: "Lichen Type B",
        3: "Substrate",
    }
)
```

### From NumPy Array

```python
from spectral_select import GroundTruth
import numpy as np

# Your labeled array (0 = background, 1+ = classes)
labels = np.load("my_labels.npy")

ground_truth = GroundTruth(
    labels=labels,
    class_names={
        1: "Material A",
        2: "Material B",
    }
)
```

### Manual Annotation with ImageJ/Fiji

1. Open your hyperspectral image (e.g., a single band)
2. Use ROI Manager to draw regions
3. Fill each region with a unique color
4. Save as PNG
5. Load with `load_ground_truth_from_png()`

## Complete Validation Workflow

```python
from spectral_select import (
    SpectraData, Config, Analyzer, Validator,
    load_ground_truth_from_png
)
from sklearn.cluster import KMeans

# 1. Load data
data = SpectraData.from_pickle("Data/processed/MySample/data.pkl")

# 2. Run wavelength selection
config = Config(
    sample_name="MySample",
    n_bands_to_select=30,
    training_epochs=100,
)
analyzer = Analyzer(config)
analyzer.fit(data)

# 3. Get selected bands data
reduced_data = analyzer.transform(data)

# 4. Run clustering
n_clusters = 5  # Match expected number of materials
flat_data = reduced_data.reshape(-1, reduced_data.shape[-1])
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(flat_data)
cluster_labels = cluster_labels.reshape(data.spatial_shape)

# 5. Load ground truth
ground_truth = load_ground_truth_from_png("ground_truth.png")

# 6. Validate
validator = Validator()
validator.fit(cluster_labels, ground_truth)

print(f"ARI: {validator.score():.3f}")
metrics = validator.get_metrics()
```

## Comparing Configurations

Run multiple configurations and compare:

```python
results = {}

configs = {
    "activation": Config(dimension_selection_method="activation", ...),
    "variance": Config(dimension_selection_method="variance", ...),
    "pca": Config(dimension_selection_method="pca", ...),
}

for name, config in configs.items():
    analyzer = Analyzer(config)
    analyzer.fit(data)

    # Cluster and validate
    reduced = analyzer.transform(data)
    # ... clustering code ...

    validator = Validator()
    validator.fit(predictions, ground_truth)
    results[name] = validator.score()

# Compare
for name, ari in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{name}: ARI = {ari:.3f}")
```

## Troubleshooting Validation

### "Shape mismatch between predictions and ground truth"

Resize one to match the other:

```python
from skimage.transform import resize

predictions_resized = resize(
    predictions,
    ground_truth.labels.shape,
    order=0,  # Nearest neighbor (preserves integer labels)
    preserve_range=True
).astype(int)
```

### "All metrics are 0"

Check if your data has valid values:

```python
# Check predictions
print(f"Unique predictions: {np.unique(predictions)}")

# Check ground truth
print(f"Unique ground truth: {np.unique(ground_truth.labels)}")

# Check overlap
mask = ground_truth.labels > 0
print(f"Valid GT pixels: {mask.sum()}")
print(f"Predictions in valid region: {np.unique(predictions[mask])}")
```

### Low validation scores

This isn't necessarily wrong. Consider:

1. **Select more bands**: Try 40-50 instead of 30
2. **Use diversity constraints**: Ensure spectral coverage
3. **Tune clustering**: Try different algorithms or parameters
4. **Check ground truth quality**: Ensure labels are accurate

## Interpreting Results

### What's a "good" score?

| ARI | Interpretation |
|-----|----------------|
| > 0.8 | Excellent - near-perfect separation |
| 0.6 - 0.8 | Good - clear separation with some errors |
| 0.4 - 0.6 | Moderate - useful but noisy |
| 0.2 - 0.4 | Weak - marginal separation |
| < 0.2 | Poor - little better than random |

### Scores depend on your data

- Complex samples with subtle differences: lower scores expected
- Clear material boundaries: higher scores expected
- Noisy data: lower scores expected

### Multiple metrics matter

Don't rely on just one metric:
- High purity but low ARI? Too many small clusters
- High ARI but low class F1? Some classes perform poorly

## Next Steps

After validation:
1. **If scores are good**: Document your configuration and proceed
2. **If scores are low**: Try different parameters (see [WAVELENGTH_SELECTION.md](WAVELENGTH_SELECTION.md))
3. **For publication**: Report multiple metrics with confidence intervals
