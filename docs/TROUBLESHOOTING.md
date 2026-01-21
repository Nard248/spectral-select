# Troubleshooting Guide

Common problems and their solutions when using Spectral-Select.

## Installation Issues

### "No module named 'spectral_select'"

**Cause**: Package not installed or wrong Python environment.

**Solutions**:
1. Make sure you've activated your virtual environment:
   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

2. Install the package:
   ```bash
   pip install -e ".[dev]"
   ```

3. Verify installation:
   ```bash
   python -c "import spectral_select; print('OK')"
   ```

### "CUDA out of memory"

**Cause**: GPU doesn't have enough memory for the data/model.

**Solutions**:
1. Use CPU instead:
   ```python
   config = Config(device="cpu", ...)
   ```

2. Reduce batch size / chunk size:
   ```python
   config = Config(training_chunk_size=128, ...)
   ```

3. Process a smaller region of your image first.

### PyTorch installation fails

**Solution**: Install PyTorch separately first:
```bash
# CPU only
pip install torch torchvision

# With CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest:
```bash
pip install -e ".[dev]"
```

## Data Loading Issues

### "FileNotFoundError: Data file not found"

**Cause**: Incorrect file path.

**Solutions**:
1. Check the path exists:
   ```python
   from pathlib import Path
   p = Path("your/path/to/data.pkl")
   print(f"Exists: {p.exists()}")
   print(f"Absolute: {p.absolute()}")
   ```

2. Use absolute paths or correct relative paths.

### "Shape mismatch" errors

**Cause**: Data arrays have inconsistent dimensions.

**Solution**: Verify all excitations have same spatial dimensions:
```python
for ex in data.excitation_wavelengths:
    ex_data = data.get_excitation(ex)
    print(f"Ex {ex}: shape = {ex_data.shape}")
```

All should have the same (height, width, _).

## Analysis Issues

### Analysis is extremely slow

**Causes and solutions**:

1. **Using CPU instead of GPU**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   # If True, use device="cuda"
   ```

2. **Too many training epochs**:
   ```python
   config = Config(training_epochs=20, ...)  # Start small
   ```

3. **Large image size**: Consider cropping or downscaling.

### "No wavelengths selected" or empty results

**Causes**:

1. **All data is masked**: Check your mask isn't all zeros:
   ```python
   print(f"Mask sum: {mask.sum()}")
   print(f"Valid pixels: {(mask > 0).sum()}")
   ```

2. **Data is all zeros or NaN**: Check for valid data:
   ```python
   cube = data.get_excitation(365.0).cube
   print(f"Min: {cube.min()}, Max: {cube.max()}")
   print(f"NaN count: {np.isnan(cube).sum()}")
   ```

### "NaN in influence scores"

**Cause**: Division by zero or invalid operations.

**Solutions**:
1. Check for zero-variance dimensions
2. Try a different normalization method:
   ```python
   config = Config(normalization_method="none", ...)
   ```

### Results don't make sense

**Check these things**:

1. **Data range**: Values should be positive and reasonable
2. **Model quality**: Check training loss decreased over epochs
3. **Sanity check**: Visualize a few selected bands

## Visualization Issues

### "No result available for heatmap"

**Cause**: Trying to visualize before fitting.

**Solution**: Make sure analysis is complete:
```python
analyzer.fit(data)  # Run analysis first
if analyzer.is_fitted:
    viz = Visualizer.from_analyzer(analyzer)
    viz.plot_influence_heatmap()
```

### Plots look wrong / cut off

**Solution**: Adjust figure size:
```python
viz = Visualizer(figsize=(15, 10), dpi=150)
```

## Validation Issues

### "Ground truth shape doesn't match predictions"

**Cause**: Arrays have different dimensions.

**Solution**: Resize to match:
```python
from skimage.transform import resize

predictions_resized = resize(
    predictions,
    ground_truth.labels.shape,
    order=0,  # Nearest neighbor for labels
    preserve_range=True
).astype(int)
```

### "All metrics are 0 or NaN"

**Causes**:
1. **Only one class in predictions**: Check cluster labels:
   ```python
   print(f"Unique clusters: {np.unique(predictions)}")
   ```

2. **Labels don't overlap**: Ensure same pixels are valid in both arrays

### Low validation scores

**This is not necessarily a bug**. Low scores might indicate:
- The selected wavelengths don't capture distinguishing features
- Try selecting more bands
- Try using diversity constraints
- Your clustering method may need tuning

## Memory Issues

### "MemoryError" or system becomes unresponsive

**Solutions**:

1. **Process smaller regions**: Crop data to region of interest
2. **Reduce chunk size**: `config = Config(training_chunk_size=128, ...)`
3. **Use a machine with more RAM**
4. **Close other applications**

## Getting More Help

If you can't solve your problem:

1. **Check existing issues** on GitHub
2. **Create a new issue** with:
   - Operating system and Python version
   - Full error message and traceback
   - Minimal code to reproduce the problem
   - Data description (dimensions, file format)

**Useful diagnostic information:**
```python
import sys
import torch
import numpy as np

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy: {np.__version__}")
```
