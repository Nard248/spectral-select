# Data Processing Guide

This guide explains how to prepare your hyperspectral data for analysis with Spectral-Select.

## Overview

Before running wavelength selection, your raw hyperspectral data needs to be:

1. **Loaded** from your instrument's file format
2. **Preprocessed** to remove artifacts
3. **Normalized** for consistent comparison
4. **Saved** in the expected format

## Supported Input Formats

### Raw Formats (Requires Fiji/ImageJ)
- `.im3` - Nuance/CRI hyperspectral format
- Other formats supported by Bio-Formats plugin

### Pre-processed Formats
- `.pkl` - Python pickle files (recommended for processed data)
- NumPy arrays in memory

## Data Organization

Organize your raw data like this:

```
Data/
└── Raw/
    └── YourSample/
        ├── 310.im3          # Named by excitation wavelength
        ├── 325.im3
        ├── 340.im3
        ├── 365.im3
        ├── 385.im3
        ├── 400.im3
        ├── 415.im3
        ├── 430.im3
        ├── metadata.xlsx    # Exposure times
        └── TLS Scans/
            └── average_power.xlsx  # Laser powers
```

### Metadata Files

**metadata.xlsx** - Contains exposure times:
| Excitation | Exposure |
|------------|----------|
| 310 | 5432.9 |
| 325 | 924.52 |
| 340 | 656.47 |
| ... | ... |

**average_power.xlsx** - Contains laser powers:
| Excitation Wavelength (nm) | Average Power (W) |
|----------------------------|-------------------|
| 310 | 0.000416 |
| 325 | 0.000305 |
| 340 | 0.000903 |
| ... | ... |

## Processing Steps

### Step 1: Load Raw Data

```python
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, "scripts")

from data_processing import HyperspectralProcessor

# Define paths
data_path = Path("Data/Raw/YourSample")
metadata_path = data_path / "metadata.xlsx"
laser_power_path = data_path / "TLS Scans" / "average_power.xlsx"
output_dir = Path("Data/processed/YourSample")

# Create processor
processor = HyperspectralProcessor(
    data_path=str(data_path),
    metadata_path=str(metadata_path),
    laser_power_excel=str(laser_power_path),
    cutoff_offset=40,  # nm offset for spectral cutoffs
    verbose=True,
)
```

### Step 2: Apply Spectral Cutoffs

Hyperspectral fluorescence data has artifacts that need to be removed:

**Rayleigh Scattering**: At emission wavelengths near the excitation wavelength, you see scattered laser light, not fluorescence.

**Second-Order Diffraction**: The spectrometer grating creates artifacts at 2x the excitation wavelength.

```python
# Process with automatic cutoffs
output_files = processor.process_full_pipeline(
    output_dir=str(output_dir),
    exposure_reference="max",   # Normalize to max exposure
    power_reference="min",      # Normalize to min laser power
)
```

The `cutoff_offset` parameter (in nm) controls how aggressively artifacts are removed:
- **40nm**: Conservative, keeps more data but may have some artifacts
- **60nm**: Aggressive, removes more potential artifacts

### Step 3: Understand Normalization

#### Exposure Time Normalization

Different excitation wavelengths may have different exposure times. Longer exposures = more signal. We normalize so all excitations have comparable intensities.

```
Normalized_value = Raw_value * (Reference_exposure / Actual_exposure)
```

Using `exposure_reference="max"` means all data is scaled as if using the longest exposure time.

#### Laser Power Normalization

Different lasers have different powers. Higher power = more fluorescence. We normalize to account for this.

```
Normalized_value = Exposure_normalized * (Reference_power / Actual_power)
```

Using `power_reference="min"` scales to the weakest laser.

### Step 4: Review Processing Results

```python
# Print processing summary
processor.print_summary()
```

This shows:
- Number of emission bands per excitation
- Wavelength ranges after cutoff
- Normalization factors applied

### Step 5: Load Processed Data

```python
from spectral_select import SpectraData

# Load the processed data
data = SpectraData.from_pickle(
    "Data/processed/YourSample/data_cutoff_40nm_exposure_max_power_min.pkl"
)

# Check what you loaded
print(f"Sample name: {data.sample_name}")
print(f"Spatial shape: {data.spatial_shape}")
print(f"Excitation wavelengths: {data.excitation_wavelengths}")

# Details per excitation
for ex in data.excitation_wavelengths:
    ex_data = data.get_excitation(ex)
    print(f"  Ex {ex}nm: {ex_data.n_bands} bands, shape {ex_data.shape}")
```

## Working with Different Data Sources

### If You Have TIFF Stacks

```python
import numpy as np
from spectral_select import SpectraData, ExcitationData
from tifffile import imread

# Load TIFF stacks (one per excitation)
cube_365 = imread("ex_365nm.tif")  # Shape: (height, width, n_bands)

# Make sure shape is (height, width, n_bands)
if cube_365.shape[0] < cube_365.shape[2]:
    cube_365 = np.transpose(cube_365, (1, 2, 0))

# Create ExcitationData
ex_365 = ExcitationData(
    excitation_nm=365.0,
    cube=cube_365,
    emission_wavelengths=[400 + i*10 for i in range(cube_365.shape[2])],
)

# Combine into SpectraData
data = SpectraData(
    excitations={365.0: ex_365},
    sample_name="MyTiffData",
)
```

### If You Have NumPy Arrays

```python
import numpy as np
from spectral_select import SpectraData, ExcitationData

# Your data arrays (shape: height, width, n_bands)
cube_dict = {
    365.0: my_array_365,  # (H, W, 50)
    405.0: my_array_405,  # (H, W, 45)
}

# Emission wavelengths for each excitation
wavelengths_dict = {
    365.0: [400 + i*6 for i in range(50)],
    405.0: [420 + i*6 for i in range(45)],
}

# Create SpectraData
excitations = {}
for ex_nm, cube in cube_dict.items():
    excitations[ex_nm] = ExcitationData(
        excitation_nm=ex_nm,
        cube=cube,
        emission_wavelengths=wavelengths_dict[ex_nm],
    )

data = SpectraData(excitations=excitations, sample_name="MyData")
```

## Creating Masks

Masks define which pixels to include in the analysis. Masked pixels (value 0) are excluded.

### From Binary Image

```python
import numpy as np
from PIL import Image

# Load mask image (should be grayscale or binary)
mask_img = Image.open("mask.png").convert("L")
mask = np.array(mask_img)

# Convert to binary (1 = include, 0 = exclude)
mask = (mask > 128).astype(np.float32)

# Apply to SpectraData
data = SpectraData(
    excitations=excitations,
    mask=mask,
    sample_name="MaskedData",
)
```

## Quality Checks

Before running analysis, verify your data:

### Check for NaN Values

```python
for ex in data.excitation_wavelengths:
    cube = data.get_excitation(ex).cube
    nan_count = np.isnan(cube).sum()
    if nan_count > 0:
        print(f"Warning: Ex {ex}nm has {nan_count} NaN values")
```

### Check Value Ranges

```python
for ex in data.excitation_wavelengths:
    cube = data.get_excitation(ex).cube
    print(f"Ex {ex}nm: min={cube.min():.4f}, max={cube.max():.4f}")
```

### Visualize a Single Band

```python
import matplotlib.pyplot as plt

ex_data = data.get_excitation(365.0)
band_index = 10  # Middle band

plt.figure(figsize=(10, 8))
plt.imshow(ex_data.cube[:, :, band_index], cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'Ex=365nm, Em={ex_data.emission_wavelengths[band_index]}nm')
plt.show()
```

## Common Issues

### "Mask shape doesn't match data"

Ensure your mask has the same height and width as your data cubes:
```python
print(f"Mask shape: {mask.shape}")
print(f"Data shape: {data.spatial_shape}")
```

### "No data loaded"

Check that your file paths are correct and files exist:
```python
from pathlib import Path
data_path = Path("Data/Raw/YourSample")
print(f"Path exists: {data_path.exists()}")
print(f"Files found: {list(data_path.glob('*.im3'))}")
```

### "Cutoff removed all data"

Your cutoff offset might be too large. Try a smaller value (30nm instead of 60nm).

## Next Steps

Once your data is processed:

1. **Configure analysis**: See [../CONFIGURATION.md](../CONFIGURATION.md)
2. **Run wavelength selection**: See [WAVELENGTH_SELECTION.md](WAVELENGTH_SELECTION.md)
3. **Validate results**: See [VALIDATION.md](VALIDATION.md)
