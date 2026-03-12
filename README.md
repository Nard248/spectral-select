# spectral-select

Autoencoder-based wavelength selection for 4D hyperspectral imaging via latent space perturbation analysis.

<!-- Paper link: TODO -->

## Overview

`spectral-select` identifies the most informative wavelength combinations in multi-excitation hyperspectral datasets by:

1. Training a convolutional autoencoder on the hyperspectral data
2. Analyzing the latent space to identify important dimensions
3. Perturbing latent dimensions and measuring reconstruction sensitivity
4. Ranking and selecting wavelength bands based on influence scores

This enables dimensionality reduction while preserving the most discriminative spectral information for downstream clustering and classification.

## Installation

```bash
pip install -e "."
```

For development (testing, notebooks):

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.11. See `pyproject.toml` for the full dependency list.

## Quick Start

```python
from spectral_select import Analyzer, Config, SpectraData

# Configure the analysis
config = Config(
    sample_name="my_sample",
    n_bands_to_select=30,
    model_path="models/autoencoder.pth",
    output_dir="results/",
)

# Load preprocessed hyperspectral data
data = SpectraData.from_raw_dict(your_data_dict)

# Run wavelength selection
analyzer = Analyzer(config)
results = analyzer.fit(data)

# Access selected wavelengths
print(results.selected_wavelengths)
```

## Repository Structure

```
spectral_select/           Core library (pip-installable)
  models/                  Autoencoder architecture, dataset, training
  analyzer.py              Main analysis engine
  config.py                Configuration management
  types.py                 Data types (SpectraData, WavelengthResult, etc.)
  loader.py                Data loading utilities
  validation.py            Ground-truth validation
  results.py               Results management and export
  visualizer.py            Plotting and visualization
  viewer.py                Interactive hyperspectral viewer
mehsi_preprocessor/        Optional preprocessing GUI (requires PyQt6)
  io/                      Raw .im3 file loading
  steps/                   Processing pipeline steps
  widgets/                 GUI components
experiments/               Paper reproduction scripts
examples/                  Tutorial notebooks
tests/                     Test suite
docs/                      Documentation
```

## Reproducing Paper Results

See [`experiments/README.md`](experiments/README.md) for instructions to reproduce all paper figures and tables.

## Documentation

See the [`docs/`](docs/) directory and [`examples/`](examples/) notebooks for detailed usage guides.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{meloyan2025spectralselect,
  title={Autoencoder-Based Wavelength Selection for 4D Hyperspectral Imaging},
  author={Meloyan, Narek},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE).
