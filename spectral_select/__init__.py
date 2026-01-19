"""
spectral_select - Wavelength selection for hyperspectral imaging.

A library for reproducible wavelength selection analysis using autoencoder-based
latent space perturbation to identify informative spectral bands in hyperspectral
imaging data.

Usage:
    from spectral_select import Analyzer, Config

    config = Config(sample_name="Lichens_2")
    analyzer = Analyzer(config)
    results = analyzer.fit(data)

For more information, see the documentation and examples in the repository.
"""

from .analyzer import Analyzer
from .config import Config
from .validation import Validator
from .visualizer import Visualizer

__version__ = "0.1.0"

__all__ = [
    "Analyzer",
    "Config",
    "Validator",
    "Visualizer",
    "__version__",
]
