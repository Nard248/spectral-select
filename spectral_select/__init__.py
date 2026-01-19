"""
spectral_select: 4D Hyperspectral Wavelength Selection Library

A library for reproducible wavelength selection analysis using autoencoder-based
latent space perturbation to identify informative spectral bands in hyperspectral
imaging data.

Example usage:
    >>> from spectral_select import Analyzer
    >>> analyzer = Analyzer(config)
    >>> results = analyzer.run()

For more information, see the documentation and examples in the repository.
"""

__version__ = "0.1.0"

# Public API exports (populated in subsequent phases)
__all__ = [
    "__version__",
    # Future exports:
    # "Analyzer",
    # "Config",
    # "Visualizer",
]
