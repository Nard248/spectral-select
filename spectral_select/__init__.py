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

For custom components, implement the protocols:
    from spectral_select.protocols import ClassifierProtocol

For more information, see the documentation and examples in the repository.
"""

from .analyzer import Analyzer
from .config import Config
from .protocols import (
    AutoencoderProtocol,
    ClassifierProtocol,
    ClusteringProtocol,
    WavelengthRankerProtocol,
)
from .types import (
    AnalysisMetrics,
    ExcitationData,
    GroundTruth,
    LoadingOptions,
    SpectraData,
    ValidationMetrics,
    WavelengthBand,
    WavelengthResult,
)
from .loader import DataLoader, DataLoadingError
from .results import ResultsManager
from .validation import Validator, load_ground_truth_from_png
from .visualizer import Visualizer

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Analyzer",
    "Config",
    "DataLoader",
    "ResultsManager",
    "Validator",
    "Visualizer",
    # Utilities
    "load_ground_truth_from_png",
    # Exceptions
    "DataLoadingError",
    # Data types
    "AnalysisMetrics",
    "ExcitationData",
    "GroundTruth",
    "LoadingOptions",
    "SpectraData",
    "ValidationMetrics",
    "WavelengthBand",
    "WavelengthResult",
    # Protocols for extensibility
    "AutoencoderProtocol",
    "ClassifierProtocol",
    "ClusteringProtocol",
    "WavelengthRankerProtocol",
    # Metadata
    "__version__",
]
