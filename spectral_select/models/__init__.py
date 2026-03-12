"""
Model components for spectral_select.

Provides the autoencoder architecture, dataset handling, and training
routines used by the wavelength selection analysis.
"""

from .autoencoder import HyperspectralCAEWithMasking
from .dataset import MaskedHyperspectralDataset
from .training import train_with_masking

__all__ = [
    "HyperspectralCAEWithMasking",
    "MaskedHyperspectralDataset",
    "train_with_masking",
]
