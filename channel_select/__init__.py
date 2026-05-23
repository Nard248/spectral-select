"""Domain-agnostic dependency-aware channel selection.

Generalizes the spectral_select perturbation-autoencoder method to any
group-structured multi-channel data (hyperspectral cubes, sensor time series).
"""
from .protocols import SelectionConfig, GroupStructuredModel, GroupedChannelData

__all__ = ["SelectionConfig", "GroupStructuredModel", "GroupedChannelData"]
