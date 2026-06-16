"""Shared perturbation-based selection engine.

Single source of truth for the Perturbation-Based Autoencoder selection algorithm,
used by both spectral_select (hyperspectral imaging) and channel_select (general
grouped/temporal channels).
"""
from selection_core.engine import (
    select_important_dimensions,
    latent_statistics,
    perturbation_amount,
    measure_influence,
    accumulate_influence,
    normalize_influence,
)

__all__ = [
    "select_important_dimensions",
    "latent_statistics",
    "perturbation_amount",
    "measure_influence",
    "accumulate_influence",
    "normalize_influence",
]
