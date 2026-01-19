"""Pytest configuration and fixtures for spectral_select tests.

This module provides shared fixtures for generating synthetic test data
that mirrors the structure of real hyperspectral data without requiring
large data files.

Fixtures:
    sample_config: Minimal Config for testing
    synthetic_spectra_data: SpectraData with small synthetic cube
    synthetic_excitation_data: ExcitationData for single excitation
    synthetic_wavelength_bands: List of WavelengthBand objects
    tmp_output_dir: Temporary directory for test outputs
"""

import pytest

import spectral_select
