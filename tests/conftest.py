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

from pathlib import Path
from typing import List

import numpy as np
import pytest

from spectral_select import (
    Config,
    ExcitationData,
    SpectraData,
    WavelengthBand,
)


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def sample_config() -> Config:
    """Return a minimal Config for testing.

    The config uses default values with a test sample name.
    Mutable fixture (function scope) so tests can modify safely.
    """
    return Config(sample_name="test_sample")


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================


@pytest.fixture
def synthetic_excitation_data() -> ExcitationData:
    """Return ExcitationData for a single excitation wavelength.

    Creates a small 10x10 spatial cube with 5 emission bands.
    Uses fixed seed for reproducibility.
    """
    np.random.seed(42)

    height, width, n_bands = 10, 10, 5
    cube = np.random.rand(height, width, n_bands).astype(np.float32)

    # Emission wavelengths from 500-540nm in 10nm steps
    emission_wavelengths = [500.0, 510.0, 520.0, 530.0, 540.0]

    return ExcitationData(
        excitation_nm=365.0,
        cube=cube,
        emission_wavelengths=emission_wavelengths,
        exposure_time=1.0,
        laser_power=100.0,
    )


@pytest.fixture
def synthetic_spectra_data() -> SpectraData:
    """Return SpectraData with multiple excitation wavelengths.

    Creates a small dataset with:
    - 10x10 spatial dimensions
    - 3 excitation wavelengths (365, 405, 450nm)
    - 5 emission bands per excitation
    - All-valid mask

    Uses fixed seed for reproducibility.
    """
    np.random.seed(42)

    height, width, n_bands = 10, 10, 5
    excitation_wavelengths = [365.0, 405.0, 450.0]

    excitations = {}
    for ex_nm in excitation_wavelengths:
        cube = np.random.rand(height, width, n_bands).astype(np.float32)
        # Emission wavelengths shift with excitation
        base_em = ex_nm + 50  # Start 50nm above excitation
        emission_wls = [base_em + i * 10 for i in range(n_bands)]

        excitations[ex_nm] = ExcitationData(
            excitation_nm=ex_nm,
            cube=cube,
            emission_wavelengths=emission_wls,
            exposure_time=1.0,
            laser_power=100.0,
        )

    # All-valid mask
    mask = np.ones((height, width), dtype=bool)

    return SpectraData(
        excitations=excitations,
        mask=mask,
        sample_name="synthetic_sample",
        metadata={"source": "test_fixture"},
    )


@pytest.fixture
def synthetic_wavelength_bands() -> List[WavelengthBand]:
    """Return a list of 5 WavelengthBand objects for testing.

    Creates bands with sequential ranks and varying excitations.
    """
    bands = [
        WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=0.95,
        ),
        WavelengthBand(
            rank=2,
            excitation_nm=405.0,
            emission_nm=520.0,
            emission_band_index=2,
            influence_score=0.85,
        ),
        WavelengthBand(
            rank=3,
            excitation_nm=365.0,
            emission_nm=530.0,
            emission_band_index=3,
            influence_score=0.75,
        ),
        WavelengthBand(
            rank=4,
            excitation_nm=450.0,
            emission_nm=540.0,
            emission_band_index=4,
            influence_score=0.65,
        ),
        WavelengthBand(
            rank=5,
            excitation_nm=405.0,
            emission_nm=510.0,
            emission_band_index=1,
            influence_score=0.55,
        ),
    ]
    return bands


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test outputs.

    Uses pytest's tmp_path fixture which is automatically
    cleaned up after the test session.
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
