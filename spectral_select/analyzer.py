"""Core wavelength selection analysis engine.

This module provides the Analyzer class, the main interface for running
wavelength selection analysis on hyperspectral data.

Example:
    from spectral_select import Analyzer, Config, SpectraData

    config = Config(sample_name="Lichens_2")
    analyzer = Analyzer(config)

    # Load data
    data = SpectraData.from_pickle("processed_data.pkl")

    # Run analysis
    analyzer.fit(data)

    # Get selected wavelengths
    bands = analyzer.get_wavelengths()
    print(f"Selected {len(bands)} wavelength combinations")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config
from .types import SpectraData, WavelengthBand, WavelengthResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Analyzer:
    """Main analyzer for wavelength selection using autoencoder perturbation.

    This class provides a scikit-learn-style interface for running wavelength
    selection analysis on hyperspectral data. It identifies the most informative
    wavelength combinations by analyzing how perturbations in the autoencoder's
    latent space affect the reconstruction.

    Attributes:
        config: Configuration object controlling analysis parameters.
        device: PyTorch device used for computation.

    Example:
        config = Config(sample_name="sample", n_bands_to_select=30)
        analyzer = Analyzer(config)
        analyzer.fit(data)
        wavelengths = analyzer.get_wavelengths()
    """

    def __init__(self, config: Config) -> None:
        """Initialize analyzer with configuration.

        Args:
            config: Configuration object with analysis parameters.
        """
        self._config = config

        # Setup compute device
        if config.device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif config.device == "mps" and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Internal state (populated during fit)
        self._data: Optional[SpectraData] = None
        self._dataset: Optional[Any] = None  # MaskedHyperspectralDataset
        self._model: Optional[Any] = None
        self._baseline_latent: Optional[Any] = None
        self._baseline_reconstruction: Optional[Dict[float, Any]] = None
        self._patch_coords: Optional[List[Tuple[int, int]]] = None
        self._influence_matrix: Optional[Dict[str, Any]] = None
        self._result: Optional[WavelengthResult] = None

        # Create output directory if specified
        if config.output_dir is not None:
            config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Analyzer initialized for '{config.sample_name}' "
            f"on device '{self._device}'"
        )

    @property
    def config(self) -> Config:
        """The configuration object."""
        return self._config

    @property
    def device(self) -> torch.device:
        """The PyTorch device used for computation."""
        return self._device

    @property
    def is_fitted(self) -> bool:
        """Whether the analyzer has been fitted to data."""
        return self._result is not None

    @property
    def result(self) -> Optional[WavelengthResult]:
        """The analysis result, or None if not yet fitted."""
        return self._result

    @property
    def influence_matrix(self) -> Optional[Dict[str, Any]]:
        """The computed influence matrix, or None if not yet fitted."""
        return self._influence_matrix

    def fit(self, data: SpectraData) -> Analyzer:
        """Run the full wavelength selection analysis pipeline.

        This method executes the complete analysis workflow:
        1. Load or use existing autoencoder model
        2. Compute baseline latent representations
        3. Perform latent space perturbations
        4. Calculate wavelength sensitivity/influence scores
        5. Select top wavelength combinations

        Args:
            data: The hyperspectral data to analyze.

        Returns:
            self, for method chaining.

        Raises:
            NotImplementedError: Implemented in Phase 04-02/04-03.
        """
        raise NotImplementedError("Implemented in 04-02/04-03")

    def transform(self, data: SpectraData) -> SpectraData:
        """Apply fitted selection to extract only selected wavelengths.

        Uses the wavelength selection from a previous fit() call to
        extract a reduced dataset containing only the selected bands.

        Args:
            data: The hyperspectral data to transform.

        Returns:
            A new SpectraData object with only the selected wavelengths.

        Raises:
            RuntimeError: If the analyzer has not been fitted.
            NotImplementedError: Implemented in Phase 04-03.
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer must be fitted before transform()")
        raise NotImplementedError("Implemented in 04-03")

    def fit_transform(self, data: SpectraData) -> SpectraData:
        """Convenience method: fit the analyzer and transform in one step.

        Equivalent to calling fit(data).transform(data).

        Args:
            data: The hyperspectral data to analyze and transform.

        Returns:
            A new SpectraData object with only the selected wavelengths.
        """
        return self.fit(data).transform(data)

    def get_wavelengths(self) -> List[WavelengthBand]:
        """Return the list of selected wavelength combinations.

        Returns the selected wavelength bands from the analysis, ordered
        by their importance/influence score (highest first).

        Returns:
            List of WavelengthBand objects representing the selection.

        Raises:
            RuntimeError: If the analyzer has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Analyzer must be fitted before get_wavelengths(). "
                "Call fit(data) first."
            )
        return self._result.selected_bands

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """Save analysis results to disk.

        Saves the wavelength selection results to JSON format, and
        optionally saves TIFF layers for visualization.

        Args:
            output_dir: Directory to save results. If None, uses the
                output_dir from config.

        Returns:
            Path to the saved results directory.

        Raises:
            RuntimeError: If the analyzer has not been fitted.
            NotImplementedError: Implemented in Phase 04-04.
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer must be fitted before save_results()")
        raise NotImplementedError("Implemented in 04-04")

    def __repr__(self) -> str:
        """Return a readable string representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_bands = self._result.n_bands if self.is_fitted else "-"
        return (
            f"Analyzer("
            f"sample='{self._config.sample_name}', "
            f"device='{self._device}', "
            f"status={status}, "
            f"n_bands={n_bands})"
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _load_data(self, data: SpectraData) -> None:
        """Load and prepare data for analysis.

        Converts SpectraData into the format expected by MaskedHyperspectralDataset,
        which is used by the autoencoder model.

        Args:
            data: The hyperspectral SpectraData object.
        """
        # Store reference to original data
        self._data = data

        logger.info(
            f"Loading data: {data.n_excitations} excitations, "
            f"spatial shape {data.spatial_shape}, "
            f"mask {'present' if data.mask is not None else 'absent'}"
        )

        # TODO: Phase 8 cleanup - remove sys.path manipulation
        # The scripts.models package should be properly installed
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.models import MaskedHyperspectralDataset

        # Convert SpectraData to format expected by MaskedHyperspectralDataset
        # Expected format: {'data': {ex_str: {'cube': array, 'wavelengths': list}},
        #                   'excitation_wavelengths': [...]}
        data_dict: Dict[str, Any] = {
            "data": {},
            "excitation_wavelengths": data.excitation_wavelengths,
        }

        for ex_nm in data.excitation_wavelengths:
            ex_data = data.get_excitation(ex_nm)
            ex_str = str(ex_nm)
            data_dict["data"][ex_str] = {
                "cube": ex_data.cube,
                "wavelengths": ex_data.emission_wavelengths,
            }

        # Extract mask from SpectraData
        mask = data.mask

        # Create dataset with normalization
        self._dataset = MaskedHyperspectralDataset(
            data_dict=data_dict,
            mask=mask,
            normalize=True,
        )

        # Store spatial dimensions for later use
        self._spatial_shape = self._dataset.get_spatial_dimensions()

        logger.info(
            f"Dataset created: {len(self._dataset.excitation_wavelengths)} excitations, "
            f"spatial {self._spatial_shape[0]}x{self._spatial_shape[1]}"
        )
