"""Data loading utilities for raw hyperspectral files.

This module provides the DataLoader class for loading raw .im3 hyperspectral
files by wrapping the existing HyperspectralDataLoader implementation.

Example:
    from spectral_select import DataLoader

    loader = DataLoader("Data/Raw/Sample1")
    data = loader.load()

    # Or use SpectraData.from_raw() for direct conversion:
    from spectral_select import SpectraData
    spectra = SpectraData.from_raw("Data/Raw/Sample1")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DataLoadingError(Exception):
    """Exception raised when data loading fails.

    Provides detailed error messages for troubleshooting data loading issues
    including missing files, format errors, or ImageJ initialization failures.

    Attributes:
        message: Human-readable error description.
        path: Path that was being loaded when error occurred.
        cause: Original exception that caused this error (if any).

    Example:
        try:
            loader.load()
        except DataLoadingError as e:
            print(f"Failed to load {e.path}: {e}")
    """

    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize DataLoadingError.

        Args:
            message: Human-readable error description.
            path: Path that was being loaded when error occurred.
            cause: Original exception that caused this error.
        """
        self.message = message
        self.path = Path(path) if path else None
        self.cause = cause

        full_message = message
        if path:
            full_message = f"{message} (path: {path})"
        if cause:
            full_message = f"{full_message} - Caused by: {cause}"

        super().__init__(full_message)


class DataLoader:
    """Wrapper for loading raw .im3 hyperspectral data files.

    Provides a clean interface for loading raw hyperspectral data by wrapping
    the existing HyperspectralDataLoader. Supports lazy ImageJ initialization
    to avoid startup delays when not needed.

    Attributes:
        data_path: Path to directory containing .im3 files.
        metadata_path: Optional path to Excel file with exposure metadata.
        cutoff_offset: Nanometer offset for spectral cutoff (default 30).
        verbose: Whether to log loading progress.

    Example:
        # Basic usage
        loader = DataLoader("Data/Raw/Lichens_2")
        data_dict = loader.load()

        # With metadata
        loader = DataLoader(
            "Data/Raw/Lichens_2",
            metadata_path="Data/Raw/Lichens_2/metadata.xlsx",
            cutoff_offset=20,
        )
        data_dict = loader.load()
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        cutoff_offset: int = 30,
        verbose: bool = True,
    ):
        """Initialize the DataLoader.

        Args:
            data_path: Path to directory containing .im3 files.
            metadata_path: Optional path to Excel file with exposure metadata.
            cutoff_offset: Nanometer offset for Rayleigh/second-order cutoff.
            verbose: Whether to log loading progress.
        """
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.cutoff_offset = cutoff_offset
        self.verbose = verbose

        # Validate data_path exists at initialization time
        if not self.data_path.exists():
            raise DataLoadingError(
                f"Data path does not exist: {self.data_path}",
                path=self.data_path,
            )

        # Lazy-loaded internal loader
        self._loader: Optional[Any] = None
        self._imagej_available: Optional[bool] = None

    @property
    def imagej_available(self) -> bool:
        """Check if ImageJ/pyimagej is available without initializing it.

        Returns:
            True if pyimagej can be imported, False otherwise.
        """
        if self._imagej_available is None:
            try:
                import imagej  # noqa: F401

                self._imagej_available = True
            except ImportError:
                self._imagej_available = False
                logger.warning(
                    "pyimagej not installed - .im3 file loading requires ImageJ.\n"
                    "Install with: pip install pyimagej\n"
                    "Note: First run will download ImageJ/Fiji which takes several minutes."
                )
        return self._imagej_available

    def _get_loader(self) -> Any:
        """Get or create the internal HyperspectralDataLoader.

        Returns:
            HyperspectralDataLoader instance.

        Raises:
            DataLoadingError: If loader cannot be created.
        """
        if self._loader is None:
            try:
                from mehsi_preprocessor.io.hyperspectral_loader import (
                    HyperspectralDataLoader,
                )

                self._loader = HyperspectralDataLoader(
                    data_path=str(self.data_path),
                    metadata_path=str(self.metadata_path) if self.metadata_path else None,
                    cutoff_offset=self.cutoff_offset,
                    use_fiji=self.imagej_available,
                    verbose=self.verbose,
                )
            except ImportError:
                raise DataLoadingError(
                    "Raw .im3 file loading requires mehsi_preprocessor.\n"
                    "For preprocessed data, use: SpectraData.from_raw_dict() or SpectraData constructor",
                    path=self.data_path,
                )
            except Exception as e:
                raise DataLoadingError(
                    f"Failed to create HyperspectralDataLoader: {e}",
                    path=self.data_path,
                    cause=e,
                )

        return self._loader

    def load(
        self,
        apply_cutoff: bool = True,
        pattern: str = "*.im3",
    ) -> Dict[str, Any]:
        """Load hyperspectral data from .im3 files.

        Loads all .im3 files matching the pattern from the data directory,
        applies spectral cutoff to remove artifacts, and returns the data
        in a format compatible with SpectraData.

        Args:
            apply_cutoff: Whether to apply Rayleigh/second-order cutoff.
            pattern: File pattern for hyperspectral data files.

        Returns:
            Dictionary with structure:
            {
                "data": {
                    str(excitation_nm): {
                        "cube": np.ndarray,  # [H, W, bands]
                        "wavelengths": List[float],  # emission wavelengths
                        "excitation": float,  # excitation nm
                    },
                    ...
                },
                "excitation_wavelengths": List[float],
                "metadata": Dict[str, Any],
            }

        Raises:
            DataLoadingError: If loading fails.

        Example:
            loader = DataLoader("Data/Raw/Sample")
            data = loader.load()
            for ex_nm, ex_data in data["data"].items():
                print(f"Ex {ex_nm}: cube shape {ex_data['cube'].shape}")
        """
        if not self.data_path.exists():
            raise DataLoadingError(
                f"Data path does not exist: {self.data_path}",
                path=self.data_path,
            )

        if self.verbose:
            logger.info(f"Loading data from {self.data_path}")

        loader = self._get_loader()

        try:
            # Call the internal loader
            loader.load_data(apply_cutoff=apply_cutoff, pattern=pattern)
        except FileNotFoundError as e:
            # List directory contents to help debug
            try:
                contents = [f.name for f in self.data_path.iterdir()]
                if len(contents) > 10:
                    contents_str = str(contents[:10]) + f" ... ({len(contents)} total)"
                else:
                    contents_str = str(contents) if contents else "(empty)"
            except Exception:
                contents_str = "(unable to list)"

            raise DataLoadingError(
                f"No .im3 files found in {self.data_path}\n\n"
                f"Directory contents: {contents_str}\n\n"
                f"Expected: Directory containing .im3 hyperspectral image files.\n"
                f"Hint: Check that data_path points to the raw data directory, "
                f"not the processed output.",
                path=self.data_path,
                cause=e,
            )
        except NotImplementedError as e:
            # Direct loading not implemented - need ImageJ
            if not self.imagej_available:
                raise DataLoadingError(
                    "Cannot load .im3 files: pyimagej is not installed.\n\n"
                    "To install pyimagej:\n"
                    "  pip install pyimagej\n\n"
                    "Note: First run will download ImageJ/Fiji (~500MB) which takes "
                    "several minutes. Subsequent runs use the cached installation.",
                    path=self.data_path,
                    cause=e,
                )
            raise DataLoadingError(
                f"Direct .im3 loading failed: {e}\n\n"
                f"ImageJ is installed but direct loading is not supported.\n"
                f"This may indicate an issue with the ImageJ initialization.",
                path=self.data_path,
                cause=e,
            )
        except Exception as e:
            raise DataLoadingError(
                f"Error loading hyperspectral data: {e}",
                path=self.data_path,
                cause=e,
            )

        # Convert to SpectraData-compatible format
        return self._convert_to_spectra_format(loader)

    def _convert_to_spectra_format(self, loader: Any) -> Dict[str, Any]:
        """Convert HyperspectralDataLoader output to SpectraData format.

        Args:
            loader: HyperspectralDataLoader instance with loaded data.

        Returns:
            Dictionary in format expected by SpectraData.from_dict().
        """
        data_dict: Dict[str, Any] = {}

        for ex_str, ex_data in loader.data.items():
            ex_nm = float(ex_str)
            data_dict[ex_str] = {
                "cube": ex_data["cube"],
                "wavelengths": ex_data["wavelengths"],
                "excitation": ex_nm,
            }

        return {
            "data": data_dict,
            "excitation_wavelengths": [float(e) for e in loader.excitation_wavelengths],
            "metadata": loader.metadata,
        }

    def get_excitation_wavelengths(self) -> List[float]:
        """Get list of available excitation wavelengths.

        Must call load() first.

        Returns:
            Sorted list of excitation wavelengths.

        Raises:
            DataLoadingError: If data hasn't been loaded yet.
        """
        loader = self._get_loader()
        if not loader.excitation_wavelengths:
            raise DataLoadingError(
                "No data loaded. Call load() first.",
                path=self.data_path,
            )
        return sorted(loader.excitation_wavelengths)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about loaded data.

        Must call load() first.

        Returns:
            Dictionary with data summary.

        Raises:
            DataLoadingError: If data hasn't been loaded yet.
        """
        loader = self._get_loader()
        if not loader.data:
            raise DataLoadingError(
                "No data loaded. Call load() first.",
                path=self.data_path,
            )
        return loader.get_summary()
