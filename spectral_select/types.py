"""Data types and type aliases for spectral analysis.

This module provides typed data classes for hyperspectral data handling:
- LoadingOptions: Configuration for data preprocessing
- ExcitationData: Per-excitation wavelength spectral cube
- SpectraData: Multi-excitation hyperspectral data container
- GroundTruth: Ground truth labels for clustering validation
- ValidationMetrics: Clustering evaluation metrics

Example:
    # Loading preprocessed data
    from spectral_select.types import SpectraData, LoadingOptions

    data = SpectraData.from_pickle("processed_data.pkl")
    print(f"Excitations: {data.excitation_wavelengths}")

    # Custom loading options
    opts = LoadingOptions(
        apply_rayleigh_cutoff=True,
        normalize_exposure=True,
        downscale_factor=2,
    )

    # Validation
    from spectral_select.types import GroundTruth, ValidationMetrics

    gt = GroundTruth.from_array(labels_2d)
    print(f"Classes: {gt.n_classes}")
"""

from __future__ import annotations

import pickle
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import pandas as pd

# Type aliases for clarity
SpectraArray: TypeAlias = "np.ndarray"


@dataclass
class LoadingOptions:
    """Configuration options for loading and preprocessing hyperspectral data.

    Controls preprocessing steps applied when loading raw .im3 files or
    when re-processing existing data.

    Attributes:
        cutoff_offset: Nanometer offset for Rayleigh/second-order cutoff.
        apply_rayleigh_cutoff: Whether to remove laser line artifacts.
        apply_second_order_cutoff: Whether to remove second-order diffraction.
        normalize_exposure: Whether to normalize by acquisition/exposure time.
        normalize_laser_power: Whether to normalize by excitation lamp power.
        roi: Region of interest as (row_min, row_max, col_min, col_max).
        downscale_factor: Spatial downscaling factor (1 = no downscaling).

    Example:
        opts = LoadingOptions(
            cutoff_offset=30,
            apply_rayleigh_cutoff=True,
            downscale_factor=2,
        )
    """

    cutoff_offset: int = 30
    apply_rayleigh_cutoff: bool = True
    apply_second_order_cutoff: bool = True
    normalize_exposure: bool = True
    normalize_laser_power: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None
    downscale_factor: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.cutoff_offset < 0:
            raise ValueError(
                f"cutoff_offset must be >= 0, got {self.cutoff_offset}"
            )

        if self.downscale_factor < 1:
            raise ValueError(
                f"downscale_factor must be >= 1, got {self.downscale_factor}"
            )

        if self.roi is not None:
            row_min, row_max, col_min, col_max = self.roi
            if row_min >= row_max:
                raise ValueError(
                    f"roi row_min ({row_min}) must be < row_max ({row_max})"
                )
            if col_min >= col_max:
                raise ValueError(
                    f"roi col_min ({col_min}) must be < col_max ({col_max})"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with all configuration values.
        """
        return {
            "cutoff_offset": self.cutoff_offset,
            "apply_rayleigh_cutoff": self.apply_rayleigh_cutoff,
            "apply_second_order_cutoff": self.apply_second_order_cutoff,
            "normalize_exposure": self.normalize_exposure,
            "normalize_laser_power": self.normalize_laser_power,
            "roi": self.roi,
            "downscale_factor": self.downscale_factor,
        }


@dataclass
class ExcitationData:
    """Data for a single excitation wavelength.

    Contains the 3D spectral cube and associated metadata for one
    excitation wavelength in a multi-excitation hyperspectral dataset.

    Attributes:
        excitation_nm: The excitation wavelength in nanometers.
        cube: 3D array of shape [height, width, n_emission_bands].
        emission_wavelengths: Emission wavelength values for each band.
        exposure_time: Acquisition time in seconds (if available).
        laser_power: Excitation lamp power (if available).

    Example:
        ed = ExcitationData(
            excitation_nm=365.0,
            cube=np.zeros((100, 100, 50)),
            emission_wavelengths=list(range(400, 700, 6)),
        )
        print(f"Shape: {ed.shape}, n_bands: {ed.n_bands}")
    """

    excitation_nm: float
    cube: np.ndarray
    emission_wavelengths: List[float]
    exposure_time: Optional[float] = None
    laser_power: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        if self.cube.ndim != 3:
            raise ValueError(
                f"cube must be 3D, got shape {self.cube.shape}"
            )

        if len(self.emission_wavelengths) != self.cube.shape[2]:
            raise ValueError(
                f"len(emission_wavelengths) ({len(self.emission_wavelengths)}) "
                f"must equal cube.shape[2] ({self.cube.shape[2]})"
            )

        if self.excitation_nm <= 0:
            raise ValueError(
                f"excitation_nm must be positive, got {self.excitation_nm}"
            )

    @property
    def height(self) -> int:
        """Height of the spatial dimension."""
        return self.cube.shape[0]

    @property
    def width(self) -> int:
        """Width of the spatial dimension."""
        return self.cube.shape[1]

    @property
    def n_bands(self) -> int:
        """Number of emission wavelength bands."""
        return self.cube.shape[2]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the spectral cube (height, width, n_bands)."""
        return self.cube.shape

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Note:
            The cube array is converted to a list for JSON compatibility.
            For large arrays, consider using pickle or numpy-specific formats.

        Returns:
            Dictionary with all data (cube as nested list).
        """
        return {
            "excitation_nm": self.excitation_nm,
            "cube": self.cube.tolist(),
            "emission_wavelengths": self.emission_wavelengths,
            "exposure_time": self.exposure_time,
            "laser_power": self.laser_power,
        }


@dataclass
class SpectraData:
    """Container for multi-excitation hyperspectral data.

    The main data structure for holding hyperspectral data from multiple
    excitation wavelengths, along with masks and metadata.

    Attributes:
        excitations: Mapping from excitation wavelength to ExcitationData.
        mask: Binary mask where 1=valid pixel, 0=masked.
        sample_name: Identifier for the sample.
        loading_options: Preprocessing settings used during loading.
        metadata: Arbitrary metadata dictionary.

    Example:
        # Load from pickle
        data = SpectraData.from_pickle("processed_data.pkl")

        # Access specific excitation
        ex_365 = data.get_excitation(365.0)
        print(f"365nm cube shape: {ex_365.shape}")
    """

    excitations: Dict[float, ExcitationData]
    mask: Optional[np.ndarray] = None
    sample_name: str = "sample"
    loading_options: Optional[LoadingOptions] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        if not self.excitations:
            raise ValueError("excitations cannot be empty - at least one excitation required")

        # Get reference spatial shape from first excitation
        first_ex = next(iter(self.excitations.values()))
        ref_height, ref_width = first_ex.height, first_ex.width

        # Verify all excitations have same spatial dimensions
        for ex_nm, ex_data in self.excitations.items():
            if ex_data.height != ref_height or ex_data.width != ref_width:
                raise ValueError(
                    f"Excitation {ex_nm}nm has shape ({ex_data.height}, {ex_data.width}), "
                    f"but expected ({ref_height}, {ref_width}) based on first excitation"
                )

        # Verify mask shape if provided
        if self.mask is not None:
            if self.mask.shape != (ref_height, ref_width):
                raise ValueError(
                    f"mask shape {self.mask.shape} doesn't match "
                    f"spatial dimensions ({ref_height}, {ref_width})"
                )

    @property
    def excitation_wavelengths(self) -> List[float]:
        """Sorted list of excitation wavelengths."""
        return sorted(self.excitations.keys())

    @property
    def n_excitations(self) -> int:
        """Number of excitation wavelengths."""
        return len(self.excitations)

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        """Spatial dimensions (height, width) from first excitation."""
        if not self.excitations:
            return (0, 0)
        first_ex = next(iter(self.excitations.values()))
        return (first_ex.height, first_ex.width)

    def get_excitation(self, ex_nm: float) -> ExcitationData:
        """Get data for a specific excitation wavelength.

        Args:
            ex_nm: The excitation wavelength in nanometers.

        Returns:
            ExcitationData for the specified wavelength.

        Raises:
            KeyError: If the excitation wavelength is not found.
        """
        if ex_nm not in self.excitations:
            available = self.excitation_wavelengths
            raise KeyError(
                f"Excitation {ex_nm}nm not found. Available: {available}"
            )
        return self.excitations[ex_nm]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Note:
            Large arrays (cube, mask) are excluded to keep the dict lightweight.
            Use to_pickle() for full serialization with arrays.

        Returns:
            Dictionary with metadata (arrays excluded).
        """
        return {
            "sample_name": self.sample_name,
            "excitation_wavelengths": self.excitation_wavelengths,
            "n_excitations": self.n_excitations,
            "spatial_shape": self.spatial_shape,
            "has_mask": self.mask is not None,
            "loading_options": self.loading_options.to_dict() if self.loading_options else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectraData":
        """Create from dictionary (for deserialization).

        Note:
            This expects a dictionary with the same structure as produced by
            to_dict(). For loading with full array data, use from_pickle().

        Args:
            data: Dictionary containing SpectraData fields.

        Returns:
            A new SpectraData instance.
        """
        # Handle excitations if present as ExcitationData dicts
        excitations = {}
        if "excitations" in data:
            for ex_nm, ex_dict in data["excitations"].items():
                if isinstance(ex_dict, dict):
                    # Reconstruct numpy array from list
                    cube = np.array(ex_dict["cube"])
                    excitations[float(ex_nm)] = ExcitationData(
                        excitation_nm=ex_dict["excitation_nm"],
                        cube=cube,
                        emission_wavelengths=ex_dict["emission_wavelengths"],
                        exposure_time=ex_dict.get("exposure_time"),
                        laser_power=ex_dict.get("laser_power"),
                    )
                elif isinstance(ex_dict, ExcitationData):
                    excitations[float(ex_nm)] = ex_dict

        # Handle mask
        mask = data.get("mask")
        if mask is not None and not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        # Handle loading_options
        loading_options = data.get("loading_options")
        if loading_options is not None and isinstance(loading_options, dict):
            loading_options = LoadingOptions(**loading_options)

        return cls(
            excitations=excitations,
            mask=mask,
            sample_name=data.get("sample_name", "sample"),
            loading_options=loading_options,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> "SpectraData":
        """Load from processed pickle file format.

        Supports the existing pickle format used in the codebase with
        'data' and 'excitation_wavelengths' keys.

        Args:
            path: Path to the pickle file.

        Returns:
            A new SpectraData instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the pickle format is not recognized.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")

        with open(path, "rb") as f:
            pkl_data = pickle.load(f)

        # Handle existing pickle format: {'data': {ex_nm: cube}, 'excitation_wavelengths': [...]}
        if isinstance(pkl_data, dict):
            # Check for existing format with 'data' and 'excitation_wavelengths' keys
            if "data" in pkl_data and "excitation_wavelengths" in pkl_data:
                raw_data = pkl_data["data"]
                excitation_wavelengths = pkl_data["excitation_wavelengths"]

                # Build ExcitationData objects
                excitations = {}
                for ex_nm, ex_data in raw_data.items():
                    ex_nm_float = float(ex_nm)

                    # Handle nested dict format: {'cube': array, 'wavelengths': [...]}
                    if isinstance(ex_data, dict) and "cube" in ex_data:
                        cube = ex_data["cube"]
                        emission_wls = ex_data.get("wavelengths", [])
                        if not emission_wls:
                            # Generate placeholder if not provided
                            n_bands = cube.shape[2] if cube.ndim == 3 else 0
                            emission_wls = list(range(n_bands))
                    else:
                        # Direct array format (legacy)
                        cube = ex_data
                        n_bands = cube.shape[2] if hasattr(cube, 'ndim') and cube.ndim == 3 else 0
                        emission_wls = list(range(n_bands))

                    excitations[ex_nm_float] = ExcitationData(
                        excitation_nm=ex_nm_float,
                        cube=cube,
                        emission_wavelengths=emission_wls,
                        exposure_time=pkl_data.get("exposure_times", {}).get(ex_nm),
                        laser_power=pkl_data.get("laser_powers", {}).get(ex_nm),
                    )

                # Extract optional fields
                mask = pkl_data.get("mask")
                metadata = {k: v for k, v in pkl_data.items()
                           if k not in ("data", "excitation_wavelengths", "mask",
                                       "exposure_times", "laser_powers")}

                return cls(
                    excitations=excitations,
                    mask=mask,
                    sample_name=path.stem,
                    metadata=metadata,
                )

            # Check for SpectraData serialized format
            if "excitations" in pkl_data:
                return cls.from_dict(pkl_data)

        raise ValueError(
            f"Unrecognized pickle format. Expected dict with 'data' and "
            f"'excitation_wavelengths' keys, or SpectraData format."
        )

    @classmethod
    def from_raw(
        cls,
        data_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        mask: Optional[np.ndarray] = None,
        sample_name: Optional[str] = None,
        loading_options: Optional[LoadingOptions] = None,
        **loader_kwargs,
    ) -> "SpectraData":
        """Load from raw .im3 hyperspectral files.

        Convenience factory method that uses DataLoader internally to load
        raw .im3 files and convert them to a SpectraData instance.

        Requires pyimagej to be installed for .im3 file loading.

        Args:
            data_path: Path to directory containing .im3 files.
            metadata_path: Optional path to Excel file with exposure metadata.
            mask: Optional binary mask (1=valid, 0=masked pixels).
            sample_name: Identifier for the sample (defaults to directory name).
            loading_options: Preprocessing options (cutoff, normalization, etc.).
            **loader_kwargs: Additional kwargs passed to DataLoader.load().

        Returns:
            A new SpectraData instance with loaded data.

        Raises:
            DataLoadingError: If loading fails (missing files, ImageJ issues, etc.).

        Example:
            # Basic usage
            data = SpectraData.from_raw("Data/Raw/Lichens_2")

            # With options
            opts = LoadingOptions(cutoff_offset=20)
            data = SpectraData.from_raw(
                "Data/Raw/Lichens_2",
                loading_options=opts,
                mask=my_mask,
            )
        """
        # Import lazily to avoid circular imports
        from .loader import DataLoader, DataLoadingError

        data_path = Path(data_path)

        # Derive sample name from directory if not provided
        if sample_name is None:
            sample_name = data_path.name

        # Get cutoff_offset from loading_options or use default
        cutoff_offset = 30
        if loading_options is not None:
            cutoff_offset = loading_options.cutoff_offset

        try:
            loader = DataLoader(
                data_path=data_path,
                metadata_path=metadata_path,
                cutoff_offset=cutoff_offset,
            )

            # Apply cutoff based on loading_options
            apply_cutoff = True
            if loading_options is not None:
                apply_cutoff = (
                    loading_options.apply_rayleigh_cutoff
                    or loading_options.apply_second_order_cutoff
                )

            raw_data = loader.load(apply_cutoff=apply_cutoff, **loader_kwargs)

        except Exception as e:
            # Re-raise with context about the data path
            if isinstance(e, DataLoadingError):
                raise
            raise DataLoadingError(
                f"Failed to load raw data from {data_path}: {e}",
                path=data_path,
                cause=e,
            )

        # Build ExcitationData objects from loaded cubes
        excitations: Dict[float, ExcitationData] = {}

        for ex_str, ex_data in raw_data["data"].items():
            ex_nm = float(ex_str)
            excitations[ex_nm] = ExcitationData(
                excitation_nm=ex_nm,
                cube=ex_data["cube"],
                emission_wavelengths=ex_data["wavelengths"],
                exposure_time=ex_data.get("exposure_time"),
                laser_power=ex_data.get("laser_power"),
            )

        if not excitations:
            raise DataLoadingError(
                "No excitation wavelengths loaded from raw data",
                path=data_path,
            )

        return cls(
            excitations=excitations,
            mask=mask,
            sample_name=sample_name,
            loading_options=loading_options,
            metadata=raw_data.get("metadata", {}),
        )


# ============================================================================
# Output/Result Data Types
# ============================================================================


@dataclass
class WavelengthBand:
    """A single selected wavelength combination from analysis.

    Represents one excitation-emission pair selected during wavelength
    selection analysis, ranked by its influence/importance score.

    Attributes:
        rank: 1-indexed rank in selection order (1 = most important).
        excitation_nm: Excitation wavelength in nanometers.
        emission_nm: Emission wavelength in nanometers.
        emission_band_index: Index into the emission band array.
        influence_score: Importance/sensitivity score from analysis.

    Example:
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=8,
            influence_score=0.85,
        )
        print(band)  # WavelengthBand(rank=1, ex=365.0nm, em=500.0nm, score=0.85)
    """

    rank: int
    excitation_nm: float
    emission_nm: float
    emission_band_index: int
    influence_score: float

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.excitation_nm <= 0:
            raise ValueError(
                f"excitation_nm must be positive, got {self.excitation_nm}"
            )
        if self.emission_nm <= 0:
            raise ValueError(
                f"emission_nm must be positive, got {self.emission_nm}"
            )
        if self.emission_band_index < 0:
            raise ValueError(
                f"emission_band_index must be >= 0, got {self.emission_band_index}"
            )

    def __repr__(self) -> str:
        """Readable representation."""
        # Use adaptive formatting: scientific for very small values, fixed for larger
        if abs(self.influence_score) < 0.001 and self.influence_score != 0:
            score_str = f"{self.influence_score:.2e}"
        else:
            score_str = f"{self.influence_score:.4f}"
        return (
            f"WavelengthBand(rank={self.rank}, ex={self.excitation_nm}nm, "
            f"em={self.emission_nm}nm, score={score_str})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all field values.
        """
        return {
            "rank": self.rank,
            "excitation_nm": self.excitation_nm,
            "emission_nm": self.emission_nm,
            "emission_band_index": self.emission_band_index,
            "influence_score": self.influence_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WavelengthBand":
        """Create from dictionary.

        Args:
            data: Dictionary with WavelengthBand fields.

        Returns:
            A new WavelengthBand instance.
        """
        return cls(
            rank=data["rank"],
            excitation_nm=data["excitation_nm"],
            emission_nm=data["emission_nm"],
            emission_band_index=data["emission_band_index"],
            influence_score=data["influence_score"],
        )


@dataclass
class AnalysisMetrics:
    """Performance metrics for wavelength selection analysis.

    Summarizes the selection results with counts and score statistics.

    Attributes:
        total_bands_available: Total wavelength combinations in input.
        bands_selected: Number of bands actually selected.
        compression_ratio: Ratio of total_bands / bands_selected.
        max_influence_score: Highest influence score among selected.
        min_influence_score: Lowest influence score among selected.
        mean_influence_score: Average influence score of selected bands.

    Example:
        metrics = AnalysisMetrics.from_bands(selected_bands, total_available=1000)
        print(f"Compression: {metrics.compression_ratio:.1f}x")
    """

    total_bands_available: int
    bands_selected: int
    compression_ratio: float
    max_influence_score: float
    min_influence_score: float
    mean_influence_score: float

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if self.total_bands_available < 1:
            raise ValueError(
                f"total_bands_available must be >= 1, got {self.total_bands_available}"
            )
        if self.bands_selected < 1:
            raise ValueError(
                f"bands_selected must be >= 1, got {self.bands_selected}"
            )
        if self.compression_ratio < 1.0:
            raise ValueError(
                f"compression_ratio must be >= 1.0, got {self.compression_ratio}"
            )
        # Score ordering: min <= mean <= max
        if not (self.min_influence_score <= self.mean_influence_score <= self.max_influence_score):
            raise ValueError(
                f"Score ordering violated: min ({self.min_influence_score}) <= "
                f"mean ({self.mean_influence_score}) <= max ({self.max_influence_score})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all metric values.
        """
        return {
            "total_bands_available": self.total_bands_available,
            "bands_selected": self.bands_selected,
            "compression_ratio": self.compression_ratio,
            "max_influence_score": self.max_influence_score,
            "min_influence_score": self.min_influence_score,
            "mean_influence_score": self.mean_influence_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary with AnalysisMetrics fields.

        Returns:
            A new AnalysisMetrics instance.
        """
        return cls(
            total_bands_available=data["total_bands_available"],
            bands_selected=data["bands_selected"],
            compression_ratio=data["compression_ratio"],
            max_influence_score=data["max_influence_score"],
            min_influence_score=data["min_influence_score"],
            mean_influence_score=data["mean_influence_score"],
        )

    @classmethod
    def from_bands(
        cls,
        bands: List["WavelengthBand"],
        total_available: int,
    ) -> "AnalysisMetrics":
        """Compute metrics from a list of selected bands.

        Convenience factory method to calculate all metrics from
        the actual selection results.

        Args:
            bands: List of selected WavelengthBand objects.
            total_available: Total number of wavelength combinations available.

        Returns:
            A new AnalysisMetrics instance with computed values.

        Raises:
            ValueError: If bands list is empty.
        """
        if not bands:
            raise ValueError("Cannot compute metrics from empty bands list")

        scores = [b.influence_score for b in bands]
        n_selected = len(bands)

        return cls(
            total_bands_available=total_available,
            bands_selected=n_selected,
            compression_ratio=total_available / n_selected,
            max_influence_score=max(scores),
            min_influence_score=min(scores),
            mean_influence_score=sum(scores) / n_selected,
        )


@dataclass
class WavelengthResult:
    """Complete output container for wavelength selection analysis.

    The main result type capturing all outputs from a wavelength selection
    analysis run, including selected bands, metrics, and configuration.

    Attributes:
        sample_name: Identifier for the analyzed sample.
        selected_bands: Ordered list of selected wavelength combinations.
        metrics: Performance metrics for the selection.
        timestamp: When the analysis was run.
        config_snapshot: Config.to_dict() at time of analysis (for reproducibility).
        method_summary: Methods used (dimension_selection, perturbation, etc.).

    Example:
        result = WavelengthResult(
            sample_name="Lichens_2",
            selected_bands=bands,
            metrics=AnalysisMetrics.from_bands(bands, 1000),
        )
        result.to_json("results/analysis_output.json")
    """

    sample_name: str
    selected_bands: List[WavelengthBand]
    metrics: AnalysisMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    config_snapshot: Optional[Dict[str, Any]] = None
    method_summary: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize data after initialization."""
        if not self.selected_bands:
            raise ValueError("selected_bands cannot be empty")

        # Sort by rank if not already sorted
        self.selected_bands = sorted(self.selected_bands, key=lambda b: b.rank)

        # Validate sequential ranks starting from 1
        expected_ranks = list(range(1, len(self.selected_bands) + 1))
        actual_ranks = [b.rank for b in self.selected_bands]
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"Ranks must be sequential starting from 1. "
                f"Expected {expected_ranks}, got {actual_ranks}"
            )

    @property
    def n_bands(self) -> int:
        """Number of selected bands."""
        return len(self.selected_bands)

    @property
    def top_band(self) -> WavelengthBand:
        """The highest-ranked (first) selected band."""
        return self.selected_bands[0]

    @property
    def excitation_wavelengths(self) -> List[float]:
        """Unique excitation wavelengths in selected bands (sorted)."""
        return sorted(set(b.excitation_nm for b in self.selected_bands))

    def get_bands_for_excitation(self, ex_nm: float) -> List[WavelengthBand]:
        """Get all selected bands for a specific excitation wavelength.

        Args:
            ex_nm: The excitation wavelength in nanometers.

        Returns:
            List of WavelengthBand objects for that excitation.
        """
        return [b for b in self.selected_bands if b.excitation_nm == ex_nm]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all result data.
        """
        return {
            "sample_name": self.sample_name,
            "selected_bands": [b.to_dict() for b in self.selected_bands],
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "config_snapshot": self.config_snapshot,
            "method_summary": self.method_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WavelengthResult":
        """Create from dictionary.

        Args:
            data: Dictionary with WavelengthResult fields.

        Returns:
            A new WavelengthResult instance.
        """
        bands = [WavelengthBand.from_dict(b) for b in data["selected_bands"]]
        metrics = AnalysisMetrics.from_dict(data["metrics"])

        # Parse timestamp
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.now()

        return cls(
            sample_name=data["sample_name"],
            selected_bands=bands,
            metrics=metrics,
            timestamp=timestamp,
            config_snapshot=data.get("config_snapshot"),
            method_summary=data.get("method_summary", {}),
        )

    def to_json(self, path: Union[str, Path]) -> None:
        """Save result to JSON file.

        Args:
            path: Path to the output JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "WavelengthResult":
        """Load result from JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A new WavelengthResult instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_excel(self, path: Union[str, Path], include_metrics: bool = True) -> None:
        """Save result to Excel file with wavelength bands table.

        Creates an Excel file with a "Wavelengths" sheet containing the selected
        bands as a flat table. Optionally includes a "Metrics" sheet with analysis
        metrics.

        Args:
            path: Path to the output Excel file (.xlsx).
            include_metrics: If True, add a second sheet with analysis metrics.

        Example:
            result.to_excel("results/wavelengths.xlsx")
            result.to_excel("results/wavelengths.xlsx", include_metrics=False)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create wavelengths DataFrame from selected bands
        wavelengths_data = {
            "Rank": [b.rank for b in self.selected_bands],
            "Excitation_nm": [b.excitation_nm for b in self.selected_bands],
            "Emission_nm": [b.emission_nm for b in self.selected_bands],
            "Band_Index": [b.emission_band_index for b in self.selected_bands],
            "Score": [b.influence_score for b in self.selected_bands],
        }
        wavelengths_df = pd.DataFrame(wavelengths_data)

        # Write to Excel
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            wavelengths_df.to_excel(writer, sheet_name="Wavelengths", index=False)

            if include_metrics:
                # Create metrics DataFrame (single row)
                metrics_data = {
                    "Total_Bands": [self.metrics.total_bands_available],
                    "Bands_Selected": [self.metrics.bands_selected],
                    "Compression_Ratio": [self.metrics.compression_ratio],
                    "Max_Score": [self.metrics.max_influence_score],
                    "Min_Score": [self.metrics.min_influence_score],
                    "Mean_Score": [self.metrics.mean_influence_score],
                }
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name="Metrics", index=False)


# ============================================================================
# Ground Truth and Validation Data Types
# ============================================================================


@dataclass
class GroundTruth:
    """Ground truth labels for clustering validation.

    Container for 2D label arrays from annotated images, with support
    for background pixels (label -1) and color-to-class mappings.

    Attributes:
        labels: 2D integer array where -1 indicates background.
        color_mapping: Mapping from label integers to RGBA tuples.
        class_names: Optional human-readable names for each class.

    Example:
        gt = GroundTruth.from_array(labels_2d)
        print(f"Classes: {gt.n_classes}, valid pixels: {gt.valid_mask.sum()}")
    """

    labels: np.ndarray
    color_mapping: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    class_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate ground truth data after initialization."""
        if self.labels.ndim != 2:
            raise ValueError(
                f"labels must be 2D array, got shape {self.labels.shape}"
            )

        # Ensure labels are integer type
        if not np.issubdtype(self.labels.dtype, np.integer):
            self.labels = self.labels.astype(np.int32)

        # Validate class_names length if provided
        if self.class_names is not None:
            if len(self.class_names) != self.n_classes:
                raise ValueError(
                    f"class_names length ({len(self.class_names)}) must equal "
                    f"n_classes ({self.n_classes})"
                )

    @property
    def n_classes(self) -> int:
        """Number of ground truth classes (excluding background -1)."""
        unique_labels = np.unique(self.labels)
        return int(np.sum(unique_labels >= 0))

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask where True indicates non-background pixels."""
        return self.labels >= 0

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial shape (height, width) of the labels array."""
        return self.labels.shape

    @classmethod
    def from_array(
        cls,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> "GroundTruth":
        """Create GroundTruth from a simple label array.

        Convenience factory that auto-generates color mapping.

        Args:
            labels: 2D integer array with -1 for background.
            class_names: Optional list of class names.

        Returns:
            A new GroundTruth instance.
        """
        # Generate default color mapping
        unique_labels = np.unique(labels)
        color_mapping: Dict[int, Tuple[int, int, int, int]] = {
            -1: (0, 0, 0, 0)  # Background transparent
        }

        # Use distinct colors for each class
        for label in unique_labels:
            if label >= 0:
                # Generate colors using golden ratio for visual separation
                hue = (label * 137.5) % 360
                # Simple HSV to RGB conversion for distinct colors
                h = hue / 60
                x = int(255 * (1 - abs(h % 2 - 1)))
                i = int(h) % 6
                rgb_map = [
                    (255, x, 0), (x, 255, 0), (0, 255, x),
                    (0, x, 255), (x, 0, 255), (255, 0, x)
                ]
                r, g, b = rgb_map[i]
                color_mapping[int(label)] = (r, g, b, 255)

        return cls(
            labels=labels,
            color_mapping=color_mapping,
            class_names=class_names,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Note: Labels array is converted to list; use to_pickle for large data.

        Returns:
            Dictionary with all ground truth data.
        """
        return {
            "labels": self.labels.tolist(),
            "color_mapping": {
                str(k): list(v) for k, v in self.color_mapping.items()
            },
            "class_names": self.class_names,
            "n_classes": self.n_classes,
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruth":
        """Create from dictionary.

        Args:
            data: Dictionary with GroundTruth fields.

        Returns:
            A new GroundTruth instance.
        """
        labels = np.array(data["labels"], dtype=np.int32)
        color_mapping = {
            int(k): tuple(v) for k, v in data.get("color_mapping", {}).items()
        }

        return cls(
            labels=labels,
            color_mapping=color_mapping,
            class_names=data.get("class_names"),
        )


@dataclass
class ValidationMetrics:
    """Clustering evaluation metrics against ground truth.

    Contains comprehensive metrics for evaluating clustering results,
    including sklearn-compatible scores and per-class statistics.

    Attributes:
        adjusted_rand_score: Adjusted Rand Index (-1 to 1, 1 = perfect).
        normalized_mutual_info: Normalized Mutual Information (0 to 1).
        adjusted_mutual_info: Adjusted Mutual Information.
        fowlkes_mallows_score: Fowlkes-Mallows Index.
        v_measure: V-measure (harmonic mean of homogeneity and completeness).
        homogeneity: Homogeneity score (0 to 1).
        completeness: Completeness score (0 to 1).
        purity: Cluster purity (fraction of correctly assigned pixels).
        cluster_to_gt_mapping: Optimal mapping from cluster IDs to GT classes.
        confusion_matrix: Confusion matrix (rows=GT, cols=predicted).
        per_class_precision: Precision for each ground truth class.
        per_class_recall: Recall for each ground truth class.
        per_class_f1: F1 score for each ground truth class.
        n_ground_truth_classes: Number of classes in ground truth.
        n_predicted_clusters: Number of predicted clusters.

    Example:
        metrics = validator.metrics
        print(f"ARI: {metrics.adjusted_rand_score:.3f}")
        print(metrics.summary())
    """

    adjusted_rand_score: float
    normalized_mutual_info: float
    adjusted_mutual_info: float
    fowlkes_mallows_score: float
    v_measure: float
    homogeneity: float
    completeness: float
    purity: float
    cluster_to_gt_mapping: Dict[int, int]
    confusion_matrix: np.ndarray
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    per_class_f1: Dict[int, float]
    n_ground_truth_classes: int
    n_predicted_clusters: int

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        # Score range validation
        score_attrs = [
            ("adjusted_rand_score", -1.0, 1.0),
            ("normalized_mutual_info", 0.0, 1.0),
            ("v_measure", 0.0, 1.0),
            ("homogeneity", 0.0, 1.0),
            ("completeness", 0.0, 1.0),
            ("purity", 0.0, 1.0),
        ]
        for attr, min_val, max_val in score_attrs:
            val = getattr(self, attr)
            if not (min_val <= val <= max_val):
                raise ValueError(
                    f"{attr} must be in [{min_val}, {max_val}], got {val}"
                )

        # Ensure confusion_matrix is 2D numpy array
        if not isinstance(self.confusion_matrix, np.ndarray):
            self.confusion_matrix = np.array(self.confusion_matrix)
        if self.confusion_matrix.ndim != 2:
            raise ValueError(
                f"confusion_matrix must be 2D, got shape {self.confusion_matrix.shape}"
            )

    def summary(self) -> str:
        """Generate a formatted summary string of all metrics.

        Returns:
            Multi-line string with all metrics formatted for display.
        """
        lines = [
            "=" * 50,
            "Clustering Validation Metrics",
            "=" * 50,
            "",
            f"Clusters: {self.n_predicted_clusters}  |  "
            f"Ground Truth Classes: {self.n_ground_truth_classes}",
            "",
            "Global Metrics:",
            f"  Purity:             {self.purity:.4f}",
            f"  Adjusted Rand:      {self.adjusted_rand_score:.4f}",
            f"  NMI:                {self.normalized_mutual_info:.4f}",
            f"  AMI:                {self.adjusted_mutual_info:.4f}",
            f"  V-Measure:          {self.v_measure:.4f}",
            f"  Homogeneity:        {self.homogeneity:.4f}",
            f"  Completeness:       {self.completeness:.4f}",
            f"  Fowlkes-Mallows:    {self.fowlkes_mallows_score:.4f}",
            "",
            "Per-Class F1 Scores:",
        ]

        for cls_id in sorted(self.per_class_f1.keys()):
            f1 = self.per_class_f1[cls_id]
            prec = self.per_class_precision.get(cls_id, 0.0)
            rec = self.per_class_recall.get(cls_id, 0.0)
            lines.append(f"  Class {cls_id}: F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}")

        lines.extend(["", "Cluster → GT Mapping:"])
        for cluster_id, gt_id in sorted(self.cluster_to_gt_mapping.items()):
            lines.append(f"  Cluster {cluster_id} → Class {gt_id}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all metric values.
        """
        return {
            "adjusted_rand_score": self.adjusted_rand_score,
            "normalized_mutual_info": self.normalized_mutual_info,
            "adjusted_mutual_info": self.adjusted_mutual_info,
            "fowlkes_mallows_score": self.fowlkes_mallows_score,
            "v_measure": self.v_measure,
            "homogeneity": self.homogeneity,
            "completeness": self.completeness,
            "purity": self.purity,
            "cluster_to_gt_mapping": {
                str(k): v for k, v in self.cluster_to_gt_mapping.items()
            },
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_precision": {
                str(k): v for k, v in self.per_class_precision.items()
            },
            "per_class_recall": {
                str(k): v for k, v in self.per_class_recall.items()
            },
            "per_class_f1": {
                str(k): v for k, v in self.per_class_f1.items()
            },
            "n_ground_truth_classes": self.n_ground_truth_classes,
            "n_predicted_clusters": self.n_predicted_clusters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary with ValidationMetrics fields.

        Returns:
            A new ValidationMetrics instance.
        """
        return cls(
            adjusted_rand_score=data["adjusted_rand_score"],
            normalized_mutual_info=data["normalized_mutual_info"],
            adjusted_mutual_info=data["adjusted_mutual_info"],
            fowlkes_mallows_score=data["fowlkes_mallows_score"],
            v_measure=data["v_measure"],
            homogeneity=data["homogeneity"],
            completeness=data["completeness"],
            purity=data["purity"],
            cluster_to_gt_mapping={
                int(k): v for k, v in data["cluster_to_gt_mapping"].items()
            },
            confusion_matrix=np.array(data["confusion_matrix"]),
            per_class_precision={
                int(k): v for k, v in data["per_class_precision"].items()
            },
            per_class_recall={
                int(k): v for k, v in data["per_class_recall"].items()
            },
            per_class_f1={
                int(k): v for k, v in data["per_class_f1"].items()
            },
            n_ground_truth_classes=data["n_ground_truth_classes"],
            n_predicted_clusters=data["n_predicted_clusters"],
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ValidationMetrics":
        """Load ValidationMetrics from JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A new ValidationMetrics instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)
