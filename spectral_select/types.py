"""Data types and type aliases for spectral analysis.

This module provides typed data classes for hyperspectral data handling:
- LoadingOptions: Configuration for data preprocessing
- ExcitationData: Per-excitation wavelength spectral cube
- SpectraData: Multi-excitation hyperspectral data container

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
"""

from __future__ import annotations

import pickle
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np

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
            return  # Allow empty initialization

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
                for ex_nm, cube in raw_data.items():
                    ex_nm_float = float(ex_nm)
                    # For existing format, we don't have emission wavelengths per-cube
                    # Generate placeholder based on cube shape
                    n_bands = cube.shape[2] if cube.ndim == 3 else 0
                    # Estimate emission wavelengths (placeholder - actual values not in pkl)
                    emission_wls = list(range(n_bands))  # Will need to be set properly

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
        return (
            f"WavelengthBand(rank={self.rank}, ex={self.excitation_nm}nm, "
            f"em={self.emission_nm}nm, score={self.influence_score:.2f})"
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
