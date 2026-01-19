"""Configuration system for spectral analysis.

This module provides the unified Config dataclass that controls the entire
wavelength selection pipeline. It supports pluggable components through
protocol-based interfaces.

Example:
    # Using built-in components via string identifiers
    config = Config(
        sample_name="Lichens_2",
        classifier="knn",
        clustering="kmeans",
    )

    # Using custom implementations
    config = Config(
        sample_name="Lichens_2",
        classifier=MyCustomClassifier,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type, Union

if TYPE_CHECKING:
    from .protocols import (
        AutoencoderProtocol,
        ClassifierProtocol,
        ClusteringProtocol,
        WavelengthRankerProtocol,
    )

# Built-in component registries
# Actual implementations will be added in later phases
BUILT_IN_CLASSIFIERS: dict[str, Any] = {"knn": None}
BUILT_IN_CLUSTERING: dict[str, Any] = {"kmeans": None}
BUILT_IN_AUTOENCODERS: dict[str, Any] = {"standard": None}
BUILT_IN_WAVELENGTH_RANKERS: dict[str, Any] = {"perturbation": None}

# Valid options for validation
_VALID_DIMENSION_METHODS = frozenset({"variance", "activation", "pca"})
_VALID_PERTURBATION_METHODS = frozenset(
    {"percentile", "standard_deviation", "absolute_range"}
)
_VALID_NORMALIZATION_METHODS = frozenset({"variance", "max_per_excitation", "none"})
_VALID_DIVERSITY_METHODS = frozenset({"mmr", "min_distance", "none"})
_VALID_DEVICES = frozenset({"cuda", "cpu", "mps"})


@dataclass
class Config:
    """Unified configuration for the wavelength selection pipeline.

    This dataclass serves as the control center for the entire analysis.
    Configure analysis parameters and pluggable components here.

    Attributes:
        sample_name: Identifier for the sample being analyzed.
        data_path: Path to input spectral data file.
        mask_path: Path to mask file for valid pixels.
        model_path: Path to trained autoencoder model.
        output_dir: Directory for saving results.
        dimension_selection_method: Method for selecting important latent dimensions.
            One of "variance", "activation", or "pca".
        n_important_dimensions: Number of latent dimensions to analyze.
        perturbation_method: Method for perturbing latent dimensions.
            One of "percentile", "standard_deviation", or "absolute_range".
        perturbation_magnitudes: List of perturbation strengths to apply.
        perturbation_directions: Direction(s) of perturbation.
        normalization_method: How to normalize sensitivity scores.
            One of "variance", "max_per_excitation", or "none".
        n_bands_to_select: Target number of wavelength bands to select.
        n_layers_to_extract: Number of layers to extract from model.
        use_diversity_constraint: Whether to enforce spectral diversity.
        diversity_method: Diversity enforcement method.
            One of "mmr", "min_distance", or "none".
        lambda_diversity: Trade-off parameter for MMR (0=relevance, 1=diversity).
        min_distance_nm: Minimum spectral distance in nanometers.
        save_tiff_layers: Whether to save intermediate TIFF layers.
        save_visualizations: Whether to generate and save plots.
        save_detailed_results: Whether to save detailed analysis results.
        device: Computation device ("cuda", "cpu", or "mps").
        n_baseline_patches: Number of patches for baseline computation.
        patch_size: Size of image patches in pixels.
        patch_stride: Stride between patches.
        random_seed: Seed for reproducibility.
        classifier: Classification component (string ID or class).
        clustering: Clustering component (string ID or class).
        autoencoder_architecture: Autoencoder architecture (string ID or class).
        wavelength_ranker: Wavelength ranking component (string ID or class).
    """

    # Data configuration
    sample_name: str = "sample"
    data_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    model_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Analysis parameters
    dimension_selection_method: str = "activation"
    n_important_dimensions: int = 15
    perturbation_method: str = "percentile"
    perturbation_magnitudes: List[float] = field(default_factory=lambda: [10, 20, 30])
    perturbation_directions: List[str] = field(
        default_factory=lambda: ["bidirectional"]
    )
    normalization_method: str = "variance"

    # Selection parameters
    n_bands_to_select: int = 30
    n_layers_to_extract: int = 10

    # Diversity constraint parameters
    use_diversity_constraint: bool = False
    diversity_method: str = "mmr"
    lambda_diversity: float = 0.5
    min_distance_nm: float = 15.0

    # Output configuration
    save_tiff_layers: bool = True
    save_visualizations: bool = True
    save_detailed_results: bool = True

    # Technical parameters
    device: str = "cuda"
    n_baseline_patches: int = 50
    patch_size: int = 32
    patch_stride: int = 16
    random_seed: int = 42

    # Pluggable components
    # Type hint uses Union for flexibility with string identifiers or class references
    classifier: Union[str, Type[ClassifierProtocol], Callable[..., Any]] = "knn"
    clustering: Union[str, Type[ClusteringProtocol], Callable[..., Any]] = "kmeans"
    autoencoder_architecture: Union[
        str, Type[AutoencoderProtocol], Callable[..., Any]
    ] = "standard"
    wavelength_ranker: Union[
        str, Type[WavelengthRankerProtocol], Callable[..., Any]
    ] = "perturbation"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_string_options()
        self._validate_numeric_ranges()
        self._convert_paths()

    def _validate_string_options(self) -> None:
        """Validate string-based configuration options."""
        if self.dimension_selection_method not in _VALID_DIMENSION_METHODS:
            raise ValueError(
                f"dimension_selection_method must be one of {sorted(_VALID_DIMENSION_METHODS)}, "
                f"got '{self.dimension_selection_method}'"
            )

        if self.perturbation_method not in _VALID_PERTURBATION_METHODS:
            raise ValueError(
                f"perturbation_method must be one of {sorted(_VALID_PERTURBATION_METHODS)}, "
                f"got '{self.perturbation_method}'"
            )

        if self.normalization_method not in _VALID_NORMALIZATION_METHODS:
            raise ValueError(
                f"normalization_method must be one of {sorted(_VALID_NORMALIZATION_METHODS)}, "
                f"got '{self.normalization_method}'"
            )

        if self.diversity_method not in _VALID_DIVERSITY_METHODS:
            raise ValueError(
                f"diversity_method must be one of {sorted(_VALID_DIVERSITY_METHODS)}, "
                f"got '{self.diversity_method}'"
            )

        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_VALID_DEVICES)}, "
                f"got '{self.device}'"
            )

    def _validate_numeric_ranges(self) -> None:
        """Validate numeric parameter ranges."""
        if self.n_important_dimensions <= 0:
            raise ValueError(
                f"n_important_dimensions must be positive, got {self.n_important_dimensions}"
            )

        if self.n_bands_to_select <= 0:
            raise ValueError(
                f"n_bands_to_select must be positive, got {self.n_bands_to_select}"
            )

        if self.n_layers_to_extract <= 0:
            raise ValueError(
                f"n_layers_to_extract must be positive, got {self.n_layers_to_extract}"
            )

        if not 0.0 <= self.lambda_diversity <= 1.0:
            raise ValueError(
                f"lambda_diversity must be between 0 and 1, got {self.lambda_diversity}"
            )

        if self.min_distance_nm < 0:
            raise ValueError(
                f"min_distance_nm must be non-negative, got {self.min_distance_nm}"
            )

        if self.n_baseline_patches <= 0:
            raise ValueError(
                f"n_baseline_patches must be positive, got {self.n_baseline_patches}"
            )

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {self.patch_stride}")

        if not self.perturbation_magnitudes:
            raise ValueError("perturbation_magnitudes cannot be empty")

    def _convert_paths(self) -> None:
        """Convert string paths to Path objects if needed."""
        if self.data_path is not None and not isinstance(self.data_path, Path):
            object.__setattr__(self, "data_path", Path(self.data_path))
        if self.mask_path is not None and not isinstance(self.mask_path, Path):
            object.__setattr__(self, "mask_path", Path(self.mask_path))
        if self.model_path is not None and not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    def resolve_classifier(self) -> Any:
        """Resolve classifier to actual implementation.

        Returns:
            The classifier class or callable.

        Raises:
            ValueError: If classifier string is not a known built-in.
        """
        return self._resolve_component(
            self.classifier, BUILT_IN_CLASSIFIERS, "classifier"
        )

    def resolve_clustering(self) -> Any:
        """Resolve clustering to actual implementation.

        Returns:
            The clustering class or callable.

        Raises:
            ValueError: If clustering string is not a known built-in.
        """
        return self._resolve_component(
            self.clustering, BUILT_IN_CLUSTERING, "clustering"
        )

    def resolve_autoencoder(self) -> Any:
        """Resolve autoencoder architecture to actual implementation.

        Returns:
            The autoencoder class or callable.

        Raises:
            ValueError: If autoencoder_architecture string is not a known built-in.
        """
        return self._resolve_component(
            self.autoencoder_architecture, BUILT_IN_AUTOENCODERS, "autoencoder_architecture"
        )

    def resolve_wavelength_ranker(self) -> Any:
        """Resolve wavelength ranker to actual implementation.

        Returns:
            The wavelength ranker class or callable.

        Raises:
            ValueError: If wavelength_ranker string is not a known built-in.
        """
        return self._resolve_component(
            self.wavelength_ranker, BUILT_IN_WAVELENGTH_RANKERS, "wavelength_ranker"
        )

    def _resolve_component(
        self,
        component: Union[str, Type[Any], Callable[..., Any]],
        registry: dict[str, Any],
        component_name: str,
    ) -> Any:
        """Resolve a component from string identifier or return as-is.

        Args:
            component: String identifier or class/callable.
            registry: Built-in component registry.
            component_name: Name for error messages.

        Returns:
            The resolved component.

        Raises:
            ValueError: If string is not in registry or component is invalid.
        """
        if isinstance(component, str):
            if component not in registry:
                raise ValueError(
                    f"Unknown {component_name}: '{component}'. "
                    f"Available: {sorted(registry.keys())}"
                )
            impl = registry[component]
            if impl is None:
                # Placeholder - actual implementation not yet available
                return component  # Return string for now
            return impl

        # If not a string, assume it's a class or callable
        if not callable(component):
            raise ValueError(
                f"{component_name} must be a string identifier or callable, "
                f"got {type(component).__name__}"
            )
        return component
