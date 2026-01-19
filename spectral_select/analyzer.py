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

    def _load_or_train_model(self) -> None:
        """Load pretrained model or train a new one if needed.

        Attempts to load model weights from config.model_path. If the file
        is missing or architecture doesn't match, trains a new model.
        """
        if self._dataset is None:
            raise RuntimeError("_load_data must be called before _load_or_train_model")

        # TODO: Phase 8 cleanup - remove sys.path manipulation
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.models import HyperspectralCAEWithMasking

        # Get data from dataset to initialize model
        all_data = self._dataset.get_all_data()

        # Initialize model with correct architecture parameters
        # Convert tensors to numpy for model initialization
        excitations_data = {ex: data.numpy() for ex, data in all_data.items()}
        self._model = HyperspectralCAEWithMasking(
            excitations_data=excitations_data,
            k1=20,
            k3=20,
            filter_size=5,
        )

        # Attempt to load pretrained weights
        model_path = self._config.model_path
        if model_path is not None and model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self._device)
                self._model.load_state_dict(state_dict)
                self._model = self._model.to(self._device)
                # Set model to evaluation mode (not inference mode)
                self._model.train(False)
                logger.info(f"Model loaded from {model_path}")
                return
            except RuntimeError as e:
                error_msg = str(e)
                if "Missing key(s)" in error_msg or "size mismatch" in error_msg:
                    logger.warning(
                        f"Model architecture mismatch: {error_msg[:100]}... "
                        "Training new model."
                    )
                else:
                    raise
        else:
            if model_path is not None:
                logger.info(f"Model file not found at {model_path}")
            else:
                logger.info("No model path specified")

        # Train new model if loading failed or no file exists
        self._train_new_model()

    def _train_new_model(self) -> None:
        """Train a new autoencoder model from scratch.

        Uses the training parameters from scripts.models.training with
        reasonable defaults for wavelength selection analysis.
        """
        logger.warning(
            "Training new autoencoder model. "
            "This may take several minutes..."
        )

        # TODO: Phase 8 cleanup - remove sys.path manipulation
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.models.training import train_with_masking

        # Determine output directory for training
        if self._config.output_dir is not None:
            train_output_dir = self._config.output_dir / "model_training"
        else:
            train_output_dir = Path("model_output")

        # Get mask from dataset
        mask = self._dataset.processed_mask

        # Train with reasonable defaults
        self._model, losses = train_with_masking(
            model=self._model,
            dataset=self._dataset,
            num_epochs=3000,
            learning_rate=0.001,
            device=str(self._device),
            mask=mask,
            output_dir=str(train_output_dir),
            verbose=True,
        )

        # Move to device and set to evaluation mode
        self._model = self._model.to(self._device)
        self._model.train(False)

        # Save model if path specified
        model_path = self._config.model_path
        if model_path is not None:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

        logger.info(
            f"Model training complete. Final loss: {losses[-1]:.6f}"
        )

    def _setup_baseline(self) -> None:
        """Setup baseline latent representations for perturbation analysis.

        Extracts patches from the spatial domain, validates them against the mask,
        encodes them through the model, and stores baseline latent representations.
        These baselines are used for measuring the effect of wavelength perturbations.
        """
        if self._dataset is None or self._model is None:
            raise RuntimeError(
                "_load_data and _load_or_train_model must be called before _setup_baseline"
            )

        logger.info("Setting up baseline latent representations...")

        # Get all data and spatial dimensions from dataset
        all_data = self._dataset.get_all_data()
        height, width = self._dataset.get_spatial_dimensions()
        mask = self._dataset.processed_mask

        # Get patch parameters from config
        patch_size = self._config.patch_size
        stride = self._config.patch_stride
        n_baseline_patches = self._config.n_baseline_patches

        # Find valid patch coordinates (more than 50% valid pixels in mask)
        patch_coords: List[Tuple[int, int]] = []

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                if mask is not None:
                    patch_mask = mask[y : y + patch_size, x : x + patch_size]
                    valid_ratio = np.sum(patch_mask) / (patch_size * patch_size)
                    if valid_ratio <= 0.5:
                        continue

                patch_coords.append((y, x))
                if len(patch_coords) >= n_baseline_patches:
                    break
            if len(patch_coords) >= n_baseline_patches:
                break

        if not patch_coords:
            raise ValueError(
                f"No valid patches found with size {patch_size} and stride {stride}. "
                "Check mask coverage or reduce patch_size."
            )

        # Store patch coordinates for reproducibility
        self._patch_coords = patch_coords

        logger.info(
            f"Selected {len(patch_coords)} patches of size {patch_size}x{patch_size}"
        )

        # Extract patches for each excitation wavelength
        patches_data: Dict[float, torch.Tensor] = {}
        for ex in all_data:
            patches = []
            for y, x in patch_coords:
                patch = all_data[ex][y : y + patch_size, x : x + patch_size, :]
                patches.append(patch)
            patches_data[ex] = torch.stack(patches)

        # Encode patches through model to get baseline latent representations
        with torch.no_grad():
            # Move patches to device
            patches_data_device = {
                ex: data.to(self._device) for ex, data in patches_data.items()
            }

            # Encode patches
            self._baseline_latent = self._model.encode(patches_data_device)

            # Decode for baseline reconstruction
            self._baseline_reconstruction = self._model.decode(self._baseline_latent)

        logger.info(f"Baseline latent shape: {self._baseline_latent.shape}")

    def _select_important_dimensions(self) -> None:
        """Select the most important latent dimensions for perturbation.

        Analyzes the baseline latent representations to identify which dimensions
        carry the most information, using one of three methods:
        - variance: Dimensions with highest variance across samples
        - activation: Dimensions with highest mean absolute activation
        - pca: Dimensions with highest PCA loading contributions

        Stores results in self._important_dims as List[(score, (c, l, h, w))].
        """
        if self._baseline_latent is None:
            raise RuntimeError("_setup_baseline must be called before _select_important_dimensions")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        logger.info(
            f"Selecting important dimensions using {self._config.dimension_selection_method} method..."
        )

        latent = self._baseline_latent
        batch_size, n_channels, n_latent, h_latent, w_latent = latent.shape

        # Flatten spatial dimensions for analysis: (batch, features)
        latent_flat = latent.reshape(batch_size, -1)

        # Compute importance scores based on selected method
        method = self._config.dimension_selection_method

        if method == "variance":
            # Higher variance = more informative dimension
            importance_scores = torch.var(latent_flat, dim=0)

        elif method == "activation":
            # Higher mean absolute activation = more used by model
            importance_scores = torch.mean(torch.abs(latent_flat), dim=0)

        elif method == "pca":
            # Use PCA loadings to identify important dimensions
            scaler = StandardScaler()
            latent_scaled = scaler.fit_transform(latent_flat.cpu().numpy())
            n_samples, n_features = latent_scaled.shape

            # PCA components limited by samples and target dimensions
            n_components = min(
                self._config.n_important_dimensions * 2,
                n_features,
                n_samples - 1,
            )

            pca = PCA(n_components=n_components)
            pca.fit(latent_scaled)

            # Sum absolute loadings across components to get importance
            components = torch.tensor(pca.components_)
            importance_scores = torch.sum(torch.abs(components), dim=0)

        else:
            raise ValueError(f"Unknown dimension selection method: {method}")

        # Convert flat indices to (channel, latent, h, w) coordinates
        coordinate_importance: List[Tuple[float, Tuple[int, int, int, int]]] = []
        for i, score in enumerate(importance_scores):
            coords = np.unravel_index(i, (n_channels, n_latent, h_latent, w_latent))
            coords_tuple: Tuple[int, int, int, int] = tuple(int(c) for c in coords)
            coordinate_importance.append((score.item(), coords_tuple))

        # Sort by importance (descending) and keep top N
        coordinate_importance.sort(reverse=True, key=lambda x: x[0])
        self._important_dims = coordinate_importance[: self._config.n_important_dimensions]

        logger.info(f"Selected top {len(self._important_dims)} dimensions")
        # Log top 5 for visibility
        for i, (score, coords) in enumerate(self._important_dims[:5]):
            logger.info(f"  {i + 1}. {coords}: importance = {score:.6f}")
