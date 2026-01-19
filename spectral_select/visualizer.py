"""Visualization utilities for spectral analysis.

This module provides the Visualizer class for creating publication-quality
visualizations of wavelength selection analysis results.

Example:
    from spectral_select import Visualizer, WavelengthResult

    result = WavelengthResult.from_json("analysis_results.json")
    viz = Visualizer.from_result(result)
    viz.plot_influence_heatmap()
    viz.plot_wavelength_scatter()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .types import WavelengthResult

if TYPE_CHECKING:
    from .analyzer import Analyzer


class Visualizer:
    """Publication-quality visualizations for wavelength selection analysis.

    Creates and saves visualizations for wavelength selection results including
    influence heatmaps, wavelength distributions, and validation plots.

    Attributes:
        output_dir: Directory where figures are saved.
        dpi: Resolution for saved figures.
        figsize: Default figure size (width, height).

    Example:
        # From result
        viz = Visualizer.from_result(result)
        path = viz.plot_influence_heatmap()
        print(f"Saved to: {path}")

        # From analyzer
        analyzer.fit(data)
        viz = Visualizer.from_analyzer(analyzer)
        paths = viz.plot_all()
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        figsize: Tuple[int, int] = (12, 8),
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        """Initialize visualizer with output settings.

        Args:
            output_dir: Directory for saving figures. Defaults to ./visualizations.
            dpi: Resolution for saved figures (300 for publication quality).
            figsize: Default figure size as (width, height) in inches.
            style: Matplotlib style to apply (seaborn styles recommended).
        """
        # Store private attributes
        self._output_dir = Path(output_dir) if output_dir else Path.cwd() / "visualizations"
        self._dpi = dpi
        self._figsize = figsize
        self._style = style

        # Optional bound data (set by factory methods)
        self._result: Optional[WavelengthResult] = None
        self._analyzer: Optional["Analyzer"] = None

        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Setup matplotlib style
        self._setup_style()

        # Initialize color palette
        self._colors = sns.color_palette("husl", 12)

    @property
    def output_dir(self) -> Path:
        """Directory where figures are saved."""
        return self._output_dir

    @property
    def dpi(self) -> int:
        """Resolution for saved figures."""
        return self._dpi

    @property
    def figsize(self) -> Tuple[int, int]:
        """Default figure size (width, height)."""
        return self._figsize

    @property
    def has_result(self) -> bool:
        """Whether a result is available (from _result or _analyzer.result_)."""
        if self._result is not None:
            return True
        if self._analyzer is not None:
            return hasattr(self._analyzer, "result_") and self._analyzer.result_ is not None
        return False

    def _setup_style(self) -> None:
        """Apply matplotlib style and seaborn settings."""
        try:
            plt.style.use(self._style)
        except OSError:
            # Fallback to default if style not available
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn" in plt.style.available else "default")

        # Set seaborn defaults
        sns.set_palette("husl")
        sns.set_context("paper", font_scale=1.2)

    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save figure to output directory.

        Args:
            fig: Matplotlib figure to save.
            name: Filename (without extension).

        Returns:
            Path to the saved figure.
        """
        path = self._output_dir / f"{name}.png"
        fig.savefig(path, dpi=self._dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    def _get_color(self, index: int) -> str:
        """Get color from palette by index (cycles if index > palette size).

        Args:
            index: Color index.

        Returns:
            Color as hex string or RGB tuple.
        """
        return self._colors[index % len(self._colors)]

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_result(
        cls,
        result: WavelengthResult,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "Visualizer":
        """Create visualizer bound to a specific result.

        Args:
            result: WavelengthResult to visualize.
            output_dir: Output directory. Defaults to visualizations/{sample_name}.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            Visualizer instance with result bound.
        """
        if output_dir is None:
            output_dir = Path("visualizations") / result.sample_name

        instance = cls(output_dir=output_dir, **kwargs)
        instance._result = result
        return instance

    @classmethod
    def from_analyzer(
        cls,
        analyzer: "Analyzer",
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "Visualizer":
        """Create visualizer bound to an analyzer.

        The visualizer will use analyzer.result_ after fit() is called.

        Args:
            analyzer: Analyzer instance (should have fit() called).
            output_dir: Output directory. Defaults to visualizations/{sample_name}.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            Visualizer instance with analyzer bound.
        """
        if output_dir is None:
            output_dir = Path("visualizations") / analyzer.config.sample_name

        instance = cls(output_dir=output_dir, **kwargs)
        instance._analyzer = analyzer
        return instance

    # =========================================================================
    # Wavelength Analysis Plots (Phase 5 Plan 02)
    # =========================================================================

    def plot_influence_heatmap(self) -> Path:
        """Plot heatmap of influence scores across excitation-emission space.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_influence_heatmap: See Phase 5 Plan 02")

    def plot_wavelength_scatter(self) -> Path:
        """Plot scatter of selected wavelengths in excitation-emission space.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_wavelength_scatter: See Phase 5 Plan 02")

    def plot_excitation_distribution(self) -> Path:
        """Plot distribution of selected bands across excitation wavelengths.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_excitation_distribution: See Phase 5 Plan 02")

    def plot_influence_ranking(self) -> Path:
        """Plot ranked bar chart of influence scores.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_influence_ranking: See Phase 5 Plan 02")

    def plot_wavelength_coverage(self) -> Path:
        """Plot coverage of selected wavelengths across spectrum.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_wavelength_coverage: See Phase 5 Plan 02")

    # =========================================================================
    # Clustering/Validation Plots (Phase 5 Plan 03)
    # =========================================================================

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Path:
        """Plot confusion matrix with annotations.

        Args:
            cm: Confusion matrix as 2D array.
            class_names: Optional class labels for axes.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_confusion_matrix: See Phase 5 Plan 03")

    def plot_accuracy_heatmap(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> Path:
        """Plot spatial accuracy heatmap comparing predictions to ground truth.

        Args:
            ground_truth: Ground truth label map.
            predictions: Predicted label map.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_accuracy_heatmap: See Phase 5 Plan 03")

    def plot_roi_overlay(
        self,
        cluster_map: np.ndarray,
        roi_regions: List[Dict],
    ) -> Path:
        """Plot cluster map with ROI region overlays.

        Args:
            cluster_map: Cluster assignment map.
            roi_regions: List of ROI region definitions.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_roi_overlay: See Phase 5 Plan 03")

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def plot_summary_dashboard(self) -> Path:
        """Create multi-panel summary dashboard of analysis results.

        Returns:
            Path to saved figure.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_summary_dashboard: See Phase 5 Plan 02")

    def plot_all(self) -> List[Path]:
        """Generate all available plots.

        Returns:
            List of paths to all saved figures.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("plot_all: See Phase 5 Plan 03")
