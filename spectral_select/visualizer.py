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

    def plot_influence_heatmap(
        self,
        influence_matrix: Optional[Dict[float, np.ndarray]] = None,
    ) -> Path:
        """Plot heatmap of influence scores across excitation-emission space.

        Creates a 2D heatmap showing influence scores with excitation wavelengths
        on the y-axis and emission band indices on the x-axis. Uses log scale
        for better visualization of wide value ranges.

        Args:
            influence_matrix: Optional pre-computed matrix as {excitation_nm: influence_array}.
                If not provided and has_result, builds from selected_bands.

        Returns:
            Path to saved figure.

        Raises:
            ValueError: If no result or influence_matrix available.
        """
        # Get or build influence matrix
        if influence_matrix is not None:
            matrix_dict = influence_matrix
        elif self.has_result:
            # Build from result's selected bands
            result = self._result if self._result else self._analyzer.result_
            matrix_dict: Dict[float, np.ndarray] = {}

            # Get unique excitations and their band counts
            for band in result.selected_bands:
                ex = band.excitation_nm
                if ex not in matrix_dict:
                    # Find max emission band index for this excitation
                    max_idx = max(
                        b.emission_band_index
                        for b in result.selected_bands
                        if b.excitation_nm == ex
                    )
                    matrix_dict[ex] = np.zeros(max_idx + 1)

                matrix_dict[ex][band.emission_band_index] = band.influence_score
        else:
            raise ValueError("No result or influence_matrix available for heatmap")

        # Prepare data for heatmap
        excitations = sorted(matrix_dict.keys())
        max_bands = max(len(matrix_dict[ex]) for ex in excitations)

        influence_array = np.zeros((len(excitations), max_bands))
        for i, ex in enumerate(excitations):
            influences = matrix_dict[ex]
            influence_array[i, :len(influences)] = influences

        # Create figure
        fig, ax = plt.subplots(figsize=self._figsize)

        # Use log scale for better visualization
        influence_log = np.log10(influence_array + 1e-10)

        heatmap = ax.imshow(influence_log, aspect='auto', cmap='YlOrRd', interpolation='nearest')

        # Customize plot
        cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8)
        cbar.set_label('Log10(Influence Score)', fontsize=10)

        ax.set_xlabel('Emission Band Index', fontsize=12)
        ax.set_ylabel('Excitation Wavelength (nm)', fontsize=12)
        ax.set_title('Wavelength Influence Heatmap (Log Scale)', fontsize=14, fontweight='bold')

        # Set y-axis labels
        ax.set_yticks(range(len(excitations)))
        ax.set_yticklabels([f"{ex:.0f}" for ex in excitations])

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save_figure(fig, "influence_heatmap")

    def plot_wavelength_scatter(self, top_n: int = 50) -> Path:
        """Plot scatter of selected wavelengths in excitation-emission space.

        Creates a scatter plot where each point represents a selected wavelength
        combination. Point color indicates influence score, and point size
        indicates ranking (larger = higher ranked).

        Args:
            top_n: Number of top bands to plot (default 50 for readability).

        Returns:
            Path to saved figure.

        Raises:
            ValueError: If no result is available.
        """
        if not self.has_result:
            raise ValueError("No result available for wavelength scatter plot")

        result = self._result if self._result else self._analyzer.result_

        # Extract data for plotting (limit to top_n for readability)
        top_bands = result.selected_bands[:top_n]
        ex_values = [band.excitation_nm for band in top_bands]
        em_values = [band.emission_nm for band in top_bands]
        influences = [band.influence_score for band in top_bands]
        ranks = [band.rank for band in top_bands]

        # Create figure
        fig, ax = plt.subplots(figsize=self._figsize)

        # Size decreases with rank (higher rank = larger point)
        sizes = [200 - (rank - 1) * 3 for rank in ranks]
        sizes = [max(s, 20) for s in sizes]  # Minimum size

        scatter = ax.scatter(
            ex_values, em_values,
            c=influences,
            s=sizes,
            cmap='plasma',
            edgecolors='black',
            linewidth=1,
            alpha=0.8
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Influence Score', fontsize=10)

        # Labels and title
        ax.set_xlabel('Excitation Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Emission Wavelength (nm)', fontsize=12)
        ax.set_title(
            'Top Wavelength Combinations\n(Size ~ Ranking, Color ~ Influence)',
            fontsize=14, fontweight='bold'
        )

        # Annotate top 20 with rank numbers
        for i, (ex, em, rank) in enumerate(zip(ex_values[:20], em_values[:20], ranks[:20])):
            ax.annotate(
                f'{rank}', (ex, em),
                xytext=(3, 3), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save_figure(fig, "wavelength_scatter")

    def plot_excitation_distribution(self) -> Path:
        """Plot distribution of selected bands across excitation wavelengths.

        Creates a bar chart showing how many bands were selected from each
        excitation wavelength, with value labels on top of each bar.

        Returns:
            Path to saved figure.

        Raises:
            ValueError: If no result is available.
        """
        if not self.has_result:
            raise ValueError("No result available for excitation distribution plot")

        result = self._result if self._result else self._analyzer.result_

        # Count selections per excitation
        excitation_counts: Dict[float, int] = {}
        for band in result.selected_bands:
            ex = band.excitation_nm
            excitation_counts[ex] = excitation_counts.get(ex, 0) + 1

        # Sort by wavelength
        excitations = sorted(excitation_counts.keys())
        counts = [excitation_counts[ex] for ex in excitations]

        # Create figure
        fig, ax = plt.subplots(figsize=self._figsize)

        bars = ax.bar(
            range(len(excitations)), counts,
            color='skyblue', edgecolor='navy',
            alpha=0.7, linewidth=1.5
        )

        # Labels and title
        ax.set_xlabel('Excitation Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Number of Selected Bands', fontsize=12)
        ax.set_title(
            'Distribution of Selected Bands Across Excitation Wavelengths',
            fontsize=14, fontweight='bold'
        )

        ax.set_xticks(range(len(excitations)))
        ax.set_xticklabels([f"{ex:.0f}" for ex in excitations], rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha='center', va='bottom', fontweight='bold'
            )

        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()

        return self._save_figure(fig, "excitation_distribution")

    def plot_influence_ranking(self) -> Path:
        """Plot influence scores versus band ranking.

        Creates a line plot showing how influence scores decay with rank.
        Automatically uses log scale for y-axis if the ratio of max/min
        scores exceeds 100x.

        Returns:
            Path to saved figure.

        Raises:
            ValueError: If no result is available.
        """
        if not self.has_result:
            raise ValueError("No result available for influence ranking plot")

        result = self._result if self._result else self._analyzer.result_

        # Extract ranks and influences
        ranks = [band.rank for band in result.selected_bands]
        influences = [band.influence_score for band in result.selected_bands]

        # Create figure
        fig, ax = plt.subplots(figsize=self._figsize)

        # Create line plot with markers
        ax.plot(ranks, influences, 'o-', linewidth=2, markersize=6,
                markerfacecolor='red', markeredgecolor='darkred', alpha=0.7)

        # Customize plot
        ax.set_xlabel('Band Rank', fontsize=12)
        ax.set_ylabel('Influence Score', fontsize=12)
        ax.set_title('Influence Score vs. Band Ranking', fontsize=14, fontweight='bold')

        # Use log scale if range is large
        if max(influences) / (min(influences) + 1e-10) > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Influence Score (Log Scale)', fontsize=12)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save_figure(fig, "influence_ranking")

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
