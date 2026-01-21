"""Jupyter widgets for interactive ROI selection in hyperspectral data.

This module provides ipywidgets-based tools for selecting regions of interest
(ROIs) directly in Jupyter notebooks using matplotlib's interactive selectors.

Requires the %matplotlib widget magic to be enabled in notebooks.

Example:
    %matplotlib widget
    from spectral_select import SpectraData, ROIWidget

    data = SpectraData.from_saved("sample.pkl")
    widget = ROIWidget(data)
    widget.display()

    # After drawing ROI
    mask = widget.get_mask()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from .types import SpectraData


def create_display_image(
    cube: np.ndarray,
    band_index: Optional[int] = None,
    method: str = "mean",
    percentile: float = 98.0,
) -> np.ndarray:
    """Create a 2D display image from a 3D hyperspectral cube.

    Converts a 3D (height, width, bands) cube into a 2D image for visualization
    using various projection methods.

    Args:
        cube: 3D array in (height, width, bands) format.
        band_index: If provided, extract this specific band. Otherwise use method.
        method: Projection method when band_index is None:
            - "mean": Mean across bands (default)
            - "max": Maximum projection
            - "rgb": False color using 3 representative bands
        percentile: Percentile for contrast normalization (default 98).

    Returns:
        2D or 3D array suitable for matplotlib imshow:
        - If method="rgb" or band_index given: normalized float array
        - Otherwise: normalized 2D float array

    Example:
        img = create_display_image(cube)  # Mean projection
        img = create_display_image(cube, band_index=10)  # Single band
        img = create_display_image(cube, method="rgb")  # False color
    """
    # Handle NaN values
    cube = np.nan_to_num(cube, nan=0.0)

    height, width, n_bands = cube.shape

    if band_index is not None:
        # Extract single band
        if not (0 <= band_index < n_bands):
            raise IndexError(f"band_index {band_index} out of range [0, {n_bands})")
        img = cube[:, :, band_index].copy()
        pval = np.percentile(img[img > 0], percentile) if np.any(img > 0) else 1.0
        if pval == 0:
            pval = 1.0
        return np.clip(img / pval, 0, 1).astype(np.float32)

    if method == "rgb" and n_bands >= 3:
        # Create false color from three representative bands
        r_idx = int(n_bands * 0.2)
        g_idx = int(n_bands * 0.5)
        b_idx = int(n_bands * 0.8)

        def normalize_band(band: np.ndarray) -> np.ndarray:
            pval = np.percentile(band[band > 0], percentile) if np.any(band > 0) else 1.0
            if pval == 0:
                pval = 1.0
            return np.clip(band / pval, 0, 1)

        r_norm = normalize_band(cube[:, :, r_idx])
        g_norm = normalize_band(cube[:, :, g_idx])
        b_norm = normalize_band(cube[:, :, b_idx])

        return np.stack([r_norm, g_norm, b_norm], axis=2).astype(np.float32)

    elif method == "max":
        # Maximum intensity projection
        img = np.max(cube, axis=2)
    else:
        # Default: mean projection
        img = np.mean(cube, axis=2)

    # Normalize
    pval = np.percentile(img[img > 0], percentile) if np.any(img > 0) else 1.0
    if pval == 0:
        pval = 1.0
    return np.clip(img / pval, 0, 1).astype(np.float32)


def path_to_mask(
    path_vertices: List[Tuple[float, float]],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Convert matplotlib Path vertices to a binary mask.

    Uses matplotlib.path.Path.contains_points to create a mask from
    polygon vertices, suitable for selecting pixels within drawn regions.

    Args:
        path_vertices: List of (x, y) coordinate tuples defining the path.
            Coordinates should be in data coordinates (column, row).
        shape: Output mask shape as (height, width).

    Returns:
        Boolean 2D array where True indicates points inside the path.

    Example:
        vertices = [(10, 20), (50, 20), (50, 80), (10, 80)]
        mask = path_to_mask(vertices, (100, 100))
        print(mask.sum())  # Number of selected pixels
    """
    height, width = shape

    # Create grid of pixel coordinates
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    points = np.column_stack([x_coords.ravel(), y_coords.ravel()])

    # Create matplotlib Path and check containment
    mpl_path = MplPath(path_vertices)
    mask_flat = mpl_path.contains_points(points)

    return mask_flat.reshape(height, width)


class ROIWidget:
    """Interactive ROI selection widget for Jupyter notebooks.

    Displays a hyperspectral image and allows drawing ROIs using
    matplotlib's LassoSelector. Requires %matplotlib widget magic.

    Attributes:
        fig: Matplotlib figure containing the image.
        ax: Matplotlib axes with the displayed image.
        mask: Current ROI mask (None if no selection).

    Example:
        %matplotlib widget
        from spectral_select import SpectraData, ROIWidget

        data = SpectraData.from_saved("sample.pkl")
        widget = ROIWidget(data)
        widget.display()

        # After drawing ROI
        mask = widget.get_mask()
    """

    def __init__(
        self,
        data: "SpectraData",
        excitation: Optional[float] = None,
        figsize: Tuple[float, float] = (8, 6),
    ) -> None:
        """Initialize ROI selection widget.

        Args:
            data: SpectraData object containing hyperspectral cube.
            excitation: Excitation wavelength to display. If None, uses first.
            figsize: Figure size as (width, height) in inches.
        """
        # Import SpectraData lazily to avoid circular imports
        from .types import SpectraData as SD

        self._data: SD = data
        self._figsize = figsize

        # Select excitation wavelength
        if excitation is None:
            self._excitation = data.excitation_wavelengths[0]
        else:
            if excitation not in data.excitation_wavelengths:
                raise ValueError(
                    f"Excitation {excitation}nm not found. "
                    f"Available: {data.excitation_wavelengths}"
                )
            self._excitation = excitation

        # Get cube for selected excitation
        ex_data = data.get_excitation(self._excitation)
        self._cube = ex_data.cube
        self._spatial_shape = (ex_data.height, ex_data.width)

        # State
        self._mask: Optional[np.ndarray] = None
        self._fig: Optional["Figure"] = None  # noqa: F821
        self._ax: Optional["Axes"] = None  # noqa: F821
        self._selector: Optional["LassoSelector"] = None  # noqa: F821
        self._image_artist: Optional["AxesImage"] = None  # noqa: F821
        self._overlay_artist: Optional["AxesImage"] = None  # noqa: F821
        self._output: Optional["Output"] = None  # noqa: F821

    @property
    def fig(self) -> Optional["Figure"]:  # noqa: F821
        """Matplotlib figure (None before display called)."""
        return self._fig

    @property
    def ax(self) -> Optional["Axes"]:  # noqa: F821
        """Matplotlib axes (None before display called)."""
        return self._ax

    @property
    def mask(self) -> Optional[np.ndarray]:
        """Current ROI mask (None if no selection)."""
        return self._mask.copy() if self._mask is not None else None

    def display(self) -> "VBox":  # noqa: F821
        """Display the widget in a Jupyter notebook.

        Creates an interactive matplotlib figure with LassoSelector tool
        for drawing ROI selections.

        Returns:
            ipywidgets VBox containing the figure and instruction label.

        Note:
            Requires %matplotlib widget magic to be enabled in notebook.
        """
        # Import ipywidgets lazily
        try:
            from ipywidgets import Label, VBox, Output
        except ImportError:
            raise ImportError(
                "ipywidgets is required for ROIWidget. "
                "Install with: pip install ipywidgets"
            )

        import matplotlib.pyplot as plt
        from matplotlib.widgets import LassoSelector

        # Create output widget to capture matplotlib figure
        self._output = Output()

        with self._output:
            # Create figure and axes
            self._fig, self._ax = plt.subplots(figsize=self._figsize)

            # Create display image and show it
            display_img = create_display_image(self._cube, method="mean")
            self._image_artist = self._ax.imshow(display_img, cmap="gray")

            # Set title with excitation info
            self._ax.set_title(f"Excitation: {self._excitation} nm - Draw ROI with lasso")
            self._ax.set_xlabel("X (pixels)")
            self._ax.set_ylabel("Y (pixels)")

            # Initialize LassoSelector
            self._selector = LassoSelector(
                self._ax,
                onselect=self._on_select,
                button=1,  # Left mouse button
            )

            plt.tight_layout()
            plt.show()

        # Create instruction label
        instruction_label = Label(
            "Click and drag to draw a lasso selection. "
            "Release to complete. Use widget.get_mask() to retrieve selection."
        )

        return VBox([self._output, instruction_label])

    def _on_select(self, vertices: List[Tuple[float, float]]) -> None:
        """Handle lasso selection completion.

        Args:
            vertices: List of (x, y) coordinate tuples from lasso path.
        """
        if not vertices or len(vertices) < 3:
            return

        # Convert vertices to mask
        self._mask = path_to_mask(vertices, self._spatial_shape)

        # Update display with overlay
        self._update_overlay()

    def _update_overlay(self) -> None:
        """Update the mask overlay on the image."""
        if self._ax is None or self._fig is None:
            return

        # Remove existing overlay if present
        if self._overlay_artist is not None:
            self._overlay_artist.remove()
            self._overlay_artist = None

        if self._mask is not None:
            # Create RGBA overlay (red with alpha)
            overlay = np.zeros((*self._spatial_shape, 4), dtype=np.float32)
            overlay[self._mask, 0] = 1.0  # Red channel
            overlay[self._mask, 3] = 0.3  # Alpha channel

            self._overlay_artist = self._ax.imshow(overlay)

        # Redraw canvas
        self._fig.canvas.draw_idle()

    def get_mask(self) -> Optional[np.ndarray]:
        """Get the current ROI selection mask.

        Returns:
            Boolean 2D array (height, width) where True indicates
            selected pixels, or None if no selection made.
        """
        return self._mask.copy() if self._mask is not None else None

    def clear(self) -> None:
        """Clear the current ROI selection."""
        self._mask = None
        self._update_overlay()
