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

    # Multi-class labeling
    widget.add_class("Lichen")
    widget.add_class("Bark")
    widget.set_class(0)  # Draw as Lichen
    # ... draw regions ...
    gt = widget.to_ground_truth()  # Export for validation
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from .types import GroundTruth, SpectraData


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
        Returns all-False mask if path_vertices is empty or has < 3 points.

    Example:
        vertices = [(10, 20), (50, 20), (50, 80), (10, 80)]
        mask = path_to_mask(vertices, (100, 100))
        print(mask.sum())  # Number of selected pixels
    """
    height, width = shape

    # Handle empty or degenerate paths
    if not path_vertices or len(path_vertices) < 3:
        return np.zeros((height, width), dtype=bool)

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
    matplotlib's LassoSelector or RectangleSelector. Requires %matplotlib widget.

    Supports multi-class labeling: create multiple classes, draw regions for each,
    and export as GroundTruth for validation workflows.

    Attributes:
        fig: Matplotlib figure containing the image.
        ax: Matplotlib axes with the displayed image.
        mask: Current ROI mask (None if no selection).
        bounds: Current ROI bounds as (row_min, row_max, col_min, col_max).

    Example:
        %matplotlib widget
        from spectral_select import SpectraData, ROIWidget

        # Rectangle selection (default)
        data = SpectraData.from_saved("sample.pkl")
        widget = ROIWidget(data, tool="rectangle")
        widget.display()

        # After drawing ROI
        mask = widget.get_mask()
        bounds = widget.get_bounds()  # (row_min, row_max, col_min, col_max)
        print(widget.get_roi_code())  # Copy-pasteable code

        # Multi-class labeling
        widget.add_class("Lichen")
        widget.add_class("Bark")
        widget.set_class(0)  # Select Lichen for drawing
        # ... draw regions ...
        gt = widget.to_ground_truth()  # Export for Validator
    """

    # Class colors for multi-class overlay visualization
    CLASS_COLORS: List[str] = [
        "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"
    ]

    # RGBA values for CLASS_COLORS (for GroundTruth export)
    CLASS_COLORS_RGBA: Dict[str, Tuple[int, int, int, int]] = {
        "red": (255, 0, 0, 255),
        "blue": (0, 0, 255, 255),
        "green": (0, 255, 0, 255),
        "orange": (255, 165, 0, 255),
        "purple": (128, 0, 128, 255),
        "cyan": (0, 255, 255, 255),
        "magenta": (255, 0, 255, 255),
        "yellow": (255, 255, 0, 255),
    }

    def __init__(
        self,
        data: "SpectraData",
        excitation: Optional[float] = None,
        figsize: Tuple[float, float] = (8, 6),
        tool: str = "rectangle",
    ) -> None:
        """Initialize ROI selection widget.

        Args:
            data: SpectraData object containing hyperspectral cube.
            excitation: Excitation wavelength to display. If None, uses first.
            figsize: Figure size as (width, height) in inches.
            tool: Selection tool - "rectangle" (default) or "lasso".
        """
        # Import SpectraData lazily to avoid circular imports
        from .types import SpectraData as SD

        self._data: SD = data
        self._figsize = figsize
        self._tool = tool.lower()

        if self._tool not in ("rectangle", "lasso"):
            raise ValueError(f"tool must be 'rectangle' or 'lasso', got '{tool}'")

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

        # Single-mask state (backward compatibility)
        self._mask: Optional[np.ndarray] = None
        self._bounds: Optional[Tuple[int, int, int, int]] = None  # (row_min, row_max, col_min, col_max)
        self._vertices: Optional[List[Tuple[float, float]]] = None  # For lasso

        # Multi-class ROI state
        self._class_labels: Dict[int, np.ndarray] = {}  # class_id -> binary mask
        self._current_class: int = 0  # Currently selected class for drawing
        self._class_names: List[str] = ["Class 0"]  # Human-readable names

        # GUI state
        self._fig: Optional["Figure"] = None  # noqa: F821
        self._ax: Optional["Axes"] = None  # noqa: F821
        self._selector = None  # LassoSelector or RectangleSelector
        self._image_artist: Optional["AxesImage"] = None  # noqa: F821
        self._overlay_artist: Optional["AxesImage"] = None  # noqa: F821
        self._output: Optional["Output"] = None  # noqa: F821

        # ipywidgets controls (initialized in display())
        self._class_dropdown = None
        self._class_name_input = None

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

    @property
    def bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Current ROI bounds as (row_min, row_max, col_min, col_max)."""
        return self._bounds

    def display(self) -> "VBox":  # noqa: F821
        """Display the widget in a Jupyter notebook.

        Creates an interactive matplotlib figure with selection tool
        for drawing ROI selections, plus class management controls.

        Returns:
            ipywidgets VBox containing the figure, class controls, and instructions.

        Note:
            Requires %matplotlib widget magic to be enabled in notebook.
        """
        # Import ipywidgets lazily
        try:
            from ipywidgets import (
                Button, Dropdown, HBox, Label, Output, Text, VBox
            )
        except ImportError:
            raise ImportError(
                "ipywidgets is required for ROIWidget. "
                "Install with: pip install ipywidgets"
            )

        import matplotlib.pyplot as plt
        from matplotlib.widgets import LassoSelector, RectangleSelector

        # Create output widget to capture matplotlib figure
        self._output = Output()

        with self._output:
            # Create figure and axes
            self._fig, self._ax = plt.subplots(figsize=self._figsize)

            # Create display image and show it
            display_img = create_display_image(self._cube, method="mean")
            self._image_artist = self._ax.imshow(display_img, cmap="gray")

            # Set title with excitation and tool info
            tool_name = "rectangle" if self._tool == "rectangle" else "lasso"
            self._ax.set_title(f"Excitation: {self._excitation} nm - Draw ROI ({tool_name})")
            self._ax.set_xlabel("X (pixels)")
            self._ax.set_ylabel("Y (pixels)")

            # Initialize selector based on tool choice
            if self._tool == "rectangle":
                self._selector = RectangleSelector(
                    self._ax,
                    onselect=self._on_rect_select,
                    useblit=True,
                    button=[1],  # Left mouse button
                    interactive=True,  # Allow resizing after drawing
                    spancoords="data",
                )
            else:
                self._selector = LassoSelector(
                    self._ax,
                    onselect=self._on_lasso_select,
                    button=1,  # Left mouse button
                )

            plt.tight_layout()
            plt.show()

        # Create class management controls
        self._class_dropdown = Dropdown(
            options=[(name, i) for i, name in enumerate(self._class_names)],
            value=self._current_class,
            description="Class:",
            layout={"width": "200px"},
        )
        self._class_dropdown.observe(self._on_class_change, names="value")

        self._class_name_input = Text(
            value=self._class_names[self._current_class],
            placeholder="Class name",
            description="Name:",
            layout={"width": "200px"},
        )
        self._class_name_input.on_submit(self._on_rename_submit)

        add_class_btn = Button(description="Add Class", button_style="success")
        add_class_btn.on_click(self._on_add_class_click)

        clear_current_btn = Button(description="Clear Current", button_style="warning")
        clear_current_btn.on_click(self._on_clear_current_click)

        clear_all_btn = Button(description="Clear All", button_style="danger")
        clear_all_btn.on_click(self._on_clear_all_click)

        # Control row layout
        controls_row = HBox([
            self._class_dropdown,
            self._class_name_input,
            add_class_btn,
            clear_current_btn,
            clear_all_btn,
        ])

        # Create instruction label
        if self._tool == "rectangle":
            instruction_text = (
                "Draw ROIs for the selected class. Use Add Class for multiple classes. "
                "Export with widget.to_ground_truth()."
            )
        else:
            instruction_text = (
                "Draw lasso ROIs for the selected class. Use Add Class for multiple classes. "
                "Export with widget.to_ground_truth()."
            )

        instruction_label = Label(instruction_text)

        return VBox([self._output, controls_row, instruction_label])

    def _on_rect_select(self, eclick, erelease) -> None:
        """Handle rectangle selection completion.

        Args:
            eclick: Mouse click event (start corner).
            erelease: Mouse release event (end corner).
        """
        # Get coordinates (note: x=column, y=row in matplotlib)
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))

        # Ensure proper ordering and clip to image bounds
        height, width = self._spatial_shape
        col_min = max(0, min(x1, x2))
        col_max = min(width, max(x1, x2))
        row_min = max(0, min(y1, y2))
        row_max = min(height, max(y1, y2))

        # Store bounds for get_roi_code() compatibility
        self._bounds = (row_min, row_max, col_min, col_max)

        # Create mask for this selection
        new_mask = np.zeros(self._spatial_shape, dtype=bool)
        new_mask[row_min:row_max, col_min:col_max] = True

        # Add to current class mask using logical OR
        self._add_to_class_mask(self._current_class, new_mask)

        # Update single-mask for backward compatibility
        self._mask = self.get_combined_mask() >= 0  # Any class assigned

        # Update display with overlay
        self._update_overlay()

    def _on_lasso_select(self, vertices: List[Tuple[float, float]]) -> None:
        """Handle lasso selection completion.

        Args:
            vertices: List of (x, y) coordinate tuples from lasso path.
        """
        if not vertices or len(vertices) < 3:
            return

        # Store vertices
        self._vertices = vertices

        # Convert vertices to mask
        new_mask = path_to_mask(vertices, self._spatial_shape)

        # Add to current class mask using logical OR
        self._add_to_class_mask(self._current_class, new_mask)

        # Update single-mask for backward compatibility
        self._mask = self.get_combined_mask() >= 0  # Any class assigned

        # Compute bounding box from mask
        rows, cols = np.where(new_mask)
        if len(rows) > 0:
            self._bounds = (int(rows.min()), int(rows.max()) + 1,
                           int(cols.min()), int(cols.max()) + 1)
        else:
            self._bounds = None

        # Update display with overlay
        self._update_overlay()

    def _update_overlay(self) -> None:
        """Update the mask overlay on the image showing all classes."""
        if self._ax is None or self._fig is None:
            return

        # Remove existing overlay if present
        if self._overlay_artist is not None:
            self._overlay_artist.remove()
            self._overlay_artist = None

        # Create RGBA overlay for all classes
        overlay = np.zeros((*self._spatial_shape, 4), dtype=np.float32)

        # Color name to normalized RGB
        color_to_rgb = {
            "red": (1.0, 0.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "green": (0.0, 1.0, 0.0),
            "orange": (1.0, 0.65, 0.0),
            "purple": (0.5, 0.0, 0.5),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
        }

        # Draw each class with its color
        for class_id, mask in self._class_labels.items():
            if mask is None or not mask.any():
                continue

            # Get color for this class (cycle if > 8 classes)
            color_name = self.CLASS_COLORS[class_id % len(self.CLASS_COLORS)]
            r, g, b = color_to_rgb.get(color_name, (1.0, 0.0, 0.0))

            # Apply color to masked pixels
            overlay[mask, 0] = r
            overlay[mask, 1] = g
            overlay[mask, 2] = b
            overlay[mask, 3] = 0.4  # Alpha

        if overlay[:, :, 3].any():
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
        """Clear the current class's ROI selection."""
        if self._current_class in self._class_labels:
            del self._class_labels[self._current_class]
        self._mask = None
        self._bounds = None
        self._vertices = None
        self._update_overlay()

    def clear_all(self) -> None:
        """Clear all ROI selections for all classes."""
        self._class_labels.clear()
        self._mask = None
        self._bounds = None
        self._vertices = None
        self._update_overlay()

    # =========================================================================
    # Multi-class ROI management
    # =========================================================================

    def _add_to_class_mask(self, class_id: int, mask: np.ndarray) -> None:
        """Add a mask region to a class using logical OR.

        Args:
            class_id: Class ID to add to.
            mask: Boolean mask to add.
        """
        if class_id not in self._class_labels:
            self._class_labels[class_id] = np.zeros(self._spatial_shape, dtype=bool)

        self._class_labels[class_id] = np.logical_or(
            self._class_labels[class_id], mask
        )

    def add_class(self, name: Optional[str] = None) -> int:
        """Add a new class for ROI labeling.

        Args:
            name: Human-readable name for the class. If None, auto-generates.

        Returns:
            The new class ID (0-indexed).
        """
        new_id = len(self._class_names)
        if name is None:
            name = f"Class {new_id}"
        self._class_names.append(name)

        # Update dropdown if it exists
        if self._class_dropdown is not None:
            self._class_dropdown.options = [
                (n, i) for i, n in enumerate(self._class_names)
            ]
            self._class_dropdown.value = new_id

        self._current_class = new_id
        return new_id

    def set_class(self, class_id: int) -> None:
        """Set the current class for drawing.

        Args:
            class_id: Class ID to make active (0-indexed).

        Raises:
            ValueError: If class_id is out of range.
        """
        if class_id < 0 or class_id >= len(self._class_names):
            raise ValueError(
                f"class_id {class_id} out of range [0, {len(self._class_names)})"
            )
        self._current_class = class_id

        # Update dropdown if it exists
        if self._class_dropdown is not None:
            self._class_dropdown.value = class_id

        # Update name input if it exists
        if self._class_name_input is not None:
            self._class_name_input.value = self._class_names[class_id]

    def rename_class(self, class_id: int, name: str) -> None:
        """Rename a class.

        Args:
            class_id: Class ID to rename.
            name: New name for the class.

        Raises:
            ValueError: If class_id is out of range.
        """
        if class_id < 0 or class_id >= len(self._class_names):
            raise ValueError(
                f"class_id {class_id} out of range [0, {len(self._class_names)})"
            )
        self._class_names[class_id] = name

        # Update dropdown if it exists
        if self._class_dropdown is not None:
            self._class_dropdown.options = [
                (n, i) for i, n in enumerate(self._class_names)
            ]

    def get_class_mask(self, class_id: int) -> Optional[np.ndarray]:
        """Get the mask for a specific class.

        Args:
            class_id: Class ID to get mask for.

        Returns:
            Boolean mask for the class, or None if no pixels assigned.
        """
        mask = self._class_labels.get(class_id)
        return mask.copy() if mask is not None else None

    def get_combined_mask(self) -> np.ndarray:
        """Get combined labels array with class IDs per pixel.

        Returns:
            2D integer array where:
            - -1 = background (no ROI selected)
            - 0, 1, 2, ... = class labels
        """
        combined = np.full(self._spatial_shape, -1, dtype=np.int32)

        # Apply class masks in order (later classes overwrite earlier on overlap)
        for class_id in sorted(self._class_labels.keys()):
            mask = self._class_labels[class_id]
            if mask is not None:
                combined[mask] = class_id

        return combined

    @property
    def n_classes(self) -> int:
        """Number of defined classes."""
        return len(self._class_names)

    @property
    def class_names(self) -> List[str]:
        """List of class names."""
        return self._class_names.copy()

    @property
    def current_class(self) -> int:
        """Currently selected class ID."""
        return self._current_class

    # =========================================================================
    # UI event handlers
    # =========================================================================

    def _on_class_change(self, change) -> None:
        """Handle class dropdown selection change."""
        if change["type"] == "change" and change["name"] == "value":
            self._current_class = change["new"]
            if self._class_name_input is not None:
                self._class_name_input.value = self._class_names[self._current_class]

    def _on_rename_submit(self, text_widget) -> None:
        """Handle class name text input submission."""
        self.rename_class(self._current_class, text_widget.value)

    def _on_add_class_click(self, button) -> None:
        """Handle Add Class button click."""
        self.add_class()

    def _on_clear_current_click(self, button) -> None:
        """Handle Clear Current button click."""
        self.clear()

    def _on_clear_all_click(self, button) -> None:
        """Handle Clear All button click."""
        self.clear_all()

    def get_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the ROI bounding box coordinates.

        Returns:
            Tuple of (row_min, row_max, col_min, col_max) as integers,
            or None if no selection made. Can be used directly for array slicing:
            `data[row_min:row_max, col_min:col_max]`
        """
        return self._bounds

    def get_roi_code(self, var_name: str = "roi") -> Optional[str]:
        """Get copy-pasteable Python code to recreate this ROI.

        Returns code that can be pasted into a notebook cell to define
        the ROI coordinates as a tuple or to create a mask directly.

        Args:
            var_name: Variable name to use in the generated code.

        Returns:
            String containing Python code, or None if no selection.

        Example:
            >>> print(widget.get_roi_code())
            # ROI bounds: (row_min, row_max, col_min, col_max)
            roi = (50, 150, 100, 200)

            # To slice data:
            # region = cube[50:150, 100:200, :]

            # To create mask:
            # mask = np.zeros((height, width), dtype=bool)
            # mask[50:150, 100:200] = True
        """
        if self._bounds is None:
            return None

        row_min, row_max, col_min, col_max = self._bounds
        height, width = self._spatial_shape

        code = f'''# ROI bounds: (row_min, row_max, col_min, col_max)
{var_name} = ({row_min}, {row_max}, {col_min}, {col_max})

# To slice data:
# region = cube[{row_min}:{row_max}, {col_min}:{col_max}, :]

# To create mask:
# mask = np.zeros(({height}, {width}), dtype=bool)
# mask[{row_min}:{row_max}, {col_min}:{col_max}] = True'''

        return code

    def print_roi_code(self, var_name: str = "roi") -> None:
        """Print copy-pasteable Python code to recreate this ROI.

        Convenience method that prints the output of get_roi_code().

        Args:
            var_name: Variable name to use in the generated code.
        """
        code = self.get_roi_code(var_name)
        if code is None:
            print("No ROI selection made yet.")
        else:
            print(code)

    # =========================================================================
    # GroundTruth export and file I/O
    # =========================================================================

    def to_ground_truth(self) -> "GroundTruth":
        """Export ROI labels as GroundTruth for validation.

        Creates a GroundTruth instance from the current ROI selections,
        suitable for use with Validator.fit() for clustering evaluation.

        Returns:
            GroundTruth instance with labels array where:
            - -1 = background (no ROI selected)
            - 0, 1, 2... = class labels

        Raises:
            ValueError: If no ROIs have been drawn.

        Example:
            widget.add_class("Lichen")
            widget.add_class("Bark")
            # ... draw ROIs ...
            gt = widget.to_ground_truth()
            validator.fit(predictions, gt)
        """
        # Import here to avoid circular imports
        from .types import GroundTruth

        # Check if any ROIs exist
        has_roi = any(
            mask is not None and mask.any()
            for mask in self._class_labels.values()
        )
        if not has_roi:
            raise ValueError(
                "No ROIs have been drawn. Draw some regions before exporting."
            )

        # Build labels array
        labels = self.get_combined_mask()

        # Build color mapping from CLASS_COLORS_RGBA
        color_mapping: Dict[int, Tuple[int, int, int, int]] = {
            -1: (0, 0, 0, 0)  # Background transparent
        }
        for class_id in range(len(self._class_names)):
            color_name = self.CLASS_COLORS[class_id % len(self.CLASS_COLORS)]
            color_mapping[class_id] = self.CLASS_COLORS_RGBA[color_name]

        return GroundTruth(
            labels=labels,
            color_mapping=color_mapping,
            class_names=self._class_names.copy(),
        )

    @classmethod
    def from_spectra_data(
        cls,
        data: "SpectraData",
        excitation: Optional[float] = None,
        figsize: Tuple[float, float] = (8, 6),
        tool: str = "rectangle",
    ) -> "ROIWidget":
        """Create ROI widget from SpectraData.

        Convenience factory that handles excitation selection and provides
        a cleaner interface for common usage patterns.

        Args:
            data: SpectraData object containing hyperspectral cube.
            excitation: Excitation wavelength to display. If None, uses first.
            figsize: Figure size as (width, height) in inches.
            tool: Selection tool - "rectangle" (default) or "lasso".

        Returns:
            New ROIWidget instance.

        Raises:
            ValueError: If excitation is not in data.excitation_wavelengths.

        Example:
            data = SpectraData.from_pickle("sample.pkl")
            widget = ROIWidget.from_spectra_data(data, excitation=365.0)
            widget.display()
        """
        # Validate excitation if provided
        if excitation is not None:
            if excitation not in data.excitation_wavelengths:
                raise ValueError(
                    f"Excitation {excitation}nm not found. "
                    f"Available: {data.excitation_wavelengths}"
                )

        return cls(data, excitation, figsize, tool)

    def save_mask(self, path: Path) -> None:
        """Save the combined class mask to a PNG file.

        Encodes class IDs as pixel intensity values:
        - 0 = background (class -1)
        - 1 = class 0
        - 2 = class 1
        - etc.

        Args:
            path: Output PNG file path.

        Example:
            widget.save_mask(Path("masks/sample_roi.png"))
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL/Pillow is required for save_mask(). "
                "Install with: pip install Pillow"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get combined mask and shift values so -1 becomes 0
        combined = self.get_combined_mask()
        # Shift: -1 -> 0, 0 -> 1, 1 -> 2, etc.
        mask_shifted = (combined + 1).astype(np.uint8)

        # Save as grayscale PNG
        img = Image.fromarray(mask_shifted, mode="L")
        img.save(path)

    def load_mask(self, path: Path) -> None:
        """Load a class mask from a PNG file.

        Reads pixel intensity values and reconstructs class labels:
        - 0 = background (becomes class -1, no mask)
        - 1 = class 0
        - 2 = class 1
        - etc.

        This clears any existing ROI selections.

        Args:
            path: Input PNG file path.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the mask dimensions don't match the data.

        Example:
            widget.load_mask(Path("masks/sample_roi.png"))
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL/Pillow is required for load_mask(). "
                "Install with: pip install Pillow"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {path}")

        # Load grayscale PNG
        img = Image.open(path).convert("L")
        mask_shifted = np.array(img, dtype=np.int32)

        # Validate dimensions
        if mask_shifted.shape != self._spatial_shape:
            raise ValueError(
                f"Mask shape {mask_shifted.shape} doesn't match "
                f"data shape {self._spatial_shape}"
            )

        # Clear existing ROIs
        self._class_labels.clear()

        # Shift back: 0 -> -1 (background), 1 -> 0, 2 -> 1, etc.
        labels = mask_shifted - 1

        # Populate _class_labels from unique values
        unique_classes = np.unique(labels)
        for class_id in unique_classes:
            if class_id >= 0:  # Skip background (-1)
                self._class_labels[int(class_id)] = (labels == class_id)

                # Ensure class names exist for loaded classes
                while len(self._class_names) <= class_id:
                    self._class_names.append(f"Class {len(self._class_names)}")

        # Update dropdown if it exists
        if self._class_dropdown is not None:
            self._class_dropdown.options = [
                (n, i) for i, n in enumerate(self._class_names)
            ]

        # Update overlay
        self._update_overlay()
