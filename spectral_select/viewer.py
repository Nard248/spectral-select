"""ME-HSI Viewer - Interactive viewer for multi-excitation hyperspectral data.

This module provides a tkinter-based GUI application for viewing and interacting
with hyperspectral imaging data stored in SpectraData format.

Example:
    from spectral_select import ViewerApp, launch_viewer

    # Launch standalone viewer
    launch_viewer()

    # Or with tkinter root
    import tkinter as tk
    root = tk.Tk()
    app = ViewerApp(root)
    root.mainloop()
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def detect_cube_format(cube: np.ndarray) -> Tuple[int, int, int]:
    """Detect the dimension ordering of a 3D hyperspectral cube.

    Identifies which dimension is spectral bands vs spatial dimensions
    based on array shape heuristics.

    Args:
        cube: 3D numpy array representing a hyperspectral cube.

    Returns:
        Tuple of (bands_dim, height_dim, width_dim) indices.

    Raises:
        ValueError: If cube is not 3D.

    Example:
        bands_dim, height_dim, width_dim = detect_cube_format(cube)
        n_bands = cube.shape[bands_dim]
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

    d0, d1, d2 = cube.shape

    # Heuristic: spectral dimension is usually smallest
    # SpectraData format: (height, width, n_bands) - bands is dim 2
    # Some formats: (n_bands, height, width) - bands is dim 0

    if d2 < d0 and d2 < d1:
        # Last dimension smallest -> (height, width, bands)
        return (2, 0, 1)
    elif d0 < d1 and d0 < d2:
        # First dimension smallest -> (bands, height, width)
        return (0, 1, 2)
    else:
        # Default to SpectraData format (height, width, bands)
        return (2, 0, 1)


def get_pixel_value(
    cube: np.ndarray,
    x: int,
    y: int,
    band: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Get pixel value(s) from a hyperspectral cube.

    Args:
        cube: 3D hyperspectral cube in (height, width, bands) format.
        x: X coordinate (column).
        y: Y coordinate (row).
        band: Optional specific band index. If None, returns all bands.

    Returns:
        Single float if band specified, otherwise 1D array of all band values.

    Raises:
        IndexError: If coordinates are out of bounds.
    """
    height, width = cube.shape[0], cube.shape[1]

    if not (0 <= y < height and 0 <= x < width):
        raise IndexError(f"Coordinates ({x}, {y}) out of bounds for shape ({height}, {width})")

    if band is not None:
        return float(cube[y, x, band])
    return cube[y, x, :]


def create_rgb_image(
    cube: np.ndarray,
    method: str = "rgb",
    percentile: float = 98.0,
) -> np.ndarray:
    """Create an RGB visualization from a hyperspectral cube.

    Converts a 3D hyperspectral cube to an RGB image for display using
    various projection methods.

    Args:
        cube: 3D array in (height, width, bands) format.
        method: Visualization method:
            - "rgb": Use three bands as R, G, B channels
            - "mean": Mean projection across bands (grayscale)
            - "max": Maximum projection across bands (grayscale)
        percentile: Percentile for contrast normalization (default 98).

    Returns:
        RGB image as (height, width, 3) float array in [0, 1] range.

    Example:
        rgb = create_rgb_image(cube, method="rgb", percentile=99)
        plt.imshow(rgb)
    """
    # Handle NaN values
    cube = np.nan_to_num(cube, nan=0.0)

    height, width, n_bands = cube.shape

    if method == "rgb" and n_bands >= 3:
        # Select three representative bands for RGB
        r_idx = int(n_bands * 0.2)
        g_idx = int(n_bands * 0.5)
        b_idx = int(n_bands * 0.8)

        r_band = cube[:, :, r_idx]
        g_band = cube[:, :, g_idx]
        b_band = cube[:, :, b_idx]

        # Normalize each channel independently
        def normalize_band(band: np.ndarray) -> np.ndarray:
            pval = np.percentile(band[band > 0], percentile) if np.any(band > 0) else 1.0
            if pval == 0:
                pval = 1.0
            return np.clip(band / pval, 0, 1)

        r_norm = normalize_band(r_band)
        g_norm = normalize_band(g_band)
        b_norm = normalize_band(b_band)

        rgb = np.stack([r_norm, g_norm, b_norm], axis=2)

    elif method == "max":
        # Maximum intensity projection
        proj = np.max(cube, axis=2)
        pval = np.percentile(proj[proj > 0], percentile) if np.any(proj > 0) else 1.0
        if pval == 0:
            pval = 1.0
        proj_norm = np.clip(proj / pval, 0, 1)
        rgb = np.stack([proj_norm, proj_norm, proj_norm], axis=2)

    else:  # method == "mean" or fallback
        # Mean intensity projection
        proj = np.mean(cube, axis=2)
        pval = np.percentile(proj[proj > 0], percentile) if np.any(proj > 0) else 1.0
        if pval == 0:
            pval = 1.0
        proj_norm = np.clip(proj / pval, 0, 1)
        rgb = np.stack([proj_norm, proj_norm, proj_norm], axis=2)

    return rgb.astype(np.float32)


class ViewerApp:
    """Interactive viewer for multi-excitation hyperspectral data.

    Provides a tkinter-based GUI for loading, viewing, and interacting with
    hyperspectral data stored in SpectraData format (pkl files).

    Attributes:
        root: The tkinter root window.
        spectra_data: Currently loaded SpectraData instance.
        current_excitation: Selected excitation wavelength.
        current_image: Currently displayed RGB image.

    Example:
        import tkinter as tk
        root = tk.Tk()
        app = ViewerApp(root)
        root.mainloop()
    """

    def __init__(self, root: tk.Tk):
        """Initialize the ViewerApp.

        Args:
            root: The tkinter root window.
        """
        self.root = root
        self.root.title("ME-HSI Viewer")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # State variables
        self._spectra_data: Optional[Any] = None  # SpectraData
        self._current_excitation: Optional[float] = None
        self._current_image: Optional[np.ndarray] = None
        self._current_cube: Optional[np.ndarray] = None
        self._last_directory: Optional[str] = None

        # Display settings
        self._auto_contrast = tk.BooleanVar(value=True)
        self._rgb_method = tk.StringVar(value="rgb")
        self._min_val = tk.DoubleVar(value=0.0)
        self._max_val = tk.DoubleVar(value=100.0)

        # Build the UI
        self._create_widgets()
        self._bind_events()

        # Initial state
        self._update_status("Ready")
        self._toggle_controls(enabled=False)

        logger.info("ViewerApp initialized")

    @property
    def spectra_data(self) -> Optional[Any]:
        """Currently loaded SpectraData instance."""
        return self._spectra_data

    @property
    def current_excitation(self) -> Optional[float]:
        """Currently selected excitation wavelength."""
        return self._current_excitation

    @property
    def current_image(self) -> Optional[np.ndarray]:
        """Currently displayed RGB image."""
        return self._current_image

    def _create_widgets(self) -> None:
        """Create all UI widgets and layout."""
        # Main container with toolbar at top
        self._toolbar_frame = ttk.Frame(self.root)
        self._toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Toolbar buttons
        ttk.Button(self._toolbar_frame, text="Open", command=self._on_open).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(self._toolbar_frame, text="Save Mask", command=self._on_save_mask).pack(
            side=tk.LEFT, padx=2
        )

        # Status label on right side of toolbar
        self._status_label = ttk.Label(self._toolbar_frame, text="Ready")
        self._status_label.pack(side=tk.RIGHT, padx=10)

        # Main content area
        self._main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self._main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: Control panel (fixed width)
        self._control_panel = ttk.Frame(self._main_frame, width=250)
        self._main_frame.add(self._control_panel, weight=0)

        # Right: Canvas area
        self._canvas_frame = ttk.Frame(self._main_frame)
        self._main_frame.add(self._canvas_frame, weight=1)

        # Build control panel sections
        self._create_data_section()
        self._create_excitation_section()
        self._create_display_section()
        self._create_tools_section()

        # Build canvas area
        self._create_canvas()

        # Info bar at bottom of canvas
        self._create_info_bar()

    def _create_data_section(self) -> None:
        """Create the Data control section."""
        data_frame = ttk.LabelFrame(self._control_panel, text="Data")
        data_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(data_frame, text="Open File...", command=self._on_open).pack(
            fill=tk.X, padx=5, pady=2
        )

        self._file_label = ttk.Label(data_frame, text="No file loaded", wraplength=230)
        self._file_label.pack(fill=tk.X, padx=5, pady=2)

    def _create_excitation_section(self) -> None:
        """Create the Excitation wavelength selection section."""
        ex_frame = ttk.LabelFrame(self._control_panel, text="Excitation")
        ex_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(ex_frame, text="Wavelength (nm):").pack(anchor=tk.W, padx=5, pady=2)

        self._excitation_var = tk.StringVar()
        self._excitation_combo = ttk.Combobox(
            ex_frame,
            textvariable=self._excitation_var,
            state="readonly",
            width=15,
        )
        self._excitation_combo.pack(fill=tk.X, padx=5, pady=2)
        self._excitation_combo.bind("<<ComboboxSelected>>", self._on_excitation_changed)

    def _create_display_section(self) -> None:
        """Create the Display controls section."""
        display_frame = ttk.LabelFrame(self._control_panel, text="Display")
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        # Auto-contrast checkbox
        ttk.Checkbutton(
            display_frame,
            text="Auto Contrast",
            variable=self._auto_contrast,
            command=self._on_display_changed,
        ).pack(anchor=tk.W, padx=5, pady=2)

        # Min/Max manual controls
        manual_frame = ttk.Frame(display_frame)
        manual_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(manual_frame, text="Min:").pack(side=tk.LEFT)
        self._min_entry = ttk.Entry(manual_frame, textvariable=self._min_val, width=6)
        self._min_entry.pack(side=tk.LEFT, padx=2)

        ttk.Label(manual_frame, text="Max:").pack(side=tk.LEFT)
        self._max_entry = ttk.Entry(manual_frame, textvariable=self._max_val, width=6)
        self._max_entry.pack(side=tk.LEFT, padx=2)

        # RGB method selection
        ttk.Label(display_frame, text="RGB Method:").pack(anchor=tk.W, padx=5, pady=2)

        for method, label in [("rgb", "RGB (3-band)"), ("mean", "Mean projection"), ("max", "Max projection")]:
            ttk.Radiobutton(
                display_frame,
                text=label,
                variable=self._rgb_method,
                value=method,
                command=self._on_display_changed,
            ).pack(anchor=tk.W, padx=15)

        # Apply button for manual contrast
        ttk.Button(display_frame, text="Apply", command=self._on_display_changed).pack(
            fill=tk.X, padx=5, pady=5
        )

    def _create_tools_section(self) -> None:
        """Create the Tools section (placeholder for masking tools)."""
        tools_frame = ttk.LabelFrame(self._control_panel, text="Tools")
        tools_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(tools_frame, text="(Masking tools coming soon)").pack(padx=5, pady=10)

    def _create_canvas(self) -> None:
        """Create the matplotlib canvas for image display."""
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure
        except ImportError:
            logger.error("matplotlib not available - canvas creation failed")
            ttk.Label(self._canvas_frame, text="matplotlib required for display").pack(
                expand=True
            )
            return

        # Create figure and axes
        self._figure = Figure(figsize=(8, 6), dpi=100)
        self._ax = self._figure.add_subplot(111)
        self._figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Create canvas
        self._canvas = FigureCanvasTkAgg(self._figure, master=self._canvas_frame)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar
        toolbar_frame = ttk.Frame(self._canvas_frame)
        toolbar_frame.pack(fill=tk.X)
        self._nav_toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._nav_toolbar.update()

        # Initial message
        self._ax.text(
            0.5, 0.5,
            "Open a SpectraData file to begin",
            ha="center", va="center",
            fontsize=14,
            transform=self._ax.transAxes,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw()

        # Store image object for updates
        self._img_display = None

    def _create_info_bar(self) -> None:
        """Create the info bar showing cursor position and pixel value."""
        info_frame = ttk.Frame(self._canvas_frame)
        info_frame.pack(fill=tk.X, pady=2)

        # Position label
        ttk.Label(info_frame, text="Position:").pack(side=tk.LEFT, padx=5)
        self._position_label = ttk.Label(info_frame, text="---, ---")
        self._position_label.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(info_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Value label
        ttk.Label(info_frame, text="Value:").pack(side=tk.LEFT, padx=5)
        self._value_label = ttk.Label(info_frame, text="---")
        self._value_label.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(info_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Excitation/band info
        self._band_info_label = ttk.Label(info_frame, text="")
        self._band_info_label.pack(side=tk.LEFT, padx=5)

    def _bind_events(self) -> None:
        """Bind keyboard and mouse events."""
        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self._on_open())
        self.root.bind("<r>", lambda e: self._reset_view())
        self.root.bind("<Escape>", lambda e: self._on_cancel())

        # Canvas mouse events (if canvas exists)
        if hasattr(self, "_canvas"):
            self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _toggle_controls(self, enabled: bool = True) -> None:
        """Enable or disable controls based on data loading state."""
        state = "normal" if enabled else "disabled"

        self._excitation_combo.config(state="readonly" if enabled else "disabled")

        # Min/max entries
        self._min_entry.config(state=state)
        self._max_entry.config(state=state)

    def _update_status(self, message: str) -> None:
        """Update the status bar message."""
        self._status_label.config(text=message)
        self.root.update_idletasks()

    def _on_open(self) -> None:
        """Handle Open file action."""
        initial_dir = self._last_directory or str(Path.home())

        file_path = filedialog.askopenfilename(
            title="Open SpectraData File",
            initialdir=initial_dir,
            filetypes=[
                ("SpectraData files", "*.pkl"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        self._last_directory = str(Path(file_path).parent)
        self._load_file(file_path)

    def _load_file(self, file_path: str) -> None:
        """Load a SpectraData file."""
        path = Path(file_path)
        self._update_status(f"Loading {path.name}...")

        try:
            # Import SpectraData lazily to avoid circular imports
            from .types import SpectraData

            self._spectra_data = SpectraData.from_pickle(path)

            # Update UI
            self._file_label.config(text=path.name)
            self.root.title(f"ME-HSI Viewer - {path.name}")

            # Populate excitation selector
            wavelengths = self._spectra_data.excitation_wavelengths
            self._excitation_combo["values"] = [str(w) for w in wavelengths]

            # Select first excitation
            if wavelengths:
                self._excitation_var.set(str(wavelengths[0]))
                self._current_excitation = wavelengths[0]

            # Enable controls
            self._toggle_controls(enabled=True)

            # Update display
            self._update_display()

            # Update status
            h, w = self._spectra_data.spatial_shape
            n_ex = self._spectra_data.n_excitations
            self._update_status(f"Loaded: {path.name} ({n_ex} excitations, {h}x{w})")

            logger.info(f"Loaded SpectraData from {path}")

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            messagebox.showerror("Error", f"Failed to load file:\n\n{e}")
            self._update_status("Error loading file")

    def _on_excitation_changed(self, event: Optional[Any] = None) -> None:
        """Handle excitation wavelength selection change."""
        try:
            ex_str = self._excitation_var.get()
            if ex_str:
                self._current_excitation = float(ex_str)
                self._update_display()
        except ValueError:
            pass

    def _on_display_changed(self) -> None:
        """Handle display settings change."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the image display with current settings."""
        if self._spectra_data is None or self._current_excitation is None:
            return

        try:
            # Get current excitation data
            ex_data = self._spectra_data.get_excitation(self._current_excitation)
            self._current_cube = ex_data.cube

            # Create RGB image
            method = self._rgb_method.get()
            percentile = 98 if self._auto_contrast.get() else 100

            self._current_image = create_rgb_image(
                self._current_cube,
                method=method,
                percentile=percentile,
            )

            # Apply manual contrast if not auto
            if not self._auto_contrast.get():
                min_val = self._min_val.get() / 100.0
                max_val = self._max_val.get() / 100.0
                if max_val > min_val:
                    self._current_image = np.clip(
                        (self._current_image - min_val) / (max_val - min_val),
                        0, 1,
                    )

            # Update display
            self._ax.clear()
            self._img_display = self._ax.imshow(
                self._current_image,
                aspect="equal",
                interpolation="nearest",
            )
            self._ax.set_title(f"Excitation: {self._current_excitation} nm")
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._canvas.draw()

            # Update band info
            n_bands = ex_data.n_bands
            self._band_info_label.config(
                text=f"Ex: {self._current_excitation}nm | {n_bands} bands"
            )

        except Exception as e:
            logger.error(f"Display update failed: {e}")
            self._update_status(f"Display error: {e}")

    def _on_mouse_move(self, event: Any) -> None:
        """Handle mouse movement over the canvas."""
        if event.inaxes != self._ax or self._current_cube is None:
            self._position_label.config(text="---, ---")
            self._value_label.config(text="---")
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        x_int, y_int = int(x), int(y)
        height, width = self._current_cube.shape[0], self._current_cube.shape[1]

        if 0 <= x_int < width and 0 <= y_int < height:
            # Update position
            self._position_label.config(text=f"X: {x_int}, Y: {y_int}")

            # Get mean pixel value across bands
            pixel_vals = self._current_cube[y_int, x_int, :]
            mean_val = np.mean(pixel_vals)

            # Format value (scientific notation for small values)
            if abs(mean_val) < 0.001 and mean_val != 0:
                val_str = f"{mean_val:.2e}"
            else:
                val_str = f"{mean_val:.4f}"

            self._value_label.config(text=val_str)
        else:
            self._position_label.config(text="Outside image")
            self._value_label.config(text="---")

    def _reset_view(self) -> None:
        """Reset the view to fit the image."""
        if hasattr(self, "_nav_toolbar"):
            self._nav_toolbar.home()

    def _on_cancel(self) -> None:
        """Handle Escape key - cancel current operation."""
        # Placeholder for future mask drawing cancellation
        pass

    def _on_save_mask(self) -> None:
        """Handle Save Mask action."""
        messagebox.showinfo("Info", "Mask saving will be implemented in a future phase.")


def launch_viewer() -> None:
    """Launch the ME-HSI Viewer application.

    Convenience function to start the viewer in standalone mode.

    Example:
        from spectral_select import launch_viewer
        launch_viewer()
    """
    root = tk.Tk()
    _ = ViewerApp(root)
    root.mainloop()
