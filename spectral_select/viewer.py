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

import csv
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .types import SpectraData

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


def compose_false_color(
    cube: np.ndarray,
    r_band: int,
    g_band: int,
    b_band: int,
    percentile: float = 98.0,
) -> np.ndarray:
    """Create a false color RGB image from specified bands.

    Allows arbitrary band-to-channel assignment for custom RGB visualization.
    Each channel is independently normalized using percentile-based contrast.

    Args:
        cube: 3D array in (height, width, bands) format.
        r_band: Band index to map to red channel.
        g_band: Band index to map to green channel.
        b_band: Band index to map to blue channel.
        percentile: Percentile for contrast normalization (default 98).

    Returns:
        RGB image as (height, width, 3) float array in [0, 1] range.

    Raises:
        IndexError: If any band index is out of range.

    Example:
        # Create false color with bands 10, 25, 40 as RGB
        rgb = compose_false_color(cube, r_band=10, g_band=25, b_band=40)
        plt.imshow(rgb)
    """
    # Handle NaN values
    cube = np.nan_to_num(cube, nan=0.0)

    height, width, n_bands = cube.shape

    # Validate band indices
    for name, idx in [("r_band", r_band), ("g_band", g_band), ("b_band", b_band)]:
        if not (0 <= idx < n_bands):
            raise IndexError(f"{name}={idx} out of range for cube with {n_bands} bands")

    # Extract bands
    r_data = cube[:, :, r_band]
    g_data = cube[:, :, g_band]
    b_data = cube[:, :, b_band]

    # Normalize each channel independently
    def normalize_band(band: np.ndarray) -> np.ndarray:
        if np.any(band > 0):
            pval = np.percentile(band[band > 0], percentile)
        else:
            pval = 1.0
        if pval == 0:
            pval = 1.0
        return np.clip(band / pval, 0, 1)

    r_norm = normalize_band(r_data)
    g_norm = normalize_band(g_data)
    b_norm = normalize_band(b_data)

    rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
    return rgb.astype(np.float32)


# False color presets: name -> (r_fraction, g_fraction, b_fraction)
# Fractions are converted to band indices based on total bands
FALSE_COLOR_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "Default (20/50/80%)": (0.2, 0.5, 0.8),
    "Lower Spectrum": (0.1, 0.2, 0.3),
    "Upper Spectrum": (0.7, 0.8, 0.9),
    "Wide Spread": (0.0, 0.5, 1.0),
}

# Maximum number of spectra to compare in multi-point mode
MAX_COMPARE_SPECTRA = 5

# Colors for multi-point spectrum comparison
SPECTRUM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def extract_spectrum(cube: np.ndarray, x: int, y: int) -> np.ndarray:
    """Extract spectrum at pixel (x, y) from cube.

    Retrieves all spectral band values for a single pixel location.

    Args:
        cube: 3D hyperspectral cube in (height, width, bands) format.
        x: X coordinate (column).
        y: Y coordinate (row).

    Returns:
        1D array of spectral values for the pixel.

    Raises:
        IndexError: If coordinates are out of bounds.

    Example:
        spec = extract_spectrum(cube, 50, 50)
        print(f"Spectrum shape: {spec.shape}")
    """
    height, width = cube.shape[0], cube.shape[1]

    if not (0 <= y < height and 0 <= x < width):
        raise IndexError(f"Coordinates ({x}, {y}) out of bounds for shape ({height}, {width})")

    return cube[y, x, :].copy()


def extract_multi_excitation_spectrum(
    spectra_data: "SpectraData", x: int, y: int
) -> Dict[float, np.ndarray]:
    """Extract spectra at pixel across all excitations.

    Retrieves the spectrum at a given pixel for each excitation wavelength,
    useful for analyzing the full 4D response at a location.

    Args:
        spectra_data: SpectraData instance with multiple excitations.
        x: X coordinate (column).
        y: Y coordinate (row).

    Returns:
        Dictionary mapping excitation wavelength to spectrum array.

    Example:
        spectra = extract_multi_excitation_spectrum(data, 50, 50)
        for ex_nm, spec in spectra.items():
            print(f"Ex {ex_nm}nm: {spec.shape}")
    """
    result: Dict[float, np.ndarray] = {}

    for ex_nm in spectra_data.excitation_wavelengths:
        ex_data = spectra_data.get_excitation(ex_nm)
        result[ex_nm] = extract_spectrum(ex_data.cube, x, y)

    return result


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

        # Band browser state
        self._current_band = tk.IntVar(value=0)
        self._display_mode = tk.StringVar(value="composite")  # "composite" or "single"
        self._colormap = tk.StringVar(value="viridis")
        self._animation_running = False
        self._animation_speed = tk.IntVar(value=100)  # milliseconds
        self._animation_loop = tk.BooleanVar(value=True)
        self._animation_id: Optional[str] = None
        self._emission_wavelengths: List[float] = []

        # False color composer state
        self._false_color_enabled = tk.BooleanVar(value=False)
        self._false_color_live = tk.BooleanVar(value=False)
        self._fc_r_band = tk.IntVar(value=0)
        self._fc_g_band = tk.IntVar(value=0)
        self._fc_b_band = tk.IntVar(value=0)
        self._fc_preset = tk.StringVar(value="Default (20/50/80%)")

        # Zoom state
        self._zoom_level = 1.0  # 1.0 = 100%
        self._zoom_factor = 1.2  # Zoom step multiplier
        self._image_shape: Optional[Tuple[int, int]] = None  # (height, width)

        # Spectrum panel state
        self._spectrum_compare_mode = tk.BooleanVar(value=False)
        self._spectrum_auto_scale = tk.BooleanVar(value=True)
        self._spectrum_traces: List[Tuple[int, int, np.ndarray]] = []  # [(x, y, spectrum), ...]
        self._clicked_pixel: Optional[Tuple[int, int]] = None  # Most recent click

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

        # Center: Canvas area
        self._canvas_frame = ttk.Frame(self._main_frame)
        self._main_frame.add(self._canvas_frame, weight=1)

        # Right: Analysis panels (spectrum, histogram, stats)
        self._right_panel = ttk.Frame(self._main_frame, width=300)
        self._main_frame.add(self._right_panel, weight=0)

        # Build control panel sections
        self._create_data_section()
        self._create_excitation_section()
        self._create_display_section()
        self._create_band_browser_section()
        self._create_false_color_section()
        self._create_tools_section()

        # Build canvas area
        self._create_canvas()

        # Info bar at bottom of canvas
        self._create_info_bar()

        # Build right panel sections
        self._create_spectrum_panel()

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

    def _create_band_browser_section(self) -> None:
        """Create the Band Browser section for single-band viewing."""
        browser_frame = ttk.LabelFrame(self._control_panel, text="Band Browser")
        browser_frame.pack(fill=tk.X, padx=5, pady=5)

        # Display mode selection (Composite vs Single Band)
        mode_frame = ttk.Frame(browser_frame)
        mode_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Radiobutton(
            mode_frame,
            text="Composite",
            variable=self._display_mode,
            value="composite",
            command=self._on_display_mode_changed,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Radiobutton(
            mode_frame,
            text="Single Band",
            variable=self._display_mode,
            value="single",
            command=self._on_display_mode_changed,
        ).pack(side=tk.LEFT, padx=2)

        # Band slider
        slider_frame = ttk.Frame(browser_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(slider_frame, text="Band:").pack(side=tk.LEFT)

        self._band_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=0,
            variable=self._current_band,
            orient=tk.HORIZONTAL,
            command=self._on_band_slider_changed,
        )
        self._band_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Band spinbox for direct entry
        self._band_spinbox = ttk.Spinbox(
            slider_frame,
            from_=0,
            to=0,
            width=5,
            textvariable=self._current_band,
            command=self._on_band_spinbox_changed,
        )
        self._band_spinbox.pack(side=tk.LEFT, padx=2)
        self._band_spinbox.bind("<Return>", lambda e: self._on_band_spinbox_changed())

        # Wavelength label
        self._wavelength_label = ttk.Label(browser_frame, text="Wavelength: -- nm")
        self._wavelength_label.pack(fill=tk.X, padx=5, pady=2)

        # Colormap selector
        cmap_frame = ttk.Frame(browser_frame)
        cmap_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(cmap_frame, text="Colormap:").pack(side=tk.LEFT)

        self._cmap_combo = ttk.Combobox(
            cmap_frame,
            textvariable=self._colormap,
            values=["viridis", "gray", "hot", "cool", "plasma", "inferno"],
            state="readonly",
            width=10,
        )
        self._cmap_combo.pack(side=tk.LEFT, padx=5)
        self._cmap_combo.bind("<<ComboboxSelected>>", lambda e: self._update_display())

        # Animation controls
        anim_frame = ttk.LabelFrame(browser_frame, text="Animation")
        anim_frame.pack(fill=tk.X, padx=5, pady=5)

        # Play/Stop buttons
        btn_frame = ttk.Frame(anim_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=2)

        self._play_btn = ttk.Button(btn_frame, text="▶ Play", command=self._on_play_animation)
        self._play_btn.pack(side=tk.LEFT, padx=2)

        self._stop_btn = ttk.Button(btn_frame, text="■ Stop", command=self._on_stop_animation, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=2)

        # Loop checkbox
        ttk.Checkbutton(
            btn_frame,
            text="Loop",
            variable=self._animation_loop,
        ).pack(side=tk.LEFT, padx=5)

        # Speed slider
        speed_frame = ttk.Frame(anim_frame)
        speed_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)

        self._speed_slider = ttk.Scale(
            speed_frame,
            from_=500,  # Slow
            to=50,  # Fast (reversed for intuitive control)
            variable=self._animation_speed,
            orient=tk.HORIZONTAL,
        )
        self._speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self._speed_label = ttk.Label(speed_frame, text="100ms")
        self._speed_label.pack(side=tk.LEFT, padx=2)

        # Update speed label when slider moves
        self._animation_speed.trace_add("write", self._update_speed_label)

    def _create_false_color_section(self) -> None:
        """Create the False Color Composer section."""
        fc_frame = ttk.LabelFrame(self._control_panel, text="False Color")
        fc_frame.pack(fill=tk.X, padx=5, pady=5)

        # Enable false color checkbox
        enable_frame = ttk.Frame(fc_frame)
        enable_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Checkbutton(
            enable_frame,
            text="Enable False Color",
            variable=self._false_color_enabled,
            command=self._on_false_color_toggled,
        ).pack(side=tk.LEFT)

        ttk.Checkbutton(
            enable_frame,
            text="Live Preview",
            variable=self._false_color_live,
        ).pack(side=tk.LEFT, padx=10)

        # Preset dropdown
        preset_frame = ttk.Frame(fc_frame)
        preset_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)

        self._fc_preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self._fc_preset,
            values=list(FALSE_COLOR_PRESETS.keys()) + ["Custom"],
            state="readonly",
            width=20,
        )
        self._fc_preset_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self._fc_preset_combo.bind("<<ComboboxSelected>>", self._on_fc_preset_changed)

        # Channel assignment rows
        self._fc_r_combo: Optional[ttk.Combobox] = None
        self._fc_g_combo: Optional[ttk.Combobox] = None
        self._fc_b_combo: Optional[ttk.Combobox] = None

        # R channel
        r_frame = ttk.Frame(fc_frame)
        r_frame.pack(fill=tk.X, padx=5, pady=1)

        ttk.Label(r_frame, text="R:", width=3, foreground="red").pack(side=tk.LEFT)
        self._fc_r_combo = ttk.Combobox(r_frame, state="readonly", width=20)
        self._fc_r_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._fc_r_combo.bind("<<ComboboxSelected>>", self._on_fc_channel_changed)
        self._fc_r_label = ttk.Label(r_frame, text="", width=10)
        self._fc_r_label.pack(side=tk.LEFT, padx=2)

        # G channel
        g_frame = ttk.Frame(fc_frame)
        g_frame.pack(fill=tk.X, padx=5, pady=1)

        ttk.Label(g_frame, text="G:", width=3, foreground="green").pack(side=tk.LEFT)
        self._fc_g_combo = ttk.Combobox(g_frame, state="readonly", width=20)
        self._fc_g_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._fc_g_combo.bind("<<ComboboxSelected>>", self._on_fc_channel_changed)
        self._fc_g_label = ttk.Label(g_frame, text="", width=10)
        self._fc_g_label.pack(side=tk.LEFT, padx=2)

        # B channel
        b_frame = ttk.Frame(fc_frame)
        b_frame.pack(fill=tk.X, padx=5, pady=1)

        ttk.Label(b_frame, text="B:", width=3, foreground="blue").pack(side=tk.LEFT)
        self._fc_b_combo = ttk.Combobox(b_frame, state="readonly", width=20)
        self._fc_b_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._fc_b_combo.bind("<<ComboboxSelected>>", self._on_fc_channel_changed)
        self._fc_b_label = ttk.Label(b_frame, text="", width=10)
        self._fc_b_label.pack(side=tk.LEFT, padx=2)

        # Buttons
        btn_frame = ttk.Frame(fc_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="Apply", command=self._on_fc_apply).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset", command=self._on_fc_reset).pack(side=tk.LEFT, padx=2)

    def _create_tools_section(self) -> None:
        """Create the Tools section (placeholder for masking tools)."""
        tools_frame = ttk.LabelFrame(self._control_panel, text="Tools")
        tools_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(tools_frame, text="(Masking tools coming soon)").pack(padx=5, pady=10)

    def _create_spectrum_panel(self) -> None:
        """Create the spectrum panel for click-to-plot visualization."""
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            logger.error("matplotlib not available - spectrum panel creation failed")
            return

        # Create spectrum frame in the right panel area
        self._spectrum_frame = ttk.LabelFrame(self._right_panel, text="Spectrum")
        self._spectrum_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Controls at top
        controls_frame = ttk.Frame(self._spectrum_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=2)

        # Compare mode checkbox
        ttk.Checkbutton(
            controls_frame,
            text="Compare Mode",
            variable=self._spectrum_compare_mode,
        ).pack(side=tk.LEFT, padx=2)

        # Auto scale checkbox
        ttk.Checkbutton(
            controls_frame,
            text="Auto Scale",
            variable=self._spectrum_auto_scale,
        ).pack(side=tk.LEFT, padx=5)

        # Clear button
        ttk.Button(
            controls_frame,
            text="Clear",
            command=self._clear_spectrum_traces,
        ).pack(side=tk.LEFT, padx=5)

        # Export button
        ttk.Button(
            controls_frame,
            text="Export CSV",
            command=self._export_spectrum_csv,
        ).pack(side=tk.LEFT, padx=5)

        # Create matplotlib figure for spectrum
        self._spectrum_figure = Figure(figsize=(4, 2), dpi=100)
        self._spectrum_ax = self._spectrum_figure.add_subplot(111)
        self._spectrum_figure.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)

        # Create canvas
        self._spectrum_canvas = FigureCanvasTkAgg(
            self._spectrum_figure, master=self._spectrum_frame
        )
        self._spectrum_canvas_widget = self._spectrum_canvas.get_tk_widget()
        self._spectrum_canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial empty state
        self._spectrum_ax.text(
            0.5, 0.5,
            "Click on image to plot spectrum",
            ha="center", va="center",
            transform=self._spectrum_ax.transAxes,
            fontsize=10,
            color="gray",
        )
        self._spectrum_ax.set_xticks([])
        self._spectrum_ax.set_yticks([])
        self._spectrum_canvas.draw()

    def _clear_spectrum_traces(self) -> None:
        """Clear all spectrum traces from compare mode."""
        self._spectrum_traces = []
        self._clicked_pixel = None
        self._update_spectrum_plot()

    def _export_spectrum_csv(self) -> None:
        """Export current spectrum data to CSV file."""
        if not self._spectrum_traces and self._clicked_pixel is None:
            messagebox.showinfo("Info", "No spectrum data to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Spectrum Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="spectrum_data.csv",
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Header row
                header = ["Wavelength_nm"]
                if self._spectrum_traces:
                    for x, y, _ in self._spectrum_traces:
                        header.append(f"Pixel_({x},{y})")
                elif self._clicked_pixel:
                    x, y = self._clicked_pixel
                    header.append(f"Pixel_({x},{y})")
                writer.writerow(header)

                # Data rows
                if self._emission_wavelengths:
                    wavelengths = self._emission_wavelengths
                else:
                    # Use band indices if wavelengths not available
                    if self._spectrum_traces:
                        wavelengths = list(range(len(self._spectrum_traces[0][2])))
                    elif self._clicked_pixel and self._current_cube is not None:
                        wavelengths = list(range(self._current_cube.shape[2]))
                    else:
                        wavelengths = []

                for i, wl in enumerate(wavelengths):
                    row = [wl]
                    if self._spectrum_traces:
                        for _, _, spec in self._spectrum_traces:
                            row.append(spec[i] if i < len(spec) else "")
                    elif self._clicked_pixel and self._current_cube is not None:
                        x, y = self._clicked_pixel
                        row.append(self._current_cube[y, x, i])
                    writer.writerow(row)

            self._update_status(f"Exported spectrum to {Path(file_path).name}")
            logger.info(f"Spectrum data exported to {file_path}")

        except Exception as e:
            logger.error(f"Failed to export spectrum: {e}")
            messagebox.showerror("Error", f"Failed to export spectrum:\n\n{e}")

    def _update_spectrum_plot(self) -> None:
        """Update the spectrum plot with current data."""
        if not hasattr(self, "_spectrum_ax"):
            return

        self._spectrum_ax.clear()

        # Check if we have data to plot
        has_data = False

        if self._spectrum_compare_mode.get() and self._spectrum_traces:
            # Multi-trace compare mode
            for i, (x, y, spectrum) in enumerate(self._spectrum_traces):
                color = SPECTRUM_COLORS[i % len(SPECTRUM_COLORS)]
                if self._emission_wavelengths:
                    self._spectrum_ax.plot(
                        self._emission_wavelengths[:len(spectrum)],
                        spectrum,
                        color=color,
                        label=f"({x}, {y})",
                        linewidth=1,
                        marker=".",
                        markersize=3,
                    )
                else:
                    self._spectrum_ax.plot(
                        spectrum,
                        color=color,
                        label=f"({x}, {y})",
                        linewidth=1,
                        marker=".",
                        markersize=3,
                    )
                has_data = True

            if has_data:
                self._spectrum_ax.legend(loc="best", fontsize=8)

        elif self._clicked_pixel and self._current_cube is not None:
            # Single spectrum mode
            x, y = self._clicked_pixel
            try:
                spectrum = extract_spectrum(self._current_cube, x, y)
                if self._emission_wavelengths:
                    self._spectrum_ax.plot(
                        self._emission_wavelengths[:len(spectrum)],
                        spectrum,
                        color=SPECTRUM_COLORS[0],
                        linewidth=1,
                        marker=".",
                        markersize=3,
                    )
                else:
                    self._spectrum_ax.plot(
                        spectrum,
                        color=SPECTRUM_COLORS[0],
                        linewidth=1,
                        marker=".",
                        markersize=3,
                    )
                has_data = True
            except IndexError:
                pass

        if has_data:
            # Configure axes
            self._spectrum_ax.set_xlabel("Wavelength (nm)" if self._emission_wavelengths else "Band Index", fontsize=9)
            self._spectrum_ax.set_ylabel("Intensity", fontsize=9)
            self._spectrum_ax.grid(True, alpha=0.3)

            if self._clicked_pixel:
                x, y = self._clicked_pixel
                self._spectrum_ax.set_title(f"Spectrum at ({x}, {y})", fontsize=10)
            else:
                self._spectrum_ax.set_title("Spectrum", fontsize=10)

            # Auto-scale or fixed range
            if not self._spectrum_auto_scale.get():
                self._spectrum_ax.set_ylim(0, None)

            self._spectrum_ax.tick_params(axis="both", labelsize=8)

        else:
            # No data - show placeholder
            self._spectrum_ax.text(
                0.5, 0.5,
                "Click on image to plot spectrum",
                ha="center", va="center",
                transform=self._spectrum_ax.transAxes,
                fontsize=10,
                color="gray",
            )
            self._spectrum_ax.set_xticks([])
            self._spectrum_ax.set_yticks([])

        self._spectrum_figure.tight_layout()
        self._spectrum_canvas.draw()

    def _on_canvas_click(self, event: Any) -> None:
        """Handle left-click on canvas for spectrum extraction."""
        if event.inaxes != self._ax or self._current_cube is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        x_int, y_int = int(x), int(y)
        height, width = self._current_cube.shape[0], self._current_cube.shape[1]

        if not (0 <= x_int < width and 0 <= y_int < height):
            return

        # Extract spectrum at clicked point
        try:
            spectrum = extract_spectrum(self._current_cube, x_int, y_int)
        except IndexError:
            return

        self._clicked_pixel = (x_int, y_int)

        # Handle compare mode
        if self._spectrum_compare_mode.get():
            # Add to traces (max 5)
            if len(self._spectrum_traces) >= MAX_COMPARE_SPECTRA:
                # Remove oldest trace
                self._spectrum_traces.pop(0)
            self._spectrum_traces.append((x_int, y_int, spectrum))
        else:
            # Single mode - clear traces and just track clicked pixel
            self._spectrum_traces = []

        # Update spectrum plot
        self._update_spectrum_plot()

        logger.debug(f"Spectrum extracted at ({x_int}, {y_int})")

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

        # Zoom controls
        zoom_frame = ttk.Frame(self._canvas_frame)
        zoom_frame.pack(fill=tk.X, pady=2)

        ttk.Button(zoom_frame, text="+", width=3, command=self._zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="-", width=3, command=self._zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Fit", command=self._zoom_fit).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="1:1", command=self._zoom_actual).pack(side=tk.LEFT, padx=2)

        self._zoom_label = ttk.Label(zoom_frame, text="100%")
        self._zoom_label.pack(side=tk.LEFT, padx=10)

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

        # Pixel position label
        ttk.Label(info_frame, text="Pixel:").pack(side=tk.LEFT, padx=5)
        self._position_label = ttk.Label(info_frame, text="(---, ---)")
        self._position_label.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(info_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Band info (when in single-band mode)
        self._pixel_band_label = ttk.Label(info_frame, text="")
        self._pixel_band_label.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(info_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Value label
        ttk.Label(info_frame, text="Value:").pack(side=tk.LEFT, padx=2)
        self._value_label = ttk.Label(info_frame, text="---")
        self._value_label.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(info_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Excitation/band info
        self._band_info_label = ttk.Label(info_frame, text="")
        self._band_info_label.pack(side=tk.LEFT, padx=5)

    def _bind_events(self) -> None:
        """Bind keyboard and mouse events."""
        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self._on_open())
        self.root.bind("<r>", lambda e: self._reset_view())
        self.root.bind("<Escape>", lambda e: self._on_cancel())
        self.root.bind("<plus>", lambda e: self._zoom_in())
        self.root.bind("<equal>", lambda e: self._zoom_in())  # = and + same key
        self.root.bind("<minus>", lambda e: self._zoom_out())
        self.root.bind("<0>", lambda e: self._zoom_fit())
        self.root.bind("<1>", lambda e: self._zoom_actual())

        # Canvas mouse events (if canvas exists)
        if hasattr(self, "_canvas"):
            self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
            self._canvas.mpl_connect("scroll_event", self._on_mouse_scroll)
            self._canvas.mpl_connect("button_press_event", self._on_canvas_click)

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
            title="Open Hyperspectral Data",
            initialdir=initial_dir,
            filetypes=[
                ("SpectraData files", "*.pkl"),
                ("Raw hyperspectral", "*.im3"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        self._last_directory = str(Path(file_path).parent)
        self._load_file(file_path)

    def _on_open_directory(self) -> None:
        """Handle Open Directory action for raw .im3 files."""
        initial_dir = self._last_directory or str(Path.home())

        dir_path = filedialog.askdirectory(
            title="Select Directory with .im3 Files",
            initialdir=initial_dir,
        )

        if not dir_path:
            return

        self._last_directory = dir_path
        self._load_raw_directory(dir_path)

    def _load_file(self, file_path: str) -> None:
        """Load a hyperspectral data file.

        Supports both SpectraData pickle files (.pkl) and raw .im3 files.
        For .im3 files, attempts to load via DataLoader (requires ImageJ).
        """
        path = Path(file_path)
        self._update_status(f"Loading {path.name}...")

        try:
            # Import SpectraData lazily to avoid circular imports
            from .types import SpectraData

            # Check file type
            if path.suffix.lower() == ".im3":
                # Raw .im3 file - need to load directory containing it
                self._load_raw_directory(str(path.parent))
                return

            # Load SpectraData pickle
            self._spectra_data = SpectraData.from_pickle(path)
            self._finalize_load(path.name)

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            messagebox.showerror("Error", f"Failed to load file:\n\n{e}")
            self._update_status("Error loading file")

    def _load_raw_directory(self, dir_path: str) -> None:
        """Load raw .im3 files from a directory.

        Uses DataLoader/SpectraData.from_raw() to load raw hyperspectral data.
        Gracefully handles ImageJ unavailability with an info dialog.
        """
        path = Path(dir_path)
        self._update_status(f"Loading raw data from {path.name}...")

        try:
            from .loader import DataLoadingError
            from .types import SpectraData

            # Attempt to load raw data
            self._spectra_data = SpectraData.from_raw(path)
            self._finalize_load(path.name)

        except DataLoadingError as e:
            # Check if it's an ImageJ/pyimagej issue
            if "pyimagej" in str(e).lower() or "imagej" in str(e).lower():
                messagebox.showinfo(
                    "ImageJ Required",
                    "Loading raw .im3 files requires pyimagej.\n\n"
                    "To install:\n"
                    "  pip install pyimagej\n\n"
                    "Note: First run will download ImageJ (~500MB).\n\n"
                    "Alternative: Load a pre-processed .pkl file instead.",
                )
                logger.info("ImageJ not available for raw file loading")
            else:
                logger.error(f"Failed to load raw data from {path}: {e}")
                messagebox.showerror("Error", f"Failed to load raw data:\n\n{e}")

            self._update_status("Ready")

        except Exception as e:
            logger.error(f"Failed to load raw data from {path}: {e}")
            messagebox.showerror("Error", f"Failed to load raw data:\n\n{e}")
            self._update_status("Error loading file")

    def _finalize_load(self, file_name: str) -> None:
        """Finalize data loading - update UI after successful load."""
        if self._spectra_data is None:
            return

        # Update UI
        self._file_label.config(text=file_name)
        self.root.title(f"ME-HSI Viewer - {file_name}")

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
        self._update_status(f"Loaded: {file_name} ({n_ex} excitations, {h}x{w})")

        logger.info(f"Loaded SpectraData: {file_name}")

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

    def _on_display_mode_changed(self) -> None:
        """Handle display mode toggle (Composite vs Single Band)."""
        self._update_display()

    def _on_band_slider_changed(self, value: str) -> None:
        """Handle band slider movement."""
        # Debounce rapid slider movements
        self._update_band_display()

    def _on_band_spinbox_changed(self) -> None:
        """Handle band spinbox value change."""
        try:
            band = self._current_band.get()
            n_bands = len(self._emission_wavelengths)
            if n_bands > 0:
                # Clamp to valid range
                band = max(0, min(band, n_bands - 1))
                self._current_band.set(band)
            self._update_band_display()
        except (ValueError, tk.TclError):
            pass

    def _update_band_display(self) -> None:
        """Update the wavelength label and display for current band."""
        band = self._current_band.get()

        # Update wavelength label
        if self._emission_wavelengths and 0 <= band < len(self._emission_wavelengths):
            wavelength = self._emission_wavelengths[band]
            self._wavelength_label.config(text=f"Wavelength: {wavelength:.1f} nm")
        else:
            self._wavelength_label.config(text="Wavelength: -- nm")

        # Update display if in single-band mode
        if self._display_mode.get() == "single":
            self._update_display()

    def _update_speed_label(self, *args: Any) -> None:
        """Update the animation speed label."""
        speed = self._animation_speed.get()
        self._speed_label.config(text=f"{speed}ms")

    def _on_play_animation(self) -> None:
        """Start band animation playback."""
        if self._animation_running:
            return

        self._animation_running = True
        self._play_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)

        # Switch to single-band mode for animation
        self._display_mode.set("single")
        self._animate_next_band()

    def _on_stop_animation(self) -> None:
        """Stop band animation playback."""
        self._animation_running = False
        self._play_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)

        # Cancel any pending animation callback
        if self._animation_id is not None:
            self.root.after_cancel(self._animation_id)
            self._animation_id = None

    def _animate_next_band(self) -> None:
        """Advance to next band in animation sequence."""
        if not self._animation_running:
            return

        n_bands = len(self._emission_wavelengths)
        if n_bands == 0:
            self._on_stop_animation()
            return

        # Get current band and advance
        current = self._current_band.get()
        next_band = current + 1

        # Check if we've reached the end
        if next_band >= n_bands:
            if self._animation_loop.get():
                next_band = 0
            else:
                self._on_stop_animation()
                return

        # Update band and display
        self._current_band.set(next_band)
        self._update_band_display()

        # Schedule next frame
        speed = self._animation_speed.get()
        self._animation_id = self.root.after(speed, self._animate_next_band)

    def _on_false_color_toggled(self) -> None:
        """Handle false color enable/disable toggle."""
        if self._false_color_enabled.get():
            # Switch to composite mode when enabling false color
            self._display_mode.set("composite")
        self._update_display()

    def _on_fc_preset_changed(self, event: Optional[Any] = None) -> None:
        """Handle false color preset selection change."""
        preset_name = self._fc_preset.get()

        if preset_name == "Custom":
            # Keep current selections
            return

        if preset_name not in FALSE_COLOR_PRESETS:
            return

        n_bands = len(self._emission_wavelengths)
        if n_bands == 0:
            return

        # Get fractions from preset and convert to band indices
        r_frac, g_frac, b_frac = FALSE_COLOR_PRESETS[preset_name]
        r_idx = int(n_bands * r_frac)
        g_idx = int(n_bands * g_frac)
        b_idx = min(int(n_bands * b_frac), n_bands - 1)

        # Update combo selections
        self._update_fc_channel_selection(self._fc_r_combo, r_idx, self._fc_r_label)
        self._update_fc_channel_selection(self._fc_g_combo, g_idx, self._fc_g_label)
        self._update_fc_channel_selection(self._fc_b_combo, b_idx, self._fc_b_label)

        # Store band indices
        self._fc_r_band.set(r_idx)
        self._fc_g_band.set(g_idx)
        self._fc_b_band.set(b_idx)

        # Apply if live preview enabled
        if self._false_color_live.get() and self._false_color_enabled.get():
            self._update_display()

    def _on_fc_channel_changed(self, event: Optional[Any] = None) -> None:
        """Handle false color channel dropdown change."""
        # Parse band index from combo selection
        def get_band_from_combo(combo: ttk.Combobox, label: ttk.Label) -> int:
            selection = combo.get()
            if not selection:
                return 0
            try:
                # Format: "15: 520.0 nm"
                band_idx = int(selection.split(":")[0])
                if 0 <= band_idx < len(self._emission_wavelengths):
                    wavelength = self._emission_wavelengths[band_idx]
                    label.config(text=f"{wavelength:.1f} nm")
                return band_idx
            except (ValueError, IndexError):
                return 0

        self._fc_r_band.set(get_band_from_combo(self._fc_r_combo, self._fc_r_label))
        self._fc_g_band.set(get_band_from_combo(self._fc_g_combo, self._fc_g_label))
        self._fc_b_band.set(get_band_from_combo(self._fc_b_combo, self._fc_b_label))

        # Mark as custom
        self._fc_preset.set("Custom")

        # Apply if live preview enabled
        if self._false_color_live.get() and self._false_color_enabled.get():
            self._update_display()

    def _on_fc_apply(self) -> None:
        """Apply false color composition."""
        # Enable false color if not already
        self._false_color_enabled.set(True)
        self._display_mode.set("composite")
        self._update_display()

    def _on_fc_reset(self) -> None:
        """Reset false color to default preset."""
        self._fc_preset.set("Default (20/50/80%)")
        self._on_fc_preset_changed()

    def _update_fc_channel_selection(
        self, combo: Optional[ttk.Combobox], band_idx: int, label: ttk.Label
    ) -> None:
        """Update a false color channel combo to select the specified band."""
        if combo is None:
            return

        n_bands = len(self._emission_wavelengths)
        if n_bands == 0 or band_idx < 0 or band_idx >= n_bands:
            return

        # Find and select the matching entry
        wavelength = self._emission_wavelengths[band_idx]
        target = f"{band_idx}: {wavelength:.1f} nm"

        values = combo["values"]
        if target in values:
            combo.set(target)
            label.config(text=f"{wavelength:.1f} nm")

    def _populate_false_color_combos(self) -> None:
        """Populate false color channel dropdowns with current bands."""
        n_bands = len(self._emission_wavelengths)
        if n_bands == 0:
            return

        # Build options list: "index: wavelength nm"
        options = [
            f"{i}: {self._emission_wavelengths[i]:.1f} nm"
            for i in range(n_bands)
        ]

        # Update all combos
        for combo in [self._fc_r_combo, self._fc_g_combo, self._fc_b_combo]:
            if combo is not None:
                combo["values"] = options

        # Set default selection using preset
        self._on_fc_preset_changed()

    def _update_display(self) -> None:
        """Update the image display with current settings."""
        if self._spectra_data is None or self._current_excitation is None:
            return

        try:
            # Get current excitation data
            ex_data = self._spectra_data.get_excitation(self._current_excitation)
            self._current_cube = ex_data.cube

            # Store emission wavelengths and update slider range
            self._emission_wavelengths = list(ex_data.emission_wavelengths)
            n_bands = len(self._emission_wavelengths)

            # Update band slider range
            if n_bands > 0:
                self._band_slider.config(to=n_bands - 1)
                self._band_spinbox.config(to=n_bands - 1)

                # Clamp current band to valid range
                current = self._current_band.get()
                if current >= n_bands:
                    self._current_band.set(n_bands - 1)

                # Populate false color combos if not already done
                if self._fc_r_combo is not None and len(self._fc_r_combo["values"]) != n_bands:
                    self._populate_false_color_combos()

            # Check display mode
            display_mode = self._display_mode.get()

            if display_mode == "single" and n_bands > 0:
                # Single-band mode with colormap
                band_idx = self._current_band.get()
                band_data = self._current_cube[:, :, band_idx]

                # Normalize band data
                percentile = 98 if self._auto_contrast.get() else 100
                if np.any(band_data > 0):
                    pval = np.percentile(band_data[band_data > 0], percentile)
                else:
                    pval = 1.0
                if pval == 0:
                    pval = 1.0
                band_norm = np.clip(band_data / pval, 0, 1)

                # Apply manual contrast if not auto
                if not self._auto_contrast.get():
                    min_val = self._min_val.get() / 100.0
                    max_val = self._max_val.get() / 100.0
                    if max_val > min_val:
                        band_norm = np.clip(
                            (band_norm - min_val) / (max_val - min_val),
                            0, 1,
                        )

                # Get colormap
                cmap_name = self._colormap.get()

                # Update display with colormap
                self._ax.clear()
                self._img_display = self._ax.imshow(
                    band_norm,
                    aspect="equal",
                    interpolation="nearest",
                    cmap=cmap_name,
                )

                # Title with band info
                wavelength = self._emission_wavelengths[band_idx] if band_idx < len(self._emission_wavelengths) else 0
                self._ax.set_title(
                    f"Ex: {self._current_excitation} nm | Band {band_idx}: {wavelength:.1f} nm"
                )

                # Update wavelength label
                self._wavelength_label.config(text=f"Wavelength: {wavelength:.1f} nm")

                # Store for mouse tracking (use the single band repeated for compatibility)
                self._current_image = np.stack([band_norm, band_norm, band_norm], axis=2)

            else:
                # Composite mode - Create RGB image
                percentile = 98 if self._auto_contrast.get() else 100

                # Check if false color is enabled
                if self._false_color_enabled.get() and n_bands > 0:
                    # Use custom band assignment
                    r_band = self._fc_r_band.get()
                    g_band = self._fc_g_band.get()
                    b_band = self._fc_b_band.get()

                    # Validate bands are in range
                    r_band = min(r_band, n_bands - 1)
                    g_band = min(g_band, n_bands - 1)
                    b_band = min(b_band, n_bands - 1)

                    self._current_image = compose_false_color(
                        self._current_cube,
                        r_band=r_band,
                        g_band=g_band,
                        b_band=b_band,
                        percentile=percentile,
                    )
                    title_suffix = f" (FC: R={r_band}, G={g_band}, B={b_band})"
                else:
                    # Use default RGB method
                    method = self._rgb_method.get()
                    self._current_image = create_rgb_image(
                        self._current_cube,
                        method=method,
                        percentile=percentile,
                    )
                    title_suffix = ""

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
                self._ax.set_title(f"Excitation: {self._current_excitation} nm{title_suffix}")

            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._canvas.draw()

            # Update band info
            self._band_info_label.config(
                text=f"Ex: {self._current_excitation}nm | {n_bands} bands"
            )

        except Exception as e:
            logger.error(f"Display update failed: {e}")
            self._update_status(f"Display error: {e}")

    def _on_mouse_move(self, event: Any) -> None:
        """Handle mouse movement over the canvas."""
        if event.inaxes != self._ax or self._current_cube is None:
            self._position_label.config(text="(---, ---)")
            self._pixel_band_label.config(text="")
            self._value_label.config(text="---")
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        x_int, y_int = int(x), int(y)
        height, width = self._current_cube.shape[0], self._current_cube.shape[1]

        if 0 <= x_int < width and 0 <= y_int < height:
            # Update position
            self._position_label.config(text=f"({x_int}, {y_int})")

            # Get value based on display mode
            display_mode = self._display_mode.get()

            if display_mode == "single":
                # Single-band mode: show current band value
                band_idx = self._current_band.get()
                pixel_val = self._current_cube[y_int, x_int, band_idx]

                # Show band info
                if band_idx < len(self._emission_wavelengths):
                    wavelength = self._emission_wavelengths[band_idx]
                    self._pixel_band_label.config(text=f"Band {band_idx}: {wavelength:.0f}nm")
                else:
                    self._pixel_band_label.config(text=f"Band {band_idx}")
            else:
                # Composite mode: show mean value across bands
                pixel_vals = self._current_cube[y_int, x_int, :]
                pixel_val = np.mean(pixel_vals)
                self._pixel_band_label.config(text="Mean")

            # Format value (scientific notation for small values)
            if abs(pixel_val) < 0.001 and pixel_val != 0:
                val_str = f"{pixel_val:.2e}"
            else:
                val_str = f"{pixel_val:.4f}"

            self._value_label.config(text=val_str)
        else:
            self._position_label.config(text="Outside")
            self._pixel_band_label.config(text="")
            self._value_label.config(text="---")

    def _on_mouse_scroll(self, event: Any) -> None:
        """Handle mouse wheel scrolling for zoom."""
        if event.inaxes != self._ax:
            return

        # Get the current axis limits
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()

        # Get cursor position in data coordinates
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        # Calculate zoom factor based on scroll direction
        if event.button == "up":
            scale = 1.0 / self._zoom_factor  # Zoom in
            self._zoom_level *= self._zoom_factor
        elif event.button == "down":
            scale = self._zoom_factor  # Zoom out
            self._zoom_level /= self._zoom_factor
        else:
            return

        # Calculate new limits centered on cursor
        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[0] - ylim[1]) * scale  # ylim is inverted for images

        # Maintain cursor position as zoom center
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[1]) / (ylim[0] - ylim[1])

        new_xlim = [xdata - new_width * relx, xdata + new_width * (1 - relx)]
        new_ylim = [ydata + new_height * (1 - rely), ydata - new_height * rely]

        self._ax.set_xlim(new_xlim)
        self._ax.set_ylim(new_ylim)
        self._canvas.draw_idle()

        # Update zoom label
        self._update_zoom_label()

    def _zoom_in(self) -> None:
        """Zoom in by one step, centered on image."""
        self._zoom_by_factor(1.0 / self._zoom_factor)

    def _zoom_out(self) -> None:
        """Zoom out by one step, centered on image."""
        self._zoom_by_factor(self._zoom_factor)

    def _zoom_by_factor(self, scale: float) -> None:
        """Zoom by a given scale factor, centered on the current view."""
        if not hasattr(self, "_ax"):
            return

        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()

        # Calculate center of current view
        xcenter = (xlim[0] + xlim[1]) / 2
        ycenter = (ylim[0] + ylim[1]) / 2

        # Calculate new dimensions
        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[0] - ylim[1]) * scale

        # Update zoom level
        self._zoom_level /= scale

        # Set new limits
        self._ax.set_xlim([xcenter - new_width / 2, xcenter + new_width / 2])
        self._ax.set_ylim([ycenter + new_height / 2, ycenter - new_height / 2])
        self._canvas.draw_idle()

        self._update_zoom_label()

    def _zoom_fit(self) -> None:
        """Fit image to window (reset zoom)."""
        if hasattr(self, "_nav_toolbar"):
            self._nav_toolbar.home()
        self._zoom_level = 1.0
        self._update_zoom_label()

    def _zoom_actual(self) -> None:
        """Zoom to 1:1 (actual pixels)."""
        if self._current_cube is None:
            return

        height, width = self._current_cube.shape[0], self._current_cube.shape[1]

        # Set axes limits to actual image size
        self._ax.set_xlim([-0.5, width - 0.5])
        self._ax.set_ylim([height - 0.5, -0.5])
        self._canvas.draw_idle()

        # Calculate effective zoom level
        # This is approximate - depends on figure size and DPI
        fig_width, fig_height = self._figure.get_size_inches()
        dpi = self._figure.dpi
        display_width = fig_width * dpi * 0.9  # Account for margins
        display_height = fig_height * dpi * 0.9

        self._zoom_level = 1.0
        self._update_zoom_label()

    def _update_zoom_label(self) -> None:
        """Update the zoom level display."""
        percentage = int(self._zoom_level * 100)
        self._zoom_label.config(text=f"{percentage}%")

    def _reset_view(self) -> None:
        """Reset the view to fit the image."""
        self._zoom_fit()

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
