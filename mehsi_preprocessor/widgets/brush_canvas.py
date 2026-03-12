"""Paintable canvas for drawing class masks on top of a base image."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from mehsi_preprocessor.state import ClassDef


def _disk_kernel(radius: int) -> np.ndarray:
    """Return a boolean disk structuring element of given radius."""
    d = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


class BrushCanvas(QWidget):
    """Canvas that allows painting class IDs onto a mask array.

    Signals:
        mask_updated()  – emitted after any paint stroke
        click_for_fill(row, col) – emitted on single click when fill mode is active
    """

    mask_updated = pyqtSignal()
    click_for_fill = pyqtSignal(int, int)  # row, col for auto-fill

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._figure = Figure(figsize=(6, 5), tight_layout=True)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # Data
        self._base_image: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None  # (H, W) int32
        self._overlay_handle = None
        self._base_handle = None

        # Brush state
        self._painting = False
        self._erasing = False
        self._fill_mode = False  # Auto-fill mode
        self._brush_radius: int = 5
        self._current_class_id: int = 1
        self._class_defs: List[ClassDef] = []
        self._alpha: float = 0.40

        # Brush cursor (circle patch)
        self._cursor_circle: Optional[Circle] = None
        self._cursor_visible = False

        # Edge overlay for preview
        self._edge_overlay_handle = None
        self._edge_mask: Optional[np.ndarray] = None  # For auto-fill boundaries

        # Connect events
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("axes_leave_event", self._on_leave)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    @property
    def base_image(self) -> Optional[np.ndarray]:
        return self._base_image

    def set_base_image(self, img: np.ndarray) -> None:
        """Set the background image (H,W) or (H,W,3)."""
        self._base_image = img
        self._ax.clear()
        self._cursor_circle = None  # Reset cursor
        self._overlay_handle = None
        self._edge_overlay_handle = None
        if img.ndim == 2:
            self._base_handle = self._ax.imshow(img, cmap="gray", aspect="equal")
        else:
            self._base_handle = self._ax.imshow(img, aspect="equal")
        self._ax.set_axis_off()
        self._init_cursor()
        self._canvas.draw_idle()

    def set_mask(self, mask: np.ndarray) -> None:
        """Set the class mask (H,W) int32."""
        self._mask = mask.astype(np.int32)
        self._refresh_overlay()

    def init_mask(self, shape: Tuple[int, int]) -> None:
        """Create a blank mask."""
        self._mask = np.zeros(shape, dtype=np.int32)
        self._refresh_overlay()

    def set_class_defs(self, defs: List[ClassDef]) -> None:
        self._class_defs = defs

    def set_current_class(self, class_id: int) -> None:
        self._current_class_id = class_id

    def set_brush_radius(self, r: int) -> None:
        self._brush_radius = max(1, r)
        self._update_cursor_size()

    def set_erase_mode(self, on: bool) -> None:
        self._erasing = on
        self._fill_mode = False  # Disable fill mode when erasing

    def set_fill_mode(self, on: bool) -> None:
        """Enable/disable auto-fill mode (click to fill detected region)."""
        self._fill_mode = on
        if on:
            self._erasing = False

    def set_edge_mask(self, edge_mask: Optional[np.ndarray]) -> None:
        """Set the edge mask used for auto-fill boundaries."""
        self._edge_mask = edge_mask

    def show_edge_preview(self, edges: Optional[np.ndarray]) -> None:
        """Show/hide edge detection preview overlay."""
        if edges is None:
            if self._edge_overlay_handle is not None:
                self._edge_overlay_handle.remove()
                self._edge_overlay_handle = None
                self._canvas.draw_idle()
            return

        # Create red overlay for edges
        h, w = edges.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        overlay[edges > 0, 0] = 1.0  # Red
        overlay[edges > 0, 3] = 0.7  # Alpha

        if self._edge_overlay_handle is None:
            self._edge_overlay_handle = self._ax.imshow(overlay, aspect="equal")
        else:
            self._edge_overlay_handle.set_data(overlay)
        self._canvas.draw_idle()

    def fill_region(self, row: int, col: int) -> bool:
        """Fill a region starting from (row, col) using the edge mask.

        Returns True if any pixels were filled.
        """
        if self._mask is None or self._edge_mask is None:
            return False

        from scipy import ndimage

        h, w = self._mask.shape
        if not (0 <= row < h and 0 <= col < w):
            return False

        # Create a binary mask where edges are barriers
        # We'll flood fill from the click point, stopping at edges
        barriers = self._edge_mask > 0

        # Label connected regions (areas between edges)
        # Invert barriers: 1 = fillable, 0 = barrier
        fillable = ~barriers
        labeled, num_features = ndimage.label(fillable)

        if labeled[row, col] == 0:
            # Clicked on an edge
            return False

        # Get the region label at click point
        region_label = labeled[row, col]

        # Fill all pixels with this label
        region_mask = labeled == region_label
        value = 0 if self._erasing else self._current_class_id
        self._mask[region_mask] = value

        self._refresh_overlay()
        self.mask_updated.emit()
        return True

    # ------------------------------------------------------------------
    # Brush cursor
    # ------------------------------------------------------------------

    def _init_cursor(self) -> None:
        """Initialize the brush cursor circle."""
        self._cursor_circle = Circle(
            (0, 0),
            self._brush_radius,
            fill=False,
            edgecolor='white',
            linewidth=1.5,
            linestyle='-',
            visible=False,
        )
        self._ax.add_patch(self._cursor_circle)

    def _update_cursor_size(self) -> None:
        """Update cursor circle radius."""
        if self._cursor_circle is not None:
            self._cursor_circle.set_radius(self._brush_radius)
            if self._cursor_visible:
                self._canvas.draw_idle()

    def _update_cursor_position(self, x: float, y: float) -> None:
        """Move cursor circle to (x, y) in data coordinates."""
        if self._cursor_circle is None:
            return
        self._cursor_circle.set_center((x, y))
        if not self._cursor_visible:
            self._cursor_circle.set_visible(True)
            self._cursor_visible = True
        self._canvas.draw_idle()

    def _hide_cursor(self) -> None:
        """Hide the brush cursor."""
        if self._cursor_circle is not None and self._cursor_visible:
            self._cursor_circle.set_visible(False)
            self._cursor_visible = False
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _stamp(self, row: int, col: int) -> None:
        if self._mask is None:
            return
        h, w = self._mask.shape
        r = self._brush_radius
        kernel = _disk_kernel(r)

        r0 = max(row - r, 0)
        r1 = min(row + r + 1, h)
        c0 = max(col - r, 0)
        c1 = min(col + r + 1, w)

        kr0 = r0 - (row - r)
        kr1 = kernel.shape[0] - ((row + r + 1) - r1)
        kc0 = c0 - (col - r)
        kc1 = kernel.shape[1] - ((col + r + 1) - c1)

        patch = kernel[kr0:kr1, kc0:kc1]
        value = 0 if self._erasing else self._current_class_id
        self._mask[r0:r1, c0:c1][patch] = value

    def _on_press(self, event) -> None:
        if event.inaxes != self._ax or event.button != 1:
            return
        # Don't paint if toolbar is in zoom/pan mode
        if self._toolbar.mode:
            return

        row, col = int(round(event.ydata)), int(round(event.xdata))

        if self._fill_mode:
            # Auto-fill mode: emit signal for fill
            self.click_for_fill.emit(row, col)
        else:
            # Normal brush mode
            self._painting = True
            self._stamp(row, col)

    def _on_release(self, event) -> None:
        if self._painting:
            self._painting = False
            self._refresh_overlay()
            self.mask_updated.emit()

    def _on_motion(self, event) -> None:
        # Always update cursor position when in axes
        if event.inaxes == self._ax and event.xdata is not None:
            self._update_cursor_position(event.xdata, event.ydata)

            # Paint if dragging
            if self._painting and not self._fill_mode:
                row, col = int(round(event.ydata)), int(round(event.xdata))
                self._stamp(row, col)
        else:
            self._hide_cursor()

    def _on_leave(self, event) -> None:
        """Hide cursor when mouse leaves the axes."""
        self._hide_cursor()

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    def _refresh_overlay(self) -> None:
        if self._mask is None or self._base_image is None:
            return

        h, w = self._mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)

        # Build colour lookup
        color_map = {cd.id: cd.color for cd in self._class_defs}

        for cls_id, rgb in color_map.items():
            where = self._mask == cls_id
            overlay[where, 0] = rgb[0] / 255.0
            overlay[where, 1] = rgb[1] / 255.0
            overlay[where, 2] = rgb[2] / 255.0
            overlay[where, 3] = self._alpha

        if self._overlay_handle is None:
            self._overlay_handle = self._ax.imshow(overlay, aspect="equal")
        else:
            self._overlay_handle.set_data(overlay)

        self._canvas.draw_idle()
