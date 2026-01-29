"""Paintable canvas for drawing class masks on top of a base image."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
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
    """

    mask_updated = pyqtSignal()

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
        self._brush_radius: int = 5
        self._current_class_id: int = 1
        self._class_defs: List[ClassDef] = []
        self._alpha: float = 0.40

        # Connect events
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def set_base_image(self, img: np.ndarray) -> None:
        """Set the background image (H,W) or (H,W,3)."""
        self._base_image = img
        self._ax.clear()
        if img.ndim == 2:
            self._base_handle = self._ax.imshow(img, cmap="gray", aspect="equal")
        else:
            self._base_handle = self._ax.imshow(img, aspect="equal")
        self._ax.set_axis_off()
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

    def set_erase_mode(self, on: bool) -> None:
        self._erasing = on

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
        self._painting = True
        row, col = int(round(event.ydata)), int(round(event.xdata))
        self._stamp(row, col)

    def _on_release(self, event) -> None:
        if self._painting:
            self._painting = False
            self._refresh_overlay()
            self.mask_updated.emit()

    def _on_motion(self, event) -> None:
        if not self._painting or event.inaxes != self._ax:
            return
        row, col = int(round(event.ydata)), int(round(event.xdata))
        self._stamp(row, col)

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
