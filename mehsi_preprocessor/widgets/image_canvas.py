"""Matplotlib canvas widget embedded in Qt for hyperspectral image display."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class ImageCanvas(QWidget):
    """Zoomable / pannable matplotlib canvas for displaying 2-D images.

    Wraps a ``FigureCanvasQTAgg`` with an embedded ``NavigationToolbar2QT``
    providing the standard zoom/pan/home controls.

    Usage::

        canvas = ImageCanvas()
        canvas.show_image(img_2d)           # grayscale [0,1]
        canvas.show_image(img_rgb)          # (H, W, 3) [0,1]
        canvas.set_title("Band 12")
    """

    def __init__(self, parent: QWidget | None = None, figsize: tuple = (5, 4)) -> None:
        super().__init__(parent)

        self._figure = Figure(figsize=figsize, tight_layout=True)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        self._im_handle = None  # AxesImage handle

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def ax(self):
        return self._ax

    @property
    def canvas(self) -> FigureCanvasQTAgg:
        return self._canvas

    def show_image(self, img: np.ndarray, cmap: str = "gray") -> None:
        """Display *img* (H,W) or (H,W,3) on the canvas."""
        self._ax.clear()
        if img.ndim == 2:
            self._im_handle = self._ax.imshow(img, cmap=cmap, aspect="equal")
        else:
            self._im_handle = self._ax.imshow(img, aspect="equal")
        self._ax.set_axis_off()
        self._canvas.draw_idle()

    def set_title(self, title: str) -> None:
        self._ax.set_title(title, fontsize=10)
        self._canvas.draw_idle()

    def clear(self) -> None:
        self._ax.clear()
        self._ax.set_axis_off()
        self._im_handle = None
        self._canvas.draw_idle()
