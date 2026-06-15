"""Rectangle selection tool using matplotlib RectangleSelector."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from mehsi_preprocessor.widgets.image_canvas import ImageCanvas


class RectSelector(ImageCanvas):
    """ImageCanvas with an interactive rectangle selector.

    Signals:
        rect_selected(row_min, row_max, col_min, col_max)
    """

    rect_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selector: Optional[RectangleSelector] = None
        self._last_rect: Optional[Tuple[int, int, int, int]] = None

    def enable_selector(self) -> None:
        """Activate the rectangle selector on the current axes."""
        if self._selector is not None:
            self._selector.set_active(False)

        self._selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            interactive=True,
            spancoords="data",
        )
        self.canvas.draw_idle()

    def disable_selector(self) -> None:
        if self._selector is not None:
            self._selector.set_active(False)
            self._selector = None

    @property
    def last_rect(self) -> Optional[Tuple[int, int, int, int]]:
        return self._last_rect

    def _on_select(self, eclick, erelease) -> None:
        x0, y0 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))

        # matplotlib coords: x=col, y=row
        row_min = max(min(y0, y1), 0)
        row_max = max(y0, y1)
        col_min = max(min(x0, x1), 0)
        col_max = max(x0, x1)

        self._last_rect = (row_min, row_max, col_min, col_max)
        self.rect_selected.emit(row_min, row_max, col_min, col_max)
