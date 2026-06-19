"""Canvas panel: paint the active material into the active layer's amount map."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from mehsi_preprocessor.widgets.image_canvas import ImageCanvas


class CanvasPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._canvas = ImageCanvas(self)
        root = QVBoxLayout(self)
        root.addWidget(self._canvas)
        self.refresh()

    def _active(self):
        i = self.state.active_layer
        return self.state.layers[i] if 0 <= i < len(self.state.layers) else None

    def paint_rect(self, r0, r1, c0, c1, amount=1.0):
        layer = self._active()
        if layer is not None:
            layer.amount_map[r0:r1, c0:c1] += amount
            self.refresh()

    def paint_circle(self, cy, cx, radius, amount=1.0):
        layer = self._active()
        if layer is None:
            return
        yy, xx = np.ogrid[: self.state.height, : self.state.width]
        layer.amount_map[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] += amount
        self.refresh()

    def composite(self):
        """False-color RGB composite of visible layers (stable per-layer hue)."""
        rgb = np.zeros((self.state.height, self.state.width, 3))
        visible = [layer for layer in self.state.layers if layer.visible]
        for n, layer in enumerate(visible):
            color = np.array([(n * 53) % 255, (n * 101) % 255, (n * 151) % 255]) / 255.0
            a = layer.amount_map
            norm = a / a.max() if a.max() > 0 else a
            rgb += norm[:, :, None] * color[None, None, :]
        return np.clip(rgb, 0, 1)

    def refresh(self):
        self._canvas.show_image(self.composite())
