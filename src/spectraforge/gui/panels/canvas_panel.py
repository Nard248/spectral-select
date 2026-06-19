"""Canvas panel: paint the active material into the active layer's amount map.

Tools: brush (drag to paint a disc), rectangle, circle. Painting is disabled while the
matplotlib zoom/pan toolbar is active so navigation doesn't paint.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from mehsi_preprocessor.state import DEFAULT_CLASS_COLORS
from mehsi_preprocessor.widgets.image_canvas import ImageCanvas


class CanvasPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._tool = "brush"
        self._radius = 4
        self._brush_value = 1.0
        self._press_px = None        # (row, col) at button press

        self._canvas = ImageCanvas(self)

        # --- tool bar ---
        self._tool_combo = QComboBox()
        self._tool_combo.addItems(["brush", "eraser", "rect", "circle"])
        self._tool_combo.currentTextChanged.connect(self.set_tool)
        self._radius_spin = QSpinBox()
        self._radius_spin.setRange(1, 50)
        self._radius_spin.setValue(self._radius)
        self._radius_spin.valueChanged.connect(self.set_radius)
        self._value_spin = QDoubleSpinBox()
        self._value_spin.setRange(0.0, 100.0)
        self._value_spin.setSingleStep(0.1)
        self._value_spin.setValue(self._brush_value)
        self._value_spin.valueChanged.connect(self.set_brush_value)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Tool"))
        bar.addWidget(self._tool_combo)
        bar.addWidget(QLabel("Brush radius"))
        bar.addWidget(self._radius_spin)
        bar.addWidget(QLabel("Brush value"))
        bar.addWidget(self._value_spin)
        bar.addStretch(1)

        root = QVBoxLayout(self)
        root.addLayout(bar)
        root.addWidget(self._canvas)

        fc = self._canvas.canvas
        fc.mpl_connect("button_press_event", self._on_press)
        fc.mpl_connect("motion_notify_event", self._on_motion)
        fc.mpl_connect("button_release_event", self._on_release)

        self.refresh()

    # ------------------------------------------------------------------
    # Tool state
    # ------------------------------------------------------------------

    def set_tool(self, name: str) -> None:
        self._tool = name

    def set_radius(self, r: int) -> None:
        self._radius = int(r)

    def set_brush_value(self, v: float) -> None:
        self._brush_value = float(v)

    # ------------------------------------------------------------------
    # Painting primitives
    # ------------------------------------------------------------------

    def _active(self):
        i = self.state.active_layer
        return self.state.layers[i] if 0 <= i < len(self.state.layers) else None

    def _disc(self, row, col):
        yy, xx = np.ogrid[: self.state.height, : self.state.width]
        return (yy - row) ** 2 + (xx - col) ** 2 <= self._radius ** 2

    def brush_at(self, row, col) -> None:
        """Set a disc of the current radius around (row, col) to the brush value."""
        layer = self._active()
        if layer is None:
            return
        disc = self._disc(row, col)
        layer.amount_map[disc] = np.maximum(layer.amount_map[disc], self._brush_value)
        self.refresh()

    def erase_at(self, row, col) -> None:
        """Zero a disc of the current radius around (row, col)."""
        layer = self._active()
        if layer is None:
            return
        layer.amount_map[self._disc(row, col)] = 0.0
        self.refresh()

    def _dab_at(self, row, col) -> None:
        (self.erase_at if self._tool == "eraser" else self.brush_at)(row, col)

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

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def _painting_allowed(self) -> bool:
        try:
            return str(self._canvas._toolbar.mode) == ""   # no zoom/pan active
        except AttributeError:
            return True

    def _event_pixel(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return None
        row = int(round(event.ydata))
        col = int(round(event.xdata))
        if 0 <= row < self.state.height and 0 <= col < self.state.width:
            return row, col
        return None

    def _on_press(self, event):
        if getattr(event, "button", None) != 1 or not self._painting_allowed():
            return
        px = self._event_pixel(event)
        if px is None:
            return
        self._press_px = px
        if self._tool in ("brush", "eraser"):
            self._dab_at(*px)

    def _on_motion(self, event):
        if self._press_px is None or self._tool not in ("brush", "eraser") or not self._painting_allowed():
            return
        px = self._event_pixel(event)
        if px is not None:
            self._dab_at(*px)

    def _on_release(self, event):
        if self._press_px is None:
            return
        start = self._press_px
        self._press_px = None
        px = self._event_pixel(event)
        if px is None or not self._painting_allowed():
            return
        r0, c0 = start
        r1, c1 = px
        if self._tool == "rect":
            self.paint_rect(min(r0, r1), max(r0, r1) + 1, min(c0, c1), max(c0, c1) + 1)
        elif self._tool == "circle":
            radius = int(round(((r1 - r0) ** 2 + (c1 - c0) ** 2) ** 0.5))
            self.paint_circle(r0, c0, radius)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def composite(self):
        """False-color RGB composite of visible layers (stable per-layer hue)."""
        rgb = np.zeros((self.state.height, self.state.width, 3))
        visible = [layer for layer in self.state.layers if layer.visible]
        for n, layer in enumerate(visible):
            rgb255 = DEFAULT_CLASS_COLORS[n % len(DEFAULT_CLASS_COLORS)]
            color = np.array(rgb255, dtype=float) / 255.0
            a = layer.amount_map
            norm = a / a.max() if a.max() > 0 else a
            rgb += norm[:, :, None] * color[None, None, :]
        return np.clip(rgb, 0, 1)

    def refresh(self):
        self._canvas.show_image(self.composite())
