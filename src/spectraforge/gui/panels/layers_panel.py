"""Layers panel: the layer stack (add / reorder / toggle / select active)."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QListWidget, QPushButton, QVBoxLayout, QWidget


class LayersPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_select)
        self._combo = QComboBox()
        add = QPushButton("Add layer")
        add.clicked.connect(self._on_add)
        h = QHBoxLayout()
        h.addWidget(self._combo)
        h.addWidget(add)
        root = QVBoxLayout(self)
        root.addWidget(self._list)
        root.addLayout(h)
        self._refresh_combo()
        self._refresh_list()

    def _refresh_combo(self):
        self._combo.clear()
        self._combo.addItems(sorted(self.state.materials))

    def _refresh_list(self):
        self._list.clear()
        self._list.addItems(
            [f"{'☑' if layer.visible else '☐'} {layer.name}" for layer in self.state.layers]
        )

    def _on_add(self):
        if self._combo.currentText():
            self.add_layer_with_material(self._combo.currentText())

    def _on_select(self, row):
        if 0 <= row < len(self.state.layers):
            self.state.active_layer = row

    def add_layer_with_material(self, material_name):
        mat = self.state.materials[material_name]
        self.state.add_layer(material_name, mat)
        self._refresh_list()
        self.changed.emit()

    def set_visible(self, i, visible):
        self.state.layers[i].visible = visible
        self._refresh_list()
        self.changed.emit()
