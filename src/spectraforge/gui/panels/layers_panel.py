"""Layers panel: the layer stack (add / reorder / remove / toggle visibility / select)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)


class LayersPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_select)
        self._list.itemChanged.connect(self._on_item_changed)

        self._combo = QComboBox()
        add = QPushButton("Add layer")
        add.clicked.connect(self._on_add)
        add_row = QHBoxLayout()
        add_row.addWidget(self._combo)
        add_row.addWidget(add)

        up = QPushButton("↑")
        down = QPushButton("↓")
        remove = QPushButton("Remove")
        up.clicked.connect(lambda: self.move_up(self._list.currentRow()))
        down.clicked.connect(lambda: self.move_down(self._list.currentRow()))
        remove.clicked.connect(lambda: self.remove_layer(self._list.currentRow()))
        btn_row = QHBoxLayout()
        for b in (up, down, remove):
            btn_row.addWidget(b)

        root = QVBoxLayout(self)
        root.addWidget(self._list)
        root.addLayout(add_row)
        root.addLayout(btn_row)
        self._refresh_combo()
        self._refresh_list()

    def _refresh_combo(self):
        self._combo.clear()
        self._combo.addItems(sorted(self.state.materials))

    def _refresh_list(self):
        self._list.blockSignals(True)   # avoid itemChanged firing while we rebuild
        self._list.clear()
        for layer in self.state.layers:
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)

    # ------------------------------------------------------------------

    def _on_add(self):
        if self._combo.currentText():
            self.add_layer_with_material(self._combo.currentText())

    def _on_select(self, row):
        if 0 <= row < len(self.state.layers):
            self.state.active_layer = row

    def _on_item_changed(self, item):
        row = self._list.row(item)
        if 0 <= row < len(self.state.layers):
            self.state.layers[row].visible = item.checkState() == Qt.CheckState.Checked
            self.changed.emit()

    def add_layer_with_material(self, material_name):
        mat = self.state.materials[material_name]
        self.state.add_layer(material_name, mat)
        self._refresh_list()
        self.changed.emit()

    def set_visible(self, i, visible):
        self.state.layers[i].visible = visible
        self._refresh_list()
        self.changed.emit()

    def _swap(self, i, j):
        layers = self.state.layers
        layers[i], layers[j] = layers[j], layers[i]
        self.state.active_layer = j
        self._refresh_list()
        self._list.setCurrentRow(j)
        self.changed.emit()

    def move_up(self, i):
        if 0 < i < len(self.state.layers):
            self._swap(i, i - 1)

    def move_down(self, i):
        if 0 <= i < len(self.state.layers) - 1:
            self._swap(i, i + 1)

    def remove_layer(self, i):
        if 0 <= i < len(self.state.layers):
            self.state.layers.pop(i)
            self.state.active_layer = min(i, len(self.state.layers) - 1)
            self._refresh_list()
            self.changed.emit()
