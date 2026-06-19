"""Material composer panel: mix fluorophores into a material (brush)."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QListWidget, QPushButton,
    QVBoxLayout, QWidget,
)

from spectraforge.material import Material
from spectraforge.gui.widgets.spectrum_plot import SpectrumPlot


class MaterialPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._name = QLineEdit()
        self._rows: list[tuple[QComboBox, QDoubleSpinBox]] = []
        self._rows_box = QVBoxLayout()
        self._plot = SpectrumPlot()
        self._materials_list = QListWidget()

        add_row = QPushButton("+ fluorophore")
        add_row.clicked.connect(self._add_row)
        compose = QPushButton("Compose material")
        compose.clicked.connect(self._on_compose)

        root = QVBoxLayout(self)
        root.addWidget(self._name)
        root.addLayout(self._rows_box)
        root.addWidget(add_row)
        root.addWidget(compose)
        root.addWidget(self._plot)
        root.addWidget(self._materials_list)
        self._add_row()

    def _add_row(self):
        combo = QComboBox()
        combo.addItems(sorted(self.state.library))
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setValue(1.0)
        self._rows.append((combo, spin))
        h = QHBoxLayout()
        h.addWidget(combo)
        h.addWidget(spin)
        self._rows_box.addLayout(h)

    def _on_compose(self):
        recipe = {c.currentText(): s.value() for c, s in self._rows if s.value() > 0}
        self.compose_material(self._name.text() or "material", recipe)

    def compose_material(self, name, recipe):
        mat = Material(name, dict(recipe))
        self.state.materials[name] = mat
        self._materials_list.clear()
        self._materials_list.addItems(sorted(self.state.materials))
        ex = self.state.acquisition.excitations[0]
        self._plot.plot_material(mat, self.state.library, ex)

    def refresh(self):
        """Re-sync the materials list and fluorophore combos from state (e.g. after load)."""
        self._materials_list.clear()
        self._materials_list.addItems(sorted(self.state.materials))
        for combo, _ in self._rows:
            combo.clear()
            combo.addItems(sorted(self.state.library))
