"""Library panel: browse fluorophore spectra and define new fluorophores."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from spectraforge.fluorophore import Fluorophore
from spectraforge.gui.widgets.spectrum_plot import SpectrumPlot


class LibraryPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._list = QListWidget()
        self._plot = SpectrumPlot()
        self._list.currentTextChanged.connect(self._on_select)

        self._name = QLineEdit()
        self._spins = {k: QDoubleSpinBox() for k in
                       ("ex_peak", "ex_fwhm", "em_peak", "em_fwhm", "qy", "eps")}
        for sp in self._spins.values():
            sp.setRange(0, 2000)
            sp.setDecimals(2)
        self._spins["ex_peak"].setValue(480)
        self._spins["ex_fwhm"].setValue(40)
        self._spins["em_peak"].setValue(520)
        self._spins["em_fwhm"].setValue(45)
        self._spins["qy"].setRange(0, 1)
        self._spins["qy"].setValue(0.6)
        self._spins["eps"].setRange(0, 100)
        self._spins["eps"].setValue(1.0)

        form = QFormLayout()
        form.addRow("Name", self._name)
        for label, key in (("Ex peak", "ex_peak"), ("Ex FWHM", "ex_fwhm"),
                           ("Em peak", "em_peak"), ("Em FWHM", "em_fwhm"),
                           ("Quantum yield", "qy"), ("Extinction", "eps")):
            form.addRow(label, self._spins[key])
        add = QPushButton("Add fluorophore")
        add.clicked.connect(self._on_add)

        root = QVBoxLayout(self)
        root.addWidget(self._list)
        root.addWidget(self._plot)
        root.addLayout(form)
        root.addWidget(add)
        self._refresh()

    def _refresh(self):
        self._list.clear()
        self._list.addItems(sorted(self.state.library))

    def refresh(self):
        """Public re-sync of the fluorophore list (e.g. after project load)."""
        self._refresh()

    def _on_select(self, name):
        if name in self.state.library:
            self._plot.plot_fluorophore(self.state.library[name])

    def _on_add(self):
        s = self._spins
        self.define_fluorophore(self._name.text() or "dye",
                                s["ex_peak"].value(), s["ex_fwhm"].value(),
                                s["em_peak"].value(), s["em_fwhm"].value(),
                                s["qy"].value(), s["eps"].value())

    def define_fluorophore(self, name, ex_peak, ex_fwhm, em_peak, em_fwhm, qy, eps):
        self.state.library[name] = Fluorophore(name, ex_peak, ex_fwhm, em_peak, em_fwhm, qy, eps)
        self._refresh()
