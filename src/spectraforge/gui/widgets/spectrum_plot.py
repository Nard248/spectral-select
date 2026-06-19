"""Matplotlib widget that plots fluorophore excitation/emission curves."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class SpectrumPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(4, 2.4))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas)
        self._grid = np.arange(250, 750, 2.0)

    def plot_fluorophore(self, fluor) -> None:
        self._ax.clear()
        self._ax.plot(self._grid, fluor.excitation(self._grid), label="excitation")
        em = fluor.emission(self._grid)
        em = em / em.max() if em.max() > 0 else em
        self._ax.plot(self._grid, em, label="emission")
        self._ax.set_xlabel("wavelength (nm)")
        self._ax.legend(fontsize=7)
        self._ax.set_title(fluor.name, fontsize=9)
        self._canvas.draw_idle()

    def plot_material(self, material, library, excitation: float) -> None:
        self._ax.clear()
        total = None
        for fname, conc in material.recipe.items():
            f = library[fname]
            contrib = (
                conc * f.extinction * f.quantum_yield
                * float(f.excitation(excitation)) * f.emission(self._grid)
            )
            total = contrib if total is None else total + contrib
        if total is not None and total.max() > 0:
            self._ax.plot(self._grid, total / total.max(), label=f"mix @ {excitation:.0f} nm")
            self._ax.legend(fontsize=7)
        self._ax.set_xlabel("wavelength (nm)")
        self._ax.set_title(material.name, fontsize=9)
        self._canvas.draw_idle()
