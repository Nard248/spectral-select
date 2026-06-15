"""Visual bar chart showing kept (green) vs removed (red) spectral bands."""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class SpectralBarChart(QWidget):
    """Horizontal bar chart of emission bands coloured by keep/remove."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(8, 1.5), tight_layout=True)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def update_chart(
        self,
        wavelengths: List[float],
        keep_mask: np.ndarray,
        excitation_nm: float,
    ) -> None:
        """Redraw the bar chart.

        Parameters
        ----------
        wavelengths : list of float
            All emission wavelengths for this excitation.
        keep_mask : bool array
            True = kept, False = removed.
        excitation_nm : float
            For the title.
        """
        ax = self._ax
        ax.clear()

        wl = np.asarray(wavelengths)
        colours = ["#4CAF50" if k else "#F44336" for k in keep_mask]
        ax.bar(wl, np.ones_like(wl), width=np.diff(wl, append=wl[-1] + 5), color=colours, edgecolor="none")
        ax.set_xlabel("Emission wavelength (nm)", fontsize=8)
        ax.set_yticks([])
        ax.set_title(
            f"Ex {excitation_nm:.0f} nm – {int(keep_mask.sum())} kept / "
            f"{int((~keep_mask).sum())} removed",
            fontsize=9,
        )
        self._canvas.draw_idle()
