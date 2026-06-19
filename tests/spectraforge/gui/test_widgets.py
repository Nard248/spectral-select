import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from spectraforge.fluorophore import Fluorophore

_app = QApplication.instance() or QApplication([])


def test_spectrum_plot_builds_and_plots():
    from spectraforge.gui.widgets.spectrum_plot import SpectrumPlot
    w = SpectrumPlot()
    w.plot_fluorophore(Fluorophore("X", 480, 40, 520, 40))  # must not raise
