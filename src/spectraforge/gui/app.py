"""SpectraForge ("the Forge") main window — workbench painter app."""
from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QMainWindow, QTabWidget

from spectraforge.gui.panels.acquire_render_panel import AcquireRenderPanel
from spectraforge.gui.panels.canvas_panel import CanvasPanel
from spectraforge.gui.panels.layers_panel import LayersPanel
from spectraforge.gui.panels.library_panel import LibraryPanel
from spectraforge.gui.panels.material_panel import MaterialPanel
from spectraforge.gui.state import ForgeState


class ForgeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraForge")
        self.resize(1500, 950)
        self._state = ForgeState(height=64, width=64)

        self._canvas = CanvasPanel(self._state)
        self.setCentralWidget(self._canvas)

        left = QTabWidget()
        left.addTab(LibraryPanel(self._state), "Library")
        left.addTab(MaterialPanel(self._state), "Materials")
        self._add_dock("Library / Materials", left, Qt.DockWidgetArea.LeftDockWidgetArea)

        self._layers = LayersPanel(self._state)
        self._layers.changed.connect(self._canvas.refresh)
        self._add_dock("Layers", self._layers, Qt.DockWidgetArea.RightDockWidgetArea)

        self._add_dock(
            "Acquire / Render / Export",
            AcquireRenderPanel(self._state),
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )

    def _add_dock(self, title, widget, area):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpectraForge")
    win = ForgeWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
