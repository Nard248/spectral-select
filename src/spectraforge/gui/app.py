"""SpectraForge ("the Forge") main window — workbench painter app."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QFileDialog, QMainWindow, QTabWidget

from spectraforge.gui.panels.acquire_render_panel import AcquireRenderPanel
from spectraforge.gui.panels.canvas_panel import CanvasPanel
from spectraforge.gui.panels.layers_panel import LayersPanel
from spectraforge.gui.panels.library_panel import LibraryPanel
from spectraforge.gui.panels.material_panel import MaterialPanel
from spectraforge.gui.project import load_project, save_project
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
        self._library = LibraryPanel(self._state)
        self._material = MaterialPanel(self._state)
        left.addTab(self._library, "Library")
        left.addTab(self._material, "Materials")
        self._add_dock("Library / Materials", left, Qt.DockWidgetArea.LeftDockWidgetArea)

        self._layers = LayersPanel(self._state)
        self._layers.changed.connect(self._canvas.refresh)
        self._add_dock("Layers", self._layers, Qt.DockWidgetArea.RightDockWidgetArea)

        self._acquire = AcquireRenderPanel(self._state)
        self._add_dock("Acquire / Render / Export", self._acquire, Qt.DockWidgetArea.BottomDockWidgetArea)

        self._build_menu()

    def _add_dock(self, title, widget, area):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("New", self._new)
        file_menu.addAction("Open…", self._open)
        file_menu.addAction("Save…", self._save)

    # ------------------------------------------------------------------
    # Project new / open / save (dialog handlers + testable cores)
    # ------------------------------------------------------------------

    def _replace_state(self, new: ForgeState) -> None:
        s = self._state
        s.height, s.width = new.height, new.width
        s.library = new.library
        s.materials = new.materials
        s.layers = new.layers
        s.acquisition = new.acquisition
        s.artifacts = new.artifacts
        s.seed = new.seed
        s.active_layer = new.active_layer
        s.last_render = None
        self._refresh_all()

    def _refresh_all(self) -> None:
        self._library.refresh()
        self._material.refresh()
        self._layers._refresh_combo()
        self._layers._refresh_list()
        self._canvas.refresh()
        self._acquire.refresh()

    def save_to(self, path) -> None:
        save_project(self._state, path)

    def open_from(self, path) -> None:
        self._replace_state(load_project(path))

    def new_project(self) -> None:
        self._replace_state(ForgeState(height=64, width=64))

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "forge_project.npz", "Forge project (*.npz)")
        if path:
            self.save_to(Path(path))

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Forge project (*.npz)")
        if path:
            self.open_from(Path(path))

    def _new(self) -> None:
        self.new_project()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpectraForge")
    win = ForgeWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
