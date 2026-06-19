"""Acquire / Render / Export bar: configure excitations, render, export."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from spectraforge.gui.render_ops import export_dataset, render_state
from spectraforge.gui.workers import RenderWorker


class AcquireRenderPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker = None
        self._ex = QLineEdit(",".join(f"{e:.0f}" for e in state.acquisition.excitations))
        self._status = QLabel("")
        render_btn = QPushButton("Render")
        render_btn.clicked.connect(self._on_render)
        export_btn = QPushButton("Export pkl + ground truth")
        export_btn.clicked.connect(self._on_export)

        root = QVBoxLayout(self)
        h = QHBoxLayout()
        h.addWidget(QLabel("Excitations (nm)"))
        h.addWidget(self._ex)
        root.addLayout(h)
        for b in (render_btn, export_btn, self._status):
            root.addWidget(b)

    def _apply_excitations(self):
        try:
            ex = [float(x) for x in self._ex.text().split(",") if x.strip()]
            if ex:
                self.state.acquisition.excitations = ex
        except ValueError:
            pass

    def render_now(self):
        """Synchronous render (used headless / in tests)."""
        self._apply_excitations()
        self.state.last_render = render_state(self.state)
        self._status.setText("Rendered.")

    def _on_render(self):
        self._apply_excitations()
        self._worker = RenderWorker(self.state)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(lambda m: self._status.setText(f"Render failed: {m}"))
        self._status.setText("Rendering…")
        self._worker.start()

    def _on_done(self, result):
        self.state.last_render = result
        self._status.setText(f"Rendered {result[0].n_excitations} excitations.")

    def export_to(self, out_dir):
        if self.state.last_render is None:
            self._status.setText("Render first.")
            return
        spectra, gt = self.state.last_render
        export_dataset(spectra, gt, out_dir)
        self._status.setText(f"Exported to {out_dir}")

    def _on_export(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.export_to(Path(d))
