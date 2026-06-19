"""Acquire / Render / Export bar: configure excitations, render, preview slices, export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from mehsi_preprocessor.widgets.image_canvas import ImageCanvas
from spectraforge.gui.render_ops import export_dataset, render_state, validate_state
from spectraforge.gui.workers import RenderWorker, ValidateWorker


class AcquireRenderPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker = None
        self._vworker = None

        self._ex = QLineEdit(",".join(f"{e:.0f}" for e in state.acquisition.excitations))
        self._status = QLabel("")
        render_btn = QPushButton("Render")
        render_btn.clicked.connect(self._on_render)
        export_btn = QPushButton("Export pkl + ground truth")
        export_btn.clicked.connect(self._on_export)
        self._validate_btn = QPushButton("Validate selection vs ground truth")
        self._validate_btn.clicked.connect(self._on_validate)
        self._metrics = QLabel("")
        self._metrics.setWordWrap(True)

        # --- slice preview ---
        self._preview = ImageCanvas(self)
        self._ex_combo = QComboBox()
        self._ex_combo.currentIndexChanged.connect(self._update_preview)
        self._band = QSlider(Qt.Orientation.Horizontal)
        self._band.setRange(0, 0)
        self._band.valueChanged.connect(self._update_preview)

        root = QVBoxLayout(self)
        ex_row = QHBoxLayout()
        ex_row.addWidget(QLabel("Excitations (nm)"))
        ex_row.addWidget(self._ex)
        root.addLayout(ex_row)
        for b in (render_btn, export_btn, self._status, self._validate_btn, self._metrics):
            root.addWidget(b)
        prev_row = QHBoxLayout()
        prev_row.addWidget(QLabel("Excitation"))
        prev_row.addWidget(self._ex_combo)
        prev_row.addWidget(QLabel("Emission band"))
        prev_row.addWidget(self._band)
        root.addLayout(prev_row)
        root.addWidget(self._preview)

    # ------------------------------------------------------------------
    # Acquisition + render
    # ------------------------------------------------------------------

    def refresh(self):
        """Re-sync the excitation field from state (e.g. after project load)."""
        self._ex.setText(",".join(f"{e:.0f}" for e in self.state.acquisition.excitations))

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
        self._after_render()
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
        self._after_render()
        self._status.setText(f"Rendered {result[0].n_excitations} excitations.")

    # ------------------------------------------------------------------
    # Validate the band selection against ground truth
    # ------------------------------------------------------------------

    @staticmethod
    def _format_metrics(m):
        # Lead with the TIGHT peak metric (the meaningful one) and expose mask saturation; compare
        # to a random baseline before reading anything into precision (the broad mask saturates it).
        marks = "  ".join(f"{name} {'OK' if ok else '--'}" for name, ok in m["peak_hits"].items())
        return (f"peak-recovery {m['peak_recovery'] * 100:.0f}%   precision {m['precision']:.2f}   "
                f"f1 {m['f1']:.2f}\nmask covers {m['mask_coverage'] * 100:.0f}% of grid "
                f"(compare vs a random baseline)   peaks: {marks}")

    def validate_now(self, config=None):
        """Synchronous validate (used headless / in tests): render->select->score vs ground truth."""
        self._apply_excitations()
        self._metrics.setText("Validating...")
        metrics = validate_state(self.state, config=config)
        self._metrics.setText(self._format_metrics(metrics))
        return metrics

    def _on_validate(self):
        if self._vworker is not None and self._vworker.isRunning():
            return                                   # guard: never drop an in-flight QThread
        self._apply_excitations()
        self._metrics.setText("Validating... (training the autoencoder, this can take a while)")
        self._validate_btn.setEnabled(False)
        self._vworker = ValidateWorker(self.state)
        self._vworker.finished_ok.connect(self._on_validated)
        self._vworker.failed.connect(self._on_validate_failed)
        self._vworker.start()

    def _on_validated(self, metrics):
        self._metrics.setText(self._format_metrics(metrics))
        self._validate_btn.setEnabled(True)

    def _on_validate_failed(self, message):
        self._metrics.setText(f"Validate failed: {message}")
        self._validate_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Slice preview
    # ------------------------------------------------------------------

    def _excitations(self):
        return self.state.last_render[0].excitation_wavelengths if self.state.last_render else []

    def preview_slice(self, ex_index, band_index):
        """Return the 2-D image at the given excitation index + emission band index."""
        spectra = self.state.last_render[0]
        ex_nm = self._excitations()[ex_index]
        return spectra.get_excitation(ex_nm).cube[:, :, band_index]

    def _after_render(self):
        exes = self._excitations()
        self._ex_combo.blockSignals(True)
        self._ex_combo.clear()
        self._ex_combo.addItems([f"{e:.0f} nm" for e in exes])
        self._ex_combo.blockSignals(False)
        if exes:
            n_bands = self.state.last_render[0].get_excitation(exes[0]).n_bands
            self._band.blockSignals(True)
            self._band.setRange(0, n_bands - 1)
            self._band.setValue(n_bands // 2)
            self._band.blockSignals(False)
        self._update_preview()

    def _update_preview(self, *_):
        if self.state.last_render is None or not self._excitations():
            return
        ex_index = max(0, self._ex_combo.currentIndex())
        band = self._band.value()
        img = self.preview_slice(ex_index, band)
        self._preview.show_image(img)
        self._preview.set_title(f"{self._excitations()[ex_index]:.0f} nm, band {band}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

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
