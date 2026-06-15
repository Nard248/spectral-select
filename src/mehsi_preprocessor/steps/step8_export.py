"""Step 8: Export PKL, PNG mask, and ROI JSON."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import PipelineState
from mehsi_preprocessor.steps.base import AbstractStepWidget


class Step8Export(AbstractStepWidget):

    @property
    def step_index(self) -> int:
        return 8

    @property
    def title(self) -> str:
        return "Export"

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._output_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Output folder
        grp_dir = QGroupBox("Output Folder")
        dl = QHBoxLayout(grp_dir)
        self._lbl_dir = QLabel("Not selected")
        self._btn_dir = QPushButton("Browse...")
        self._btn_dir.clicked.connect(self._browse_dir)
        dl.addWidget(self._lbl_dir, 1)
        dl.addWidget(self._btn_dir)
        layout.addWidget(grp_dir)

        # Checkboxes
        grp_opts = QGroupBox("Outputs")
        ol = QVBoxLayout(grp_opts)
        self._chk_masked = QCheckBox("Masked PKL  (NaN outside mask)")
        self._chk_masked.setChecked(True)
        self._chk_unmasked = QCheckBox("Unmasked PKL  (full cubes)")
        self._chk_unmasked.setChecked(True)
        self._chk_png = QCheckBox("Mask PNG  (coloured by class)")
        self._chk_png.setChecked(True)
        self._chk_json = QCheckBox("ROI JSON  (structured coordinates)")
        self._chk_json.setChecked(True)
        ol.addWidget(self._chk_masked)
        ol.addWidget(self._chk_unmasked)
        ol.addWidget(self._chk_png)
        ol.addWidget(self._chk_json)
        layout.addWidget(grp_opts)

        # Export button
        self._btn_export = QPushButton("Export")
        self._btn_export.setStyleSheet("font-weight:bold; padding:8px;")
        self._btn_export.clicked.connect(self._export)
        layout.addWidget(self._btn_export)

        # Status
        self._lbl_status = QLabel("")
        layout.addWidget(self._lbl_status)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Browse
    # ------------------------------------------------------------------

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self._output_dir = Path(d)
            self._lbl_dir.setText(str(self._output_dir))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self) -> None:
        from mehsi_preprocessor.processing.export import (
            export_mask_png,
            export_masked_pkl,
            export_roi_json,
            export_unmasked_pkl,
        )

        if self._output_dir is None:
            QMessageBox.warning(self, "No Folder", "Select an output folder first.")
            return

        spectra = self.state.current_spectra
        if spectra is None:
            QMessageBox.warning(self, "No Data", "No spectra available to export.")
            return

        out = self._output_dir
        out.mkdir(parents=True, exist_ok=True)
        results = []

        try:
            if self._chk_masked.isChecked():
                mask = self.state.class_mask
                if mask is None:
                    QMessageBox.warning(self, "No Mask", "Draw a class mask in Step 6.")
                    return
                p = out / "spectra_masked.pkl"
                export_masked_pkl(spectra, mask, p)
                results.append(f"Masked PKL: {p}")

            if self._chk_unmasked.isChecked():
                p = out / "spectra_unmasked.pkl"
                export_unmasked_pkl(spectra, p)
                results.append(f"Unmasked PKL: {p}")

            if self._chk_png.isChecked():
                mask = self.state.class_mask
                if mask is not None:
                    p = out / "class_mask.png"
                    export_mask_png(mask, self.state.class_definitions, p)
                    results.append(f"Mask PNG: {p}")
                else:
                    results.append("Mask PNG: skipped (no mask)")

            if self._chk_json.isChecked():
                p = out / "roi_regions.json"
                export_roi_json(
                    self.state.roi_regions,
                    self.state.class_definitions,
                    p,
                )
                results.append(f"ROI JSON: {p}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            return

        self._lbl_status.setText("Export complete:\n" + "\n".join(results))
        QMessageBox.information(self, "Done", "Export complete!")

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        # Show summary of what's available
        lines = []
        spectra = self.state.current_spectra
        if spectra:
            h, w = spectra.spatial_shape
            lines.append(f"Spectra: {spectra.n_excitations} excitations, {h}x{w}")
        if self.state.class_mask is not None:
            n_classes = len(set(self.state.class_mask.flat) - {0})
            lines.append(f"Mask: {n_classes} classes defined")
        lines.append(f"ROI regions: {len(self.state.roi_regions)}")
        self._lbl_status.setText("\n".join(lines))
