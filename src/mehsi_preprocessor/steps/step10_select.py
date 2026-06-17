"""Step 10 — run Perturbation-Based AE band selection on the prepared model."""
from __future__ import annotations

from dataclasses import replace

from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QLabel, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.workers import SelectWorker


class Step10Select(AbstractStepWidget):
    """Tune selection parameters and run selection on the prepared model (re-runnable)."""

    @property
    def step_index(self) -> int:
        return 10

    @property
    def title(self) -> str:
        return "Select Bands"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._worker: SelectWorker | None = None

        root = QVBoxLayout(self)
        root.addWidget(QLabel("<b>Perturbation-Based AE selection.</b> "
                              "Adjust parameters and run; re-run freely without retraining."))

        form = QFormLayout()
        self._n_bands = QSpinBox(); self._n_bands.setRange(1, 500); self._n_bands.setValue(30)
        self._dim = QComboBox(); self._dim.addItems(["variance", "activation", "pca"])
        self._norm = QComboBox(); self._norm.addItems(["variance", "max_per_excitation", "none"])
        self._div = QComboBox(); self._div.addItems(["mmr", "min_distance", "none"])
        form.addRow("Bands to select", self._n_bands)
        form.addRow("Dimension method", self._dim)
        form.addRow("Normalization", self._norm)
        form.addRow("Diversity", self._div)
        root.addLayout(form)

        self._btn_run = QPushButton("Run selection")
        self._btn_run.clicked.connect(self._start)
        self._progress = QProgressBar(); self._progress.setRange(0, 0); self._progress.hide()
        self._status = QLabel("")
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Rank", "Excitation (nm)", "Emission (nm)", "Influence"])
        self._btn_export = QPushButton("Export results (CSV / JSON / TIFF)")
        self._btn_export.clicked.connect(self._export)
        self._btn_export.setEnabled(False)
        for w in (self._btn_run, self._progress, self._status, self._table, self._btn_export):
            root.addWidget(w)

    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        ready = self.state.analyzer is not None
        self._btn_run.setEnabled(ready)
        self._status.setText("" if ready else "Train or load a model in Step 9 first.")

    def _config(self):
        base = self.state.analyzer.config
        return replace(
            base,
            n_bands_to_select=self._n_bands.value(),
            dimension_selection_method=self._dim.currentText(),
            normalization_method=self._norm.currentText(),
            use_diversity_constraint=(self._div.currentText() != "none"),
            diversity_method=self._div.currentText(),
        )

    def _start(self) -> None:
        if self.state.analyzer is None:
            return
        config = self._config()
        self.state.selection_config = config
        self._btn_run.setEnabled(False)
        self._progress.show()
        self._worker = SelectWorker(self.state.analyzer, config)
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _done(self, result) -> None:
        self.state.selection_result = result
        self._progress.hide()
        self._btn_run.setEnabled(True)
        self._btn_export.setEnabled(True)
        bands = result.selected_bands
        self._table.setRowCount(len(bands))
        for r, band in enumerate(bands):
            cells = (band.rank, f"{band.excitation_nm:.0f}",
                     f"{band.emission_nm:.1f}", f"{band.influence_score:.3e}")
            for c, val in enumerate(cells):
                self._table.setItem(r, c, QTableWidgetItem(str(val)))
        self._status.setText(f"Selected {len(bands)} bands.")

    def _error(self, msg: str) -> None:
        self._progress.hide()
        self._btn_run.setEnabled(True)
        self._status.setText(f"Selection failed: {msg}")

    def _export(self) -> None:
        if self.state.selection_result is None or self.state.analyzer is None:
            return
        out = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if out:
            self.state.analyzer.save_results(out)  # ResultsManager writes CSV / JSON / TIFF
            self._status.setText(f"Exported to {out}")
