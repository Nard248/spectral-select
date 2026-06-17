"""Step 9 — train the autoencoder (or load a pretrained model)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QLabel, QProgressBar,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout,
)

from spectral_select import Analyzer, Config

from mehsi_preprocessor.state import STEP_TRAIN
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.workers import TrainWorker


class Step9Train(AbstractStepWidget):
    """Train a new CAE on the preprocessed data, or load an existing ``.pth``."""

    @property
    def step_index(self) -> int:
        return 9

    @property
    def title(self) -> str:
        return "Train Autoencoder"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._worker: TrainWorker | None = None
        self._model_path: Path | None = None

        root = QVBoxLayout(self)
        root.addWidget(QLabel("<b>Train the autoencoder</b> (or load a saved model), "
                              "then continue to Step 10 to select bands."))

        # --- Train-new group ---
        self._rb_train = QRadioButton("Train a new model")
        self._rb_train.setChecked(True)
        form = QFormLayout()
        self._epochs = QSpinBox(); self._epochs.setRange(1, 2000); self._epochs.setValue(50)
        self._lr = QDoubleSpinBox(); self._lr.setDecimals(5)
        self._lr.setRange(0.00001, 1.0); self._lr.setSingleStep(0.0001); self._lr.setValue(0.001)
        form.addRow("Epochs", self._epochs)
        form.addRow("Learning rate", self._lr)
        box = QGroupBox("New model")
        box_layout = QVBoxLayout(); box_layout.addWidget(self._rb_train); box_layout.addLayout(form)
        box.setLayout(box_layout)
        root.addWidget(box)

        # --- Load-pretrained group ---
        self._rb_load = QRadioButton("Load a pretrained model (.pth)")
        self._btn_browse = QPushButton("Browse…")
        self._btn_browse.clicked.connect(self._browse)
        load_box = QGroupBox("Pretrained model")
        load_layout = QVBoxLayout()
        load_layout.addWidget(self._rb_load)
        load_layout.addWidget(self._btn_browse)
        self._lbl_path = QLabel("(no file selected)")
        load_layout.addWidget(self._lbl_path)
        load_box.setLayout(load_layout)
        root.addWidget(load_box)

        # --- Action + progress ---
        self._btn_train = QPushButton("Train / Load")
        self._btn_train.clicked.connect(self._start)
        self._progress = QProgressBar()
        self._status = QLabel("No model yet.")
        for w in (self._btn_train, self._progress, self._status):
            root.addWidget(w)
        root.addStretch(1)

    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        ready = self.state.analyzer is not None
        self._status.setText("Model ready ✓ — continue to Step 10." if ready else "No model yet.")

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select model", "", "PyTorch model (*.pth)")
        if path:
            self._model_path = Path(path)
            self._lbl_path.setText(self._model_path.name)
            self._rb_load.setChecked(True)

    def _build_config(self) -> Config:
        spec = self.state.current_spectra
        return Config(
            sample_name=(spec.sample_name if spec is not None else "gui"),
            training_epochs=self._epochs.value(),
            training_lr=self._lr.value(),
            model_path=(self._model_path if self._rb_load.isChecked() else None),
            device="cpu",
        )

    def _start(self) -> None:
        data = self.state.current_spectra
        if data is None:
            self._status.setText("No preprocessed data — complete steps 1–7 first.")
            return
        if self._rb_load.isChecked() and self._model_path is None:
            self._status.setText("Choose a .pth file or switch to 'Train a new model'.")
            return

        analyzer = Analyzer(self._build_config())
        self._btn_train.setEnabled(False)
        if self._rb_load.isChecked():
            self._progress.setRange(0, 0)  # busy: loading is quick, no epochs
        else:
            self._progress.setRange(0, self._epochs.value())
            self._progress.setValue(0)

        self._worker = TrainWorker(analyzer, data)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _on_progress(self, epoch: int, total: int, loss: float) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(epoch)
        self._status.setText(f"Epoch {epoch}/{total} — loss {loss:.4f}")

    def _done(self, analyzer) -> None:
        self.state.invalidate_from(STEP_TRAIN)  # clear any stale selection result
        self.state.analyzer = analyzer
        self.state.training_losses = list(getattr(analyzer, "_training_losses", []) or [])
        self.state.model_source = "loaded" if self._model_path and self._rb_load.isChecked() else "trained"
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self._btn_train.setEnabled(True)
        self._status.setText("Model ready ✓ — continue to Step 10.")

    def _error(self, msg: str) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(0)
        self._btn_train.setEnabled(True)
        self._status.setText(f"Training failed: {msg}")
