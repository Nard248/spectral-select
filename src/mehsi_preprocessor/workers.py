"""Background workers so training / selection never freeze the UI.

``run_selection_job`` is a pure, import-safe function unit-tested without Qt. The
``QThread`` subclasses wrap the slow analyzer calls and emit progress / finished /
error signals the step widgets connect to.
"""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal


def run_selection_job(analyzer, config):
    """Run band selection on an already-prepared analyzer; returns WavelengthResult."""
    return analyzer.select(config)


class TrainWorker(QThread):
    """Runs Analyzer.prepare(data) (load-or-train + baseline) off the UI thread."""

    progress = pyqtSignal(int, int, float)   # epoch, total_epochs, loss
    finished_ok = pyqtSignal(object)         # the prepared Analyzer
    failed = pyqtSignal(str)

    def __init__(self, analyzer, data):
        super().__init__()
        self._analyzer = analyzer
        self._data = data

    def run(self):
        try:
            self._analyzer.prepare(
                self._data,
                progress_callback=lambda e, t, loss: self.progress.emit(e, t, loss),
            )
            self.finished_ok.emit(self._analyzer)
        except Exception as exc:  # surface to the UI; never crash the thread
            self.failed.emit(str(exc))


class SelectWorker(QThread):
    """Runs analyzer.select(config) off the UI thread (fast, re-runnable)."""

    finished_ok = pyqtSignal(object)         # WavelengthResult
    failed = pyqtSignal(str)

    def __init__(self, analyzer, config):
        super().__init__()
        self._analyzer = analyzer
        self._config = config

    def run(self):
        try:
            self.finished_ok.emit(run_selection_job(self._analyzer, self._config))
        except Exception as exc:
            self.failed.emit(str(exc))
