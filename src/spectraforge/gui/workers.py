"""Background render worker for the Forge GUI."""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from spectraforge.gui.render_ops import render_state


def run_render_job(state):
    """Pure render entry point (unit-tested without Qt)."""
    return render_state(state)


class RenderWorker(QThread):
    finished_ok = pyqtSignal(object)   # (SpectraData, GroundTruth)
    failed = pyqtSignal(str)

    def __init__(self, state):
        super().__init__()
        self._state = state

    def run(self):
        try:
            self.finished_ok.emit(run_render_job(self._state))
        except Exception as exc:  # surface to the UI; never crash the thread
            self.failed.emit(str(exc))
