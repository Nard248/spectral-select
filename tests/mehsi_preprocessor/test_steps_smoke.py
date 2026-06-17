"""Offscreen smoke tests: the new wizard steps build and react to empty state.

Run headless via QT_QPA_PLATFORM=offscreen (set below so the file is import-safe).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from mehsi_preprocessor.state import PipelineState

_app = QApplication.instance() or QApplication([])


def test_step9_builds():
    from mehsi_preprocessor.steps.step9_train import Step9Train
    w = Step9Train(PipelineState())
    assert w.step_index == 9
    assert "Train" in w.title
    w.on_enter()  # must not raise on empty state


def test_step10_builds_and_gates_on_model():
    from mehsi_preprocessor.steps.step10_select import Step10Select
    state = PipelineState()
    w = Step10Select(state)
    assert w.step_index == 10
    assert "Select" in w.title
    w.on_enter()  # no model yet -> run disabled
    assert w._btn_run.isEnabled() is False
