"""Main application window – step-by-step wizard with sidebar navigation."""

from __future__ import annotations

import sys
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import PipelineState
from mehsi_preprocessor.steps.base import AbstractStepWidget


class PreprocessorWindow(QMainWindow):
    """Wizard-style window with a sidebar step list and a stacked content area.

    Supports step locking when PKL is imported to skip earlier steps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MEHSI Preprocessor")
        self.resize(1400, 900)

        self._state = PipelineState()
        self._steps: List[AbstractStepWidget] = []
        self._current_index: int = 0

        self._build_ui()
        self._register_steps()
        self._sidebar.setCurrentRow(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # --- Sidebar ---
        sidebar_container = QVBoxLayout()
        sidebar_label = QLabel("Steps")
        sidebar_label.setFont(QFont("", 12, QFont.Weight.Bold))
        sidebar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_container.addWidget(sidebar_label)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(200)
        self._sidebar.currentRowChanged.connect(self._on_sidebar_clicked)
        sidebar_container.addWidget(self._sidebar)

        root.addLayout(sidebar_container)

        # --- Right panel: stack + nav buttons ---
        right = QVBoxLayout()
        self._stack = QStackedWidget()
        right.addWidget(self._stack, 1)

        # Navigation buttons
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("← Previous")
        self._btn_prev.clicked.connect(self._go_prev)
        self._btn_next = QPushButton("Next →")
        self._btn_next.clicked.connect(self._go_next)
        nav.addStretch()
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._btn_next)
        right.addLayout(nav)

        root.addLayout(right, 1)

    # ------------------------------------------------------------------
    # Step registration
    # ------------------------------------------------------------------

    def _register_steps(self) -> None:
        """Instantiate and add all step widgets."""
        from mehsi_preprocessor.steps.step1_load import Step1Load
        from mehsi_preprocessor.steps.step2_metadata import Step2Metadata
        from mehsi_preprocessor.steps.step3_normalize import Step3Normalize
        from mehsi_preprocessor.steps.step4_spatial_crop import Step4SpatialCrop
        from mehsi_preprocessor.steps.step5_spectral_crop import Step5SpectralCrop
        from mehsi_preprocessor.steps.step6_draw_classes import Step6DrawClasses
        from mehsi_preprocessor.steps.step7_roi_regions import Step7ROIRegions
        from mehsi_preprocessor.steps.step8_export import Step8Export

        step_classes = [
            Step1Load,
            Step2Metadata,
            Step3Normalize,
            Step4SpatialCrop,
            Step5SpectralCrop,
            Step6DrawClasses,
            Step7ROIRegions,
            Step8Export,
        ]

        for cls in step_classes:
            widget = cls(self._state, parent=self)
            self._steps.append(widget)
            self._stack.addWidget(widget)

            item = QListWidgetItem(f"{widget.step_index}. {widget.title}")
            self._sidebar.addItem(item)

        # Connect PKL import signal from Step1
        step1 = self._steps[0]
        if hasattr(step1, "pkl_imported"):
            step1.pkl_imported.connect(self._on_pkl_imported)

        self._update_nav_buttons()

    # ------------------------------------------------------------------
    # PKL import handling
    # ------------------------------------------------------------------

    def _on_pkl_imported(self, start_step: int) -> None:
        """Handle PKL import completion – update sidebar and navigate."""
        self._update_sidebar_for_locked_steps()
        # Navigate to the start step (0-indexed)
        target_index = start_step - 1
        self._sidebar.setCurrentRow(target_index)

    def _update_sidebar_for_locked_steps(self) -> None:
        """Update sidebar items to show locked steps with visual indicator."""
        for i in range(self._sidebar.count()):
            item = self._sidebar.item(i)
            step_num = i + 1  # 1-indexed
            widget = self._steps[i]

            if self._state.is_step_locked(step_num):
                # Locked step: gray out and add lock indicator
                item.setText(f"🔒 {step_num}. {widget.title}")
                item.setForeground(QColor(128, 128, 128))
            else:
                # Unlocked step: restore normal appearance
                item.setText(f"{step_num}. {widget.title}")
                item.setForeground(QColor(0, 0, 0))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate_to(self, index: int) -> None:
        if index == self._current_index:
            return
        if not (0 <= index < len(self._steps)):
            return

        # Check if target step is locked
        target_step = index + 1  # 1-indexed
        if self._state.is_step_locked(target_step):
            QMessageBox.information(
                self,
                "Step Locked",
                f"Step {target_step} is locked because data was imported from PKL.\n"
                f"This step was skipped during import."
            )
            # Restore sidebar selection
            self._sidebar.blockSignals(True)
            self._sidebar.setCurrentRow(self._current_index)
            self._sidebar.blockSignals(False)
            return

        # Leave current step
        current = self._steps[self._current_index]
        if not current.on_leave():
            # Step refused to leave – restore sidebar selection
            self._sidebar.blockSignals(True)
            self._sidebar.setCurrentRow(self._current_index)
            self._sidebar.blockSignals(False)
            return

        self._current_index = index
        self._stack.setCurrentIndex(index)
        self._steps[index].on_enter()
        self._update_nav_buttons()

    def _on_sidebar_clicked(self, row: int) -> None:
        self._navigate_to(row)

    def _go_prev(self) -> None:
        # Find previous unlocked step
        target = self._current_index - 1
        while target >= 0 and self._state.is_step_locked(target + 1):
            target -= 1
        if target >= 0:
            self._sidebar.setCurrentRow(target)

    def _go_next(self) -> None:
        self._sidebar.setCurrentRow(
            min(self._current_index + 1, len(self._steps) - 1)
        )

    def _update_nav_buttons(self) -> None:
        # Previous: enabled if there's an unlocked step before current
        prev_enabled = False
        for i in range(self._current_index - 1, -1, -1):
            if not self._state.is_step_locked(i + 1):
                prev_enabled = True
                break
        self._btn_prev.setEnabled(prev_enabled)
        self._btn_next.setEnabled(self._current_index < len(self._steps) - 1)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("MEHSI Preprocessor")
    window = PreprocessorWindow()
    window.show()
    # Trigger first step on_enter
    if window._steps:
        window._steps[0].on_enter()
    app.exec()
