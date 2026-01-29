"""Step 3: Exposure-time and laser-power normalization with before/after view."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import STEP_NORMALIZE, PipelineState
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.widgets.band_navigator import BandNavigator
from mehsi_preprocessor.widgets.image_canvas import ImageCanvas


class Step3Normalize(AbstractStepWidget):

    @property
    def step_index(self) -> int:
        return 3

    @property
    def title(self) -> str:
        return "Normalize"

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Options
        opts = QGroupBox("Normalization Options")
        ol = QHBoxLayout(opts)
        self._chk_exposure = QCheckBox("By Exposure Time")
        self._chk_exposure.setChecked(True)
        self._chk_power = QCheckBox("By Laser Power")
        self._chk_power.setChecked(True)
        self._btn_apply = QPushButton("Apply Normalization")
        self._btn_apply.clicked.connect(self._apply)
        ol.addWidget(self._chk_exposure)
        ol.addWidget(self._chk_power)
        ol.addStretch()
        ol.addWidget(self._btn_apply)
        layout.addWidget(opts)

        # Shared navigator
        self._navigator = BandNavigator(self)
        self._navigator.band_changed.connect(self._refresh_images)
        layout.addWidget(self._navigator)

        # Side-by-side canvases
        canvases = QHBoxLayout()
        self._canvas_before = ImageCanvas(self)
        self._canvas_after = ImageCanvas(self)
        lbl_b = QLabel("Before")
        lbl_a = QLabel("After")
        lbl_b.setStyleSheet("font-weight:bold")
        lbl_a.setStyleSheet("font-weight:bold")

        left = QVBoxLayout()
        left.addWidget(lbl_b)
        left.addWidget(self._canvas_before, 1)
        right = QVBoxLayout()
        right.addWidget(lbl_a)
        right.addWidget(self._canvas_after, 1)

        canvases.addLayout(left)
        canvases.addLayout(right)
        layout.addLayout(canvases, 1)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _apply(self) -> None:
        from mehsi_preprocessor.processing.normalization import normalize_spectra

        src = self.state.raw_spectra
        if src is None:
            QMessageBox.warning(self, "No Data", "Load data first (Step 1).")
            return

        # Warn about missing metadata
        warnings = []
        if self._chk_exposure.isChecked():
            missing = [
                f"{ex:.0f}" for ex, ed in src.excitations.items()
                if ed.exposure_time is None or ed.exposure_time == 0
            ]
            if missing:
                warnings.append(
                    f"Exposure time missing for: {', '.join(missing)} nm.\n"
                    f"Those excitations will NOT be normalized by exposure."
                )

        if self._chk_power.isChecked():
            missing = [
                f"{ex:.0f}" for ex, ed in src.excitations.items()
                if ed.laser_power is None or ed.laser_power == 0
            ]
            if missing:
                warnings.append(
                    f"Laser power missing for: {', '.join(missing)} nm.\n"
                    f"Those excitations will NOT be normalized by power."
                )

        if warnings:
            proceed = QMessageBox.question(
                self, "Missing Metadata",
                "\n\n".join(warnings) + "\n\nProceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if proceed != QMessageBox.StandardButton.Yes:
                return

        normalized = normalize_spectra(
            src,
            by_exposure=self._chk_exposure.isChecked(),
            by_laser_power=self._chk_power.isChecked(),
        )
        self.state.normalized_spectra = normalized
        self.state.invalidate_from(STEP_NORMALIZE)

        # Refresh navigator to use normalized data for "after"
        self._navigator.set_spectra(src)

    def _refresh_images(self, excitation: float, band_idx: int) -> None:
        from spectral_select.widgets import create_display_image

        raw = self.state.raw_spectra
        if raw is not None:
            cube = raw.get_excitation(excitation).cube
            self._canvas_before.show_image(create_display_image(cube, band_index=band_idx))
            self._canvas_before.set_title(f"Raw – Ex {excitation:.0f}")

        norm = self.state.normalized_spectra
        if norm is not None:
            cube_n = norm.get_excitation(excitation).cube
            self._canvas_after.show_image(create_display_image(cube_n, band_index=band_idx))
            self._canvas_after.set_title(f"Normalized – Ex {excitation:.0f}")

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        src = self.state.raw_spectra
        if src is not None:
            self._navigator.set_spectra(src)
