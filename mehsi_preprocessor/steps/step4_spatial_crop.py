"""Step 4: Spatial crop using rectangle selection."""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import STEP_SPATIAL_CROP, PipelineState
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.widgets.image_canvas import ImageCanvas
from mehsi_preprocessor.widgets.rect_selector import RectSelector


class Step4SpatialCrop(AbstractStepWidget):

    @property
    def step_index(self) -> int:
        return 4

    @property
    def title(self) -> str:
        return "Spatial Crop"

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._pending_rect: Optional[Tuple[int, int, int, int]] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Info / controls
        ctrl = QGroupBox("Crop Region")
        cl = QHBoxLayout(ctrl)
        self._lbl_coords = QLabel("Draw a rectangle on the image below")
        self._btn_apply = QPushButton("Apply Crop")
        self._btn_apply.clicked.connect(self._apply_crop)
        self._btn_apply.setEnabled(False)
        cl.addWidget(self._lbl_coords, 1)
        cl.addWidget(self._btn_apply)
        layout.addWidget(ctrl)

        # Side-by-side: selector + preview
        canvases = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(QLabel("Select crop region:"))
        self._selector = RectSelector(self)
        self._selector.rect_selected.connect(self._on_rect)
        left.addWidget(self._selector, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Preview:"))
        self._preview = ImageCanvas(self)
        right.addWidget(self._preview, 1)

        canvases.addLayout(left)
        canvases.addLayout(right)
        layout.addLayout(canvases, 1)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_rect(self, r0: int, r1: int, c0: int, c1: int) -> None:
        self._pending_rect = (r0, r1, c0, c1)
        self._lbl_coords.setText(
            f"ROI: rows [{r0}, {r1}), cols [{c0}, {c1})  "
            f"({r1 - r0} x {c1 - c0} px)"
        )
        self._btn_apply.setEnabled(True)

        # Show quick preview
        self._show_preview()

    def _show_preview(self) -> None:
        from spectral_select.widgets import create_display_image

        spectra = self._source_spectra()
        if spectra is None or self._pending_rect is None:
            return
        r0, r1, c0, c1 = self._pending_rect
        ex_nm = spectra.excitation_wavelengths[0]
        cube = spectra.get_excitation(ex_nm).cube
        cropped = cube[r0:r1, c0:c1, :]
        self._preview.show_image(create_display_image(cropped))
        self._preview.set_title("Cropped preview")

    def _apply_crop(self) -> None:
        from mehsi_preprocessor.processing.cropping import spatial_crop

        spectra = self._source_spectra()
        if spectra is None:
            QMessageBox.warning(self, "No Data", "No spectra available to crop.")
            return
        if self._pending_rect is None:
            return

        cropped = spatial_crop(spectra, self._pending_rect)
        self.state.crop_roi = self._pending_rect
        self.state.cropped_spectra = cropped
        self.state.invalidate_from(STEP_SPATIAL_CROP)

        h, w = cropped.spatial_shape
        self._lbl_coords.setText(
            f"Crop applied: {h} x {w} px  (origin reset to 0,0)"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _source_spectra(self):
        """Return the best available spectra before this step."""
        return self.state.normalized_spectra or self.state.raw_spectra

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        from spectral_select.widgets import create_display_image

        spectra = self._source_spectra()
        if spectra is None:
            return

        ex_nm = spectra.excitation_wavelengths[0]
        cube = spectra.get_excitation(ex_nm).cube
        img = create_display_image(cube)
        self._selector.show_image(img)
        self._selector.set_title(f"Ex {ex_nm:.0f} nm – draw crop region")
        self._selector.enable_selector()
