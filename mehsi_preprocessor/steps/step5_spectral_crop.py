"""Step 5: Spectral / emission crop (Rayleigh cutoff + manual range)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import STEP_SPECTRAL_CROP, PipelineState
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.widgets.spectral_bar_chart import SpectralBarChart


class Step5SpectralCrop(AbstractStepWidget):

    @property
    def step_index(self) -> int:
        return 5

    @property
    def title(self) -> str:
        return "Spectral Crop"

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._manual_spins: Dict[float, Tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Rayleigh mode ---
        grp_ray = QGroupBox("Rayleigh Cutoff")
        rl = QHBoxLayout(grp_ray)
        self._chk_rayleigh = QCheckBox("Apply Rayleigh cutoff")
        self._chk_rayleigh.setChecked(True)
        self._spin_offset = QSpinBox()
        self._spin_offset.setRange(0, 200)
        self._spin_offset.setValue(30)
        self._spin_offset.setSuffix(" nm")
        self._chk_second_order = QCheckBox("Remove second-order")
        self._chk_second_order.setChecked(True)
        rl.addWidget(self._chk_rayleigh)
        rl.addWidget(QLabel("Offset:"))
        rl.addWidget(self._spin_offset)
        rl.addWidget(self._chk_second_order)
        rl.addStretch()
        layout.addWidget(grp_ray)

        # --- Manual mode ---
        grp_man = QGroupBox("Manual Emission Range (per excitation)")
        self._manual_layout = QFormLayout(grp_man)
        self._manual_container = grp_man
        layout.addWidget(grp_man)

        # --- Apply + bar chart ---
        btn_row = QHBoxLayout()
        self._btn_preview = QPushButton("Preview")
        self._btn_preview.clicked.connect(self._preview)
        self._btn_apply = QPushButton("Apply Spectral Crop")
        self._btn_apply.clicked.connect(self._apply)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_preview)
        btn_row.addWidget(self._btn_apply)
        layout.addLayout(btn_row)

        # Excitation selector + bar chart
        bar_row = QHBoxLayout()
        bar_row.addWidget(QLabel("Show excitation:"))
        self._ex_combo = QComboBox()
        self._ex_combo.currentIndexChanged.connect(self._update_bar)
        bar_row.addWidget(self._ex_combo, 1)
        layout.addLayout(bar_row)

        self._bar_chart = SpectralBarChart(self)
        layout.addWidget(self._bar_chart)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Populate manual spinboxes
    # ------------------------------------------------------------------

    def _populate_manual_ranges(self) -> None:
        spectra = self._source_spectra()
        if spectra is None:
            return

        # Clear old
        while self._manual_layout.rowCount():
            self._manual_layout.removeRow(0)
        self._manual_spins.clear()
        self._ex_combo.clear()

        for ex_nm in spectra.excitation_wavelengths:
            ex = spectra.get_excitation(ex_nm)
            wl = ex.emission_wavelengths
            wl_min, wl_max = min(wl), max(wl)

            sp_min = QDoubleSpinBox()
            sp_min.setRange(wl_min, wl_max)
            sp_min.setValue(wl_min)
            sp_min.setSuffix(" nm")

            sp_max = QDoubleSpinBox()
            sp_max.setRange(wl_min, wl_max)
            sp_max.setValue(wl_max)
            sp_max.setSuffix(" nm")

            row = QHBoxLayout()
            row.addWidget(QLabel("Min:"))
            row.addWidget(sp_min)
            row.addWidget(QLabel("Max:"))
            row.addWidget(sp_max)
            container = QWidget()
            container.setLayout(row)

            self._manual_layout.addRow(f"Ex {ex_nm:.0f} nm", container)
            self._manual_spins[ex_nm] = (sp_min, sp_max)
            self._ex_combo.addItem(f"{ex_nm:.0f} nm")

    # ------------------------------------------------------------------
    # Compute keep masks
    # ------------------------------------------------------------------

    def _compute_keep_masks(self):
        """Return {ex_nm: (wavelengths, keep_mask)} for preview."""
        spectra = self._source_spectra()
        if spectra is None:
            return {}

        masks = {}
        for ex_nm in spectra.excitation_wavelengths:
            wl = np.array(spectra.get_excitation(ex_nm).emission_wavelengths)
            keep = np.ones(len(wl), dtype=bool)

            # Rayleigh
            if self._chk_rayleigh.isChecked():
                offset = self._spin_offset.value()
                keep &= wl >= (ex_nm + offset)
                if self._chk_second_order.isChecked():
                    so_min = 2 * ex_nm - offset
                    so_max = 2 * ex_nm + offset
                    keep &= (wl < so_min) | (wl > so_max)

            # Manual range
            if ex_nm in self._manual_spins:
                sp_min, sp_max = self._manual_spins[ex_nm]
                keep &= (wl >= sp_min.value()) & (wl <= sp_max.value())

            masks[ex_nm] = (wl.tolist(), keep)

        return masks

    # ------------------------------------------------------------------
    # Preview / apply
    # ------------------------------------------------------------------

    def _preview(self) -> None:
        self._update_bar()

    def _update_bar(self) -> None:
        masks = self._compute_keep_masks()
        idx = self._ex_combo.currentIndex()
        spectra = self._source_spectra()
        if spectra is None or idx < 0:
            return
        ex_nm = spectra.excitation_wavelengths[idx]
        if ex_nm in masks:
            wl, keep = masks[ex_nm]
            self._bar_chart.update_chart(wl, keep, ex_nm)

    def _apply(self) -> None:
        from mehsi_preprocessor.processing.spectral_filter import (
            apply_manual_emission_crop,
            apply_rayleigh_cutoff,
        )

        spectra = self._source_spectra()
        if spectra is None:
            QMessageBox.warning(self, "No Data", "No spectra available.")
            return

        result = spectra

        # Apply Rayleigh
        if self._chk_rayleigh.isChecked():
            result = apply_rayleigh_cutoff(
                result,
                cutoff_offset=self._spin_offset.value(),
                apply_second_order=self._chk_second_order.isChecked(),
            )

        # Apply manual ranges
        ranges: Dict[float, Tuple[float, float]] = {}
        for ex_nm, (sp_min, sp_max) in self._manual_spins.items():
            ranges[ex_nm] = (sp_min.value(), sp_max.value())
        if ranges:
            result = apply_manual_emission_crop(result, ranges)

        self.state.filtered_spectra = result
        self.state.invalidate_from(STEP_SPECTRAL_CROP)

        # Summary
        total_kept = sum(
            result.get_excitation(e).n_bands for e in result.excitation_wavelengths
        )
        QMessageBox.information(
            self, "Spectral Crop Applied",
            f"Total bands kept: {total_kept}",
        )
        self._update_bar()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _source_spectra(self):
        return (
            self.state.cropped_spectra
            or self.state.normalized_spectra
            or self.state.raw_spectra
        )

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        self._populate_manual_ranges()
