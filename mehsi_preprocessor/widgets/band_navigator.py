"""Band navigation widget – excitation dropdown + emission slider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from spectral_select.types import SpectraData


class BandNavigator(QWidget):
    """Dropdown for excitation + slider/spinbox for emission band index.

    Signals:
        band_changed(excitation_nm: float, band_idx: int)
    """

    band_changed = pyqtSignal(float, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # --- Excitation dropdown ---
        self._ex_combo = QComboBox()
        self._ex_combo.currentIndexChanged.connect(self._on_excitation_changed)

        # --- Emission slider + spinbox ---
        self._em_slider = QSlider(Qt.Orientation.Horizontal)
        self._em_slider.setMinimum(0)
        self._em_slider.valueChanged.connect(self._on_slider_moved)

        self._em_spin = QSpinBox()
        self._em_spin.setMinimum(0)
        self._em_spin.valueChanged.connect(self._on_spin_changed)

        self._em_label = QLabel("Band 0 / 0  (— nm)")

        # Layout
        top = QHBoxLayout()
        top.addWidget(QLabel("Excitation:"))
        top.addWidget(self._ex_combo, 1)

        bot = QHBoxLayout()
        bot.addWidget(QLabel("Emission band:"))
        bot.addWidget(self._em_slider, 1)
        bot.addWidget(self._em_spin)
        bot.addWidget(self._em_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addLayout(bot)

        # Internal bookkeeping
        self._spectra: Optional[SpectraData] = None
        self._excitations: List[float] = []
        self._current_wavelengths: List[float] = []
        self._updating = False  # guard against re-entrancy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_spectra(self, spectra: SpectraData) -> None:
        """Populate from a SpectraData container."""
        self._spectra = spectra
        self._excitations = spectra.excitation_wavelengths

        self._updating = True
        self._ex_combo.clear()
        for ex in self._excitations:
            self._ex_combo.addItem(f"{ex:.1f} nm")
        self._updating = False

        if self._excitations:
            self._ex_combo.setCurrentIndex(0)
            self._on_excitation_changed(0)

    @property
    def current_excitation(self) -> Optional[float]:
        idx = self._ex_combo.currentIndex()
        if 0 <= idx < len(self._excitations):
            return self._excitations[idx]
        return None

    @property
    def current_band_index(self) -> int:
        return self._em_slider.value()

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_excitation_changed(self, index: int) -> None:
        if self._updating or self._spectra is None:
            return
        if not (0 <= index < len(self._excitations)):
            return

        ex_nm = self._excitations[index]
        ex_data = self._spectra.get_excitation(ex_nm)
        n_bands = ex_data.n_bands
        self._current_wavelengths = ex_data.emission_wavelengths

        self._updating = True
        self._em_slider.setMaximum(max(n_bands - 1, 0))
        self._em_spin.setMaximum(max(n_bands - 1, 0))
        self._em_slider.setValue(0)
        self._em_spin.setValue(0)
        self._updating = False

        self._update_label(0)
        self.band_changed.emit(ex_nm, 0)

    def _on_slider_moved(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self._em_spin.setValue(value)
        self._updating = False
        self._update_label(value)
        ex = self.current_excitation
        if ex is not None:
            self.band_changed.emit(ex, value)

    def _on_spin_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self._em_slider.setValue(value)
        self._updating = False
        self._update_label(value)
        ex = self.current_excitation
        if ex is not None:
            self.band_changed.emit(ex, value)

    def _update_label(self, band_idx: int) -> None:
        n = len(self._current_wavelengths)
        if 0 <= band_idx < n:
            wl = self._current_wavelengths[band_idx]
            self._em_label.setText(f"Band {band_idx} / {n - 1}  ({wl:.1f} nm)")
        else:
            self._em_label.setText(f"Band {band_idx} / {max(n - 1, 0)}")
