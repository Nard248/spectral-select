"""Step 2: Verify metadata.xlsx and average_power.xlsx."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import STEP_METADATA, PipelineState
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.steps.step1_load import _detect_columns, _find_metadata_xlsx, _find_power_xlsx
from mehsi_preprocessor.widgets.metadata_table import MetadataTable


def _status_icon(ok: bool) -> str:
    return "\u2705" if ok else "\u274C"


class Step2Metadata(AbstractStepWidget):
    """Locate and display metadata.xlsx and average_power.xlsx.

    Step 1 already auto-detects and patches exposure / power data.
    This step displays the results and allows manual overrides.
    """

    @property
    def step_index(self) -> int:
        return 2

    @property
    def title(self) -> str:
        return "Verify Metadata"

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._metadata_path: Optional[Path] = None
        self._power_path: Optional[Path] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Exposure summary ---
        grp1 = QGroupBox("Exposure Times  (metadata.xlsx)")
        g1 = QHBoxLayout()
        self._lbl_meta_status = QLabel(_status_icon(False))
        self._lbl_meta_path = QLabel("Not loaded")
        self._btn_meta_browse = QPushButton("Browse...")
        self._btn_meta_browse.clicked.connect(self._browse_metadata)
        g1.addWidget(self._lbl_meta_status)
        g1.addWidget(self._lbl_meta_path, 1)
        g1.addWidget(self._btn_meta_browse)

        g1v = QVBoxLayout(grp1)
        g1v.addLayout(g1)
        self._meta_table = MetadataTable(self)
        g1v.addWidget(self._meta_table)
        layout.addWidget(grp1)

        # --- Power summary ---
        grp2 = QGroupBox("Laser Powers  (average_power.xlsx)")
        g2 = QHBoxLayout()
        self._lbl_power_status = QLabel(_status_icon(False))
        self._lbl_power_path = QLabel("Not loaded")
        self._btn_power_browse = QPushButton("Browse...")
        self._btn_power_browse.clicked.connect(self._browse_power)
        g2.addWidget(self._lbl_power_status)
        g2.addWidget(self._lbl_power_path, 1)
        g2.addWidget(self._btn_power_browse)

        g2v = QVBoxLayout(grp2)
        g2v.addLayout(g2)
        self._power_table = MetadataTable(self)
        g2v.addWidget(self._power_table)
        layout.addWidget(grp2)

        # --- Current values per excitation ---
        grp3 = QGroupBox("Values Applied to Spectra")
        g3 = QVBoxLayout(grp3)
        self._applied_table = MetadataTable(self)
        g3.addWidget(self._applied_table)
        layout.addWidget(grp3)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Refresh from state
    # ------------------------------------------------------------------

    def _refresh_from_state(self) -> None:
        """Show what Step 1 already loaded, and allow manual file upload."""
        folder = self.state.data_folder

        # --- Exposure file ---
        if folder:
            meta_path = _find_metadata_xlsx(folder)
            if meta_path and meta_path.exists():
                self._show_metadata_file(meta_path)
            elif not self._metadata_path:
                self._lbl_meta_status.setText(_status_icon(False))
                self._lbl_meta_path.setText("Not found in data folder")
                self._meta_table.clear()

        # --- Power file ---
        if folder:
            power_path = _find_power_xlsx(folder)
            if power_path and power_path.exists():
                self._show_power_file(power_path)
            elif not self._power_path:
                self._lbl_power_status.setText(_status_icon(False))
                self._lbl_power_path.setText("Not found in data folder")
                self._power_table.clear()

        # --- Applied values table ---
        self._refresh_applied_table()

    def _refresh_applied_table(self) -> None:
        """Show a table of excitation -> exposure_time, laser_power from the
        actual ExcitationData objects."""
        spectra = self.state.raw_spectra
        if spectra is None:
            self._applied_table.clear()
            return

        rows = []
        for ex_nm in spectra.excitation_wavelengths:
            ex = spectra.get_excitation(ex_nm)
            rows.append({
                "Excitation (nm)": f"{ex_nm:.0f}",
                "Exposure Time": ex.exposure_time if ex.exposure_time is not None else "—",
                "Laser Power": ex.laser_power if ex.laser_power is not None else "—",
            })
        df = pd.DataFrame(rows)
        self._applied_table.set_dataframe(df)

    # ------------------------------------------------------------------
    # File display helpers
    # ------------------------------------------------------------------

    def _show_metadata_file(self, path: Path) -> None:
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()
            self._metadata_path = path
            self._lbl_meta_status.setText(_status_icon(True))
            self._lbl_meta_path.setText(str(path))
            self._meta_table.set_dataframe(df)
        except Exception as e:
            self._lbl_meta_status.setText(_status_icon(False))
            self._lbl_meta_path.setText(f"Error: {e}")

    def _show_power_file(self, path: Path) -> None:
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()
            self._power_path = path
            self._lbl_power_status.setText(_status_icon(True))
            self._lbl_power_path.setText(str(path))
            self._power_table.set_dataframe(df)
        except Exception as e:
            self._lbl_power_status.setText(_status_icon(False))
            self._lbl_power_path.setText(f"Error: {e}")

    # ------------------------------------------------------------------
    # Manual browse -> re-parse and re-patch
    # ------------------------------------------------------------------

    def _browse_metadata(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata.xlsx", "", "Excel Files (*.xlsx *.xls)"
        )
        if not path:
            return
        p = Path(path)
        self._show_metadata_file(p)
        self._reparse_and_patch_exposure(p)

    def _browse_power(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select average_power.xlsx", "", "Excel Files (*.xlsx *.xls)"
        )
        if not path:
            return
        p = Path(path)
        self._show_power_file(p)
        self._reparse_and_patch_power(p)

    def _reparse_and_patch_exposure(self, path: Path) -> None:
        spectra = self.state.raw_spectra
        if spectra is None:
            return
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()
            ex_col, exp_col = _detect_columns(
                df,
                excitation_hints=["Excitation", "excitation", "Wavelength",
                                  "wavelength", "Laser", "laser",
                                  "Excitation Wavelength (nm)"],
                value_hints=["Exposure", "exposure", "Time", "time",
                             "Integration", "integration", "Exposure Time"],
            )
            if not ex_col or not exp_col:
                QMessageBox.warning(
                    self, "Column Mismatch",
                    f"Could not find excitation + exposure columns.\n"
                    f"Available columns: {list(df.columns)}",
                )
                return

            times: Dict[float, float] = {}
            for _, row in df.iterrows():
                try:
                    times[float(row[ex_col])] = float(row[exp_col])
                except (ValueError, TypeError):
                    pass

            self.state.exposure_times = times
            patched = 0
            for ex_nm, ex_data in spectra.excitations.items():
                if ex_nm in times:
                    ex_data.exposure_time = times[ex_nm]
                    patched += 1
            self.state.invalidate_from(STEP_METADATA)
            self._refresh_applied_table()
            QMessageBox.information(
                self, "Exposure Applied",
                f"Patched {patched}/{spectra.n_excitations} excitations.",
            )
        except Exception as e:
            QMessageBox.warning(self, "Parse Error", str(e))

    def _reparse_and_patch_power(self, path: Path) -> None:
        spectra = self.state.raw_spectra
        if spectra is None:
            return
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()
            ex_col, pw_col = _detect_columns(
                df,
                excitation_hints=["Excitation Wavelength (nm)", "Excitation",
                                  "excitation", "Wavelength", "wavelength",
                                  "Laser"],
                value_hints=["Average Power (W)", "Power", "power",
                             "Average", "average", "Intensity", "intensity"],
            )
            if not ex_col or not pw_col:
                QMessageBox.warning(
                    self, "Column Mismatch",
                    f"Could not find excitation + power columns.\n"
                    f"Available columns: {list(df.columns)}",
                )
                return

            powers: Dict[float, float] = {}
            for _, row in df.iterrows():
                try:
                    powers[float(row[ex_col])] = float(row[pw_col])
                except (ValueError, TypeError):
                    pass

            self.state.laser_powers = powers
            patched = 0
            for ex_nm, ex_data in spectra.excitations.items():
                if ex_nm in powers:
                    ex_data.laser_power = powers[ex_nm]
                    patched += 1
            self.state.invalidate_from(STEP_METADATA)
            self._refresh_applied_table()
            QMessageBox.information(
                self, "Power Applied",
                f"Patched {patched}/{spectra.n_excitations} excitations.",
            )
        except Exception as e:
            QMessageBox.warning(self, "Parse Error", str(e))

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        self._refresh_from_state()
