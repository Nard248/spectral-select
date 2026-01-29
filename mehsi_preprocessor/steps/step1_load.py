"""Step 1: Load raw .im3 data from a folder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import STEP_LOAD, PipelineState
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.widgets.band_navigator import BandNavigator
from mehsi_preprocessor.widgets.image_canvas import ImageCanvas


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_metadata_xlsx(folder: Path) -> Optional[Path]:
    """Auto-detect metadata.xlsx in the data folder."""
    for name in ("metadata.xlsx", "Metadata.xlsx", "metadata.xls"):
        p = folder / name
        if p.exists():
            return p
    return None


def _find_power_xlsx(folder: Path) -> Optional[Path]:
    """Auto-detect average_power.xlsx in the data folder or TLS Scans/."""
    for candidate in (
        folder / "TLS Scans" / "average_power.xlsx",
        folder / "tls_scans" / "average_power.xlsx",
        folder / "TLS Scans" / "Average_Power.xlsx",
        folder / "average_power.xlsx",
    ):
        if candidate.exists():
            return candidate
    return None


def _detect_columns(df, excitation_hints: list, value_hints: list):
    """Find the best-matching excitation and value columns.

    Strategy: try exact-name match first, then substring match.
    """
    cols = list(df.columns)

    def _find(hints):
        # Exact match
        for h in hints:
            if h in cols:
                return h
        # Substring (case-insensitive)
        for h in hints:
            for c in cols:
                if h.lower() in c.lower():
                    return c
        return None

    return _find(excitation_hints), _find(value_hints)


def _parse_exposure_times(folder: Path) -> Dict[float, float]:
    """Read metadata.xlsx and return {excitation_nm: exposure_time}."""
    import pandas as pd

    meta_path = _find_metadata_xlsx(folder)
    if meta_path is None:
        return {}
    try:
        df = pd.read_excel(meta_path)
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
            return {}
        result = {}
        for _, row in df.iterrows():
            try:
                result[float(row[ex_col])] = float(row[exp_col])
            except (ValueError, TypeError):
                pass
        return result
    except Exception:
        return {}


def _parse_laser_powers(folder: Path) -> Dict[float, float]:
    """Read average_power.xlsx and return {excitation_nm: power}."""
    import pandas as pd

    power_path = _find_power_xlsx(folder)
    if power_path is None:
        return {}
    try:
        df = pd.read_excel(power_path)
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
            return {}
        result = {}
        for _, row in df.iterrows():
            try:
                result[float(row[ex_col])] = float(row[pw_col])
            except (ValueError, TypeError):
                pass
        return result
    except Exception:
        return {}


# ------------------------------------------------------------------
# Background loader thread
# ------------------------------------------------------------------

class _LoaderThread(QThread):
    """Loads SpectraData in a background thread.

    Bypasses ``SpectraData.from_raw()`` because the underlying
    ``HyperspectralDataLoader.load_data(apply_cutoff=False)`` returns an
    empty dict (it only populates ``self.data`` inside ``_process_data``
    which is skipped when cutoff is off).  Instead we read ``raw_data``
    directly and build ``SpectraData`` ourselves.
    """

    finished = pyqtSignal(object)   # (SpectraData, summary_dict) or Exception
    progress = pyqtSignal(str)

    def __init__(self, folder: Path, parent=None) -> None:
        super().__init__(parent)
        self._folder = folder

    def run(self) -> None:
        try:
            self._do_load()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.finished.emit(exc)

    def _do_load(self) -> None:
        import numpy as np
        from spectral_select.types import ExcitationData, SpectraData

        folder = self._folder
        meta_path = _find_metadata_xlsx(folder)

        # ----------------------------------------------------------
        # 1.  Use HyperspectralDataLoader to load .im3 files
        # ----------------------------------------------------------
        self.progress.emit("Initializing loader (ImageJ may take a moment)...")

        from scripts.data_processing.hyperspectral_loader import (
            HyperspectralDataLoader,
        )

        try:
            # Check if imagej is available
            imagej_ok = False
            try:
                import imagej  # noqa: F401
                imagej_ok = True
            except ImportError:
                pass

            loader = HyperspectralDataLoader(
                data_path=str(folder),
                metadata_path=str(meta_path) if meta_path else None,
                cutoff_offset=30,
                use_fiji=imagej_ok,
                verbose=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create data loader: {e}") from e

        self.progress.emit("Loading .im3 files (this may take a while)...")

        # Call load_data.  We pass apply_cutoff=False so we get raw cubes
        # but note: this means loader.data will be EMPTY (known bug in the
        # underlying loader).  We read from loader.raw_data instead.
        try:
            loader.load_data(apply_cutoff=False)
        except NotImplementedError:
            # Fallback: try with tifffile directly
            self.progress.emit("ImageJ loading failed, trying tifffile fallback...")
            self._load_tifffile_fallback(folder, meta_path)
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load .im3 files: {e}") from e

        # ----------------------------------------------------------
        # 2.  Build SpectraData from raw_data
        # ----------------------------------------------------------
        self.progress.emit("Building spectral data...")

        if not loader.raw_data:
            raise RuntimeError(
                f"No .im3 files were successfully loaded from {folder}.\n"
                f"The loader's raw_data is empty.  Check that the directory "
                f"contains valid .im3 hyperspectral image files."
            )

        # Parse metadata files independently (more robust than relying
        # on the internal loader's metadata handling)
        exposure_times = _parse_exposure_times(folder)
        laser_powers = _parse_laser_powers(folder)

        excitations = {}
        for ex_str, raw in loader.raw_data.items():
            ex_nm = float(raw["ex"])
            cube = raw["data"]
            wavelengths = raw["em_arr"]

            # Ensure cube is 3-D (height, width, bands)
            if cube.ndim == 2:
                cube = cube[:, :, np.newaxis]

            excitations[ex_nm] = ExcitationData(
                excitation_nm=ex_nm,
                cube=cube.astype(np.float64),
                emission_wavelengths=[float(w) for w in wavelengths],
                exposure_time=exposure_times.get(ex_nm) or raw.get("expos_val"),
                laser_power=laser_powers.get(ex_nm),
            )

        spectra = SpectraData(
            excitations=excitations,
            sample_name=folder.name,
        )

        # ----------------------------------------------------------
        # 3.  Build summary for the UI
        # ----------------------------------------------------------
        summary = self._build_summary(spectra, exposure_times, laser_powers, folder)
        self.finished.emit((spectra, summary))

    def _load_tifffile_fallback(self, folder: Path, meta_path) -> None:
        """Fallback loader using tifffile for .im3 files (TIFF-based)."""
        import numpy as np
        import tifffile
        from spectral_select.types import ExcitationData, SpectraData

        im3_files = sorted(folder.glob("*.im3"))
        if not im3_files:
            raise RuntimeError(f"No .im3 files found in {folder}")

        exposure_times = _parse_exposure_times(folder)
        laser_powers = _parse_laser_powers(folder)
        excitations = {}

        for im3_path in im3_files:
            name = im3_path.stem
            try:
                ex_nm = float(name)
            except ValueError:
                continue  # skip non-numeric filenames

            self.progress.emit(f"Loading {im3_path.name} via tifffile...")
            try:
                cube = tifffile.imread(str(im3_path)).astype(np.float64)
            except Exception as e:
                print(f"Warning: failed to load {im3_path.name}: {e}")
                continue

            # tifffile may return (bands, H, W) — transpose to (H, W, bands)
            if cube.ndim == 3 and cube.shape[0] < cube.shape[1]:
                cube = np.transpose(cube, (1, 2, 0))
            elif cube.ndim == 2:
                cube = cube[:, :, np.newaxis]

            n_bands = cube.shape[2]
            em_start = 420.0 if ex_nm <= 400.0 else ex_nm + 20.0
            wavelengths = [em_start + i * 10 for i in range(n_bands)]

            excitations[ex_nm] = ExcitationData(
                excitation_nm=ex_nm,
                cube=cube,
                emission_wavelengths=wavelengths,
                exposure_time=exposure_times.get(ex_nm),
                laser_power=laser_powers.get(ex_nm),
            )

        if not excitations:
            raise RuntimeError(f"No .im3 files could be loaded from {folder}")

        spectra = SpectraData(
            excitations=excitations,
            sample_name=folder.name,
        )
        summary = self._build_summary(spectra, exposure_times, laser_powers, folder)
        self.finished.emit((spectra, summary))

    @staticmethod
    def _build_summary(spectra, exposure_times, laser_powers, folder) -> dict:
        summary = {
            "exposure_file": None,
            "power_file": None,
            "exposure_times": exposure_times,
            "laser_powers": laser_powers,
            "warnings": [],
            "patched_exposure": 0,
            "patched_power": 0,
        }

        meta_path = _find_metadata_xlsx(folder)
        power_path = _find_power_xlsx(folder)
        if meta_path:
            summary["exposure_file"] = str(meta_path)
        if power_path:
            summary["power_file"] = str(power_path)

        missing_exp = []
        missing_pow = []
        for ex_nm, ex_data in spectra.excitations.items():
            if ex_data.exposure_time is not None:
                summary["patched_exposure"] += 1
            else:
                missing_exp.append(ex_nm)
            if ex_data.laser_power is not None:
                summary["patched_power"] += 1
            else:
                missing_pow.append(ex_nm)

        if missing_exp:
            summary["warnings"].append(
                f"No exposure time for: "
                f"{', '.join(f'{x:.0f}' for x in sorted(missing_exp))} nm"
            )
        if missing_pow:
            summary["warnings"].append(
                f"No laser power for: "
                f"{', '.join(f'{x:.0f}' for x in sorted(missing_pow))} nm"
            )
        return summary


# ------------------------------------------------------------------
# Step widget
# ------------------------------------------------------------------

class Step1Load(AbstractStepWidget):
    """Pick a folder of .im3 files and browse the loaded hyperspectral data."""

    @property
    def step_index(self) -> int:
        return 1

    @property
    def title(self) -> str:
        return "Load Data"

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._thread: _LoaderThread | None = None
        self._progress_dlg: QProgressDialog | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Folder selection ---
        grp = QGroupBox("Data Folder")
        g = QHBoxLayout(grp)
        self._lbl_folder = QLabel("No folder selected")
        self._btn_browse = QPushButton("Browse...")
        self._btn_browse.clicked.connect(self._browse)
        g.addWidget(self._lbl_folder, 1)
        g.addWidget(self._btn_browse)
        layout.addWidget(grp)

        # --- Info ---
        self._lbl_info = QLabel("")
        self._lbl_info.setWordWrap(True)
        layout.addWidget(self._lbl_info)

        # --- Band navigator + canvas ---
        self._canvas = ImageCanvas(parent=self)
        self._navigator = BandNavigator(parent=self)
        self._navigator.band_changed.connect(self._on_band_changed)

        layout.addWidget(self._navigator)
        layout.addWidget(self._canvas, 1)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select .im3 data folder")
        if not folder:
            return
        path = Path(folder)
        self._lbl_folder.setText(str(path))
        self._start_loading(path)

    def _start_loading(self, folder: Path) -> None:
        self._btn_browse.setEnabled(False)
        self._lbl_info.setText("Loading... please wait.")

        self._progress_dlg = QProgressDialog(
            "Initializing...", None, 0, 0, self  # No cancel button
        )
        self._progress_dlg.setWindowTitle("Loading Data")
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.setCancelButton(None)
        self._progress_dlg.show()

        self._thread = _LoaderThread(folder, parent=self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_loaded)
        self._thread.start()

    def _on_progress(self, msg: str) -> None:
        if self._progress_dlg:
            self._progress_dlg.setLabelText(msg)

    def _on_loaded(self, result: object) -> None:
        self._btn_browse.setEnabled(True)
        if self._progress_dlg:
            self._progress_dlg.close()
            self._progress_dlg = None

        if isinstance(result, Exception):
            self._lbl_info.setText(f"Load failed: {result}")
            QMessageBox.critical(self, "Load Error", str(result))
            return

        spectra, summary = result
        self.state.raw_spectra = spectra
        self.state.data_folder = Path(self._lbl_folder.text())
        self.state.invalidate_from(STEP_LOAD)

        # Store parsed metadata in state for Step 2 to display
        self.state.exposure_times = summary.get("exposure_times", {})
        self.state.laser_powers = summary.get("laser_powers", {})

        # Build info text
        ex_list = spectra.excitation_wavelengths
        h, w = spectra.spatial_shape
        n_ex = spectra.n_excitations
        lines = [f"Loaded {n_ex} excitations | Spatial: {h} x {w}"]

        bands_info = ", ".join(
            f"{ex:.0f} nm ({spectra.get_excitation(ex).n_bands} bands)"
            for ex in ex_list
        )
        lines.append(bands_info)

        # Metadata summary
        if summary.get("exposure_file"):
            lines.append(
                f"\nExposure file: {Path(summary['exposure_file']).name} "
                f"({summary['patched_exposure']}/{n_ex} matched)"
            )
        else:
            lines.append("\nNo metadata.xlsx found")

        if summary.get("power_file"):
            lines.append(
                f"Power file: {Path(summary['power_file']).name} "
                f"({summary['patched_power']}/{n_ex} matched)"
            )
        else:
            lines.append("No average_power.xlsx found")

        for w_msg in summary.get("warnings", []):
            lines.append(f"Warning: {w_msg}")

        self._lbl_info.setText("\n".join(lines))
        self._navigator.set_spectra(spectra)

    def _on_band_changed(self, excitation: float, band_idx: int) -> None:
        from spectral_select.widgets import create_display_image

        spectra = self.state.raw_spectra
        if spectra is None:
            return
        cube = spectra.get_excitation(excitation).cube
        img = create_display_image(cube, band_index=band_idx)
        self._canvas.show_image(img)
        self._canvas.set_title(f"Ex {excitation:.0f} nm – Band {band_idx}")

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        if self.state.raw_spectra is not None:
            self._navigator.set_spectra(self.state.raw_spectra)
