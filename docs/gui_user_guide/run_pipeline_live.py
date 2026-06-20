"""Drive the WHOLE pipeline on real data (Collagen_Acetic_Acid .im3), headless.

Loads raw .im3 -> normalize -> spatial crop -> spectral crop -> classes/ROI ->
export -> train autoencoder -> select bands, calling each step's real handler and
screenshotting every populated page. This is both an end-to-end test and the source
of the "with real data" screenshots used in the user guide.

Run:  QT_QPA_PLATFORM=offscreen python docs/gui_user_guide/run_pipeline_live.py
Output: docs/gui_user_guide/screenshots_live/*.png  (+ exported files in a temp dir)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data" / "Raw" / "Collagen_Acetic_Acid"
OUT = Path(__file__).parent / "screenshots_live"
OUT.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = Path(tempfile.mkdtemp(prefix="mehsi_export_"))

SLUGS = {1: "load", 2: "metadata", 3: "normalize", 4: "spatial_crop", 5: "spectral_crop",
         6: "draw_classes", 7: "roi_regions", 8: "export", 9: "train", 10: "select"}

# --- Neutralise modal dialogs so handlers run headless -------------------------
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(EXPORT_DIR))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))


def main() -> None:
    app = QApplication.instance() or QApplication([])
    from mehsi_preprocessor.app import PreprocessorWindow
    win = PreprocessorWindow()
    win.resize(1300, 850)
    win.show()
    app.processEvents()
    state = win._state
    steps = {s.step_index: s for s in win._steps}

    def goto(i: int) -> None:
        """Make page i the current (sized) widget BEFORE running its handlers,
        so matplotlib canvases/selectors have a positive figure size offscreen."""
        win._stack.setCurrentIndex(i - 1)
        win._sidebar.blockSignals(True)
        win._sidebar.setCurrentRow(i - 1)
        win._sidebar.blockSignals(False)
        app.processEvents()

    def shot(i: int) -> None:
        app.processEvents()
        p = OUT / f"step_{i:02d}_{SLUGS[i]}.png"
        win.grab().save(str(p))
        print(f"  [shot] step {i} -> {p.name}")

    # --- Step 1: load real .im3 (synchronous, mirrors _LoaderThread + _on_loaded) ---
    print("STEP 1: loading raw .im3 from", RAW)
    from mehsi_preprocessor.steps.step1_load import _LoaderThread
    goto(1)
    s1 = steps[1]
    s1._lbl_folder.setText(str(RAW))
    captured = {}
    lt = _LoaderThread(RAW)
    lt.finished.connect(lambda r: captured.setdefault("r", r))
    lt._do_load()
    result = captured["r"]
    assert not isinstance(result, Exception), f"load failed: {result}"
    s1._on_loaded(result)
    spec = state.raw_spectra
    ex0 = spec.excitation_wavelengths[0]
    nb = spec.get_excitation(ex0).n_bands
    s1._on_band_changed(ex0, nb // 2)
    print(f"  loaded: {spec.n_excitations} ex, {spec.spatial_shape}, {nb} bands @ {ex0}nm")
    shot(1)

    # --- Step 2: metadata ---
    print("STEP 2: metadata")
    goto(2)
    steps[2].on_enter()
    shot(2)

    # --- Step 3: normalize ---
    print("STEP 3: normalize")
    goto(3)
    s3 = steps[3]
    s3.on_enter()
    s3._apply()
    s3._refresh_images(ex0, nb // 2)
    print("  normalized:", state.normalized_spectra.spatial_shape)
    shot(3)

    # --- Step 4: spatial crop to a small region (keeps training fast) ---
    print("STEP 4: spatial crop")
    goto(4)
    s4 = steps[4]
    s4.on_enter()
    r0, r1, c0, c1 = 90, 170, 140, 220   # 80 x 80
    s4._on_rect(r0, r1, c0, c1)
    s4._apply_crop()
    print("  cropped:", state.cropped_spectra.spatial_shape)
    shot(4)

    # --- Step 5: spectral crop (Rayleigh + defaults) ---
    print("STEP 5: spectral crop")
    goto(5)
    s5 = steps[5]
    s5.on_enter()
    s5._preview()
    s5._apply()
    filt = state.filtered_spectra
    total_bands = sum(filt.get_excitation(e).n_bands for e in filt.excitation_wavelengths)
    print("  filtered:", filt.spatial_shape, "total bands kept:", total_bands)
    shot(5)

    # --- Step 6 & 7: classes + ROIs (real structures + mask_ops) ---
    print("STEP 6/7: classes + ROIs")
    from mehsi_preprocessor.state import ClassDef, ROIRegion
    from mehsi_preprocessor.processing.mask_ops import roi_regions_to_mask
    h, w = state.current_spectra.spatial_shape
    state.class_definitions = [
        ClassDef(1, "Sample", (220, 40, 40)),
        ClassDef(2, "Lesion", (40, 90, 220)),
        ClassDef(3, "Background", (40, 170, 70)),
    ]
    state.roi_regions = [
        ROIRegion(1, "Sample", (8, 34, 8, 36)),
        ROIRegion(2, "Lesion", (44, 72, 44, 72)),
        ROIRegion(3, "Background", (8, 30, 46, 74)),
    ]
    state.class_mask = roi_regions_to_mask(state.roi_regions, (h, w))
    goto(6)
    steps[6].on_enter()
    shot(6)
    goto(7)
    steps[7].on_enter()
    shot(7)

    # --- Step 8: export to disk (real files) ---
    print("STEP 8: export ->", EXPORT_DIR)
    goto(8)
    s8 = steps[8]
    s8.on_enter()
    s8._browse_dir()      # getExistingDirectory monkeypatched -> EXPORT_DIR
    s8._export()
    print("  exported files:", sorted(p.name for p in EXPORT_DIR.glob("*")))
    shot(8)

    # --- Step 9: train autoencoder (few epochs, on the cropped data) ---
    print("STEP 9: train autoencoder (this runs a real CAE training)")
    from spectral_select import Analyzer
    goto(9)
    s9 = steps[9]
    s9._epochs.setValue(4)
    cfg = s9._build_config()
    analyzer = Analyzer(cfg)
    analyzer.prepare(
        state.current_spectra,
        progress_callback=lambda e, t, loss: print(f"    epoch {e}/{t} loss={loss:.4f}"),
    )
    s9._done(analyzer)
    print("  model prepared:", analyzer.is_prepared)
    shot(9)

    # --- Step 10: select bands (real perturbation selection) ---
    print("STEP 10: select bands")
    from mehsi_preprocessor.workers import run_selection_job
    goto(10)
    s10 = steps[10]
    s10.on_enter()
    s10._n_bands.setValue(12)
    cfg10 = s10._config()
    res = run_selection_job(state.analyzer, cfg10)
    s10._done(res)
    print("  selected bands:", len(res.selected_bands))
    for b in res.selected_bands[:5]:
        print(f"    #{b.rank} ex{b.excitation_nm:.0f} em{b.emission_nm:.1f} infl={b.influence_score:.3e}")
    shot(10)

    # overview = step 1 with data
    win._stack.setCurrentIndex(0)
    win._sidebar.blockSignals(True); win._sidebar.setCurrentRow(0); win._sidebar.blockSignals(False)
    app.processEvents()
    win.grab().save(str(OUT / "overview.png"))
    print("DONE. Live screenshots in", OUT)
    print("Exported pipeline outputs in", EXPORT_DIR)


if __name__ == "__main__":
    main()
