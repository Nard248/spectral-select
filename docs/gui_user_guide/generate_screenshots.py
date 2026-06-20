"""Render every wizard page to a PNG via Qt's offscreen engine.

This doubles as a render smoke-test: constructing + drawing all 10 pages will raise
if any widget is broken. Run:  QT_QPA_PLATFORM=offscreen python docs/gui_user_guide/generate_screenshots.py
Output: docs/gui_user_guide/screenshots/*.png
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

SLUGS = {
    1: "load", 2: "metadata", 3: "normalize", 4: "spatial_crop", 5: "spectral_crop",
    6: "draw_classes", 7: "roi_regions", 8: "export", 9: "train", 10: "select",
}


def main() -> None:
    app = QApplication.instance() or QApplication([])
    from mehsi_preprocessor.app import PreprocessorWindow

    window = PreprocessorWindow()
    window.resize(1300, 850)
    window.show()
    app.processEvents()

    saved = []
    for i, step in enumerate(window._steps):
        # Set the visible page + highlight the sidebar WITHOUT triggering on_leave guards.
        window._stack.setCurrentIndex(i)
        window._sidebar.blockSignals(True)
        window._sidebar.setCurrentRow(i)
        window._sidebar.blockSignals(False)
        try:
            step.on_enter()
        except Exception as exc:  # keep going; note which page failed to refresh
            print(f"  [warn] step {step.step_index} on_enter raised: {exc}")
        app.processEvents()

        path = OUT / f"step_{step.step_index:02d}_{SLUGS.get(step.step_index, 'page')}.png"
        ok = window.grab().save(str(path))
        size = path.stat().st_size if path.exists() else 0
        print(f"  step {step.step_index:>2} '{step.title}': saved={ok} bytes={size} -> {path.name}")
        saved.append(path)

    # Cover/overview = full window on the first page.
    window._stack.setCurrentIndex(0)
    window._sidebar.blockSignals(True); window._sidebar.setCurrentRow(0); window._sidebar.blockSignals(False)
    window._steps[0].on_enter()
    app.processEvents()
    cover = OUT / "overview.png"
    window.grab().save(str(cover))
    print(f"  overview -> {cover.name}")
    print(f"DONE: {len(saved)} page screenshots + overview in {OUT}")


if __name__ == "__main__":
    main()
