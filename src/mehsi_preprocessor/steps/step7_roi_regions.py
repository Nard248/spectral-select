"""Step 7: Draw rectangular ROI regions and assign to classes."""

from __future__ import annotations

from typing import Optional

import matplotlib.patches as mpatches
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import (
    DEFAULT_CLASS_COLORS,
    STEP_ROI_REGIONS,
    PipelineState,
    ROIRegion,
)
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.widgets.rect_selector import RectSelector


class Step7ROIRegions(AbstractStepWidget):

    @property
    def step_index(self) -> int:
        return 7

    @property
    def title(self) -> str:
        return "ROI Regions"

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        # --- Left panel: controls + table ---
        left = QVBoxLayout()

        # Class selector
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Assign to class:"))
        self._class_combo = QComboBox()
        cls_row.addWidget(self._class_combo, 1)
        left.addLayout(cls_row)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add Selected Rect")
        self._btn_add.clicked.connect(self._add_roi)
        self._btn_add.setEnabled(False)
        self._btn_dup = QPushButton("Duplicate Last")
        self._btn_dup.clicked.connect(self._duplicate_last)
        self._btn_del = QPushButton("Delete Selected")
        self._btn_del.clicked.connect(self._delete_selected)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_dup)
        btn_row.addWidget(self._btn_del)
        left.addLayout(btn_row)

        # ROI table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["ID", "Class", "Row min", "Row max", "Col min→max"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        left.addWidget(self._table, 1)

        layout.addLayout(left)

        # --- Right panel: image with rectangles ---
        self._canvas = RectSelector(self)
        self._canvas.rect_selected.connect(self._on_rect_selected)
        layout.addWidget(self._canvas, 1)

    # ------------------------------------------------------------------
    # ROI management
    # ------------------------------------------------------------------

    def _on_rect_selected(self, r0: int, r1: int, c0: int, c1: int) -> None:
        self._btn_add.setEnabled(True)

    def _add_roi(self) -> None:
        rect = self._canvas.last_rect
        if rect is None:
            return

        idx = self._class_combo.currentIndex()
        if idx < 0 or not self.state.class_definitions:
            QMessageBox.warning(self, "No Class", "Define classes in Step 6 first.")
            return

        cd = self.state.class_definitions[idx]
        region = ROIRegion(
            class_id=cd.id,
            class_name=cd.name,
            rect=rect,
        )
        self.state.roi_regions.append(region)
        self._refresh_table()
        self._draw_all_rects()

    def _duplicate_last(self) -> None:
        if not self.state.roi_regions:
            return
        last = self.state.roi_regions[-1]
        r0, r1, c0, c1 = last.rect
        # Offset by 10px
        new_rect = (r0 + 10, r1 + 10, c0 + 10, c1 + 10)

        # Increment class if possible
        defs = self.state.class_definitions
        curr_idx = next(
            (i for i, d in enumerate(defs) if d.id == last.class_id), 0
        )
        next_idx = (curr_idx + 1) % len(defs) if defs else 0
        cd = defs[next_idx] if defs else None
        if cd is None:
            return

        region = ROIRegion(class_id=cd.id, class_name=cd.name, rect=new_rect)
        self.state.roi_regions.append(region)
        self._refresh_table()
        self._draw_all_rects()

    def _delete_selected(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self.state.roi_regions):
            self.state.roi_regions.pop(row)
            self._refresh_table()
            self._draw_all_rects()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        self._table.setRowCount(len(self.state.roi_regions))
        for i, reg in enumerate(self.state.roi_regions):
            r0, r1, c0, c1 = reg.rect
            self._table.setItem(i, 0, QTableWidgetItem(str(i)))
            self._table.setItem(i, 1, QTableWidgetItem(reg.class_name))
            self._table.setItem(i, 2, QTableWidgetItem(str(r0)))
            self._table.setItem(i, 3, QTableWidgetItem(str(r1)))
            self._table.setItem(i, 4, QTableWidgetItem(f"{c0} → {c1}"))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_all_rects(self) -> None:
        # Remove old patches
        ax = self._canvas.ax
        for p in list(ax.patches):
            p.remove()

        # Colour map from class defs
        cmap = {}
        for cd in self.state.class_definitions:
            cmap[cd.id] = tuple(c / 255.0 for c in cd.color)

        for reg in self.state.roi_regions:
            r0, r1, c0, c1 = reg.rect
            color = cmap.get(reg.class_id, (1, 0, 0))
            rect = mpatches.Rectangle(
                (c0, r0), c1 - c0, r1 - r0,
                linewidth=2, edgecolor=color, facecolor=(*color, 0.15),
            )
            ax.add_patch(rect)

        self._canvas.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        from spectral_select.widgets import create_display_image

        # Refresh class combo
        self._class_combo.clear()
        for cd in self.state.class_definitions:
            self._class_combo.addItem(f"[{cd.id}] {cd.name}")

        spectra = self.state.current_spectra
        if spectra is None:
            return
        ex_nm = spectra.excitation_wavelengths[0]
        cube = spectra.get_excitation(ex_nm).cube
        self._canvas.show_image(create_display_image(cube))
        self._canvas.set_title("Draw ROI rectangles")
        self._canvas.enable_selector()
        self._refresh_table()
        self._draw_all_rects()
