"""Step 6: Draw class masks using a brush / eraser tool."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mehsi_preprocessor.state import (
    DEFAULT_CLASS_COLORS,
    STEP_DRAW_CLASSES,
    ClassDef,
    PipelineState,
)
from mehsi_preprocessor.steps.base import AbstractStepWidget
from mehsi_preprocessor.widgets.brush_canvas import BrushCanvas


class Step6DrawClasses(AbstractStepWidget):

    @property
    def step_index(self) -> int:
        return 6

    @property
    def title(self) -> str:
        return "Draw Classes"

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(state, parent)
        self._next_id = 1
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        # --- Left panel: class list + tools ---
        left = QVBoxLayout()

        # Class list
        cls_grp = QGroupBox("Classes")
        cg = QVBoxLayout(cls_grp)
        self._class_list = QListWidget()
        self._class_list.currentRowChanged.connect(self._on_class_selected)
        cg.addWidget(self._class_list)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add")
        self._btn_add.clicked.connect(self._add_class)
        self._btn_rename = QPushButton("Rename")
        self._btn_rename.clicked.connect(self._rename_class)
        self._btn_delete = QPushButton("Delete")
        self._btn_delete.clicked.connect(self._delete_class)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_rename)
        btn_row.addWidget(self._btn_delete)
        cg.addLayout(btn_row)
        left.addWidget(cls_grp)

        # Tool selection
        tool_grp = QGroupBox("Tool")
        tl = QVBoxLayout(tool_grp)
        self._radio_brush = QRadioButton("Brush")
        self._radio_brush.setChecked(True)
        self._radio_brush.toggled.connect(self._tool_changed)
        self._radio_eraser = QRadioButton("Eraser")
        tl.addWidget(self._radio_brush)
        tl.addWidget(self._radio_eraser)

        # Brush radius
        rad_row = QHBoxLayout()
        rad_row.addWidget(QLabel("Radius:"))
        self._spin_radius = QSpinBox()
        self._spin_radius.setRange(1, 50)
        self._spin_radius.setValue(5)
        self._spin_radius.valueChanged.connect(self._radius_changed)
        rad_row.addWidget(self._spin_radius)
        tl.addLayout(rad_row)

        left.addWidget(tool_grp)
        left.addStretch()

        layout.addLayout(left)

        # --- Right panel: brush canvas ---
        self._canvas = BrushCanvas(self)
        self._canvas.mask_updated.connect(self._on_mask_updated)
        layout.addWidget(self._canvas, 1)

    # ------------------------------------------------------------------
    # Class management
    # ------------------------------------------------------------------

    def _add_class(self) -> None:
        idx = len(self.state.class_definitions)
        color = DEFAULT_CLASS_COLORS[idx % len(DEFAULT_CLASS_COLORS)]
        cd = ClassDef(id=self._next_id, name=f"Class {self._next_id}", color=color)
        self._next_id += 1
        self.state.class_definitions.append(cd)
        self._refresh_class_list()
        self._canvas.set_class_defs(self.state.class_definitions)

    def _rename_class(self) -> None:
        row = self._class_list.currentRow()
        if row < 0 or row >= len(self.state.class_definitions):
            return
        cd = self.state.class_definitions[row]
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Rename Class", "New name:", text=cd.name)
        if ok and name:
            cd.name = name
            self._refresh_class_list()

    def _delete_class(self) -> None:
        row = self._class_list.currentRow()
        if row < 0 or row >= len(self.state.class_definitions):
            return
        cd = self.state.class_definitions.pop(row)
        # Clear mask pixels for this class
        if self.state.class_mask is not None:
            self.state.class_mask[self.state.class_mask == cd.id] = 0
        self._refresh_class_list()
        self._canvas.set_class_defs(self.state.class_definitions)
        self._canvas._refresh_overlay()

    def _refresh_class_list(self) -> None:
        self._class_list.clear()
        for cd in self.state.class_definitions:
            item = QListWidgetItem(f"[{cd.id}] {cd.name}")
            item.setForeground(QColor(*cd.color))
            self._class_list.addItem(item)

    def _on_class_selected(self, row: int) -> None:
        if 0 <= row < len(self.state.class_definitions):
            self._canvas.set_current_class(self.state.class_definitions[row].id)

    # ------------------------------------------------------------------
    # Tool state
    # ------------------------------------------------------------------

    def _tool_changed(self, checked: bool) -> None:
        self._canvas.set_erase_mode(not self._radio_brush.isChecked())

    def _radius_changed(self, val: int) -> None:
        self._canvas.set_brush_radius(val)

    # ------------------------------------------------------------------
    # Mask sync
    # ------------------------------------------------------------------

    def _on_mask_updated(self) -> None:
        self.state.class_mask = self._canvas.mask
        self.state.invalidate_from(STEP_DRAW_CLASSES)

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        from spectral_select.widgets import create_display_image

        spectra = self.state.current_spectra
        if spectra is None:
            return

        ex_nm = spectra.excitation_wavelengths[0]
        cube = spectra.get_excitation(ex_nm).cube
        img = create_display_image(cube)
        self._canvas.set_base_image(img)

        h, w = spectra.spatial_shape
        if self.state.class_mask is None or self.state.class_mask.shape != (h, w):
            self._canvas.init_mask((h, w))
            self.state.class_mask = self._canvas.mask
        else:
            self._canvas.set_mask(self.state.class_mask)

        self._canvas.set_class_defs(self.state.class_definitions)
        self._refresh_class_list()
