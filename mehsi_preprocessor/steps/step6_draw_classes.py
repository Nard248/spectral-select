"""Step 6: Draw class masks using a brush / eraser tool with auto-fill support."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
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
        self._edge_mask: np.ndarray | None = None
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
        self._radio_eraser.toggled.connect(self._tool_changed)
        self._radio_fill = QRadioButton("Auto-Fill (click to fill)")
        self._radio_fill.toggled.connect(self._tool_changed)
        tl.addWidget(self._radio_brush)
        tl.addWidget(self._radio_eraser)
        tl.addWidget(self._radio_fill)

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

        # Edge detection controls
        edge_grp = QGroupBox("Edge Detection (for Auto-Fill)")
        el = QVBoxLayout(edge_grp)

        # Algorithm selection
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self._combo_algorithm = QComboBox()
        self._combo_algorithm.addItem("Canny", "canny")
        self._combo_algorithm.addItem("Sobel", "sobel")
        algo_row.addWidget(self._combo_algorithm)
        el.addLayout(algo_row)

        # Threshold slider
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self._slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self._slider_threshold.setRange(1, 100)
        self._slider_threshold.setValue(30)
        self._slider_threshold.valueChanged.connect(self._on_threshold_changed)
        self._lbl_threshold = QLabel("30")
        thresh_row.addWidget(self._slider_threshold)
        thresh_row.addWidget(self._lbl_threshold)
        el.addLayout(thresh_row)

        # Preview toggle
        self._chk_preview = QCheckBox("Preview edges")
        self._chk_preview.toggled.connect(self._on_preview_toggled)
        el.addWidget(self._chk_preview)

        # Compute button
        self._btn_compute_edges = QPushButton("Compute Edges")
        self._btn_compute_edges.clicked.connect(self._compute_edges)
        el.addWidget(self._btn_compute_edges)

        # Status label
        self._lbl_edge_status = QLabel("Edges not computed")
        self._lbl_edge_status.setStyleSheet("color: gray; font-style: italic;")
        el.addWidget(self._lbl_edge_status)

        left.addWidget(edge_grp)
        left.addStretch()

        layout.addLayout(left)

        # --- Right panel: brush canvas ---
        self._canvas = BrushCanvas(self)
        self._canvas.mask_updated.connect(self._on_mask_updated)
        self._canvas.click_for_fill.connect(self._on_fill_click)
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
        if self._radio_brush.isChecked():
            self._canvas.set_erase_mode(False)
            self._canvas.set_fill_mode(False)
        elif self._radio_eraser.isChecked():
            self._canvas.set_erase_mode(True)
            self._canvas.set_fill_mode(False)
        elif self._radio_fill.isChecked():
            self._canvas.set_erase_mode(False)
            self._canvas.set_fill_mode(True)

    def _radius_changed(self, val: int) -> None:
        self._canvas.set_brush_radius(val)

    # ------------------------------------------------------------------
    # Edge detection
    # ------------------------------------------------------------------

    def _on_threshold_changed(self, val: int) -> None:
        self._lbl_threshold.setText(str(val))

    def _on_preview_toggled(self, checked: bool) -> None:
        if checked and self._edge_mask is not None:
            self._canvas.show_edge_preview(self._edge_mask)
        else:
            self._canvas.show_edge_preview(None)

    def _compute_edges(self) -> None:
        """Compute edge detection on the base image."""
        img = self._canvas.base_image
        if img is None:
            QMessageBox.warning(self, "No Image", "Load data first.")
            return

        algorithm = self._combo_algorithm.currentData()
        threshold = self._slider_threshold.value()

        try:
            self._edge_mask = self._detect_edges(img, algorithm, threshold)
            self._canvas.set_edge_mask(self._edge_mask)

            n_edge_pixels = np.sum(self._edge_mask > 0)
            self._lbl_edge_status.setText(f"Edges computed ({n_edge_pixels:,} edge pixels)")
            self._lbl_edge_status.setStyleSheet("color: green;")

            if self._chk_preview.isChecked():
                self._canvas.show_edge_preview(self._edge_mask)

        except Exception as e:
            QMessageBox.critical(self, "Edge Detection Error", str(e))
            self._lbl_edge_status.setText(f"Error: {e}")
            self._lbl_edge_status.setStyleSheet("color: red;")

    def _detect_edges(self, img: np.ndarray, algorithm: str, threshold: int) -> np.ndarray:
        """Run edge detection algorithm on the image.

        Returns a binary edge mask.
        """
        from scipy import ndimage

        # Ensure grayscale
        if img.ndim == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img.copy()

        # Normalize to 0-255 range
        gray = gray.astype(np.float64)
        if gray.max() > gray.min():
            gray = (gray - gray.min()) / (gray.max() - gray.min()) * 255
        gray = gray.astype(np.uint8)

        if algorithm == "canny":
            return self._canny_edge(gray, threshold)
        elif algorithm == "sobel":
            return self._sobel_edge(gray, threshold)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _canny_edge(self, gray: np.ndarray, threshold: int) -> np.ndarray:
        """Canny edge detection using scipy."""
        from scipy import ndimage

        # Gaussian blur first
        blurred = ndimage.gaussian_filter(gray.astype(np.float64), sigma=1.5)

        # Sobel gradients
        gx = ndimage.sobel(blurred, axis=1)
        gy = ndimage.sobel(blurred, axis=0)

        # Gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize magnitude
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max() * 255

        # Threshold (scaled: 1-100 -> 5-200)
        thresh_val = 5 + (threshold / 100) * 195
        edges = magnitude > thresh_val

        # Optional: thin edges using morphological operations
        from scipy.ndimage import binary_erosion
        struct = np.ones((3, 3))
        # Light erosion to thin thick edges
        if np.sum(edges) > 1000:
            edges = binary_erosion(edges, structure=struct, iterations=1)
            # Restore with original to keep connectivity
            edges = magnitude > thresh_val * 0.8

        return edges.astype(np.uint8)

    def _sobel_edge(self, gray: np.ndarray, threshold: int) -> np.ndarray:
        """Sobel edge detection."""
        from scipy import ndimage

        # Sobel gradients
        gx = ndimage.sobel(gray.astype(np.float64), axis=1)
        gy = ndimage.sobel(gray.astype(np.float64), axis=0)

        # Gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max() * 255

        # Threshold (scaled: 1-100 -> 5-150)
        thresh_val = 5 + (threshold / 100) * 145

        return (magnitude > thresh_val).astype(np.uint8)

    # ------------------------------------------------------------------
    # Auto-fill
    # ------------------------------------------------------------------

    def _on_fill_click(self, row: int, col: int) -> None:
        """Handle click in fill mode."""
        if self._edge_mask is None:
            QMessageBox.warning(
                self, "No Edges",
                "Compute edges first using the 'Compute Edges' button."
            )
            return

        if not self.state.class_definitions:
            QMessageBox.warning(
                self, "No Class",
                "Add a class first before filling."
            )
            return

        # Perform the fill
        filled = self._canvas.fill_region(row, col)
        if not filled:
            # Might have clicked on an edge
            QMessageBox.information(
                self, "Cannot Fill",
                "Clicked on an edge or outside bounds. Try clicking inside a region."
            )

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

        # Reset edge state
        self._edge_mask = None
        self._lbl_edge_status.setText("Edges not computed")
        self._lbl_edge_status.setStyleSheet("color: gray; font-style: italic;")
