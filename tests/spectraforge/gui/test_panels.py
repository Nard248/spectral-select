import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from spectraforge.material import Material
from spectraforge.gui.state import ForgeState

_app = QApplication.instance() or QApplication([])


def test_library_panel_builds_and_defines():
    from spectraforge.gui.panels.library_panel import LibraryPanel
    st = ForgeState(height=8, width=8)
    p = LibraryPanel(st)
    n0 = len(st.library)
    p.define_fluorophore("MyDye", 500, 40, 560, 50, 0.7, 0.9)
    assert len(st.library) == n0 + 1 and "MyDye" in st.library


def test_material_panel_composes():
    from spectraforge.gui.panels.material_panel import MaterialPanel
    st = ForgeState(height=8, width=8)
    p = MaterialPanel(st)
    p.compose_material("tissue", {"collagen": 1.0, "NADH": 0.3})
    assert "tissue" in st.materials
    assert st.materials["tissue"].recipe["collagen"] == 1.0


def test_layers_panel_add_toggle():
    from spectraforge.gui.panels.layers_panel import LayersPanel
    st = ForgeState(height=8, width=8)
    st.materials["m"] = Material("m", {"collagen": 1.0})
    p = LayersPanel(st)
    p.add_layer_with_material("m")
    assert len(st.layers) == 1 and st.layers[0].material.name == "m"
    p.set_visible(0, False)
    assert st.layers[0].visible is False


def test_canvas_panel_paint_rect_writes_active_layer():
    from spectraforge.gui.panels.canvas_panel import CanvasPanel
    st = ForgeState(height=10, width=10)
    st.materials["m"] = Material("m", {"collagen": 1.0})
    st.add_layer("L", st.materials["m"])
    p = CanvasPanel(st)
    p.paint_rect(2, 6, 2, 6, amount=1.0)
    assert st.layers[0].amount_map[3, 3] == 1.0
    assert st.layers[0].amount_map[0, 0] == 0.0


def test_acquire_render_panel_render_and_export(tmp_path):
    from spectraforge.gui.panels.acquire_render_panel import AcquireRenderPanel
    st = ForgeState(height=10, width=10)
    st.materials["m"] = Material("m", {"EGFP": 1.0})
    layer = st.add_layer("L", st.materials["m"])
    layer.amount_map[:] = 1.0
    p = AcquireRenderPanel(st)
    p.render_now()
    assert st.last_render is not None
    p.export_to(tmp_path)
    assert (tmp_path / "spectra_unmasked.pkl").exists()


class _Ev:
    """Minimal stand-in for a matplotlib MouseEvent."""
    def __init__(self, ax, x, y, button=1):
        self.inaxes = ax
        self.xdata = x
        self.ydata = y
        self.button = button


def _canvas_state():
    st = ForgeState(height=12, width=12)
    st.materials["m"] = Material("m", {"collagen": 1.0})
    st.add_layer("L", st.materials["m"])
    return st


def test_layers_panel_reorder_and_remove():
    from spectraforge.gui.panels.layers_panel import LayersPanel
    st = ForgeState(height=8, width=8)
    st.materials["a"] = Material("a", {"collagen": 1.0})
    st.materials["b"] = Material("b", {"NADH": 1.0})
    p = LayersPanel(st)
    p.add_layer_with_material("a")
    p.add_layer_with_material("b")
    assert [layer.name for layer in st.layers] == ["a", "b"]
    p.move_up(1)
    assert [layer.name for layer in st.layers] == ["b", "a"]
    p.move_down(0)
    assert [layer.name for layer in st.layers] == ["a", "b"]
    p.remove_layer(0)
    assert [layer.name for layer in st.layers] == ["b"]


def test_layers_panel_checkbox_toggles_visibility():
    from PyQt6.QtCore import Qt
    from spectraforge.gui.panels.layers_panel import LayersPanel
    st = ForgeState(height=8, width=8)
    st.materials["a"] = Material("a", {"collagen": 1.0})
    p = LayersPanel(st)
    p.add_layer_with_material("a")
    item = p._list.item(0)
    item.setCheckState(Qt.CheckState.Unchecked)
    assert st.layers[0].visible is False
    item.setCheckState(Qt.CheckState.Checked)
    assert st.layers[0].visible is True


def test_canvas_brush_at_paints_disc():
    from spectraforge.gui.panels.canvas_panel import CanvasPanel
    st = _canvas_state()
    p = CanvasPanel(st)
    p.set_tool("brush")
    p.set_radius(2)
    p.brush_at(6, 6)
    am = st.layers[0].amount_map
    assert am[6, 6] == 1.0
    assert am[6, 8] == 1.0    # distance 2, within radius 2
    assert am[0, 0] == 0.0


def test_canvas_mouse_press_paints_with_brush():
    from spectraforge.gui.panels.canvas_panel import CanvasPanel
    st = _canvas_state()
    p = CanvasPanel(st)
    p.set_tool("brush")
    p.set_radius(1)
    p._on_press(_Ev(p._canvas.ax, x=4.0, y=7.0))
    assert st.layers[0].amount_map[7, 4] == 1.0


def test_canvas_mouse_rect_paints_on_release():
    from spectraforge.gui.panels.canvas_panel import CanvasPanel
    st = _canvas_state()
    p = CanvasPanel(st)
    p.set_tool("rect")
    p._on_press(_Ev(p._canvas.ax, x=2.0, y=3.0))
    p._on_release(_Ev(p._canvas.ax, x=6.0, y=8.0))
    am = st.layers[0].amount_map
    assert am[5, 4] > 0        # inside rows 3..8, cols 2..6
    assert am[0, 0] == 0.0


def test_canvas_eraser_zeros_disc():
    from spectraforge.gui.panels.canvas_panel import CanvasPanel
    st = _canvas_state()
    p = CanvasPanel(st)
    p.set_radius(3)
    p.brush_at(6, 6)
    assert st.layers[0].amount_map[6, 6] == 1.0
    p.set_tool("eraser")
    p._on_press(_Ev(p._canvas.ax, x=6.0, y=6.0))   # erase via mouse
    assert st.layers[0].amount_map[6, 6] == 0.0


def test_composite_first_layer_is_visible():
    # Regression: the first layer must not render as black (invisible on black bg).
    from spectraforge.gui.panels.canvas_panel import CanvasPanel
    st = _canvas_state()
    p = CanvasPanel(st)
    p.set_radius(3)
    p.brush_at(6, 6)
    rgb = p.composite()
    assert rgb[6, 6].sum() > 0.0      # painted pixel of layer 0 is a visible color
    assert rgb[0, 0].sum() == 0.0     # unpainted stays black


def test_acquire_render_panel_slice_preview():
    import numpy as np
    from spectraforge.gui.panels.acquire_render_panel import AcquireRenderPanel
    st = ForgeState(height=10, width=10)
    st.materials["m"] = Material("m", {"EGFP": 1.0})
    layer = st.add_layer("L", st.materials["m"])
    layer.amount_map[:] = 1.0
    p = AcquireRenderPanel(st)
    p.render_now()
    img = p.preview_slice(0, 0)
    assert img.shape == (10, 10)
    grid = st.acquisition.emission_grid()
    ex_idx = list(st.acquisition.excitations).index(488.0)   # EGFP excites at 488
    band_507 = int(np.argmin(np.abs(grid - 507)))            # its emission peak
    band_far = int(np.argmin(np.abs(grid - 380)))
    assert p.preview_slice(ex_idx, band_507).mean() > p.preview_slice(ex_idx, band_far).mean()


def test_forge_window_builds():
    from spectraforge.gui.app import ForgeWindow
    w = ForgeWindow()
    assert w._state is not None
