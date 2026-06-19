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


def test_forge_window_builds():
    from spectraforge.gui.app import ForgeWindow
    w = ForgeWindow()
    assert w._state is not None
