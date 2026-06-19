import numpy as np
from spectraforge.material import Material
from spectraforge.gui.state import ForgeState
from spectraforge.gui.layer import build_scene


def test_state_defaults_and_add_layer():
    st = ForgeState(height=8, width=8)
    assert "collagen" in st.library            # built-in library loaded
    assert st.layers == []
    layer = st.add_layer("patch", Material("m", {"collagen": 1.0}))
    assert layer.amount_map.shape == (8, 8)
    assert st.layers[-1] is layer
    assert st.active_layer == 0
    layer.amount_map[0, 0] = 1.0
    assert build_scene(st).resolve()["collagen"][0, 0] == 1.0


def test_state_has_default_acquisition_and_artifacts():
    st = ForgeState(height=4, width=4)
    assert len(st.acquisition.excitations) >= 1
    assert st.artifacts is not None
