import numpy as np
from spectraforge.fluorophore import Fluorophore
from spectraforge.material import Material
from spectraforge.gui.state import ForgeState
from spectraforge.gui.project import save_project, load_project


def test_project_roundtrip(tmp_path):
    st = ForgeState(height=8, width=8)
    st.library["MyDye"] = Fluorophore("MyDye", 500, 40, 560, 50, 0.7, 0.9)
    st.materials["m"] = Material("m", {"collagen": 1.0, "MyDye": 0.5})
    layer = st.add_layer("L", st.materials["m"])
    layer.amount_map[2:5, 2:5] = 0.7
    layer.visible = False
    st.acquisition.exposure[340.0] = 2.0
    st.seed = 7

    path = tmp_path / "proj.npz"
    save_project(st, path)
    st2 = load_project(path)

    assert (st2.height, st2.width) == (8, 8)
    assert "MyDye" in st2.library and st2.library["MyDye"].em_peak_nm == 560
    assert st2.materials["m"].recipe["MyDye"] == 0.5
    assert len(st2.layers) == 1
    assert st2.layers[0].name == "L"
    assert st2.layers[0].visible is False
    assert st2.layers[0].material.recipe["collagen"] == 1.0
    assert np.allclose(st2.layers[0].amount_map, layer.amount_map)
    assert st2.acquisition.exposure_for(340.0) == 2.0
    assert st2.seed == 7
