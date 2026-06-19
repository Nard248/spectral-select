import numpy as np
from spectraforge.material import Material
from spectraforge.gui.layer import Layer, build_scene


class _State:  # minimal duck-typed state for the unit test
    def __init__(self, h, w, layers):
        self.height, self.width, self.layers = h, w, layers


def _layer(name, fmap_val, mat, visible=True):
    amt = np.zeros((6, 6))
    amt[:3, :3] = fmap_val
    return Layer(name=name, material=mat, amount_map=amt, visible=visible)


def test_build_scene_sums_visible_layers():
    s = _State(6, 6, [
        _layer("a", 1.0, Material("a", {"collagen": 1.0})),
        _layer("b", 2.0, Material("b", {"collagen": 0.5, "NADH": 1.0})),
        _layer("hidden", 5.0, Material("h", {"collagen": 9.0}), visible=False),
    ])
    maps = build_scene(s).resolve()
    assert maps["collagen"][0, 0] == 1.0 * 1.0 + 0.5 * 2.0   # visible a + b; hidden excluded
    assert maps["NADH"][0, 0] == 1.0 * 2.0
    assert maps["collagen"][5, 5] == 0.0
