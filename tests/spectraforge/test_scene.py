import numpy as np
from spectraforge.material import Material
from spectraforge.scene import Scene


def test_paint_rect_sets_concentration():
    s = Scene(20, 20)
    s.paint_rect(Material("m", {"collagen": 2.0}), 5, 10, 5, 10)
    maps = s.resolve()
    assert maps["collagen"][7, 7] == 2.0
    assert maps["collagen"][0, 0] == 0.0


def test_painting_is_additive_and_mixes():
    s = Scene(10, 10)
    s.paint_rect(Material("a", {"collagen": 1.0}), 0, 10, 0, 10)
    s.paint_rect(Material("b", {"collagen": 0.5, "NADH": 1.0}), 0, 5, 0, 5)
    maps = s.resolve()
    assert maps["collagen"][2, 2] == 1.5   # overlap accumulates
    assert maps["collagen"][7, 7] == 1.0
    assert maps["NADH"][2, 2] == 1.0


def test_scene_addition_sums_maps():
    a = Scene(8, 8)
    a.paint_rect(Material("m", {"collagen": 1.0}), 0, 4, 0, 4)
    b = Scene(8, 8)
    b.paint_rect(Material("m", {"collagen": 1.0}), 0, 8, 0, 8)
    c = a + b
    assert c.resolve()["collagen"][1, 1] == 2.0
    assert c.resolve()["collagen"][6, 6] == 1.0


def test_paint_circle():
    s = Scene(21, 21)
    s.paint_circle(Material("m", {"EGFP": 1.0}), cy=10, cx=10, radius=5)
    maps = s.resolve()
    assert maps["EGFP"][10, 10] == 1.0
    assert maps["EGFP"][0, 0] == 0.0
