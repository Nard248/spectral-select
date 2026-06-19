from spectraforge.material import Material


def test_material_recipe():
    m = Material("tissue", {"collagen": 1.0, "NADH": 0.4})
    assert m.recipe["collagen"] == 1.0
    assert set(m.fluorophores()) == {"collagen", "NADH"}
