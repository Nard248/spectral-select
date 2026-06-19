from spectraforge.library import load_builtin_library
from spectraforge.fluorophore import Fluorophore


def test_library_loads_known_fluorophores():
    lib = load_builtin_library()
    assert "collagen" in lib and "EGFP" in lib and "fluorescein" in lib
    assert isinstance(lib["collagen"], Fluorophore)
    assert lib["collagen"].em_peak_nm == 390
    assert len(lib) >= 10
