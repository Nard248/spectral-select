import numpy as np
from spectraforge.groundtruth import GroundTruth


def test_groundtruth_save_roundtrip(tmp_path):
    gt = GroundTruth(
        concentration_maps={"collagen": np.ones((4, 4)), "NADH": np.zeros((4, 4))},
        clean_cubes={340.0: np.ones((4, 4, 3))},
        emission_grid=np.array([400.0, 450.0, 500.0]),
        excitations=[340.0],
    )
    gt.save(tmp_path)
    assert (tmp_path / "groundtruth.npz").exists()
    assert (tmp_path / "groundtruth.json").exists()
    loaded = np.load(tmp_path / "groundtruth.npz")
    assert "conc__collagen" in loaded
    assert np.array_equal(loaded["conc__collagen"], np.ones((4, 4)))


def test_informative_bands_flags_signal():
    cube = np.zeros((2, 2, 4))
    cube[..., 2] = 5.0  # band 2 carries all the signal
    gt = GroundTruth(
        concentration_maps={"A": np.ones((2, 2))},
        clean_cubes={488.0: cube},
        emission_grid=np.array([400.0, 450.0, 500.0, 550.0]),
        excitations=[488.0],
    )
    info = gt.informative_bands(threshold=0.05)
    assert info[488.0].tolist() == [False, False, True, False]
