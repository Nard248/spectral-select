import numpy as np
from spectraforge.artifacts import ArtifactConfig, add_scatter_lines, add_noise


def test_rayleigh_adds_energy_at_excitation():
    em = np.arange(400, 700, 5.0)
    cube = np.zeros((3, 3, len(em)))
    cfg = ArtifactConfig(rayleigh_strength=1.0, rayleigh_fwhm=10, second_order=False)
    add_scatter_lines(cube, ex=500.0, em_grid=em, cfg=cfg, scale=1.0)
    peak = int(np.argmax(cube[0, 0]))
    assert em[peak] == 500.0
    assert cube[0, 0].max() > 0


def test_second_order_line_at_double_excitation():
    em = np.arange(400, 900, 5.0)
    cube = np.zeros((2, 2, len(em)))
    cfg = ArtifactConfig(rayleigh_strength=1.0, rayleigh_fwhm=10, second_order=True)
    add_scatter_lines(cube, ex=320.0, em_grid=em, cfg=cfg, scale=1.0)
    assert abs(em[int(np.argmax(cube[0, 0]))] - 640.0) <= 5.0  # 2*320


def test_noise_is_seed_deterministic():
    cube = np.full((4, 4, 5), 10.0)
    cfg = ArtifactConfig(photon_scale=1.0, read_sigma=0.5)
    a = add_noise(cube.copy(), cfg, np.random.default_rng(0))
    b = add_noise(cube.copy(), cfg, np.random.default_rng(0))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, cube)          # noise actually applied
    assert abs(a.mean() - 10.0) < 1.0           # mean approximately preserved
