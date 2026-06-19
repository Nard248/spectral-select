from spectraforge.acquisition import AcquisitionConfig


def test_emission_grid_and_scalars():
    a = AcquisitionConfig(excitations=[340, 488], em_min=400, em_max=700, em_step=5,
                          exposure={340: 2.0}, power={488: 0.5})
    grid = a.emission_grid()
    assert grid[0] == 400 and grid[-1] == 700 and grid[1] - grid[0] == 5
    assert a.exposure_for(340) == 2.0
    assert a.exposure_for(488) == 1.0  # default
    assert a.power_for(488) == 0.5
    assert a.lamp_for(340) == 1.0
