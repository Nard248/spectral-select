import numpy as np
import torch
from channel_select.engine import normalize_influence


def test_max_per_group():
    infl = {"g": np.array([2.0, 4.0])}
    out = normalize_influence(infl, data={}, method="max_per_group")
    assert np.allclose(out["g"], [0.5, 1.0])


def test_variance_divides_by_per_channel_variance():
    # data (window=3, time=1, channel=2); channel 0 has var 0, channel 1 var>0
    data = {"g": torch.tensor([[[1.0, 0.0]], [[1.0, 3.0]], [[1.0, 6.0]]])}
    infl = {"g": np.array([1.0, 9.0])}
    out = normalize_influence(infl, data=data, method="variance")
    var1 = np.var(np.array([0.0, 3.0, 6.0]))  # = 6.0 (biased, matches np.var default)
    assert out["g"][0] == 1.0 / 1e-10          # zero variance clamped
    assert np.isclose(out["g"][1], 9.0 / var1)
