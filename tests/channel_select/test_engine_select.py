import numpy as np
import torch
from channel_select.engine import select_channels


def test_topk_no_diversity():
    infl = {"g": np.array([0.1, 0.9, 0.5])}
    sel = select_channels(infl, data={}, K=2, method="none", lambda_diversity=0.5)
    assert sel == [("g", 1), ("g", 2)]  # by influence desc


def test_mmr_penalizes_redundant_channel():
    # channels 0 and 1 identical profile (redundant); channel 2 distinct.
    data = {"g": torch.tensor([
        [[1.0, 1.0, 0.0]],
        [[2.0, 2.0, 9.0]],
        [[3.0, 3.0, 1.0]],
    ])}  # (window=3, time=1, channel=3)
    # ch1 and ch2 are equally relevant, so redundancy is the deciding factor:
    # ch1 is a perfect copy of the already-picked ch0, ch2 is distinct.
    infl = {"g": np.array([1.0, 0.6, 0.6])}
    sel = select_channels(data=data, influence=infl, K=2, method="mmr", lambda_diversity=0.9)
    # picks ch0 first (highest), then ch2 (distinct) over ch1 (redundant copy of ch0)
    assert sel[0] == ("g", 0)
    assert sel[1] == ("g", 2)
