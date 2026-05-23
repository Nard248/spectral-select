import numpy as np
from channel_select.adapters.pamap2_monster import (
    dataset_from_arrays, GROUP_STARTS, IMU_LOCAL_CHANNELS,
)


def test_dataset_from_arrays_shapes_and_grouping():
    N, C, T = 20, 52, 100
    X = np.zeros((N, C, T), dtype=np.float32)
    # mark hand-acc-x (global channel GROUP_STARTS['hand']+1) with a known value
    hand_acc_x = GROUP_STARTS["hand"] + IMU_LOCAL_CHANNELS[0]
    X[:, hand_acc_x, :] = 7.0
    y = np.arange(N) % 12
    subj = np.array([1] * 10 + [2] * 10)

    ds = dataset_from_arrays(X, y, subj)
    assert ds.axis_type == "temporal1d"
    assert set(ds.groups) == {"hand", "chest", "ankle"}
    assert ds.channels_per_group == {"hand": 9, "chest": 9, "ankle": 9}
    assert ds.data["hand"].shape == (N, T, 9)
    # the marked channel is local index 0 of the hand group
    assert np.allclose(ds.data["hand"][:, :, 0].numpy(), 7.0)
    # LOSO split works on subject ids
    tr, te = ds.loso_split(2)
    assert len(te) == 10 and len(tr) == 10
