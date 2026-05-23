import numpy as np
from channel_select.adapters.pamap2 import windows_from_array, IMU_LOCAL_CHANNELS, GROUP_COL_OFFSETS


def test_windowing_groups_and_labels():
    # Build 100 rows, 54 cols. activity col=1. constant activity 4.
    arr = np.zeros((100, 54), dtype=float)
    arr[:, 1] = 4
    # put a ramp in hand-acc-x to verify it lands in the window
    hand_acc_x = GROUP_COL_OFFSETS["hand"] + IMU_LOCAL_CHANNELS[0]
    arr[:, hand_acc_x] = np.arange(100)
    ds = windows_from_array(arr, window=32, step=16, sampling_drop_transient=True)
    assert ds.axis_type == "temporal1d"
    assert ds.channels_per_group["hand"] == len(IMU_LOCAL_CHANNELS)
    assert "ankle" in ds.groups and "chest" in ds.groups
    assert ds.labels is not None
    assert int(ds.labels[0]) == 4
    # first window's hand-acc-x channel should be 0..31
    w0 = ds.data["hand"][0, :, 0].numpy()
    assert np.allclose(w0, np.arange(32))


def test_transient_activity_dropped():
    arr = np.zeros((64, 54), dtype=float)
    arr[:, 1] = 0  # all transient
    ds = windows_from_array(arr, window=32, step=16, sampling_drop_transient=True)
    assert ds.n_windows == 0
