import torch
from channel_select.data import GroupedChannelDataset


def _toy():
    data = {
        "hand": torch.zeros(4, 8, 3),   # (window, time, channel)
        "ankle": torch.zeros(4, 8, 2),
    }
    labels = torch.tensor([0, 0, 1, 1])
    subjects = torch.tensor([1, 1, 2, 2])
    return GroupedChannelDataset(data, axis_type="temporal1d", labels=labels, subject_ids=subjects)


def test_properties():
    ds = _toy()
    assert ds.groups == ["hand", "ankle"]
    assert ds.channels_per_group == {"hand": 3, "ankle": 2}
    assert ds.axis_type == "temporal1d"
    assert ds.n_windows == 4


def test_subset_by_index():
    ds = _toy()
    sub = ds.subset([0, 2])
    assert sub.n_windows == 2
    assert sub.data["hand"].shape == (2, 8, 3)
    assert sub.labels.tolist() == [0, 1]
    assert sub.subject_ids.tolist() == [1, 2]


def test_loso_split_indices():
    ds = _toy()
    train_idx, test_idx = ds.loso_split(holdout_subject=2)
    assert train_idx == [0, 1]
    assert test_idx == [2, 3]
