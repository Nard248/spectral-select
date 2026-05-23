from __future__ import annotations
from typing import Hashable, Optional

import torch


class GroupedChannelDataset:
    """In-memory container conforming to GroupedChannelData.

    data: group -> Tensor[window, *axis_dims, channel]  (channel axis LAST)
    """

    def __init__(
        self,
        data: dict[Hashable, torch.Tensor],
        axis_type: str,
        labels: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> None:
        if axis_type not in ("spatial2d", "temporal1d"):
            raise ValueError(f"axis_type must be spatial2d|temporal1d, got {axis_type}")
        self.data = data
        self.axis_type = axis_type
        self.labels = labels
        self.subject_ids = subject_ids

    @property
    def groups(self) -> list[Hashable]:
        return list(self.data.keys())

    @property
    def channels_per_group(self) -> dict[Hashable, int]:
        return {g: int(t.shape[-1]) for g, t in self.data.items()}

    @property
    def n_windows(self) -> int:
        return int(next(iter(self.data.values())).shape[0])

    def get_all_data(self) -> dict[Hashable, torch.Tensor]:
        return self.data

    def subset(self, indices: list[int]) -> "GroupedChannelDataset":
        idx = torch.as_tensor(indices, dtype=torch.long)
        return GroupedChannelDataset(
            {g: t.index_select(0, idx) for g, t in self.data.items()},
            axis_type=self.axis_type,
            labels=None if self.labels is None else self.labels.index_select(0, idx),
            subject_ids=None if self.subject_ids is None else self.subject_ids.index_select(0, idx),
        )

    def loso_split(self, holdout_subject) -> tuple[list[int], list[int]]:
        if self.subject_ids is None:
            raise ValueError("loso_split requires subject_ids")
        train, test = [], []
        for i, s in enumerate(self.subject_ids.tolist()):
            (test if s == holdout_subject else train).append(i)
        return train, test
