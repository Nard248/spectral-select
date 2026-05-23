"""PAMAP2 -> GroupedChannelDataset.

Dataset: UCI PAMAP2 Physical Activity Monitoring.
Manual download (if auto-fetch unavailable):
  https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip
Unzip to Data/Raw/PAMAP2/Protocol/*.dat (subject101.dat ... subject109.dat).

Column layout (54 cols per row):
  0 timestamp, 1 activity_id, 2 heart_rate, then 3 IMU blocks of 17 cols each
  (hand @3, chest @20, ankle @37). Within an IMU block the local offsets are:
  0 temp; 1-3 acc(+/-16g); 4-6 acc(+/-6g); 7-9 gyro; 10-12 mag; 13-16 orientation.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from ..data import GroupedChannelDataset

# Start column of each IMU block.
GROUP_COL_OFFSETS = {"hand": 3, "chest": 20, "ankle": 37}
# Local offsets within an IMU block: acc16g (1,2,3), gyro (7,8,9), mag (10,11,12).
IMU_LOCAL_CHANNELS = [1, 2, 3, 7, 8, 9, 10, 11, 12]


def windows_from_array(arr: np.ndarray, window: int = 256, step: int = 128,
                       sampling_drop_transient: bool = True,
                       subject_id: int | None = None) -> GroupedChannelDataset:
    activity = arr[:, 1].astype(int)
    per_group_windows: dict[str, list[np.ndarray]] = {g: [] for g in GROUP_COL_OFFSETS}
    labels: list[int] = []
    subjects: list[int] = []
    for start in range(0, len(arr) - window + 1, step):
        sl = slice(start, start + window)
        win_act = activity[sl]
        vals, counts = np.unique(win_act, return_counts=True)
        lab = int(vals[np.argmax(counts)])
        if sampling_drop_transient and lab == 0:
            continue
        for g, off in GROUP_COL_OFFSETS.items():
            cols = [off + c for c in IMU_LOCAL_CHANNELS]
            block = arr[sl][:, cols]                       # (window, n_ch)
            block = np.nan_to_num(block, nan=0.0)
            per_group_windows[g].append(block)
        labels.append(lab)
        subjects.append(subject_id if subject_id is not None else -1)

    n_ch = len(IMU_LOCAL_CHANNELS)
    data = {
        g: (torch.tensor(np.stack(w), dtype=torch.float32) if w
            else torch.zeros(0, window, n_ch, dtype=torch.float32))
        for g, w in per_group_windows.items()
    }
    return GroupedChannelDataset(
        data, axis_type="temporal1d",
        labels=torch.tensor(labels, dtype=torch.long),
        subject_ids=torch.tensor(subjects, dtype=torch.long),
    )


def load_pamap2(protocol_dir: str | Path, window: int = 256, step: int = 128,
                subjects: list[int] | None = None) -> GroupedChannelDataset:
    protocol_dir = Path(protocol_dir)
    files = sorted(protocol_dir.glob("subject1*.dat"))
    if not files:
        raise FileNotFoundError(
            f"No PAMAP2 .dat files in {protocol_dir}. See module docstring for download URL."
        )
    parts: list[GroupedChannelDataset] = []
    for f in files:
        sid = int(f.stem.replace("subject", ""))
        if subjects is not None and sid not in subjects:
            continue
        arr = np.loadtxt(f)
        parts.append(windows_from_array(arr, window, step, subject_id=sid))
    groups = parts[0].groups
    data = {g: torch.cat([p.data[g] for p in parts], dim=0) for g in groups}
    labels = torch.cat([p.labels for p in parts])
    subjects_t = torch.cat([p.subject_ids for p in parts])
    return GroupedChannelDataset(data, "temporal1d", labels=labels, subject_ids=subjects_t)
