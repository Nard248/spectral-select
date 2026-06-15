"""MONSTER-preprocessed PAMAP2 -> GroupedChannelDataset.

Source (ungated): https://huggingface.co/datasets/monster-monash/PAMAP2
Files: PAMAP2_X.npy (N, 52, 100), PAMAP2_y.csv, PAMAP2_subject_id.csv (headerless,
one value per line). Series are fixed 100-sample windows at 100 Hz, 12 classes.

The 52 channels follow the raw PAMAP2 column set with timestamp + activity removed:
  ch0 = heart rate; then three 17-column IMU blocks: hand @1, chest @18, ankle @35.
Within each 17-col block the local offsets are:
  0 temp; 1-3 acc(+/-16g); 4-6 acc(+/-6g); 7-9 gyro; 10-12 mag; 13-16 orientation.
We keep acc16g (1,2,3), gyro (7,8,9), mag (10,11,12) = 9 channels/IMU, matching the
raw-.dat adapter in pamap2.py.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from ..data import GroupedChannelDataset

GROUP_STARTS = {"hand": 1, "chest": 18, "ankle": 35}  # global start of each IMU block
IMU_LOCAL_CHANNELS = [1, 2, 3, 7, 8, 9, 10, 11, 12]   # acc16 xyz, gyro xyz, mag xyz


def dataset_from_arrays(X: np.ndarray, y: np.ndarray,
                        subject_ids: np.ndarray) -> GroupedChannelDataset:
    """X: (N, 52, time). Returns groups of shape (N, time, 9)."""
    data = {}
    for g, start in GROUP_STARTS.items():
        idx = [start + c for c in IMU_LOCAL_CHANNELS]
        block = X[:, idx, :]                              # (N, 9, time)
        block = np.nan_to_num(block, nan=0.0)
        data[g] = torch.tensor(block.transpose(0, 2, 1), dtype=torch.float32)  # (N, time, 9)
    return GroupedChannelDataset(
        data, axis_type="temporal1d",
        labels=torch.tensor(np.asarray(y).astype(int), dtype=torch.long),
        subject_ids=torch.tensor(np.asarray(subject_ids).astype(int), dtype=torch.long),
    )


def load_pamap2_monster(data_dir: str | Path) -> GroupedChannelDataset:
    data_dir = Path(data_dir)
    xpath = data_dir / "PAMAP2_X.npy"
    if not xpath.exists():
        raise FileNotFoundError(
            f"{xpath} missing. Download (ungated) from "
            "https://huggingface.co/datasets/monster-monash/PAMAP2 "
            "(PAMAP2_X.npy, PAMAP2_y.csv, PAMAP2_subject_id.csv)."
        )
    X = np.load(xpath)                                    # (N, 52, 100)
    y = np.loadtxt(data_dir / "PAMAP2_y.csv")             # headerless
    subj = np.loadtxt(data_dir / "PAMAP2_subject_id.csv")
    return dataset_from_arrays(X, y, subj)
