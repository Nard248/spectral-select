"""Load the Lichens 2 processed cube without modifying source data."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from scipy import ndimage

from .rayleigh_filter import RejectedPair, WavelengthPair, apply_rayleigh_filter


@dataclass
class SelectionDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    pairs: list[WavelengthPair]
    raw_pair_count: int
    physically_accepted_count: int
    rejected: list[RejectedPair]
    source_path: Path
    mask_path: Path


def _pairs_from_record(record: dict, excitation: float) -> list[WavelengthPair]:
    wavelengths = record.get("em_arr", record.get("wavelengths"))
    return [WavelengthPair(float(excitation), float(w)) for w in np.asarray(wavelengths)]


def _connected_groups(mask: np.ndarray) -> np.ndarray:
    groups = np.full(mask.shape, -1, dtype=np.int32)
    next_group = 0
    for cls in sorted(int(v) for v in np.unique(mask) if v >= 0):
        components, count = ndimage.label(mask == cls)
        for component in range(1, count + 1):
            groups[components == component] = next_group
            next_group += 1
    return groups


def load_lichen_dataset(
    processed_path: str | Path,
    mask_path: str | Path,
    offset_nm: float = 40.0,
    max_pixels_per_group: int | None = 200,
    seed: int = 42,
) -> SelectionDataset:
    """Load normalized features and object-group labels for leakage-safe CV.

    Candidate filtering is derived from the embedded raw wavelength grid. Features
    are taken from the already exposure- and lamp-power-normalized processed cube.
    """
    processed_path, mask_path = Path(processed_path), Path(mask_path)
    with processed_path.open("rb") as handle:
        payload = pickle.load(handle)
    mask = np.load(mask_path)
    processed = payload["data"]
    raw = payload.get("raw_data", processed)

    raw_pairs: list[WavelengthPair] = []
    for key, record in raw.items():
        raw_pairs.extend(_pairs_from_record(record, float(record.get("ex", key))))
    accepted, rejected = apply_rayleigh_filter(raw_pairs, offset_nm)

    cube_by_pair: dict[WavelengthPair, np.ndarray] = {}
    for key, record in processed.items():
        excitation = float(record.get("excitation", record.get("ex", key)))
        cube = np.asarray(record.get("cube", record.get("data")))
        for index, pair in enumerate(_pairs_from_record(record, excitation)):
            cube_by_pair[pair] = cube[..., index]
    usable = [pair for pair in accepted if pair in cube_by_pair]
    if not usable:
        raise ValueError("No physically accepted wavelength pairs exist in the processed cube")

    group_image = _connected_groups(mask)
    coords = np.argwhere(mask >= 0)
    if max_pixels_per_group is not None:
        rng = np.random.default_rng(seed)
        chosen: list[np.ndarray] = []
        for group in np.unique(group_image[mask >= 0]):
            candidates = np.argwhere(group_image == group)
            if len(candidates) > max_pixels_per_group:
                candidates = candidates[rng.choice(len(candidates), max_pixels_per_group, replace=False)]
            chosen.append(candidates)
        coords = np.concatenate(chosen)
    rows, cols = coords[:, 0], coords[:, 1]
    X = np.column_stack([cube_by_pair[p][rows, cols] for p in usable]).astype(np.float32)
    return SelectionDataset(
        X=X,
        y=mask[rows, cols].astype(np.int32),
        groups=group_image[rows, cols],
        pairs=usable,
        raw_pair_count=len(raw_pairs),
        physically_accepted_count=len(accepted),
        rejected=rejected,
        source_path=processed_path,
        mask_path=mask_path,
    )
