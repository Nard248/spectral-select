#!/usr/bin/env python3
"""Export Lichens 2 to the compact binary format consumed by the native search."""

import argparse
from pathlib import Path
import pickle
import struct

import numpy as np
from scipy import ndimage


MAGIC = b"HSIEXACT"


def pair_key(excitation, emission):
    return round(float(excitation), 6), round(float(emission), 6)


def physically_valid(excitation, emission, offset):
    return emission >= excitation + offset and not (2 * excitation - offset <= emission <= 2 * excitation + offset)


def connected_groups(mask):
    groups = np.full(mask.shape, -1, dtype=np.int32)
    group_id = 0
    for cls in sorted(int(v) for v in np.unique(mask) if v >= 0):
        components, count = ndimage.label(mask == cls)
        for component in range(1, count + 1):
            groups[components == component] = group_id
            group_id += 1
    return groups


def load_data(processed_path, mask_path, offset, max_pixels, seed):
    with processed_path.open("rb") as handle:
        payload = pickle.load(handle)
    mask = np.load(mask_path)
    available = {}
    for key, record in payload["data"].items():
        ex = float(record.get("excitation", record.get("ex", key)))
        emissions = np.asarray(record.get("wavelengths", record.get("em_arr")))
        cube = np.asarray(record.get("cube", record.get("data")))
        for index, emission in enumerate(emissions):
            available[pair_key(ex, emission)] = cube[..., index]
    raw_pairs = []
    for key, record in payload.get("raw_data", payload["data"]).items():
        ex = float(record.get("ex", record.get("excitation", key)))
        emissions = np.asarray(record.get("em_arr", record.get("wavelengths")))
        raw_pairs.extend(pair_key(ex, emission) for emission in emissions)
    pairs = [p for p in raw_pairs if physically_valid(*p, offset) and p in available]

    group_image = connected_groups(mask)
    rng = np.random.default_rng(seed)
    coordinates = []
    for group in np.unique(group_image[mask >= 0]):
        choices = np.argwhere(group_image == group)
        if max_pixels and len(choices) > max_pixels:
            choices = choices[rng.choice(len(choices), max_pixels, replace=False)]
        coordinates.append(choices)
    coordinates = np.concatenate(coordinates)
    rows, columns = coordinates[:, 0], coordinates[:, 1]
    X = np.column_stack([available[p][rows, columns] for p in pairs]).astype("<f4")
    y = mask[rows, columns].astype("<i4")
    groups = group_image[rows, columns].astype("<i4")
    return X, y, groups, np.asarray(pairs, dtype="<f4")


def main():
    personal = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=personal / "Processed/Lichens_2/data_cutoff_40nm_exposure_max_power_min.pkl")
    parser.add_argument("--mask", type=Path, default=personal / "Processed/Lichens_2/lichens_2_class_mask.npy")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data/lichens2.bin")
    parser.add_argument("--offset-nm", type=float, default=40.0)
    parser.add_argument("--max-pixels-per-object", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    X, y, groups, pairs = load_data(args.processed, args.mask, args.offset_nm, args.max_pixels_per_object, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<IIII", X.shape[0], X.shape[1], int(y.max() + 1), int(groups.max() + 1)))
        handle.write(pairs.tobytes(order="C"))
        handle.write(y.tobytes(order="C"))
        handle.write(groups.tobytes(order="C"))
        handle.write(X.tobytes(order="C"))
    print(f"Exported {X.shape[0]} samples, {X.shape[1]} pairs, {y.max()+1} classes and {groups.max()+1} objects")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
