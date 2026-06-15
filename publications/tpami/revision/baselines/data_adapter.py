"""Data adapters - load each dataset into a common feature/label format.

Each loader returns a dict with:
    features  : (n_pixels, n_valid_bands)  float32  pixel-by-band matrix
    labels    : (n_pixels,) int  class labels (or -1 for unlabeled/background)
    band_catalog : list[(ex_nm: int, em_nm: int)] in same order as features columns
    dataset   : str  short name
    valid_pixel_mask : (n_pixels,) bool  optional, True for labeled/in-ROI

For unlabeled datasets like Drop Data we additionally return:
    drop_id : (n_pixels,) int  per-pixel drop index (0 = background)
    drop_types : (n_drops,) int  Ward-on-full type per drop
    drop_mean_spectra_full : (n_drops, n_valid_bands) float  mean spectrum / drop
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Drop Data (full_cr, ruler-cropped)
# ---------------------------------------------------------------------------
def load_drop_data_full_cr():
    """Load Drop Data full_cr cube + drop labels + Ward type assignments.

    Reads from the safe `.npz` cache. Returns features as (n_pixels, n_bands)
    where pixels are restricted to the drop_mask (only inside-drop pixels).
    """
    NPZ = ROOT / "revision" / "figures" / "drop_data" / "full_cr_cube.npz"
    DROP_LABELS = ROOT / "Data" / "processed" / "Drop Data Cropped" / "drop_labels.npy"
    DROP_TYPES = ROOT / "results" / "Drop_Data_Cropped_Sweep" / "full_cr" / "drop_types.npy"
    DROP_MEAN_SPECTRA = ROOT / "results" / "Drop_Data_Cropped_Sweep" / "full_cr" / "drop_mean_spectra.npy"

    if not NPZ.exists():
        raise FileNotFoundError(
            f"Drop Data cache missing: {NPZ}. Run the pkl->npz conversion first."
        )

    # Build per-excitation cubes + band catalog
    with np.load(NPZ) as z:
        ex_grid = z["ex_grid"].tolist()
        cubes = {ex: z[f"cube_{ex}"] for ex in ex_grid}
        wls = {ex: z[f"wl_{ex}"].tolist() for ex in ex_grid}

    band_catalog = []
    for ex in ex_grid:
        for em in wls[ex]:
            band_catalog.append((int(ex), int(em)))
    n_bands = len(band_catalog)

    drop_labels = np.load(DROP_LABELS)  # (H, W) int, 0=bg, 1..16=drops
    drop_pixel_mask = drop_labels > 0
    pixel_yx = np.argwhere(drop_pixel_mask)  # (n_pix, 2)
    n_pix = len(pixel_yx)

    features = np.zeros((n_pix, n_bands), dtype=np.float32)
    col = 0
    for ex in ex_grid:
        cube = cubes[ex]
        n_em = len(wls[ex])
        # Slice pixels for this excitation: (n_pix, n_em)
        pix_vals = cube[pixel_yx[:, 0], pixel_yx[:, 1], :]
        features[:, col:col + n_em] = pix_vals
        col += n_em
    assert col == n_bands

    drop_id = drop_labels[drop_pixel_mask]  # (n_pix,) int 1..16

    drop_types = np.load(DROP_TYPES)  # (16,) int
    drop_mean_spectra_full = np.load(DROP_MEAN_SPECTRA)  # (16, n_bands_full)

    # For ARI evaluation, we use Ward(full-spectrum) types as ground truth labels.
    # Map per-pixel via drop_id -> type (drop_id=1 -> drop_types[0], etc).
    labels = np.full(n_pix, -1, dtype=int)
    for d in range(1, drop_types.size + 1):
        labels[drop_id == d] = int(drop_types[d - 1])

    return {
        "features": features,
        "labels": labels,  # per-pixel type label (0..2)
        "band_catalog": band_catalog,
        "dataset": "drop_data_full_cr",
        "drop_id": drop_id,
        "drop_types": drop_types,
        "drop_mean_spectra_full": drop_mean_spectra_full,
    }


# ---------------------------------------------------------------------------
# Stub loaders for Lichens and Pepsin
#   These will be implemented as Phase 2 / Phase 3 work loads them.
# ---------------------------------------------------------------------------
def load_lichens(subsample_per_class: int | None = 5000):
    """Load Lichens Dataset 1 from cached npz + labels.npy.

    Parameters
    ----------
    subsample_per_class : int, optional
        If set, randomly subsample this many pixels per class for the
        feature matrix (otherwise we'd have ~191k pixels x 192 bands x
        every method call - too slow for the runner). Default 5000.
        Set to None for the full 191046-pixel matrix.
    """
    NPZ = ROOT / "revision" / "baselines" / "lichens_cube.npz"
    LABELS = ROOT / "revision" / "baselines" / "lichens_labels.npy"
    if not NPZ.exists():
        raise FileNotFoundError(
            f"Lichens cube cache missing: {NPZ}. Run the one-time conversion."
        )
    if not LABELS.exists():
        raise FileNotFoundError(
            f"Lichens labels missing: {LABELS}. Run the one-time conversion."
        )

    with np.load(NPZ) as z:
        ex_grid = z["ex_grid"].tolist()
        cubes = {ex: z[f"cube_{ex}"] for ex in ex_grid}
        wls = {ex: z[f"wl_{ex}"].tolist() for ex in ex_grid}

    band_catalog = []
    for ex in ex_grid:
        for em in wls[ex]:
            band_catalog.append((int(ex), int(em)))
    n_bands = len(band_catalog)

    labels_img = np.load(LABELS)  # (H, W) int, -1 for unlabeled
    labeled_mask = labels_img >= 1
    pixel_yx = np.argwhere(labeled_mask)
    pixel_labels = labels_img[labeled_mask]

    # Optional subsampling per class to keep runner-friendly
    if subsample_per_class is not None:
        rng = np.random.default_rng(0)
        keep = []
        for cid in np.unique(pixel_labels):
            ids = np.where(pixel_labels == cid)[0]
            if len(ids) > subsample_per_class:
                ids = rng.choice(ids, size=subsample_per_class, replace=False)
            keep.append(ids)
        keep = np.concatenate(keep)
        pixel_yx = pixel_yx[keep]
        pixel_labels = pixel_labels[keep]

    n_pix = len(pixel_yx)
    features = np.zeros((n_pix, n_bands), dtype=np.float32)
    col = 0
    for ex in ex_grid:
        cube = cubes[ex]
        n_em = len(wls[ex])
        pix_vals = cube[pixel_yx[:, 0], pixel_yx[:, 1], :]
        features[:, col:col + n_em] = pix_vals
        col += n_em

    return {
        "features": features,
        "labels": pixel_labels.astype(int),
        "band_catalog": band_catalog,
        "dataset": "lichens",
        "pixel_yx": pixel_yx,
    }


def load_pepsin(subsample_per_class: int | None = 2000):
    """Load Pepsin (Collagen) dataset from cached npz + labels.npy.

    6 excitations x variable emission = 158 valid bands. 3 classes (IDs 2, 3, 4).
    """
    NPZ = ROOT / "revision" / "baselines" / "pepsin_cube.npz"
    LABELS = ROOT / "revision" / "baselines" / "pepsin_labels.npy"
    if not NPZ.exists() or not LABELS.exists():
        raise FileNotFoundError(f"Run the one-time Pepsin conversion first.")

    with np.load(NPZ) as z:
        ex_grid = z["ex_grid"].tolist()
        cubes = {ex: z[f"cube_{ex}"] for ex in ex_grid}
        wls = {ex: z[f"wl_{ex}"].tolist() for ex in ex_grid}

    band_catalog = []
    for ex in ex_grid:
        for em in wls[ex]:
            band_catalog.append((int(ex), int(em)))
    n_bands = len(band_catalog)

    labels_img = np.load(LABELS)
    labeled_mask = labels_img >= 1
    pixel_yx = np.argwhere(labeled_mask)
    pixel_labels = labels_img[labeled_mask]

    if subsample_per_class is not None:
        rng = np.random.default_rng(0)
        keep = []
        for cid in np.unique(pixel_labels):
            ids = np.where(pixel_labels == cid)[0]
            if len(ids) > subsample_per_class:
                ids = rng.choice(ids, size=subsample_per_class, replace=False)
            keep.append(ids)
        keep = np.concatenate(keep)
        pixel_yx = pixel_yx[keep]
        pixel_labels = pixel_labels[keep]

    n_pix = len(pixel_yx)
    features = np.zeros((n_pix, n_bands), dtype=np.float32)
    col = 0
    for ex in ex_grid:
        cube = cubes[ex]
        n_em = len(wls[ex])
        pix_vals = cube[pixel_yx[:, 0], pixel_yx[:, 1], :]
        features[:, col:col + n_em] = pix_vals
        col += n_em

    return {
        "features": features,
        "labels": pixel_labels.astype(int),
        "band_catalog": band_catalog,
        "dataset": "pepsin",
        "pixel_yx": pixel_yx,
    }


DATASET_LOADERS = {
    "drop_data_full_cr": load_drop_data_full_cr,
    "lichens": load_lichens,
    "pepsin": load_pepsin,
}
