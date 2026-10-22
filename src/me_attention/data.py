"""Loaders for the four ME fluorescence datasets -> ``MEData`` (see docs/me_attention/SPEC.md).

Wavelength alignment is by real nm values (never by index): every cube must come with a per-excitation
``wavelengths`` vector, otherwise a ``ValueError`` is raised (review bug B1).
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label

from .types import MEData

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "Data" / "processed"
CACHE_DIR = REPO / "results" / "me_attention" / "cache"
MIN_SPECIMEN_PIXELS = 100

_STRUCT8 = np.ones((3, 3), dtype=np.uint8)

DATASETS: dict[str, dict[str, Any]] = {
    "collagen_acetic": dict(kind="png", dir="Collagen_Acetic_Acid", pkl="spectra_masked.pkl"),
    "collagen_pepsin": dict(kind="png", dir="Collagen Pepsin", pkl="spectra_masked.pkl"),
    "lichens1": dict(kind="png", dir="Lichens Dataset 1", pkl="spectra_masked.pkl"),
    "lichens2": dict(kind="npy", dir="Lichens_2", pkl="data_cutoff_40nm_exposure_max_power_min.pkl"),
}

# Emission-axis correction (dataset audit 2026-10-22, docs/datasets/audit_2026-10-22). The processed pickles were made by a
# loader that did not read the emission axis from the .im3 files: it assumed the first band is 420 nm for excitations
# <= 400 nm and ex + 20 nm above. The .im3 headers (Data/Raw/*, Zenodo 18640119 HSI.zip) record different axes for the
# cubes below. Adding these offsets (nm) to the stored labels gives the recorded wavelengths; band values, the set of
# retained bands and their order are unchanged. Opt-in via ME_ATTENTION_RECORDED_WL=1 (default keeps the labels used in
# the 2026-10-19 sweep, for reproducibility).
RECORDED_WL_OFFSET: dict[str, dict[float, float]] = {
    "lichens1": {385.0: 20.0, 400.0: 40.0, 415.0: 35.0, 430.0: 40.0},  # recorded 440/460/470/490-720 nm
    "lichens2": {415.0: -15.0, 430.0: -30.0},  # recorded 420-720 nm
    "collagen_pepsin": {400.0: 30.0},  # recorded 450-720 nm
    "collagen_acetic": {400.0: 30.0},  # recorded 450-720 nm
}


def use_recorded_wavelengths() -> bool:
    return os.environ.get("ME_ATTENTION_RECORDED_WL", "0") == "1"


# ----------------------------------------------------------------------------- masks
def _class_map_png(d: Path) -> tuple[np.ndarray, list[str]]:
    from PIL import Image

    roi = json.loads((d / "roi_regions.json").read_text())
    rgb = np.array(Image.open(d / "class_mask.png").convert("RGB"))
    cmap = np.full(rgb.shape[:2], -1, dtype=np.int16)
    names = []
    for cls, info in enumerate(roi["classes"]):
        cmap[np.all(rgb == np.asarray(info["color"]), axis=2)] = cls
        names.append(str(info.get("name", f"Class {cls}")))
    return cmap, names


def _class_map_npy(d: Path) -> tuple[np.ndarray, list[str]]:
    cmap = np.load(d / "lichens_2_class_mask.npy").astype(np.int16)
    info = json.loads((d / "lichens_2_class_info.json").read_text())
    n = int(cmap.max()) + 1
    names = [str(info[str(c)]["name"]) for c in range(n)]
    return cmap, names


def specimens_from_class_map(class_map: np.ndarray, n_classes: int, min_px: int = MIN_SPECIMEN_PIXELS):
    """8-connected components per class; components < ``min_px`` dropped. Returns (specimen_map, classes, dropped)."""
    spec = np.full(class_map.shape, -1, dtype=np.int64)
    classes: list[int] = []
    dropped_px = 0
    dropped_n = 0
    sid = 0
    for c in range(n_classes):
        comp, n = label(class_map == c, structure=_STRUCT8)
        for k in range(1, n + 1):
            m = comp == k
            size = int(m.sum())
            if size < min_px:
                dropped_px += size
                dropped_n += 1
                continue
            spec[m] = sid
            classes.append(c)
            sid += 1
    return spec, np.asarray(classes, dtype=np.int64), dropped_px, dropped_n


# ----------------------------------------------------------------------------- features
def _wavelength_vector(entry: dict, ex: float) -> np.ndarray:
    for key, value in entry.items():
        if key == "cube":
            continue
        k = key.lower()
        arr = np.asarray(value)
        if ("wavelength" in k or "emission" in k) and arr.ndim == 1 and arr.dtype.kind in "fiu":
            return arr.astype(float)
    raise ValueError(
        f"excitation {ex}: no 1-D wavelength vector among keys {list(entry.keys())}; "
        "refusing index alignment (bug B1)")


def build_features(cubes: list[tuple[float, np.ndarray, np.ndarray]], foreground: np.ndarray):
    """cubes: [(ex_nm, cube [H,W,B], wavelengths [B])]. Returns X, ex_nm, em_nm, excitations, emissions, grid_index."""
    cubes = sorted(cubes, key=lambda t: t[0])
    excitations = np.array([c[0] for c in cubes], dtype=float)
    emissions = np.array(sorted({round(float(w), 6) for _, _, w in cubes for w in w}), dtype=float)
    lookup = {round(float(w), 6): i for i, w in enumerate(emissions)}
    cols, ex_l, em_l, gi = [], [], [], []
    for e_idx, (ex, cube, wl) in enumerate(cubes):
        if cube.shape[:2] != foreground.shape:
            raise ValueError(f"excitation {ex}: cube {cube.shape[:2]} != mask {foreground.shape}")
        if len(wl) != cube.shape[-1]:
            raise ValueError(f"excitation {ex}: {len(wl)} wavelengths vs {cube.shape[-1]} bands")
        order = np.argsort(wl, kind="stable")
        sub = cube[foreground][:, order].astype(np.float32)
        cols.append(sub)
        for w in wl[order]:
            ex_l.append(ex)
            em_l.append(float(w))
            gi.append((e_idx, lookup[round(float(w), 6)]))
    X = np.concatenate(cols, axis=1)
    return (X, np.asarray(ex_l, float), np.asarray(em_l, float), excitations, emissions,
            np.asarray(gi, dtype=np.int64))


# ----------------------------------------------------------------------------- cache
def _cache_path(key: str) -> Path:
    return CACHE_DIR / (f"{key}_recwl.npz" if use_recorded_wavelengths() else f"{key}.npz")


def _save_cache(key: str, d: MEData) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _cache_path(key), X=d.X, ex_nm=d.ex_nm, em_nm=d.em_nm, y=d.y, groups=d.groups, coords=d.coords,
        specimen_classes=d.specimen_classes, excitations=d.excitations, emissions=d.emissions,
        grid_index=d.grid_index, specimen_map=d.specimen_map,
        meta=np.array(json.dumps(dict(name=d.name, class_names=d.class_names,
                                      image_shape=list(d.image_shape), info=d.info))))


def _load_cache(key: str) -> MEData | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return MEData(
        name=meta["name"], X=z["X"], ex_nm=z["ex_nm"], em_nm=z["em_nm"], y=z["y"], groups=z["groups"],
        coords=z["coords"], specimen_classes=z["specimen_classes"], class_names=meta["class_names"],
        image_shape=tuple(meta["image_shape"]), excitations=z["excitations"], emissions=z["emissions"],
        grid_index=z["grid_index"], specimen_map=z["specimen_map"], info=meta["info"])


# ----------------------------------------------------------------------------- public API
def available() -> list[str]:
    return list(DATASETS)


def load(key: str, use_cache: bool = True, data_root: Path | None = None,
         min_specimen_pixels: int = MIN_SPECIMEN_PIXELS) -> MEData:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; choose from {available()}")
    if use_cache and data_root is None and min_specimen_pixels == MIN_SPECIMEN_PIXELS:
        cached = _load_cache(key)
        if cached is not None:
            return cached
    cfg = DATASETS[key]
    d = Path(data_root or DATA_ROOT) / cfg["dir"]
    class_map, names = (_class_map_png if cfg["kind"] == "png" else _class_map_npy)(d)
    n_classes = len(names)
    spec_map, spec_classes, dropped_px, dropped_n = specimens_from_class_map(class_map, n_classes, min_specimen_pixels)
    fg = spec_map >= 0

    with open(d / cfg["pkl"], "rb") as f:
        raw = pickle.load(f)
    cubes = []
    for ex in raw["excitation_wavelengths"]:
        entry = raw["data"][str(float(ex))]
        wl = _wavelength_vector(entry, ex)
        if use_recorded_wavelengths():
            wl = wl + RECORDED_WL_OFFSET.get(key, {}).get(float(ex), 0.0)
        cubes.append((float(ex), np.asarray(entry["cube"]), wl))
    if "mask" in raw:
        sm = np.asarray(raw["mask"]).astype(bool)
        if sm.shape != fg.shape:
            raise ValueError(f"spectral mask {sm.shape} != class mask {fg.shape}")
        if not sm[fg].all():
            raise ValueError("some kept specimen pixels have no spectral measurement")
    X, ex_nm, em_nm, excitations, emissions, grid_index = build_features(cubes, fg)
    del raw, cubes
    rows, cols = np.nonzero(fg)
    y = class_map[rows, cols].astype(np.int64)
    groups = spec_map[rows, cols].astype(np.int64)
    assert (spec_classes[groups] == y).all()
    info = dict(
        source=str(Path(cfg["dir"]) / cfg["pkl"]), recorded_wavelengths=use_recorded_wavelengths(), dropped_px=int(dropped_px), dropped_components=int(dropped_n),
        min_specimen_pixels=min_specimen_pixels, n_nonfinite=int((~np.isfinite(X)).sum()),
        specimens_per_class=np.bincount(spec_classes, minlength=n_classes).tolist())
    data = MEData(
        name=key, X=X, ex_nm=ex_nm, em_nm=em_nm, y=y, groups=groups,
        coords=np.stack([rows, cols], 1).astype(np.int64), specimen_classes=spec_classes, class_names=names,
        image_shape=(int(fg.shape[0]), int(fg.shape[1])), excitations=excitations, emissions=emissions,
        grid_index=grid_index, specimen_map=spec_map, info=info)
    if use_cache and data_root is None and min_specimen_pixels == MIN_SPECIMEN_PIXELS:
        _save_cache(key, data)
    return data


def to_grid(X: np.ndarray, data: MEData) -> tuple[np.ndarray, np.ndarray]:
    """Scatter [n, F] features onto the [n, E, M] grid (zeros where unmeasured). Returns (grid, mask [E, M])."""
    E, M = len(data.excitations), len(data.emissions)
    grid = np.zeros((X.shape[0], E, M), dtype=np.float32)
    gi = data.grid_index
    grid[:, gi[:, 0], gi[:, 1]] = X
    mask = np.zeros((E, M), dtype=bool)
    mask[gi[:, 0], gi[:, 1]] = True
    return grid, mask


def summary(data: MEData) -> dict[str, Any]:
    return dict(
        name=data.name, n_pixels=int(len(data.y)), n_classes=data.n_classes,
        specimens_per_class=np.bincount(data.specimen_classes, minlength=data.n_classes).tolist(),
        pixels_per_class=np.bincount(data.y, minlength=data.n_classes).tolist(),
        dropped_px=data.info.get("dropped_px"), dropped_components=data.info.get("dropped_components"),
        E=len(data.excitations), M=len(data.emissions), F=data.n_features,
        ex_nm=[float(data.excitations.min()), float(data.excitations.max())],
        em_nm=[float(data.emissions.min()), float(data.emissions.max())],
        image_shape=list(data.image_shape))
