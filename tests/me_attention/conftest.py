"""Synthetic fixtures for the ME-attention tests (no real data)."""
import json
import pickle

import numpy as np
import pytest
from PIL import Image

from me_attention.types import MEData

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# per-excitation wavelength grids: regular / with a gap / 5-nm offset
GRIDS = {310.0: [420, 430, 440, 450], 340.0: [430, 450, 470], 400.0: [425, 435, 445]}


def write_dataset(root, key_name="wavelengths", with_wl=True, H=12, W=30):
    """3 classes x 3 specimens (4x4 blocks => 16 px) + a 3-px artifact; cube value = ex*1000 + wavelength."""
    d = root / "Collagen_Acetic_Acid"
    d.mkdir(parents=True)
    mask = np.zeros((H, W, 3), np.uint8)
    for c in range(3):
        for s in range(3):
            r0, c0 = 4 * c, 1 + s * 9
            mask[r0:r0 + 4, c0:c0 + 4] = COLORS[c]
    mask[11, 29] = COLORS[0]  # tiny artifact component (1 px)
    Image.fromarray(mask).save(d / "class_mask.png")
    (d / "roi_regions.json").write_text(json.dumps(
        {"classes": [{"id": i, "name": f"C{i}", "color": list(COLORS[i])} for i in range(3)]}))
    data = {}
    for ex, wl in GRIDS.items():
        cube = np.zeros((H, W, len(wl)))
        cube[:] = ex * 1000 + np.asarray(wl, float)
        entry = {"cube": cube}
        if with_wl:
            entry[key_name] = np.asarray(wl, float)
        data[str(float(ex))] = entry
    fg = mask.any(2)
    with open(d / "spectra_masked.pkl", "wb") as f:
        pickle.dump({"data": data, "excitation_wavelengths": list(GRIDS), "mask": fg}, f)
    return d


def make_me(counts, px=6, seed=0, F=4):
    """MEData with `counts[c]` specimens for class c, each a px-pixel horizontal strip (rows keep them disjoint)."""
    rng = np.random.default_rng(seed)
    classes = [c for c, n in enumerate(counts) for _ in range(n)]
    S = len(classes)
    H, W = 2 * S, px + 2
    smap = np.full((H, W), -1, np.int64)
    coords, groups, y = [], [], []
    for s, c in enumerate(classes):
        for j in range(px):
            smap[2 * s, 1 + j] = s
            coords.append((2 * s, 1 + j))
            groups.append(s)
            y.append(c)
    n = len(y)
    X = rng.normal(size=(n, F)).astype(np.float32)
    ex = np.repeat([1.0, 2.0], F // 2)
    em = np.tile([10.0, 20.0], F // 2)
    return MEData(
        name=f"syn_{'_'.join(map(str, counts))}", X=X, ex_nm=ex, em_nm=em, y=np.array(y, np.int64),
        groups=np.array(groups, np.int64), coords=np.array(coords, np.int64),
        specimen_classes=np.array(classes, np.int64), class_names=[f"c{i}" for i in range(len(counts))],
        image_shape=(H, W), excitations=np.array([1.0, 2.0]), emissions=np.array([10.0, 20.0]),
        grid_index=np.array([(0, 0), (0, 1), (1, 0), (1, 1)]), specimen_map=smap)


@pytest.fixture
def syn_dir(tmp_path):
    return write_dataset(tmp_path).parent
