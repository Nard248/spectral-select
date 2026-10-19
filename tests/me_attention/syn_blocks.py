"""2-D synthetic MEData: 3 classes x 3 specimens as 6x6 blocks with class-dependent spectra (patch tests)."""
import numpy as np

from me_attention.types import MEData


def make_blocks(F=8, side=6, seed=0, counts=(3, 3, 3)) -> MEData:
    rng = np.random.default_rng(seed)
    classes = [c for c, n in enumerate(counts) for _ in range(n)]
    S = len(classes)
    H, W = 2 * side + 2, S * (side + 2) + 2
    smap = np.full((H, W), -1, np.int64)
    coords, groups, y, X = [], [], [], []
    protos = rng.normal(size=(len(counts), F))
    for s, c in enumerate(classes):
        r0, c0 = 2, 1 + s * (side + 2)
        for i in range(side):
            for j in range(side):
                smap[r0 + i, c0 + j] = s
                coords.append((r0 + i, c0 + j)); groups.append(s); y.append(c)
                X.append(10 + 3 * protos[c] + rng.normal(0, 1, F))
    X = np.abs(np.array(X)).astype(np.float32) + 1.0
    ex = np.repeat([1.0, 2.0], F // 2)
    em = np.tile(np.arange(F // 2) * 10.0 + 10, 2)
    gi = np.stack([np.repeat([0, 1], F // 2), np.tile(np.arange(F // 2), 2)], 1)
    return MEData(
        name=f"blocks_{F}_{seed}", X=X, ex_nm=ex, em_nm=em, y=np.array(y, np.int64), groups=np.array(groups, np.int64),
        coords=np.array(coords, np.int64), specimen_classes=np.array(classes, np.int64),
        class_names=[f"c{i}" for i in range(len(counts))], image_shape=(H, W), excitations=np.array([1.0, 2.0]),
        emissions=em[:F // 2].copy(), grid_index=gi, specimen_map=smap)
