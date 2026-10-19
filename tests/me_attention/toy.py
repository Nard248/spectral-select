"""Synthetic helper: 3 classes x 3 specimens x 300 px, F=40 (5 fake excitations)."""
import numpy as np

EX = np.repeat([300.0, 350, 400, 450, 500], 8)
EM = np.tile(np.arange(8) * 10.0 + 400, 5)


def meta(F=40, C=3):
    return dict(ex_nm=EX, em_nm=EM, n_features=F, n_classes=C)


def toy(n=300, seed=0, sep=1.0):
    rng = np.random.default_rng(seed)
    protos = rng.normal(0, 1, (3, 40))
    X, y, g = [], [], []
    for c in range(3):
        for s in range(3):
            shift = rng.normal(0, 0.3, 40)
            amp = np.exp(rng.normal(0, 0.2))
            x = 10 + sep * 2 * protos[c] + shift + rng.normal(0, 1.0, (n, 40))
            X.append(np.abs(x) * amp); y += [c] * n; g += [c * 3 + s] * n
    return np.concatenate(X).astype(np.float32), np.array(y), np.array(g)
