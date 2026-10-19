"""Evaluation protocol (docs/me_attention/SPEC.md): outer folds, inner split, patches, metrics, statistics."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .types import MEData


# ----------------------------------------------------------------------------- outer folds
def n_outer_folds(data: MEData) -> int:
    return int(np.bincount(data.specimen_classes, minlength=data.n_classes).min())


def specimen_folds(data: MEData, repeat: int) -> list[list[int]]:
    """Class-covering deal of specimens into k folds (round-robin after a seeded per-class shuffle)."""
    k = n_outer_folds(data)
    rng = np.random.default_rng(int(repeat))
    folds: list[list[int]] = [[] for _ in range(k)]
    for c in range(data.n_classes):
        sids = np.flatnonzero(data.specimen_classes == c)
        rng.shuffle(sids)
        for i, s in enumerate(sids):
            folds[i % k].append(int(s))
    for f in folds:
        assert set(data.specimen_classes[f].tolist()) == set(range(data.n_classes)), "fold misses a class"
    return folds


def outer_folds(data: MEData, repeat: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for f in specimen_folds(data, repeat):
        te_mask = np.isin(data.groups, f)
        tr, te = np.flatnonzero(~te_mask), np.flatnonzero(te_mask)
        assert set(np.unique(data.y[te]).tolist()) == set(range(data.n_classes))
        assert set(data.groups[tr]).isdisjoint(set(data.groups[te]))
        out.append((tr, te))
    return out


# ----------------------------------------------------------------------------- inner split
def inner_split(data: MEData, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out one whole specimen per class (if >= 2 in train); otherwise a contiguous ~25% block of the
    single specimen along its longer axis. Returns (inner_train, inner_val) as pixel indices."""
    rng = np.random.default_rng(int(seed))
    train_idx = np.asarray(train_idx)
    g = data.groups[train_idx]
    val_mask = np.zeros(len(train_idx), dtype=bool)
    for c in range(data.n_classes):
        sids = np.unique(g[data.y[train_idx] == c])
        if len(sids) == 0:
            continue
        if len(sids) >= 2:
            val_mask |= g == int(rng.choice(sids))
        else:
            sel = np.flatnonzero(g == sids[0])
            xy = data.coords[train_idx[sel]].astype(float)
            axis = int(np.argmax(xy.max(0) - xy.min(0)))
            order = np.argsort(xy[:, axis], kind="stable")
            n_val = max(1, int(round(0.25 * len(sel))))
            chosen = order[-n_val:] if rng.random() < 0.5 else order[:n_val]
            val_mask[sel[chosen]] = True
    return train_idx[~val_mask], train_idx[val_mask]


# ----------------------------------------------------------------------------- patches
_PATCH_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, int]] = {}


def _cache_key(data: MEData) -> tuple:
    return (data.name, tuple(data.X.shape), id(data.X), tuple(data.image_shape))


def _padded_cube(data: MEData, pad: int):
    key = _cache_key(data)
    hit = _PATCH_CACHE.get(key)
    if hit is not None and hit[2] >= pad and hit[0].shape[-1] == data.n_features:
        return hit
    while len(_PATCH_CACHE) >= 2 and key not in _PATCH_CACHE:  # bound memory: padded cubes are large
        _PATCH_CACHE.pop(next(iter(_PATCH_CACHE)))
    H, W = data.image_shape
    cube = np.zeros((H + 2 * pad, W + 2 * pad, data.n_features), dtype=np.float32)
    spec = np.full((H + 2 * pad, W + 2 * pad), -1, dtype=np.int64)
    r, c = data.coords[:, 0] + pad, data.coords[:, 1] + pad
    cube[r, c] = data.X
    spec[r, c] = data.groups
    _PATCH_CACHE[key] = (cube, spec, pad)
    return _PATCH_CACHE[key]


def clear_patch_cache() -> None:
    _PATCH_CACHE.clear()


def make_patches(data: MEData, idx: np.ndarray, P: int, chunk: int = 4096) -> np.ndarray:
    """[n, P, P, F] float32 patches centred on pixels ``idx``; pixels outside the centre pixel's specimen are zero."""
    if P % 2 != 1:
        raise ValueError("P must be odd")
    h = P // 2
    cube, spec, pad = _padded_cube(data, h)
    off = np.arange(-h, h + 1)
    idx = np.asarray(idx)
    out = np.empty((len(idx), P, P, data.n_features), dtype=np.float32)
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        rr = (data.coords[sl, 0] + pad)[:, None, None] + off[None, :, None]
        cc = (data.coords[sl, 1] + pad)[:, None, None] + off[None, None, :]
        rr, cc = np.broadcast_arrays(rr, cc)
        patch = cube[rr, cc]
        same = spec[rr, cc] == data.groups[sl][:, None, None]
        out[s:s + chunk] = patch * same[..., None]
    return out


# ----------------------------------------------------------------------------- metrics
def metrics(y_true, proba_or_pred, groups, n_classes: int, whole_specimen: bool = True) -> dict:
    y_true = np.asarray(y_true)
    p = np.asarray(proba_or_pred)
    pred = p.argmax(1) if p.ndim == 2 else p.astype(np.int64)
    groups = np.asarray(groups)
    labels = np.arange(n_classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, pred), 1)
    support = cm.sum(1)
    recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    pred_n = cm.sum(0)
    prec = np.where(pred_n > 0, np.diag(cm) / np.maximum(pred_n, 1), 0.0)
    rec0 = np.nan_to_num(recall, nan=0.0)
    f1 = np.where(prec + rec0 > 0, 2 * prec * rec0 / np.maximum(prec + rec0, 1e-12), 0.0)
    present = support > 0
    sids = np.unique(groups)
    per_spec = [float((pred[groups == s] == y_true[groups == s]).mean()) for s in sids]
    out = dict(
        balanced_acc=float(np.nanmean(recall)),
        macro_f1=float(f1.mean()),  # fixed labels 0..C-1, zero_division=0 (== sklearn)
        pixel_acc=float((pred == y_true).mean()),
        specimen_avg_acc=float(np.mean(per_spec)),
        per_class_recall=[None if np.isnan(r) else float(r) for r in recall],
        confusion=cm.tolist(),
        n=int(len(y_true)), n_specimens=int(len(sids)), labels=labels.tolist(),
    )
    if whole_specimen:
        ok = 0
        for s in sids:
            m = groups == s
            ok += int(np.bincount(pred[m], minlength=n_classes).argmax() == np.bincount(y_true[m]).argmax())
        out["specimen_vote_acc"] = ok / len(sids)
    else:
        out["specimen_vote_acc"] = None
    return out


# ----------------------------------------------------------------------------- statistics
def stats_paired(a, b, test_train_ratio: float, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of a vs b over (repeat, fold) units."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    if n < 2 or np.allclose(d, 0):
        p_w = 1.0
    else:
        p_w = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    if n < 2 or not sd > 0:
        t, p_nb = float("nan"), (1.0 if mean == 0 else 0.0)
    else:
        t = mean / np.sqrt((1.0 / n + test_train_ratio) * sd ** 2)
        p_nb = float(2 * stats.t.sf(abs(t), df=n - 1))
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=n, mean_diff=mean, std_diff=sd, wilcoxon_p=p_w, nb_t=float(t), nb_p=p_nb,
                ci95=[float(lo), float(hi)], wins=int((d > 0).sum()), ties=int((d == 0).sum()),
                losses=int((d < 0).sum()))


def holm(pvals) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()
