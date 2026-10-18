"""Metrics and paired statistics (copied from the ME-attention protocol)."""
from __future__ import annotations

import numpy as np
from scipy import stats


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
