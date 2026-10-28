"""Outcome maps: what does a selection of n excitation-emission pairs actually *produce*?

For one dataset, one n and one repeat, build a (selection method) x (downstream algorithm) grid of
per-pixel maps:

  columns : Ground truth | kNN, LDA, SVM, RF (specimen-held-out CV, exactly the paper's protocol)
            | k-means, GMM (label-free, fitted on all annotated pixels) | NMF (linear unmixing)
  rows    : All pairs | RL add/remove | RL add-only | CSS greedy | Random | SA (supervised search)

The supervised maps are assembled from the outer folds: every annotated pixel is predicted by a model that
never saw its specimen, using the very same `_cap_idx` + `_make_clf` as `rl_select.evaluate.evaluate`.
Every (method, fold, classifier) balanced accuracy is recomputed from the predictions and checked against the
stored JSONL record (see <out>/maps_<key>_n<n>_validation.txt).

  PYTHONPATH=src .venv/bin/python experiments/rl_select/outcome_maps.py \
      --dataset collagen_pepsin --n 8 --repeat 0 --out publications/rl_band_selection/figures/proto/

`--render-only` re-draws the figure from the saved maps_<key>_n<n>.npz without recomputing anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "rl_select"

SUP = ("knn", "lda", "svm", "rf")
UNSUP = ("kmeans", "gmm")
ALGOS = SUP + UNSUP + ("nmf",)
ALGO_HDR = {"knn": "kNN", "lda": "LDA", "svm": "SVM", "rf": "RF", "kmeans": "k-means", "gmm": "GMM", "nmf": "NMF"}
# (key, display label, row-label for the figure)
METHODS = (
    ("all", "All pairs", "All pairs\n(reference)"),
    ("rl_addremove", "RL add/remove", "RL add/remove\n(transfer)"),
    ("rl_add", "RL add-only", "RL add-only\n(transfer)"),
    ("css", "CSS (greedy, label-free)", "CSS greedy\n(label-free)"),
    ("random", "Random", "Random"),
    ("sa", "SA (supervised search)", "SA\n(supervised)"),
)
CROSS = {"lichens1": "C2L", "lichens2": "C2L", "collagen_acetic": "L2C", "collagen_pepsin": "L2C"}


def cross_tag(key: str) -> str:
    """Transfer-source tag of a dataset key (new keys such as lichens2_sub3 follow their family)."""
    return CROSS.get(key, "C2L" if key.startswith("lichens") else "L2C")


def set_results_root(root) -> Path:
    """Point this module (and rl_select.data's cache dir, if <root>/cache exists) at another results root."""
    global RES
    root = Path(root)
    RES = root if root.is_absolute() else ROOT / root
    if (RES / "cache").exists():
        import rl_select.data as _rd
        _rd.CACHE_DIR = RES / "cache"
    return RES


def class_names_for(key: str, meta_names=None) -> list[str]:
    return CLASS_NAMES.get(key) or [f"Class {i + 1}" for i in range(len(meta_names or []))]
PAL = np.array([(230, 25, 75), (60, 180, 75), (0, 130, 200), (255, 225, 25), (240, 50, 230)], dtype=np.float32) / 255.0
CLASS_NAMES = {
    "lichens1": ["R. sinensis (lower)", "R. sinensis (upper)", "F. soredica", "R. fraxinea"],
    "collagen_pepsin": ["GA level 1", "GA level 2", "GA level 3"],
    "collagen_acetic": ["GA level 1", "GA level 2", "GA level 3"],
    "lichens2": ["Class A", "Class B", "Class C", "Class D", "Class E"],
}
# lichens1 cache class names must be Class 1,3,6,7 (= R.s. lower, R.s. upper, F. soredica, R. fraxinea)
EXPECT_META = {"lichens1": ["Class 1", "Class 3", "Class 6", "Class 7"]}
TOL = 1e-9


# ============================================================================ selections
def _read_jsonl(path: Path) -> list[dict]:
    rows, seen = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen[(r["method"], r["task"], r["n"], r["seed"], r["budget"])] = r  # last write wins (paper's dedupe)
    rows = list(seen.values())
    return rows


def load_selections(key: str, n: int, repeat: int, nf: int) -> dict[str, dict[int, dict]]:
    """{method_key: {fold: record}} for every non-reference method (records carry selected/seed/eval)."""
    src = cross_tag(key)
    s2 = _read_jsonl(RES / "S2" / f"{key}.jsonl")
    s2rl = _read_jsonl(RES / "S2" / f"{key}_rl.jsonl")
    s1 = _read_jsonl(RES / "S1" / f"{key}.jsonl")

    def pick(rows, method, seed0=True, first=False):
        out: dict[int, dict] = {}
        for r in rows:
            if r["method"] != method or r["n"] != n or r["repeat"] != repeat:
                continue
            if seed0 and r["seed"] != 0:
                continue
            if r["fold"] not in out:  # first record per fold in file order
                out[r["fold"]] = r
        return out

    sel = {
        "rl_addremove": pick(s2rl, f"rl_addremove_nowl_k16@{src}"),
        "rl_add": pick(s2rl, f"rl_add_nowl_k16@{src}"),
        "css": pick(s2, "css_greedy"),
        "random": pick(s2, "random", seed0=False),
        "sa": pick(s1, "sa"),
    }
    for m, d in sel.items():
        miss = sorted(set(range(nf)) - set(d))
        if miss:
            raise RuntimeError(f"{key}: method {m} has no record for folds {miss} (n={n}, repeat={repeat})")
    allp = {}
    for r in s2:
        if r["method"] == "all_pairs" and r["repeat"] == repeat:
            allp[r["fold"]] = r
    sel["all_records"] = allp  # stored all-pairs reference per fold (for validation only)
    return sel


# ============================================================================ supervised
def fit_predict(task, S, clf: str, seed: int):
    from rl_select.evaluate import CAPS, _cap_idx, _make_clf
    idx = _cap_idx(task, CAPS[clf], seed)
    model = _make_clf(clf, seed).fit(task.Xtr[idx][:, S], task.ytr[idx])
    return model.predict(task.Xte[:, S])


def matched_labels(y, lab, K):
    """Hungarian match of cluster ids to classes. Returns (cluster->class mapping, relabelled array)."""
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, lab), 1)
    r, c = linear_sum_assignment(-cm)  # rows = classes, cols = clusters
    mp = np.zeros(K, dtype=np.int64)
    mp[c] = r
    return mp, mp[lab]


def run_supervised(key, repeat, nf, sel, n, val_rows, log):
    from rl_select.data import load_dataset, make_task, specimen_folds
    from rl_select.evaluate import evaluate
    from rl_select.stats import metrics

    d = load_dataset(key)
    H, W = d["specimen_map"].shape
    K = d["n_classes"]
    folds = specimen_folds(d, repeat)
    maps = {(m, c): np.full((H, W), -1, dtype=np.int8) for m, _, _ in METHODS for c in SUP}
    fold_ba = {(m, c): [None] * nf for m, _, _ in METHODS for c in SUP}
    chosen = {}  # (m, f) -> selected set
    for f in range(nf):
        task = make_task(key, repeat, f)
        te = np.flatnonzero(np.isin(d["groups"], folds[f]))
        assert np.array_equal(d["y"][te], task.yte), "held-out pixel order mismatch"
        rc = d["coords"][te]
        for m, label, _ in METHODS:
            if m == "all":
                S = np.flatnonzero(task.valid)
                rec = sel["all_records"].get(f)
            else:
                rec = sel[m][f]
                S = np.asarray(rec["selected"], dtype=np.int64)
                assert len(S) == n, (m, f, len(S))
            chosen[(m, f)] = S
            t0 = time.time()
            for clf in SUP:
                stored = None if rec is None else rec["eval"].get(clf)
                seeds = [0] if rec is None else list(dict.fromkeys([int(rec["seed"]), 0]))
                best = None
                for sd in seeds:
                    pred = fit_predict(task, S, clf, sd)
                    ba = metrics(task.yte, pred, task.gte, K)["balanced_acc"]
                    diff = abs(ba - stored["balanced_acc"]) if stored else None
                    cand = (diff if diff is not None else 0.0, sd, pred, ba)
                    if best is None or cand[0] < best[0]:
                        best = cand
                    if diff is None or diff < TOL:
                        break
                _, sd, pred, ba = best
                maps[(m, clf)][rc[:, 0], rc[:, 1]] = pred
                fold_ba[(m, clf)][f] = ba
                if stored is not None:
                    val_rows.append(dict(method=label, fold=f, clf=clf, ref="stored record", ref_ba=stored["balanced_acc"],
                                         my_ba=ba, diff=abs(ba - stored["balanced_acc"]), seed=sd,
                                         tried=",".join(map(str, seeds))))
                if stored is None or m == "all":
                    # independent path: evaluate() itself (no stored value for this classifier, or all-pairs spec)
                    ev = evaluate(task, S, [clf], seed=0)[clf]["balanced_acc"]
                    val_rows.append(dict(method=label, fold=f, clf=clf, ref="evaluate()", ref_ba=ev, my_ba=ba,
                                         diff=abs(ba - ev), seed=sd, tried=",".join(map(str, seeds))))
            log(f"  supervised fold {f} {label}: {time.time() - t0:.1f}s")
    return maps, fold_ba, chosen


def check_all_pairs_csv(key, nf, val_rows, log, n_repeats_csv=None):
    """Compare the mean all-pairs BA over all repeats with summary/all_pairs_reference.csv (if present)."""
    from rl_select.data import make_task
    from rl_select.evaluate import evaluate
    path = RES / "summary" / "all_pairs_reference.csv"
    if not path.exists():
        return None
    row = None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] == key and r["method"] == "all_pairs":
                row = r
    if row is None:
        return None
    n_tasks = int(row["n_tasks"])
    reps = n_tasks // nf
    vals = {c: [] for c in SUP}
    for rep in range(reps):
        for f in range(nf):
            task = make_task(key, rep, f)
            ev = evaluate(task, np.flatnonzero(task.valid), SUP, seed=0)
            for c in SUP:
                vals[c].append(ev[c]["balanced_acc"])
        log(f"  csv-check: repeat {rep} evaluated")
    out = {}
    for c in SUP:
        out[c] = (float(np.mean(vals[c])), float(row[f"{c}_ba_mean"]))
    return reps, out


# ============================================================================ unsupervised / unmixing
def run_unsup(key, repeat, nf, chosen, n, log):
    from rl_select.data import load_dataset, make_task, specimen_folds
    from rl_select.stats import metrics
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.mixture import GaussianMixture

    d = load_dataset(key)
    H, W = d["specimen_map"].shape
    K = d["n_classes"]
    folds = specimen_folds(d, repeat)
    t0 = make_task(key, repeat, 0)
    te_mask = np.isin(d["groups"], folds[0])
    order = np.concatenate([np.flatnonzero(~te_mask), np.flatnonzero(te_mask)])
    Xall = np.vstack([t0.Xtr, t0.Xte])  # fold-0 log1p + standardise
    y = np.concatenate([t0.ytr, t0.yte])
    g = np.concatenate([t0.gtr, t0.gte])
    assert np.array_equal(d["y"][order], y) and np.array_equal(d["groups"][order], g)
    rc = d["coords"][order]
    Xraw = np.clip(d["X"][order].astype(np.float64), 0, None)

    maps = {(m, a): np.full((H, W), -1, dtype=np.int8) for m, _, _ in METHODS for a in UNSUP + ("nmf",)}
    nmf_ab = {m: np.zeros((H, W, K), dtype=np.float16) for m, _, _ in METHODS}
    scores = {}  # (m, a) -> dict(ARI, NMI, BA, extra)
    for m, label, _ in METHODS:
        S = chosen[(m, 0)]
        Z = Xall[:, S]
        for a in UNSUP:
            ts = time.time()
            extra = {}
            if a == "kmeans":
                lab = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Z)
            else:
                reg = 1e-6
                for attempt in range(4):
                    try:
                        with warnings.catch_warnings(record=True) as wl:
                            warnings.simplefilter("always")
                            gm = GaussianMixture(n_components=K, covariance_type="full", random_state=0, reg_covar=reg).fit(Z)
                        break
                    except ValueError:
                        reg *= 100
                else:
                    raise
                lab = gm.predict(Z)
                extra = dict(converged=bool(gm.converged_), n_iter=int(gm.n_iter_), reg_covar=reg)
            mp, ml = matched_labels(y, lab, K)
            maps[(m, a)][rc[:, 0], rc[:, 1]] = ml
            scores[(m, a)] = dict(ARI=float(adjusted_rand_score(y, lab)), NMI=float(normalized_mutual_info_score(y, lab)),
                                  BA=float(metrics(y, ml, g, K)["balanced_acc"]), **extra)
            log(f"  {a} {label}: {time.time() - ts:.1f}s ARI {scores[(m, a)]['ARI']:.3f}")
        # NMF on raw non-negative selected features
        ts = time.time()
        A = Xraw[:, S].copy()
        p99 = np.percentile(A, 99, axis=0)
        p99[p99 <= 0] = 1.0
        A /= p99
        nmf = NMF(n_components=K, init="nndsvda", max_iter=1000, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Wm = nmf.fit_transform(A)
        Hm = nmf.components_
        hn = np.linalg.norm(Hm, axis=1)
        hn[hn == 0] = 1.0
        Wm = Wm * hn[None, :]  # move component scale into the abundances so components are comparable
        tot = Wm.sum(1, keepdims=True)
        zero = (tot[:, 0] <= 1e-12)
        Wn = np.where(zero[:, None], 1.0 / K, Wm / np.maximum(tot, 1e-12))
        M = np.stack([Wn[y == c].mean(0) for c in range(K)])  # class x component
        r, c = linear_sum_assignment(-M)
        comp2class = np.zeros(K, dtype=np.int64)
        comp2class[c] = r
        Ab = np.zeros_like(Wn)
        for comp in range(K):
            Ab[:, comp2class[comp]] = Wn[:, comp]
        pred = Ab.argmax(1)
        maps[(m, "nmf")][rc[:, 0], rc[:, 1]] = pred
        nmf_ab[m][rc[:, 0], rc[:, 1]] = Ab.astype(np.float16)
        scores[(m, "nmf")] = dict(ARI=float(adjusted_rand_score(y, pred)), NMI=float(normalized_mutual_info_score(y, pred)),
                                  BA=float(metrics(y, pred, g, K)["balanced_acc"]), n_iter=int(nmf.n_iter_),
                                  n_zero_abundance_px=int(zero.sum()))
        log(f"  nmf {label}: {time.time() - ts:.1f}s BA {scores[(m, 'nmf')]['BA']:.3f} (iters {nmf.n_iter_})")
    return maps, nmf_ab, scores


# ============================================================================ outputs
def loc_word(rf, cf):
    v = "top" if rf < 1 / 3 else ("middle" if rf < 2 / 3 else "bottom")
    h = "left" if cf < 1 / 3 else ("centre" if cf < 2 / 3 else "right")
    return f"{v}-{h}"


def specimen_table(key, repeat, labels, gt_map, crop):
    """Per-specimen accuracy (fraction of pixels with the right class) for every (method, algorithm) panel."""
    from rl_select.data import load_dataset
    d = load_dataset(key)
    g, y, rc = d["groups"], d["y"], d["coords"]
    r0, r1, c0, c1 = crop
    rows = []
    for s in np.unique(g):
        m = g == s
        cen = rc[m].mean(0)
        rows.append(dict(specimen=int(s), true_class=int(y[m][0]), n_px=int(m.sum()), centroid_row=float(cen[0]),
                         centroid_col=float(cen[1]), location=loc_word((cen[0] - r0) / (r1 - r0), (cen[1] - c0) / (c1 - c0))))
    names = []
    for (mk, lab, _), L in zip(METHODS, labels):
        for ai, a in enumerate(ALGOS):
            names.append((mk, a))
            pr = L[ai][rc[:, 0], rc[:, 1]]
            for row, s in zip(rows, np.unique(g)):
                msk = g == s
                row[f"acc|{mk}|{a}"] = float((pr[msk] == y[msk]).mean())
                row[f"maj|{mk}|{a}"] = int(np.bincount(pr[msk], minlength=d["n_classes"]).argmax())
    return rows, names


def save_outputs(key, n, gt_map, crop, labels, nmf_ab, scores_rows, meta, specimen_map, out):
    H, W = gt_map.shape
    np.savez_compressed(
        out / f"maps_{key}_n{n}.npz",
        gt=gt_map, labels=np.stack([np.stack([L[a] for a in range(len(ALGOS))]) for L in labels]).astype(np.int8),
        nmf_abund=np.stack([nmf_ab[m] for m, _, _ in METHODS]), crop=np.asarray(crop),
        method_keys=np.array([m for m, _, _ in METHODS]), method_labels=np.array([l for _, l, _ in METHODS]),
        row_labels=np.array([r for _, _, r in METHODS]), algo_keys=np.array(ALGOS),
        class_names=np.array(class_names_for(key, list(range(int(gt_map.max()) + 1)))), scores_json=np.array(json.dumps(scores_rows)),
        meta_json=np.array(json.dumps(meta)), specimen_map=specimen_map,
    )


# ============================================================================ rendering
def load_means(csv_path, key: str, n: int) -> dict:
    """{(method_key, algo_key): mean score over all stored repeats} from a score_vs_n summary CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; run experiments/rl_select/score_vs_n.py first "
                                f"(--label-mean-over-repeats reads the repeat means from it)")
    out = {}
    with open(csv_path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("dataset,"):
                continue
            r = line.rstrip("\n").split(",")
            if r[0] == key and int(r[1]) == n:
                out[(r[2], r[3])] = float(r[4])
    return out


def render(npz_path: Path, out: Path, key: str, n: int, score_loc: str = "auto", algos=None, suffix: str = "",
           means=None, score_fs=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image

    z = np.load(npz_path)  # only plain numeric / unicode arrays, no pickles
    gt, labels, nmf_ab = z["gt"], z["labels"], z["nmf_abund"]
    r0, r1, c0, c1 = [int(v) for v in z["crop"]]
    row_labels = [str(s) for s in z["row_labels"]]
    mkeys = [str(s) for s in z["method_keys"]]
    names = [str(s) for s in z["class_names"]]
    scores = json.loads(str(z["scores_json"]))
    K = len(names)
    sc = {(r["method_key"], r["algorithm_key"]): r for r in scores}
    all_algos = [str(a) for a in z["algo_keys"]]
    algos = [a for a in (algos or all_algos) if a in all_algos]  # keep the canonical column order
    algos = [a for a in all_algos if a in algos]

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    nrows, ncols = len(mkeys), 1 + len(algos)
    score_fs = score_fs or (7.0 if ncols <= 6 else 5.6)
    h, w = r1 - r0, c1 - c0
    FW, L, R, GAP = 7.16, 0.80, 0.03, 0.02
    pw = (FW - L - R - (ncols - 1) * GAP) / ncols
    ph = pw * h / w
    HDR_G, HDR_C, PADT, LEG = 0.12, 0.13, 0.03, 0.22
    top = HDR_G + HDR_C + PADT
    FH = top + nrows * ph + (nrows - 1) * GAP + LEG
    fig = plt.figure(figsize=(FW, FH))
    dpi = 300
    tw, th = int(round(pw * dpi)), int(round(ph * dpi))

    def to_img(rgb):
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        if im.size[0] > tw:
            im = im.resize((tw, th), Image.BOX)
        return np.asarray(im)

    def lab_rgb(lab):
        sub = lab[r0:r1, c0:c1]
        rgb = np.ones(sub.shape + (3,), dtype=np.float32)
        for k in range(K):
            rgb[sub == k] = PAL[k]
        return rgb

    def ab_rgb(ab, mask):
        rgb = ab[r0:r1, c0:c1].astype(np.float32) @ PAL[:K]
        rgb[~mask[r0:r1, c0:c1]] = 1.0
        return rgb

    ann = gt >= 0
    gt_img = to_img(lab_rgb(gt))
    if score_loc == "auto":  # corner (28% x 13% of the crop) with the fewest annotated pixels
        sub = ann[r0:r1, c0:c1]
        hh, ww = max(int(0.13 * h), 1), max(int(0.28 * w), 1)
        cnt = {"tl": sub[:hh, :ww].sum(), "tr": sub[:hh, -ww:].sum(), "bl": sub[-hh:, :ww].sum(), "br": sub[-hh:, -ww:].sum()}
        score_loc = min(cnt, key=lambda k: (cnt[k], k))

    def place(ri, ci, img, text=None):
        x = (L + ci * (pw + GAP)) / FW
        y = (FH - top - (ri + 1) * ph - ri * GAP) / FH
        ax = fig.add_axes([x, y, pw / FW, ph / FH])
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3); sp.set_color("#b0b0b0")
        if text:
            ha, va, tx, ty = {"bl": ("left", "bottom", 0.04, 0.03), "br": ("right", "bottom", 0.96, 0.03),
                              "tl": ("left", "top", 0.04, 0.97), "tr": ("right", "top", 0.96, 0.97)}[score_loc]
            ax.text(tx, ty, text, transform=ax.transAxes, ha=ha, va=va, fontsize=score_fs, color="black",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.6))
        return ax

    for ri, mk in enumerate(mkeys):
        fig.text((L - 0.05) / FW, (FH - top - ri * (ph + GAP) - ph / 2) / FH, row_labels[ri], ha="right", va="center",
                 fontsize=6.3, linespacing=1.05)
        place(ri, 0, gt_img)
        for ci, a in enumerate(algos):
            ai = all_algos.index(a)
            if a == "nmf":
                img = to_img(ab_rgb(nmf_ab[ri], ann))
            else:
                img = to_img(lab_rgb(labels[ri, ai]))
            s = sc[(mk, a)]
            if a in SUP:
                v0, nm = s['value'], "BA"
            elif a in UNSUP:
                v0, nm = s['ARI'], "ARI"
            else:
                v0, nm = s['matched_balanced_acc'], "BA"
            if means is not None and (mk, a) in means:
                txt = f"{nm} {means[(mk, a)]:.2f} ({v0:.2f})"
            else:
                txt = f"{nm} {v0:.2f}"
            place(ri, 1 + ci, img, txt)

    # headers
    hdr = ["Ground truth"] + [ALGO_HDR[a] for a in algos]
    for ci, t in enumerate(hdr):
        fig.text((L + ci * (pw + GAP) + pw / 2) / FW, (FH - HDR_G - PADT / 2 - HDR_C / 2) / FH, t, ha="center",
                 va="center", fontsize=6.6, fontweight="bold")
    groups = []
    for t, members in (("Supervised, specimen-held-out CV", SUP), ("Unsupervised", UNSUP), ("Unmixing", ("nmf",))):
        cols = [1 + i for i, a in enumerate(algos) if a in members]
        if cols:
            groups.append((t if (len(cols) > 1 or t != "Supervised, specimen-held-out CV") else "Supervised", cols[0], cols[-1]))
    for t, a, b in groups:
        x0 = (L + a * (pw + GAP)) / FW
        x1 = (L + b * (pw + GAP) + pw) / FW
        yline = (FH - HDR_G + 0.01) / FH
        fig.add_artist(plt.Line2D([x0, x1], [yline, yline], color="#555555", lw=0.5, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, (FH - HDR_G / 2 + 0.01) / FH, t, ha="center", va="center", fontsize=5.8, color="#333333")

    handles = [Patch(fc=PAL[k], ec="none", label=names[k]) for k in range(K)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=K, frameon=False, fontsize=6.4,
               handlelength=1.1, handleheight=0.9, columnspacing=1.4, handletextpad=0.4, borderpad=0.2)
    fig.savefig(out / f"maps_{key}_n{n}{suffix}.pdf", dpi=dpi)
    fig.savefig(out / f"maps_{key}_n{n}{suffix}.png", dpi=dpi)
    rep0 = json.loads(str(z["meta_json"])).get("repeat", 0)
    cap = (f"Maps show repeat {rep0}" + (", fold-0 selection for the clustering/unmixing columns" if any(a in UNSUP or a == "nmf" for a in algos) else "")
           + (f". Panel labels: mean over all stored repeats, with the repeat-{rep0} value of the map in parentheses."
              if means is not None else f". Panel labels: repeat-{rep0} value of the map."))
    (out / f"maps_{key}_n{n}{suffix}_caption.txt").write_text(cap + "\n")
    plt.close(fig)
    return FW, FH


def render_specimen_ids(key, out, crop, gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_select.data import load_dataset
    d = load_dataset(key)
    r0, r1, c0, c1 = crop
    K = d["n_classes"]
    rgb = np.ones(gt.shape + (3,), dtype=np.float32)
    for k in range(K):
        rgb[gt == k] = PAL[k]
    fig, ax = plt.subplots(figsize=(7.16, 7.16 * (r1 - r0) / (c1 - c0)))
    ax.imshow(rgb[r0:r1, c0:c1], interpolation="nearest")
    for s in np.unique(d["groups"]):
        m = d["groups"] == s
        cy, cx = d["coords"][m].mean(0)
        ax.text(cx - c0, cy - r0, str(int(s)), ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out / f"maps_{key}_specimen_ids.png", dpi=200)
    plt.close(fig)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--out", default="publications/rl_band_selection/figures/proto/")
    ap.add_argument("--render-only", action="store_true", help="redraw from the saved npz, compute nothing")
    ap.add_argument("--no-csv-check", action="store_true", help="skip all-pairs vs all_pairs_reference.csv check")
    ap.add_argument("--score-loc", default="auto", choices=["auto", "bl", "br", "tl", "tr"])
    ap.add_argument("--results-root", default="results/rl_select")
    ap.add_argument("--label-mean-over-repeats", action="store_true",
                    help="panel label = mean over all stored repeats (repeat-0 value in parentheses); needs score_vs_n output")
    ap.add_argument("--means-csv", default=None, help="score_vs_n summary CSV (default <out>/score_vs_n_summary.csv)")
    ap.add_argument("--algorithms", default=",".join(ALGOS), help="comma list of columns to draw, e.g. knn,svm,kmeans,nmf")
    ap.add_argument("--suffix", default=None, help="output suffix (default: '' for all columns, else _<algos>)")
    ap.add_argument("--score-fontsize", type=float, default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious Accelerate-BLAS matmul warnings on macOS
    key, n, repeat = args.dataset, args.n, args.repeat
    set_results_root(args.results_root)
    algos = [a for a in args.algorithms.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        ap.error(f"unknown algorithms {bad}; choose from {ALGOS}")
    suffix = args.suffix if args.suffix is not None else ("" if set(algos) == set(ALGOS) else "_" + "-".join(algos))
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()

    def log(s):
        print(f"[{time.time() - T0:7.1f}s] {s}", flush=True)

    npz = out / f"maps_{key}_n{n}.npz"
    means = (load_means(args.means_csv or out / "score_vs_n_summary.csv", key, n)
             if args.label_mean_over_repeats else None)
    rkw = dict(algos=algos, suffix=suffix, means=means, score_fs=args.score_fontsize)
    if args.render_only:
        render(npz, out, key, n, args.score_loc, **rkw)
        log(f"re-rendered from {npz}")
        return

    from rl_select.data import load_dataset, n_outer_folds
    d = load_dataset(key)
    K = d["n_classes"]
    meta_names = json.loads(str(d["meta"]))["class_names"]
    if key in EXPECT_META:
        assert meta_names == EXPECT_META[key], (meta_names, EXPECT_META[key])
    nf = n_outer_folds(key)
    gt_map = np.full(d["specimen_map"].shape, -1, dtype=np.int8)
    gt_map[d["coords"][:, 0], d["coords"][:, 1]] = d["y"]
    rr, cc = d["coords"][:, 0], d["coords"][:, 1]
    H, W = gt_map.shape
    crop = (max(rr.min() - 5, 0), min(rr.max() + 6, H), max(cc.min() - 5, 0), min(cc.max() + 6, W))
    log(f"{key}: {len(d['y'])} px, {K} classes {meta_names}, {nf} outer folds, F={d['X'].shape[1]}, crop={crop}")

    sel = load_selections(key, n, repeat, nf)
    val_rows: list[dict] = []
    log("supervised ...")
    sup_maps, fold_ba, chosen = run_supervised(key, repeat, nf, sel, n, val_rows, log)
    log("unsupervised / unmixing ...")
    un_maps, nmf_ab, un_scores = run_unsup(key, repeat, nf, chosen, n, log)

    # ---- assemble
    from rl_select.stats import metrics
    labels, scores_rows = [], []
    y_all, g_all, rc = d["y"], d["groups"], d["coords"]
    for mk, label, _ in METHODS:
        L = []
        for a in ALGOS:
            L.append(sup_maps[(mk, a)] if a in SUP else un_maps[(mk, a)])
        labels.append(L)
        for a in SUP:
            fb = fold_ba[(mk, a)]
            pr = sup_maps[(mk, a)][rc[:, 0], rc[:, 1]]
            assert (pr >= 0).all(), "unpredicted annotated pixel"
            pooled = metrics(y_all, pr, g_all, K)["balanced_acc"]
            scores_rows.append(dict(method=label, method_key=mk, algorithm=ALGO_HDR[a], algorithm_key=a,
                                    metric="balanced_acc", value=float(np.mean(fb)), per_fold=[float(v) for v in fb],
                                    pooled_balanced_acc=float(pooled)))
        for a in UNSUP + ("nmf",):
            s = un_scores[(mk, a)]
            pr = un_maps[(mk, a)][rc[:, 0], rc[:, 1]]
            assert (pr >= 0).all()
            scores_rows.append(dict(method=label, method_key=mk, algorithm=ALGO_HDR[a], algorithm_key=a,
                                    metric="ARI" if a in UNSUP else "matched_balanced_acc",
                                    value=s["ARI"] if a in UNSUP else s["BA"], ARI=s["ARI"], NMI=s["NMI"],
                                    matched_balanced_acc=s["BA"], per_fold=[], info={k: v for k, v in s.items()
                                                                                    if k not in ("ARI", "NMI", "BA")}))
    meta = dict(dataset=key, n=n, repeat=repeat, nf=nf, class_names=class_names_for(key, meta_names), cache_class_names=meta_names,
                selections={m: {str(f): {"selected": [int(i) for i in chosen[(m, f)]],
                                         "seed": (None if m == "all" else int(sel[m][f]["seed"])),
                                         "method": (None if m == "all" else sel[m][f]["method"])}
                                for f in range(nf)} for m, _, _ in METHODS})
    save_outputs(key, n, gt_map, crop, labels, nmf_ab, scores_rows, meta, d["specimen_map"], out)

    # ---- scores csv
    with open(out / f"maps_{key}_n{n}_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "algorithm", "metric", "value", "per_fold"])
        for r in scores_rows:
            w.writerow([r["method"], r["algorithm"], r["metric"], f"{r['value']:.6f}",
                        ";".join(f"{v:.6f}" for v in r["per_fold"])])
            if r["algorithm_key"] in SUP:
                w.writerow([r["method"], r["algorithm"], "balanced_acc_pooled", f"{r['pooled_balanced_acc']:.6f}", ""])
            else:
                for mname, v in (("ARI", r["ARI"]), ("NMI", r["NMI"]), ("matched_balanced_acc", r["matched_balanced_acc"])):
                    if mname != r["metric"]:
                        w.writerow([r["method"], r["algorithm"], mname, f"{v:.6f}", ""])

    # ---- specimens
    spec_rows, spec_names = specimen_table(key, repeat, labels, gt_map, crop)
    cols = ["specimen", "true_class", "n_px", "centroid_row", "centroid_col", "location"] + \
           [f"{p}|{mk}|{a}" for (mk, a) in spec_names for p in ("acc", "maj")]
    with open(out / f"maps_{key}_n{n}_specimens.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in spec_rows:
            w.writerow(r)
    log("specimens whose pixels are mostly misclassified (acc < 0.5), counted over the 6 x 7 panels:")
    for r in spec_rows:
        bad = [(mk, a) for (mk, a) in spec_names if r[f"acc|{mk}|{a}"] < 0.5]
        if len(bad) >= 6:
            per_m = Counter(mk for mk, _ in bad)
            maj = Counter(r[f"maj|{mk}|{a}"] for mk, a in bad).most_common(1)[0]
            log(f"   specimen {r['specimen']} (true class {r['true_class']}, {r['location']}, centroid "
                f"({r['centroid_row']:.0f},{r['centroid_col']:.0f}), {r['n_px']} px): wrong in {len(bad)}/{len(spec_names)} panels"
                f" {dict(per_m)}; most common wrong class {maj[0]}")

    # ---- validation txt
    csvres = None
    if not args.no_csv_check:
        log("all-pairs CSV check ...")
        csvres = check_all_pairs_csv(key, nf, val_rows, log)
    mx = max(r["diff"] for r in val_rows)
    lines = [f"Validation: {key} n={n} repeat={repeat}  (recomputed fold-wise balanced accuracy vs stored reference)",
             f"max |BA difference| over all {len(val_rows)} comparisons: {mx:.3e}  (tolerance {TOL:.0e})", ""]
    by = {}
    for r in val_rows:
        by.setdefault((r["method"], r["ref"]), []).append(r)
    lines.append("per method / reference: max abs diff, #comparisons, seeds that reproduced")
    for (mth, ref), rs in by.items():
        lines.append(f"  {mth:28s} vs {ref:14s} max {max(x['diff'] for x in rs):.3e}  n={len(rs):3d}  "
                     f"seeds {dict(Counter(x['seed'] for x in rs))}")
    lines.append("")
    stored_only = [r for r in val_rows if r["ref"] == "stored record"]
    ev_only = [r for r in val_rows if r["ref"] == "evaluate()"]
    lines.append(f"max diff vs stored records: {max(r['diff'] for r in stored_only):.3e} ({len(stored_only)} comparisons)")
    lines.append(f"max diff vs fresh evaluate(): {max(r['diff'] for r in ev_only):.3e} ({len(ev_only)} comparisons; used where no "
                 f"stored value exists, e.g. RF for SA, and for the all-pairs row)" if ev_only else "no evaluate() comparisons")
    if csvres:
        reps, o = csvres
        lines.append("")
        lines.append(f"All pairs vs summary/all_pairs_reference.csv (mean BA over {reps} repeats x {nf} folds, fresh evaluate()):")
        for c in SUP:
            mine, ref = o[c]
            lines.append(f"  {c:4s} mine {mine:.12f}  csv {ref:.12f}  diff {abs(mine - ref):.3e}")
        lines.append(f"  max diff vs CSV: {max(abs(a - b) for a, b in o.values()):.3e}")
    elif not args.no_csv_check:
        lines.append("(all_pairs_reference.csv not found / no row for this dataset)")
    lines += ["", "detail: method | fold | clf | reference | ref BA | my BA | |diff| | seed used | seeds tried"]
    for r in val_rows:
        lines.append(f"{r['method']:28s} f{r['fold']} {r['clf']:4s} {r['ref']:14s} {r['ref_ba']:.12f} {r['my_ba']:.12f} "
                     f"{r['diff']:.3e} seed={r['seed']} tried={r['tried']}")
    (out / f"maps_{key}_n{n}_validation.txt").write_text("\n".join(lines) + "\n")
    log(f"VALIDATION max abs diff = {mx:.3e}")

    render(npz, out, key, n, args.score_loc, **rkw)
    render_specimen_ids(key, out, crop, gt_map)
    log(f"done: {out}/maps_{key}_n{n}.pdf")


if __name__ == "__main__":
    main()
