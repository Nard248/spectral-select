"""Dedicated linear-unmixing figure for the RL band-selection paper (PROTOTYPE).

For one dataset, n pairs and fold 0 of repeat 0 (the same sets as outcome_maps.py) and four selections
(All pairs, RL add/remove, CSS, SA):

  block (a)  NMF unmixing, r = number of classes, on the raw non-negative selected features, scaled exactly as in
             outcome_maps.py (per-pair 99th percentile, nndsvda, 1000 iterations, abundances sum-normalised, each
             endmember Hungarian-matched to a class).  Fitted on all annotated pixels (unsupervised).
  block (b)  supervised fully constrained least squares (abundance >= 0, sum = 1) with the class-mean spectra of the
             TRAINING specimens of fold 0 as endmembers (same scaling, class-mean = mean of specimen means).
             scipy.optimize.nnls with the standard sum-to-one augmentation (a delta-weighted row of ones).

Columns: endmember signatures over the selected pairs, one abundance map per endmember (white -> class colour, 0-1,
cropped to the annotated region), and a class x endmember mean-abundance heatmap.

  PYTHONPATH=src .venv/bin/python experiments/rl_select/unmixing_figure.py \
      --datasets lichens1,collagen_pepsin --results-root results/rl_select

Wavelengths are the RECORDED ones (ME_ATTENTION_RECORDED_WL=1 is forced; feature order == RL cache).
Writes fig_unmixing_<key>_n<n>.{pdf,png}, unmixing_<key>_n<n>_{endmembers,class_by_endmember,summary}.csv to --out.
"""
from __future__ import annotations

import os

os.environ["ME_ATTENTION_RECORDED_WL"] = "1"  # before me_attention.data is used

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment, nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outcome_maps as om  # noqa: E402

ROOT = om.ROOT
PAL = om.PAL
ROWS = (("all", "All pairs"), ("rl_addremove", "RL add/\nremove"), ("css", "CSS\n(label-free)"), ("sa", "SA\n(supervised)"))
DELTA = 1e3          # weight of the sum-to-one row in the FCLS augmentation
FS = 7.0

plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "axes.labelpad": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "figure.dpi": 100, "savefig.dpi": 300,
})


# ------------------------------------------------------------------------------------------------ numerics
def p99_scale(A: np.ndarray):
    p = np.percentile(A, 99, axis=0)
    p[p <= 0] = 1.0
    return p


def balanced_acc(y, pred, K):
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    sup = cm.sum(1)
    return float(np.mean(np.diag(cm)[sup > 0] / sup[sup > 0]))


def nmf_unmix(Xraw: np.ndarray, y: np.ndarray, K: int):
    """Same recipe as outcome_maps.run_unsup. Returns class-ordered abundances [N,K], endmembers [K,F] (unit L2, scaled
    space), the per-feature p99 scale, and the NMF iteration count."""
    from sklearn.decomposition import NMF
    A = Xraw / p99_scale(Xraw)
    nmf = NMF(n_components=K, init="nndsvda", max_iter=1000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W = nmf.fit_transform(A)
    H = nmf.components_
    hn = np.linalg.norm(H, axis=1)
    hn[hn == 0] = 1.0
    Wm = W * hn[None, :]
    tot = Wm.sum(1, keepdims=True)
    zero = tot[:, 0] <= 1e-12
    Wn = np.where(zero[:, None], 1.0 / K, Wm / np.maximum(tot, 1e-12))
    M = np.stack([Wn[y == c].mean(0) for c in range(K)])
    r, c = linear_sum_assignment(-M)
    comp2class = np.zeros(K, dtype=np.int64)
    comp2class[c] = r
    Ab = np.zeros_like(Wn)
    Em = np.zeros_like(H)
    for comp in range(K):
        Ab[:, comp2class[comp]] = Wn[:, comp]
        Em[comp2class[comp]] = H[comp] / hn[comp]
    return Ab, Em, nmf.n_iter_


def fcls(E: np.ndarray, X: np.ndarray, delta: float = DELTA):
    """Fully constrained LS: min ||E a - x||, a >= 0, sum(a) = 1 for every row of X. E: [F,K]; X: [N,F].
    Standard augmentation: stack a delta-weighted row of ones under E (target delta) and solve with NNLS."""
    F, K = E.shape
    Ea = np.vstack([E, delta * np.ones((1, K))])
    out = np.empty((len(X), K))
    for i in range(len(X)):
        out[i], _ = nnls(Ea, np.append(X[i], delta), maxiter=50 * K)
    return out


def purity(Ab, y, K):
    M = np.stack([Ab[y == c].mean(0) for c in range(K)])   # class x endmember
    return M, float(np.mean(np.diag(M)))


# ------------------------------------------------------------------------------------------------ data
def load_wavelengths(key: str, F_expected: int, X_cache: np.ndarray):
    from me_attention.data import load
    dd = load(key)
    assert dd.X.shape == X_cache.shape, (key, dd.X.shape, X_cache.shape)
    same = bool(np.array_equal(dd.X, X_cache))
    if not same:
        print(f"WARNING {key}: me_attention X differs from the RL cache X; wavelength labels may be misaligned", flush=True)
    return dd.ex_nm.astype(float), dd.em_nm.astype(float), same


def compute(key, n, repeat=0, log=print):
    from rl_select.data import load_dataset, n_outer_folds, specimen_folds
    d = load_dataset(key)
    K = d["n_classes"]
    nf = n_outer_folds(key)
    ex, em, same = load_wavelengths(key, d["X"].shape[1], d["X"])
    sel = om.load_selections(key, n, repeat, nf)
    folds = specimen_folds(d, repeat)
    te = np.isin(d["groups"], folds[0])
    y = d["y"].astype(np.int64)
    Xraw = np.clip(d["X"].astype(np.float64), 0, None)
    F = Xraw.shape[1]
    res = {}
    for mk, label in ROWS:
        S = np.arange(F) if mk == "all" else np.asarray(sel[mk][0]["selected"], dtype=np.int64)
        if mk != "all":
            S = np.sort(S)  # display order: by feature index (= by excitation, then emission)
            assert len(S) == n
        A = Xraw[:, S]
        t0 = time.time()
        Ab, Em, it = nmf_unmix(A, y, K)
        Mn, pur = purity(Ab, y, K)
        ba = balanced_acc(y, Ab.argmax(1), K)
        # ---- FCLS with training-specimen class means as endmembers
        sc = p99_scale(A)
        As = A / sc
        spec_means = np.stack([As[d["groups"] == s].mean(0) for s in range(len(d["specimen_classes"]))])
        train_spec = np.array([s for s in range(len(d["specimen_classes"])) if not np.isin(s, folds[0])])
        E = np.stack([spec_means[train_spec[d["specimen_classes"][train_spec] == c]].mean(0) for c in range(K)])  # [K,F]
        Af = fcls(E.T, As)
        Mf, purf = purity(Af, y, K)
        baf = balanced_acc(y, Af.argmax(1), K)
        res[mk] = dict(S=S, nmf=dict(Ab=Ab, Em=Em, M=Mn, pur=pur, ba=ba, it=it,
                                     ba_te=balanced_acc(y[te], Ab[te].argmax(1), K), pur_te=purity(Ab[te], y[te], K)[1]),
                       fcls=dict(Ab=Af, Em=E, M=Mf, pur=purf, ba=baf, ba_te=balanced_acc(y[te], Af[te].argmax(1), K),
                                 pur_te=purity(Af[te], y[te], K)[1], sum_dev=float(np.abs(Af.sum(1) - 1).max())))
        log(f"  {key} {label.replace(chr(10), ' ')}: NMF BA {ba:.3f} pur {pur:.3f} | FCLS BA {baf:.3f} pur {purf:.3f} "
            f"(held-out BA {res[mk]['fcls']['ba_te']:.3f}) {time.time() - t0:.0f}s")
    return d, ex, em, res, same


# ------------------------------------------------------------------------------------------------ figure
def class_cmap(c):
    return LinearSegmentedColormap.from_list(f"c{c}", [(1, 1, 1), tuple(PAL[c])])


def draw(key, n, d, ex, em, res, out_dir):
    from dataset_figure import InchFig, draw_legend
    K = d["n_classes"]
    names = om.class_names_for(key, list(range(K)))
    coords = d["coords"]
    H, W = d["specimen_map"].shape
    crop = (max(coords[:, 0].min() - 5, 0), min(coords[:, 0].max() + 6, H), max(coords[:, 1].min() - 5, 0),
            min(coords[:, 1].max() + 6, W))
    r0, r1, c0, c1 = crop
    asp = (r1 - r0) / (c1 - c0)
    W_FIG, LBL, GAP = 7.16, 0.74, 0.08
    HEAT_W, HEAT_LAB = 0.62, 0.34
    mw = min(0.95, (W_FIG - LBL - 1.5 - HEAT_W - HEAT_LAB - 0.10 - (K + 2) * GAP) / K)
    SIG_W = W_FIG - LBL - HEAT_W - HEAT_LAB - 0.10 - (K + 2) * GAP - K * mw
    mh = mw * asp
    ROW_H = max(mh, 1.32)
    TITLE, HDR, LEG = 0.20, 0.30, 0.20
    H_FIG = LEG + 2 * (TITLE + HDR + len(ROWS) * (ROW_H + 0.06)) + 0.02
    F = InchFig(W_FIG, H_FIG)
    draw_legend(F, LBL, 0.07, W_FIG, [f"{i + 1} {nm}" for i, nm in enumerate(names)], fs=FS)
    y_top = LEG
    for bi, (bk, btitle) in enumerate((("nmf", f"(a) NMF unmixing, r = {K} (unsupervised, fitted on all annotated pixels)"),
                                       ("fcls", "(b) FCLS, training-specimen class means as endmembers (supervised; ho = held-out BA/purity)"))):
        F.text(0.02, y_top + TITLE / 2, btitle, fontsize=FS + 0.5, fontweight="bold", va="center", ha="left")
        y_top += TITLE
        F.text(LBL + SIG_W / 2, y_top + HDR / 2, "Endmember signatures (ex/em nm;\nall pairs: grouped by excitation)", fontsize=FS, ha="center", va="center")
        F.text(LBL + SIG_W + GAP + (K * (mw + GAP)) / 2, y_top + HDR / 2, "Abundance maps (0 white - 1 class colour)",
               fontsize=FS, ha="center", va="center")
        xh = LBL + SIG_W + GAP + K * (mw + GAP) + GAP
        F.text(xh + HEAT_W / 2 + 0.05, y_top + HDR / 2, "Mean abundance,\nclass (row) x endm.", fontsize=FS, ha="center", va="center")
        y_top += HDR
        for mk, label in ROWS:
            r = res[mk][bk]
            S = res[mk]["S"]
            Ab, Em, M = r["Ab"], r["Em"], r["M"]
            F.text(LBL - 0.06, y_top + ROW_H / 2, label, fontsize=FS, ha="right", va="center", linespacing=1.05)
            # ---- signatures
            ax = F.axes(LBL + 0.30, y_top + 0.02, SIG_W - 0.30, ROW_H - 0.58)
            Emn = Em / np.maximum(Em.max(1, keepdims=True), 1e-12)
            if mk == "all":
                x = np.arange(len(S))
                for c in range(K):
                    ax.plot(x, Emn[c], color=PAL[c], lw=0.7)
                ex_s = ex[S]
                starts = [int(np.flatnonzero(ex_s == e)[0]) for e in np.unique(ex_s)]
                ax.set_xticks(starts)
                ax.set_xticklabels([f"{e:g}" for e in np.unique(ex_s)], rotation=90)
                ax.set_xlim(0, len(S) - 1)
            else:
                x = np.arange(len(S))
                bw = 0.8 / K
                for c in range(K):
                    ax.bar(x - 0.4 + bw * (c + 0.5), Emn[c], width=bw, color=PAL[c], lw=0)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{ex[i]:g}/{em[i]:g}" for i in S], rotation=90)
                ax.set_xlim(-0.6, len(S) - 0.4)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 1])
            ax.set_ylabel("rel. intensity" if True else "", fontsize=FS)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            # ---- abundance maps
            for c in range(K):
                img = np.full((H, W), np.nan)
                img[coords[:, 0], coords[:, 1]] = Ab[:, c]
                a = F.axes(LBL + SIG_W + GAP + c * (mw + GAP), y_top + (ROW_H - mh) / 2, mw, mh)
                cm = class_cmap(c).copy()
                cm.set_bad("white")
                a.imshow(np.ma.masked_invalid(img[r0:r1, c0:c1]), cmap=cm, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_edgecolor(PAL[c]); sp.set_linewidth(1.0)
            # ---- class x endmember heatmap
            a = F.axes(xh, y_top + 0.05, HEAT_W, HEAT_W)
            a.imshow(M, cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            for i in range(K):
                for j in range(K):
                    a.text(j, i, f"{M[i, j]:.2f}".lstrip("0") if M[i, j] < 1 else "1", ha="center", va="center", fontsize=FS - 0.5,
                           color="white" if M[i, j] > 0.55 else "black")
            a.set_xticks(range(K)); a.set_yticks(range(K))
            a.set_xticklabels([str(i + 1) for i in range(K)]); a.set_yticklabels([str(i + 1) for i in range(K)])
            for i, (tx, ty) in enumerate(zip(a.get_xticklabels(), a.get_yticklabels())):
                tx.set_color(PAL[i] * 0.75); ty.set_color(PAL[i] * 0.75); tx.set_fontweight("bold"); ty.set_fontweight("bold")
            a.tick_params(length=0)
            for sp in a.spines.values():
                sp.set_linewidth(0.4)
            txt = f"BA {r['ba']:.2f}\npur. {r['pur']:.2f}"
            if bk == "fcls":
                txt += f"\nho {r['ba_te']:.2f}/{r['pur_te']:.2f}"
            F.text(xh + HEAT_W / 2, y_top + 0.05 + HEAT_W + 0.17, txt, fontsize=FS, ha="center", va="top", linespacing=1.0)
            y_top += ROW_H + 0.06
    for ext in ("pdf", "png"):
        F.fig.savefig(out_dir / f"fig_unmixing_{key}_n{n}.{ext}")
    plt.close(F.fig)
    return F.w, F.h


def write_csvs(key, n, d, ex, em, res, out_dir):
    K = d["n_classes"]
    names = om.class_names_for(key, list(range(K)))
    with open(out_dir / f"unmixing_{key}_n{n}_endmembers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "selection", "endmember_class", "pair_index", "ex_nm", "em_nm", "value_unit_L2_scaled_space"])
        for bk in ("nmf", "fcls"):
            for mk, label in ROWS:
                Em, S = res[mk][bk]["Em"], res[mk]["S"]
                for c in range(K):
                    en = Em[c] / max(np.linalg.norm(Em[c]), 1e-12)
                    for j, i in enumerate(S):
                        w.writerow([bk, mk, names[c], int(i), f"{ex[i]:g}", f"{em[i]:g}", f"{en[j]:.6f}"])
    with open(out_dir / f"unmixing_{key}_n{n}_class_by_endmember.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "selection", "class", "endmember_class", "mean_abundance"])
        for bk in ("nmf", "fcls"):
            for mk, label in ROWS:
                M = res[mk][bk]["M"]
                for i in range(K):
                    for j in range(K):
                        w.writerow([bk, mk, names[i], names[j], f"{M[i, j]:.6f}"])
    with open(out_dir / f"unmixing_{key}_n{n}_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "selection", "matched_BA_all_px", "purity_all_px", "BA_heldout_fold0", "purity_heldout_fold0", "extra"])
        for bk in ("nmf", "fcls"):
            for mk, label in ROWS:
                r = res[mk][bk]
                extra = f"nmf_iters={r['it']}" if bk == "nmf" else f"max|sum(a)-1|={r['sum_dev']:.2e}; delta={DELTA:g}"
                w.writerow([bk, mk, f"{r['ba']:.6f}", f"{r['pur']:.6f}", f"{r['ba_te']:.6f}", f"{r['pur_te']:.6f}", extra])


def check_against_maps(key, n, res, out_dir):
    """The NMF matched BA must equal the one stored by outcome_maps.py (same recipe) - report max difference."""
    p = out_dir / f"maps_{key}_n{n}_scores.csv"
    if not p.exists():
        return None
    ref = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r["algorithm"] == "NMF" and r["metric"] == "matched_balanced_acc":
                ref[r["method"]] = float(r["value"])
    lab = {"all": "All pairs", "rl_addremove": "RL add/remove", "css": "CSS (greedy, label-free)", "sa": "SA (supervised search)"}
    return {mk: abs(res[mk]["nmf"]["ba"] - ref[lab[mk]]) for mk in lab if lab[mk] in ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/rl_select")
    ap.add_argument("--datasets", default="lichens1,collagen_pepsin")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="publications/rl_band_selection/figures/proto/")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    om.set_results_root(args.results_root)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for key in [k for k in args.datasets.split(",") if k]:
        t0 = time.time()
        d, ex, em, res, same = compute(key, args.n)
        write_csvs(key, args.n, d, ex, em, res, out)
        chk = check_against_maps(key, args.n, res, out)
        w, h = draw(key, args.n, d, ex, em, res, out)
        print(f"{key}: saved fig_unmixing_{key}_n{args.n} ({w:.2f} x {h:.2f} in) in {time.time() - t0:.0f}s; "
              f"NMF-vs-outcome_maps max |dBA| = {max(chk.values()) if chk else 'n/a'}", flush=True)


if __name__ == "__main__":
    main()
