"""Selection overlay on the per-pair Fisher-ratio map (PROTOTYPE).

Background: Fisher ratio of every (excitation, recorded-emission) pair on specimen means of log1p intensities, as in
dataset_figure.py.  Overlay: how often each pair is chosen at n pairs, over ALL stored repeats x folds (RL add/remove,
CSS) or repeat 0 x folds x search seeds (SA).  Marker area is proportional to the selection frequency.
One column per method, one row per dataset, recorded wavelengths (feature order == RL cache).

  PYTHONPATH=src .venv/bin/python experiments/rl_select/eem_overlay.py --results-root results/rl_select \
      --datasets lichens1,lichens2,collagen_acetic,collagen_pepsin

Writes fig_eem_overlay.{pdf,png} and eem_overlay_frequencies.csv (every pair of every dataset/method) to --out.
"""
from __future__ import annotations

import os

os.environ["ME_ATTENTION_RECORDED_WL"] = "1"

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outcome_maps as om  # noqa: E402
import score_vs_n as sv  # noqa: E402
from dataset_figure import fisher_ratio, specimen_means, to_ex_grid  # noqa: E402

ROOT = om.ROOT
METHODS = (("rl_addremove", "RL add/remove"), ("css", "CSS (label-free)"), ("sa", "SA (supervised search)"))
FS = 7.0
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.8, "ytick.major.size": 1.8,
                     "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "pdf.fonttype": 42, "font.family": "sans-serif"})


def frequencies(key, n, rl_seeds):
    col = sv.collect(key, {n}, rl_seeds)
    out = {}
    for m, _ in METHODS:
        sets = [r["selected"] for recs in col[(m, n)].values() for r in recs]
        F = None
        cnt = None
        for S in sets:
            cnt = np.zeros(1 + max(max(S), 0)) if cnt is None else cnt
        out[m] = sets
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/rl_select")
    ap.add_argument("--datasets", default=",".join(sv.DATASETS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--rl-seeds", default="0")
    ap.add_argument("--out", default="publications/rl_band_selection/figures/proto/")
    args = ap.parse_args()
    om.set_results_root(args.results_root)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rl_seeds = None if args.rl_seeds == "all" else {int(v) for v in args.rl_seeds.split(",")}
    from me_attention.data import load
    from rl_select.data import load_dataset

    keys = [k for k in args.datasets.split(",") if k]
    P = {}
    rows = []
    for key in keys:
        dd = load(key)
        cache = load_dataset(key)
        assert dd.X.shape == cache["X"].shape, (key, dd.X.shape, cache["X"].shape)
        if not np.array_equal(dd.X, cache["X"]):
            print(f"WARNING {key}: me_attention X != RL cache X; overlay positions may be wrong", flush=True)
        C = dd.n_classes
        Zlog = specimen_means(dd, np.log1p)
        fisher = to_ex_grid(dd, fisher_ratio(Zlog, dd.specimen_classes, C))
        sets = frequencies(key, args.n, rl_seeds)
        freq = {}
        for m, _ in METHODS:
            cnt = np.zeros(dd.n_features)
            for S in sets[m]:
                cnt[np.asarray(S)] += 1
            freq[m] = cnt / max(len(sets[m]), 1)
            for p in np.flatnonzero(cnt):
                rows.append([key, m, int(p), f"{dd.ex_nm[p]:g}", f"{dd.em_nm[p]:g}", f"{freq[m][p]:.4f}", len(sets[m])])
        P[key] = dict(d=dd, fisher=fisher, freq=freq, nsets={m: len(sets[m]) for m, _ in METHODS})
        print(key, "sets per method:", P[key]["nsets"], flush=True)
    with open(out / "eem_overlay_frequencies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "method", "pair_index", "ex_nm", "em_nm", "selection_frequency", "n_sets"])
        w.writerows(rows)

    vals = np.concatenate([P[k]["fisher"][np.isfinite(P[k]["fisher"])] for k in keys])
    vmin, vmax = 0.03, 30.0   # fixed display range (values outside are clipped to the end colours)
    print(f"Fisher ratio range {vals.min():.3g} .. {vals.max():.3g}; displayed {vmin}..{vmax}")
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#d4d4d4")
    W, LBL, CB, GX = 7.16, 0.55, 0.12, 0.08
    pw = (W - LBL - 0.50 - 2 * GX) / 3
    CELL = pw / 31.0
    CELL_H = max(CELL, 0.095)
    TOPH, GY, BOT = 0.22, 0.07, 0.55
    hs = [len(P[k]["d"].excitations) * CELL_H for k in keys]
    H = TOPH + sum(hs) + GY * (len(keys) - 1) + BOT
    fig = plt.figure(figsize=(W, H))
    ytop = TOPH
    SMAX = 60.0
    for i, key in enumerate(keys):
        d = P[key]["d"]
        E = len(d.excitations)
        for j, (m, mname) in enumerate(METHODS):
            ax = fig.add_axes([(LBL + j * (pw + GX)) / W, (H - ytop - hs[i]) / H, pw / W, hs[i] / H])
            im = ax.imshow(np.ma.masked_invalid(P[key]["fisher"]), cmap=cm, norm=norm, aspect="auto", origin="lower",
                           extent=(d.emissions[0] - 5, d.emissions[-1] + 5, -0.5, E - 0.5), interpolation="nearest")
            f = P[key]["freq"][m]
            idx = np.flatnonzero(f > 0)
            ax.scatter(d.emissions[d.grid_index[idx, 1]], d.grid_index[idx, 0], s=SMAX * f[idx], fc="#ff3b30", ec="white",
                       lw=0.5, alpha=0.9, zorder=3)
            ax.set_yticks(range(E))
            ax.set_yticklabels([f"{e:g}" for e in d.excitations] if j == 0 else [])
            if j != 0:
                ax.tick_params(axis="y", length=0)
            ax.set_xticks([450, 550, 650])
            if i != len(keys) - 1:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_title(mname, fontsize=FS, pad=2)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if j == 2:
                ax.text(1.02, 0.5, f"{P[key]['nsets'][m]} sets", transform=ax.transAxes, fontsize=FS - 1, rotation=90, va="center",
                        ha="left", color="#444444") if False else None
        fig.text(0.012, (H - ytop - hs[i] / 2) / H, sv.TITLE.get(key, key).replace(" ", "\n", 1), rotation=90, fontsize=FS, linespacing=1.0, fontweight="bold",
                 ha="center", va="center")
        ytop += hs[i] + GY
    # axis labels, colour bar, size legend
    fig.text((LBL + (3 * pw + 2 * GX) / 2) / W, 0.36 / H, "emission wavelength (nm)", fontsize=FS, ha="center", va="center")
    fig.text(0.045, (H - TOPH - (H - TOPH - BOT) / 2) / H, "excitation (nm)", rotation=90, fontsize=FS, ha="center", va="center") if False else None
    cax = fig.add_axes([(W - 0.34) / W, (BOT) / H, CB / W, (H - TOPH - BOT) / H])
    cb = plt.colorbar(im, cax=cax, extend='both')
    cb.set_ticks([0.1, 1, 10]); cb.set_ticklabels(['0.1', '1', '10'])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1)
    cb.ax.minorticks_off()
    fig.text(W - 0.04, (BOT + (H - TOPH - BOT) / 2) / H, "Fisher ratio (log)", rotation=90, fontsize=FS, ha="center", va="center")
    lax = fig.add_axes([0.5 / W, 0.0, 1, 1]); lax.set_axis_off(); lax.set_xlim(0, W); lax.set_ylim(0, H)
    lax.set_position([0, 0, 1, 1])
    x0 = LBL + 3.4
    lax.text(x0, 0.11, "selection frequency:", fontsize=FS, ha="right", va="center")
    xx = x0 + 0.25
    for fv in (0.1, 0.5, 1.0):
        lax.scatter([xx], [0.11], s=SMAX * fv, fc="#ff3b30", ec="white", lw=0.5)
        lax.text(xx + 0.10, 0.11, f"{fv:g}", fontsize=FS, ha="left", va="center")
        xx += 0.55
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_eem_overlay.{ext}", dpi=300)
    print(f"saved fig_eem_overlay ({W:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
