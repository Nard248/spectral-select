"""Figures of E9 (unsupervised clustering vs cover thickness).  .venv/bin/python .../E9_figures.py [results_dir]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from E2_figures import OI, panel_label, save, style  # noqa: E402

FIG = HERE.parent / "figures"
DS = Path.home() / ".cache" / "spectralforge" / "e9_dataset"
MCOL = {"kmeans_pca": OI["blue"], "gmm_pca": OI["orange"], "kmeans_l2": OI["green"], "cae_kmeans": OI["verm"]}
MLAB = {"kmeans_pca": "k-means (PCA)", "gmm_pca": "GMM (PCA)", "kmeans_l2": "k-means (PCA, L2-norm.)", "cae_kmeans": "CAE latent + k-means"}
SLAB = {"all": "all 217 ch.", "e5_thick15": "E5-optimal 15 ch.", "uniform15": "uniform 15 ch.", "random15": "random 15 ch."}
SCOL = {"all": "k", "e5_thick15": OI["verm"], "uniform15": OI["sky"], "random15": OI["purple"]}


def curve(ent, key):
    x = np.arange(len(ent["bins"]))
    m = np.array([b[key]["mean"] if b[key]["mean"] is not None else np.nan for b in ent["bins"]], float)
    se = np.array([(b[key]["se"] or 0) if b[key]["mean"] is not None else np.nan for b in ent["bins"]], float)
    return x, m, se


def binticks(a, R):
    e = R["settings"]["bin_edges_mm"]
    a.set_xticks(range(len(e) - 1))
    a.set_xticklabels([f"{e[i]:g}-{e[i+1]:g}" for i in range(len(e) - 1)], rotation=40, ha="right", fontsize=6.3)
    a.set_xlabel("local cover thickness bin (mm)")


def fig_scene(R, resdir):
    from spectraforge.studio.dataset import load_index, load_sample
    idx = load_index(DS)
    e = idx["samples"][0]
    sd, lab, ex = load_sample(DS, e)
    th = np.nansum([ex["maps"][f"layer_thickness:cover_{i}"] for i in range(3)], axis=0)
    th = np.where(lab > 0, th, np.nan)
    P = np.load(resdir / "example_preds.npz")
    sid = e["id"]
    rr, cc = np.where(lab > 0)
    sl = (slice(rr.min() - 1, rr.max() + 2), slice(cc.min() - 1, cc.max() + 2))
    fig, ax = plt.subplots(1, 5, figsize=(7.16, 1.7), constrained_layout=True)
    cm = matplotlib.colors.ListedColormap(["#dddddd", OI["blue"], OI["orange"], OI["green"]])
    ax[0].imshow(lab[sl], cmap=cm, vmin=0, vmax=3, interpolation="nearest")
    ax[0].set_title("tissue type\n(NADH / FAD / stroma)", fontsize=6.8)
    im = ax[1].imshow(th[sl], cmap="cividis", vmin=0, vmax=1.5, interpolation="nearest")
    ax[1].set_title("cover thickness\n(wedge, mm)", fontsize=6.8)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.06, pad=0.03, ticks=[0, 0.75, 1.5])
    cb.ax.tick_params(labelsize=6)
    for a, (key, title) in zip(ax[2:], [(f"{sid}|all|kmeans_pca_s3", "k-means (PCA, 3x3 bin)\nall channels"), (f"{sid}|all|cae_kmeans", "CAE + k-means\nall channels"),
                                        (f"{sid}|e5_thick15|kmeans_pca_s3", "k-means (PCA, 3x3 bin)\nE5 15 ch.")]):
        p = np.ma.masked_where(P[key] < 0, P[key])
        a.imshow(p[sl], cmap=matplotlib.colors.ListedColormap([OI["purple"], OI["yellow"], OI["sky"]]), vmin=0, vmax=2, interpolation="nearest")
        a.set_title(title, fontsize=6.8)
    for a, s_ in zip(ax, "abcde"):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        a.text(0.0, 0.99, f"({s_})", transform=a.transAxes, fontsize=8, fontweight="bold", va="top")
    save(fig, FIG, "E9_scene")


def fig_methods(R):
    A = R["aggregate"]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.6), gridspec_kw=dict(wspace=0.42))
    spec = [("all|lda_oracle_s3", "supervised LDA acc., 3x3 bin (reference)", "k", "--", "s"),
            ("all|lda_oracle_s1", "supervised LDA acc., raw (reference)", "0.45", ":", "s"),
            ("all|binfit_kmeans_s3", "k-means fitted per bin, 3x3 bin", OI["blue"], "-", "o"),
            ("all|binfit_kmeans_s1", "k-means fitted per bin, raw", OI["sky"], "-", "o")]
    for key, lab, col, ls, mk in spec:
        ent = A.get(key)
        if ent:
            x, y, se = curve(ent, "acc" if "lda" in key else "ARI")
            ax[0].errorbar(x, y, yerr=se, fmt=mk + ls, color=col, ms=3, lw=1.0, capsize=1.5, label=lab)
    ax[0].set_ylabel("ARI (clustering) / accuracy (LDA)"); ax[0].set_ylim(-0.05, 1.05); binticks(ax[0], R)
    ax[0].set_ylim(-0.05, 1.5); ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax[0].legend(frameon=False, fontsize=5.6, loc="upper center", ncol=1); ax[0].set_title("local (per-bin) fit", fontsize=7.5)
    for key, lab, col in [("all|gmm_pca_s3", "GMM, 3x3 bin", OI["orange"]), ("all|kmeans_pca_s3", "k-means, 3x3 bin", OI["blue"]),
                          ("all|kmeans_pca_s1", "k-means, raw", OI["sky"]), ("all|cae_kmeans", "CAE + k-means", OI["verm"])]:
        ent = A.get(key)
        if ent:
            x, y, se = curve(ent, "ARI")
            ax[1].errorbar(x, y, yerr=se, fmt="o-", color=col, ms=3, lw=1.0, capsize=1.5, label=lab)
    ax[1].set_ylabel("ARI vs tissue type (within bin)"); ax[1].set_ylim(-0.05, 1.05); binticks(ax[1], R)
    ax[1].legend(frameon=False, fontsize=5.8, loc="upper right"); ax[1].set_title("global (whole-capture) fit", fontsize=7.5)
    keys = [("all|kmeans_pca_s1", "k-means\nraw"), ("all|kmeans_pca_s3", "k-means\n3x3"), ("all|gmm_pca_s3", "GMM\n3x3"), ("all|cae_kmeans", "CAE")]
    xs = np.arange(len(keys)); w = 0.38
    for j, (k_, lab, col) in enumerate((("ARI_all", "ARI vs tissue", OI["blue"]), ("NMI_cluster_vs_thickbin", "NMI vs thickness bin", OI["verm"]))):
        mu = [A[k][k_]["mean"] for k, _ in keys]; se = [A[k][k_]["se"] or 0 for k, _ in keys]
        ax[2].bar(xs + (j - 0.5) * w, mu, w, yerr=se, color=col, label=lab, capsize=1.5, error_kw=dict(lw=0.7))
    ax[2].set_xticks(xs); ax[2].set_xticklabels([l for _, l in keys], fontsize=6.5)
    ax[2].set_ylabel("whole-capture score"); ax[2].legend(frameon=False, fontsize=6.0, loc="upper left"); ax[2].set_ylim(0, 0.45)
    for a, s_ in zip(ax, "abc"):
        panel_label(a, f"({s_})")
    save(fig, FIG, "E9_ari_vs_thickness")


def fig_channels(R):
    A = R["aggregate"]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.6), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.1]))
    off = {"all": -0.12, "e5_thick15": -0.04, "uniform15": 0.04, "random15": 0.12}
    for a, (m, key, yl, ttl) in zip(ax[:2], (("binfit_kmeans_s3", "ARI", "ARI (per-bin k-means, 3x3 bin)", "label-free"),
                                              ("lda_oracle_s3", "acc", "LDA accuracy (3x3 bin)", "supervised reference"))):
        for s_ in ("all", "e5_thick15", "uniform15", "random15"):
            ent = A.get(f"{s_}|{m}")
            if ent:
                x, y, se = curve(ent, key)
                a.errorbar(x + off[s_], y, yerr=se, fmt="o-", color=SCOL[s_], ms=2.8, lw=0.9, capsize=1.2, label=SLAB[s_])
        a.set_ylabel(yl); a.set_title(ttl, fontsize=7.5); a.set_ylim(-0.05, 1.05); binticks(a, R)
    ax[0].legend(frameon=False, fontsize=6.0, loc="upper right")
    groups = [("k = 5", ("e5_jointD5", "uniform5", "random5")), ("k = 15", ("e5_thick15", "uniform15", "random15")),
              ("k = 26", ("e5_thickamp26", "uniform26", "random26"))]
    cols = [OI["verm"], OI["sky"], OI["purple"]]; names = ["E5-optimal", "uniform", "random"]; w = 0.26
    allm = np.mean([b["ARI"]["mean"] for b in A["all|binfit_kmeans_s3"]["bins"][:4]])
    ax[2].axhline(allm, color="k", lw=0.8, ls="--", label=f"all 217 ch. ({allm:.2f})")
    for gi, (lab, sets) in enumerate(groups):
        for j, s_ in enumerate(sets):
            ent = A.get(f"{s_}|binfit_kmeans_s3")
            if ent:
                v = [b["ARI"]["mean"] for b in ent["bins"][:4]]
                ax[2].bar(gi + (j - 1) * w, np.mean(v), w, color=cols[j], label=names[j] if gi == 0 else None)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels([g[0] for g in groups], fontsize=7)
    ax[2].set_ylabel("mean per-bin ARI, bins 0-0.35 mm"); ax[2].legend(frameon=False, fontsize=6.0, loc="upper right"); ax[2].set_ylim(0, 1.05)
    for a, s_ in zip(ax, "abc"):
        panel_label(a, f"({s_})")
    save(fig, FIG, "E9_channels")


if __name__ == "__main__":
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results" / "E9"
    style()
    R = json.loads((rd / "results.json").read_text())
    fig_scene(R, rd)
    fig_methods(R)
    fig_channels(R)
