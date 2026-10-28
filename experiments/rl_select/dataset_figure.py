"""PROTOTYPE dataset figures for the RL band-selection paper (what do the samples look like, where do classes differ).

  PYTHONPATH=src ME_ATTENTION_RECORDED_WL=1 .venv/bin/python experiments/rl_select/dataset_figure.py

Writes to publications/rl_band_selection/figures/proto/:
  fig_datasets.pdf/.png           two-column: appearance | annotated specimens | class-mean EEM | 365 nm class spectra
  fig_eem_discriminability.pdf/.png  single-column: per-pair Fisher ratio on the excitation x recorded-emission grid
  dataset_table.csv               all counts computed from the loaded data

Everything is plotted on the *recorded* emission wavelengths (ME_ATTENTION_RECORDED_WL=1 is forced below). The processed
files mislabel the emission axis of some excitations (docs/datasets/audit_2026-10-22); the me_attention loader corrects it.
Appearance images are reused from docs/datasets/audit_2026-10-22/renders (white-light PNGs for the lichens, 365 nm
false-colour PNGs for collagen); they live in the same pixel frame as the class masks (checked, see report).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ["ME_ATTENTION_RECORDED_WL"] = "1"  # must be set before me_attention.data is used

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

from me_attention.data import DATASETS, _class_map_npy, _class_map_png, DATA_ROOT, load, to_grid

ROOT = Path(__file__).resolve().parents[2]
RENDERS = ROOT / "docs" / "datasets" / "audit_2026-10-22" / "renders"
OUT = ROOT / "publications" / "rl_band_selection" / "figures" / "proto"
OUT.mkdir(parents=True, exist_ok=True)
RL_CACHE = ROOT / "results" / "rl_select" / "cache"

W1, W2 = 3.5, 7.16
ORDER = ("lichens1", "lichens2", "collagen_pepsin", "collagen_acetic")
TITLE = {"lichens1": "Lichens 1", "lichens2": "Lichens 2", "collagen_pepsin": "Collagen Pepsin",
         "collagen_acetic": "Collagen Acetic"}
PALETTE = np.array([(230, 25, 75), (60, 180, 75), (0, 130, 200), (255, 225, 25), (240, 50, 230)]) / 255.0
# class names by class index (0..C-1). Lichens 1 order = cache names Class 1, 3, 6, 7 (asserted below).
CLASS_LABELS = {
    "lichens1": ["R. sinensis (lower)", "R. sinensis (upper)", "F. soredica", "R. fraxinea"],
    "lichens2": ["Class A", "Class B", "Class C", "Class D", "Class E"],
    "collagen_pepsin": ["GA 0.002 %", "GA 0.0075 %", "GA 0.013 %"],
    "collagen_acetic": ["GA 0.002 %", "GA 0.0075 %", "GA 0.013 %"],
}
EXPECTED_CACHE_NAMES = {
    "lichens1": ["Class 1", "Class 3", "Class 6", "Class 7"],
    "lichens2": ["Class 1 (Red)", "Class 2 (Green)", "Class 3 (Blue)", "Class 4 (Yellow)", "Class 5 (Magenta)"],
}
APPEARANCE = {  # (png, tag drawn on the image)
    "lichens1": ("lichens1_whitelight_rgb.png", "white light"),
    "lichens2": ("lichens2_whitelight_rgb.png", "white light"),
    "collagen_pepsin": ("collagen_pepsin_fluorescence_rgb.png", "365 nm, false colour"),
    "collagen_acetic": ("collagen_acetic_fluorescence_rgb.png", "365 nm, false colour"),
}
EX_SPECTRA = 365.0
SPECIMEN_NUMBER_OFFSET = 0  # figure shows the repo's specimen/group ids (0-based); set to 1 for 1-based numbering

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "axes.labelpad": 1.0, "lines.linewidth": 0.9,
    "pdf.fonttype": 42, "font.family": "sans-serif", "figure.dpi": 100, "savefig.dpi": 400,
})


# ------------------------------------------------------------------------------------------------ data
def specimen_means(d, transform=None) -> np.ndarray:
    """[S, F] mean over each specimen's pixels of transform(X) (float64 accumulation)."""
    S = len(d.specimen_classes)
    out = np.empty((S, d.X.shape[1]))
    for s in range(S):
        x = d.X[d.groups == s].astype(np.float64)
        out[s] = (transform(x) if transform else x).mean(0)
    return out


def to_ex_grid(d, vec: np.ndarray) -> np.ndarray:
    """[F] feature vector -> [E, M] grid on (excitation, recorded emission), NaN = pair not measured."""
    g = np.full((len(d.excitations), len(d.emissions)), np.nan)
    g[d.grid_index[:, 0], d.grid_index[:, 1]] = vec
    return g


def fisher_ratio(Zlog: np.ndarray, spec_cls: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-feature Fisher ratio on specimen means Zlog [S, F] (already log1p-transformed).

    between = variance (ddof 0) of the class means (class means = mean of that class's specimen means, classes weighted
    equally); within = mean over classes of the sample variance (ddof 1) of that class's specimen means.
    """
    mu = np.stack([Zlog[spec_cls == c].mean(0) for c in range(n_classes)])
    within = np.stack([Zlog[spec_cls == c].var(0, ddof=1) for c in range(n_classes)]).mean(0)
    between = mu.var(0)
    return between / np.maximum(within, 1e-12)


def raw_class_map(key: str):
    cfg = DATASETS[key]
    d = DATA_ROOT / cfg["dir"]
    return (_class_map_png if cfg["kind"] == "png" else _class_map_npy)(d)


def prepare(key: str) -> dict:
    d = load(key)
    z = np.load(RL_CACHE / f"{key}.npz")
    x_equal = bool(np.array_equal(z["X"], d.X)) and z["X"].shape == d.X.shape
    # sanity checks the figure relies on
    if key in EXPECTED_CACHE_NAMES:
        assert d.class_names == EXPECTED_CACHE_NAMES[key], (key, d.class_names)
    assert d.n_classes == len(CLASS_LABELS[key])
    assert np.isfinite(d.X).all() and (d.X >= 0).all()
    if key.startswith("collagen"):  # classes must run left to right across the 3x3 grid
        cx = [d.coords[d.y == c, 1].mean() for c in range(d.n_classes)]
        assert cx == sorted(cx), (key, cx)
    Z = specimen_means(d)
    Zlog = specimen_means(d, np.log1p)
    cls = d.specimen_classes
    C = d.n_classes
    class_mean = np.stack([Z[cls == c].mean(0) for c in range(C)])  # [C, F] mean of specimen means
    assert (class_mean > 0).all()
    sel = d.ex_nm == EX_SPECTRA
    order = np.argsort(d.em_nm[sel])
    em365 = d.em_nm[sel][order]
    spec365 = Z[:, sel][:, order]  # [S, B]
    cm_map, _ = raw_class_map(key)
    return dict(
        key=key, d=d, x_equal=x_equal, Z=Z, cls=cls, class_mean=class_mean, em365=em365, spec365=spec365,
        fisher=to_ex_grid(d, fisher_ratio(Zlog, cls, C)),
        class_mean_grid=[to_ex_grid(d, np.log10(class_mean[c])) for c in range(C)],
        raw_class_map=cm_map,
    )


# ------------------------------------------------------------------------------------------------ figure helpers
class InchFig:
    """Figure with axes placed in inches from the top-left corner (tight, explicit layout)."""

    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))

    def axes(self, x: float, y_top: float, w: float, h: float, **kw):
        return self.fig.add_axes([x / self.w, 1 - (y_top + h) / self.h, w / self.w, h / self.h], **kw)

    def text(self, x: float, y_top: float, s: str, **kw):
        return self.fig.text(x / self.w, 1 - y_top / self.h, s, **kw)


OUTLINE = [pe.withStroke(linewidth=1.2, foreground="black")]


def crop_box(mask: np.ndarray, frac: float = 0.04):
    """Bounding box of the annotation + margin (frac of the longer box side), clipped to the image."""
    rr, cc = np.nonzero(mask)
    m = int(round(frac * max(rr.max() - rr.min(), cc.max() - cc.min())))
    r0, r1 = max(rr.min() - m, 0), min(rr.max() + m + 1, mask.shape[0])
    c0, c1 = max(cc.min() - m, 0), min(cc.max() + m + 1, mask.shape[1])
    return r0, r1, c0, c1


def draw_legend(ifig: InchFig, x0: float, y_top: float, width: float, labels, fs: float = 5.0):
    """One-line class legend (colour square + name) laid out left to right inside [x0, x0+width] (inches)."""
    r = ifig.fig.canvas.get_renderer()
    sq, pad_t, pad_i = 0.06, 0.028, 0.095
    x = x0
    for c, lab in enumerate(labels):
        ax = ifig.axes(x, y_top, sq, sq)
        ax.set_facecolor(PALETTE[c]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.3)
        t = ifig.text(x + sq + pad_t, y_top + sq / 2, lab, fontsize=fs, va="center", ha="left")
        wt = t.get_window_extent(r).width / ifig.fig.dpi
        x += sq + pad_t + wt + pad_i
    return x - pad_i - x0  # used width


def scaled_axis(ax, lo: float, hi: float):
    """Linear y axis in units of 10^k (avoids the offset text); returns the exponent k."""
    k = int(np.floor(np.log10(max(abs(hi), 1e-12))))
    k = k if k >= 3 else 0
    ax.set_ylim(0, hi * 1.06)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 10 ** k:g}"))
    return k


# ------------------------------------------------------------------------------------------------ fig 1
def fig_datasets(P: dict):
    IMG_W, G1, LBL = 1.30, 0.03, 0.19
    TOP, BAND, W = 0.17, 0.26, W2
    xa = LBL
    xb = xa + IMG_W + G1
    xc = xb + IMG_W + 0.30          # first heatmap left edge (room for the y tick labels + axis label)
    STRIP, CB_GAP, CB_W = 1.95, 0.035, 0.065
    xcb = xc + STRIP + CB_GAP
    xd = xcb + CB_W + 0.19 + 0.33   # (d) axes left edge (cbar tick labels + y tick labels + axis label)
    WD = W - 0.03 - xd

    # geometry first: crop boxes and row heights
    geo = {}
    for k in ORDER:
        p = P[k]
        r0, r1, c0, c1 = crop_box(p["d"].specimen_map >= 0)
        geo[k] = dict(box=(r0, r1, c0, c1), h=IMG_W * (r1 - r0) / (c1 - c0))
    H = TOP + sum(geo[k]["h"] + BAND for k in ORDER) + 0.01
    F = InchFig(W, H)

    # column headers
    hdr = dict(fontsize=6.2, fontweight="bold", ha="center", va="center")
    F.text(xa + IMG_W / 2, TOP / 2, "(a) Appearance", **hdr)
    F.text(xb + IMG_W / 2, TOP / 2, "(b) Annotated specimens", **hdr)
    F.text(xc + STRIP / 2 - 0.05, TOP / 2, "(c) Class-mean EEM, log$_{10}$ intensity", **hdr)
    F.text(xd + WD / 2 - 0.15, TOP / 2, f"(d) {EX_SPECTRA:.0f} nm emission, mean ± s.d.", **hdr)

    y = TOP
    for k in ORDER:
        p, g = P[k], geo[k]
        d = p["d"]
        h = g["h"]
        r0, r1, c0, c1 = g["box"]
        C = d.n_classes
        # ---- (a), (b)
        png, tag = APPEARANCE[k]
        img = np.asarray(Image.open(RENDERS / png).convert("RGB"), dtype=float) / 255.0
        assert img.shape[:2] == d.image_shape, (k, img.shape, d.image_shape)
        cmap = p["raw_class_map"]
        over = img * 0.45
        for c in range(C):
            m = cmap == c
            over[m] = 0.35 * over[m] + 0.65 * PALETTE[c]
        for x0, im, name in ((xa, img, "a"), (xb, over, "b")):
            ax = F.axes(x0, y, IMG_W, h)
            ax.imshow(im[r0:r1, c0:c1], interpolation="bilinear", aspect="equal")
            ax.set_axis_off()
            if name == "a":
                ax.text(0.03, 0.03, tag, transform=ax.transAxes, color="white", fontsize=4.6, va="bottom",
                        ha="left", path_effects=OUTLINE)
            else:
                nS = len(d.specimen_classes)
                for s in range(nS):
                    rr, cc = np.nonzero(d.specimen_map == s)
                    ax.text(cc.mean() - c0, rr.mean() - r0, str(s + SPECIMEN_NUMBER_OFFSET), color="white", fontsize=5.0 if nS <= 16 else 4.0,
                            ha="center", va="center", path_effects=OUTLINE, fontweight="bold")
        # ---- row label (rotated), vertically centred on the images
        F.text(0.075, y + h / 2, TITLE[k], rotation=90, fontsize=6.4, fontweight="bold", ha="center", va="center")
        nS, Fp = len(d.specimen_classes), d.n_features
        F.text(0.152, y + h / 2, f"{nS} specimens, {Fp} pairs", rotation=90, fontsize=4.8, ha="center",
               va="center", color="#444444")

        # ---- (c) class-mean EEM strip, shared colour scale per dataset
        grids = p["class_mean_grid"]
        vmin = float(np.nanmin(grids)); vmax = float(np.nanmax(grids))
        norm = Normalize(vmin, vmax)
        cm = plt.get_cmap("viridis").copy(); cm.set_bad("#d4d4d4")
        gap = 0.03
        wh = (STRIP - (C - 1) * gap) / C
        E = len(d.excitations)
        i365 = int(np.where(d.excitations == EX_SPECTRA)[0][0])
        for c in range(C):
            ax = F.axes(xc + c * (wh + gap), y, wh, h)
            im = ax.imshow(np.ma.masked_invalid(grids[c]), cmap=cm, norm=norm, aspect="auto", origin="lower",
                           extent=(d.emissions[0] - 5, d.emissions[-1] + 5, -0.5, E - 0.5), interpolation="nearest")
            ax.set_xticks([450, 700])
            ax.set_yticks(range(E))
            if c == 0:
                ax.set_yticklabels([f"{e:g}" for e in d.excitations])
                ax.set_ylabel("Excitation (nm)")
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            tl = ax.get_xticklabels()  # first label left-aligned, last right-aligned: neighbours never collide
            tl[0].set_ha('left'); tl[-1].set_ha('right')
            ax.tick_params(axis='x', labelsize=4.4 if C >= 5 else 5)
            for s in ax.spines.values():
                s.set_edgecolor(PALETTE[c]); s.set_linewidth(1.3)
            ax.add_patch(Rectangle((d.emissions[0] - 5, i365 - 0.5), d.emissions[-1] - d.emissions[0] + 10, 1,
                                   fill=False, ec="white", lw=0.5, ls=(0, (2, 1.5))))
        F.text(xc + STRIP / 2, y + h + 0.185, "Emission (nm)", fontsize=5.2, ha="center", va="center")
        cax = F.axes(xcb, y, CB_W, h)
        cb = plt.colorbar(im, cax=cax)
        cb.outline.set_linewidth(0.4)
        cb.set_ticks(np.arange(np.ceil(vmin), np.floor(vmax) + 1))
        cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)

        # ---- (d) class-mean spectra at 365 nm
        ax = F.axes(xd, y, WD, h)
        em, S365 = p["em365"], p["spec365"]
        hi = 0
        for c in range(C):
            v = S365[p["cls"] == c]
            mu, sd = v.mean(0), v.std(0, ddof=1)
            ax.fill_between(em, mu - sd, mu + sd, color=PALETTE[c], alpha=0.25, lw=0)
            ax.plot(em, mu, color=PALETTE[c], lw=1.0)
            hi = max(hi, (mu + sd).max())
        ax.set_xlim(em[0], em[-1]); ax.set_xticks([450, 550, 650])
        kexp = scaled_axis(ax, 0, hi)
        ax.set_ylabel("Intensity (a.u.)" if kexp == 0 else f"Intensity (a.u. $\\times$10$^{kexp}$)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        F.text(xd + WD / 2, y + h + 0.185, "Emission (nm)", fontsize=5.2, ha="center", va="center")

        # ---- legend band under (a), (b)
        draw_legend(F, xa, y + h + 0.09, 2 * IMG_W + G1, CLASS_LABELS[k])
        y += h + BAND
    return F


# ------------------------------------------------------------------------------------------------ fig 2
def fig_discriminability(P: dict):
    LEFT, HEAT_W, GAP, CB_W = 0.34, 2.60, 0.05, 0.07
    CELL_H = HEAT_W / 31.0          # square cells on the 31-band emission axis
    TITLE_H, TOP, BOT = 0.125, 0.02, 0.27
    vals = np.concatenate([P[k]["fisher"][np.isfinite(P[k]["fisher"])] for k in ORDER])
    norm = matplotlib.colors.LogNorm(vmin=0.04, vmax=16)
    assert vals.min() >= 0.04 and vals.max() <= 16
    heights = [len(P[k]["d"].excitations) * CELL_H for k in ORDER]
    H = TOP + sum(h + TITLE_H for h in heights) + BOT
    F = InchFig(W1, H)
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("#d4d4d4")
    y = TOP
    first_y = y + TITLE_H
    for i, k in enumerate(ORDER):
        d = P[k]["d"]
        E = len(d.excitations)
        h = heights[i]
        ax = F.axes(LEFT, y + TITLE_H, HEAT_W, h)
        im = ax.imshow(np.ma.masked_invalid(P[k]["fisher"]), cmap=cm, norm=norm, aspect="auto", origin="lower",
                       extent=(d.emissions[0] - 5, d.emissions[-1] + 5, -0.5, E - 0.5), interpolation="nearest")
        ax.set_yticks(range(E)); ax.set_yticklabels([f"{e:g}" for e in d.excitations])
        ax.set_xticks(np.arange(450, 701, 50))
        if i < len(ORDER) - 1:
            ax.set_xticklabels([])
        for s_ in ax.spines.values():
            s_.set_linewidth(0.4)
        F.text(LEFT, y + TITLE_H / 2 + 0.005, f"{TITLE[k]}", fontsize=6.2, fontweight="bold", ha="left", va="center")
        ft = P[k]["fisher"]
        F.text(LEFT + HEAT_W, y + TITLE_H / 2 + 0.005,
               f"{d.n_classes} classes, {len(d.specimen_classes)} specimens, {d.n_features} pairs", fontsize=5, ha="right",
               va="center", color="#444444")
        last_ax = ax
        y += h + TITLE_H
    F.text(LEFT + HEAT_W / 2, y + 0.19, "Emission wavelength (nm)", fontsize=6, ha="center", va="center")
    F.text(0.07, (first_y + y) / 2, "Excitation wavelength (nm)", fontsize=6, rotation=90, ha="center", va="center")
    cax = F.axes(LEFT + HEAT_W + GAP, first_y, CB_W, y - first_y)
    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([0.1, 1, 10]); cb.set_ticklabels(["0.1", "1", "10"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    F.text(W1 - 0.045, (first_y + y) / 2, "Fisher ratio (log scale)", fontsize=5.5, rotation=90, ha="center", va="center")
    return F


# ------------------------------------------------------------------------------------------------ table
def missing_ranges(d) -> str:
    """'ex:lo-hi' ranges (nm) of emission bins on the 10 nm union grid that have no retained pair at that excitation."""
    out = []
    for e in d.excitations:
        have = set(d.emissions[d.grid_index[d.ex_nm == e][:, 1]].tolist())
        miss = [m for m in d.emissions.tolist() if m not in have]
        rng = []
        for m in miss:
            if rng and m - rng[-1][1] == 10:
                rng[-1][1] = m
            else:
                rng.append([m, m])
        out.append(f"{e:g}:" + (",".join(f"{a:g}-{b:g}" for a, b in rng) if rng else "none"))
    return " | ".join(out)


def build_table(P: dict):
    rows = []
    for k in ORDER:
        p = P[k]
        d = p["d"]
        C = d.n_classes
        S_per = np.bincount(d.specimen_classes, minlength=C)
        px_kept = np.bincount(d.y, minlength=C)
        px_raw = np.array([(p["raw_class_map"] == c).sum() for c in range(C)])
        E, M = len(d.excitations), len(d.emissions)
        join = lambda a: "/".join(str(int(v)) for v in a)
        rows.append(dict(
            dataset=TITLE[k], key=k,
            image_rows_x_cols=f"{d.image_shape[0]}x{d.image_shape[1]}",
            n_classes=C, class_names_in_cache="; ".join(d.class_names), class_labels_in_figure="; ".join(CLASS_LABELS[k]),
            n_specimens=int(S_per.sum()), specimens_per_class=join(S_per),
            annotated_px_per_class_used=join(px_kept), annotated_px_per_class_raw_mask=join(px_raw),
            n_px_used=int(px_kept.sum()), px_dropped_small_components=int(px_raw.sum() - px_kept.sum()),
            specimen_px_min=int(np.bincount(d.groups).min()), specimen_px_max=int(np.bincount(d.groups).max()),
            n_excitations=E, excitations_nm=" ".join(f"{e:g}" for e in d.excitations),
            n_emission_bins_union=M, emission_range_nm=f"{d.emissions.min():g}-{d.emissions.max():g}",
            emission_step_nm=f"{np.unique(np.diff(d.emissions)).tolist()}",
            F_pairs=d.n_features, grid_cells_ExM=E * M, pairs_not_retained=E * M - d.n_features,
            pairs_per_excitation=join([(d.ex_nm == e).sum() for e in d.excitations]),
            emission_nm_not_retained_per_excitation=missing_ranges(d),
            X_equals_rl_cache=p["x_equal"],
            X_min=float(d.X.min()), X_median=float(np.median(d.X)), X_max=float(d.X.max()),
            fisher_min=float(np.nanmin(p["fisher"])), fisher_median=float(np.nanmedian(p["fisher"])),
            fisher_max=float(np.nanmax(p["fisher"])),
        ))
    with open(OUT / "dataset_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    P = {k: prepare(k) for k in ORDER}
    for name, F in (("fig_datasets", fig_datasets(P)), ("fig_eem_discriminability", fig_discriminability(P))):
        for ext in ("pdf", "png"):
            F.fig.savefig(OUT / f"{name}.{ext}")
        print("saved", name, f"{F.w:.2f} x {F.h:.2f} in")
        plt.close(F.fig)
    for r in build_table(P):
        print({k: v for k, v in r.items() if k in ("dataset", "image_rows_x_cols", "specimens_per_class", "annotated_px_per_class_used", "annotated_px_per_class_raw_mask", "F_pairs", "X_equals_rl_cache")})
