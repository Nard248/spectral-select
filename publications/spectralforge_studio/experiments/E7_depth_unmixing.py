"""E7: depth-dependent failure of linear spectral unmixing (SpectralForge Studio paper).

Scene: E5 tissue (4 mm dermis / 1 mm buried NADH+FAD+collagen+elastin layer / dense-collagen cover of uniform thickness t),
lab protocol (8 TLS lines x 31 LCTF bands), noisy `fast` capture at scale 0.25, ROI = central 0.4-0.6 of the image.
Endmember groups (all layers of a dye summed): NADH, FAD, matrix (collagen + elastin), scatter (elastic + glint + leak).
Ground truth: simulator components (`explain_pixel`) at 12 random ROI pixels, in the abundance units of `unmix`
(signal summed over all channels, electrons).  Unmixing: NNLS per ROI pixel, mean over the ROI.

Endmember libraries
  library      = endmembers measured on the SAME tissue with a thin cover (t_ref = 0.05 mm): what a user gets from "pure"
                 reference spectra of the fluorophores that are not buried (the standard practice).
  right-depth  = endmembers from the forward model of the capture at its own thickness t (`endmembers_from_forward_model`).
Channels: joint 8-excitation EEM, joint 4-excitation (340-430 nm, the 365-ish UV set), single 365 nm excitation.
Reported: relative bias (est/gt - 1) of every abundance vs t with a bootstrap CI over pixels, the redox ratio
FAD/(FAD+NADH) error, the spectral angle between the library and the right-depth endmembers (the depth distortion), the
condition number of E, the mean residual fraction.

Run: .venv/bin/python publications/spectralforge_studio/experiments/E7_depth_unmixing.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import C3_common as C  # noqa: E402
from spectraforge.studio import analysis, api  # noqa: E402
from spectraforge.studio import unmix as um  # noqa: E402
from spectraforge.studio.surrogate import ChannelTable, resolve_roi  # noqa: E402

GROUPS = {"NADH": ["dye:NADH@*"], "FAD": ["dye:FAD@*"], "matrix": ["dye:collagen@*", "dye:elastin@*"],
          "scatter": ["elastic", "elastic@*", "glint", "glint@*", "leak_source", "leak_source@*"]}
GROUPS_L = {"NADH@buried": ["dye:NADH@buried"], "NADH@deep": ["dye:NADH@substrate"], "FAD@buried": ["dye:FAD@buried"],
            "FAD@deep": ["dye:FAD@substrate"], "matrix@cover": ["dye:collagen@cover", "dye:elastin@cover"],
            "matrix@below": ["dye:collagen@buried", "dye:collagen@substrate", "dye:elastin@buried", "dye:elastin@substrate"],
            "scatter": GROUPS["scatter"]}
T_GRID = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3]
T_REF = 0.05
ROI = {"shape": "rect", "normalized": True, "x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6}
CHSETS = {"joint8": None, "joint4": [340.0, 365.0, 400.0, 430.0], "single365": [365.0]}


def capture_uniform(t: float, quick: bool, seed: int = 2):
    exc = [340.0, 365.0] if quick else None
    exp = C.build_scene(t, t, seed=seed, excitations=exc)
    exp = api.set(exp, C.T_LAYER + "kind", "uniform")
    cap = C.cached(f"E7_cap_{'q' if quick else 'f'}_t{t:g}_s{seed}", lambda: api.capture(exp, "fast", scale=C.SCALE))
    return exp, cap


def group_signals(exp, cap, pix, groups, table):
    """Per-pixel electron signal of every component group through the capture's own path: ``(n_pix, K, C)``, from the
    simulator's exact component decomposition of ``explain_pixel`` (sum over all components == the capture's mean signal)."""
    expls = um.explain_pixels(exp, pix, scale=C.SCALE, excitations=sorted(set(table.ex_nm.tolist())))
    out = np.zeros((len(pix), len(groups), table.n))
    for i, e in enumerate(expls):
        comps = um._components_of(e, table)
        ratio = um._electron_ratio(e, table, exp)
        for k, pats in enumerate(groups.values()):
            out[i, k] = um._group_sum(comps, pats, table.n) * ratio
    return out


def library(sig, names, table, label):
    """Endmembers = unit-sum mean signature of each group over the pixels; gt = mean group totals (electrons)."""
    mean = sig.mean(axis=0)                                  # (K, C)
    em = um.Endmembers(list(names), np.stack([um._unit_sum(v) for v in mean], axis=1), table, label, "electrons")
    return em, mean.sum(axis=1)


def solve(em, y, mask):
    """NNLS of one spectrum ``y`` (C,) on the channels ``mask``; returns abundances (K,), residual fraction, cond(E)."""
    E = em.matrix[mask]
    r = um.unmix(y[mask][None, None, :], um.Endmembers(em.names, E, ChannelTable(em.channels.ex_nm[mask], em.channels.em_nm[mask]),
                                                        em.source, em.signal), "nnls")
    return r.abundances[0, 0], float(r.residual_fraction[0, 0]), float(np.linalg.cond(E))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-pixels", type=int, default=24)
    args = ap.parse_args()
    q = args.quick
    T0 = time.time()
    C.style()
    tg = [0.05, 0.4, 1.0] if q else T_GRID
    chsets = {"joint2": None} if q else CHSETS
    npix = 6 if q else args.n_pixels
    out = {"quick": q, "t_grid_mm": tg, "t_ref_mm": T_REF, "groups": GROUPS, "groups_layered": GROUPS_L, "channel_sets": list(chsets),
           "n_pixels": npix, "units": {"bias": "relative (est/gt - 1)", "angle": "degrees"},
           "method": "signatures and ground truth from the SAME npix ROI pixels (oracle: noise-free right-depth unmixing is exact "
                     "by construction); 'heldout': signatures from the first half of the pixels, data+GT from the second half",
           "runs": []}
    lib_thin = None
    brng = np.random.default_rng(0)
    for t in [T_REF] + [x for x in tg if x != T_REF]:
        exp, cap = capture_uniform(t, q)
        table = um.ChannelTable.from_result(cap)
        roi = resolve_roi(ROI, cap.width, cap.height)
        m = analysis.roi_mask(roi, cap.height, cap.width)
        ys, xs = np.nonzero(m)
        pick = np.random.default_rng(3).choice(xs.size, size=min(npix, xs.size), replace=False)
        pix = [(int(xs[i]), int(ys[i])) for i in pick]
        sig = group_signals(exp, cap, pix, {**GROUPS, **{"L:" + k: v for k, v in GROUPS_L.items()}}, table)
        sg, sl = sig[:, :len(GROUPS)], sig[:, len(GROUPS):]
        Ye, _ = um.capture_matrix(cap, "electrons")
        Yn, _ = um.capture_matrix(cap, "adu_net")
        rows, cols = zip(*[(y, x) for x, y in pix])
        y_clean, y_noisy = Ye[rows, cols].mean(axis=0), Yn[rows, cols].mean(axis=0)
        em_t, gt_t = library(sg, GROUPS, table, "forward_model")
        em_L, gt_L = library(sl, list(GROUPS_L), table, "forward_model")
        h = len(pix) // 2
        em_h, _ = library(sg[:h], GROUPS, table, "heldout")
        gt_h = sg[h:].mean(axis=0).sum(axis=1)
        y_h = {"noiseless": Ye[rows[h:], cols[h:]].mean(axis=0), "noisy": Yn[rows[h:], cols[h:]].mean(axis=0)}
        if t == T_REF:
            lib_thin = em_t
        angles = {n: um.spectral_angle_deg(lib_thin.matrix[:, i], em_t.matrix[:, i]) for i, n in enumerate(GROUPS)}
        # consistency check of the endmember path: the explained signal must equal the capture's mean signal
        out.setdefault("consistency", {})[f"{t:g}"] = {
            "explain_over_capture_electrons": float(sg.sum(axis=1).mean(axis=0).sum() / y_clean.sum()),
            "noisy_over_clean_signal": float(y_noisy.sum() / y_clean.sum())}
        for data, y in (("noiseless", y_clean), ("noisy", y_noisy)):
            for cs, ex in chsets.items():
                mask = np.ones(table.n, bool) if ex is None else np.isin(table.ex_nm, ex)
                for lib_name, em, gt in (("library_thin", lib_thin, gt_t), ("right_depth", em_t, gt_t)):
                    runs = _run(out, t, cs, lib_name, data, em, gt, y, mask, angles, brng, ye=(Yn if data == "noisy" else Ye), rows=rows, cols=cols)
                if cs == "joint8":
                    _run(out, t, cs, "right_depth_layered", data, em_L, gt_L, y, mask, angles, brng, layered=True)
                    _run(out, t, cs, "right_depth_heldout", data, em_h, gt_h, y_h[data], mask, angles, brng)
        print(f"t={t:g}: {time.time() - T0:.0f} s", flush=True)
    ref = {r["t_mm"]: r["gt"] for r in out["runs"]}[T_REF if not q else tg[0]]
    for r in out["runs"]:
        r["err_over_gt_ref"] = {n: (r["abundance"][n] - r["gt"][n]) / ref[n] for n in ref}
        r["gt_over_gt_ref"] = {n: r["gt"][n] / ref[n] for n in ref}
    out["gt_ref"] = ref
    out["elapsed_s"] = time.time() - T0
    C.write_results("E7" if not q else "E7_quick", out)
    make_figure(out, q)


def _ratio(d):
    den = d["FAD"] + d["NADH"]
    return d["FAD"] / den if den > 0 else float("nan")


def _run(out, t, cs, lib_name, data, em, gt, y, mask, angles, brng, ye=None, rows=None, cols=None, layered=False):
    est, rf, cond = solve(em, y, mask)
    names = list(em.names)
    d_est, d_gt = dict(zip(names, est.tolist())), dict(zip(names, np.asarray(gt).tolist()))
    layers = None
    if layered:
        layers = {"abundance_layers": d_est, "gt_layers": d_gt}
        tot = lambda d, key: sum(v for k, v in d.items() if k.startswith(key))   # noqa: E731
        d_est = {"NADH": tot(d_est, "NADH"), "FAD": tot(d_est, "FAD"), "matrix": tot(d_est, "matrix"), "scatter": d_est["scatter"]}
        d_gt = {"NADH": tot(layers["gt_layers"], "NADH"), "FAD": tot(layers["gt_layers"], "FAD"),
                "matrix": tot(layers["gt_layers"], "matrix"), "scatter": layers["gt_layers"]["scatter"]}
    rec = {"t_mm": t, "channels": cs, "endmembers": lib_name, "data": data, "abundance": d_est, "gt": d_gt,
           "rel_bias": {n: d_est[n] / d_gt[n] - 1 for n in d_est},
           "redox_est": _ratio(d_est), "redox_gt": _ratio(d_gt),
           "residual_fraction": rf, "cond_E": cond}
    rec["redox_err"] = rec["redox_est"] - rec["redox_gt"]
    if data == "noisy" and ye is not None:           # bootstrap over the pixels of the noisy mean spectrum (data noise only)
        bs = []
        for _ in range(100):
            b = brng.integers(0, len(rows), len(rows))
            yb = ye[np.asarray(rows)[b], np.asarray(cols)[b]].mean(axis=0)
            a, _, _ = solve(em, yb, mask)
            bs.append((a[names.index("FAD")] / (a[names.index("FAD")] + a[names.index("NADH")]) - rec["redox_gt"]) if "FAD" in names else np.nan)
        rec["redox_err_ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    if layers:
        rec.update(layers)
    rec["library_vs_rightdepth_angle_deg"] = angles
    out["runs"].append(rec)
    return rec


def _rows(out, cs, lib, data):
    return sorted([r for r in out["runs"] if r["channels"] == cs and r["endmembers"] == lib and r["data"] == data],
                  key=lambda r: r["t_mm"])


LIB = {"library_thin": ("thin-cover library", C.OI["verm"], "o"), "right_depth": ("right depth", C.OI["blue"], "s"),
       "right_depth_layered": ("right depth, per layer", C.OI["green"], "^"), "right_depth_heldout": ("right depth, held-out pixels", C.OI["purple"], "D")}


def make_figure(out, q):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D as L
    cs_main = "joint8" if "joint8" in out["channel_sets"] else out["channel_sets"][0]
    dye = {"NADH": C.OI["blue"], "FAD": C.OI["green"], "matrix": C.OI["orange"], "scatter": "0.4"}
    fig, ax = plt.subplots(1, 4, figsize=(C.DCOL, 2.2))
    a = ax[0]
    rs = _rows(out, cs_main, "right_depth", "noisy")
    t = [r["t_mm"] for r in rs]
    for n, c in dye.items():
        a.plot(t, np.maximum([r["gt_over_gt_ref"][n] for r in rs], 1e-9), "-o", ms=2.5, color=c, label=n)
    a.set_yscale("log"); a.set_xlabel("cover thickness (mm)"); a.set_ylabel("true signal / signal at 0.05 mm")
    a.set_title("(a) detected signal", fontsize=8); a.legend(fontsize=7, loc="lower left")
    for j, (name, ttl) in enumerate((("NADH", "(b) NADH: error vs true signal"), ("FAD", "(c) FAD: error vs true signal"))):
        a = ax[1 + j]
        rs0 = _rows(out, cs_main, "right_depth", "noisy")
        a.plot([r["t_mm"] for r in rs0], [r["gt_over_gt_ref"][name] for r in rs0], "k-", lw=1.4, zorder=1)
        for lib, (lab, col, mk) in LIB.items():
            for data, ls in (("noisy", "-"), ("noiseless", "--")):
                if data == "noiseless" and lib in ("right_depth", "right_depth_layered"):
                    continue                                     # exact by construction (same pixels), not informative
                rs = _rows(out, cs_main, lib, data)
                if rs:
                    a.plot([r["t_mm"] for r in rs], np.where(np.abs([r["err_over_gt_ref"][name] for r in rs]) > 1e-6, np.abs([r["err_over_gt_ref"][name] for r in rs]), np.nan), ls=ls,
                           marker=mk, ms=2.2, lw=0.8, color=col, zorder=2)
        a.set_yscale("log"); a.set_ylim(1e-6, 3); a.set_xlabel("cover thickness (mm)"); a.set_title(ttl, fontsize=8)
        if j == 0:
            a.set_ylabel("abs. error and true signal\n(relative to true at 0.05 mm)")
    a = ax[3]
    for lib, (lab, col, mk) in LIB.items():
        for cs, ls in zip(out["channel_sets"], ("-", "--", ":")):
            if lib in ("right_depth_layered", "right_depth_heldout") and cs != "joint8":
                continue
            rs = _rows(out, cs, lib, "noisy")
            if rs:
                a.plot([r["t_mm"] for r in rs], [r["redox_err"] for r in rs], ls=ls, marker=mk, ms=2.2, lw=0.9, color=col)
    a.axhline(0, color="0.5", lw=0.5); a.set_ylim(-1.05, 1.05); a.set_xlabel("cover thickness (mm)"); a.set_ylabel("redox ratio error")
    a.set_title("(d) FAD/(FAD+NADH)", fontsize=8)
    fig.tight_layout(w_pad=0.7)
    hl = [L([], [], color=c, marker=mk, ms=3, label=lab) for lab, c, mk in LIB.values()]
    hl += [L([], [], color="k", lw=1.4, label="true signal"), L([], [], color="0.3", ls="-", label="noisy data"),
           L([], [], color="0.3", ls="--", label="noise-free")]
    fig.legend(handles=hl, title="(b, c): joint 8-line EEM", title_fontsize=7, loc="upper center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.40, 0.02), frameon=False)
    hd = [L([], [], color="0.3", ls=ls, label=cs) for cs, ls in zip(out["channel_sets"], ("-", "--", ":"))]
    fig.legend(handles=hd, title="(d): channel set", title_fontsize=7, loc="upper center", ncol=3,
               fontsize=7, bbox_to_anchor=(0.90, 0.02), frameon=False)
    C.savefig(fig, "E7_depth_unmixing" + ("_quick" if q else ""))


if __name__ == "__main__":
    main()
