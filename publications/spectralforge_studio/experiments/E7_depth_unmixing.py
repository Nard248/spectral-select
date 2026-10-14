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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    T0 = time.time()
    C.style()
    tg = [0.05, 0.4, 1.0] if q else T_GRID
    chsets = {"joint2": None} if q else CHSETS
    npix = 6 if q else 12
    exp_ref, cap_ref = capture_uniform(T_REF, q)
    table_all = um.ChannelTable.from_result(cap_ref)
    em_lib = um.endmembers_from_forward_model(exp_ref, cap_ref, scale=C.SCALE, groups=GROUPS, n_average=3, seed=0,
                                              signal="adu_net")
    print(f"library built {time.time() - T0:.0f} s", flush=True)
    out = {"quick": q, "t_grid_mm": tg, "t_ref_mm": T_REF, "groups": GROUPS, "channel_sets": list(chsets), "n_pixels_gt": npix,
           "units": {"bias": "relative (est/gt - 1)", "angle": "degrees"}, "runs": []}
    for t in tg:
        exp, cap = capture_uniform(t, q)
        em_t = um.endmembers_from_forward_model(exp, cap, scale=C.SCALE, groups=GROUPS, n_average=3, seed=0, signal="adu_net")
        roi = resolve_roi(ROI, cap.width, cap.height)
        m = analysis.roi_mask(roi, cap.height, cap.width)
        ys, xs = np.nonzero(m)
        rng = np.random.default_rng(3)
        pick = rng.choice(xs.size, size=min(npix, xs.size), replace=False)
        pix = [(int(xs[i]), int(ys[i])) for i in pick]
        gt = um.gt_component_amounts(exp, cap, em_t, pix, scale=C.SCALE, groups=GROUPS)         # (n, K) endmember order
        angles = {n: um.spectral_angle_deg(em_lib.matrix[:, i], em_t.matrix[:, i]) for i, n in enumerate(em_lib.names)}
        for data, sig in (("noisy", "adu_net"), ("noiseless", "electrons")):
          for cs, ex in chsets.items():
            for lib_name, em0 in (("library_thin", em_lib), ("right_depth", em_t)):
                em = replace(em0, signal=sig)
                r = um.unmix(cap, em, "nnls", mask=m, excitations=ex)
                A = r.abundances[m]                    # all ROI pixels (n_roi, K)
                gtm = gt.mean(axis=0)                  # ground truth at sampled pixels (uniform scene: pixel-to-pixel spread = noise)
                est = A.mean(axis=0)
                boot = []
                brng = np.random.default_rng(0)
                for _ in range(200):
                    b = brng.integers(0, A.shape[0], A.shape[0])
                    boot.append(A[b].mean(axis=0) / gtm - 1)
                boot = np.array(boot)
                ni, fi = em.names.index("NADH"), em.names.index("FAD")
                redox = est[fi] / (est[fi] + est[ni])
                redox_gt = gtm[fi] / (gtm[fi] + gtm[ni])
                Ee = em.matrix if ex is None else um._align(em, ChannelTable.from_result(cap, ex))
                out["runs"].append({
                    "t_mm": t, "channels": cs, "endmembers": lib_name, "data": data,
                    "abundance": dict(zip(em.names, est.tolist())), "gt": dict(zip(em.names, gtm.tolist())),
                    "rel_bias": dict(zip(em.names, (est / gtm - 1).tolist())),
                    "rel_bias_ci": {n: [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))] for i, n in enumerate(em.names)},
                    "redox_est": float(redox), "redox_gt": float(redox_gt), "redox_err": float(redox - redox_gt),
                    "residual_fraction": float(np.nanmean(r.residual_fraction[m])), "cond_E": float(np.linalg.cond(Ee)),
                    "library_vs_rightdepth_angle_deg": angles})
        # layer-resolved right-depth library: one endmember per (dye, layer), the "fix" that the simulator makes possible
        em_L = um.endmembers_from_forward_model(exp, cap, scale=C.SCALE, groups=GROUPS_L, n_average=3, seed=0, signal="adu_net")
        gtL = um.gt_component_amounts(exp, cap, em_L, pix, scale=C.SCALE, groups=GROUPS_L).mean(axis=0)
        for data, sig in (("noisy", "adu_net"), ("noiseless", "electrons")):
            r = um.unmix(cap, replace(em_L, signal=sig), "nnls", mask=m)
            est = r.abundances[m].mean(axis=0)
            d_est, d_gt = dict(zip(em_L.names, est.tolist())), dict(zip(em_L.names, gtL.tolist()))
            tot = lambda d, key: sum(v for k, v in d.items() if k.startswith(key))   # noqa: E731
            ab = {n: tot(d_est, n) for n in ("NADH", "FAD")}
            gg = {n: tot(d_gt, n) for n in ("NADH", "FAD")}
            ab["matrix"], gg["matrix"] = tot(d_est, "matrix"), tot(d_gt, "matrix")
            ab["scatter"], gg["scatter"] = d_est["scatter"], d_gt["scatter"]
            out["runs"].append({"t_mm": t, "channels": "joint8", "endmembers": "right_depth_layered", "data": data,
                                "abundance": ab, "gt": gg, "rel_bias": {n: ab[n] / gg[n] - 1 for n in ab},
                                "redox_est": ab["FAD"] / (ab["FAD"] + ab["NADH"]), "redox_gt": gg["FAD"] / (gg["FAD"] + gg["NADH"]),
                                "redox_err": ab["FAD"] / (ab["FAD"] + ab["NADH"]) - gg["FAD"] / (gg["FAD"] + gg["NADH"]),
                                "residual_fraction": float(np.nanmean(r.residual_fraction[m])), "cond_E": float(np.linalg.cond(em_L.matrix)),
                                "abundance_layers": d_est, "gt_layers": d_gt})
        print(f"t={t:g}: {time.time() - T0:.0f} s", flush=True)
    ref = {r["t_mm"]: r["gt"] for r in out["runs"]}[T_REF if not q else tg[0]]
    for r in out["runs"]:
        r["err_over_gt_ref"] = {n: (r["abundance"][n] - r["gt"][n]) / ref[n] for n in ref}
        r["gt_over_gt_ref"] = {n: r["gt"][n] / ref[n] for n in ref}
    out["gt_ref"] = ref
    out["elapsed_s"] = time.time() - T0
    C.write_results("E7" if not q else "E7_quick", out)
    make_figure(out, q)


def _rows(out, cs, lib, data):
    return sorted([r for r in out["runs"] if r["channels"] == cs and r["endmembers"] == lib and r["data"] == data],
                  key=lambda r: r["t_mm"])


def make_figure(out, q):
    import matplotlib.pyplot as plt
    cs_main = out["channel_sets"][0]
    cols = {"NADH": C.OI["blue"], "FAD": C.OI["green"], "matrix": C.OI["orange"], "scatter": "0.4"}
    fig, ax = plt.subplots(1, 4, figsize=(C.DCOL, 2.3))
    a = ax[0]
    rs = _rows(out, cs_main, "right_depth", "noisy")
    t = [r["t_mm"] for r in rs]
    for n, c in cols.items():
        a.plot(t, np.maximum([r["gt_over_gt_ref"][n] for r in rs], 1e-9), "-o", ms=2.5, color=c, label=n)
    a.set_yscale("log"); a.set_xlabel("cover thickness (mm)"); a.set_ylabel("true signal / signal at 0.05 mm")
    a.set_title("(a) detected signal", fontsize=8); a.legend(fontsize=6)
    for j, (data, ttl) in enumerate((("noiseless", "(b) noise-free data"), ("noisy", "(c) noisy data"))):
        a = ax[1 + j]
        for n in ("NADH", "FAD"):
            for lib, ls, mk in (("library_thin", "-", "o"), ("right_depth", "--", "s"), ("right_depth_layered", ":", "^")):
                rs = _rows(out, cs_main, lib, data)
                a.plot([r["t_mm"] for r in rs], [r["err_over_gt_ref"][n] for r in rs], ls=ls, marker=mk, ms=2.5, color=cols[n])
        a.plot([], [], "-", color="0.3", label="thin-cover library"); a.plot([], [], "--", color="0.3", label="right depth"); a.plot([], [], ":", color="0.3", label="right depth, per layer")
        a.plot([], [], color=cols["NADH"], label="NADH"); a.plot([], [], color=cols["FAD"], label="FAD")
        a.axhline(0, color="0.5", lw=0.5); a.set_yscale("symlog", linthresh=0.1); a.set_xlabel("cover thickness (mm)")
        a.set_title(ttl, fontsize=8)
        if j == 0:
            a.set_ylabel("(est - true) / true at 0.05 mm")
    a = ax[3]
    rs = _rows(out, "joint8", "right_depth_layered", "noisy")
    a.plot([r["t_mm"] for r in rs], [r["redox_err"] for r in rs], "-", marker="^", ms=2.5, color=C.OI["green"], lw=1.4)
    a.plot([], [], color=C.OI["green"], marker="^", ms=2.5, label="per-layer, joint8")
    for cs, ls in zip(out["channel_sets"], ("-", "--", ":")):
        for lib, c in (("library_thin", C.OI["verm"]), ("right_depth", C.OI["blue"])):
            rs = _rows(out, cs, lib, "noisy")
            a.plot([r["t_mm"] for r in rs], [r["redox_err"] for r in rs], ls=ls, marker="o", ms=2, color=c)
    for cs, ls in zip(out["channel_sets"], ("-", "--", ":")):
        a.plot([], [], ls=ls, color="0.3", label=cs)
    a.plot([], [], color=C.OI["verm"], label="thin library"); a.plot([], [], color=C.OI["blue"], label="right depth")
    a.axhline(0, color="0.5", lw=0.5); a.set_xlabel("cover thickness (mm)"); a.set_ylabel("redox ratio error")
    a.set_title("(d) FAD/(FAD+NADH)", fontsize=8); a.set_ylim(-1.05, 1.0)
    fig.tight_layout(w_pad=0.7)
    L = plt.Line2D
    h1 = [L([], [], color="0.3", ls="-", label="thin-cover library"), L([], [], color="0.3", ls="--", label="right depth"),
          L([], [], color="0.3", ls=":", label="right depth, per layer"), L([], [], color=cols["NADH"], label="NADH"),
          L([], [], color=cols["FAD"], label="FAD")]
    fig.legend(handles=h1, loc="upper center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, 0.0), frameon=False)
    h2 = [L([], [], color=C.OI["verm"], label="thin library"), L([], [], color=C.OI["blue"], label="right depth"),
          L([], [], color=C.OI["green"], label="per layer"), L([], [], color="0.3", ls="-", label="joint8"),
          L([], [], color="0.3", ls="--", label="joint4"), L([], [], color="0.3", ls=":", label="365 nm only")]
    fig.legend(handles=h2, loc="upper center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, -0.08), frameon=False)
    C.savefig(fig, "E7_depth_unmixing" + ("_quick" if q else ""))


if __name__ == "__main__":
    main()
