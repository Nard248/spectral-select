"""A tissue-like cube with a fluorescent sphere at its centre, captured with the 3D Monte Carlo and studied.

Scene (all numbers are arguments of `build_scene`):
  * ONE layered block = a cube, footprint 10 x 10 mm, a single 10 mm layer of `intralipid_1pct` (a turbid,
    non-fluorescent, tissue-mimicking phantom, mu_s' ~ 1.7-2 /mm, mu_a ~ water), i.e. a cube whose only layer is the host;
  * ONE sphere inclusion (radius 1.5 mm) at the cube centre (block-local y = 5 mm, i.e. 5 mm below the top face),
    whose material is the host + 10 uM fluorescein (so its optics = host, its only difference = the dye);
  * lamp: the lab's tunable xenon source, lines 470 and 490 nm; camera: Nuance-class LCTF, 348 x 256 px, 150 mm above.

Run (about 1.5-2 min on a laptop):
    .venv/bin/python publications/spectralforge_studio/examples/cube_central_fluorophore.py [--out DIR] [--quick]

Outputs (in --out, default ./cube_guide_out): scene .sfx.json (loadable in the web UI: File > Open .sfx.json...),
capture .npz (reload with `api.load_capture`), and the figures fig1..fig5 (.png).
Axes of every cube:  res.adu / res.radiance / res.electrons_mean  ->  dict {excitation_nm: array (H, W, B)}:
    excitation (dict key) x y (axis 0) x x (axis 1) x emission band (axis 2; wavelengths in res.emission_nm[ex]).
"""
from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectraforge.studio import api, block_capture, capture as cap, optics, presets as P, transport3d as t3
from spectraforge.studio.schema import DyeAmount, Inclusion, LayeredBlockShape

HERE = Path(__file__).resolve().parent


# ============================================================================================ 1. the scene
def build_scene(*, cube_mm=10.0, radius_mm=1.5, depth_mm=None, host="intralipid_1pct", dye="fluorescein",
                dye_uM=10.0, excitations=(470.0, 490.0), photons=400_000, inclusion_voxel_mm=0.1,
                exit_cell_mm=0.25, exposure_ms=100.0, power_mW=4.7, with_inclusion=True, null_inclusion=False,
                seed=0, name="Cube with a central fluorophore"):
    """Experiment = cube (single layer of `host`) + sphere inclusion of `host`+`dye` + lamp + LCTF camera.

    depth_mm : depth of the sphere CENTRE below the top face (default cube/2 = centre of the cube).
    null_inclusion=True : same sphere, same voxelisation, but filled with the plain host (matched control).
    """
    depth = cube_mm / 2 if depth_mm is None else depth_mm
    mats = {m.id: m for m in P.builtin_materials()}
    host_m = mats[host].model_copy(update={"id": "host", "name": f"Host ({host})"})
    inc_m = host_m.model_copy(update={"id": "inc_mat", "name": f"Host + {dye} {dye_uM:g} uM",
                                      "dyes": list(host_m.dyes) + [DyeAmount(fluorophore=dye, concentration_uM=dye_uM)]})
    incs = []
    if with_inclusion:
        incs = [Inclusion(name="inc", material_id="host" if null_inclusion else "inc_mat", kind="sphere",
                          center_mm=(0.0, cube_mm - depth, 0.0), radius_mm=radius_mm)]   # y is UP, bottom face y = 0
    block = LayeredBlockShape(footprint_mm=(cube_mm, cube_mm), layers=[P._layer("tissue", "host", cube_mm)],
                              inclusions=incs)
    exp = P.mehsi_lab_replica()                      # gives the validated TLS + Nuance hardware and physics defaults
    src = P.tunable_lamp(target=(0.0, cube_mm / 2, 0.0),
                         sequence=[(float(w), power_mW, exposure_ms) for w in excitations])
    cam = P.nuance_camera(position=(0.0, 150.0, 0.0), target=(0.0, cube_mm, 0.0))
    cam = cam.model_copy(update={"lens": cam.lens.model_copy(update={"focus_distance_mm": 150.0 - cube_mm})})
    vs = exp.acquisition.volume_mc.model_copy(update={
        "excitation_photons": photons, "emission_photons": photons,   # per excitation step / per emission node
        "preview_fraction": 1.0,                                       # preview/fast use ALL photons (default 0.1)
        "inclusion_voxel_mm": inclusion_voxel_mm,                      # voxel of the inclusion map (default 0.2)
        "exit_cell_mm": exit_cell_mm,                                  # image-plane tally cell (default 0.5)
        "emission_node_step_nm": 40.0})                                # emission wavelength nodes (default 20)
    return exp.model_copy(update={
        "name": name,
        "description": "10 mm tissue-like cube, fluorescein sphere at the centre, 3D layered Monte Carlo.",
        "materials": [mats["black_anodized"], host_m, inc_m],   # table + host + sphere (builtin dyes are looked up by name)
        "objects": [P._obj("cube", "Cube", block, "host", (0.0, 0.0, 0.0))],
        "sources": [src], "cameras": [cam], "active_source_id": src.id, "active_camera_id": cam.id,
        "acquisition": exp.acquisition.model_copy(update={"volume_mc": vs, "seed": seed})})


def install_crn():
    """Common random numbers: every photon gets its own random stream (function of seed + photon index), so two
    captures that differ only by the inclusion share photon histories -> far less noise in their difference.
    `block_capture` does not expose the flag, so wrap the two transport entry points (idempotent)."""
    if getattr(t3.excitation_pass, "_crn", False):
        return
    for name in ("excitation_pass", "emission_pass"):
        f = getattr(t3, name)

        @functools.wraps(f)
        def g(*a, _f=f, **k):
            k.setdefault("per_photon_streams", True)
            return _f(*a, **k)
        g._crn = True
        setattr(t3, name, g)


def uninstall_crn():
    for name in ("excitation_pass", "emission_pass"):
        f = getattr(t3, name)
        if getattr(f, "_crn", False):
            setattr(t3, name, f.__wrapped__)


# ============================================================================================ 2. analysis helpers
def geometry(res, r_mm, cube_mm, margin_mm=1.0):
    """Pixel masks from the ground truth: top-face pixels, the sphere's footprint, a background ring."""
    gt = res.ground_truth
    mask = np.nan_to_num(gt["inclusion_mask:inc"], nan=-1)             # 1 inside the sphere's vertical projection
    top = gt["block_face"] == 0                                         # pixels that see the top face (NaN/-1 elsewhere)
    ys, xs = np.nonzero(top)
    px_mm = (xs.max() - xs.min() + 1) / cube_mm                         # pixels per mm on the top face
    cy, cx = np.nonzero(mask == 1)[0].mean(), np.nonzero(mask == 1)[1].mean()
    yy, xx = np.indices(mask.shape)
    d_mm = np.hypot(yy - cy, xx - cx) / px_mm
    inner = (xx - xs.min() >= margin_mm * px_mm) & (xs.max() - xx >= margin_mm * px_mm) \
        & (yy - ys.min() >= margin_mm * px_mm) & (ys.max() - yy >= margin_mm * px_mm)   # away from the cube edges
    return dict(top=top, core=(mask == 1) & (d_mm <= r_mm * 0.6), inc=(mask == 1),
                bg=top & inner & (d_mm > r_mm + 1.0), centre=(int(round(cx)), int(round(cy))), px_mm=px_mm,
                bg_pixel=(int(round(cx)) + int(round(2.5 * px_mm)), int(round(cy))))   # x, y of a background pixel


def window_signal(res, ex, key="electrons_mean", offset_nm=40.0):
    """Sum over the 'clean fluorescence' bands >= ex + offset (the first band, ex+20 nm, is the leaking excitation)."""
    cube = getattr(res, key)[ex]
    em = res.emission_nm[ex]
    return cube[:, :, em >= ex + offset_nm].sum(axis=2)


def residual_std(img, geo):
    """Pixel-to-pixel scatter of `img` in the background ring after removing what is NOT noise: a plane (lamp gradient)
    plus a radial profile around the sphere (its diffuse halo).  Estimates the MC (+shot) noise per pixel."""
    sel = geo["bg"]
    yy, xx = np.nonzero(sel)
    cx, cy = geo["centre"]
    r = np.hypot(xx - cx, yy - cy) / geo["px_mm"]
    ring = np.minimum((r / 0.5).astype(int), int(r.max() / 0.5))
    cols = [np.ones(len(r)), xx - cx, yy - cy] + [(ring == k).astype(float) for k in range(1, ring.max() + 1)]
    A = np.column_stack([c for c in cols if c.any()])
    v = img[yy, xx].astype(float)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return float(np.std(v - (A * coef).sum(axis=1)))        # (not A @ coef: macOS Accelerate emits spurious matmul warnings)


def contrast_metrics(res, geo, ex):
    """C = (S_centre - <S_bg>)/<S_bg>   (expected e-, ring around the sphere: includes its diffuse halo, so conservative)
    CNR = (<ADU_core> - <ADU_bg>) / sigma_pix   with sigma_pix = residual_std of the ADU window image (MC + shot + FPN)."""
    s = window_signal(res, ex)                       # expected electrons (MC noise only)
    a = window_signal(res, ex, key="adu")            # noisy counts as recorded
    cx, cy = geo["centre"]
    bg = s[geo["bg"]].mean()
    return dict(peak_contrast=(s[cy, cx] - bg) / bg, core_contrast=(s[geo["core"]].mean() - bg) / bg,
                cnr=(a[geo["core"]].mean() - a[geo["bg"]].mean()) / residual_std(a, geo),
                signal_centre_e=s[cy, cx], signal_bg_e=bg, mc_noise_rel=residual_std(s, geo) / bg)


def delta_mm(material, nm):
    """Diffusion-theory penetration depth 1/mu_eff, mu_eff = sqrt(3 mua (mua + mus')), of `material` at wavelength nm."""
    o = optics.material_optics(material, {}, grid=np.array([nm - 20.0, nm, nm + 20.0]))
    mua, mus_p = float(o.mua_per_mm[1]), float(o.mus_per_mm[1] * (1 - o.g))
    return 1.0 / np.sqrt(3 * mua * (mua + mus_p)), mua, mus_p


# ============================================================================================ 3. main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cube_guide_out")
    ap.add_argument("--quick", action="store_true", help="fewer sweep points / photons (about 1 min)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    EX, R, L = 470.0, 1.5, 10.0
    PH = 200_000 if a.quick else 400_000

    # ---- 3.1 build, validate, save as a web-UI scene ------------------------------------------------------
    exp = build_scene(cube_mm=L, radius_mm=R, photons=PH)
    api.validate(exp)                                                   # raises with a readable path on error
    api.dump(exp, HERE / "cube_central_fluorophore.sfx.json")           # File > Open .sfx.json... in the web UI
    print("scene saved:", HERE / "cube_central_fluorophore.sfx.json")

    # ---- 3.2 capture: mode "full" = every excitation, full sensor 348x256, full photons ------------------------
    t = time.time()
    res = api.capture(exp, mode="full", seed=0)
    print(f"[capture] {time.time() - t:.1f} s (first run includes numba warm-up); timings {res.timings_s}")
    api.save(res, out / "cube_central.npz")                             # ground truth + 3D volumes included
    print("\ncube axes: excitations", res.excitations_nm, "| per excitation array shape (H, W, B) =",
          res.radiance[EX].shape, "| bands", res.emission_nm[EX][[0, 1, -1]], "... nm")
    print("fields: adu (raw counts), electrons_mean (expected e-, MC noise only), radiance (mW/mm2/sr per band)")
    print("ground-truth maps:", [k for k in res.ground_truth if k.startswith(("inclusion", "origin", "layer", "block"))])
    print("3D light-field volumes:", {k: list(v)[:4] for k, v in res.volumes.items()})
    print("warnings:", res.warnings)

    geo = geometry(res, R, L)
    print(f"\ntop face {geo['top'].sum()} px, {geo['px_mm']:.1f} px/mm; sphere footprint {geo['inc'].sum()} px; "
          f"background ring {geo['bg'].sum()} px; centre pixel (x, y) = {geo['centre']}")
    host_m = next(m for m in exp.materials if m.id == "host")
    for nm in (470.0, 520.0):
        delta, mua, mus_p = delta_mm(host_m, nm)
        print(f"host at {nm:g} nm: mua = {mua:.5f}/mm, mus' = {mus_p:.2f}/mm -> 1/mu_eff = {delta:.1f} mm")
    for ex in res.excitations_nm:
        m = contrast_metrics(res, geo, ex)
        print(f"ex {ex:g} nm: peak contrast at centre = {m['peak_contrast']:.2f}  (core mean {m['core_contrast']:.2f}), "
              f"single-capture CNR = {m['cnr']:.1f}, centre {m['signal_centre_e']:.0f} e-, bg {m['signal_bg_e']:.0f} e-, "
              f"MC noise in bg {100 * m['mc_noise_rel']:.1f} %")

    # cross-check of the noise estimate: two independent seeds of the same scene (MC + shot + fixed-pattern noise)
    res1 = api.capture(exp, mode="full", seed=1)
    a0, a1 = window_signal(res, EX, "adu"), window_signal(res1, EX, "adu")
    print(f"per-pixel noise of the ADU window image @ {EX:g} nm: two-seed estimate {np.std((a0 - a1)[geo['bg']]) / np.sqrt(2):.0f}"
          f" vs ring-residual estimate {residual_std(a0, geo):.0f} ADU")

    # ---- 3.3 figure 1: band images, ground truth, spectra, profile ----------------------------------------------
    em = res.emission_nm[EX]
    ib = int(np.argmin(abs(em - 530)))                                  # a clean fluorescence band (ex + 60 nm)
    cube = res.electrons_mean[EX]
    cx, cy = geo["centre"]
    bx, by = geo["bg_pixel"]
    gt = res.ground_truth
    box = (slice(max(cy - 70, 0), cy + 70), slice(max(cx - 80, 0), cx + 80))     # crop around the cube
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for k, (img, ttl, cm) in enumerate([
            (cube[..., 1], f"band {em[1]:g} nm (= ex+20): elastic leak, sphere not visible", "magma"),
            (cube[..., ib], f"band {em[ib]:g} nm: fluorescence, e-", "magma"),
            (np.where(geo["top"], gt["origin_depth_mm"], np.nan), "GT origin_depth_mm: mean depth where detected light was born", "viridis")]):
        h = ax.flat[k].imshow(img[box], cmap=cm)
        ax.flat[k].set_title(ttl, fontsize=9)
        plt.colorbar(h, ax=ax.flat[k], shrink=0.8)
        ax.flat[k].plot([cx - box[1].start], [cy - box[0].start], "c+", ms=10)
        ax.flat[k].plot([bx - box[1].start], [by - box[0].start], "w+", ms=10)
    sp = ax[1, 0]
    sp.semilogy(em, cube[cy, cx], "o-", label="centre pixel (above sphere)")
    sp.semilogy(em, cube[by, bx], "s-", label="background pixel (top face, 2.5 mm off)")
    sp.axvline(EX + 20, color="gray", ls=":")
    sp.text(EX + 22, sp.get_ylim()[0] * 3, "ex+20: elastic leak", fontsize=8, rotation=90)
    sp.set(xlabel="emission band (nm)", ylabel="expected electrons / pixel", title=f"spectra @ {EX:g} nm excitation")
    sp.legend(fontsize=8)
    pr = ax[1, 1]
    s_win = window_signal(res, EX)
    xmm = (np.arange(s_win.shape[1]) - cx) / geo["px_mm"]
    pr.plot(xmm, s_win[cy], label="fluorescence window (>= ex+40 nm)")
    pr.plot(xmm, np.nan_to_num(gt["inclusion_mask:inc"][cy], nan=0) * s_win[cy].max() * 0.15, "k--", lw=0.8,
            label="GT inclusion_mask (scaled)")
    pr.set(xlim=(-L / 2 - 1, L / 2 + 1), xlabel="x across the top face (mm)", ylabel="window signal (e-)",
           title="horizontal profile through the centre")
    pr.legend(fontsize=8)
    dp = ax[1, 2]
    h = dp.imshow(np.where(geo["top"], gt["inclusion_depth_mm:inc"], np.nan)[box], cmap="cividis")
    plt.colorbar(h, ax=dp, shrink=0.8)
    dp.set_title("GT inclusion_depth_mm (sphere top below surface; 0/blank = no sphere)", fontsize=9)
    fig.suptitle(f"Cube {L:g} mm, sphere r={R} mm at the centre (5 mm deep), fluorescein 10 uM, host intralipid_1pct", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "fig1_overview.png", dpi=110)
    plt.close(fig)

    # ---- 3.4 figure 2: explain_pixel = what physical component makes each pixel -------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for axx, (x, y), ttl in ((axs[0], geo["centre"], "centre pixel"), (axs[1], geo["bg_pixel"], "background pixel")):
        ex_pix = cap.explain_pixel(exp, x, y, scale=1.0, excitations=[EX])     # same scale as the capture!
        comp = ex_pix["components"][f"{EX:g}"]
        print(f"\nexplain_pixel {ttl} ({x},{y}) ground truth:",
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in list(ex_pix["ground_truth"].items())[:6]})
        tot = {k: float(np.sum(v)) for k, v in comp.items()}
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:6]:
            print(f"   {k:<28s} {100 * v / sum(tot.values()):6.2f} %")
        for k, v in comp.items():
            if np.sum(v) > 1e-4 * sum(tot.values()):
                axx.semilogy(em, np.maximum(v, 1e-12), label=k)
        axx.set(title=f"{ttl}: radiance per physical component ({EX:g} nm)", xlabel="emission (nm)", ylim=(1e-10, None))
        axx.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig2_explain_pixel.png", dpi=110)
    plt.close(fig)

    # ---- 3.5 built-in analysis API: ROI EEM, response vs ground truth, line profile --------------------------------
    roi_in = {"shape": "disk", "x": cx, "y": cy, "radius_px": 0.6 * R * geo["px_mm"]}
    roi_bg = {"shape": "disk", "x": bx, "y": by, "radius_px": 0.6 * R * geo["px_mm"]}
    e_in, e_bg = api.eem(res, roi_in, {"mode": "electrons"}), api.eem(res, roi_bg, {"mode": "electrons"})
    print("\nEEM object:", e_in.unit, "excitations", e_in.excitations_nm, "| fields:", [k for k in vars(e_in)][:8])
    rc = api.response(res, "inclusion_depth_mm:inc", {"kind": "band", "ex": EX, "band_nm": 530.0}, bins=6,
                      norm={"mode": "electrons"})
    print("response of the 530 nm band vs sphere-top depth: bins", np.round(rc.bin_centres, 2), "mean", np.round(rc.mean, 0))

    # ---- 3.6 matched control + difference images (MC noise, common random numbers) ----------------------------
    # (a) the same cube with the sphere filled with plain host (identical voxelisation): the true "no dye" reference
    nul = build_scene(cube_mm=L, radius_mm=R, photons=PH, null_inclusion=True)
    res_null = api.capture(nul, mode="full", seed=0)
    s_w, s_n = window_signal(res, EX), window_signal(res_null, EX)
    print(f"\nvs matched null cube (sphere = plain host): centre {s_w[cy, cx]:.0f} vs {s_n[cy, cx]:.0f} e- ; "
          f"far background {s_w[geo['bg']].mean():.0f} vs {s_n[geo['bg']].mean():.0f} e- "
          f"(the ring is lit by the sphere's halo, so the ring-based contrast above underestimates the true one)")
    # (b) difference of two captures (dye 12 uM minus 10 uM): plain vs common random numbers
    exp12 = build_scene(cube_mm=L, radius_mm=R, photons=PH, dye_uM=12.0)
    d_img = {}
    for label, crn, seed_b in (("independent seeds", False, 1), ("same seed, no CRN", False, 0), ("same seed + CRN", True, 0)):
        (install_crn if crn else uninstall_crn)()
        block_capture.clear_cache()          # the 3D-MC cache key does not know about the CRN switch
        r_a = api.capture(exp, mode="fast", scale=0.5, seed=0)
        r_b = api.capture(exp12, mode="fast", scale=0.5, seed=seed_b)
        g = geometry(r_a, R, L)
        d = window_signal(r_b, EX) - window_signal(r_a, EX)
        d_img[label] = (d, residual_std(d, g), d[g["core"]].mean())
    uninstall_crn()
    block_capture.clear_cache()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for axx, (label, (d, sd, mu)) in zip(axs, d_img.items()):
        h = axx.imshow(d, cmap="RdBu_r", vmin=-abs(d).max() * 0.6, vmax=abs(d).max() * 0.6)
        axx.set_title(f"{label}\nresidual sigma (bg ring)={sd:.0f} e-, core mean={mu:.0f} e-", fontsize=9)
        plt.colorbar(h, ax=axx, shrink=0.8)
    fig.suptitle("difference image, dye 12 uM minus 10 uM (window >= ex+40 nm), 0.5x resolution")
    fig.tight_layout()
    fig.savefig(out / "fig3_crn_difference.png", dpi=110)
    plt.close(fig)
    print("noise of the (12 uM - 10 uM) difference in the ring, halo removed [e-]:",
          {k: round(float(v[1]), 1) for k, v in d_img.items()},
          "| expected core difference ~ 0.2 x core signal =", round(0.2 * float(window_signal(res, EX)[geo['core']].mean())))

    # ---- 3.7 figure 4: depth sweep for two hosts ---------------------------------------------------------------
    depths = [1.5, 2.5, 3.5, 5.0, 6.5, 8.0] if not a.quick else [1.5, 3.5, 5.0, 8.0]
    hosts = ["intralipid_1pct", "breast_normal"]       # turbid-clear phantom vs blood-containing tissue (Hb absorption)
    sweep = {h: [] for h in hosts}
    t = time.time()
    for h in hosts:
        for d in depths:
            e = build_scene(cube_mm=L, radius_mm=R, depth_mm=d, host=h, photons=PH)
            r = api.capture(e, mode="fast", scale=0.5, seed=0)          # 174 x 128 px, all excitations
            g = geometry(r, R, L)
            m = contrast_metrics(r, g, EX)
            sweep[h].append(m)
            print(f"  host {h:<16s} depth {d:3.1f} mm: peak contrast {m['peak_contrast']:7.2f}  CNR {m['cnr']:7.1f}  "
                  f"centre {m['signal_centre_e']:9.0f} e-  bg {m['signal_bg_e']:9.0f} e-")
    print(f"[depth sweep] {time.time() - t:.0f} s")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for h in hosts:
        axs[0].plot(depths, [m["peak_contrast"] for m in sweep[h]], "o-", label=h)
        axs[1].semilogy(depths, [max(abs(m["cnr"]), 1e-2) for m in sweep[h]], "o-", label=h)
        axs[2].semilogy(depths, [m["signal_centre_e"] for m in sweep[h]], "o-", label=f"{h} centre")
        axs[2].semilogy(depths, [m["signal_bg_e"] for m in sweep[h]], "x--", label=f"{h} background")
    axs[0].set_yscale("symlog", linthresh=0.1)
    axs[0].axhline(0, color="gray", lw=0.5)
    axs[1].axhline(3, color="k", ls=":")
    axs[0].set(xlabel="sphere-centre depth (mm)", ylabel="peak contrast (S_c - S_bg)/S_bg", title="contrast")
    axs[1].set(xlabel="sphere-centre depth (mm)", ylabel="single-capture CNR", title="CNR (dotted = 3)")
    axs[2].set(xlabel="sphere-centre depth (mm)", ylabel="electrons (window)", title="signal levels")
    for axx in axs:
        axx.legend(fontsize=7)
    fig.suptitle(f"{EX:g} nm excitation, fluorescein 10 uM, r={R} mm: visibility vs depth (cube {L:g} mm)")
    fig.tight_layout()
    fig.savefig(out / "fig4_depth_sweep.png", dpi=110)
    plt.close(fig)

    # ---- 3.8 cube size and photon count (MC noise ~ 1/sqrt(N)) ----------------------------------------------------
    print("\ncube size, centred sphere (depth = L/2):")
    for Lc in ([8.0, 14.0] if a.quick else [8.0, 14.0, 20.0]):
        e = build_scene(cube_mm=Lc, radius_mm=R, photons=PH)
        r = api.capture(e, mode="fast", scale=0.5, seed=0)
        m = contrast_metrics(r, geometry(r, R, Lc), EX)
        print(f"  L = {Lc:4.1f} mm (depth {Lc / 2:4.1f}): peak contrast {m['peak_contrast']:.2f}, CNR {m['cnr']:.1f}")
    print("MC noise of the background (noise-free expected signal) vs photons per pass:")
    for n in ([50_000, 200_000] if a.quick else [50_000, 200_000, 800_000]):
        e = build_scene(cube_mm=L, radius_mm=R, photons=n)
        r = api.capture(e, mode="fast", scale=0.5, seed=0)
        m = contrast_metrics(r, geometry(r, R, L), EX)
        print(f"  N = {n:>8,d}: sigma_bg/mean = {100 * m['mc_noise_rel']:.2f} %  (expect ~1/sqrt(N))")

    # ---- 3.9 reload test ---------------------------------------------------------------------------------------------
    back = api.load_capture(out / "cube_central.npz")
    assert np.allclose(back.electrons_mean[EX], res.electrons_mean[EX])
    print(f"\nreloaded npz OK. total wall time {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
