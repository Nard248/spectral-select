"""E4: detectability of buried inclusions (fluorescent nodule, absorbing blood vessel) under a collagen cover.

Scene (see E4_scene.py): 16 x 16 mm block, 4 mm dermis_reticular | 7 mm autofluorescent tissue layer (E2 'buried'
optics) | dense-collagen cover of thickness d.  The inclusion's TOP touches the cover/tissue interface, so d is the
depth of the inclusion top below the surface and the whole depth dependence is the attenuation by the cover:
  nodule  sphere r = 0.5, 1, 2 mm of `dysplasia_nodule` (NADH 135 uM; literature tissue optics)
  vessel  cylinder (axis z) r = 0.25, 0.5, 1 mm of `blood_oxygenated`
Lab TLS lines 340/365/400/430 nm + Nuance LCTF camera (31 bands) at the protocol exposures, native 348 x 256 pixels
(0.21 mm/px on the block), 3D layered Monte Carlo with voxel inclusions (0.1 mm voxels).

Contrast: matched no-inclusion capture (same seed; `per_photon_streams=True` common random numbers, installed by
monkey-patching transport3d from here because block_capture does not expose the flag) is subtracted:
    Delta = [mean(ROI) - mean(annulus)]_with  -  [mean(ROI) - mean(annulus)]_without .
Noise: (i) sensor: Poisson + read + dark of the band-summed expected electrons (analytic, validated against
`sensor.detect` realisations); (ii) Monte Carlo: across-seed spread.  Instrument CNR = |mean_seeds Delta| / sensor sigma
(MC averaged down); raw single-capture CNR = |ROI-annulus (no baseline)| / sqrt(sensor^2 + MC^2).

Run:  .venv/bin/python publications/spectralforge_studio/experiments/E4_inclusion_detectability.py [--quick] [--no-figures]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import E4_scene as S  # noqa: E402
from spectraforge.studio import block_capture, capture as cap, optics, sensor as sensor_mod  # noqa: E402
from spectraforge.studio.provenance import provenance  # noqa: E402

ROOT = HERE.parent
FIG = ROOT / "figures"
HALF = 45                       # crop half-width (px)
ANN = (0.5, 2.0)                # annulus: r + 0.5 ... r + 2.0 mm (disc) / lateral offset from the vessel edge (strip)
VESSEL_HALF_LEN = 5.0           # mm: ROI/annulus rows |z| <= 5 mm for the vessel

NODULE_R = [0.5, 1.0, 2.0]
VESSEL_R = [0.25, 0.5, 1.0]
DEPTHS_N = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
DEPTHS_V = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
SEEDS = [0, 1, 2]
PHOTONS = 1_200_000


# ------------------------------------------------------------------------------------------ geometry
def roi_masks(kind, r, shape, cy, cx, pitch):
    """Boolean ROI and annulus masks on the crop (centre (cy, cx) in crop pixels, pitch mm/px)."""
    yy, xx = np.indices(shape)
    dy, dx = (yy - cy) * pitch, (xx - cx) * pitch          # image rows = block z, columns = block x
    if kind == "nodule":
        rho = np.hypot(dx, dy)
        return rho <= r, (rho >= r + ANN[0]) & (rho <= r + ANN[1])
    rows = np.abs(dy) <= VESSEL_HALF_LEN
    ax = np.abs(dx)
    return (ax <= r) & rows, (ax >= r + ANN[0]) & (ax <= r + ANN[1]) & rows


# ------------------------------------------------------------------------------------------ capture + cache
def key(kind, r, d, seed, crn=True):
    return f"{kind or 'base'}_r{r:g}_d{d:g}_s{seed}_{'crn' if crn else 'plain'}"


def run_capture(cache: Path, kind, r, d, seed, photons, excitations, crn=True):
    """Run (or load) one capture; stores crops of the band-summed expected electrons + ROI/annulus spectra."""
    f = cache / (key(kind, r, d, seed, crn) + ".npz")
    if f.exists():
        return dict(np.load(f))
    (S.install_crn if crn else S.uninstall_crn)()
    exp = S.build(kind, r, d, seed=seed, photons=photons, excitations=excitations)
    block_capture.clear_cache()
    t0 = time.time()
    res = cap.capture(exp, scale=1.0)
    gt = res.ground_truth
    blk = np.isfinite(gt["layer_thickness:tissue"])
    ys, xs = np.nonzero(blk)
    cy, cx = 0.5 * (ys.min() + ys.max()), 0.5 * (xs.min() + xs.max())
    pitch = S.FOOT / (xs.max() - xs.min() + 1)
    r0, c0 = int(round(cy)) - HALF, int(round(cx)) - HALF
    out = dict(cy=cy - r0, cx=cx - c0, pitch=pitch, t_s=time.time() - t0)
    for ex in res.excitations_nm:
        E = res.electrons_mean[ex].astype(np.float64)                      # (H, W, B)
        crop = E[r0:r0 + 2 * HALF + 1, c0:c0 + 2 * HALF + 1]
        em_ = np.asarray(res.emission_nm[ex], float)
        out[f"E{ex:g}"] = crop[..., em_ >= ex + 25.0].sum(-1).astype(np.float32)      # fluorescence bands (first band = elastic tail excluded)
        out[f"Eall{ex:g}"] = crop.sum(-1).astype(np.float32)                           # all 31 bands (incl. elastic leak)
        out[f"spec_c{ex:g}"] = crop[HALF - 2:HALF + 3, HALF - 2:HALF + 3].mean((0, 1)).astype(np.float32)
        out[f"em{ex:g}"] = np.asarray(res.emission_nm[ex], float)
        if kind is not None:
            roi, ann = roi_masks(kind, r, crop.shape[:2], out["cy"], out["cx"], pitch)
            out[f"spec_roi{ex:g}"] = crop[roi].mean(0).astype(np.float32)
            out[f"spec_ann{ex:g}"] = crop[ann].mean(0).astype(np.float32)
        out[f"exposure_ms{ex:g}"] = res.exposure_ms[ex]
    np.savez_compressed(f, **out)
    return out


# ------------------------------------------------------------------------------------------ noise model
def pixel_var(E, exposure_ms, sensor, tg, n_bands=31):
    """Variance (electrons^2) of the band-summed detected electrons at a pixel with band-summed expectation E."""
    dark = float(sensor_mod.mean_dark_electrons(sensor, exposure_ms, tg, None))
    var = (E + n_bands * dark) if tg.shot_noise else 0.0 * E
    if tg.read_noise:
        var = var + n_bands * sensor.read_noise_e ** 2
    return var


def contrast_stats(inc, base, kind, r, ex, sensor, tg, bs="fluor"):
    """Quantities of one (seed) pair of crops, for excitation ex."""
    k = f"E{ex:g}" if bs == "fluor" else f"Eall{ex:g}"
    nb = int(np.sum(np.asarray(inc[f"em{ex:g}"]) >= ex + 25.0)) if bs == "fluor" else len(inc[f"em{ex:g}"])
    Ei, Eb = inc[k].astype(np.float64), base[k].astype(np.float64)
    roi, ann = roi_masks(kind, r, Ei.shape, float(inc["cy"]), float(inc["cx"]), float(inc["pitch"]))
    t = float(inc[f"exposure_ms{ex:g}"])
    raw = Ei[roi].mean() - Ei[ann].mean()
    raw_base = Eb[roi].mean() - Eb[ann].mean()
    vi = pixel_var(Ei, t, sensor, tg, nb)
    var_roi, var_ann = vi[roi].mean(), vi[ann].mean()
    return dict(delta=raw - raw_base, raw=raw, raw_base=raw_base, S_roi_base=Eb[roi].mean(), S_ann_base=Eb[ann].mean(),
                S_roi=Ei[roi].mean(), S_ann=Ei[ann].mean(), var_roi=var_roi, var_ann=var_ann,
                n_roi=int(roi.sum()), n_ann=int(ann.sum()),
                sigma_pix=float(np.sqrt(0.5 * (var_roi + var_ann))),
                sigma_roimean=float(np.sqrt(var_roi / roi.sum() + var_ann / ann.sum())),
                ann_diff_std=float(np.std((Ei - Eb)[ann])), ann_diff_mean=float((Ei - Eb)[ann].mean()),
                roi=roi, ann=ann)


def log_interp_crossing(d, c, level=3.0):
    """First depth where the (positive) curve c(d) falls through `level` (log-linear interpolation).
    Returns (value, flag) with flag in {'ok', 'below_dmin', 'above_dmax'}."""
    d, c = np.asarray(d, float), np.asarray(c, float)
    if not np.isfinite(c[0]) or c[0] < level:
        return float("nan"), "below_dmin"
    for i in range(len(d) - 1):
        if c[i] >= level > c[i + 1]:
            if c[i + 1] <= 0:
                return float(d[i]), "ok"
            a, b = np.log(c[i]), np.log(c[i + 1])
            return float(d[i] + (np.log(level) - a) / (b - a) * (d[i + 1] - d[i])), "ok"
    return float("nan"), "above_dmax"


def fit_exp(d, y, dmin=0.5, ymin=None, w=None):
    """OLS of ln y = ln y0 - d/delta on points with d >= dmin and y > ymin (y > 0). Returns dict."""
    d, y = np.asarray(d, float), np.asarray(y, float)
    ok = (d >= dmin) & np.isfinite(y) & (y > (0 if ymin is None else ymin))
    if ok.sum() < 3:
        return dict(delta=np.nan, y0=np.nan, n=int(ok.sum()), r2=np.nan)
    A = np.vstack([np.ones(ok.sum()), -d[ok]]).T
    ly = np.log(y[ok])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    return dict(delta=float(1.0 / coef[1]) if coef[1] > 0 else np.nan, y0=float(np.exp(coef[0])), n=int(ok.sum()),
                r2=float(1 - np.sum((ly - pred) ** 2) / ss_tot) if ss_tot > 0 else np.nan,
                d_lo=float(d[ok].min()), d_hi=float(d[ok].max()))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------ theory
def cover_optics(exp, nm):
    mats = {m.id: m for m in exp.materials}
    mo = optics.material_optics(mats["cover"], optics.merged_fluorophores(exp), np.asarray(nm, float))
    mua, musp = mo.mua_per_mm, mo.mus_per_mm * (1.0 - mo.g)
    return mua, musp, np.sqrt(3.0 * mua * (mua + musp))


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    quick = a.quick
    res_dir = ROOT / "results" / ("E4_quick" if quick else "E4")
    cache = res_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    exc = [365.0] if quick else S.EXC
    dk = {'nodule': [0.5, 1.0, 2.0] if quick else DEPTHS_N, 'vessel': [0.5, 1.0, 2.0] if quick else DEPTHS_V}
    depths = sorted(set(dk['nodule']) | set(dk['vessel']))
    seeds = [0, 1] if quick else SEEDS
    photons = 200_000 if quick else PHOTONS
    nodule_r = [1.0] if quick else NODULE_R
    vessel_r = [0.5] if quick else VESSEL_R
    t_start = time.time()
    cfgs = [("base", 0.0)] + [("nodule", r) for r in nodule_r] + [("vessel", r) for r in vessel_r]
    data = {}
    todo = [(s_, d, kind, r) for s_ in seeds for d in depths for kind, r in cfgs if kind == "base" or d in dk[kind]]
    for n_done, (s_, d, kind, r) in enumerate(todo, 1):
        data[(kind, r, d, s_)] = run_capture(cache, None if kind == "base" else kind, r, d, s_, photons, exc)
        print(f"[{n_done}/{len(todo)}] {kind} r={r:g} d={d:g} seed={s_}  elapsed {time.time() - t_start:.0f}s "
              f"(capture {float(data[(kind, r, d, s_)]['t_s']):.0f}s)", flush=True)

    exp0 = S.build("nodule", 1.0, 1.0, excitations=exc)
    sensor, tg = exp0.cameras[0].sensor, exp0.acquisition.physics

    # ---- analysis
    def analyse(bset):
        results = {}
        for kind, rl in (("nodule", nodule_r), ("vessel", vessel_r)):
            for r in rl:
                for ex in exc:
                    rows = []
                    for d in dk[kind]:
                        per = []
                        for s in seeds:
                            st = contrast_stats(data[(kind, r, d, s)], data[("base", 0.0, d, s)], kind, r, ex, sensor, tg, bset)
                            per.append(st)
                        # MC noise: across-seed std of the incl. crops over ROI + annulus pixels (rms)
                        stack = np.stack([data[(kind, r, d, s)][("E" if bset == "fluor" else "Eall") + f"{ex:g}"].astype(np.float64) for s in seeds])
                        msk = per[0]["roi"] | per[0]["ann"]
                        sd_pix = stack.std(0, ddof=1)[msk]
                        sigma_mc_pix = float(np.sqrt(np.mean(sd_pix ** 2)))
                        rows.append(dict(d=d, per=per, sigma_mc_pix=sigma_mc_pix))
                    results[(kind, r, ex)] = rows

        out = {"nodule": {}, "vessel": {}}
        rng = np.random.default_rng(0)
        for (kind, r, ex), rows in results.items():
            ds = np.array([x["d"] for x in rows])
            nS = len(seeds)

            def agg(fn):
                return np.array([[fn(p) for p in x["per"]] for x in rows])      # (n_depth, n_seed)

            delta = agg(lambda p: p["delta"])
            raw = agg(lambda p: p["raw"])
            sig_pix = agg(lambda p: p["sigma_pix"])
            sig_roi = agg(lambda p: p["sigma_roimean"])
            s_base = agg(lambda p: p["S_roi_base"])
            sigma_mc = np.array([x["sigma_mc_pix"] for x in rows])
            annd = agg(lambda p: p["ann_diff_std"])
            annm = agg(lambda p: p["ann_diff_mean"])

            def curves(idx):
                dl, rw, sp, sr, sb = (delta[:, idx], raw[:, idx], sig_pix[:, idx], sig_roi[:, idx], s_base[:, idx])
                inst = np.abs(dl.mean(1)) / sp.mean(1)
                inst_roi = np.abs(dl.mean(1)) / sr.mean(1)
                rawc = np.abs(rw).mean(1) / np.sqrt(sp.mean(1) ** 2 + sigma_mc ** 2)
                return inst, inst_roi, rawc, np.abs(dl.mean(1)), sb.mean(1)

            inst, inst_roi, rawc, adel, sbase = curves(np.arange(nS))
            # bootstrap over seeds
            bs = {k: [] for k in ("dmax", "dmax_roi", "delta_fit", "delta_sig_fit")}
            floor_c = float(np.sqrt(2.0) * np.sqrt(np.mean(np.array([[p_["raw_base"] for p_ in x["per"]] for x in rows]) ** 2)) / np.sqrt(nS))
            for _ in range(400):
                idx = rng.integers(0, nS, nS)
                ci, cr, _, ad_b, _ = curves(idx)
                bs["delta_sig_fit"].append(fit_exp(ds, ad_b, dmin=0.15, ymin=2.0 * floor_c)["delta"])
                v, f = log_interp_crossing(ds, ci)
                bs["dmax"].append(v if f == "ok" else np.nan)
                v, f = log_interp_crossing(ds, cr)
                bs["dmax_roi"].append(v if f == "ok" else np.nan)
                bs["delta_fit"].append(fit_exp(ds, ci, dmin=0.5, ymin=1.0)["delta"])

            def ci95(v):
                v = np.array(v, float)
                v = v[np.isfinite(v)]
                return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 40 else [np.nan, np.nan]

            dm, flag = log_interp_crossing(ds, inst)
            dmr, flagr = log_interp_crossing(ds, inst_roi)
            fit_c = fit_exp(ds, inst, dmin=0.5, ymin=1.0)
            fit_r = fit_exp(ds, inst_roi, dmin=0.5, ymin=1.0)
            fit_d = fit_exp(ds, adel, dmin=0.15, ymin=2.0 * floor_c)
            fit_s = fit_exp(ds, sbase, dmin=0.5)
            # extrapolated crossing when the curve never falls through 3 inside the sampled range
            def extrap(fit, level=3.0):
                if not np.isfinite(fit.get("delta", np.nan)) or fit["delta"] <= 0:
                    return float("nan")
                return float(fit["delta"] * np.log(fit["y0"] / level))

            # ---- theory
            em = np.asarray(data[(kind, r, ds[0], seeds[0])][f"em{ex:g}"], float)
            d_ref = 1.0 if 1.0 in ds else ds[len(ds) // 2]
            # contrast-spectrum weights: |ROI - annulus| spectrum of the captures with the inclusion (the annulus removes the
            # common tissue background; baseline ROI/annulus spectra are equal by symmetry up to the small ring offset)
            dspec = np.mean([data[(kind, r, d_ref, s)][f"spec_roi{ex:g}"].astype(float) - data[(kind, r, d_ref, s)][f"spec_ann{ex:g}"].astype(float)
                             for s in seeds], axis=0)
            w = np.abs(dspec)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            mua, musp, mueff = cover_optics(exp0, np.concatenate([[ex], em]))
            mu_ex, mu_b = mueff[0], mueff[1:]
            lam_em_eff = float((w * em).sum())
            mu_em_eff = float(np.interp(lam_em_eff, em, mu_b))
            mu_pred_two = float(mu_ex + mu_em_eff)
            # band-resolved prediction on the same window as the signal fit
            s_band = np.array([(w * np.exp(-dd * (mu_ex + mu_b))).sum() for dd in ds])
            fit_band = fit_exp(ds, s_band, dmin=0.5)
            # fluorescence-only (protocol) signal: S_base slope for reference
            res = dict(
                kind=kind, r_mm=r, ex_nm=ex, depths_mm=ds.tolist(), n_seeds=nS,
                delta_e=delta.mean(1).tolist(), delta_e_se=(delta.std(1, ddof=1) / np.sqrt(nS)).tolist(),
                delta_e_seeds=delta.tolist(),
            null_rms_e=float(np.sqrt(np.mean(np.array([[p_['raw_base'] for p_ in x['per']] for x in rows]) ** 2))),
            delta_noise_floor_conservative_e=float(np.sqrt(2.0) * np.sqrt(np.mean(np.array([[p_['raw_base'] for p_ in x['per']] for x in rows]) ** 2)) / np.sqrt(nS)),
            cnr_mc_floor=(delta.std(1, ddof=1) / np.sqrt(nS) / sig_pix.mean(1)).tolist(),
                sign=float(np.sign(np.nanmean(delta[:2].mean(1)))),
                S_roi_base_e=sbase.tolist(),
                sigma_pix_e=sig_pix.mean(1).tolist(), sigma_roimean_e=sig_roi.mean(1).tolist(),
                sigma_mc_pix_e=sigma_mc.tolist(),
                cnr_instrument=inst.tolist(), cnr_instrument_roi=inst_roi.tolist(), cnr_raw_single=rawc.tolist(),
                cnr_instrument_seeds=(np.abs(delta) / sig_pix).tolist(),
                ann_diff_std_e=annd.mean(1).tolist(), ann_diff_mean_e=annm.mean(1).tolist(),
                dmax_mm=dm, dmax_flag=flag, dmax_ci95_mm=ci95(bs["dmax"]), dmax_extrap_mm=extrap(fit_c) if flag == "above_dmax" else None,
                dmax_roi_mm=dmr, dmax_roi_flag=flagr, dmax_roi_ci95_mm=ci95(bs["dmax_roi"]),
                dmax_roi_extrap_mm=extrap(fit_r) if flagr == "above_dmax" else None,
                fit_cnr=fit_c, fit_cnr_roi=fit_r, fit_delta_signal=fit_d, fit_S_base=fit_s,
                delta_fit_cnr_ci95_mm=ci95(bs["delta_fit"]),
                lam_em_eff_nm=lam_em_eff, mu_eff_ex_per_mm=float(mu_ex), mu_eff_em_per_mm=mu_em_eff,
                delta_fit_signal_ci95_mm=ci95(bs["delta_sig_fit"]), delta_pred_em_only_mm=1.0 / mu_em_eff,
            delta_pred_twoway_mm=1.0 / mu_pred_two, delta_pred_ex_only_mm=1.0 / float(mu_ex),
                delta_pred_band_resolved_mm=fit_band["delta"],
                cover_mua_ex_per_mm=float(mua[0]), cover_musp_ex_per_mm=float(musp[0]),
                contrast_spectrum_weights=w.tolist(), em_nm=em.tolist())
            out[kind].setdefault(f"r{r:g}", {})[f"{ex:g}"] = res
        return out

    outs = {bs_: analyse(bs_) for bs_ in ('fluor', 'all')}
    out = outs['fluor']


    # ---- CRN benefit (only full run): plain streams, same seed / independent seed
    crn_block = {}
    if not quick:
        r_c, ex_c = 1.0, 365.0
        for d in (1.0, 3.0):
            row = {}
            inc_p = [run_capture(cache, "nodule", r_c, d, s, photons, exc, crn=False) for s in seeds]
            base_same = [run_capture(cache, None, 0.0, d, s, photons, exc, crn=False) for s in seeds]
            base_ind = [run_capture(cache, None, 0.0, d, s + 1000, photons, exc, crn=False) for s in seeds]
            for name, pairs in (("crn", [(data[("nodule", r_c, d, s)], data[("base", 0.0, d, s)]) for s in seeds]),
                                ("plain_same_seed", list(zip(inc_p, base_same))),
                                ("plain_independent_seed", list(zip(inc_p, base_ind)))):
                sts = [contrast_stats(i, b, "nodule", r_c, ex_c, sensor, tg) for i, b in pairs]
                dl = np.array([x["delta"] for x in sts])
                row[name] = dict(delta_mean_e=float(dl.mean()), delta_sd_across_seeds_e=float(dl.std(ddof=1)),
                                 ann_diff_pixel_std_e=float(np.mean([x["ann_diff_std"] for x in sts])),
                                 ann_diff_mean_e=float(np.mean([x["ann_diff_mean"] for x in sts])),
                                 delta_seeds=dl.tolist())
            crn_block[f"d{d:g}"] = row
            print(f"[crn-benefit] d={d:g}: " + ", ".join(f"{k}: sd(Delta)={v['delta_sd_across_seeds_e']:.2f} ann-pix-std={v['ann_diff_pixel_std_e']:.2f}" for k, v in row.items()), flush=True)

    # ---- sensor-noise validation: detect() realisations vs the analytic variance (one baseline cube)
    nv = {}
    kk, rr, dd = "nodule", 1.0, 1.0
    ex_v = 365.0
    cube_path = cache / "noise_check_cube.npz"
    if cube_path.exists():
        z = np.load(cube_path)
        Ecube, expo = z["E"], float(z["expo"])
    else:
        exp_v = S.build(kk, rr, dd, seed=0, photons=photons, excitations=[ex_v])
        S.install_crn()
        block_capture.clear_cache()
        res_c = cap.capture(exp_v, scale=1.0)
        Ecube, expo = res_c.electrons_mean[ex_v].astype(np.float32), float(res_c.exposure_ms[ex_v])
        np.savez_compressed(cube_path, E=Ecube, expo=expo)
    rng2 = np.random.default_rng(123)
    tg_n = tg.model_copy(update={"fixed_pattern_noise": False, "saturation": False, "quantization": False})
    n_rep = 40
    sums = []
    for _ in range(n_rep):
        adu, _ = sensor_mod.detect(Ecube, expo, sensor, tg_n, None, rng2)
        sums.append(((adu.astype(np.float64) - sensor.offset_adu) * sensor.gain_e_per_adu).sum(-1))
    sums = np.stack(sums)
    Esum = Ecube.astype(np.float64).sum(-1)
    var_emp = sums.var(0, ddof=1)
    var_an = pixel_var(Esum, expo, sensor, tg_n, n_bands=Ecube.shape[-1])
    sel = Esum > 0.2 * Esum.max()
    nv = dict(n_realisations=n_rep, n_pixels=int(sel.sum()), mean_ratio_empirical_over_analytic=float(var_emp[sel].mean() / var_an[sel].mean()),
              median_signal_e=float(np.median(Esum[sel])), median_sigma_e=float(np.sqrt(np.median(var_an[sel]))),
              note="detect() realisations (fixed-pattern, saturation, quantisation off) vs analytic shot+read+dark variance")
    print("[noise-validation]", nv, flush=True)

    settings = dict(
        excitations_nm=exc, depths_mm=dk, seeds=seeds, nodule_radii_mm=nodule_r, vessel_radii_mm=vessel_r,
        photons_per_pass=photons, preview_fraction=1.0, inclusion_voxel_mm=0.1, emission_node_step_nm=40.0,
        capture_scale=1.0, camera="Nuance LCTF, 348x256, lens f=35 mm @330 mm", pixel_mm_on_block=float(data[("nodule", nodule_r[0], dk['nodule'][0], seeds[0])]["pitch"]),
        block_mm=[S.FOOT, S.FOOT, S.H_SUB + S.H_TIS], layers="dermis_reticular 4 | tissue 7 (NADH 60/FAD 40/collagen 250/elastin 40 uM) | collagen_dense cover d",
        nodule_material="dysplasia_nodule", vessel_material="blood_oxygenated", annulus_mm=list(ANN), vessel_half_length_mm=VESSEL_HALF_LEN,
        exposures_ms={f"{e:g}": float(data[("base", 0.0, depths[0], seeds[0])][f"exposure_ms{e:g}"]) for e in exc},
        crn="per_photon_streams=True (monkey-patched into transport3d.excitation_pass/emission_pass)",
        sensor=json.loads(sensor.model_dump_json()), noise_model="shot + read + dark, band-summed over 31 bands, per native pixel",
        cnr_threshold=3.0, fit_window="d >= 0.5 mm and CNR_instrument > 1")
    results_json = dict(experiment="E4 inclusion detectability", quick=quick, settings=settings, results=out, results_all_bands=outs['all'],
                        crn_benefit=crn_block, sensor_noise_validation=nv, runtime_s=time.time() - t_start,
                        mean_capture_s=float(np.mean([float(v["t_s"]) for v in data.values()])),
                        provenance=provenance(),
                        units=dict(depth="mm", delta_e="electrons (band-summed, per native pixel, ROI mean)", cnr="dimensionless", delta_fit="mm"))
    (res_dir / "results.json").write_text(json.dumps(clean(results_json), indent=1))
    print(f"done in {time.time() - t_start:.0f}s -> {res_dir}", flush=True)
    if not a.no_figures:
        import E4_figures
        E4_figures.make(res_dir, FIG if not quick else res_dir)


if __name__ == "__main__":
    main()
