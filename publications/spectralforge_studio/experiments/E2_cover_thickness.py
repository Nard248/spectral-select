"""E2: cover-thickness response of a collagen-covered fluorescent tissue (SpectralForge Studio paper).

Scene (the `tissue_under_collagen` structure rebuilt on the W1 literature tissue materials): a 4 mm reticular-dermis
substrate (`dermis_reticular`), a 1 mm buried NADH/FAD/collagen/elastin-autofluorescent dermis-like layer
(`dermis_papillary` with NADH 60 uM, FAD 40 uM) and a dense-collagen cover (`collagen_dense`, Jacques 2013 optics) of
thickness t = 0 ... 2 mm; the lab's TLS lines (340/365/400/430 nm) + Nuance LCTF camera, 3D layered Monte Carlo.

Analyses
 (a) buried-layer signal S_bur(t) per excitation (sum of the buried layer's dye components over the 31 emission
     bands, from `explain_pixel`), fit S = I0 exp(-t/delta_eff), compared with the two-way diffusion-theory depth
     1/delta = 1/delta(lam_ex) + 1/delta(lam_em), delta = 1/sqrt(3 mua (mua + mus')).
 (b) cover (collagen/elastin) autofluorescence fraction f_cover(t) and its crossover thickness (f = 0.5).
 (c) spectral distortion (fluorescence-only spectrum): emission peak shift, centroid shift, blue/red band ratio;
     repeated with B = 0, 1, 3 % blood in the cover (Hb Soret/Q-band inner filter).
 (d) wedge phantoms (gentle 40 mm and steep 8 mm): local-thickness response vs the analytic local thickness and
     the uniform-block (1D) response, with a fitted lateral-transport Gaussian blur.

The per-pixel quantities come from `capture.explain_pixel` (noise-free expectation of the MC tallies at that pixel,
exactly the engine used by `capture`; one consistency check against `api.capture` is reported). Pixel sets are the
central 3x3 block (uniform blocks) or one image row (wedges); Monte Carlo replicates are seeds 0..2.

Run:  .venv/bin/python publications/spectralforge_studio/experiments/E2_cover_thickness.py [--quick]
Outputs: results/E2/{results.json,runs.parquet,FINDINGS.md(by hand)}, figures/E2_*.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectraforge.studio import api, capture as cap, optics, presets as P
from spectraforge.studio.provenance import provenance
from spectraforge.studio.schema import DyeAmount, LayeredBlockShape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"

EXC = [340.0, 365.0, 400.0, 430.0]
THICK = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
SEEDS = [0, 1, 2]
BLOOD = [0.0, 0.01, 0.03]
BLOOD_T = [0.1, 0.25, 0.5, 1.0]
BLOOD_EXC = [400.0, 430.0]
BLUE, RED = (450.0, 500.0), (550.0, 600.0)
FIT_DYN = 1e-4          # fit window: S/I0 > FIT_DYN (dynamic range of the MC tallies) and t >= FIT_TMIN
FIT_TMIN = 0.05


# ------------------------------------------------------------------------------------------ scene
def build(t: float, t_end: float | None = None, blood: float | None = None, footprint=(40.0, 40.0), seed: int = 0):
    mats = {m.id: m for m in P.builtin_materials()}
    cover = mats["collagen_dense"].model_copy(update={"id": "cover"})
    if blood is not None:
        cover = cover.model_copy(update={"tissue": cover.tissue.model_copy(update={"blood_volume_fraction": blood})})
    buried = mats["dermis_papillary"].model_copy(update={"id": "buried", "dyes": [
        DyeAmount(fluorophore="NADH", concentration_uM=60.0), DyeAmount(fluorophore="FAD", concentration_uM=40.0),
        DyeAmount(fluorophore="collagen", concentration_uM=250.0), DyeAmount(fluorophore="elastin", concentration_uM=40.0)]})
    exp = P.tissue_under_collagen()
    block = LayeredBlockShape(footprint_mm=footprint, layers=[
        P._layer("substrate", "dermis_reticular", 4.0), P._layer("buried", "buried", 1.0),
        P._layer("cover", "cover", t, t_end)])
    seq = [s for s in exp.sources[0].sequence if float(s.wavelength_nm) in EXC]
    src = exp.sources[0].model_copy(update={"sequence": seq})
    exp = exp.model_copy(update={
        "name": "E2 tissue under collagen cover", "materials": list(mats.values()) + [cover, buried],
        "objects": [P._obj("block", "block", block, "cover", (0, 0, 0))], "sources": [src]})
    return api.set(exp, "/acquisition/seed", seed)


def mat_optics(exp, mid: str, nm):
    mats = {m.id: m for m in exp.materials}
    mo = optics.material_optics(mats[mid], optics.merged_fluorophores(exp), np.asarray(nm, float))
    return mo.mua_per_mm, mo.mus_per_mm * (1.0 - mo.g)


def delta_diff(mua, musp):
    return 1.0 / np.sqrt(3.0 * mua * (mua + musp))


# ------------------------------------------------------------------------------------------ extraction
def explain_block(exp, pixels, excitations, scale):
    """Per-excitation component sums (n_bands arrays) averaged over `pixels`; first call runs the MC."""
    out = {f"{e:g}": None for e in excitations}
    n = 0
    for (x, y) in pixels:
        ex = cap.explain_pixel(exp, x, y, scale=scale, excitations=list(excitations))
        n += 1
        for k in out:
            comp = ex["components"][k]
            groups = {"cover": 0.0, "buried": 0.0, "substrate": 0.0}
            for name, arr in comp.items():
                if not name.startswith("dye:"):
                    continue
                layer = name.split("@")[1]
                groups[layer] = groups[layer] + np.asarray(arr)
            groups["em"] = np.asarray(ex["emission_nm"][k])
            groups["elastic"] = np.asarray(comp.get("elastic", np.zeros_like(groups["em"])))
            groups["radiance"] = np.asarray(ex["radiance"][k])
            if out[k] is None:
                out[k] = groups
            else:
                for g in groups:
                    if g != "em":
                        out[k][g] = out[k][g] + groups[g]
    for k in out:
        for g in out[k]:
            if g != "em":
                out[k][g] = np.asarray(out[k][g], float) / n
    return out


def band_sum(em, spec, lo, hi):
    m = (em >= lo) & (em <= hi)
    return float(np.sum(spec[m]))


def spectral_metrics(em, spec):
    spec = np.asarray(spec, float)
    tot = spec.sum()
    if not np.isfinite(tot) or tot <= 0:
        return dict(peak=np.nan, centroid=np.nan, br=np.nan)
    i = int(np.argmax(spec))
    pk = em[i]
    if 0 < i < len(em) - 1:                       # parabolic refinement on the 10 nm grid
        y0, y1, y2 = spec[i - 1], spec[i], spec[i + 1]
        d = y0 - 2 * y1 + y2
        if d < 0:
            pk = em[i] + 0.5 * (y0 - y2) / d * (em[i + 1] - em[i])
    cen = float((em * spec).sum() / tot)
    b, r = band_sum(em, spec, *BLUE), band_sum(em, spec, *RED)
    return dict(peak=float(pk), centroid=cen, br=b / r if r > 0 else np.nan)


def cell(t, seed, excitations, blood=None, pixels=None, scale=None):
    exp = build(t, blood=blood, seed=seed)
    sc = scale or exp.acquisition.preview_scale
    if pixels is None:
        w = max(8, int(round(exp.cameras[0].sensor.width_px * sc)))
        h = max(8, int(round(exp.cameras[0].sensor.height_px * sc)))
        pixels = [(w // 2 + dx, h // 2 + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    t0 = time.time()
    g = explain_block(exp, pixels, excitations, sc)
    rows = []
    for k, d in g.items():
        fl = d["cover"] + d["buried"] + d["substrate"]
        sm = spectral_metrics(d["em"], fl)
        sb = spectral_metrics(d["em"], d["buried"]) if np.sum(d["buried"]) > 0 else dict(peak=np.nan, centroid=np.nan, br=np.nan)
        tot = float(np.sum(fl))
        rows.append(dict(ex=float(k), t=t, seed=seed, blood=np.nan if blood is None else blood,
                         S_buried=float(np.sum(d["buried"])), S_cover=float(np.sum(d["cover"])),
                         S_substrate=float(np.sum(d["substrate"])), S_total=tot, S_elastic=float(np.sum(d["elastic"])),
                         f_cover=float(np.sum(d["cover"])) / tot if tot > 0 else np.nan,
                         peak_nm=sm["peak"], centroid_nm=sm["centroid"], br=sm["br"],
                         buried_centroid_nm=sb["centroid"],
                         em_nm=json.dumps(d["em"].tolist()), spec_total=json.dumps(fl.tolist()),
                         spec_buried=json.dumps(np.asarray(d["buried"]).tolist()),
                         t_cell_s=time.time() - t0))
    return rows


# ------------------------------------------------------------------------------------------ fits
def fit_exp(t, s, se=None, dyn=FIT_DYN, tmin=FIT_TMIN):
    """Weighted LS of ln S = ln I0 - t/delta on the window S/I0 > dyn, t >= tmin. Returns dict."""
    t, s = np.asarray(t, float), np.asarray(s, float)
    i0 = s[0] if t[0] == 0 else np.interp(0.0, t, s)
    ok = (t >= tmin) & (s > dyn * i0) & (s > 0)
    if ok.sum() < 3:
        return dict(delta=np.nan, i0=np.nan, n=int(ok.sum()), t_lo=np.nan, t_hi=np.nan, r2=np.nan)
    y = np.log(s[ok])
    w = np.ones(ok.sum()) if se is None else 1.0 / np.maximum(np.asarray(se)[ok] / s[ok], 1e-3) ** 2
    A = np.vstack([np.ones(ok.sum()), -t[ok]]).T
    W = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    pred = A @ coef
    ss_res, ss_tot = np.sum(w * (y - pred) ** 2), np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return dict(delta=float(1.0 / coef[1]) if coef[1] > 0 else np.nan, i0=float(np.exp(coef[0])), n=int(ok.sum()),
                t_lo=float(t[ok].min()), t_hi=float(t[ok].max()), r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan)


def crossover(t, f, level=0.5):
    """Smallest thickness where the (seed-mean) cover fraction reaches `level` (linear interpolation)."""
    t, f = np.asarray(t, float), np.asarray(f, float)
    for i in range(1, len(t)):
        if f[i - 1] < level <= f[i]:
            return float(t[i - 1] + (level - f[i - 1]) / (f[i] - f[i - 1]) * (t[i] - t[i - 1]))
    return float("nan")


# ------------------------------------------------------------------------------------------ analyses
def analyse_sweep(df, exp0):
    res = {}
    for ex in sorted(df.ex.unique()):
        d = df[df.ex == ex]
        ts = np.array(sorted(d.t.unique()))
        g = d.groupby("t")
        m = g.S_buried.mean().reindex(ts).values
        se = (g.S_buried.std(ddof=1) / np.sqrt(g.S_buried.count())).reindex(ts).values if d.seed.nunique() > 1 else None
        fit = fit_exp(ts, m, se)
        # bootstrap over seeds
        rng = np.random.default_rng(0)
        seeds = sorted(d.seed.unique())
        bs = []
        for _ in range(300):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            mb = np.mean([d[d.seed == s].set_index("t").S_buried.reindex(ts).values for s in pick], axis=0)
            f = fit_exp(ts, mb, None)
            if np.isfinite(f["delta"]):
                bs.append(f["delta"])
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 20 else [np.nan, np.nan]
        # theory
        d0 = d[d.t == 0]
        em = np.array(json.loads(d0.em_nm.iloc[0]))
        spec0 = np.mean([json.loads(x) for x in d0.spec_buried], axis=0)
        lam_em = float((em * spec0).sum() / spec0.sum())
        mua_c, musp_c = mat_optics(exp0, "cover", [ex, lam_em])
        dl = delta_diff(mua_c, musp_c)
        pred_two = 1.0 / (1.0 / dl[0] + 1.0 / dl[1])
        # band-resolved prediction: S(t) = sum_l w_l exp(-t (1/d_ex + 1/d_l)), same fit window as data
        mua_b, musp_b = mat_optics(exp0, "cover", np.concatenate([[ex], em]))
        db = delta_diff(mua_b, musp_b)
        w = spec0 / spec0.sum()
        tt = np.array(ts)
        s_band = np.array([(w * np.exp(-t_ * (1.0 / db[0] + 1.0 / db[1:]))).sum() for t_ in tt])
        fit_band = fit_exp(tt, s_band, None, dyn=FIT_DYN, tmin=FIT_TMIN)
        # window-matched: evaluate the band prediction on the data's fit window
        fb2 = fit_exp(tt, s_band * fit["i0"] if np.isfinite(fit["i0"]) else s_band, None)
        res[f"{ex:g}"] = dict(
            ex_nm=float(ex), I0=fit["i0"], delta_eff_mm=fit["delta"], delta_eff_ci95_mm=ci, fit_n_points=fit["n"],
            fit_window_mm=[fit["t_lo"], fit["t_hi"]], fit_r2=fit["r2"], lam_em_eff_nm=lam_em,
            cover_mua_ex_per_mm=float(mua_c[0]), cover_musp_ex_per_mm=float(musp_c[0]),
            cover_mua_em_per_mm=float(mua_c[1]), cover_musp_em_per_mm=float(musp_c[1]),
            delta_diff_ex_mm=float(dl[0]), delta_diff_em_mm=float(dl[1]),
            delta_pred_twoway_mm=float(pred_two), delta_pred_band_resolved_mm=fit_band["delta"],
            ratio_measured_over_twoway=float(fit["delta"] / pred_two) if np.isfinite(fit["delta"]) else np.nan,
            ratio_measured_over_band=float(fit["delta"] / fit_band["delta"]) if np.isfinite(fit["delta"]) else np.nan,
            S_buried_mean=dict(zip([f"{x:g}" for x in ts], map(float, m))),
            S_buried_se=None if se is None else dict(zip([f"{x:g}" for x in ts], map(float, se))))
        # (b) cover fraction
        fm = g.f_cover.mean().reindex(ts).values
        fse = (g.f_cover.std(ddof=1) / np.sqrt(g.f_cover.count())).reindex(ts).values if d.seed.nunique() > 1 else None
        res[f"{ex:g}"]["f_cover_mean"] = dict(zip([f"{x:g}" for x in ts], map(float, fm)))
        res[f"{ex:g}"]["f_cover_se"] = None if fse is None else dict(zip([f"{x:g}" for x in ts], map(float, fse)))
        res[f"{ex:g}"]["crossover_t_half_mm"] = crossover(ts, fm, 0.5)
        res[f"{ex:g}"]["crossover_t_10pct_mm"] = crossover(ts, fm, 0.1)
        # (c) spectral
        for key in ("peak_nm", "centroid_nm", "br"):
            mm = g[key].mean().reindex(ts).values
            ss = (g[key].std(ddof=1) / np.sqrt(g[key].count())).reindex(ts).values if d.seed.nunique() > 1 else None
            res[f"{ex:g}"][key] = dict(mean=dict(zip([f"{x:g}" for x in ts], map(float, mm))),
                                       se=None if ss is None else dict(zip([f"{x:g}" for x in ts], map(float, ss))))
    return res


def analyse_blood(dfb):
    out = {}
    for ex in sorted(dfb.ex.unique()):
        for b in sorted(dfb.blood.unique()):
            for t in sorted(dfb.t.unique()):
                d = dfb[(dfb.ex == ex) & (dfb.blood == b) & (dfb.t == t)]
                if d.empty:
                    continue
                spec = np.mean([json.loads(x) for x in d.spec_total], axis=0)
                em = np.array(json.loads(d.em_nm.iloc[0]))
                out[f"ex{ex:g}_B{b:g}_t{t:g}"] = dict(
                    ex_nm=float(ex), blood=float(b), t_mm=float(t), em_nm=em.tolist(), spectrum=spec.tolist(),
                    peak_nm=float(d.peak_nm.mean()), centroid_nm=float(d.centroid_nm.mean()),
                    br=float(d.br.mean()), br_se=float(d.br.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                    S_total=float(d.S_total.mean()), S_buried=float(d.S_buried.mean()),
                    S_total_se=float(d.S_total.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None)
    return out


# ------------------------------------------------------------------------------------------ wedge
def wedge_profiles(kind, seed, excitations, scale):
    if kind == "gentle":
        exp = build(0.0, 2.0, footprint=(40.0, 40.0), seed=seed)
    else:
        exp = build(0.0, 2.0, footprint=(8.0, 8.0), seed=seed)
    r = api.capture(exp, "preview", seed=seed, scale=scale, excitations=[excitations[0]])
    gt = np.asarray(r.ground_truth["layer_thickness:cover"], float)
    h, w = gt.shape
    row = h // 2
    xs = np.where(np.isfinite(gt[row]))[0]
    prof = {f"{e:g}": [] for e in excitations}
    for x in xs:
        ex_ = cap.explain_pixel(exp, int(x), int(row), scale=scale, excitations=list(excitations))
        for e in excitations:
            k = f"{e:g}"
            tot = 0.0
            for name, arr in ex_["components"][k].items():
                if name.startswith("dye:"):
                    tot += float(np.sum(arr))
            prof[k].append(tot)
    # pixel pitch in mm along x from the analytic thickness gradient
    t_loc = gt[row, xs]
    return exp, xs, t_loc, {k: np.array(v) for k, v in prof.items()}


def gauss_blur(y, sigma_px):
    if sigma_px < 1e-6:
        return y.copy()
    n = int(np.ceil(4 * sigma_px))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma_px) ** 2)
    k /= k.sum()
    yp = np.pad(y, n, mode="edge")
    return np.convolve(yp, k, mode="valid")


def wedge_analysis(kind, xs, t_loc, prof, sweep_df, excitations):
    out = {}
    pitch = None
    # pixel pitch (mm/px): the cover thickness slope per px divided by the geometric slope
    slope_mm = {"gentle": 2.0 / 40.0, "steep": 2.0 / 8.0}[kind]
    if len(xs) > 3:
        dt = np.polyfit(xs, t_loc, 1)[0]
        pitch = abs(dt) / slope_mm
    for e in excitations:
        k = f"{e:g}"
        d = sweep_df[sweep_df.ex == e]
        ts = np.array(sorted(d.t.unique()))
        sm = d.groupby("t").S_buried.mean().reindex(ts).values
        sf = d.groupby("t").S_total.mean().reindex(ts).values
        # the wedge profile sums ALL dye components (S_total of the uniform sweep)
        from scipy.interpolate import PchipInterpolator
        f1d = PchipInterpolator(ts, np.log(np.maximum(sf, 1e-300)))
        s1d = np.exp(f1d(np.clip(t_loc, ts.min(), ts.max())))
        y = prof[k]
        best = None
        for sig_mm in np.concatenate([[0.0], np.geomspace(0.05, 3.0, 40)]):
            sp = sig_mm / pitch if pitch else 0.0
            pred = gauss_blur(s1d, sp)
            ok = (pred > 0) & (y > 0)
            rms = float(np.sqrt(np.mean((np.log(y[ok]) - np.log(pred[ok])) ** 2)))
            if best is None or rms < best[1]:
                best = (float(sig_mm), rms)
        ok = (s1d > 0) & (y > 0)
        rms0 = float(np.sqrt(np.mean((np.log(y[ok]) - np.log(s1d[ok])) ** 2)))
        out[k] = dict(pitch_mm_per_px=float(pitch) if pitch else None, n_px=int(len(xs)),
                      rms_log_ratio_1D=rms0, sigma_fit_mm=best[0], rms_log_ratio_after_blur=best[1],
                      ratio_wedge_over_1D=(y / s1d).tolist(), t_local_mm=t_loc.tolist(), wedge_signal=y.tolist(),
                      one_d_signal=s1d.tolist())
    return out


def clean(o):
    """JSON-safe copy: numpy scalars/arrays to python, NaN/inf to null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    quick = a.quick
    res_dir = ROOT / "results" / ("E2_quick" if quick else "E2")
    res_dir.mkdir(parents=True, exist_ok=True)
    cache = res_dir / "cache"
    cache.mkdir(exist_ok=True)
    thick = [0.0, 0.1, 0.5, 1.0] if quick else THICK
    seeds = [0] if quick else SEEDS
    exc = [400.0] if quick else EXC
    t_start = time.time()

    def cached(name, fn):
        p = cache / f"{name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
        df = pd.DataFrame(fn())
        df.to_parquet(p)
        return df

    # ---- uniform sweep
    frames = []
    for s in seeds:
        for t in thick:
            frames.append(cached(f"sweep_t{t:g}_s{s}", lambda t=t, s=s: cell(t, s, exc)))
            print(f"[sweep] t={t:g} seed={s}  elapsed {time.time() - t_start:.0f}s", flush=True)
    sweep = pd.concat(frames, ignore_index=True)

    # ---- blood
    bl = []
    if not quick:
        for s in seeds[:2]:
            for b in BLOOD:
                for t in BLOOD_T:
                    bl.append(cached(f"blood_B{b:g}_t{t:g}_s{s}", lambda t=t, s=s, b=b: cell(t, s, BLOOD_EXC, blood=b)))
                    print(f"[blood] B={b:g} t={t:g} seed={s} elapsed {time.time() - t_start:.0f}s", flush=True)
    blood = pd.concat(bl, ignore_index=True) if bl else pd.DataFrame()

    # ---- consistency check explain_pixel vs api.capture (noise-free radiance vs capture radiance, fast mode)
    exp_c = build(0.5, seed=0)
    r = api.capture(exp_c, "fast", seed=0, excitations=[exc[0]])
    k0 = f"{exc[0]:g}"
    rad = r.radiance[exc[0]] if exc[0] in r.radiance else r.radiance[list(r.radiance)[0]]
    rad = np.asarray(rad)                         # (bands, H, W) or (H, W, bands)
    h, w = r.height, r.width
    if rad.shape[0] != h:
        rad = np.moveaxis(rad, 0, -1)
    cx, cy = w // 2, h // 2
    cap_band = rad[cy - 1:cy + 2, cx - 1:cx + 2, :].mean(axis=(0, 1))
    ex_ = explain_block(exp_c, [(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], [exc[0]],
                        exp_c.acquisition.preview_scale)[k0]
    exp_band = ex_["radiance"]
    consistency = dict(rel_diff_band_sum=float(abs(cap_band.sum() - exp_band.sum()) / cap_band.sum()))
    print("[consistency]", consistency, flush=True)

    # ---- wedges
    wedge = {}
    wedge_out = {}
    wex = [400.0] if quick else [400.0, 430.0]
    for kind in (["steep"] if quick else ["gentle", "steep"]):
        per_seed = []
        for s in ([0] if quick else [0, 1]):
            sc = 0.5
            exp_w, xs, t_loc, prof = wedge_profiles(kind, s, wex, sc)
            per_seed.append(wedge_analysis(kind, xs, t_loc, prof, sweep, wex))
            print(f"[wedge] {kind} seed={s} elapsed {time.time() - t_start:.0f}s", flush=True)
        wedge_out[kind] = per_seed

    # ---- analysis
    exp0 = build(0.5)
    sweep_res = analyse_sweep(sweep, exp0)
    blood_res = analyse_blood(blood) if len(blood) else {}
    sweep.drop(columns=["spec_total", "spec_buried", "em_nm"]).to_parquet(res_dir / "runs.parquet")
    if len(blood):
        blood.drop(columns=["spec_total", "spec_buried", "em_nm"]).to_parquet(res_dir / "runs_blood.parquet")
    results = dict(
        experiment="E2 cover-thickness response", quick=quick, units=dict(thickness="mm", delta="mm", wavelength="nm",
                                                                         signal="radiance (engine units, band-summed)"),
        settings=dict(excitations_nm=exc, thicknesses_mm=thick, seeds=seeds, blood_fractions=BLOOD, blood_t_mm=BLOOD_T,
                      blood_excitations_nm=BLOOD_EXC, blue_band_nm=BLUE, red_band_nm=RED,
                      fit_dynamic_range=FIT_DYN, fit_tmin_mm=FIT_TMIN, preview_scale=exp0.acquisition.preview_scale,
                      mc=dict(exp0.acquisition.volume_mc.model_dump()), pixels="central 3x3 (uniform), one row (wedge)"),
        sweep=sweep_res, blood=blood_res, wedge=wedge_out, consistency_explain_vs_capture=consistency,
        runtime_s=time.time() - t_start, provenance=provenance(), n_cells=int(len(sweep)))
    (res_dir / "results.json").write_text(json.dumps(clean(results), indent=1))
    print(f"done in {time.time() - t_start:.0f}s -> {res_dir}")
    if not a.no_figures:
        sys.path.insert(0, str(HERE))
        import E2_figures
        E2_figures.make(res_dir, FIG if not quick else res_dir)


if __name__ == "__main__":
    main()
