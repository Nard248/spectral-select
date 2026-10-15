"""E10: performance and convergence of the 3D layered Monte Carlo (SpectralForge Studio paper).

Scene: the E2 collagen-covered tissue block (uniform 0.5 mm dense-collagen cover over a buried NADH/FAD layer,
40 x 40 mm footprint), one excitation (400 nm), preview resolution (87 x 64 px), so that every measurement is the
cost of ONE excitation pass + the emission passes of one block (``timings_s['mc3d']``).

Photon count: N is the number of packets PER PASS actually launched (excitation pass and each emission wavelength
node).  ``excitation_photons = emission_photons = N / PEEL_PHOTON_FACTOR`` (peel-off) or ``N`` (cone), ``preview_fraction = 1``,
so the estimators are compared at EQUAL packets per pass.

Measurements
  1. runtime and MC noise vs N (peel-off estimator = production default): relative SE of the ROI-mean band-summed
     radiance across seeds, per-pixel relative spatial SD in a plane-detrended ROI; log-log slopes with bootstrap CIs.
  2. thread scaling with ``numba.set_num_threads`` (1, 2, 4, 8, 14), interleaved repetitions, paired speed-ups,
     plus bit-identical-output check across thread counts.
  3. estimators: peel-off vs acceptance-cone (raw, smoothing off; and 'production' cone with the engine's N-dependent
     lateral smoothing): noise at equal packets, runtime, efficiency = 1/(noise^2 x time), bias check.
  4. preview (87x64 px, 10 % photons) vs full (348x256 px, 100 % photons) agreement on a gentle cover wedge.
  5. voxel-inclusion overhead: same scene without / with a sphere (r = 1 mm) / a vessel cylinder (r = 1 mm) / a
     host-filled null sphere (identical machinery, no optical contrast) at equal packets; median paired time ratios.

Timing hygiene: wait for 1-min load < 4 before each block (<= 20 min), JIT warm-up first, interleaved A/B/... repetitions,
reps with wall time > 3x the condition median are discarded for timing (logged), load average recorded per run.

Run:  .venv/bin/python publications/spectralforge_studio/experiments/E10_performance.py [--quick]
Outputs: results/E10/{results.json,runs.parquet,agreement_maps.npz}, figures/E10_*.{pdf,png} (FINDINGS.md by hand).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import E2_cover_thickness as E2   # noqa: E402  (scene builder, json cleaner)
import E4_scene as E4S            # noqa: E402  (scene with voxel inclusions)
from spectraforge.studio import api, block_capture as bc, transport3d as t3   # noqa: E402
from spectraforge.studio.provenance import provenance   # noqa: E402

ROOT = HERE.parent
FIG = ROOT / "figures"
EX = 400.0
ROI = 12            # half-width (px) of the central ROI at preview resolution
SLEEP_FACTOR = 3.0
LOAD_LIMIT = 4.0
LOG: list[str] = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def sysinfo() -> dict:
    def sc(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except Exception:   # noqa: BLE001
            return ""
    return dict(cpu=sc("machdep.cpu.brand_string"), logical_cpus=os.cpu_count(), perf_cores=sc("hw.perflevel0.logicalcpu"),
                eff_cores=sc("hw.perflevel1.logicalcpu"), mem_GB=round(int(sc("hw.memsize") or 0) / 2 ** 30, 1),
                os=platform.platform(), python=platform.python_version(), numba=numba.__version__,
                numpy=np.__version__, threading_layer=str(numba.config.THREADING_LAYER),
                numba_num_threads=numba.get_num_threads(), date=time.strftime("%Y-%m-%d %H:%M"))


def load1() -> float:
    return float(os.getloadavg()[0])


MAX_WAIT = [1200.0]


def wait_load(tag: str, limit=LOAD_LIMIT, max_wait=None, poll=10.0) -> dict:
    max_wait = MAX_WAIT[0] if max_wait is None else max_wait
    t0 = time.time()
    l0 = load1()
    while load1() >= limit and time.time() - t0 < max_wait:
        time.sleep(poll)
    w = time.time() - t0
    rec = dict(tag=tag, load_before=l0, load_after=load1(), waited_s=w, reached=bool(load1() < limit))
    log(f"[load] {tag}: load1 {l0:.1f} -> {rec['load_after']:.1f} after waiting {w:.0f}s (target <{limit}) reached={rec['reached']}")
    return rec


# ---------------------------------------------------------------------------------------------- estimator variants
@contextlib.contextmanager
def variant(v: str):
    """peel = production (next-event); cone_raw = acceptance cone, smoothing off; cone_prod = cone + engine smoothing."""
    old = (bc.PEEL_OFF, bc.smooth_sigma_mm)
    bc.PEEL_OFF = (v == "peel")
    if v == "cone_raw":
        bc.smooth_sigma_mm = lambda n, face="top": 0.0
    try:
        yield
    finally:
        bc.PEEL_OFF, bc.smooth_sigma_mm = old


def scene(photons_per_pass: int, v: str = "peel", t_cover: float = 0.5, seed: int = 0):
    exp = E2.build(t_cover, None, seed=seed)
    pf = bc.PEEL_PHOTON_FACTOR if v == "peel" else 1.0
    n = int(np.clip(round(photons_per_pass / pf), 10_000, 200_000_000))
    vs = exp.acquisition.volume_mc.model_copy(update={"excitation_photons": n, "emission_photons": n, "preview_fraction": 1.0})
    return exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"volume_mc": vs})})


@contextlib.contextmanager
def recorder(store: list):
    """Record per-call seconds / steps-per-photon of the 3D MC passes and the inclusion-map build."""
    o_ex, o_em, o_map = t3.excitation_pass, t3.emission_pass, t3.make_inclusion_map

    def wrap_pass(f, name):
        def g(*a, **k):
            r = f(*a, **k)
            store.append((name, float(r.info.get("seconds", np.nan)), float(r.info.get("mean_steps_per_photon", np.nan))))
            return r
        return g

    def g_map(*a, **k):
        t0 = time.perf_counter()
        r = o_map(*a, **k)
        store.append(("map", time.perf_counter() - t0, np.nan))
        return r
    t3.excitation_pass, t3.emission_pass, t3.make_inclusion_map = wrap_pass(o_ex, "exc"), wrap_pass(o_em, "em"), g_map
    try:
        yield
    finally:
        t3.excitation_pass, t3.emission_pass, t3.make_inclusion_map = o_ex, o_em, o_map


def run(exp, scale=0.25, seed=0):
    bc.clear_cache()
    t0 = time.perf_counter()
    r = api.capture(exp, "fast", seed=seed, excitations=[EX], scale=scale)
    wall = time.perf_counter() - t0
    cube = np.asarray(r.radiance[EX], float)               # (H, W, B)
    return r, cube, dict(wall_s=wall, mc3d_s=float(r.timings_s.get("mc3d", np.nan)), total_s=float(r.timings_s.get("total", np.nan)))


def roi_slice(shape, half=ROI):
    h, w = shape
    hh = min(half, h // 2 - 1)
    return slice(h // 2 - hh, h // 2 + hh), slice(w // 2 - half, w // 2 + half)


def spatial_rel_sd(img):
    """Relative SD of a plane-detrended image patch (per-pixel MC noise in a smooth, uniform region)."""
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    A = np.column_stack([np.ones(img.size), xx.ravel(), yy.ravel()])
    c, *_ = np.linalg.lstsq(A, img.ravel(), rcond=None)
    res = img.ravel() - A @ c
    return float(res.std(ddof=3) / img.mean())


def fit_slope(x, y):
    """OLS slope of log y on log x with t-based standard error. Returns (slope, se, prefactor)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    ok = np.isfinite(ly)
    A = np.vstack([np.ones(ok.sum()), lx[ok]]).T
    coef, *_ = np.linalg.lstsq(A, ly[ok], rcond=None)
    res = ly[ok] - A @ coef
    dof = max(ok.sum() - 2, 1)
    cov = (res ** 2).sum() / dof * np.linalg.inv(A.T @ A)
    return float(coef[1]), float(np.sqrt(cov[1, 1])), float(np.exp(coef[0]))


def t_ci(se, dof, level=0.95):
    from scipy import stats
    return float(stats.t.ppf(0.5 + level / 2, max(dof, 1)) * se)


def boot_slope(groups: dict, stat, nboot=3000, seed=1):
    """groups: {N: array of replicate values}. stat(array)->scalar (e.g. relative SE). Bootstrap over replicates;
    returns (slope, lo, hi) with the 95 % percentile CI of the log-log slope of stat vs N."""
    rng = np.random.default_rng(seed)
    Ns = sorted(groups)
    s0 = fit_slope(Ns, [stat(np.asarray(groups[n])) for n in Ns])[0]
    bs = []
    for _ in range(nboot):
        y = [stat(rng.choice(np.asarray(groups[n]), size=len(groups[n]), replace=True)) for n in Ns]
        if np.isfinite(np.log(np.maximum(y, 1e-300))).all() and min(y) > 0:
            bs.append(fit_slope(Ns, y)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(s0), float(lo), float(hi)


def drop_sleep(df: pd.DataFrame, by: list[str], col="mc3d_s"):
    """Flag reps whose time exceeds SLEEP_FACTOR x the median of their condition. Returns (kept_df, list of dropped dicts)."""
    med = df.groupby(by)[col].transform("median")
    bad = df[col] > SLEEP_FACTOR * med
    dropped = [dict(zip(by, [row[b] for b in by]), **{col: float(row[col])}) for _, row in df[bad].iterrows()]
    for d in dropped:
        log(f"[sleep-artefact] dropped timing rep {d}")
    return df[~bad], dropped


def iqr(x):
    return [float(v) for v in np.percentile(x, [25, 75])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--skip", default="", help="comma list of blocks to skip: conv,threads,incl,full")
    a = ap.parse_args()
    quick = a.quick
    skip = set(a.skip.split(","))
    if quick:
        MAX_WAIT[0] = 0.0
    res_dir = ROOT / "results" / ("E10_quick" if quick else "E10")
    res_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    hw = sysinfo()
    log(str(hw))
    rows = []
    loads = []

    # JIT warm-up for every code path used (peel, cone, inclusions, per-thread-count)
    loads.append(wait_load("warmup"))
    for v in ("peel", "cone_raw"):
        with variant(v):
            run(scene(10_000, v), seed=99)
    run(E4S.build("nodule", 1.0, 0.5, photons=50_000, excitations=[EX]), seed=99)
    run(E4S.build("vessel", 1.0, 0.5, photons=50_000, excitations=[EX]), seed=99)
    log(f"warm-up done {time.time() - t_start:.0f}s")

    # ---------------- 1+3: noise, runtime vs N, three estimator variants, interleaved over (seed, N, variant)
    Ns = [10_000, 20_000, 40_000] if quick else [10_000, 20_000, 40_000, 80_000, 160_000, 320_000, 640_000]
    nseed = 3 if quick else 8
    variants = ["peel", "cone_raw", "cone_prod"]
    if "conv" not in skip:
        loads.append(wait_load("convergence"))
        for s in range(nseed):
            for N in Ns:
                for v in variants:
                    with variant(v):
                        _, cube, tm = run(scene(N, v), seed=s)
                    img = cube.sum(axis=2)
                    patch = img[roi_slice(img.shape)]
                    rows.append(dict(kind="noise", variant=v, N=N, seed=s, roi_mean=float(patch.mean()),
                                     pix_rel_sd=spatial_rel_sd(patch), load1=load1(), threads=numba.get_num_threads(), **tm))
            log(f"[noise] seed {s} done, elapsed {time.time() - t_start:.0f}s, load {load1():.1f}")

    # ---------------- 2: thread scaling (interleaved reps) + identical output
    thread_list = [1, 2] if quick else [1, 2, 4, 8, 14]
    Nthr = 20_000 if quick else 80_000
    nrep_thr = 2 if quick else 5
    ref_cube = None
    bit = {nt: [] for nt in thread_list}
    maxdiff = {nt: 0.0 for nt in thread_list}
    if "threads" not in skip:
        loads.append(wait_load("threads"))
        exp = scene(Nthr, "peel")
        for rep in range(nrep_thr):
            for nt in thread_list:
                numba.set_num_threads(nt)
                _, cube, tm = run(exp, seed=0)
                rows.append(dict(kind="threads", variant="peel", N=Nthr, seed=0, threads=nt, rep=rep, load1=load1(), **tm))
                if ref_cube is None:
                    ref_cube = cube
                bit[nt].append(bool(np.array_equal(cube, ref_cube)))
                maxdiff[nt] = max(maxdiff[nt], float(np.max(np.abs(cube - ref_cube))))
            log(f"[threads] rep {rep} elapsed {time.time() - t_start:.0f}s load {load1():.1f}")
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    # ---------------- 5: inclusion overhead (interleaved)
    Nin = 20_000 if quick else 100_000
    nrep_in = 3 if quick else 9
    kinds = [("none", None, False), ("sphere", "nodule", False), ("vessel", "vessel", False), ("sphere_null", "nodule", True)]
    rec_by_kind: dict[str, list] = {k[0]: [] for k in kinds}
    if "incl" not in skip:
        loads.append(wait_load("inclusions"))
        for rep in range(nrep_in):
            for name, kind, null in kinds:
                e = E4S.build(kind, 1.0, 0.5, photons=int(round(Nin / bc.PEEL_PHOTON_FACTOR)), excitations=[EX], null=null)
                store: list = []
                with recorder(store):
                    _, cube, tm = run(e, seed=0)
                rows.append(dict(kind="incl", variant=name, N=Nin, seed=0, rep=rep, load1=load1(),
                                 map_s=sum(x[1] for x in store if x[0] == "map"),
                                 exc_s=sum(x[1] for x in store if x[0] == "exc"), em_s=sum(x[1] for x in store if x[0] == "em"),
                                 n_em_passes=sum(1 for x in store if x[0] == "em"),
                                 steps_exc=float(np.nanmean([x[2] for x in store if x[0] == "exc"] or [np.nan])),
                                 steps_em=float(np.nanmean([x[2] for x in store if x[0] == "em"] or [np.nan])), **tm))
            log(f"[incl] rep {rep} elapsed {time.time() - t_start:.0f}s load {load1():.1f}")

    # ---------------- 4: preview vs full on the gentle wedge
    agree = {}
    if "full" not in skip and not quick:
        loads.append(wait_load("preview_full"))
        full = {}
        for s in (0, 1):
            e = E2.build(0.0, 2.0, seed=s)
            _, cube_p, tm_p = run(e, scale=0.25, seed=s)     # preview resolution, preview photons (default settings)
            rows.append(dict(kind="preview", seed=s, load1=load1(), **tm_p))
            bc.clear_cache()
            t0 = time.perf_counter()
            rf = api.capture(e, "full", seed=s, excitations=[EX])
            tw = time.perf_counter() - t0
            full[s] = (np.asarray(rf.radiance[EX], float).sum(axis=2), cube_p.sum(axis=2))
            rows.append(dict(kind="full", seed=s, wall_s=tw, mc3d_s=float(rf.timings_s.get("mc3d", np.nan)),
                             total_s=float(rf.timings_s.get("total", np.nan)), load1=load1()))
            log(f"[full] seed {s} wall {tw:.0f}s elapsed {time.time() - t_start:.0f}s")
        f0, p0 = full[0]
        f1, p1 = full[1]
        H, W = p0.shape
        fh, fw = f0.shape
        k = fh // H

        def binto(img):
            return img[:H * k, :W * k].reshape(H, k, W, k).mean(axis=(1, 3))
        b0, b1 = binto(f0), binto(f1)
        m = (b0 > 0.02 * b0.max()) & (b1 > 0.02 * b1.max()) & (p0 > 0.02 * p0.max()) & np.isfinite(b0)

        def rel(x, y):
            return float(np.sqrt(np.mean(((x[m] - y[m]) / (0.5 * (x[m] + y[m]))) ** 2)))
        agree = dict(
            n_pixels=int(m.sum()), full_shape=[int(fh), int(fw)], preview_shape=[int(H), int(W)], bin_factor=int(k),
            rms_rel_preview_vs_fullbinned=rel(p0, b0), rms_rel_fullbinned_seed0_vs_seed1=rel(b0, b1),
            rms_rel_preview_seed0_vs_preview_seed1=rel(p0, p1),
            rms_rel_preview_seed1_vs_fullbinned_seed0=rel(p1, b0),
            corr_preview_fullbinned=float(np.corrcoef(p0[m], b0[m])[0, 1]),
            corr_log_preview_fullbinned=float(np.corrcoef(np.log(p0[m]), np.log(b0[m]))[0, 1]),
            sum_ratio_preview_over_full=float(p0[m].sum() / b0[m].sum()),
            mean_rel_diff_preview_vs_full=float(np.mean((p0[m] - b0[m]) / b0[m])),
            note="band-summed radiance (all bands) of excitation 400 nm on a 0-2 mm cover wedge; mask = pixels > 2 % of max")
        np.savez_compressed(res_dir / "agreement_maps.npz", preview=p0, full_binned=b0, full_binned_seed1=b1, mask=m)

    # ======================================================================= analysis
    df = pd.DataFrame(rows)
    df.to_parquet(res_dir / "runs.parquet")
    out = dict(experiment="E10 performance and convergence", quick=quick, hardware=hw,
               scene="E2 block, cover 0.5 mm, ex 400 nm, 87x64 px preview", peel_photon_factor=bc.PEEL_PHOTON_FACTOR,
               units=dict(time="s", N="packets per pass"), load_waits=loads)
    dropped_all = []

    nz = df[df.kind == "noise"]
    if len(nz):
        nz_t, dr = drop_sleep(nz, ["variant", "N"])
        dropped_all += dr
        conv = {}
        for v in variants:
            d, dt = nz[nz.variant == v], nz_t[nz_t.variant == v]
            grp = {int(N): g.roi_mean.values for N, g in d.groupby("N")}
            tab = {}
            for N, g in d.groupby("N"):
                gt = dt[dt.N == N]
                tab[int(N)] = dict(rel_se_roi=float(g.roi_mean.std(ddof=1) / g.roi_mean.mean()), pix_rel_sd=float(g.pix_rel_sd.mean()),
                                   mc3d_s=float(gt.mc3d_s.median()), mc3d_s_iqr=iqr(gt.mc3d_s), n_seeds=int(len(g)), n_timing=int(len(gt)),
                                   load1=float(g.load1.mean()))
            Nv = sorted(tab)
            s_pix, se_pix, _ = fit_slope(d.N, d.pix_rel_sd)         # pooled over all runs (each run has its own pixel SD)
            dof = len(d) - 2
            sl_roi = boot_slope(grp, lambda y: y.std(ddof=1) / y.mean())
            sl_t = fit_slope(Nv, [tab[n]["mc3d_s"] for n in Nv])
            # t = t0 + b N (linear model with a fixed overhead)
            A = np.vstack([np.ones(len(Nv)), Nv]).T
            (t0_, b_), *_ = np.linalg.lstsq(A, np.array([tab[n]["mc3d_s"] for n in Nv]), rcond=None)
            conv[v] = dict(table=tab,
                           slope_pix_rel_sd=dict(value=s_pix, se=se_pix, ci95=[s_pix - t_ci(se_pix, dof), s_pix + t_ci(se_pix, dof)], n_runs=int(len(d))),
                           slope_rel_se_roi=dict(value=sl_roi[0], ci95=[sl_roi[1], sl_roi[2]], method="bootstrap over seeds (3000)"),
                           slope_runtime_loglog=dict(value=sl_t[0], se=sl_t[1], prefactor_s=sl_t[2]),
                           runtime_linear_fit=dict(overhead_s=float(t0_), s_per_1e5_packets=float(b_ * 1e5)),
                           runtime_per_1e5_packets_at_Nmax_s=float(tab[Nv[-1]]["mc3d_s"] / Nv[-1] * 1e5))
        out["convergence"] = conv
        out["N_range_decades"] = float(np.log10(max(Ns) / min(Ns)))

        # estimators at equal packets
        eff = {}
        rng = np.random.default_rng(3)
        for N in sorted(nz.N.unique()):
            e_ = {}
            for v in variants:
                d = nz[(nz.N == N) & (nz.variant == v)]
                dt = nz_t[(nz_t.N == N) & (nz_t.variant == v)]
                e_[v] = dict(pix_rel_sd=float(d.pix_rel_sd.mean()), rel_se_roi=float(d.roi_mean.std(ddof=1) / d.roi_mean.mean()),
                             t=float(dt.mc3d_s.median()))
            for v in ("cone_raw", "cone_prod"):
                e_[f"pix_sd_ratio_{v}_over_peel"] = e_[v]["pix_rel_sd"] / e_["peel"]["pix_rel_sd"]
                e_[f"roi_se_ratio_{v}_over_peel"] = e_[v]["rel_se_roi"] / e_["peel"]["rel_se_roi"]
                e_[f"time_ratio_{v}_over_peel"] = e_[v]["t"] / e_["peel"]["t"]
                # efficiency ratio (peel / cone) = (sigma_c^2 T_c) / (sigma_p^2 T_p): >1 means peel-off is more efficient
                e_[f"efficiency_peel_over_{v}_pix"] = (e_[v]["pix_rel_sd"] ** 2 * e_[v]["t"]) / (e_["peel"]["pix_rel_sd"] ** 2 * e_["peel"]["t"])
                e_[f"efficiency_peel_over_{v}_roi"] = (e_[v]["rel_se_roi"] ** 2 * e_[v]["t"]) / (e_["peel"]["rel_se_roi"] ** 2 * e_["peel"]["t"])
            eff[int(N)] = e_
        out["estimator_comparison"] = eff
        # bootstrap CI of the pixel-noise efficiency ratio at the middle N
        Nb = sorted(nz.N.unique())
        bias = {}
        for v in ("cone_raw", "cone_prod"):
            zs, rels = [], []
            for N in Nb:
                dp, dc = nz[(nz.N == N) & (nz.variant == "peel")], nz[(nz.N == N) & (nz.variant == v)]
                se = np.sqrt(dp.roi_mean.var(ddof=1) / len(dp) + dc.roi_mean.var(ddof=1) / len(dc))
                zs.append(float((dc.roi_mean.mean() - dp.roi_mean.mean()) / se))
                rels.append(float(dc.roi_mean.mean() / dp.roi_mean.mean() - 1.0))
            bias[v] = dict(N=[int(n) for n in Nb], rel_diff_cone_minus_peel=rels, z=zs,
                           mean_rel_diff=float(np.mean(rels)), note="ROI-mean band-summed radiance; z in units of the joint SE of the means")
        out["estimator_bias_check"] = bias

    inc = df[df.kind == "incl"]
    if len(inc):
        # discard sleep-affected rounds: a round is dropped if any of its reps is > 3x condition median
        med = inc.groupby("variant").mc3d_s.transform("median")
        bad_rounds = set(inc[inc.mc3d_s > SLEEP_FACTOR * med].rep)
        for r_ in bad_rounds:
            log(f"[sleep-artefact] dropped inclusion round {r_}")
        inc_k = inc[~inc.rep.isin(bad_rounds)]
        piv = inc_k.pivot(index="rep", columns="variant", values="mc3d_s")
        res_i = dict(N=int(Nin), n_reps=int(len(piv)), dropped_rounds=sorted(int(x) for x in bad_rounds), target_ratio=1.6,
                     time_s={k: dict(median=float(piv[k].median()), iqr=iqr(piv[k])) for k in piv.columns}, ratio_vs_none={})
        for k in piv.columns:
            if k == "none":
                continue
            r_ = piv[k] / piv["none"]
            res_i["ratio_vs_none"][k] = dict(median=float(r_.median()), iqr=iqr(r_), min=float(r_.min()), max=float(r_.max()))
        prof = {}
        for k in piv.columns:
            g = inc_k[inc_k.variant == k]
            prof[k] = dict(map_s=float(g.map_s.median()), exc_s=float(g.exc_s.median()), em_s=float(g.em_s.median()),
                           n_em_passes=int(g.n_em_passes.median()), steps_per_photon_exc=float(g.steps_exc.median()),
                           steps_per_photon_em=float(g.steps_em.median()), mc3d_s=float(g.mc3d_s.median()),
                           other_s=float((g.mc3d_s - g.map_s - g.exc_s - g.em_s).median()))
        res_i["profile_median"] = prof
        res_i["profile_ratio_vs_none"] = {k: {c: (prof[k][c] / prof["none"][c] if prof["none"][c] else None) for c in
                                              ("map_s", "exc_s", "em_s", "steps_per_photon_exc", "steps_per_photon_em", "mc3d_s")}
                                          for k in prof if k != "none"}
        out["inclusion_overhead"] = res_i

    th = df[df.kind == "threads"]
    if len(th):
        th_t, dr = drop_sleep(th, ["threads"])
        dropped_all += dr
        # paired ratios per repetition (round)
        bad = set(th[~th.index.isin(th_t.index)].rep)
        piv = th[~th.rep.isin(bad)].pivot(index="rep", columns="threads", values="mc3d_s")
        tab = {}
        for nt in piv.columns:
            sp = piv[1] / piv[nt]
            tab[int(nt)] = dict(t_median=float(piv[nt].median()), t_iqr=iqr(piv[nt]), speedup_median=float(sp.median()), speedup_iqr=iqr(sp),
                                efficiency_median=float((sp / nt).median()), efficiency_iqr=iqr(sp / nt),
                                load1_mean=float(th[th.threads == nt].load1.mean()))
        out["thread_scaling"] = dict(N=int(Nthr), n_reps_used=int(len(piv)), dropped_rounds=sorted(int(x) for x in bad), table=tab,
                                     bit_identical_all_reps={int(k): bool(all(v)) for k, v in bit.items()},
                                     max_abs_diff_vs_1_thread={int(k): float(v) for k, v in maxdiff.items()})
    out["preview_vs_full"] = agree
    if len(df[df.kind == "full"]):
        out["timings_preview_full"] = dict(preview_mc3d_s=float(df[df.kind == "preview"].mc3d_s.mean()), full_mc3d_s=float(df[df.kind == "full"].mc3d_s.mean()),
                                           preview_total_s=float(df[df.kind == "preview"].total_s.mean()), full_total_s=float(df[df.kind == "full"].total_s.mean()),
                                           note="single runs, not interleaved; load recorded in runs.parquet")
    out["dropped_timing_reps"] = dropped_all
    out["log"] = LOG
    out["runtime_s"] = time.time() - t_start
    out["provenance"] = provenance()
    (res_dir / "results.json").write_text(json.dumps(E2.clean(out), indent=1))
    log(f"done {time.time() - t_start:.0f}s")
    if not a.no_figures:
        import E10_figures
        E10_figures.make(res_dir, FIG if not quick else res_dir)


if __name__ == "__main__":
    main()
