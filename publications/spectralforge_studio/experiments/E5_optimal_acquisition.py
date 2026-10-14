"""E5: information-optimal acquisition for a collagen-covered fluorescent tissue (SpectralForge Studio paper).

Instrument: the lab's 8-line TLS (310-430 nm) x Nuance LCTF (31 bands, 420-720 nm) = 248 (excitation, band) channels;
scene: 4 mm dermis / 1 mm buried NADH+FAD+collagen+elastin layer / dense-collagen cover of thickness t (see C3_common).
Everything is computed from the PUBLIC inverse API (`surrogate`, `information`, `inverse`).

Noise models (all labelled in the outputs):
  photon-limited      Poisson-Gaussian camera noise only (eps = 0): the Cramer-Rao bound of W4.
  model-error-limited per-channel variance += (eps S_c)^2 with a systematic relative model/calibration error
                      eps in {0.5, 1, 2, 5 %} and eps_ref = the sim-to-real residual of the calibrated twin (`derive_eps`).
Parameter scenarios:
  thick       thickness alone, absolute signal level known (brief (a));
  thick+amp   thickness with one unknown multiplicative amplitude (illumination), the model the thickness map E6 fits;
  thick+gain  thickness with an unknown gain per excitation (lamp-power calibration of each line);
  joint       (thickness, buried NADH, cover mu_s'(500)) (brief (b)), finite-difference Jacobians (central, common random
              numbers) of the 3D layered Monte Carlo at two cover thicknesses.

Analyses: (1) CRLB(t) for all channels vs eps; (2) greedy+exchange target-CRLB / D-optimal selection for k = 1..30, random
(200 draws, median + IQR), uniform and cost-aware selections, k within 10 % of the full-protocol CRLB and the
acquisition-time saving; (3) the unsupervised channel-selection method of the repository (`compare_with_channel_selection`,
spectral autoencoder); (4) efficiency of the ML estimator: empirical SD of 200 noisy fits against the CRLB, with
photon-only noise and with the systematic term, on an independent simulator truth.

Run:  .venv/bin/python publications/spectralforge_studio/experiments/E5_optimal_acquisition.py [--quick] [--force]
Outputs: results/E5/{results.json,FINDINGS.md(by hand)}, figures/E5_*.{pdf,png}.
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
from spectraforge.studio import information as info  # noqa: E402
from spectraforge.studio import inverse as inv  # noqa: E402
from spectraforge.studio.surrogate import (ChannelTable, apply_params, build_surrogate,  # noqa: E402
                                           simulate_channels)

KMAX = 30
N_RANDOM = 200
TARGET = "target:cover_mm"


# ------------------------------------------------------------------------------------------------ helpers
augment = C.augment
with_eps_fisher = C.with_eps_fisher


def sd_of(fi: info.FisherResult, idx: int, sel=None) -> float:
    if sel is None:
        return float(fi.sd[idx])
    return float(info.crlb_for_channels(fi, list(sel))[idx])


def design_curve(fi: info.FisherResult, criterion: str, idx: int, kmax: int, cand=None):
    """Best k-subset for every k = 1..kmax (greedy forward selection refined by Fedorov exchange) -> sets, sd[idx] per k."""
    sets, sds = [], []
    for k in range(1, kmax + 1):
        d = info.optimal_design(fi, k, criterion, "channels", exhaustive_check=False, candidates=cand)
        sets.append(list(d.selected))
        sds.append(sd_of(fi, idx, d.selected))
    return sets, np.array(sds)


def random_curve(fi, idx, kmax, n, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((n, kmax))
    for k in range(1, kmax + 1):
        for i in range(n):
            out[i, k - 1] = sd_of(fi, idx, rng.choice(fi.channels.n, k, replace=False))
    return out


def uniform_curve(fi, idx, kmax):
    return np.array([sd_of(fi, idx, np.unique(np.round(np.linspace(0, fi.channels.n - 1, k)).astype(int)))
                     for k in range(1, kmax + 1)])


def cost_aware_greedy(fi, idx, kmax_time_s=None, n_max=120):
    """Greedy selection maximising the reduction of the target variance per second of exposure."""
    info_c = fi.channel_info
    P = info_c.shape[1]
    ridge = info.RIDGE * max(float(np.trace(info_c.sum(axis=0))) / P, 1e-300)
    cost = fi.exposure_s
    F = ridge * np.eye(P)
    chosen, times, sds = [], [], []
    free = list(range(fi.channels.n))
    v0 = np.linalg.inv(F)[idx, idx]
    for _ in range(n_max):
        Fs = F[None] + info_c[free]
        v = np.linalg.inv(Fs)[:, idx, idx]
        gain = (v0 - v) / cost[free]
        j = int(np.argmax(gain))
        c = free.pop(j)
        F = F + info_c[c]
        v0 = v[j]
        chosen.append(c)
        times.append((times[-1] if times else 0.0) + cost[c])
        sds.append(float(np.sqrt(np.linalg.inv(F)[idx, idx])))
    return chosen, np.array(times), np.array(sds)


def greedy_unit_curve(fi, idx, nmax=None):
    """Plain greedy forward selection of the target variance, unit cost per channel, all the way to ``nmax`` channels
    (the exchange-refined design curve of ``design_curve`` is only run to k = 30) -> (order, sd[idx] vs k)."""
    info_c = fi.channel_info
    P = info_c.shape[1]
    nmax = nmax or fi.channels.n
    ridge = info.RIDGE * max(float(np.trace(info_c.sum(axis=0))) / P, 1e-300)
    F = ridge * np.eye(P)
    free = list(range(fi.channels.n))
    order, sds = [], []
    for _ in range(nmax):
        v = np.linalg.inv(F[None] + info_c[free])[:, idx, idx]
        j = int(np.argmin(v))
        c = free.pop(j)
        F = F + info_c[c]
        order.append(c)
        sds.append(float(np.sqrt(v[j])))
    return order, np.array(sds)


EXT_K = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 130, 160, 200, 248]


def k_within(sds: np.ndarray, sd_all: float, tol=0.10):
    ok = np.flatnonzero(sds <= (1 + tol) * sd_all)
    return int(ok[0] + 1) if ok.size else None


def protocol_time(fi, sel) -> float:
    return float(fi.exposure_s[list(sel)].sum())


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    q = args.quick
    T0 = time.time()
    C.style()
    tag = "q" if q else "f"
    EXC = [340.0, 365.0] if q else None
    kmax = 8 if q else KMAX
    n_rand = 50 if q else N_RANDOM
    out: dict = {"quick": q, "units": {"cover_mm": "mm", "crlb_sd": "um unless key says _mm", "time": "s"}}
    eps_ref = C.derive_eps()
    EPS = C.EPS_GRID + [eps_ref["eps_ref"]]
    out["eps"] = {"grid": EPS, **eps_ref}

    # ---------------------------------------------------------------- 1-D thickness surrogate (8 excitations x 31 bands)
    exp0, sp, t_grid, sur = C.tissue_surrogate(q)
    out["surrogate"] = {"n_channels": sur.channels.n, "grid_mm": t_grid, "build_s": sur.meta.get("build_s"),
                        "cache": sur.meta.get("cache"), "loo": sur.loo_error(), "n_pix_effective": sur.noise.n_pix,
                        "roi": "central rect, normalized 0.4-0.6 of the image (~234 px at scale 0.25 = 3744 full-res pixels)"}
    print(f"surrogate: {sur.channels.n} ch, {sur.meta.get('cache')}, {time.time() - T0:.0f} s", flush=True)
    ch = sur.channels
    expo = np.array([sur.meta["exposure_ms"][f"{e:g}"] * 1e-3 for e in ch.ex_nm])
    out["protocol"] = {"n_channels": int(ch.n), "n_excitations": int(len(set(ch.ex_nm.tolist()))),
                       "total_exposure_s": float(expo.sum()), "exposure_ms_per_band": sur.meta["exposure_ms"]}

    # ---------------------------------------------------------------- (1) CRLB vs thickness and vs eps (all channels)
    th = np.linspace(0.08, 1.45, 14 if q else 40)
    crlb_t = {}
    for scen in ("thick", "thick+amp", "thick+gain"):
        arr = np.zeros((len(EPS), th.size))
        for j, t in enumerate(th):
            f0 = info.fisher_from_surrogate(sur, [t])
            f1 = f0 if scen == "thick" else augment(f0, "amp" if scen == "thick+amp" else "gain")
            for i, e in enumerate(EPS):
                arr[i, j] = with_eps_fisher(f1, e).sd[0] * 1e3
        crlb_t[scen] = arr
    out["crlb_vs_thickness"] = {"cover_mm": th.tolist(), "eps": EPS,
                                **{f"sd_um_{k}": v.tolist() for k, v in crlb_t.items()}}
    t_mid = 0.6
    fi_mid = info.fisher_from_surrogate(sur, [t_mid])
    out["floor_at_0.6mm"] = {}
    for scen, f1 in (("thick", fi_mid), ("thick+amp", augment(fi_mid, "amp")), ("thick+gain", augment(fi_mid, "gain"))):
        out["floor_at_0.6mm"][scen] = {f"eps={e:g}": float(with_eps_fisher(f1, e).sd[0] * 1e3) for e in EPS}

    # ---------------------------------------------------------------- (2) joint (thickness, NADH, cover mu_s') Fisher by FD
    jt = [0.6] if q else [0.3, 0.9]
    jparams = C.tissue_params(exp0, ("thickness", "nadh", "mus"))
    steps = {"cover_mm": 0.06, "NADH_uM": 12.0, "cover_mus500": 0.4}
    joint = {}
    for t in jt:
        e_t = C.build_scene(t, excitations=EXC)
        joint[t] = C.cached(f"E5_fdjoint_{tag}_t{t}", lambda e_t=e_t, t=t: info.fisher(
            e_t, jparams, excitations="all" if EXC is None else EXC, scale=C.SCALE, step=steps,
            scheme="forward" if q else "central", theta0=[t, 60.0, 2.92]), force=args.force)
        print(f"joint FD at t={t}: {time.time() - T0:.0f} s", flush=True)
    out["joint"] = {}
    for t, fj in joint.items():
        d = {"theta": fj.theta.tolist(), "names": fj.names, "meta": {k: v for k, v in fj.meta.items() if k != "step"}}
        d["photon_limited"] = {"crlb_sd": fj.sd.tolist(), "crlb_rel_sd": fj.rel_sd.tolist(), "identifiable": fj.identifiable,
                               "unit": ["mm", "uM", "1/mm"]}
        d["by_eps"] = {f"{e:g}": {"crlb_sd": with_eps_fisher(fj, e).sd.tolist(),
                                  "crlb_rel_sd": with_eps_fisher(fj, e).rel_sd.tolist()} for e in EPS}
        # FD vs surrogate consistency for thickness alone
        fs = info.fisher_from_surrogate(sur, [t])
        fd1 = info.fisher_from_jacobian(["cover_mm"], [t], fj.jac_e[:, :1], fj.mean_e, fj.noise, fj.channels, fj.exposure_s)
        d["check_fd_vs_surrogate_thickness_only"] = {"surrogate_sd_um": float(fs.sd[0] * 1e3), "fd_sd_um": float(fd1.sd[0] * 1e3),
                                                      "ratio": float(fd1.sd[0] / fs.sd[0])}
        jc = np.corrcoef(fj.jac_e.T / np.linalg.norm(fj.jac_e, axis=0)[:, None])
        d["jacobian_cosines"] = (fj.jac_e.T @ fj.jac_e / np.outer(np.linalg.norm(fj.jac_e, axis=0),
                                                                  np.linalg.norm(fj.jac_e, axis=0))).tolist()
        out["joint"][f"t={t}"] = d

    # ---------------------------------------------------------------- (3) channel selection k = 1..kmax
    fis = {"thick": fi_mid, "thick+amp": augment(fi_mid, "amp")}
    fj_mid = joint[jt[0] if q else 0.9]
    sel_out: dict = {}
    sel_sets: dict = {}
    for scen, f1, crit, idx in (("thick", fis["thick"], TARGET, 0), ("thick+amp", fis["thick+amp"], TARGET, 0),
                                ("joint", fj_mid, TARGET, 0), ("joint_D", fj_mid, "D", 0)):
        sel_out[scen] = {}
        for e in EPS:
            fe = with_eps_fisher(f1, e)
            sets, sds = design_curve(fe, crit, idx, kmax)
            sd_all = float(fe.sd[idx])
            rnd = random_curve(fe, idx, kmax, n_rand, seed=1)
            uni = uniform_curve(fe, idx, kmax)
            kopt = k_within(sds, sd_all)
            krnd = k_within(np.median(rnd, axis=0), sd_all)
            kuni = k_within(uni, sd_all)
            sel_out[scen][f"{e:g}"] = {
                "criterion": crit, "sd_all_um": sd_all * 1e3, "optimal_sd_um_vs_k": (sds * 1e3).tolist(),
                "random_median_um": (np.median(rnd, axis=0) * 1e3).tolist(),
                "random_q25_um": (np.percentile(rnd, 25, axis=0) * 1e3).tolist(),
                "random_q75_um": (np.percentile(rnd, 75, axis=0) * 1e3).tolist(), "n_random": n_rand,
                "uniform_sd_um": (uni * 1e3).tolist(), "k_within_10pct": {"optimal": kopt, "random_median": krnd, "uniform": kuni},
                "optimal_sets": sets if e in (0.0, 0.02, EPS[-1]) else None}
            if kopt:
                T_all = protocol_time(fe, range(fe.channels.n))
                T_sel = protocol_time(fe, sets[kopt - 1])
                sel_out[scen][f"{e:g}"]["acquisition"] = {
                    "k": kopt, "exposure_time_all_s": T_all, "exposure_time_selected_s": T_sel,
                    "time_saving_fraction": 1 - T_sel / T_all, "channel_saving_fraction": 1 - kopt / fe.channels.n,
                    "excitations_used": sorted(set(fe.channels.ex_nm[sets[kopt - 1]].tolist())),
                    "labels": [fe.channels.labels()[i] for i in sets[kopt - 1]]}
            # cost-aware greedy (information per second)
            if scen in ("thick", "thick+amp"):
                ch_c, tt, sdc = cost_aware_greedy(fe, idx, n_max=min(60, fe.channels.n))
                ok = np.flatnonzero(sdc <= 1.1 * sd_all)
                sel_out[scen][f"{e:g}"]["cost_aware"] = {
                    "k_within_10pct": int(ok[0] + 1) if ok.size else None,
                    "time_s_within_10pct": float(tt[ok[0]]) if ok.size else None,
                    "time_saving_fraction": float(1 - tt[ok[0]] / protocol_time(fe, range(fe.channels.n))) if ok.size else None,
                    "labels": [fe.channels.labels()[i] for i in ch_c[:(ok[0] + 1 if ok.size else 8)]]}
            if scen == "thick" and e in (0.0, 0.02, EPS[-1]):
                sel_sets[e] = sets
        print(f"selection {scen}: {time.time() - T0:.0f} s", flush=True)
    out["selection"] = sel_out

    # extended-k selection (greedy, k up to all channels): how many channels reach within 10 % of the full protocol when
    # the floor is set by systematic error rather than by photons?  Random / uniform baselines on the same k grid.
    ext: dict = {}
    for scen, f1 in (("thick+amp", fis["thick+amp"]), ("joint", fj_mid)):
        ext[scen] = {}
        for e in EPS:
            fe = with_eps_fisher(f1, e)
            order, sds = greedy_unit_curve(fe, 0, None if not q else 40)
            sd_all = float(fe.sd[0])
            ks = [k for k in EXT_K if k <= len(order)]
            rng = np.random.default_rng(2)
            rmed = []
            for k in ks:
                rmed.append(float(np.median([sd_of(fe, 0, rng.choice(fe.channels.n, k, replace=False)) for _ in range(40 if not q else 10)])))
            rmed = np.array(rmed)
            uni = np.array([sd_of(fe, 0, np.unique(np.round(np.linspace(0, fe.channels.n - 1, k)).astype(int))) for k in ks])
            kg = k_within(sds, sd_all)
            T_all = protocol_time(fe, range(fe.channels.n))
            ext[scen][f"{e:g}"] = {
                "sd_all_um": sd_all * 1e3, "k_grid": ks, "greedy_sd_um": (sds[np.array(ks) - 1] * 1e3).tolist(),
                "random_median_um": (rmed * 1e3).tolist(), "uniform_um": (uni * 1e3).tolist(),
                "k_within_10pct": {"greedy": kg, "random_median": next((k for k, v in zip(ks, rmed) if v <= 1.1 * sd_all), None),
                                   "uniform": next((k for k, v in zip(ks, uni) if v <= 1.1 * sd_all), None)},
                "k_within_25pct_greedy": k_within(sds, sd_all, 0.25), "k_within_50pct_greedy": k_within(sds, sd_all, 0.5),
                "time_saving_fraction_10pct": (1 - protocol_time(fe, order[:kg]) / T_all) if kg else None,
                "excitations_used_10pct": sorted(set(fe.channels.ex_nm[order[:kg]].tolist())) if kg else None}
        print(f"extended selection {scen}: {time.time() - T0:.0f} s", flush=True)
    out["selection_extended"] = ext

    # how the optimal set changes with eps (k = 8, thick+amp)
    k8 = min(8, kmax)
    chg = {}
    f_ta = fis["thick+amp"]
    sets0 = None
    for e in EPS:
        fe = with_eps_fisher(f_ta, e)
        d = info.optimal_design(fe, k8, TARGET, "channels", exhaustive_check=False)
        s = list(d.selected)
        if sets0 is None:
            sets0 = s
        rel_sens = np.abs(fe.jac_e[:, 0]) / np.maximum(fe.mean_e, 1e-9)       # |d ln S / d t|, 1/mm
        chg[f"{e:g}"] = {"selected": [fe.channels.labels()[i] for i in s], "jaccard_with_eps0": len(set(s) & set(sets0)) / len(set(s) | set(sets0)),
                         "mean_rel_sensitivity_per_mm": float(rel_sens[s].mean()), "mean_abs_sensitivity_e_per_mm": float(np.abs(fe.jac_e[s, 0]).mean()),
                         "mean_signal_e": float(fe.mean_e[s].mean()), "median_excitation_nm": float(np.median(fe.channels.ex_nm[s])),
                         "median_emission_nm": float(np.median(fe.channels.em_nm[s])), "sd_um": float(info.crlb_for_channels(fe, s)[0] * 1e3),
                         "sd_um_if_set_of_eps0": float(info.crlb_for_channels(fe, sets0)[0] * 1e3)}
    chg["all_channels_mean_rel_sensitivity_per_mm"] = float((np.abs(f_ta.jac_e[:, 0]) / np.maximum(f_ta.mean_e, 1e-9)).mean())
    chg["all_channels_median_rel_sensitivity_per_mm"] = float(np.median(np.abs(f_ta.jac_e[:, 0]) / np.maximum(f_ta.mean_e, 1e-9)))
    out["set_vs_eps_k8_thick+amp"] = chg

    # exposure allocation at the k=8 photon-limited optimum, and with eps = 2 %
    exa = {}
    for e in (0.0, 0.02):
        fe = with_eps_fisher(fis["thick"], e)
        d = info.optimal_design(fe, k8, TARGET, "exposure_time", exhaustive_check=False)
        x = d.exposure
        exa[f"{e:g}"] = {"sd_reference_um": x["crlb_sd_reference"][0] * 1e3, "sd_equal_um": x["crlb_sd_equal"][0] * 1e3,
                         "sd_optimised_um": x["crlb_sd_opt"][0] * 1e3, "total_time_s": x["total_time_s"]}
    out["exposure_allocation_k8_thick"] = exa

    # ---------------------------------------------------------------- (4) unsupervised channel selection (spectral AE)
    ae = {}
    if not q:
        try:
            cap = C.get_capture("tissue", 1.0, quick=False)
            spectra = info.spectradata_from_capture(cap, "tissue_wedge")
            for kk in (8,):
                for seed in (0, 1):
                    r = info.compare_with_channel_selection(fis["thick+amp"], spectra, k=kk, method="spectral_ae",
                                                            criterion=TARGET, seed=seed, epochs=60)
                    ae[f"k={kk},seed={seed}"] = {
                        "unsupervised_picks": r["unsupervised"]["channels"], "idx": r["unsupervised"]["indices"],
                        **{f"eps={e:g}": {
                            "unsupervised_sd_um": float(info.crlb_for_channels(with_eps_fisher(fis["thick+amp"], e), r["unsupervised"]["indices"])[0] * 1e3),
                            "optimal_sd_um": sel_out["thick+amp"][f"{e:g}"]["optimal_sd_um_vs_k"][kk - 1],
                            "random_median_um": sel_out["thick+amp"][f"{e:g}"]["random_median_um"][kk - 1],
                            "all_sd_um": sel_out["thick+amp"][f"{e:g}"]["sd_all_um"]} for e in (0.0, 0.02, EPS[-1])}}
            print(f"AE comparison: {time.time() - T0:.0f} s", flush=True)
        except Exception as ex:  # noqa: BLE001 - report, do not abort the experiment
            ae = {"error": repr(ex)}
    out["unsupervised_channel_selection"] = ae

    # ---------------------------------------------------------------- (5) efficiency of the ML estimator
    tt = [0.3, 0.7] if q else [0.25, 0.6, 1.1]
    nreal = 40 if q else 200
    eff = []
    sub_mask = np.zeros(ch.n, bool)
    for t in tt:
        truth = C.cached(f"E5_truth_{tag}_{t}", lambda t=t: simulate_channels(
            apply_params(exp0, [sp], [t], None), None, scale=C.SCALE, excitations=EXC).mean_e, force=args.force)
        f_t = info.fisher_from_surrogate(sur, [t])
        opt_set = info.optimal_design(f_t, k8, TARGET, "channels", exhaustive_check=False).selected
        sub_mask[:] = False
        sub_mask[list(opt_set)] = True
        for name, mask in (("all", np.ones(ch.n, bool)), (f"k{k8}_optimal", sub_mask.copy())):
            s_sub = sur.subset(mask)
            for e in (0.0, 0.02):
                nm_e = s_sub.noise.with_model_error(e)
                rng = np.random.default_rng(11)
                est, ses = [], []
                mu_true = truth[mask]
                for _ in range(nreal):
                    y = inv.simulate_observation(mu_true, s_sub.noise, rng)            # photon / read noise
                    if e > 0:
                        y = y + e * mu_true * rng.standard_normal(mu_true.size)         # systematic term
                    r = inv.fit(y, s_sub, nm_e, x0=[t], irls=6)
                    est.append(r.theta[0]); ses.append(r.se[0])
                est = np.array(est)
                fe = info.fisher_from_surrogate(s_sub, [t], noise=nm_e)
                eff.append({"t_mm": t, "channels": name, "eps": e, "n_real": nreal, "emp_sd_um": float(est.std(ddof=1) * 1e3),
                            "crlb_um": float(fe.sd[0] * 1e3), "ratio": float(est.std(ddof=1) / fe.sd[0]),
                            "bias_um": float((est.mean() - t) * 1e3), "mean_fit_se_um": float(np.mean(ses) * 1e3),
                            "bias_from_independent_sim_truth": "yes (truth = separate simulator evaluation off the surrogate grid)"})
        print(f"efficiency t={t}: {time.time() - T0:.0f} s", flush=True)
    out["efficiency"] = eff

    out["elapsed_s"] = time.time() - T0
    C.write_results("E5", out)
    make_figures(out, sur, fi_mid, fis, sel_sets, sel_out, ae, EPS, q)
    print(f"done in {time.time() - T0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ figures
def make_figures(out, sur, fi_mid, fis, sel_sets, sel_out, ae, EPS, q):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    cm = plt.get_cmap("viridis")
    ecol = {0.0: C.OI["black"], 0.005: C.OI["sky"], 0.01: C.OI["blue"], 0.02: C.OI["green"], 0.05: C.OI["orange"], EPS[-1]: C.OI["verm"]}

    # ---- Fig. E5_precision: CRLB(t) and floor vs eps
    cv = out["crlb_vs_thickness"]
    th = np.array(cv["cover_mm"])
    fig, ax = plt.subplots(1, 3, figsize=(C.DCOL, 2.35), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    a = ax[0]
    for i, e in enumerate(EPS):
        a.plot(th, np.array(cv["sd_um_thick"])[i], color=ecol[e], label=("photon-limited" if e == 0 else f"$\\epsilon$ = {e * 100:g} %"))
    eff = out["efficiency"]
    pts = [r for r in eff if r["channels"] == "all" and r["eps"] == 0.0]
    a.plot([r["t_mm"] for r in pts], [r["emp_sd_um"] for r in pts], "o", mfc="none", color=C.OI["purple"], ms=4, label="empirical SD (ML)")
    a.set_yscale("log"); a.set_xlabel("cover thickness $t$ (mm)"); a.set_ylabel("CRLB SD of $t$ (µm)")
    a.set_title("(a) all 248 channels, $t$ alone", fontsize=8); a.set_ylim(0.004, 40); a.legend(fontsize=5.6, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.4)
    a = ax[1]
    for key, ls, lab in (("thick", "-", "$t$ alone"), ("thick+amp", "--", "+ amplitude"), ("thick+gain", ":", "+ gain / excitation")):
        for e in (0.0, 0.02):
            a.plot(th, np.array(cv[f"sd_um_{key}"])[EPS.index(e)], ls=ls, color=ecol[e], lw=1.2)
    a.plot([], [], "-", color="0.3", label="$t$ alone"); a.plot([], [], "--", color="0.3", label="+ amplitude")
    a.plot([], [], ":", color="0.3", label="+ gain / excitation")
    a.plot([], [], "-", color=ecol[0.0], label="$\\epsilon$ = 0"); a.plot([], [], "-", color=ecol[0.02], label="$\\epsilon$ = 2 %")
    a.set_yscale("log"); a.set_ylim(0.004, 40); a.set_xlabel("cover thickness $t$ (mm)"); a.set_title("(b) nuisance parameters", fontsize=8)
    a.legend(fontsize=5.6, ncol=2, loc="lower right", columnspacing=0.8, handlelength=1.4)
    a = ax[2]
    ee = np.array(EPS) * 100
    fl = out["floor_at_0.6mm"]
    for key, ls, mk in (("thick", "-", "o"), ("thick+amp", "--", "s"), ("thick+gain", ":", "^")):
        v = np.array([fl[key][f"eps={e:g}"] for e in EPS])
        a.plot(np.maximum(ee, 0.1), v, ls=ls, marker=mk, ms=3.5, color="0.15", label=key.replace("thick", "$t$"))
    jf = out["joint"]
    jk = sorted(jf)[-1]
    vj = np.array([jf[jk]["by_eps"][f"{e:g}"]["crlb_sd"][0] * 1e3 for e in EPS])
    a.plot(np.maximum(ee, 0.1), vj, "-", marker="D", ms=3.5, color=C.OI["blue"], label="$t$ in joint (3 par.)")
    a.axhspan(25, 200, color="0.88", lw=0, zorder=0); a.text(0.12, 60, "retrieval errors\n25-200 µm\n(E6, W4)", fontsize=5.8, va="center")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("$\\epsilon$ (%; 0 drawn at 0.1)")
    a.set_ylabel("CRLB SD at $t$ = 0.6 mm (µm)"); a.set_title("(c) precision floor", fontsize=8); a.set_ylim(0.03, 300); a.legend(fontsize=5.6, loc="lower right")
    fig.tight_layout(w_pad=0.8)
    C.savefig(fig, "E5_precision")

    # ---- Fig. E5_design: information map + picks, CRLB(k) photon-limited and model-error-limited
    ch = fi_mid.channels
    exs = sorted(set(ch.ex_nm.tolist())); ems = sorted(set(ch.em_nm.tolist()))
    fig = plt.figure(figsize=(C.DCOL, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.6)
    a = fig.add_subplot(gs[0])
    f_ta = fis["thick+amp"]
    ipc = f_ta.info_per_channel()[:, 0]
    rel = np.abs(f_ta.jac_e[:, 0]) / np.maximum(f_ta.mean_e, 1e-9)
    vals = ipc[ipc > 0]
    norm = LogNorm(vmin=max(vals.max() * 1e-6, vals.min()), vmax=vals.max())
    im = None
    for r, e in enumerate(exs):
        m = np.abs(ch.ex_nm - e) < 1e-6
        em_r = ch.em_nm[m]
        edges = np.concatenate([em_r - 5.0, [em_r[-1] + 5.0]])
        im = a.pcolormesh(edges, [r - 0.5, r + 0.5], np.maximum(ipc[m], 1e-300)[None, :], cmap="cividis", norm=norm, shading="flat")
    a.set_xlim(415, 725); a.set_ylim(-0.5, len(exs) - 0.5)
    for e, mk, col, lab in ((0.0, "o", "black", "picks, photon-limited"), (0.05, "s", C.OI["verm"], "picks, $\\epsilon$ = 5 %")):
        fe = with_eps_fisher(f_ta, e)
        d = info.optimal_design(fe, min(8, len(sel_out["thick"]["0"]["optimal_sd_um_vs_k"])), TARGET, "channels", exhaustive_check=False)
        a.scatter([ch.em_nm[i] for i in d.selected], [exs.index(ch.ex_nm[i]) for i in d.selected], marker=mk, s=20, facecolor="none",
                  edgecolor=col, linewidths=1.0, label=lab)
    a.set_yticks(range(len(exs))); a.set_yticklabels([f"{e:g}" for e in exs]); a.set_xlabel("emission band (nm)"); a.set_ylabel("excitation (nm)")
    a.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=False)
    cb = fig.colorbar(im, ax=a, orientation="horizontal", pad=0.2, fraction=0.05); cb.set_label("information on $t$ (mm$^{-2}$)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    a.set_title("(a) $t$ = 0.6 mm; $k$ = 8 picks", fontsize=8)
    for j, (e, ttl) in enumerate(((0.0, "(b) photon-limited"), (0.02, "(c) model error $\\epsilon$ = 2 %"))):
        a = fig.add_subplot(gs[1 + j])
        s = sel_out["thick+amp"][f"{e:g}"]
        x = out["selection_extended"]["thick+amp"][f"{e:g}"]
        kk = np.arange(1, len(s["optimal_sd_um_vs_k"]) + 1)
        a.fill_between(kk, s["random_q25_um"], s["random_q75_um"], color="0.8", lw=0, label="random IQR ($k \\leq$ 30)")
        a.plot(x["k_grid"], x["random_median_um"], color="0.45", ls="--", label="random median")
        a.plot(x["k_grid"], x["uniform_um"], color=C.OI["sky"], ls=":", label="uniform")
        kg = np.array(x["k_grid"])
        a.plot(kk, s["optimal_sd_um_vs_k"], color=C.OI["blue"], marker="o", ms=2, label="information-optimal")
        m = kg > kk[-1]
        a.plot(kg[m], np.array(x["greedy_sd_um"])[m], color=C.OI["blue"], lw=1.1)
        a.axhline(s["sd_all_um"], color=C.OI["verm"], lw=0.9, label="all 248 channels")
        a.axhline(1.1 * s["sd_all_um"], color=C.OI["verm"], lw=0.6, ls=":")
        kopt = x["k_within_10pct"]["greedy"]
        if kopt:
            yk = np.interp(kopt, kg, x["greedy_sd_um"])
            a.plot([kopt], [yk], "*", color=C.OI["orange"], ms=8, zorder=5, label=f"within 10 %: $k$ = {kopt}")
        aekey = [v for v in ae.values() if isinstance(v, dict) and f"eps={e:g}" in v]
        if aekey:
            mm = np.mean([v[f"eps={e:g}"]["unsupervised_sd_um"] for v in aekey])
            a.plot([8], [mm], "D", color=C.OI["purple"], ms=4, label="unsupervised AE ($k$ = 8)")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("channels $k$"); a.set_title(ttl, fontsize=8)
        if j == 0:
            a.set_ylabel("CRLB SD of $t$ (µm), amplitude free")
        a.legend(fontsize=5.0, loc="upper right")
    C.savefig(fig, "E5_design")


if __name__ == "__main__":
    main()
