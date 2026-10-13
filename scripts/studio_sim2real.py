#!/usr/bin/env python
"""Sim-to-real validation study of the SpectralForge Studio digital twin (W6).

Runs, in this order (whole script < 20 min on a laptop CPU, about 2-3 min in practice):

  0. fast forward model vs a full ``capture`` (target < 3 %), and twin -> twin self-recovery of known parameters
     (noisy full captures, parameter bias and 95 % CI coverage);
  A. collagen calibration (Pepsin classes 2-4, Acetic Acid classes 1-3): nominal vs corrected instrument vs two
     components; figure ``fig27_sim2real_collagen.png`` (measured / simulated / residual EEM, emission slices,
     per-pixel spectral-angle map);
  B. hold-out validation: held-out excitations, held-out samples (leave-one-ROI-out over six collagen ROIs and
     across the two datasets), and absolute prediction of held-out excitations for the radiometrically
     calibrated droplet data; figure ``fig28_sim2real_holdout.png``;
  C. droplet data: per-drop calibration, clustering, comparison with the fluorophore library; figure
     ``fig29_sim2real_drops.png``;
  E. sensitivity of the calibrated emission descriptors to the assumed instrument (LCTF width, long-pass edge,
     QE, LCTF throughput);
  D. scattering identifiability check, and the calibrated fluorophores are written to
     ``src/spectraforge/studio/data/calibrated_fluorophores.json`` (with provenance).

Real data are read in place from ``Data/processed`` (never copied); parts that need them are skipped with a
message if ``Data/`` is absent.  Numbers go to ``docs/studio/figures/sim2real_numbers.json``.

Usage::

    .venv/bin/python scripts/studio_sim2real.py [--only 0,A,B,C,E,D] [--no-bootstrap] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "processed"
FIG_DIR = ROOT / "docs" / "studio" / "figures"

PEPSIN = DATA / "Collagen Pepsin"
ACETIC = DATA / "Collagen_Acetic_Acid"
DROPS = DATA / "Drop Data Radiometric" / "rad_merged"
DROP_LABELS = DATA / "Drop Data Cropped" / "drop_labels.npy"

SEED = 20260711


def _tick(label: str, t0: float) -> float:
    print(f"[{time.perf_counter() - t0:7.1f}s] {label}", flush=True)
    return time.perf_counter()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def cal_summary(s) -> dict:
    from spectraforge.studio import calibration as cal
    return cal.emission_summary(s)


def spec_row(fr) -> dict:
    """Parameters (+ 95 % CI where the parameter was free) of every component of a fit."""
    rows = []
    names = fr.param_names
    for k, s in enumerate(fr.specs):
        r = {"em_peak_nm": s.em_peak_nm, "em_fwhm_nm": s.em_fwhm_nm, "em_skew": s.em_skew,
             "ex_peak_nm": s.ex_peak_nm, "ex_fwhm_nm": s.ex_fwhm_nm, "share": fr.relative_brightness[k]}
        for nme, val, ci in zip(names, fr.param_values, fr.ci95):
            if nme.startswith(f"c{k}.") and not nme.endswith("log_amp"):
                r[nme.split(".", 1)[1] + "_ci95"] = None if not np.isfinite(ci) else float(ci)
        r["summary"] = cal_summary(s)
        rows.append(r)
    return {"components": rows, "r2": fr.diagnostics["r2"], "sam_mean_deg": fr.diagnostics["sam_mean_deg"],
            "sam_deg": fr.diagnostics["sam_deg"], "rms_rel": fr.diagnostics["rms_rel"], "bic": fr.diagnostics["bic"],
            "n_obs": fr.n_obs, "n_params": fr.n_params, "gains": fr.gains,
            "starts_within_1pct": fr.diagnostics["n_starts_within_1pct"], "n_starts": len(fr.start_costs)}


# ============================================================================================
# 0. verification of the fast model + self-recovery
# ============================================================================================
def part0(res: dict, t0: float) -> None:
    from spectraforge.studio import analysis, calibration as cal, capture as cap
    from spectraforge.studio.schema import FluorophoreSpec

    out = {}
    spec = FluorophoreSpec(name="truth", ex_peak_nm=350.0, ex_fwhm_nm=70.0, em_peak_nm=470.0, em_fwhm_nm=80.0,
                           em_skew=0.3, vibronic=0.1)
    v = cal.verify_fast_model(spec, excitations=(310.0, 340.0, 365.0, 385.0, 400.0))
    out["fast_vs_full_capture"] = v
    print(f"   fast model vs full capture: max {100 * v['max_rel_dev']:.3f} %, mean {100 * v['mean_rel_dev']:.3f} %")

    # twin -> twin recovery with the detector chain on
    exs = (310.0, 325.0, 340.0, 365.0, 385.0, 400.0, 415.0, 430.0)
    truth = FluorophoreSpec(name="truth", ex_peak_nm=350.0, ex_fwhm_nm=70.0, em_peak_nm=470.0, em_fwhm_nm=80.0,
                            em_skew=0.3)
    start = truth.model_copy(update=dict(ex_peak_nm=330.0, ex_fwhm_nm=50.0, em_peak_nm=500.0, em_fwhm_nm=60.0,
                                         em_skew=0.0))
    rec = {"global": [], "per_excitation": []}
    cover = {"global": {}, "per_excitation": {}}
    n_rep = 8
    for exposure, tag in ((100.0, "high_signal"), (4.0, "low_signal")):
        rec = {"global": [], "per_excitation": []}
        for rep in range(n_rep):
            exp = cal.thin_layer_experiment(truth, excitations=exs, concentration_uM=0.5, noise=True,
                                            sensor_px=(48, 36), exposure_ms=exposure, seed=1000 + rep)
            r = cap.capture(exp)
            dk = cap.capture(exp, mode="dark")
            e = analysis.eem(r, {"shape": "disk", "x": 24, "y": 18, "radius_px": 12},
                             {"mode": "dark_subtracted"}, dark=dk)       # constant P t: gain is global
            for gain in ("global", "per_excitation"):
                fr = cal.fit_fluorophores(e, gain=gain, prior=start, n_starts=4, seed=rep)
                rec[gain].append(fr)
        summ = {}
        for gain, frs in rec.items():
            d = {}
            for pname, true in (("em_peak_nm", truth.em_peak_nm), ("em_fwhm_nm", truth.em_fwhm_nm),
                                ("em_skew", truth.em_skew), ("ex_peak_nm", truth.ex_peak_nm),
                                ("ex_fwhm_nm", truth.ex_fwhm_nm)):
                vals, covered = [], []
                for fr in frs:
                    nm = f"c0.{pname}"
                    if nm not in fr.param_names:
                        continue
                    i = fr.param_names.index(nm)
                    vals.append(fr.param_values[i])
                    covered.append(abs(fr.param_values[i] - true) <= fr.ci95[i])
                if vals:
                    d[pname] = {"true": true, "mean": float(np.mean(vals)), "bias": float(np.mean(vals) - true),
                                "std": float(np.std(vals, ddof=1)), "ci95_coverage": float(np.mean(covered)),
                                "mean_ci95": float(np.mean([f.ci95[f.param_names.index(f'c0.{pname}')] for f in frs]))}
            summ[gain] = d
        out[f"self_recovery_{tag}"] = {"n_repeats": n_rep, "exposure_ms": exposure, "fits": summ}
        g = summ["global"]
        print(f"   self-recovery ({tag}): em_peak bias {g['em_peak_nm']['bias']:+.3f} nm (std {g['em_peak_nm']['std']:.3f}), "
              f"ex_peak bias {g['ex_peak_nm']['bias']:+.3f} nm")
    res["part0"] = out


# ============================================================================================
# A. collagen calibration + fig27
# ============================================================================================
def _fit_set(eem_, tag: str, deg: int = 2, n_starts: int = 16, **kw):
    from spectraforge.studio import calibration as cal
    return cal.fit_fluorophores(eem_, gain="per_excitation", response_degree=deg, n_starts=n_starts, seed=SEED,
                                name=tag, **kw)


def load_collagen():
    from spectraforge.studio import realdata as rd
    cubes = {"pepsin": rd.load_real_cube(PEPSIN, name="Collagen Pepsin"),
             "acetic": rd.load_real_cube(ACETIC, name="Collagen Acetic Acid")}
    eems = {}
    for key, cb in cubes.items():
        for r in cb.roi_names():
            eems[f"{key}:{r}"] = rd.real_roi_eem(cb, r)
    return cubes, eems


def partA(res: dict, t0: float, bootstrap: bool, outdir: Path) -> dict:
    from spectraforge.studio import calibration as cal, realdata as rd

    cubes, eems = load_collagen()
    table = {}
    fits = {}
    for key, e in eems.items():
        m0 = _fit_set(e, "c", deg=0)
        m1 = _fit_set(e, "c", deg=2)
        m2 = _fit_set(e, "c", deg=2, n_components=2)
        # the same corrected model with the shared band ripple
        m3 = _fit_set(e, "c", deg=2, ripple=True)
        fits[key] = {"nominal": m0, "corrected": m1, "two_comp": m2, "ripple": m3}
        table[key] = {"nominal_K1": spec_row(m0), "corrected_K1": spec_row(m1), "corrected_K2": spec_row(m2),
                      "corrected_ripple_K1": spec_row(m3)}
        s = m1.specs[0]
        print(f"   {key:10s} nominal R2={m0.diagnostics['r2']:.3f} SAM={m0.diagnostics['sam_mean_deg']:.1f}  |  "
              f"corrected R2={m1.diagnostics['r2']:.3f} SAM={m1.diagnostics['sam_mean_deg']:.1f}  em_peak={s.em_peak_nm:.0f} "
              f"fwhm={s.em_fwhm_nm:.0f} skew={s.em_skew:.2f} centroid>420={table[key]['corrected_K1']['components'][0]['summary']['centroid_window_nm']:.0f} "
              f"redHM={table[key]['corrected_K1']['components'][0]['summary']['red_half_max_nm']:.0f} |  2c R2={m2.diagnostics['r2']:.3f}")
    res["collagen_fits"] = table

    # excitation-dependent gain table of the main fit (radiometric consistency of the processed units)
    gtab = {}
    for key, fr in ((k, v["corrected"]) for k, v in fits.items()):
        g = fr.gains
        ref = g[340.0]
        gtab[key] = {f"{ex:g}": g[ex] / ref for ex in sorted(g)}
    res["collagen_gain_ratio_vs_340"] = gtab

    if bootstrap:
        bs = cal.bootstrap_ci(cubes["pepsin"], "2", fits["pepsin:2"]["corrected"], n_boot=20, seed=SEED,
                              fit_kwargs=dict(response_degree=2))
        res["collagen_pepsin2_bootstrap"] = bs
        print("   bootstrap (pepsin:2): centroid>420 = %.1f [%.1f, %.1f]  red half-max = %.1f [%.1f, %.1f]" % (
            bs["c0.centroid_window_nm"]["mean"], bs["c0.centroid_window_nm"]["lo95"], bs["c0.centroid_window_nm"]["hi95"],
            bs["c0.red_half_max_nm"]["mean"], bs["c0.red_half_max_nm"]["lo95"], bs["c0.red_half_max_nm"]["hi95"]))
    fig27(cubes["pepsin"], eems["pepsin:2"], fits["pepsin:2"]["corrected"], fits["pepsin:2"]["nominal"],
          outdir / "fig27_sim2real_collagen.png", res)
    return {"cubes": cubes, "eems": eems, "fits": fits}


def fig27(cube, e_meas, fit, fit_nominal, path: Path, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from spectraforge.studio import calibration as cal

    grid = e_meas.grid_emission_nm
    exs = list(e_meas.excitations_nm)
    sim = cal.ThinLayerModel(cal.TwinInstrument.lab_replica(), {ex: e_meas.emission_nm[ex] for ex in exs})
    pred = cal.predict_eem(fit, {ex: e_meas.emission_nm[ex] for ex in exs})
    pred_nom = cal.predict_eem(fit_nominal, {ex: e_meas.emission_nm[ex] for ex in exs})
    e_sim = cal.eem_from_arrays(exs, e_meas.emission_nm, pred, grid=grid)
    e_nom = cal.eem_from_arrays(exs, e_meas.emission_nm, pred_nom, grid=grid)
    peak = np.nanmax(e_meas.grid_value, axis=1, keepdims=True)
    M, S, N = e_meas.grid_value / peak, e_sim.grid_value / peak, e_nom.grid_value / peak
    R = M - S

    fig = plt.figure(figsize=(19, 8.6))
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.15, 1.0], hspace=0.42, wspace=2.6)
    ext = [grid[0], grid[-1], len(exs) - 0.5, -0.5]
    panels = [("measured (Pepsin ROI 2), fraction of excitation peak", M, "viridis", (0, 1)), ("simulated (calibrated twin)", S, "viridis", (0, 1)),
              ("residual: measured - simulated", R, "RdBu_r", (-0.25, 0.25))]
    for j, (title, arr, cmap, (lo, hi)) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3])
        im = ax.imshow(arr, aspect="auto", extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_yticks(range(len(exs)))
        ax.set_yticklabels([f"{e:g}" for e in exs])
        ax.set_xlabel("emission (nm)")
        ax.set_ylabel("excitation (nm)" if j == 0 else "")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax = fig.add_subplot(gs[0, 9:12])
    sam = cal.pixel_sam_map(cube, fit)
    im = ax.imshow(sam, cmap="magma", vmin=0, vmax=30)
    ax.set_title("per-pixel spectral angle to model (deg)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    res.setdefault("fig27", {})["pixel_sam_median_deg"] = float(np.nanmedian(sam))
    res["fig27"]["pixel_sam_mean_deg"] = float(np.nanmean(sam))
    for i, ex in enumerate(exs):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        em = np.asarray(e_meas.emission_nm[ex])
        y = e_meas.value[ex]
        sc = np.nanmax(y)
        ax.errorbar(em, y / sc, yerr=e_meas.std[ex] / sc, fmt="o", ms=3, color="k", lw=0.5, alpha=0.7,
                    label="measured (ROI mean +- sd)")
        ax.plot(em, pred[ex] / sc, "-", color="C3", lw=1.8, label="calibrated twin")
        ax.plot(em, pred_nom[ex] / sc, "--", color="C0", lw=1.0, label="nominal instrument")
        ax.axvline(ex + 20, color="0.6", lw=0.6, ls=":")
        ax.set_title(f"{ex:g} nm excitation   SAM {fit.diagnostics['sam_deg'][ex]:.1f} deg", fontsize=9)
        ax.set_xlabel("emission (nm)")
        if i == 0:
            ax.set_ylabel("normalised")
            ax.legend(fontsize=6.5, loc="upper right")
        ax.set_ylim(0, 1.25)
    d = fit.diagnostics
    s = fit.specs[0]
    fig.suptitle(f"Sim-to-real: collagen (Pepsin, ROI 2).  Calibrated: em {s.em_peak_nm:.0f} nm, FWHM {s.em_fwhm_nm:.0f} nm, "
                 f"skew {s.em_skew:.2f};  R2 = {d['r2']:.3f}, mean SAM = {d['sam_mean_deg']:.1f} deg "
                 f"(nominal instrument: R2 = {fit_nominal.diagnostics['r2']:.3f}, SAM = "
                 f"{fit_nominal.diagnostics['sam_mean_deg']:.1f} deg)", fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {path.name}")


# ============================================================================================
# B. hold-out validation
# ============================================================================================
def _sub_eem(e, exs):
    """EemResult restricted to ``exs`` (shares arrays)."""
    from dataclasses import replace
    exs = [float(x) for x in exs]
    idx = [list(e.excitations_nm).index(x) for x in exs]
    return replace(e, excitations_nm=exs, emission_nm={x: e.emission_nm[x] for x in exs},
                   value={x: e.value[x] for x in exs}, std={x: e.std[x] for x in exs},
                   grid_value=e.grid_value[idx])


def partB(res: dict, t0: float, A: dict, outdir: Path) -> None:
    from spectraforge.studio import calibration as cal, realdata as rd

    eems = A["eems"]
    out = {}
    # ---- B1: held-out excitations ---------------------------------------------------------------
    splits = {"A": ([310.0, 340.0, 385.0], [325.0, 365.0, 400.0]), "B": ([325.0, 365.0, 400.0], [310.0, 340.0, 385.0]),
              "C": ([310.0, 325.0, 340.0], [365.0, 385.0, 400.0]), "D": ([365.0, 385.0, 400.0], [310.0, 325.0, 340.0])}
    configs = {"nominal_K1": dict(deg=0), "corrected_K1": dict(deg=2), "corrected_K2": dict(deg=2, n_components=2),
               "corrected_ripple_K1": dict(deg=2, ripple=True)}
    b1 = {c: [] for c in configs}
    b1_base = {"empirical": [], "flat": []}
    b1_in = []
    for key, e in eems.items():
        for sp, (tr, te) in splits.items():
            etr, ete = _sub_eem(e, tr), _sub_eem(e, te)
            base = cal.baseline_scores(etr, ete)
            b1_base["empirical"].append(base["empirical_sam_deg"])
            b1_base["flat"].append(base["flat_sam_deg"])
            for cname, kw in configs.items():
                fr = _fit_set(etr, "ho", n_starts=8, **kw)
                ev = cal.evaluate_prediction(fr, ete, refit_gains=True)
                b1[cname].append(ev["sam_mean_deg"])
            # in-sample reference on the test excitations (fit directly to them, corrected K1)
            frs = _fit_set(ete, "ho", n_starts=8, deg=2)
            b1_in.append(frs.diagnostics["sam_mean_deg"])
    out["B1_heldout_excitations"] = {
        "n_cases": len(b1_in), "mean_sam_deg": {**{c: float(np.mean(v)) for c, v in b1.items()},
                                                **{f"baseline_{k}": float(np.mean(v)) for k, v in b1_base.items()},
                                                "in_sample_refit_K1": float(np.mean(b1_in))},
        "std_sam_deg": {**{c: float(np.std(v)) for c, v in b1.items()},
                        **{f"baseline_{k}": float(np.std(v)) for k, v in b1_base.items()}}}
    print("   B1 held-out excitations (mean SAM, deg):", {k: round(v, 2) for k, v in out["B1_heldout_excitations"]["mean_sam_deg"].items()})

    # ---- B2: held-out samples (leave-one-ROI-out, calibrate on ONE ROI and predict each other ROI) ------------
    keys = list(eems)
    pair = {c: {"within": [], "across": []} for c in list(configs) + ["baseline_empirical", "baseline_flat", "in_sample"]}
    pair_rows = []
    fit_cache = {}
    for kt in keys:
        for cname, kw in configs.items():
            fit_cache[(kt, cname)] = _fit_set(eems[kt], "ho", n_starts=8, **kw)
    for kt in keys:
        for ks in keys:
            if kt == ks:
                continue
            grp = "within" if kt.split(":")[0] == ks.split(":")[0] else "across"
            base = cal.baseline_scores(eems[kt], eems[ks])
            pair["baseline_empirical"][grp].append(base["empirical_sam_deg"])
            pair["baseline_flat"][grp].append(base["flat_sam_deg"])
            pair["in_sample"][grp].append(fit_cache[(ks, "corrected_K1")].diagnostics["sam_mean_deg"])
            row = {"train": kt, "test": ks, "group": grp, "baseline_empirical": base["empirical_sam_deg"]}
            for cname in configs:
                ev = cal.evaluate_prediction(fit_cache[(kt, cname)], eems[ks], refit_gains=True)
                pair[cname][grp].append(ev["sam_mean_deg"])
                row[cname] = ev["sam_mean_deg"]
            pair_rows.append(row)
    out["B2_heldout_samples"] = {c: {g: float(np.mean(v)) if v else None for g, v in d.items()} for c, d in pair.items()}
    out["B2_pairs"] = pair_rows
    print("   B2 held-out samples (mean SAM, deg):", {c: {g: round(v, 2) for g, v in d.items() if v is not None}
                                                      for c, d in out["B2_heldout_samples"].items()})

    # ---- B3: absolute prediction of held-out excitations (radiometrically calibrated droplets) ----------------
    drops = rd.load_real_cube(DROPS, labels=DROP_LABELS, name="Drop Data (rad_merged)")
    b3_rows = []
    split3 = ([310.0, 340.0, 385.0, 415.0], [325.0, 365.0, 400.0])
    for name in drops.roi_names():
        e = rd.real_roi_eem(drops, name)
        etr, ete = _sub_eem(e, split3[0]), _sub_eem(e, split3[1])
        for deg in (0, 2):
            fr = cal.fit_fluorophores(etr, gain="global", response_degree=deg, n_starts=8, seed=SEED)
            ev = cal.evaluate_prediction(fr, ete, refit_gains=False)
            ev_s = cal.evaluate_prediction(fr, ete, refit_gains=True)
            b3_rows.append({"drop": name, "deg": deg, "r2_abs": ev["r2_abs"], "sam_deg": ev["sam_mean_deg"],
                            "sam_refit_deg": ev_s["sam_mean_deg"],
                            "peak_ratio": ev["peak_ratio_pred_over_meas"],
                            "peak_brightness": float(np.nanmax(e.grid_value))})
    out["B3_heldout_excitations_absolute_drops"] = {
        "train_ex": split3[0], "test_ex": split3[1], "rows": b3_rows,
        "median_r2_abs": {str(d): float(np.median([r["r2_abs"] for r in b3_rows if r["deg"] == d])) for d in (0, 2)},
        "median_sam_deg": {str(d): float(np.median([r["sam_deg"] for r in b3_rows if r["deg"] == d])) for d in (0, 2)},
        "median_abs_log_peak_ratio": {str(d): float(np.median(np.abs(np.log([v for r in b3_rows if r["deg"] == d
                                                                              for v in r["peak_ratio"].values()]))))
                                      for d in (0, 2)}}
    b = out["B3_heldout_excitations_absolute_drops"]
    print("   B3 drops absolute hold-out: median R2_abs", {k: round(v, 3) for k, v in b["median_r2_abs"].items()},
          " median SAM", {k: round(v, 2) for k, v in b["median_sam_deg"].items()},
          " median |ln peak ratio|", {k: round(v, 3) for k, v in b["median_abs_log_peak_ratio"].items()})
    res["holdout"] = out
    fig28(out, outdir / "fig28_sim2real_holdout.png")


def fig28(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    b1 = out["B1_heldout_excitations"]["mean_sam_deg"]
    order = ["in_sample_refit_K1", "corrected_K1", "corrected_ripple_K1", "corrected_K2", "nominal_K1",
             "baseline_empirical", "baseline_flat"]
    lab = ["in-sample\n(refit)", "twin\n(corrected)", "twin\n(+ripple)", "twin\n(2 comp.)", "twin\n(nominal)",
           "empirical\nbaseline", "flat\nbaseline"]
    cols = ["0.5", "C3", "C4", "C1", "C0", "C2", "0.8"]
    ax = axs[0]
    ax.bar(range(len(order)), [b1[o] for o in order], color=cols)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("mean spectral angle on held-out excitations (deg)")
    ax.set_title("B1: calibrate on 3 excitations, predict the other 3\n(6 collagen ROIs x 4 splits)", fontsize=10)
    b2 = out["B2_heldout_samples"]
    ax = axs[1]
    w = 0.38
    for j, grp in enumerate(("within", "across")):
        ax.bar(np.arange(len(order)) + (j - 0.5) * w, [b2[o][grp] if o != "in_sample_refit_K1" else b2["in_sample"][grp] for o in
                                                      ["in_sample" if o == "in_sample_refit_K1" else o for o in order]],
               w, color=cols, alpha=0.55 if j else 1.0, hatch="//" if j else None, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_title("B2: calibrate on one ROI, predict another\nsolid = same dataset, hatched = other dataset", fontsize=10)
    ax = axs[2]
    rows = [r for r in out["B3_heldout_excitations_absolute_drops"]["rows"] if r["deg"] == 2]
    pm, tm = [], []
    for r in rows:
        for ex, ratio in r["peak_ratio"].items():
            pm.append(r["peak_brightness"])
            tm.append(ratio)
    ax.scatter(pm, tm, s=18, c="C3")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("drop peak brightness (relative units)")
    ax.set_ylabel("predicted / measured peak at held-out excitations")
    ax.set_title("B3: droplets, global gain: train 310/340/385/415 nm,\nabsolute prediction at 325/365/400 nm", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {path.name}")


# ============================================================================================
# C. droplets
# ============================================================================================
def partC(res: dict, t0: float, outdir: Path) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage

    from spectraforge.studio import calibration as cal, realdata as rd
    from spectraforge.studio.presets import builtin_fluorophores

    drops = rd.load_real_cube(DROPS, labels=DROP_LABELS, name="Drop Data (rad_merged)")
    names = drops.roi_names()
    rows, fits = {}, {}
    for n in names:
        e = rd.real_roi_eem(drops, n)
        fr = cal.fit_fluorophores(e, gain="global", response_degree=2, n_starts=16, seed=SEED, name=f"drop{n}")
        fits[n] = fr
        s = fr.specs[0]
        rows[n] = spec_row(fr)
        rows[n]["n_pixels"] = int(e.n_pixels)
        rows[n]["peak_brightness"] = float(np.nanmax(e.grid_value))
        print(f"   drop {n:>2s}: ex {s.ex_peak_nm:5.1f}/{s.ex_fwhm_nm:5.1f}  em {s.em_peak_nm:5.1f}/{s.em_fwhm_nm:5.1f} "
              f"skew {s.em_skew:.2f}  R2={fr.diagnostics['r2']:.3f} SAM={fr.diagnostics['sam_mean_deg']:.1f}")
    # data-driven grouping: Ward clustering of the excitation-normalised full spectra (as in the TPAMI analysis)
    feats = []
    for n in names:
        sp = drops.roi_mean_spectra(n)
        v = np.concatenate([sp[ex] / np.linalg.norm(sp[ex]) for ex in drops.excitations_nm if len(sp[ex]) == 31]
                           + [np.array([np.linalg.norm(sp[ex]) for ex in drops.excitations_nm]) /
                              max(np.linalg.norm([np.linalg.norm(sp[ex]) for ex in drops.excitations_nm]), 1e-30)] * 3)
        feats.append(v)
    labels = fcluster(linkage(np.array(feats), "ward"), 3, "maxclust")
    groups = {int(g): [n for n, l in zip(names, labels) if l == g] for g in np.unique(labels)}
    # order groups by brightness: bright / mid / dim
    mean_b = {g: np.mean([rows[n]["peak_brightness"] for n in ns]) for g, ns in groups.items()}
    order = sorted(groups, key=lambda g: -mean_b[g])
    tags = ["bright", "mid", "dim"]
    grp = {tags[i]: groups[g] for i, g in enumerate(order)}
    res["drops_groups"] = grp
    print("   groups:", grp)
    # pooled calibration per group
    pooled, pooled_fit = {}, {}
    lib = builtin_fluorophores()
    for tag, ns in grp.items():
        mask = np.zeros((drops.height, drops.width), bool)
        for n in ns:
            mask |= drops.roi_pixels(n)
        e = rd.real_roi_eem(drops, mask)
        fr = cal.fit_fluorophores(e, gain="global", response_degree=2, n_starts=16, seed=SEED, name=f"drop_{tag}")
        pooled_fit[tag] = (fr, e)
        s = fr.specs[0]
        cand = sorted(lib, key=lambda f: np.hypot(f.ex_peak_nm - s.ex_peak_nm, f.em_peak_nm - s.em_peak_nm))[:2]
        pooled[tag] = {**spec_row(fr), "drops": ns, "n_pixels": int(e.n_pixels),
                       "nearest_library": [{"name": c.name, "ex_peak_nm": c.ex_peak_nm, "em_peak_nm": c.em_peak_nm,
                                            "distance_nm": float(np.hypot(c.ex_peak_nm - s.ex_peak_nm, c.em_peak_nm - s.em_peak_nm))}
                                           for c in cand]}
        print(f"   group {tag:6s} ({len(ns)} drops): ex {s.ex_peak_nm:.0f}/{s.ex_fwhm_nm:.0f} em {s.em_peak_nm:.0f}/{s.em_fwhm_nm:.0f} "
              f"skew {s.em_skew:.2f} R2={fr.diagnostics['r2']:.3f} SAM={fr.diagnostics['sam_mean_deg']:.1f}  nearest library: "
              f"{[c.name for c in cand]}")
    # spread of the per-drop fits inside the bright group
    br = grp["bright"]
    spread = {k: {"mean": float(np.mean([getattr(fits[n].specs[0], k) for n in br])),
                  "std": float(np.std([getattr(fits[n].specs[0], k) for n in br], ddof=1))}
              for k in ("ex_peak_nm", "em_peak_nm", "em_fwhm_nm", "em_skew")}
    res["drops"] = {"per_drop": rows, "pooled": pooled, "bright_group_spread": spread}
    fig29(drops, fits, pooled_fit, grp, outdir / "fig29_sim2real_drops.png")
    # scattering identifiability
    sc = cal.calibrate_scattering(drops, names[0])
    res["scattering"] = {"drops": {"identifiable": sc.identifiable, "reason": sc.reason}}
    return {"drops": drops, "fits": fits, "pooled": pooled_fit, "groups": grp}


def fig29(drops, fits, pooled_fit, grp, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraforge.studio import calibration as cal

    fig, axs = plt.subplots(2, 4, figsize=(19, 9.0))
    fig.subplots_adjust(wspace=0.42, hspace=0.42)
    col = {"bright": "C3", "mid": "C0", "dim": "C2"}
    ax = axs[0, 0]
    for tag, ns in grp.items():
        ax.scatter([fits[n].specs[0].ex_peak_nm for n in ns], [fits[n].specs[0].em_peak_nm for n in ns], c=col[tag], label=f"{tag} ({len(ns)})")
    ax.set_xlabel("calibrated excitation peak (nm)")
    ax.set_ylabel("calibrated emission peak (nm)")
    ax.legend(fontsize=8)
    ax.set_title("per-drop calibration", fontsize=10)
    ax = axs[0, 1]
    for tag, ns in grp.items():
        ax.scatter([fits[n].diagnostics["r2"] for n in ns], [fits[n].diagnostics["sam_mean_deg"] for n in ns], c=col[tag])
    ax.set_xlabel("R2")
    ax.set_ylabel("mean SAM (deg)")
    ax.set_title("fit quality per drop", fontsize=10)
    for k, tag in enumerate(("bright", "mid", "dim")):
        fr, e = pooled_fit[tag]
        exs = list(e.excitations_nm)
        pred = cal.predict_eem(fr, {ex: e.emission_nm[ex] for ex in exs})
        if k == 0:
            ax = axs[0, 2]
        elif k == 1:
            ax = axs[0, 3]
        else:
            ax = axs[1, 0]
        cmap = plt.get_cmap("viridis")
        for i, ex in enumerate(exs):
            c = cmap(i / max(len(exs) - 1, 1))
            ax.plot(e.emission_nm[ex], e.value[ex], "o", ms=3, color=c)
            ax.plot(e.emission_nm[ex], pred[ex], "-", color=c, lw=1.2, label=f"{ex:g}")
        s = fr.specs[0]
        ax.set_title(f"{tag}: {len(grp[tag])} drops, ex {s.ex_peak_nm:.0f}, em {s.em_peak_nm:.0f} nm, R2 {fr.diagnostics['r2']:.3f}", fontsize=9)
        ax.set_xlabel("emission (nm)")
        ax.set_ylabel("radiometric units (global gain)")
        if k == 0:
            ax.legend(fontsize=6, title="excitation (nm)", ncol=2)
    # excitation spectrum of the bright group: measured peak per excitation vs fitted
    ax = axs[1, 1]
    for tag in ("bright", "mid", "dim"):
        fr, e = pooled_fit[tag]
        exs = list(e.excitations_nm)
        meas = [np.nanmax(e.value[ex]) for ex in exs]
        pred = cal.predict_eem(fr, {ex: e.emission_nm[ex] for ex in exs})
        ax.plot(exs, meas, "o", color=col[tag])
        ax.plot(exs, [np.max(pred[ex]) for ex in exs], "-", color=col[tag], label=tag)
    ax.set_xlabel("excitation (nm)")
    ax.set_ylabel("band peak (relative units)")
    ax.set_title("excitation spectra (points: measured)", fontsize=9)
    ax.legend(fontsize=8)
    # residual map of the bright group
    ax = axs[1, 2]
    fr, e = pooled_fit["bright"]
    mask = np.zeros((drops.height, drops.width), bool)
    for n in grp["bright"] + grp["mid"] + grp["dim"]:
        mask |= drops.roi_pixels(n)
    sam = cal.pixel_sam_map(drops, fr, mask)
    im = ax.imshow(sam, cmap="magma", vmin=0, vmax=30)
    ax.set_title("pixel SAM to the 'bright' model (deg)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axs[1, 3]
    for n, fr in fits.items():
        s = fr.specs[0]
        tag = next(t for t, ns in grp.items() if n in ns)
        ax.scatter(s.em_fwhm_nm, s.em_skew, c=col[tag])
    ax.set_xlabel("calibrated emission FWHM (nm)")
    ax.set_ylabel("calibrated skew")
    ax.set_title("emission shape parameters", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {path.name}")


# ============================================================================================
# E. sensitivity of the calibrated descriptors to the assumed instrument
# ============================================================================================
def partE(res: dict, t0: float, A: dict, C: dict) -> None:
    """Re-calibrate under alternative twin-instrument assumptions (LCTF FWHM, long-pass edge, QE curve, LCTF
    throughput curve) and report the spread of the identifiable emission descriptors."""
    from spectraforge.studio import calibration as cal, realdata as rd

    base = cal.TwinInstrument.lab_replica()

    def variant(filt=None, sens=None):
        f = base.camera.filter.model_copy(update=filt or {})
        s_ = base.camera.sensor.model_copy(update=sens or {})
        return cal.TwinInstrument(base.camera.model_copy(update={"filter": f, "sensor": s_}), base.source, base.toggles)

    variants = {"nominal": variant(), "LCTF FWHM 10 nm": variant({"fwhm_nm": 10.0}),
                "LCTF FWHM 14 nm": variant({"fwhm_nm": 14.0}), "long-pass 420 nm": variant({"emission_longpass_nm": 420.0}),
                "long-pass 405 nm": variant({"emission_longpass_nm": 405.0}), "UV-enhanced CCD": variant(sens={"qe_curve": "ccd_uv"}),
                "flat LCTF throughput": variant({"peak_transmission_curve": "flat"})}
    e_p = A["eems"]["pepsin:2"]
    e_b = C["pooled"]["bright"][1]
    out = {}
    for vname, inst in variants.items():
        row = {}
        for lab, e, gain in (("collagen_pepsin2", e_p, "per_excitation"), ("drops_bright", e_b, "global")):
            for deg in (0, 2):
                fr = cal.fit_fluorophores(e, gain=gain, response_degree=deg, n_starts=10, seed=SEED, instrument=inst)
                d = cal.emission_summary(fr.specs[0])
                row[f"{lab}_deg{deg}"] = {"r2": fr.diagnostics["r2"], "sam_mean_deg": fr.diagnostics["sam_mean_deg"],
                                          "em_peak_nm": fr.specs[0].em_peak_nm, "ex_peak_nm": fr.specs[0].ex_peak_nm,
                                          "em_skew": fr.specs[0].em_skew, **d}
        out[vname] = row
    res["instrument_sensitivity"] = out
    for key in ("collagen_pepsin2_deg2", "drops_bright_deg2"):
        phys = [v[key] for k, v in out.items() if k != "flat LCTF throughput"]
        print(f"   {key}: centroid>420 {min(p['centroid_window_nm'] for p in phys):.0f}-{max(p['centroid_window_nm'] for p in phys):.0f} nm, "
              f"red half-max {min(p['red_half_max_nm'] for p in phys):.0f}-{max(p['red_half_max_nm'] for p in phys):.0f} nm "
              f"(excluding the unphysical flat-throughput variant)")


# ============================================================================================
# D. scattering + write the calibrated fluorophores
# ============================================================================================
def partD(res: dict, t0: float, A: dict, C: dict | None) -> None:
    from spectraforge.studio import calibration as cal

    sc = cal.calibrate_scattering(A["cubes"]["pepsin"], "2")
    res.setdefault("scattering", {})["collagen"] = {"identifiable": sc.identifiable, "reason": sc.reason}
    entries = []
    for key, d in A["fits"].items():
        fr = d["corrected"]
        ds, roi = key.split(":")
        label = "Collagen Pepsin" if ds == "pepsin" else "Collagen Acetic Acid"
        spec = fr.specs[0].model_copy(update={"name": f"collagen_cal_{ds}_{roi}"})
        entries.append((spec, {
            "dataset": f"{label} (Data/processed, spectra_masked.pkl)", "roi": f"class {roi} (roi_regions.json)",
            "method": "per-excitation gain, shared smooth throughput correction (degree 2), 1 component, "
                      "12/16-start soft-L1", "instrument": "mehsi_lab_replica (Nuance LCTF + Xe/monochromator 10 nm)",
            "r2": fr.diagnostics["r2"], "sam_mean_deg": fr.diagnostics["sam_mean_deg"], "rms_rel": fr.diagnostics["rms_rel"],
            "n_obs": fr.n_obs, "ci95": dict(zip(fr.param_names, fr.ci95)),
            "notes": "Excitation parameters held at the collagen prior (not identifiable with a per-excitation gain); "
                     "emission peak is only constrained from above because the Nuance window starts at 420 nm."}))
    if C is not None:
        for tag, (fr, e) in C["pooled"].items():
            spec = fr.specs[0].model_copy(update={"name": f"drop_cal_{tag}"})
            entries.append((spec, {
                "dataset": "Drop Data Radiometric rad_merged (Data/processed)", "roi": f"drops {C['groups'][tag]} pooled",
                "method": "global gain (radiometrically calibrated), shared smooth throughput correction (degree 2), "
                          "1 component, 16-start soft-L1", "instrument": "mehsi_lab_replica",
                "r2": fr.diagnostics["r2"], "sam_mean_deg": fr.diagnostics["sam_mean_deg"], "rms_rel": fr.diagnostics["rms_rel"],
                "n_obs": fr.n_obs, "ci95": dict(zip(fr.param_names, fr.ci95)),
                "notes": "dye identity of the droplets is not recorded in the repository (Drop Data is an unlabeled dataset); "
                         "values are effective fluorophore parameters, not a literature dye."}))
    p = cal.write_calibrated_fluorophores(entries, note="Generated by scripts/studio_sim2real.py; see docs/studio/SIM2REAL.md. "
                                                         "Amplitude is not stored (relative data): quantum_yield/extinction are nominal.")
    print(f"   wrote {p.relative_to(ROOT)} ({len(entries)} fluorophores)")
    res["calibrated_fluorophores_written"] = [e[0].name for e in entries]


# ============================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="0,A,B,C,E,D")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--out", default=str(FIG_DIR))
    args = ap.parse_args()
    parts = set(args.only.upper().split(","))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res: dict = {}
    numbers = outdir / "sim2real_numbers.json"
    if numbers.exists() and parts != {"0", "A", "B", "C", "D", "E"}:
        res = json.loads(numbers.read_text())
    if "0" in parts:
        print("== 0. fast model verification and self-recovery")
        part0(res, t0)
        _tick("part 0", t0)
    have_data = DATA.exists()
    if not have_data:
        print("Data/processed not found: real-data parts skipped")
    A = C = None
    if have_data and parts & {"A", "B", "D", "E"}:
        print("== A. collagen calibration")
        A = partA(res, t0, not args.no_bootstrap, outdir)
        _tick("part A", t0)
    if have_data and "B" in parts:
        print("== B. hold-out validation")
        partB(res, t0, A, outdir)
        _tick("part B", t0)
    if have_data and parts & {"C", "D", "E"}:
        print("== C. droplets")
        C = partC(res, t0, outdir)
        _tick("part C", t0)
    if have_data and "E" in parts and A is not None and C is not None:
        print("== E. instrument-assumption sensitivity")
        partE(res, t0, A, C)
        _tick("part E", t0)
    if have_data and "D" in parts:
        print("== D. scattering + calibrated library")
        partD(res, t0, A, C)
        _tick("part D", t0)
    numbers.write_text(json.dumps(_jsonable(res), indent=1))
    print(f"numbers -> {numbers}")
    _tick("done", t0)


if __name__ == "__main__":
    main()
