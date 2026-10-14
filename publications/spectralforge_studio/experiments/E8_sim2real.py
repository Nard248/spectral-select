"""E8: sim-to-real consolidation + calibrated-collagen-in-simulated-cover check (SpectralForge Studio paper).

Part 1 (consolidation): W6's numbers (docs/studio/figures/sim2real_numbers.json) -> one paper figure + one table:
  (a) calibrated collagen twin vs measured emission spectra (Pepsin ROI 2; refit here, 16 starts), (b) hold-out spectral
  angle for held-out excitations (B1) and held-out samples (B2), (c) droplet absolute peak-amplitude prediction at
  held-out excitations (B3).
Part 2 (new): the calibrated collagen emission spectrum replaces the library collagen fluorophore in the 3D layered
Monte Carlo (thick collagen slab and a 0.5 mm collagen cover on dermis, lab TLS + Nuance twin), and the simulated
per-excitation emission shape is compared with the measured one (spectral angle per excitation).  Because the
ROI-fitted throughput polynomial (instrument_gain) is a property of the real instrument that the nominal twin lacks,
every comparison is reported without and with it.

Run: .venv/bin/python publications/spectralforge_studio/experiments/E8_sim2real.py [--quick]
Outputs: results/E8/{results.json,FINDINGS.md(by hand),table_sim2real.tex}, figures/E8_sim2real.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
RES = ROOT / "results" / "E8"
FIG = ROOT / "figures"
DATA = REPO / "Data" / "processed"
NUMS = REPO / "docs" / "studio" / "figures" / "sim2real_numbers.json"
SEED = 20260711
EXS = (310.0, 325.0, 340.0, 365.0, 385.0, 400.0)
MIN_OFFSET = 20.0


def sam_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ------------------------------------------------------------------------------------------ real data + fits
def load_pepsin2():
    from spectraforge.studio import calibration as cal, realdata as rd
    cube = rd.load_real_cube(DATA / "Collagen Pepsin", name="Collagen Pepsin")
    e = rd.real_roi_eem(cube, "2")
    fit = cal.fit_fluorophores(e, gain="per_excitation", response_degree=2, n_starts=16, seed=SEED, name="c")
    nom = cal.fit_fluorophores(e, gain="per_excitation", response_degree=0, n_starts=16, seed=SEED, name="c")
    return cube, e, fit, nom


# ------------------------------------------------------------------------------------------ MC cover scene
def collagen_scene(spec_name: str, fl_specs, cover_mm: float | None, photons: int):
    """Thick collagen slab (cover_mm None: 3 mm slab on a non-fluorescent absorber) or a cover of that thickness on
    reticular dermis.  The collagen dye of the cover material is replaced by ``spec_name`` (concentration of the
    library collagen cover, 1200 uM)."""
    from spectraforge.studio import presets as P
    from spectraforge.studio.presets_tissue import tissue_materials
    from spectraforge.studio.schema import DyeAmount, ExcitationStep, LayeredBlockShape
    exp = P.mehsi_lab_replica()
    mats = {m.id: m for m in exp.materials}
    for m in tissue_materials():
        mats[m.id] = m
    cov = mats["collagen_dense"]
    cov = cov.model_copy(update={"id": "cover_mat", "dyes": [DyeAmount(fluorophore=spec_name, concentration_uM=1200.0)]})
    mats["cover_mat"] = cov
    if cover_mm is None:
        layers = [P._layer("support", "india_ink_absorber", 2.0), P._layer("cover", "cover_mat", 3.0)]
    else:
        layers = [P._layer("support", "dermis_reticular", 3.0), P._layer("cover", "cover_mat", cover_mm)]
    block = LayeredBlockShape(footprint_mm=(8.0, 8.0), layers=layers)
    seq = [(310, 0.42, 4000), (325, 0.85, 2000), (340, 1.3, 1000), (365, 2.6, 250), (385, 3.4, 60), (400, 3.9, 50)]
    src = P.tunable_lamp(target=(0, 2.5, 0), sequence=[(w, p, e) for w, p, e in seq])
    cam = P.nuance_camera(target=(0, 2.5, 0))
    cam = cam.model_copy(update={"sensor": cam.sensor.model_copy(update={"width_px": 64, "height_px": 48, "binning": 2})})
    acq = exp.acquisition.model_copy(update={"preview_scale": 0.5, "volume_mc": exp.acquisition.volume_mc.model_copy(
        update={"excitation_photons": photons, "emission_photons": photons})})
    fl = {f.name: f for f in exp.fluorophores}
    for f in fl_specs:
        fl[f.name] = f
    return exp.model_copy(update={"name": f"E8 {spec_name}", "materials": list(mats.values()),
                                  "fluorophores": list(fl.values()),
                                  "objects": [P._obj("block", "block", block, "cover_mat", (0, 0, 0))],
                                  "sources": [src], "cameras": [cam], "acquisition": acq})


def sim_vs_measured(res, e_meas, inst_gain: dict | None):
    """Per-excitation spectral angle between the simulated ROI-mean emission (central half of the image) and the
    measured one on the measured bands >= ex + MIN_OFFSET inside the simulated band span."""
    from spectraforge.studio import api
    w, h = res.width, res.height
    roi = {"shape": "rect", "x0": 0.3 * w, "y0": 0.3 * h, "x1": 0.7 * w, "y1": 0.7 * h}
    e = api.eem(res, roi, {"mode": "exposure_power"})
    out = {}
    for ex in e_meas.excitations_nm:
        if ex not in e.value:
            continue
        mb = np.asarray(e_meas.emission_nm[ex], float)
        sb = np.asarray(e.emission_nm[ex], float)
        sv = np.asarray(e.value[ex], float)
        ok = np.isfinite(sv)
        sim = np.interp(mb, sb[ok], sv[ok], left=np.nan, right=np.nan)
        if inst_gain:
            wl = np.array(sorted(inst_gain)); tv = np.array([inst_gain[x] for x in wl])
            sim = sim * np.interp(mb, wl, tv)
        meas = np.asarray(e_meas.value[ex], float)
        sel = np.isfinite(sim) & np.isfinite(meas) & (mb >= ex + MIN_OFFSET)
        out[float(ex)] = {"sam_deg": sam_deg(sim[sel], meas[sel]), "n_bands": int(sel.sum()),
                          "sim": (mb[sel], sim[sel] / np.max(sim[sel])), "meas": (mb[sel], meas[sel] / np.max(meas[sel]))}
    return out


def run_cover_check(e_meas, fit, photons: int, quick: bool):
    from spectraforge.studio import api, calibration as cal
    cal_spec = fit.specs[0]
    lib = {f.name: f for f in api.builtin_fluorophores()} if hasattr(api, "builtin_fluorophores") else None
    from spectraforge.studio.presets import builtin_fluorophores
    lib = {f.name: f for f in builtin_fluorophores()}
    colg = lib["collagen"]
    # drop-in replacement: calibrated emission shape/excitation, library extinction + quantum yield + bleach rate
    calc = cal_spec.model_copy(update={"name": "collagen_cal", "extinction": colg.extinction,
                                       "quantum_yield": colg.quantum_yield, "bleach_quantum_yield": colg.bleach_quantum_yield,
                                       "display_color": colg.display_color})
    cases = {"slab_3mm": None, "cover_0.5mm_on_dermis": 0.5} if not quick else {"slab_3mm": None}
    out, curves = {}, {}
    for cname, cover in cases.items():
        for lname, spec_name, fls in (("library", "collagen", []), ("calibrated", "collagen_cal", [calc])):
            t0 = time.time()
            exp = collagen_scene(spec_name, fls, cover, photons)
            res = api.capture(exp, mode="fast", seed=SEED)
            raw = sim_vs_measured(res, e_meas, None)
            cor = sim_vs_measured(res, e_meas, fit.instrument_gain)
            key = f"{cname}/{lname}"
            out[key] = {"capture_s": time.time() - t0,
                        "sam_deg_raw": {f"{k:g}": v["sam_deg"] for k, v in raw.items()},
                        "sam_deg_with_fitted_throughput": {f"{k:g}": v["sam_deg"] for k, v in cor.items()},
                        "mean_sam_deg_raw": float(np.mean([v["sam_deg"] for v in raw.values()])),
                        "mean_sam_deg_with_fitted_throughput": float(np.mean([v["sam_deg"] for v in cor.values()]))}
            curves[key] = {"raw": raw, "cor": cor}
            print(f"   {key}: mean SAM raw {out[key]['mean_sam_deg_raw']:.1f}, with throughput "
                  f"{out[key]['mean_sam_deg_with_fitted_throughput']:.1f}  ({out[key]['capture_s']:.0f}s)", flush=True)
    # thin-layer model reference (what the fit itself achieves) and a library-collagen spectrum with the ROI-fitted gain
    out["thin_layer_fit"] = {"sam_deg": {f"{k:g}": v for k, v in fit.diagnostics["sam_deg"].items()},
                             "mean_sam_deg": fit.diagnostics["sam_mean_deg"]}
    return out, curves


# ------------------------------------------------------------------------------------------ table + figure
def holdout_table(nums):
    ho = nums["holdout"]
    b1, b1s = ho["B1_heldout_excitations"]["mean_sam_deg"], ho["B1_heldout_excitations"]["std_sam_deg"]
    n1 = ho["B1_heldout_excitations"]["n_cases"]
    pairs = ho["B2_pairs"]
    rows = []
    names = [("in_sample_refit_K1", "In-sample refit (1 comp.)"), ("corrected_K1", "Twin, throughput-corrected (1 comp.)"),
             ("corrected_ripple_K1", "Twin + band ripple"), ("corrected_K2", "Twin, 2 components"),
             ("nominal_K1", "Twin, nominal instrument"), ("baseline_empirical", "Empirical spectrum (copy)"),
             ("baseline_flat", "Flat spectrum")]
    for k, lab in names:
        r = {"model": lab, "B1_mean": b1[k], "B1_sd": b1s.get(k), "B1_se": None if k not in b1s else b1s[k] / np.sqrt(n1)}
        pk = "baseline_empirical" if k == "baseline_empirical" else k
        for grp in ("within", "across"):
            v = [p[pk] for p in pairs if p["group"] == grp and pk in p]
            if v:
                r[f"B2_{grp}_mean"] = float(np.mean(v)); r[f"B2_{grp}_sd"] = float(np.std(v, ddof=1)); r[f"B2_{grp}_n"] = len(v)
        rows.append(r)
    return rows


def b3_stats(nums):
    b3 = nums["holdout"]["B3_heldout_excitations_absolute_drops"]
    out = {}
    for deg in (0, 2):
        rows = [r for r in b3["rows"] if r["deg"] == deg]
        lr = np.log([v for r in rows for v in r["peak_ratio"].values()])
        sam = np.array([r["sam_deg"] for r in rows]); r2 = np.array([r["r2_abs"] for r in rows])
        out[str(deg)] = {"n_drops": len(rows), "median_sam_deg": float(np.median(sam)),
                         "iqr_sam_deg": [float(np.percentile(sam, 25)), float(np.percentile(sam, 75))],
                         "median_r2_abs": float(np.median(r2)),
                         "median_abs_ln_peak_ratio": float(np.median(np.abs(lr))),
                         "geom_mean_peak_ratio": float(np.exp(lr.mean())),
                         "geom_sd_ln_peak_ratio": float(lr.std(ddof=1)),
                         "frac_within_25pct": float(np.mean(np.abs(np.exp(lr) - 1) < 0.25))}
    return out


def latex_table(rows):
    f = lambda m, s=None: "--" if m is None else (f"{m:.1f}" if s is None else f"{m:.1f}\\,$\\pm$\\,{s:.1f}")
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Model & B1 held-out $\\lambda_{ex}$ & B2 same dataset & B2 other dataset \\\\", "\\midrule"]
    for r in rows:
        L.append(f"{r['model']} & {f(r['B1_mean'], r['B1_sd'])} & {f(r.get('B2_within_mean'), r.get('B2_within_sd'))} & "
                 f"{f(r.get('B2_across_mean'), r.get('B2_across_sd'))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def make_figure(e_meas, fit, nom, rows, b3, nums, cover_curves, cover_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spectraforge.studio import calibration as cal
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "sky": "#56B4E9", "orange": "#E69F00", "grey": "#777777",
          "pink": "#CC79A7"}
    fig = plt.figure(figsize=(7.16, 5.6))
    gs = fig.add_gridspec(2, 12, hspace=0.6, wspace=2.6)
    # (a) spectra at three excitations
    show = [340.0, 365.0, 400.0]
    bands = {ex: np.asarray(e_meas.emission_nm[ex], float) for ex in e_meas.excitations_nm}
    pred = cal.predict_eem(fit, bands)
    prn = cal.predict_eem(nom, bands)
    for i, ex in enumerate(show):
        ax = fig.add_subplot(gs[0, 4 * i:4 * i + 4])
        em, y = bands[ex], np.asarray(e_meas.value[ex]); sd = np.asarray(e_meas.std[ex]); sc = np.nanmax(y)
        ax.errorbar(em, y / sc, yerr=sd / sc, fmt="o", ms=2.2, color="k", lw=0.5, label="measured")
        ax.plot(em, pred[ex] / sc, "-", color=OI["verm"], lw=1.4, label="calibrated twin")
        ax.plot(em, prn[ex] / sc, "--", color=OI["blue"], lw=1.0, label="nominal instrument")
        ax.set_ylim(0, 1.3); ax.set_xlabel("emission (nm)")
        ax.set_title(f"{ex:g} nm exc.: SAM {fit.diagnostics['sam_deg'][ex]:.1f}$^\\circ$", fontsize=7.5)
        if i == 0:
            ax.set_ylabel("norm. signal"); ax.legend(loc="upper right", frameon=False, handlelength=1.6)
        ax.text(-0.02, 1.16, "(a)" if i == 0 else "", transform=ax.transAxes, fontweight="bold", fontsize=9)
    # (b) hold-out bars
    ax = fig.add_subplot(gs[1, 0:5])
    sel = ["in_sample_refit_K1", "corrected_K1", "nominal_K1", "baseline_empirical", "baseline_flat"]
    lab = ["in-sample\nrefit", "twin\ncorrected", "twin\nnominal", "empirical\ncopy", "flat"]
    cols = [OI["grey"], OI["verm"], OI["blue"], OI["green"], "0.8"]
    rr = {r["model"]: r for r in rows}
    keys = ["In-sample refit (1 comp.)", "Twin, throughput-corrected (1 comp.)", "Twin, nominal instrument",
            "Empirical spectrum (copy)", "Flat spectrum"]
    w = 0.26
    for j, (g, hatch) in enumerate((("B1_mean", None), ("B2_within_mean", "///"), ("B2_across_mean", "...") )):
        vals = [rr[k].get(g) for k in keys]
        sds = [rr[k].get(g.replace("mean", "sd")) for k in keys]
        x = np.arange(len(keys)) + (j - 1) * w
        for xi, v, sd_, c in zip(x, vals, sds, cols):
            if v is not None:
                ax.bar(xi, v, w, color=c, edgecolor="k", linewidth=0.4, hatch=hatch,
                       yerr=None if sd_ is None else sd_, error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(lab)
    ax.set_ylabel("mean spectral angle (deg)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="w", ec="k", lw=0.5, label="B1 held-out excitations"),
                       Patch(fc="w", ec="k", lw=0.5, hatch="///", label="B2 other ROI, same dataset"),
                       Patch(fc="w", ec="k", lw=0.5, hatch="...", label="B2 other ROI, other dataset")],
              frameon=False, loc="upper left", bbox_to_anchor=(0.2, 1.02), fontsize=6.5)
    ax.set_ylim(0, 30)
    ax.text(-0.02, 1.08, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    # (c) drops peak ratio
    ax = fig.add_subplot(gs[1, 5:8])
    rows3 = [r for r in nums["holdout"]["B3_heldout_excitations_absolute_drops"]["rows"] if r["deg"] == 2]
    for k, ex in enumerate(("325.0", "365.0", "400.0")):
        v = np.array([r["peak_ratio"][ex] for r in rows3])
        ax.scatter(np.full(v.size, k) + np.random.default_rng(k).uniform(-0.15, 0.15, v.size), v, s=6, color=OI["pink"], alpha=0.8, lw=0)
        ax.hlines(np.median(v), k - 0.3, k + 0.3, color="k", lw=1.2)
    ax.axhline(1, color="0.4", lw=0.6, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels(["325", "365", "400"]); ax.set_xlabel("held-out excitation (nm)")
    ax.set_ylabel("predicted / measured peak"); ax.set_ylim(0.4, 1.6)
    ax.set_title(f"drops (n={b3['2']['n_drops']}), absolute", fontsize=7.5)
    ax.text(-0.02, 1.08 + 0.0, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    # (d) cover check SAM per excitation
    ax = fig.add_subplot(gs[1, 9:12])
    exl = [f"{e:g}" for e in EXS]
    k0 = "slab_3mm/calibrated"; k1 = "slab_3mm/library"
    if k0 in cover_out:
        ax.plot(range(len(EXS)), [cover_out[k0]["sam_deg_with_fitted_throughput"].get(e) for e in exl], "o-", color=OI["verm"], ms=3, lw=1, label="MC, calibrated")
        ax.plot(range(len(EXS)), [cover_out[k1]["sam_deg_with_fitted_throughput"].get(e) for e in exl], "s--", color=OI["sky"], ms=3, lw=1, label="MC, library")
        ax.plot(range(len(EXS)), [cover_out[k0]["sam_deg_raw"].get(e) for e in exl], "o:", color=OI["verm"], ms=2.5, lw=0.8, mfc="w", label="calib., no throughput corr.")
    ax.set_xticks(range(len(EXS))); ax.set_xticklabels(exl, rotation=45)
    ax.set_xlabel("excitation (nm)"); ax.set_ylabel("SAM to measured (deg)")
    ax.legend(frameon=False, fontsize=5.5, loc="center right", bbox_to_anchor=(1.0, 0.5)); ax.set_ylim(0, 75)
    ax.text(-0.02, 1.08, "(d)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "E8_sim2real.pdf", bbox_inches="tight")
    fig.savefig(FIG / "E8_sim2real.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # second figure: MC cover check spectra
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True)
    ck = cover_curves.get("slab_3mm/calibrated")
    kl = cover_curves.get("slab_3mm/library")
    for ax, ex in zip(axs, (340.0, 365.0, 400.0)):
        if ck is None:
            continue
        m = ck["cor"][ex]["meas"]
        ax.plot(*m, "ko", ms=2.5, label="measured")
        ax.plot(*ck["cor"][ex]["sim"], "-", color=OI["verm"], lw=1.4, label="MC, calibrated collagen")
        ax.plot(*kl["cor"][ex]["sim"], "--", color=OI["sky"], lw=1.2, label="MC, library collagen")
        ax.set_title(f"{ex:g} nm: SAM {ck['cor'][ex]['sam_deg']:.1f}$^\\circ$ / {kl['cor'][ex]['sam_deg']:.1f}$^\\circ$", fontsize=7.5)
        ax.set_xlabel("emission (nm)")
    axs[0].set_ylabel("norm. signal"); axs[0].legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "E8_cover_check.pdf", bbox_inches="tight")
    fig.savefig(FIG / "E8_cover_check.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--photons", type=int, default=1_000_000)
    a = ap.parse_args()
    from spectraforge.studio.provenance import provenance
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    nums = json.loads(NUMS.read_text())
    rows = holdout_table(nums)
    b3 = b3_stats(nums)
    results = {"units": {"sam": "degrees", "gains": "model units", "nm": "nm"}, "source_numbers": str(NUMS.relative_to(REPO)),
               "holdout_table": rows, "B3_drops": b3,
               "collagen_gain_ratio_vs_340": nums["collagen_gain_ratio_vs_340"],
               "fast_vs_full_max_rel_dev": nums["part0"]["fast_vs_full_capture"]["max_rel_dev"],
               "fig27_pixel_sam_median_deg": nums["fig27"]["pixel_sam_median_deg"]}
    (RES / "table_sim2real.tex").write_text(latex_table(rows))
    cube, e_meas, fit, nom = load_pepsin2()
    results["pepsin2"] = {"corrected": {"r2": fit.diagnostics["r2"], "sam_mean_deg": fit.diagnostics["sam_mean_deg"],
                                        "sam_deg": {f"{k:g}": v for k, v in fit.diagnostics["sam_deg"].items()}},
                          "nominal": {"r2": nom.diagnostics["r2"], "sam_mean_deg": nom.diagnostics["sam_mean_deg"]},
                          "summary": __import__("spectraforge.studio.calibration", fromlist=["x"]).emission_summary(fit.specs[0]),
                          "em_fwhm_nm": fit.specs[0].em_fwhm_nm, "em_skew": fit.specs[0].em_skew,
                          "instrument_gain_range": [min(fit.instrument_gain.values()), max(fit.instrument_gain.values())]}
    print(f"[{time.time()-t0:.0f}s] fit: R2 {fit.diagnostics['r2']:.3f} SAM {fit.diagnostics['sam_mean_deg']:.2f}", flush=True)
    cover_out, curves = run_cover_check(e_meas, fit, 200_000 if a.quick else a.photons, a.quick)
    results["cover_check"] = cover_out
    results["cover_check_settings"] = {"photons": 200_000 if a.quick else a.photons, "seed": SEED, "roi": "central 40% x 40%",
                                       "band_rule": f"bands >= ex + {MIN_OFFSET:g} nm", "concentration_uM": 1200.0,
                                       "note": "calibrated spec uses library collagen extinction/QY/bleach; shape only is calibrated"}
    results["provenance"] = provenance()
    results["wall_time_s"] = time.time() - t0
    (RES / "results.json").write_text(json.dumps(jsonable(results), indent=2))
    make_figure(e_meas, fit, nom, rows, b3, nums, curves, cover_out)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
