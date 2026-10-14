"""E3: global sensitivity and confounders of the optical redox ratio (SpectralForge Studio paper).

Question: how much of the variance of the measured optical redox ratio RR = FAD/(FAD+NADH) is caused by the
collagen cover (thickness, scattering, blood, oxygenation, collagen autofluorescence) rather than by the true
metabolic state of the buried tissue (NADH, FAD concentration), and does an elastic-channel / cover-autofluorescence
correction remove it?

Scene: a buried 0.5 mm NADH/FAD layer (colon-mucosa optics, Jacques 2013 eq. 2 scattering) under a collagen
cover (dense collagen, blood B / StO2 absorption), on reticular dermis; lab TLS (365 and 450 nm lines) + a
Nuance-class LCTF camera whose band list is extended down to 360 nm so that the elastic (diffuse-reflectance)
channel at the excitation wavelength is recorded (hypothetical hardware change, documented in FINDINGS.md).

Design: 8 inputs, Saltelli (Sobol' S1/ST + bootstrap CIs), Morris cross-check, a replicate study for the MC
noise floor, a reference capture for the cover-autofluorescence leak coefficient.  Everything runs through the
public API (api.capture, study.run_study, sensitivity.sobol_indices/morris_indices).

Run:  .venv/bin/python publications/spectralforge_studio/experiments/E3_sensitivity_redox.py [--quick]
          [--n N] [--morris R] [--workers W] [--photons P]
Outputs: results/E3/{results.json,runs.parquet,FINDINGS.md(by hand)}, figures/E3_*.{pdf,png}.
"""
from __future__ import annotations

import argparse
import warnings
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectraforge.studio import api, presets as P, sensitivity as S, study as ST
from spectraforge.studio.provenance import provenance
from spectraforge.studio.schema import DyeAmount, ExcitationStep, LayeredBlockShape, TissueComposition
from spectraforge.studio.study_schema import DesignSpec, MetricSpec, ParameterSpec, StudySpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "E3"
FIG = ROOT / "figures"

NORM = {"mode": "exposure_power"}
ROI = {"shape": "rect", "x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7}
NADH_BAND, FAD_BAND = (440.0, 480.0), (510.0, 550.0)   # NADH em ~460 nm, FAD em ~530 nm
EL450_BAND = (440.0, 460.0)  # elastic channel at the 450 nm line (no usable elastic signal below 400 nm, see FINDINGS)
CAF_BAND = (400.0, 420.0)                                 # collagen autofluorescence (blue edge) at 365 nm excitation
PARAMS = ["cover_mm", "mus500", "b_mie", "blood", "sto2", "nadh_uM", "fad_uM", "collagen_uM"]
COVER_GROUP = ["cover_mm", "mus500", "b_mie", "blood", "sto2", "collagen_uM"]
META_GROUP = ["nadh_uM", "fad_uM"]
LABEL = {"cover_mm": "cover thickness", "mus500": r"cover $\mu_s'$(500)", "b_mie": r"cover $b_{Mie}$",
         "blood": "cover blood fraction", "sto2": r"cover StO$_2$", "nadh_uM": "NADH conc.",
         "fad_uM": "FAD conc.", "collagen_uM": "collagen fluor. conc."}


# ------------------------------------------------------------------------------------------ scene
def build_scene(photons: int = 2_000_000, sensor=(64, 48), node_step: float = 40.0):
    mats = {m.id: m for m in P.builtin_materials()}
    cover = mats["collagen_dense"].model_copy(update={
        "id": "cover", "name": "Collagen cover (blood-perfused)",
        "tissue": TissueComposition(blood_volume_fraction=0.01, oxygen_saturation=0.7, water_fraction=0.65)})
    buried = mats["mucosa_colon"].model_copy(update={
        "id": "buried", "name": "Buried NADH/FAD tissue",
        "dyes": [DyeAmount(fluorophore="NADH", concentration_uM=60.0), DyeAmount(fluorophore="FAD", concentration_uM=40.0)]})
    exp = P.mehsi_lab_replica()
    block = LayeredBlockShape(footprint_mm=(20.0, 20.0), layers=[
        P._layer("substrate", "dermis_reticular", 3.0), P._layer("buried", "buried", 0.5),
        P._layer("cover", "cover", 0.5)])
    seq = [ExcitationStep(wavelength_nm=365, power_mW=2.6, exposure_ms=250),
           ExcitationStep(wavelength_nm=450, power_mW=4.6, exposure_ms=8)]
    src = P.tunable_lamp(target=(0, 1.9, 0)).model_copy(update={"sequence": seq})
    cam = P.nuance_camera(target=(0, 1.9, 0))
    cam = cam.model_copy(update={
        "filter": cam.filter.model_copy(update={"band_start_policy": "fixed", "band_start_nm": 400.0, "n_bands": 25,
                                                        "peak_transmission_curve": "flat"}),
        "sensor": cam.sensor.model_copy(update={"width_px": sensor[0], "height_px": sensor[1], "binning": 4})})
    acq = exp.acquisition.model_copy(update={"volume_mc": exp.acquisition.volume_mc.model_copy(update={
        "excitation_photons": photons, "emission_photons": photons,
        "emission_node_step_nm": node_step})})
    exp = exp.model_copy(update={
        "name": "E3 tissue under collagen cover", "materials": list(mats.values()) + [cover, buried],
        "objects": [P._obj("block", "block", block, "cover", (0, 0, 0))], "sources": [src], "cameras": [cam],
        "acquisition": acq})
    return exp


def parameters(exp) -> list[ParameterSpec]:
    ids = [m.id for m in exp.materials]
    ic, ib = ids.index("cover"), ids.index("buried")
    cov = exp.materials[ic]
    col_i = [d.fluorophore for d in cov.dyes].index("collagen")
    L = lambda p, **kw: ParameterSpec(pointer=p, kind="continuous", **kw)  # noqa: E731
    return [
        L("/objects/0/shape/layers/2/thickness/thickness_mm", label="cover_mm", low=0.1, high=1.0,
          also_set=["/objects/0/shape/layers/2/thickness/end_thickness_mm"]),
        L(f"/materials/{ic}/scattering/mus_prime_500_per_mm", label="mus500", low=1.5, high=4.5),
        L(f"/materials/{ic}/scattering/power_b", label="b_mie", low=0.5, high=1.5),
        L(f"/materials/{ic}/tissue/blood_volume_fraction", label="blood", low=0.002, high=0.05),
        L(f"/materials/{ic}/tissue/oxygen_saturation", label="sto2", low=0.4, high=1.0),
        L(f"/materials/{ib}/dyes/0/concentration_uM", label="nadh_uM", low=20.0, high=150.0),
        L(f"/materials/{ib}/dyes/1/concentration_uM", label="fad_uM", low=10.0, high=100.0),
        L(f"/materials/{ic}/dyes/{col_i}/concentration_uM", label="collagen_uM", low=300.0, high=1500.0),
    ]


def metrics() -> list[MetricSpec]:
    def bs(name, ex, band):
        return MetricSpec(name=name, kind="band_sum", roi=ROI, ex=ex, from_nm=band[0], to_nm=band[1], norm=NORM)
    return [bs("nadh", 365, NADH_BAND), bs("fad", 450, FAD_BAND), bs("el450", 450, EL450_BAND), bs("caf", 365, CAF_BAND),
            MetricSpec(name="peak", kind="peak_wavelength", roi=ROI, ex=365, norm=NORM)]


# ------------------------------------------------------------------------------------------ derived outputs
def calibrate(ref: dict) -> dict:
    """One-point calibration of the corrected ratios on a reference capture (thin cover, nominal NADH/FAD):
    the constants k make every corrected ratio equal the raw ratio there, so the corrections change only how
    the ratio responds to the cover, never its absolute level at the reference."""
    n, f, e, c = ref["nadh"], ref["fad"], ref["el450"], ref["caf"]
    R = f / (f + n)
    fe, nc = f / e, n / c
    g = (1 - R) / R
    return {"R_ref": R, "k_F": fe * g / n, "k_N": f * g / nc, "k_FN": fe * g / nc}


def add_derived(df: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Redox ratios from the per-capture band signals (all in exposure-and-power-normalised units).

    rr_raw  = F/(F+N), F = FAD band (ex 450, em 510-550), N = NADH band (ex 365, em 440-480)
    rr_F    = Fe/(Fe + k_F N),   Fe = F / el450          (FAD signal divided by the elastic channel at 450 nm)
    rr_N    = F/(F + k_N Nc),    Nc = N / caf            (NADH divided by the cover-autofluorescence channel,
                                                          ex 365, em 400-420)
    rr_FN   = Fe/(Fe + k_FN Nc)                          (both)
    k_* from :func:`calibrate`."""
    d = df.copy()
    n, f = d["m:nadh"], d["m:fad"]
    fe, nc = f / d["m:el450"], n / d["m:caf"]
    d["m:rr_raw"] = f / (f + n)
    d["m:rr_F"] = fe / (fe + cal["k_F"] * n)
    d["m:rr_N"] = f / (f + cal["k_N"] * nc)
    d["m:rr_FN"] = fe / (fe + cal["k_FN"] * nc)
    return d


RR_ALL = ["rr_raw", "rr_F", "rr_N", "rr_FN"]
OUTPUTS = ["nadh", "fad", "peak"] + RR_ALL


def sobol_for(df: pd.DataFrame, n: int, k: int, col: str, n_boot: int, seed: int = 0) -> dict:
    ok = df
    pm = ok.groupby(["block", "row"], sort=False)[col].mean().reset_index()
    blocks = ["A", "B"] + [f"AB:{i}" for i in range(k)]
    Y = np.full((len(blocks), n), np.nan)
    for bi, b in enumerate(blocks):
        sub = pm[pm["block"] == b]
        Y[bi, sub["row"].to_numpy()] = sub[col].to_numpy()
    good = np.isfinite(Y).all(axis=0)
    r = S.sobol_indices(Y[:, good].reshape(-1), int(good.sum()), k, n_boot=n_boot, seed=seed)
    r["n_eff"] = int(good.sum())
    return r


# ------------------------------------------------------------------------------------------ surrogate groups
def group_indices(U: np.ndarray, y: np.ndarray, groups: dict[str, list[int]], n_boot: int, seed: int = 0,
                  n_mc: int = 20000) -> dict:
    """Closed (S_G = Var(E[Y|X_G])/V) and total (ST_G = 1 - S_{complement}) Sobol' group indices from a GP
    surrogate (ARD Matern-5/2) fitted to all design points; 5-fold CV R^2 as a quality check; bootstrap CI by
    refitting on resampled training points."""
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
    from sklearn.model_selection import KFold

    k = U.shape[1]
    mu, sd = float(np.mean(y)), float(np.std(y))
    z = (y - mu) / sd

    def fit(Ut, zt, s):
        ker = C(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(k), length_scale_bounds=(0.05, 50.0), nu=2.5) \
            + WhiteKernel(1e-3, (1e-6, 1e-1))
        return GPR(ker, normalize_y=False, n_restarts_optimizer=1, random_state=s).fit(Ut, zt)

    def indices(gp, s):
        rng = np.random.default_rng(s)
        A, B = rng.random((n_mc, k)), rng.random((n_mc, k))
        fA, fB = gp.predict(A), gp.predict(B)
        V = np.var(np.concatenate([fA, fB]))
        f0 = 0.5 * (fA.mean() + fB.mean())
        out = {}
        for g, idx in groups.items():
            comp = [j for j in range(k) if j not in idx]
            Cg = B.copy(); Cg[:, idx] = A[:, idx]          # shares X_G with A
            Cc = B.copy(); Cc[:, comp] = A[:, comp]        # shares complement with A
            s_closed = (np.mean(fA * gp.predict(Cg)) - f0 ** 2) / V
            s_comp = (np.mean(fA * gp.predict(Cc)) - f0 ** 2) / V
            out[g] = (float(s_closed), float(1.0 - s_comp))
        return out

    gp = fit(U, z, seed)
    pred = np.empty_like(z)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(U):
        pred[te] = fit(U[tr], z[tr], seed).predict(U[te])
    r2cv = float(1.0 - np.sum((z - pred) ** 2) / np.sum((z - z.mean()) ** 2))
    base = indices(gp, seed)
    rng = np.random.default_rng(seed + 1)
    boots = []
    for b in range(n_boot):
        ii = rng.integers(0, len(z), len(z))
        boots.append(indices(fit(U[ii], z[ii], seed + b + 2), seed + b + 2))
    res = {"r2_cv": r2cv, "kernel": str(gp.kernel_), "groups": {}}
    for g in groups:
        bc = np.array([b[g][0] for b in boots]); bt = np.array([b[g][1] for b in boots])
        res["groups"][g] = {"S_closed": base[g][0], "S_closed_ci": [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
                            "ST": base[g][1], "ST_ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}
    return res


def compute_indices(d: pd.DataFrame, dm: pd.DataFrame, dn: pd.DataFrame, N: int, k: int, n_boot: int):
    """Sobol' (centred estimator), Morris and MC-noise-floor tables for every output from the stored runs."""
    sob, mor, noise = {}, {}, {}
    for o in OUTPUTS + ["el450", "caf"]:
        col = f"m:{o}"
        r = sobol_for(d, N, k, col, n_boot)
        sob[o] = {"variance": r["variance"], "n_eff": r["n_eff"], "sum_S1": float(np.nansum(r["S1"])),
                  "sum_ST": float(np.nansum(r["ST"])),
                  "indices": {p: {"S1": float(r["S1"][i]), "S1_ci": [float(x) for x in r["S1_ci"][i]],
                                  "ST": float(r["ST"][i]), "ST_ci": [float(x) for x in r["ST_ci"][i]]}
                              for i, p in enumerate(PARAMS)}}
        # Morris
        pm = dm.groupby("point", sort=True)[col].mean().reset_index()
        meta = dm.drop_duplicates("point").set_index("point")
        yy = pm[col].to_numpy()
        mr = S.morris_indices(yy, meta["traj"].to_numpy(), meta["step"].to_numpy(), meta["step_param"].to_numpy(),
                              meta["direction"].to_numpy(), k, 4 / (2 * 3), n_boot=min(n_boot, 500), seed=0)
        mor[o] = {p: {"mu": float(mr["mu"][i]), "mu_star": float(mr["mu_star"][i]), "sigma": float(mr["sigma"][i]),
                      "mu_star_ci": [float(x) for x in mr["mu_star_ci"][i]]} for i, p in enumerate(PARAMS)}
        nf = S.noise_floor(dn, col)
        noise[o] = {**nf, "cv_within": float(np.sqrt(nf["within_var"]) / abs(dn.groupby("point")[col].mean().mean()))
                    if np.isfinite(nf["within_var"]) else None}
    return sob, mor, noise


def reanalyse(quick: bool = False, n_boot: int | None = None) -> None:
    """Recompute every Sobol'/Morris/noise table from the STORED runs (no simulation) with the current
    (centred) estimator, keep all non-index fields of the old results.json (group surrogate, calibration, timing),
    archive the previous file as results_pre_centering.json, and regenerate the figures."""
    out = RES / ("quick" if quick else "")
    old = json.loads((out / "results.json").read_text())
    if not (out / "results_pre_centering.json").exists():
        (out / "results_pre_centering.json").write_text(json.dumps(old, indent=2))
    d, dm, dn = (pd.read_parquet(out / f) for f in ("runs.parquet", "runs_morris.parquet", "runs_noise.parquet"))
    N, k, nb = old["N_saltelli_base"], old["k"], n_boot or old["n_boot"]
    old["sobol"], old["morris"], old["noise"] = compute_indices(d, dm, dn, N, k, nb)
    old["reanalysis"] = {"estimator": "Sobol' S1 Saltelli 2010 on outputs centred by mean(fA,fB); ST Jansen",
                         "source": "stored runs*.parquet, no re-simulation", "n_boot": nb}
    (out / "results.json").write_text(json.dumps(old, indent=2, default=float))
    print(f"re-analysed {out / 'results.json'}")
    from E3_figures import make_figures  # noqa: E402
    make_figures(out, quick)


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reanalyse", action="store_true", help="recompute indices + figures from stored runs only")
    ap.add_argument("--n", type=int, default=None, help="Saltelli base sample size")
    ap.add_argument("--morris", type=int, default=None, help="Morris trajectories")
    ap.add_argument("--noise_points", type=int, default=None)
    ap.add_argument("--noise_reps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--photons", type=int, default=None)
    ap.add_argument("--boot", type=int, default=None)
    a = ap.parse_args()
    q = a.quick
    if a.reanalyse:
        return reanalyse(q, a.boot)
    N = a.n or (2 if q else 64)
    R_M = a.morris or (2 if q else 8)
    NP, NR = a.noise_points or (2 if q else 6), a.noise_reps or (2 if q else 4)
    photons = a.photons or (50_000 if q else 1_000_000)
    n_boot = a.boot or (30 if q else 1000)
    sensor = (32, 24) if q else (64, 48)
    out = RES / ("quick" if q else "")
    out.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    exp = build_scene(photons, sensor, 100.0 if q else 40.0)
    params = parameters(exp)
    mets = metrics()
    k = len(params)

    def spec(name, kind, n, reps=1, levels=4, seed=0):
        return StudySpec(name=name, base_experiment=exp, parameters=params, metrics=mets, mode="fast", replicates=reps,
                         design=DesignSpec(kind=kind, n=n, levels=levels, seed=seed), max_runs=100000)

    def run(name, sp):
        t0 = time.time()
        r = ST.run_study(sp, out / "studies" / name, workers=a.workers, resume=True,
                         progress=lambda f, m: print(f"  [{name}] {f * 100:5.1f}% {m}", end="\r", flush=True))
        print(f"\n  [{name}] done in {time.time() - t0:.0f} s")
        return ST.load_runs(out / "studies" / name), ST.load_manifest(out / "studies" / name)

    # 1) cover-free reference capture (thin cover, nominal concentrations) -> calibration constants
    from spectraforge.studio.study import compute_metrics
    ref = exp.model_copy(deep=True)
    ref.objects[0].shape.layers[2].thickness.thickness_mm = 0.02
    ref.objects[0].shape.layers[2].thickness.end_thickness_mm = 0.02
    refm = [compute_metrics(api.capture(ref, mode="fast", seed=sd), mets) for sd in range(3 if not q else 1)]
    refd = {k_: float(np.mean([r[k_] for r in refm])) for k_ in refm[0]}
    cal = calibrate(refd)
    print("reference", refd, "calibration", cal)

    def runs(name, sp):
        df_, man_ = run(name, sp)
        ok_ = df_[df_["status"] == "ok"]
        return df_, man_, add_derived(ok_, cal)

    # 2) small studies first (MC noise floor, Morris), then the big Saltelli study
    dfn, _, dn = runs("noise", spec("E3 noise", "lhs", NP, reps=NR, seed=7))
    dfm, manm, dm = runs("morris", spec("E3 morris", "morris", R_M, levels=4))
    df, man, d = runs("saltelli", spec("E3 saltelli", "saltelli", N))
    nfail = int((df["status"] != "ok").sum())

    # ---- indices
    results: dict = {"experiment": "E3", "units": {"signals": "ADU / (mW s), 20 nm-wide bands summed over ROI mean",
                                                      "peak": "nm", "cover_mm": "mm", "mus500": "1/mm",
                                                      "blood": "volume fraction", "nadh_uM,fad_uM,collagen_uM": "uM"},
                     "quick": q, "N_saltelli_base": N, "n_runs_saltelli": int(len(df)), "n_failed_saltelli": nfail,
                     "k": k, "parameters": {p.label: [p.low, p.high] for p in params},
                     "morris_trajectories": R_M, "n_runs_morris": int(len(dfm)),
                     "noise_study": {"points": NP, "replicates": NR}, "photons_per_pass": photons,
                     "sensor_px": list(sensor), "reference": refd, "calibration": cal, "n_boot": n_boot, "excitations_nm": [365, 450],
                     "bands_nm": {"nadh": NADH_BAND, "fad": FAD_BAND, "el450": EL450_BAND,
                                  "caf": CAF_BAND}}
    sob, mor, noise = compute_indices(d, dm, dn, N, k, n_boot)
    results["sobol"], results["morris"], results["noise"] = sob, mor, noise

    # ---- confounder fraction: group indices from a GP surrogate fitted to every Saltelli point
    U = (df[df["status"] == "ok"][[f"u:{p}" for p in PARAMS]]).to_numpy()
    grp = {"cover": [PARAMS.index(p) for p in COVER_GROUP], "metabolism": [PARAMS.index(p) for p in META_GROUP]}
    gi = {}
    for o in RR_ALL:
        yv = d[f"m:{o}"].to_numpy()
        ok = np.isfinite(yv)
        # use per-row replicate means (replicates=1 here)
        gi[o] = group_indices(d[[f"u:{p}" for p in PARAMS]].to_numpy()[ok], yv[ok], grp, n_boot=25 if not q else 2)
        print(o, {g: round(v["ST"], 3) for g, v in gi[o]["groups"].items()}, "R2cv", round(gi[o]["r2_cv"], 3))
    results["group_surrogate"] = gi

    # ---- runs table + provenance
    keep = ["run_id", "block", "row", "point", "seed", *[f"u:{p}" for p in PARAMS], *[f"p:{p}" for p in PARAMS],
            *[f"m:{c}" for c in ["nadh", "fad", "el450", "caf", "peak", *RR_ALL]],
            "t_capture_s"]
    d[keep].to_parquet(out / "runs.parquet")
    dm.to_parquet(out / "runs_morris.parquet")
    dn.to_parquet(out / "runs_noise.parquet")
    results["timing"] = {"mean_capture_s": float(df["t_capture_s"].mean()), "total_wall_s": time.time() - t_all}
    results["provenance"] = provenance()
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out / 'results.json'}")

    from E3_figures import make_figures  # noqa: E402
    make_figures(out, q)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
