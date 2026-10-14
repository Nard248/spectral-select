"""Reproduce the sim-to-real numbers used in the WHISPERS 2026 'Validation against real acquisitions' paragraph.
Re-uses the helpers of scripts/studio_sim2real.py (identical protocol, seeds, starts)."""
import json, sys, importlib.util, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("s2r", ROOT / "scripts" / "studio_sim2real.py")
s2r = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2r)
from spectraforge.studio import calibration as cal, realdata as rd

OUT = Path(__file__).parent
res = {}
# ---------------- (a) drops ----------------
drops = rd.load_real_cube(s2r.DROPS, labels=s2r.DROP_LABELS, name="Drop Data (rad_merged)")
tr_ex, te_ex = [310., 340., 385., 415.], [325., 365., 400.]
rows, curves = [], {}
for name in drops.roi_names():
    e = rd.real_roi_eem(drops, name)
    etr, ete = s2r._sub_eem(e, tr_ex), s2r._sub_eem(e, te_ex)
    fr = cal.fit_fluorophores(etr, gain="global", response_degree=2, n_starts=8, seed=s2r.SEED)
    ev = cal.evaluate_prediction(fr, ete, refit_gains=False)
    rows.append(dict(drop=name, r2_abs=ev["r2_abs"], sam_deg=ev["sam_mean_deg"],
                     peak_ratio=ev["peak_ratio_pred_over_meas"], n_params=fr.n_params,
                     param_names=fr.param_names, peak_brightness=float(np.nanmax(e.grid_value))))
    bands = {ex: np.asarray(ete.emission_nm[ex], float) for ex in te_ex}
    pred = cal.predict_eem(fr, bands, gains=ev["gains"])
    curves[name] = {str(ex): dict(wl=bands[ex].tolist(), meas=np.asarray(ete.value[ex]).tolist(),
                                  pred=np.asarray(pred[ex]).tolist()) for ex in te_ex}
    print(name, round(ev["r2_abs"], 3), round(ev["sam_mean_deg"], 2), flush=True)
def q(v):
    v = np.array(v); return dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                                 q75=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max()))
allpr = [abs(np.log(x)) for r in rows for x in r["peak_ratio"].values()]
perdrop_pe = [float(np.mean([abs(x - 1) for x in r["peak_ratio"].values()])) for r in rows]
res["drops"] = dict(n_drops=len(rows), train_ex=tr_ex, test_ex=te_ex, R2_abs=q([r["r2_abs"] for r in rows]),
                    SAM_deg=q([r["sam_deg"] for r in rows]), abs_ln_peak_ratio=q(allpr),
                    peak_amp_rel_error_per_drop_mean=q(perdrop_pe), rows=rows,
                    n_fitted_params=rows[0]["n_params"], fitted_param_names=rows[0]["param_names"])
(OUT / "drop_curves.json").write_text(json.dumps(curves))
# ---------------- (b) collagen ----------------
cubes, eems = s2r.load_collagen()
splits = {"A": ([310., 340., 385.], [325., 365., 400.]), "B": ([325., 365., 400.], [310., 340., 385.]),
          "C": ([310., 325., 340.], [365., 385., 400.]), "D": ([365., 385., 400.], [310., 325., 340.])}
out = {"twin_corrected": [], "twin_nominal": [], "flat": [], "empirical": [], "in_sample": []}
per = {}
for key, e in eems.items():
    per[key] = {k: [] for k in out}
    for sp, (tr, te) in splits.items():
        etr, ete = s2r._sub_eem(e, tr), s2r._sub_eem(e, te)
        b = cal.baseline_scores(etr, ete)
        vals = dict(flat=b["flat_sam_deg"], empirical=b["empirical_sam_deg"])
        for nm, deg in (("twin_corrected", 2), ("twin_nominal", 0)):
            fr = s2r._fit_set(etr, "ho", n_starts=8, deg=deg)
            vals[nm] = cal.evaluate_prediction(fr, ete, refit_gains=True)["sam_mean_deg"]
            if nm == "twin_corrected": npar = fr.n_params; pn = fr.param_names
        vals["in_sample"] = s2r._fit_set(ete, "ho", n_starts=8, deg=2).diagnostics["sam_mean_deg"]
        for k, v in vals.items():
            out[k].append(v); per[key][k].append(v)
    print(key, {k: round(float(np.mean(v)), 2) for k, v in per[key].items()}, flush=True)
def ms(v): return dict(mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v))
sub = lambda pref: {k: ms([x for kk in per if kk.startswith(pref) for x in per[kk][k]]) for k in out}
res["collagen"] = dict(n_cases=len(out["flat"]), splits=splits, sam_deg_all=dict((k, ms(v)) for k, v in out.items()),
                       sam_deg_pepsin=sub("pepsin"), sam_deg_acetic=sub("acetic"),
                       n_fitted_params_corrected=npar, fitted_param_names=pn,
                       per_roi_mean={k: {kk: float(np.mean(vv)) for kk, vv in d.items()} for k, d in per.items()})
(OUT / "validation_numbers.json").write_text(json.dumps(res, indent=1))
print("done")
