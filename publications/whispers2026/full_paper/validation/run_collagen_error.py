"""Collagen held-out error metrics (R2 shape, relative RMS) + representative-case curves for Fig. 5b.
Same protocol/seeds/fits as run_validation.py (twin_corrected: per-excitation gain refit on the held-out
excitations, response_degree=2, n_starts=8). Adds keys to validation_numbers.json (nothing removed)."""
import json, sys, importlib.util, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("s2r", ROOT / "scripts" / "studio_sim2real.py")
s2r = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2r)
from spectraforge.studio import calibration as cal
OUT = Path(__file__).parent
cubes, eems = s2r.load_collagen()
splits = {"A": ([310., 340., 385.], [325., 365., 400.]), "B": ([325., 365., 400.], [310., 340., 385.]),
          "C": ([310., 325., 340.], [365., 385., 400.]), "D": ([365., 385., 400.], [310., 325., 340.])}
cases = []
for key, e in eems.items():
    for sp, (tr, te) in splits.items():
        etr, ete = s2r._sub_eem(e, tr), s2r._sub_eem(e, te)
        fr = s2r._fit_set(etr, "ho", n_starts=8, deg=2)
        ev = cal.evaluate_prediction(fr, ete, refit_gains=True)
        bands = {ex: np.asarray(ete.emission_nm[ex], float) for ex in te}
        pred = cal.predict_eem(fr, bands, gains=ev["gains"])
        c = dict(roi=key, split=sp, sam=ev["sam_mean_deg"], r2=ev["r2"],
                 r2_ex_mean=float(np.mean(list(ev["r2_per_ex"].values()))), rms=ev["rms_rel"], curves={})
        for ex in te:
            m = np.asarray(ete.value[ex], float); p = np.asarray(pred[ex], float)
            idx = np.array([int(np.argmin(np.abs(bands[ex] - b))) for b in bands[ex]])
            c["curves"][str(ex)] = dict(wl=bands[ex].tolist(), meas=m.tolist(), pred=p.tolist())
        cases.append(c); print(key, sp, round(c["sam"], 2), round(c["r2"], 3), round(c["rms"], 3), flush=True)
def ms(v): return dict(mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v))
sam = [c["sam"] for c in cases]; med = float(np.median(sam))
rep = min(cases, key=lambda c: abs(c["sam"] - med))
res = json.load(open(OUT / "validation_numbers.json"))
res["collagen"]["error_twin_corrected"] = dict(
    r2_shape=ms([c["r2"] for c in cases]), r2_shape_mean_per_excitation=ms([c["r2_ex_mean"] for c in cases]),
    rel_rms=ms([c["rms"] for c in cases]), sam_deg_check=ms(sam),
    representative=dict(roi=rep["roi"], split=rep["split"], sam_deg=rep["sam"], r2=rep["r2"], rel_rms=rep["rms"],
                        median_sam_all=med),
    how=("Twin_corrected fits (response_degree=2, n_starts=8, seed as run_validation.py) on 3 training excitations, "
         "evaluated on the 3 held-out excitations of each of 4 splits x 6 ROIs = 24 cases; per-excitation nuisance gain "
         "refitted on the held-out spectra (shape only). Each spectrum normalised by its own peak. R2 = 1 - SSres/SStot "
         "pooled over the 3 held-out excitations (calibration._diagnostics 'r2'); rel_rms = RMS of (measured-predicted)/peak, "
         "averaged over the 3 excitations ('rms_rel'). mean +- SD (population SD) over 24 cases."))
(OUT / "validation_numbers.json").write_text(json.dumps(res, indent=1))
(OUT / "collagen_rep_curves.json").write_text(json.dumps(dict(roi=rep["roi"], split=rep["split"], sam=rep["sam"],
    r2=rep["r2"], rms=rep["rms"], curves=rep["curves"])))
print(res["collagen"]["error_twin_corrected"])
