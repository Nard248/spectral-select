"""Random-pixel CV (Gevorg's make_pixel_splits) vs whole-specimen class-covering CV, on the dev set only.

lr, mlp, smartattention_v0 with default configs (frozen preprocessing: area/global/log1p), 3 folds, 1 repeat, cpu.
Training rows are capped at MAX_TRAIN (seeded uniform subsample) in BOTH schemes to keep the run cheap.
Run: nice -n 10 env PYTHONPATH=src python experiments/me_attention/pixel_vs_specimen.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.update({k: "2" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")})
import numpy as np
import torch

torch.set_num_threads(2)
from me_attention import data as D
from me_attention.models.registry import MODELS
from me_attention.protocol import metrics, outer_folds
from spectral_select.me_smartattention.splits import make_pixel_splits

OUT = Path(__file__).resolve().parents[2] / "results" / "me_attention" / "dev"
MAX_TRAIN = 40000
MODELS_RUN = ["lr", "mlp", "smartattention_v0"]


def run(d, tr, te, name, seed):
    rng = np.random.default_rng(seed)
    if len(tr) > MAX_TRAIN:
        tr = np.sort(rng.choice(tr, MAX_TRAIN, replace=False))
    spec = MODELS[name]
    cfg = dict(spec.default)
    if "n_jobs" in cfg or name in ("rf", "hgb"):
        cfg["n_jobs"] = 2
    est = spec.build(cfg, d.meta(), seed, "cpu")
    est.fit(d.X[tr], d.y[tr], d.groups[tr])
    p = est.predict_proba(d.X[te])
    m = metrics(d.y[te], p, d.groups[te], d.n_classes, whole_specimen=False)
    return m["balanced_acc"], m["macro_f1"], m["pixel_acc"]


def main():
    d = D.load("collagen_acetic")
    pix = [(np.asarray(a), np.asarray(b)) for a, b in make_pixel_splits(d.y, d.groups, n_folds=3, seed=42)]
    spc = outer_folds(d, 0)
    res = {"dataset": "collagen_acetic", "max_train": MAX_TRAIN, "folds": 3, "models": {}}
    for name in MODELS_RUN:
        res["models"][name] = {}
        for scheme, splits in (("pixel", pix), ("specimen", spc)):
            t0 = time.time()
            vals = [run(d, tr, te, name, 100 + i) for i, (tr, te) in enumerate(splits)]
            v = np.array(vals)
            res["models"][name][scheme] = dict(bacc_folds=v[:, 0].tolist(), bacc_mean=float(v[:, 0].mean()),
                                               bacc_std=float(v[:, 0].std(ddof=1)), f1_mean=float(v[:, 1].mean()),
                                               secs=time.time() - t0)
            print(name, scheme, f"{v[:, 0].mean():.3f}", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "pixel_vs_specimen.json").write_text(json.dumps(res, indent=1))
    L = ["# Random-pixel vs whole-specimen CV (collagen_acetic, dev set)", "",
         "Balanced accuracy, 3 folds x 1 repeat, default configs, preprocessing area/global/log1p. "
         "Pixel = balanced random pixel split (make_pixel_splits, pixels of every specimen in train and test); "
         f"specimen = whole-specimen class-covering folds. Train rows capped at {MAX_TRAIN}.", "",
         "| model | random-pixel | whole-specimen | gap |", "|---|---|---|---|"]
    for n, r in res["models"].items():
        a, b = r["pixel"], r["specimen"]
        L.append(f"| {n} | {a['bacc_mean']*100:.1f} +- {a['bacc_std']*100:.1f} | {b['bacc_mean']*100:.1f} +- "
                 f"{b['bacc_std']*100:.1f} | {100*(a['bacc_mean']-b['bacc_mean']):+.1f} |")
    (OUT / "pixel_vs_specimen.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
