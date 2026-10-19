"""Dev exploration on collagen_acetic ONLY (engineer E).  Usage:
  python experiments/me_attention/dev_explore.py run STAGE WORKER NWORKERS
  python experiments/me_attention/dev_explore.py board
Config = dict(name, kind, cfg, mode).  kind: 'sk:<baseline>' | 'nn:<baseline>' | 'att:<name>' | 'v2:<hybrid|mlp|cnn1d|meas_tokens|ex_tokens|...>'.
mode: 'inner' (fit inner-train, early stop on inner-val, test outer fold)  |  'full' (fit on full outer-train, fixed epochs).
"""
from __future__ import annotations

import json, os, sys, time, itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)

from me_attention import protocol
from me_attention.data import load
from me_attention.models import attention as A
from me_attention.models import attention_v2 as V
from me_attention.models import baselines_spectral as B

OUT = ROOT / "results/me_attention/dev"
JSONL = OUT / "dev_results.jsonl"
DEV = "collagen_acetic"
BASE = dict(samples_per_specimen=1000, epochs=40, patience=8)
FULL_EPOCHS = 25


def make(kind, cfg, meta, seed, device):
    cfg = dict(cfg)
    fam, name = kind.split(":")
    if fam in ("sk", "nn") and name in B.SPECTRAL_BASELINES:
        spec = B.SPECTRAL_BASELINES[name]
        c = {**spec.default, **cfg}
        if fam == "sk":
            return spec.build(c, meta, seed, device)
        mod = dict(mlp=B.MLP, cnn1d=B.CNN1D, spectralformer=B.SpectralFormer, bandvit=B.BandViT)[name]
        return V.TorchEstimatorV2(lambda m, cc: mod(m, cc), {**c, **BASE, **cfg}, meta, seed, device)
    if fam == "att":
        fac = A.SPECTRAL_FACTORIES[name]
        dflt = dict(smartattention_v0=A.V0_DEFAULT, meas_tokens=A.MEAS_DEFAULT, axial=A.AXIAL_DEFAULT, ex_tokens=A.EX_DEFAULT)[name]
        return V.TorchEstimatorV2(fac, {**dflt, **BASE, **cfg}, meta, seed, device)
    if fam == "v2" and name == "hybrid":
        return V.TorchEstimatorV2(V.hybrid_factory, {**BASE, **cfg}, meta, seed, device)
    raise ValueError(kind)


def run_config(c, data, device):
    units = []
    t0 = time.time()
    for r in (0, 1):
        for f, (tr, te) in enumerate(protocol.outer_folds(data, r)):
            seed = 100 * r + f
            cfg = dict(c["cfg"])
            if c.get("mode", "inner") == "inner":
                itr, iva = protocol.inner_split(data, tr, seed)
                Xv, yv = data.X[iva], data.y[iva]
            else:
                itr, Xv, yv = tr, None, None
                cfg.update(fixed_epochs=True, epochs=FULL_EPOCHS)
            if c["kind"].startswith("nn:") or c["kind"].startswith("att:") or c["kind"].startswith("v2:"):
                cfg = {**BASE, **cfg}
                if c.get("mode") == "full":
                    cfg["epochs"] = FULL_EPOCHS
            est = make(c["kind"], cfg, data.meta(), seed, device)
            t1 = time.time()
            est.fit(data.X[itr], data.y[itr], data.groups[itr], Xv, yv)
            P = est.predict_proba(data.X[te])
            m = protocol.metrics(data.y[te], P, data.groups[te], data.n_classes)
            units.append(dict(repeat=r, fold=f, balanced_acc=m["balanced_acc"], macro_f1=m["macro_f1"],
                              specimen_avg_acc=m["specimen_avg_acc"], pixel_acc=m["pixel_acc"],
                              best_epoch=getattr(est, "best_epoch_", None), sec=time.time() - t1,
                              degenerate=getattr(est, "degenerate_", False)))
    return dict(name=c["name"], kind=c["kind"], cfg=c["cfg"], mode=c.get("mode", "inner"), units=units,
                time=time.time() - t0, stage=c.get("stage"))


def done_names():
    if not JSONL.exists():
        return set()
    return {json.loads(l)["name"] for l in JSONL.read_text().splitlines() if l.strip()}


def board():
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    seen = {}
    for r in rows:
        seen[r["name"]] = r
    out = []
    for r in seen.values():
        u = r["units"]
        g = lambda k: (np.mean([x[k] for x in u]), np.std([x[k] for x in u]))
        out.append((g("balanced_acc"), g("macro_f1"), g("specimen_avg_acc"), r))
    out.sort(key=lambda t: -t[0][0])
    lines = ["# Dev leaderboard (collagen_acetic, 6 units = 2 repeats x 3 folds; mean +- std)", "",
             "| # | name | mode | bAcc | macroF1 | spec-avg acc | s/run |", "|---|---|---|---|---|---|---|"]
    for i, (b, f, s, r) in enumerate(out, 1):
        lines.append(f"| {i} | {r['name']} | {r['mode']} | {b[0]*100:.1f} +- {b[1]*100:.1f} | {f[0]*100:.1f} +- {f[1]*100:.1f} | "
                     f"{s[0]*100:.1f} +- {s[1]*100:.1f} | {r['time']:.0f} |")
    (OUT / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "board":
        board()
        print((OUT / "LEADERBOARD.md").read_text())
    else:
        stage, wi, nw = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        import importlib.util
        spec = importlib.util.spec_from_file_location("stages", Path(__file__).with_name("dev_stages.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        cfgs = [dict(c, stage=stage) for c in getattr(mod, stage)()]
        device = os.environ.get("DEV_DEVICE", "mps")
        data = load(DEV)
        mine = cfgs[wi::nw]
        for c in mine:
            if c["name"] in done_names():
                continue
            try:
                res = run_config(c, data, "cpu" if c["kind"].startswith("sk:") else device)
            except Exception as e:  # keep the sweep alive
                import traceback; traceback.print_exc()
                continue
            with open(JSONL, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            b = np.mean([u["balanced_acc"] for u in res["units"]])
            print(f"[{stage}:{wi}] {c['name']:50s} bacc={b*100:.1f} t={res['time']:.0f}s", flush=True)
