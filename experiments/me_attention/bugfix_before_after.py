"""Before/after measurement of the review bug fixes on Collagen_Acetic_Acid (whole-specimen CV).

before: ORIGINAL code from commit 584ed77 (extracted into a scratch package, einops installed in a scratch dir)
after : fixed src/spectral_select/me_smartattention (no einops)

Usage (from the worktree root):
  PYTHONPATH=src python experiments/me_attention/bugfix_before_after.py            # runs all parts as parallel subprocesses
  PYTHONPATH=src python experiments/me_attention/bugfix_before_after.py --part before|after|report
Notebook config: DIM=16 DEPTH=3 HEADS=2 DIM_HEAD=8 MLP_DIM=4 DROPOUT=0.25 EMB_DROPOUT=0.10 EPOCHS=30 BATCH=256
SAMPLES_PER_SPECIMEN=2500 LR=5e-4 WD=0 gamma=0.9 steps=10 seed=42, device=cpu (same for both).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "me_attention" / "bugfix"
SCRATCH = Path("/private/tmp/claude-501/-Users-narekmeloyan-PycharmProjects-4D-Hyperspectral-Unsupervised-Clustering/"
               "85502d4d-a594-4266-ba88-05014eaaf9b8/scratchpad")
DATA_DIR = REPO / "Data" / "processed" / "Collagen_Acetic_Acid"
CFG = dict(dim=16, depth=3, heads=2, dim_head=8, mlp_dim=4, dropout=0.25, embedding_dropout=0.10,
           epochs=30, batch_size=256, samples_per_specimen=2500, lr=5e-4, weight_decay=0.0,
           scheduler_gamma=0.9, scheduler_steps=10)
DEVICE = "cpu"
EDGE_VAR = "zero"   # edge mode for the variance study, fixed a priori (SPEC default)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def fold_metrics(res, splits, data):
    """Uniform per-fold + pooled metrics from stored predictions (same function for before and after)."""
    from me_attention.protocol import metrics
    y, g = data["y"], data["groups"]
    nc = int(len(np.unique(y)))
    preds = np.concatenate([np.asarray(p) for p in res["predictions"]]) if isinstance(res["predictions"], list) \
        else np.asarray(res["predictions"])
    folds, o = [], 0
    spec_cls = data["specimen_classes"]
    for k, (tr, te) in enumerate(splits, 1):
        p = preds[o:o + len(te)]
        o += len(te)
        m = metrics(y[te], p, g[te], nc)
        m["fold"] = k
        m["test_specimens"] = sorted(int(s) for s in np.unique(g[te]))
        m["test_classes"] = sorted(int(c) for c in spec_cls[np.unique(g[te])])
        m["reported_macro_f1"] = res["folds"][k - 1]["macro_f1"]
        m["reported_pixel_acc"] = res["folds"][k - 1]["pixel_accuracy"]
        folds.append(m)
    te_all = np.concatenate([te for _, te in splits])
    pooled = metrics(y[te_all], preds, g[te_all], nc)
    keys = ["balanced_acc", "macro_f1", "pixel_acc", "specimen_vote_acc"]
    return dict(folds=folds, pooled=pooled,
                fold_mean={k: float(np.mean([f[k] for f in folds])) for k in keys},
                reported=dict(pixel_accuracy=res["pixel_accuracy"], macro_f1=res["macro_f1"],
                              specimen=f'{res["specimen_correct"]}/{res["specimen_total"]}'))


def slim(r):
    r = dict(r)
    for f in r["folds"]:
        f.pop("confusion", None)
    return r


def part_before():
    sys.path.insert(0, str(SCRATCH / "before_pkg"))
    sys.path.insert(0, str(SCRATCH / "before_deps"))
    from me_smart_before.data import load_dataset
    from me_smart_before.splits import make_specimen_splits
    from me_smart_before.training import run_smart_attention
    from me_smart_before.baseline import run_logistic_regression
    t0 = time.time()
    d = quiet(load_dataset, DATA_DIR)
    out = dict(code="584ed77 (original)", device=DEVICE, load_s=time.time() - t0,
               emission_axis=d["emissions"].tolist(), n_emission_positions=len(d["emissions"]),
               eem_mask_valid=int(d["eem_mask"].sum()))
    splits = make_specimen_splits(d["X_flat"], d["y"], d["groups"], seed=42)
    out["splits_seed42"] = [dict(test_specimens=sorted(int(s) for s in np.unique(d["groups"][te])),
                                 test_classes=sorted(int(c) for c in d["specimen_classes"][np.unique(d["groups"][te])]))
                            for _, te in splits]
    out["runs"] = {}
    for edge in ["circular", "zero", "discard"]:
        t0 = time.time()
        res = quiet(run_smart_attention, d, splits, name=f"before-{edge}", edge_mode=edge, seed=42,
                    device=DEVICE, **CFG)
        r = fold_metrics(res, splits, d)
        r["time_s"] = time.time() - t0
        out["runs"][f"attn_{edge}"] = slim(r)
        print("before", edge, r["pooled"]["balanced_acc"], r["pooled"]["pixel_acc"], f'{r["time_s"]:.0f}s', flush=True)
    res = quiet(run_logistic_regression, d["X_flat"], d["y"], d["groups"], splits, seed=42)
    out["runs"]["lr"] = slim(fold_metrics(res, splits, d))
    print("before lr", out["runs"]["lr"]["pooled"]["balanced_acc"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_part_before.json").write_text(json.dumps(out, indent=1))


def _after_common():
    assert "einops" not in sys.modules
    from spectral_select.me_smartattention.data import load_dataset
    from spectral_select.me_smartattention.splits import make_specimen_splits
    from spectral_select.me_smartattention.training import run_smart_attention
    from spectral_select.me_smartattention.baseline import run_logistic_regression
    d = quiet(load_dataset, DATA_DIR)
    assert "einops" not in sys.modules
    return d, make_specimen_splits, run_smart_attention, run_logistic_regression


def _comp(d, sp):
    return [dict(test_specimens=sorted(int(s) for s in np.unique(d["groups"][te])),
                 test_classes=sorted(int(c) for c in d["specimen_classes"][np.unique(d["groups"][te])]))
            for _, te in sp]


def part_after():
    """Main table: fixed code, split seed 42, model seed 42, edge modes zero/discard/circular + LR."""
    t0 = time.time()
    d, mk, run, run_lr = _after_common()
    out = dict(code="fixed", device=DEVICE, load_s=time.time() - t0, emission_axis=d["emissions"].tolist(),
               n_emission_positions=len(d["emissions"]), eem_mask_valid=int(d["eem_mask"].sum()))
    sp42 = mk(d["X_flat"], d["y"], d["groups"], seed=42)
    out["splits_seed42"] = _comp(d, sp42)
    out["runs"] = {}
    for edge in ["zero", "discard", "circular"]:
        t0 = time.time()
        res = quiet(run, d, sp42, name=f"after-{edge}", edge_mode=edge, seed=42, device=DEVICE, **CFG)
        r = fold_metrics(res, sp42, d)
        r["time_s"] = time.time() - t0
        out["runs"][f"attn_{edge}"] = slim(r)
        print("after", edge, r["pooled"]["balanced_acc"], r["pooled"]["pixel_acc"], f'{r["time_s"]:.0f}s', flush=True)
    res = quiet(run_lr, d["X_flat"], d["y"], d["groups"], sp42, seed=42)
    out["runs"]["lr"] = slim(fold_metrics(res, sp42, d))
    print("after lr", out["runs"]["lr"]["pooled"]["balanced_acc"], flush=True)
    (OUT / "_part_after.json").write_text(json.dumps(out, indent=1))


def part_unit(spec, edge):
    """One variance unit: 'seed:S' (model seed S, split seed 42) or 'repeat:S' (split seed S, model seed 42; + LR)."""
    kind, s = spec.split(":")
    s = int(s)
    d, mk, run, run_lr = _after_common()
    sp = mk(d["X_flat"], d["y"], d["groups"], seed=42 if kind == "seed" else s)
    res = quiet(run, d, sp, name=spec, edge_mode=edge, seed=s if kind == "seed" else 42, device=DEVICE, **CFG)
    out = dict(kind=kind, seed=s, edge_mode=edge, composition=_comp(d, sp), attn=slim(fold_metrics(res, sp, d)))
    if kind == "repeat":
        lr = quiet(run_lr, d["X_flat"], d["y"], d["groups"], sp, seed=42)
        out["lr"] = slim(fold_metrics(lr, sp, d))
    (OUT / f"_unit_{kind}_{s}.json").write_text(json.dumps(out, indent=1))
    print("unit", spec, out["attn"]["pooled"]["balanced_acc"], flush=True)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def part_report():
    b = json.loads((OUT / "_part_before.json").read_text())
    a = json.loads((OUT / "_part_after.json").read_text())
    a["best_edge_mode"] = EDGE_VAR
    a["seed_study"] = [json.loads((OUT / f"_unit_seed_{s}.json").read_text())["attn"] | {"seed": s} for s in range(42, 47)]
    a["repeat_study"] = []
    for s in range(42, 47):
        u = json.loads((OUT / f"_unit_repeat_{s}.json").read_text())
        a["repeat_study"].append(dict(split_seed=s, composition=u["composition"], attn=u["attn"], lr=u["lr"]))
    (OUT / "before_after.json").write_text(json.dumps(dict(before=b, after=a, config=CFG), indent=1))
    L = ["# Bug-fix before/after on Collagen_Acetic_Acid (whole-specimen CV, 3 folds)", "",
         f"Notebook config {CFG}; device {DEVICE}. All metrics computed by one function from stored predictions "
         "(macro-F1 with fixed labels). `spec vote` = specimens correct by majority vote / total, as a fraction.", "",
         "## Alignment (B1)", "",
         f"- before: emission axis has {b['n_emission_positions']} positions = indices 0..{b['n_emission_positions']-1} "
         f"(index fallback), {b['eem_mask_valid']} valid cells.",
         f"- after: {a['n_emission_positions']} real wavelengths, {a['emission_axis'][0]:.0f}-{a['emission_axis'][-1]:.0f} nm, "
         f"{a['eem_mask_valid']} valid cells.", "",
         "## Test-class composition per fold, seed 42 (B2)", "", "| fold | before (test specimens -> classes) | after |", "|---|---|---|"]
    for i, (x, y) in enumerate(zip(b["splits_seed42"], a["splits_seed42"]), 1):
        L.append(f"| {i} | {x['test_specimens']} -> {x['test_classes']} | {y['test_specimens']} -> {y['test_classes']} |")
    L += ["", "## Results, split seed 42, model seed 42 (pooled over folds; per-fold in JSON)", "",
          "| code | model | pixel acc | balanced acc | macro-F1 | spec vote | per-fold bal. acc | time s |", "|---|---|---|---|---|---|---|---|"]
    for tag, r in [("before", b), ("after", a)]:
        for k, v in r["runs"].items():
            p = v["pooled"]
            L.append(f"| {tag} | {k} | {pct(p['pixel_acc'])} | {pct(p['balanced_acc'])} | {pct(p['macro_f1'])} | "
                     f"{pct(p['specimen_vote_acc'])} | {' / '.join(pct(f['balanced_acc']) for f in v['folds'])} | "
                     f"{v.get('time_s', 0):.0f} |")
    L += ["", f"## Variance (after code, edge_mode={a['best_edge_mode']}, pre-registered default, not selected on test results)", "",
          "Model seeds 42-46 (split seed 42), pooled balanced accuracy %: "
          + ", ".join(pct(s["pooled"]["balanced_acc"]) for s in a["seed_study"])
          + f" -> mean {pct(np.mean([s['pooled']['balanced_acc'] for s in a['seed_study']]))} "
            f"+/- {pct(np.std([s['pooled']['balanced_acc'] for s in a['seed_study']], ddof=1))}", "",
          "| split seed | test-fold classes (all folds covered) | attention bal. acc | LR bal. acc | attention spec vote | LR spec vote |", "|---|---|---|---|---|---|"]
    for r in a["repeat_study"]:
        L.append(f"| {r['split_seed']} | {[c['test_classes'] for c in r['composition']]} | {pct(r['attn']['pooled']['balanced_acc'])} | "
                 f"{pct(r['lr']['pooled']['balanced_acc'])} | {pct(r['attn']['pooled']['specimen_vote_acc'])} | {pct(r['lr']['pooled']['specimen_vote_acc'])} |")
    A = [r["attn"]["pooled"]["balanced_acc"] for r in a["repeat_study"]]
    Lr = [r["lr"]["pooled"]["balanced_acc"] for r in a["repeat_study"]]
    L += ["", f"Over 5 fold assignments: attention {pct(np.mean(A))} +/- {pct(np.std(A, ddof=1))}, LR {pct(np.mean(Lr))} +/- {pct(np.std(Lr, ddof=1))} (balanced acc %).", ""]
    (OUT / "before_after.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["before", "after", "unit", "report", "all"], default="all")
    ap.add_argument("--unit", help="seed:S or repeat:S (with --part unit)")
    args = ap.parse_args()
    if args.part == "all":  # sequential driver; in practice the parts were launched as parallel processes
        procs = [subprocess.Popen([sys.executable, __file__, "--part", p]) for p in ["before", "after"]]
        procs += [subprocess.Popen([sys.executable, __file__, "--part", "unit", "--unit", f"{k}:{s}"])
                  for k in ["seed", "repeat"] for s in range(42, 47)]
        assert all(p.wait() == 0 for p in procs)
        part_report()
    elif args.part == "unit":
        part_unit(args.unit, EDGE_VAR)
    else:
        {"before": part_before, "after": part_after, "report": part_report}[args.part]()
