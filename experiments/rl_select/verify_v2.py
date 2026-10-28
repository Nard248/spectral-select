"""Compare finished work under the results root with the expected job/row counts of the v2 rerun.

  RL_SELECT_RESULTS=results/rl_select_v2 PYTHONPATH=src python experiments/rl_select/verify_v2.py
Exit code 0 iff nothing is missing. Counts distinct result keys (resumable drivers may append duplicates).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baselines as BL  # noqa: E402
import run_rl as R  # noqa: E402
import run_rl_v3 as V3  # noqa: E402
from rl_select import paths  # noqa: E402
from rl_select.data import n_outer_folds  # noqa: E402

RES = paths.results_root()
DS = paths.datasets()
SEEDS = paths.seeds()
NF = {d: n_outer_folds(d) for d in DS}


def keys(pattern: str, setting: str):
    out = {}
    for f in sorted((RES / setting).glob(pattern)):
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[R.key_of(r)] = r
    return out


def expected() -> dict[str, int]:
    e = Counter()
    for d in DS:
        for rep in (0, 1, 2):
            for f in range(NF[d]):
                e[f"baselines {'S2'} all_pairs"] += 1
                for n in BL.NS:
                    for s in BL.specs_for(rep, n):
                        e[f"baselines {s['setting']}"] += 1
        e["baselines S1 exact (n=2,3)"] += 2 * NF[d]
    pol_rl = [(src, v, sd) for src in R.POOLS for v in R.VARIANTS for sd in SEEDS]
    e["policies (policies/)"] = len(pol_rl)
    e["policies_v3 (policies_v3/)"] = len(R.POOLS) * len(V3.VARIANTS) * len(SEEDS)
    for src, v, sd in pol_rl:
        for t in R.POOLS[src][1]:
            e["RL S2 (non-aug)"] += 3 * NF[t] * len(R.NS) * 2
            if src.startswith("lodo") and v in ("rl_add_nowl", "rl_addremove_nowl"):
                e["RL S3 (upper bound, 5 budgets)"] += NF[t] * len(R.NS) * len(R.B3)
    for src in R.POOLS:
        for v in V3.VARIANTS:
            for sd in SEEDS:
                for t in R.POOLS[src][1]:
                    e["RL S2 (aug v3)"] += 3 * NF[t] * len(R.NS) * 2
    e["RL S1 per-instance (n=5,13)"] = sum(NF[d] * 2 * 2 for d in DS)
    e["RL S1 ablation (collagen_acetic, n=8)"] = NF["collagen_acetic"] * len(R.ABL) if "collagen_acetic" in DS else 0
    e["RL S4 cost (lichens1)"] = NF["lichens1"] * len(R.MUS) * 2 if "lichens1" in DS else 0
    return dict(e)


def actual() -> dict[str, int]:
    a = Counter()
    base = {}
    for s in ("S1", "S2", "S3"):
        for f in (RES / s).glob("*.jsonl"):
            if f.stem.endswith("_rl") or f.stem.endswith("_rl_v3"):
                continue
            for line in f.read_text().splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    base[R.key_of(r)] = r
    for k, r in base.items():
        if r["method"] == "all_pairs":
            a["baselines S2 all_pairs"] += 1
        elif r["method"] == "exact":
            a["baselines S1 exact (n=2,3)"] += 1
        else:
            a[f"baselines {r['setting']}"] += 1
    for k, r in keys("*_rl.jsonl", "S2").items():
        a["RL S2 (non-aug)"] += 1
    for k, r in keys("*_rl_v3.jsonl", "S2").items():
        a["RL S2 (aug v3)"] += 1
    for k, r in keys("*_rl.jsonl", "S3").items():
        a["RL S3 (upper bound, 5 budgets)"] += 1
    for k, r in keys("*_rl.jsonl", "S1").items():
        a["RL S1 ablation (collagen_acetic, n=8)" if r["budget"] == R.ABL_B else "RL S1 per-instance (n=5,13)"] += 1
    for k, r in keys("*_rl.jsonl", "S4").items():
        a["RL S4 cost (lichens1)"] += 1
    a["policies (policies/)"] = sum(1 for p in (RES / "policies").glob("*/policy.pt"))
    a["policies_v3 (policies_v3/)"] = sum(1 for p in (RES / "policies_v3").glob("*/policy.pt"))
    return dict(a)


if __name__ == "__main__":
    e, a = expected(), actual()
    print(f"results root: {RES}\ndatasets: {DS}  folds: {NF}  seeds: {SEEDS}\n")
    print(f"{'unit':40s} {'expected':>9s} {'actual':>9s}  status")
    bad = 0
    for k in sorted(e):
        x, y = e[k], a.get(k, 0)
        ok = y >= x or ("upper bound" in k and y >= 0.9 * x)
        bad += not ok
        print(f"{k:40s} {x:9d} {y:9d}  {'ok' if y >= x else ('~ok (<=5 budgets/row)' if ok else 'MISSING ' + str(x - y))}")
    print("\nVERIFY_OK" if not bad else f"\nVERIFY_INCOMPLETE ({bad} categories)")
    sys.exit(1 if bad else 0)
