"""Tiny end-to-end smoke of the v2 pipeline (minutes): baselines unit, RL train (3 PPO updates), S2, S3, v3 augmented.

  PYTHONPATH=src python experiments/rl_select/smoke_v2.py [--keep]
Runs on collagen_acetic + collagen_pepsin + lichens2_sub3 (the Lichens 1 cache is never loaded), n = 3, outer fold 0
only, into a temporary results root that is deleted at the end (unless --keep). Does not touch results/rl_select*.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="rl_v2_smoke_"))
os.environ.update(RL_SELECT_RESULTS=str(tmp), RL_SELECT_DATASETS="collagen_acetic,collagen_pepsin,lichens2_sub3",
                  RL_SELECT_SEEDS="0", RL_SELECT_WORKERS="1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import run_baselines as BL  # noqa: E402
import run_rl as R  # noqa: E402
import run_rl_v3 as V3  # noqa: E402
from rl_select import data  # noqa: E402

assert R.RES == tmp and R.DATASETS == ("collagen_acetic", "collagen_pepsin", "lichens2_sub3"), (R.RES, R.DATASETS)
data.n_outer_folds = lambda key: 1            # 1 outer fold per dataset; all_tasks / job_s2 / job_s3 see this
R.N_UPDATES = 3
R.NS = (3,)
R.B3 = (25,)
BL.B1, BL.B3, BL.NS = 100, (25,), (3,)
for sub in ("S1", "S2", "S3", "S4"):
    (tmp / sub).mkdir(parents=True, exist_ok=True)

try:
    # 1. baselines: one (dataset, repeat, fold, n) unit incl. S1/S2/S3 wrappers at tiny budgets
    rows = BL.unit(("lichens2_sub3", 0, 0, 3, set(), {"s3_seeds": 1, "ga_seeds": 1, "s2_rf": False}))
    BL.write_rows(rows)
    print("baseline rows:", len(rows))
    # 2. RL training (3 PPO updates) cross-family C2L (collagen -> lichens2_sub3) + lodo-lichens2_sub3 for S3
    for src, var in (("C2L", "rl_addremove_nowl"), ("lodo-lichens2_sub3", "rl_add_nowl")):
        R.run_job("train", (src, var, 0))
        assert R.policy_done(R.POL / R.pol_name(src, var, 0)), src
    # 3. S2 deployment (label-free) and S3 few-shot
    r2, _ = R.run_job("s2", ("C2L", "rl_addremove_nowl", 0, "lichens2_sub3", 0, set()))
    R.write_rows(r2)
    r3, _ = R.run_job("s3", ("lodo-lichens2_sub3", "rl_add_nowl", 0, "lichens2_sub3", set()))
    R.write_rows(r3)
    print("S2 rows", len(r2), "S3 rows", len(r3))
    assert r2 and r3 and all(r["dataset"] == "lichens2_sub3" and r["fold"] == 0 for r in r2 + r3)
    # 4. v3 (augmented) training + deployment
    V3.POL3.mkdir(parents=True, exist_ok=True)
    V3.run_job("train", ("C2L", "rl_addremove_aug", 0, 3))
    rv, _ = V3.run_job("s2", ("C2L", "rl_addremove_aug", 0, "lichens2_sub3", 0, set()))
    V3.write_rows(rv)
    print("v3 rows", len(rv))
    assert rv
    n_jsonl = sum(1 for f in tmp.rglob("*.jsonl") for _ in open(f))
    print(f"SMOKE_OK: {n_jsonl} jsonl rows under {tmp}")
finally:
    if "--keep" not in sys.argv:
        shutil.rmtree(tmp, ignore_errors=True)
        print("temp results removed")
