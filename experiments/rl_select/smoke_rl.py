"""Smoke test of the RL selector on lichens1 repeat0 fold0 (n=8): train rl_add and rl_addremove, deploy,
compare inner JSup against random sets. Usage: PYTHONPATH=src python experiments/rl_select/smoke_rl.py [seconds]"""
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

N = 8


def run(mode: str, seconds: float) -> str:
    import torch
    torch.set_num_threads(1)
    from rl_select.data import make_task
    from rl_select.objective import JSup
    from rl_select.rl_search import deploy_few_shot, deploy_label_free, train_policy

    t = make_task("lichens1", 0, 0)
    obj = JSup(t)
    t0 = time.time()
    pol = train_policy([(t, obj)], mode=mode, ns=(N,), dropout=0.2, max_seconds=seconds, seed=0,
                       out_dir=f"results/rl_select/smoke_{mode}", n_envs=64)
    wall = time.time() - t0
    info = pol.train_info
    log = info["log"]
    k = max(1, len(log) // 10)
    f0, f1 = np.mean([r["mean_final_J"] for r in log[:k]]), np.mean([r["mean_final_J"] for r in log[-k:]])
    e0, e1 = log[0]["entropy"], log[-1]["entropy"]
    out = [f"== rl_{mode} (lichens1 r0 f0, n={N}, JSup reward, label-free features, {wall:.0f}s) ==",
           f"updates={info['updates']} episodes={info['episodes']} steps={info['steps']} "
           f"steps/s={info['steps'] / wall:.1f} JSup evals={obj.n_evals} evals/s={obj.n_evals / wall:.1f}",
           f"mean final J (train episodes, 20% dropout): first {k} updates {f0:.4f} -> last {k} updates {f1:.4f}; "
           f"entropy {e0:.2f} -> {e1:.2f}"]
    rng = np.random.default_rng(0)
    rand = [obj(list(rng.choice(t.n_features, N, replace=False))) for _ in range(200)]
    out.append(f"random 8-sets JSup: mean {np.mean(rand):.4f} sd {np.std(rand):.4f} max {np.max(rand):.4f}")
    for ns in (1, 16):
        r = deploy_label_free(pol, t, N, mode=mode, n_samples=ns, seed=0)
        out.append(f"deploy label-free n_samples={ns}: JSup={obj(r.selected):.4f} set={r.selected}")
    n0 = obj.n_evals
    fo = JSup(t)
    r = deploy_few_shot(pol, t, N, fo, budget=100, mode=mode, mc_lookahead=(mode == "addremove"), seed=0)
    out.append(f"deploy few-shot (budget 100, mc={mode == 'addremove'}): JSup={r.inner_J:.4f} evals={r.n_evals}")
    return "\n".join(out)


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    res = Parallel(n_jobs=2)(delayed(run)(m, seconds) for m in ("add", "addremove"))
    text = "\n\n".join(res)
    print(text)
    p = Path("results/rl_select/smoke_rl.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text + "\n")
