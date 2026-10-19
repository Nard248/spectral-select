"""Build results/rl_select/summary/digest_rl_v3.md from S2 jsonl files."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[2] / "results" / "rl_select"
DS = ("lichens1", "lichens2", "collagen_acetic", "collagen_pepsin")
NS = (3, 5, 8, 13)
V3 = ["rl_add_aug", "rl_addremove_aug", "rl_addremove_css_aug", "rl_addremove_mix_aug"]
V2OF = {"rl_add_aug": "rl_add_nowl", "rl_addremove_aug": "rl_addremove_nowl",
        "rl_addremove_css_aug": "rl_addremove_css_nowl", "rl_addremove_mix_aug": "rl_addremove_nowl"}
SCEN = {"lichens1": ["C2L", "lodo"], "lichens2": ["C2L", "lodo"], "collagen_acetic": ["L2C", "lodo"],
        "collagen_pepsin": ["L2C", "lodo"]}


def rows(path):
    if not path.exists():
        return
    for l in path.read_text().splitlines():
        if l.strip():
            yield json.loads(l)


def load(ds):
    """unit=(repeat,fold,n) -> method -> kNN BA (mean over seeds for baselines)."""
    tmp = defaultdict(lambda: defaultdict(list))
    for fn in (f"{ds}.jsonl", f"{ds}_rl.jsonl", f"{ds}_rl_v3.jsonl"):
        for r in rows(RES / "S2" / fn):
            if r.get("seed", 0) != 0 and "@" in r["method"]:
                continue
            tmp[(r["repeat"], r["fold"], r["n"])][r["method"]].append(r["eval"]["knn"]["balanced_acc"])
    return {u: {m: float(np.mean(v)) for m, v in d.items()} for u, d in tmp.items()}


def ms(vals):
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if len(vals) else "-"


def wl(a, b):
    w = sum(x > y + 1e-9 for x, y in zip(a, b)); l = sum(x < y - 1e-9 for x, y in zip(a, b))
    return f"{w}W/{l}L/{len(a) - w - l}T"


def main():
    out = ["# RL v3 (task-augmented meta-training) vs v2 vs label-free baselines",
           "", "Outer-test kNN balanced accuracy, label-free deployment, repeats 0-2 x all outer folds "
           "(mean±std over units). `_k16` = best-of-16 by JRecon. Win/loss counts are per (repeat, fold) unit "
           "(ties within 1e-9). v2 counterpart: add->rl_add_nowl, addremove & mix->rl_addremove_nowl, "
           "css->rl_addremove_css_nowl.", ""]
    summ = defaultdict(lambda: defaultdict(list))   # (variant, kind) -> list of (n, w, l)
    for ds in DS:
        D = load(ds)
        for sc in SCEN[ds]:
            have = [v for v in V3 if any(f"{v}@{sc}" in m for m in D.values())]
            if not have:
                continue
            out += [f"## {ds} ({sc})", ""]
            cols = ["random", "uniform_grid", "css_greedy"] + [f"{m}@{sc}" for m in dict.fromkeys(V2OF[v] for v in have)]
            cols += [f"{v}{k}@{sc}" for v in have for k in ("", "_k16")]
            out += ["| n | units | " + " | ".join(c.replace(f"@{sc}", "") for c in cols) + " |",
                    "|---|---|" + "---|" * len(cols)]
            for n in NS:
                us = [u for u in D if u[2] == n]
                cells = []
                for c in cols:
                    vals = [D[u][c] for u in us if c in D[u]]
                    cells.append(ms(vals) + ("" if len(vals) == len(us) else f" (n={len(vals)})"))
                out.append(f"| {n} | {len(us)} | " + " | ".join(cells) + " |")
            out += ["", "Paired W/L/T (variant vs reference), all units pooled per n:", "",
                    "| variant | n | vs css_greedy | vs random | vs v2 counterpart | vs add-only(aug) |", "|---|---|---|---|---|---|"]
            for v in have:
                for k in ("", "_k16"):
                    for n in NS:
                        us = [u for u in D if u[2] == n and f"{v}{k}@{sc}" in D[u]]
                        if not us:
                            continue
                        a = [D[u][f"{v}{k}@{sc}"] for u in us]
                        cs = wl(a, [D[u]["css_greedy"] for u in us])
                        rd = wl(a, [D[u]["random"] for u in us])
                        v2n = f"{V2OF[v]}{k}@{sc}"
                        u2 = [u for u in us if v2n in D[u]]
                        v2 = wl([D[u][f"{v}{k}@{sc}"] for u in u2], [D[u][v2n] for u in u2]) if u2 else "-"
                        ad = "-"
                        if v != "rl_add_aug" and "rl_add_aug" in have:
                            u3 = [u for u in us if f"rl_add_aug{k}@{sc}" in D[u]]
                            ad = wl([D[u][f"{v}{k}@{sc}"] for u in u3], [D[u][f"rl_add_aug{k}@{sc}"] for u in u3]) if u3 else "-"
                        out.append(f"| {v}{k} | {n} | {cs} | {rd} | {v2} | {ad} |")
            out.append("")
    # pooled summary: mean kNN BA over all units/targets per scenario
    out += ["## Pooled mean kNN BA over all units of each scenario (strict cross-family C2L/L2C; LODO)", "",
            "| scenario | n | random | css_greedy | " + " | ".join(V3) + " | v2 add_nowl | v2 addremove_nowl |", "|---|---|" + "---|" * (4 + len(V3))]
    for scn in ("cross", "lodo"):
        for n in NS:
            vals = defaultdict(list)
            for ds in DS:
                D = load(ds)
                sc = [s for s in SCEN[ds] if (s != "lodo") == (scn == "cross")][0]
                for u, d in D.items():
                    if u[2] != n:
                        continue
                    for m, col in (("random", "random"), ("css_greedy", "css_greedy"),
                                   (f"rl_add_nowl@{sc}", "v2a"), (f"rl_addremove_nowl@{sc}", "v2ar")):
                        if m in d:
                            vals[col].append(d[m])
                    for v in V3:
                        if f"{v}@{sc}" in d:
                            vals[v].append(d[f"{v}@{sc}"])
            order = ["random", "css_greedy"] + V3 + ["v2a", "v2ar"]
            out.append(f"| {scn} | {n} | " + " | ".join(f"{np.mean(vals[c]):.3f}" if vals[c] else "-" for c in order) + " |")
    p = RES / "summary" / "digest_rl_v3.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(out) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
