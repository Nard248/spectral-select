"""Prototype figure: one real label-free episode of the transferred add/remove policy (reversible mechanism).

Policy C2L_rl_addremove_nowl_s0 (trained on collagen, wavelength-free features) deployed greedily and label-free on
lichens1, repeat 0, fold 0, n = 8 (the paper's headline S2 setting, n_samples = 1), against the matched add-only
policy C2L_rl_add_nowl_s0. Every step is recorded; J_sup (train specimens) and outer kNN balanced accuracy are
computed after the fact for the figure only and never reach the policy.

  PYTHONPATH=src .venv/bin/python experiments/rl_select/episode_figure.py [--fold F --n N --repeat R]

Writes publications/rl_band_selection/figures/proto/{fig_episode.pdf,.png,fig_episode_steps.csv,fig_episode_swaps.csv}.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"                     # same thread setting as run_rl.py (bit-reproducible with the stored rows)
os.environ["ME_ATTENTION_RECORDED_WL"] = "1"

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "rl_select"
OUT = ROOT / "publications" / "rl_band_selection" / "figures" / "proto"
TARGET = "lichens1"
NS = (3, 5, 8, 13)
POL_AR, POL_ADD = "C2L_rl_addremove_nowl_s0", "C2L_rl_add_nowl_s0"
M_AR, M_ADD = "rl_addremove_nowl@C2L", "rl_add_nowl@C2L"
EVAL_SEED = 0                                 # run_rl.Evals uses evaluate(..., seed=0)

# colours (Okabe-Ito based, as in paper_figures.py)
C_REC, C_BA, C_ADD, C_REM = "#0072B2", "#222222", "#009E73", "#D7191C"
C_SEL, C_ADM = "#243447", "#d9d9d9"


# --------------------------------------------------------------------------------------------- rollout
def rollout(policy, tf, n, mode, init="empty", seed=0):
    """Greedy rollout identical to rl_search.deploy_label_free(n_samples=1); returns the step list."""
    from rl_select.env import EnvConfig, EpSpec, VecSelectEnv
    from rl_select.policy import obs_to_tensors
    ecfg = EnvConfig(mode=mode, init=init, reward="terminal", dropout=0.0, T=None, ns=(n,))
    venv = VecSelectEnv([(tf, None)], ecfg, seed)
    torch.manual_seed(seed)
    venv.reset([EpSpec(0, n, 0.0)])
    ep = venv.episodes[0]
    F = tf.F
    steps = []
    while not ep.done:
        obs = venv.observe()
        with torch.no_grad():
            logits, _ = policy(*obs_to_tensors(obs))
        probs = torch.softmax(logits, -1)[0].numpy()
        a = int(logits.argmax(-1)[0])
        phase, constructing, cycle = ep.phase, ep.constructing, ep.cycle
        p_stop = float(probs[F]) if ep.stop_allowed() else float("nan")
        venv.step(obs.env_idx, np.array([a]))
        kind = "stop" if a == F else ("add" if phase == 0 else "remove")
        steps.append(dict(step=len(steps) + 1, kind=kind, pair=None if a == F else a, p_action=float(probs[a]),
                          p_stop=p_stop, phase="construction" if constructing else "refinement", cycle=cycle,
                          S=sorted(int(i) for i in ep.order)))
    return steps


def annotate(task, tf, steps, obj_sup, em_rec):
    """Add J_rec (what the policy sees), J_sup (train specimens) and outer kNN BA to every step."""
    from rl_select.evaluate import evaluate
    for s in steps:
        S = s["S"]
        s["size"] = len(S)
        s["J_rec"] = float(tf.jrecon(S))
        s["J_sup"] = float(obj_sup(S))
        s["knn_ba"] = float(evaluate(task, S, ("knn",), seed=EVAL_SEED)["knn"]["balanced_acc"])
        s["ex_nm"] = None if s["pair"] is None else float(task.ex_nm[s["pair"]])
        s["em_nm"] = None if s["pair"] is None else float(em_rec[s["pair"]])           # recorded axis
        s["em_cache_nm"] = None if s["pair"] is None else float(task.em_nm[s["pair"]])  # label in the RL cache
    return steps


def stored_record(method, repeat, fold, n, seed=0):
    for line in open(RES / "S2" / f"{TARGET}_rl.jsonl"):
        r = json.loads(line)
        if (r["method"], r["repeat"], r["fold"], r["n"], r["seed"]) == (method, repeat, fold, n, seed):
            return r
    raise KeyError((method, repeat, fold, n))


# --------------------------------------------------------------------------------------------- recorded EEM grid
def eem_grid(task):
    """Recorded (ex, em) coordinates of the F candidates; asserts they match the RL cache feature order."""
    from me_attention.data import RECORDED_WL_OFFSET, load
    d = load(TARGET)
    assert d.ex_nm.shape == task.ex_nm.shape == (192,), (d.ex_nm.shape, task.ex_nm.shape)
    assert np.array_equal(d.ex_nm, task.ex_nm), "excitation order differs between me_attention and the RL cache"
    off = np.array([RECORDED_WL_OFFSET[TARGET].get(float(e), 0.0) for e in task.ex_nm])
    assert np.allclose(d.em_nm, task.em_nm + off), "recorded emission != cached emission + audit offset"
    ex, em = d.ex_nm.astype(float), d.em_nm.astype(float)
    uex, uem = np.unique(ex), np.unique(em)
    ie, im = np.searchsorted(uex, ex), np.searchsorted(uem, em)
    assert len({(a, b) for a, b in zip(ie, im)}) == len(ex), "two candidates share a grid cell"
    return uex, uem, ie, im, em


# --------------------------------------------------------------------------------------------- figure
def fig_episode(task, steps_ar, steps_add, n, out):
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    uex, uem, ie, im, _ = eem_grid(task)
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "font.family": "sans-serif", "lines.linewidth": 1.1})
    # snapshots: after construction, then after each completed swap (remove + add)
    cons_end = max(s["step"] for s in steps_ar if s["phase"] == "construction")
    snaps = [dict(step=cons_end, S=steps_ar[cons_end - 1]["S"], added=None, removed=None,
                  J=steps_ar[cons_end - 1]["J_rec"], label="construction")]
    k = 0
    for i, s in enumerate(steps_ar):
        if s["kind"] == "remove":
            k += 1
            a = steps_ar[i + 1]
            snaps.append(dict(step=a["step"], S=a["S"], added=a["pair"], removed=s["pair"], J=a["J_rec"],
                              label=f"swap {k}", rem=s, add=a))
    ns = len(snaps)

    W, H = 7.16, 3.2
    fig = plt.figure(figsize=(W, H))

    def axin(x0, y0, w, h):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H])

    # ---------------- (a) snapshots, manual layout in inches
    ax0, aw = 0.40, 2.86
    title_h, gap, foot = 0.14, 0.05, 0.50           # title row, gap between snapshots, legend + xticks + xlabel
    gh = min(0.74, (H - 0.04 - foot - ns * (title_h + gap)) / ns)
    cmap = ListedColormap(["#ffffff", C_ADM, C_SEL, C_ADD])
    shape = (len(uex), len(uem))
    for r, sn in enumerate(snaps):
        y0 = H - 0.04 - (r + 1) * (gh + title_h + gap) + gap
        ax = axin(ax0, y0, aw, gh)
        code = np.zeros(shape, int)
        code[ie, im] = 1
        sel = np.array(sn["S"])
        code[ie[sel], im[sel]] = 2
        if sn["added"] is not None:
            code[ie[sn["added"]], im[sn["added"]]] = 3
        ax.pcolormesh(np.arange(shape[1] + 1) - 0.5, np.arange(shape[0] + 1) - 0.5, code, cmap=cmap, vmin=0, vmax=3,
                      edgecolors="white", linewidth=0.5)
        if sn["removed"] is not None:
            ax.add_patch(Rectangle((im[sn["removed"]] - 0.5, ie[sn["removed"]] - 0.5), 1, 1, fill=False, ec=C_REM,
                                   lw=1.5, zorder=5))
        ax.set_xlim(-0.5, shape[1] - 0.5)
        ax.set_ylim(-0.5, shape[0] - 0.5)
        ax.set_yticks(range(shape[0]))
        ax.set_yticklabels([f"{int(v)}" for v in uex])
        ax.set_xticks(range(0, shape[1], 5))
        if r == ns - 1:
            ax.set_xticklabels([f"{int(uem[i])}" for i in range(0, shape[1], 5)])
            ax.set_xlabel("emission (nm)", labelpad=1.0)
        else:
            ax.set_xticklabels([])
        ax.tick_params(length=1.5, width=0.5, pad=1.2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if sn["label"] == "construction":
            ttl = f"step {sn['step']}: construction ({n} adds)"
        else:
            rem, add = sn["rem"], sn["add"]
            ttl = (f"step {sn['step']}: swap {sn['label'].split()[1]}  "
                   f"$-$({rem['ex_nm']:.0f}, {rem['em_nm']:.0f})  $+$({add['ex_nm']:.0f}, {add['em_nm']:.0f})")
        fig.text(ax0 / W, (y0 + gh + 0.025) / H, ttl + f"   $J_{{rec}}$ = {sn['J']:.3f}", fontsize=7,
                 ha="left", va="bottom")
    fig.text(0.075 / W, (H / 2 - 0.05) / H, "excitation (nm)", rotation=90, va="center", ha="center", fontsize=7)
    fig.text(0.02 / W, (H - 0.09) / H, "(a)", fontsize=8, fontweight="bold", va="center")
    lax = axin(ax0 - 0.3, 0.0, aw + 0.3, 0.17)
    lax.axis("off")
    handles = [Patch(fc=C_ADM, label="admissible"), Patch(fc=C_SEL, label="selected"),
               Patch(fc=C_ADD, label="added"), Patch(fc="white", ec=C_REM, lw=1.3, label="just removed")]
    lax.legend(handles=handles, ncol=4, loc="center", frameon=False, handlelength=1.1, handletextpad=0.4,
               columnspacing=0.9, borderaxespad=0)

    # ---------------- data for (b), (c)
    t = np.array([s["step"] for s in steps_ar])
    jr = np.array([s["J_rec"] for s in steps_ar])
    ba = np.array([s["knn_ba"] for s in steps_ar])
    size = np.array([s["size"] for s in steps_ar])
    kinds = np.array([s["kind"] for s in steps_ar])
    full, rm = size == n, kinds == "remove"
    jr_add, ba_add = steps_add[-1]["J_rec"], steps_add[-1]["knn_ba"]

    def curves(ax, ax2, sl, ms=3.4):
        ax.plot(t[sl], jr[sl], "-", color=C_REC, zorder=3)
        f, h = full[sl], ~full[sl]
        ax.plot(t[sl][f], jr[sl][f], "o", color=C_REC, ms=ms, zorder=4)
        ax.plot(t[sl][h], jr[sl][h], "o", mfc="white", mec=C_REC, ms=ms, zorder=4)
        ax2.plot(t[sl][rm[sl]], jr[sl][rm[sl]], "v", color=C_REM, ms=6.2, mec="white", mew=0.5, zorder=9,
                 transform=ax.transData, clip_on=True)      # drawn on the top twin axes so it is never hidden
        ax2.plot(t[sl], ba[sl], "--", color=C_BA, lw=1.0, zorder=3)
        ax2.plot(t[sl][f], ba[sl][f], "s", color=C_BA, ms=ms - 0.4, zorder=4)
        ax2.plot(t[sl][h], ba[sl][h], "s", mfc="white", mec=C_BA, ms=ms - 0.4, zorder=4)
        ax.tick_params(axis="y", colors=C_REC, length=2, width=0.5, pad=1.5)
        ax2.tick_params(axis="y", colors=C_BA, length=2, width=0.5, pad=1.5)
        ax.tick_params(axis="x", length=2, width=0.5, pad=1.5)
        ax.grid(axis="y", alpha=0.25, lw=0.4)

    def shade(ax, lo, hi):
        ax.axvspan(lo, cons_end + 0.5, color="#eaf1f8", lw=0, zorder=0)
        ax.axvspan(cons_end + 0.5, hi, color="#fbeee3", lw=0, zorder=0)

    bx0, bw = 4.12, 2.42
    # ---------------- (b) whole episode
    ax = axin(bx0, 1.93, bw, 1.17)
    ax2 = ax.twinx()
    shade(ax, 0.5, t[-1] + 0.5)
    curves(ax, ax2, slice(None))
    ax.axhline(jr_add, color=C_REC, lw=0.9, ls=":", zorder=2)
    ax2.axhline(ba_add, color=C_BA, lw=0.9, ls=":", zorder=2)
    ax.set_xlim(0.5, t[-1] + 0.5)
    ax.set_xticks(t)
    ax.set_ylim(0.50, 0.90)
    ax2.set_ylim(0.55, 1.07)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylabel("$J_{rec}$ (label-free)", color=C_REC, labelpad=2)
    ax2.set_ylabel("held-out kNN BA", color=C_BA, labelpad=2)
    for xm, lab, col in ((0.5 + (cons_end) / 2, "construction", "#33556f"),
                         (cons_end + 0.5 + (t[-1] - cons_end) / 2, "refinement", "#8a4f1d")):
        ax.text(xm, 0.895, lab, ha="center", va="top", fontsize=7, color=col)
    leg = [Line2D([], [], color=C_REC, marker="o", ms=3.4, label="$J_{rec}$"),
           Line2D([], [], color=C_BA, ls="--", marker="s", ms=3, label="kNN BA"),
           Line2D([], [], color=C_REM, ls="", marker="v", ms=5.5, label="removal"),
           Line2D([], [], color="#777777", ls="", marker="o", mfc="white", ms=3.4, label="$|S|<n$"),
           Line2D([], [], color=C_REC, ls=":", label="add-only, final $J_{rec}$"),
           Line2D([], [], color=C_BA, ls=":", label="add-only, final BA")]
    ax.legend(handles=leg, loc="lower right", ncol=2, frameon=True, framealpha=0.93, edgecolor="none", borderpad=0.25,
              handlelength=1.6, handletextpad=0.4, labelspacing=0.2, columnspacing=0.9, bbox_to_anchor=(1.0, 0.0))
    fig.text(3.45 / W, (H - 0.09) / H, "(b)", fontsize=8, fontweight="bold", va="center")

    # ---------------- (c) zoom on the refinement phase
    ax = axin(bx0, 0.50, bw, 0.93)
    ax2 = ax.twinx()
    sl = slice(cons_end - 1, None)
    lo, hi = t[sl][0] - 0.5, t[sl][-1] + 0.5
    shade(ax, lo, hi)
    curves(ax, ax2, sl, ms=3.8)
    ax.set_xlim(lo, hi)
    ax.set_xticks(t[sl])
    ax.set_xticklabels([f"{ti}\n{('' if i == 0 else kinds[sl][i])}" for i, ti in enumerate(t[sl])])
    ax.tick_params(axis="x", pad=1.2)
    ax.set_xlabel("step $t$ and action", labelpad=0)
    jlo, jhi = jr[sl].min(), jr[sl].max()
    blo, bhi = ba[sl].min(), ba[sl].max()
    ax.set_ylim(jlo - 0.18 * (jhi - jlo), jhi + 0.18 * (jhi - jlo))
    ax2.set_ylim(blo - 0.18 * (bhi - blo), bhi + 0.18 * (bhi - blo))
    for a_ in (ax, ax2):
        a_.yaxis.set_major_locator(MaxNLocator(4))
        a_.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_ylabel("$J_{rec}$", color=C_REC, labelpad=1)
    ax2.set_ylabel("kNN BA", color=C_BA, labelpad=1)
    fig.text(3.45 / W, 1.62 / H, "(c)", fontsize=8, fontweight="bold", va="center")
    fig.text((bx0 + 0.02) / W, 1.62 / H, "refinement phase, zoomed (expanded y-axes)", fontsize=7,
             va="center", ha="left")
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- main
def write_steps_csv(path, steps_ar, steps_add, case):
    cols = ["policy", "task", "n", "step", "phase", "action", "pair", "ex_nm", "em_nm", "em_nm_cache_label", "set_size", "set_after",
            "J_rec", "J_sup", "knn_ba", "p_action", "p_stop"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, steps in (("addremove", steps_ar), ("add_only", steps_add)):
            for s in steps:
                w.writerow([name, case, s["n"], s["step"], s["phase"],
                            s["kind"], "" if s["pair"] is None else s["pair"],
                            "" if s["ex_nm"] is None else f"{s['ex_nm']:.0f}",
                            "" if s["em_nm"] is None else f"{s['em_nm']:.0f}",
                            "" if s["em_cache_nm"] is None else f"{s['em_cache_nm']:.0f}", s["size"],
                            " ".join(map(str, s["S"])), f"{s['J_rec']:.6f}", f"{s['J_sup']:.6f}",
                            f"{s['knn_ba']:.6f}", f"{s['p_action']:.4f}",
                            "" if np.isnan(s["p_stop"]) else f"{s['p_stop']:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--fold", type=int, default=None, help="force a case (default: first case with >=1 swap)")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    from rl_select.data import make_task, n_outer_folds
    from rl_select.features import TaskFeatures
    from rl_select.objective import JSup
    from rl_select.policy import SelectPolicy
    from rl_select.rl_search import deploy_label_free

    OUT.mkdir(parents=True, exist_ok=True)
    pol_ar = SelectPolicy.load(RES / "policies" / POL_AR)
    pol_add = SelectPolicy.load(RES / "policies" / POL_ADD)
    print("params", pol_ar.n_params(), "meta", pol_ar.meta)

    def tfeat(task):
        return TaskFeatures(task, None, sup="none", feat=pol_ar.meta.get("feat", "full"))

    # ---- swap counts over cases (greedy label-free rollouts only: no evaluation needed)
    from rl_select.evaluate import evaluate
    rows = []
    nf = n_outer_folds(TARGET)
    for rep in (0, 1, 2):
        for fold in range(nf):
            task = make_task(TARGET, rep, fold)
            tf = tfeat(task)
            jsup = JSup(task)
            for n in NS:
                st = rollout(pol_ar, tf, n, "addremove")
                S_c = st[n - 1]["S"]                  # set after the n construction adds
                S_f = st[-1]["S"]
                ba = lambda S: evaluate(task, S, ("knn",), seed=EVAL_SEED)["knn"]["balanced_acc"]
                nsw = sum(s["kind"] == "remove" for s in st)
                stopped = st[-1]["kind"] == "stop"
                rec = stored_record(M_AR, rep, fold, n)
                same = sorted(rec["selected"]) == st[-1]["S"]
                rows.append(dict(repeat=rep, fold=fold, n=n, swaps=nsw, stopped_by_policy=int(stopped),
                                 steps=len(st), stored_steps=rec["extra"]["steps"], matches_stored=int(same),
                                 set_changed=int(S_c != S_f), n_pairs_changed=len(set(S_f) - set(S_c)),
                                 jrec_cons=tf.jrecon(S_c), jrec_final=tf.jrecon(S_f),
                                 jsup_cons=jsup(S_c), jsup_final=jsup(S_f), knn_cons=ba(S_c), knn_final=ba(S_f)))
    with open(OUT / "fig_episode_swaps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("swap counts (repeat, fold, n -> swaps):")
    for n in NS:
        sub = [r for r in rows if r["n"] == n]
        print(f"  n={n}: swaps per unit {[r['swaps'] for r in sub]}  mean {np.mean([r['swaps'] for r in sub]):.2f}; "
              f"episodes with >=1 swap {sum(r['swaps'] > 0 for r in sub)}/{len(sub)}; "
              f"all stored finals reproduced {all(r['matches_stored'] for r in sub)}; "
              f"final set differs from post-construction set in {sum(r['set_changed'] for r in sub)}/{len(sub)}; "
              f"mean delta (final - post-construction): J_rec {np.mean([r['jrec_final'] - r['jrec_cons'] for r in sub]):+.4f}, "
              f"J_sup {np.mean([r['jsup_final'] - r['jsup_cons'] for r in sub]):+.4f}, "
              f"kNN BA {np.mean([r['knn_final'] - r['knn_cons'] for r in sub]):+.4f}")

    # ---- pick the case
    if a.fold is not None:
        order = [(a.fold, a.n)]
    else:
        order = [(f, n) for n in (8, 13) for f in range(nf)]
    chosen, tried = None, []
    for fold, n in order:
        sw = next(r["swaps"] for r in rows if r["repeat"] == a.repeat and r["fold"] == fold and r["n"] == n)
        tried.append((fold, n, sw))
        if sw >= 1:
            chosen = (fold, n)
            break
    print("cases tried (fold, n, swaps):", tried)
    fold, n = chosen
    task = make_task(TARGET, a.repeat, fold)
    tf = tfeat(task)
    obj_sup = JSup(task)
    case = f"{task.name}/n{n}"

    em_rec = eem_grid(task)[4]
    steps_ar = annotate(task, tf, rollout(pol_ar, tf, n, "addremove"), obj_sup, em_rec)
    steps_add = annotate(task, tf, rollout(pol_add, tf, n, "add"), obj_sup, em_rec)
    for s in steps_ar + steps_add:
        s["n"] = n

    # ---- reproduce stored records (via the library entry point as well)
    res_ar = deploy_label_free(pol_ar, task, n, mode="addremove", init="empty", n_samples=1, seed=0)
    res_add = deploy_label_free(pol_add, task, n, mode="add", init="empty", n_samples=1, seed=0)
    for nm, steps, res, meth in (("add/remove", steps_ar, res_ar, M_AR), ("add-only", steps_add, res_add, M_ADD)):
        rec = stored_record(meth, a.repeat, fold, n)
        print(f"[{nm}] final set (manual) {steps[-1]['S']}")
        print(f"[{nm}] final set (deploy_label_free) {sorted(res.selected)} | stored {sorted(rec['selected'])}")
        print(f"[{nm}] steps manual {len(steps)} deploy {res.extra['steps']} stored {rec['extra']['steps']}")
        print(f"[{nm}] knn BA recomputed {steps[-1]['knn_ba']:.10f} stored {rec['eval']['knn']['balanced_acc']:.10f}")
        print(f"[{nm}] J_rec {steps[-1]['J_rec']:.10f} stored inner_J {rec['inner_J']:.10f}")
        assert steps[-1]["S"] == sorted(res.selected) == sorted(rec["selected"]), "final set differs from stored"
        assert abs(steps[-1]["knn_ba"] - rec["eval"]["knn"]["balanced_acc"]) < 1e-9, "kNN BA differs from stored"

    write_steps_csv(OUT / "fig_episode_steps.csv", steps_ar, steps_add, case)
    for nm, steps in (("add/remove", steps_ar), ("add-only", steps_add)):
        print(f"--- {nm}  {case}")
        for s in steps:
            pr = "" if s["pair"] is None else f"pair {s['pair']:3d} (ex {s['ex_nm']:.0f}, em {s['em_nm']:.0f}rec)"
            print(f" t={s['step']:2d} {s['phase'][:5]} {s['kind']:6s} {pr:28s} |S|={s['size']:2d} "
                  f"Jrec={s['J_rec']:.5f} Jsup={s['J_sup']:.4f} kNN={s['knn_ba']:.4f} p={s['p_action']:.2f} "
                  f"pstop={s['p_stop']:.2f}")
    fig_episode(task, steps_ar, steps_add, n, OUT / "fig_episode")
    print("wrote", OUT / "fig_episode.pdf")


if __name__ == "__main__":
    main()
