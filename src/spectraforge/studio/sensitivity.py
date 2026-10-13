"""Global sensitivity analysis: Sobol' variance-based indices and Morris elementary effects.

Sobol' indices (Sobol' 1993, Math. Model. Comput. Exp. 1:407; Saltelli 2010, Comput. Phys. Commun. 181:259,
doi:10.1016/j.cpc.2009.09.018).  For a model Y = f(X), X in [0,1]^k with independent inputs and matrices A, B and
AB_i (A with column i from B), N base samples:

    Var(Y)  = sample variance of the 2N values [f(A), f(B)]
    V_i     = (1/N) sum_j f(B)_j ( f(AB_i)_j - f(A)_j )              first-order effect   (Saltelli 2010, eq. 6)
    VT_i    = (1/2N) sum_j ( f(A)_j - f(AB_i)_j )^2                  total effect         (Jansen 1999, eq. 4.3,
              Comput. Phys. Commun. 117:35, doi:10.1016/S0010-4655(98)00154-4)
    S_i = V_i / Var(Y),  ST_i = VT_i / Var(Y)

95 % confidence intervals are percentile bootstrap intervals over the N base rows (Archer, Saltelli & Sobol' 1997,
J. Stat. Comput. Simul. 58:99), resampling A, B and every AB_i jointly.

Morris screening (Morris 1991, Technometrics 33:161; Campolongo, Cariboni & Saltelli 2007, Environ. Model. Softw.
22:1509, doi:10.1016/j.envsoft.2006.10.004): for every trajectory step that changes only input i by +-Delta,
EE_i = (f(x') - f(x)) / (+-Delta); mu = mean EE, mu* = mean |EE| (robust to sign cancellation), sigma = sample std of
EE (non-linearity / interactions).  Inputs are on the unit cube, so EE has the units of the metric per full range.

Monte Carlo noise.  When a study has R > 1 replicates per design point, the metric analysed is the replicate mean.
The within-point replicate variance s_w^2 (mean of the unbiased per-point variances) gives the *noise floor*:
``noise_floor = (s_w^2 / R) / Var(replicate means)``, the fraction of the output variance that is pure engine noise.
Indices of inputs whose effect is below this floor are not distinguishable from zero.  (The replicate r of every
design point shares one seed, i.e. common random numbers; the floor is therefore an upper bound on the noise that
survives in index differences.)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------ test functions
def ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """Ishigami & Homma (1990) function on [0,1]^3 mapped to [-pi, pi]^3.  Analytic indices for a=7, b=0.1:
    S1 = (0.3139, 0.4424, 0), ST = (0.5576, 0.4424, 0.2437)."""
    z = -np.pi + 2 * np.pi * np.asarray(x, dtype=float)
    return np.sin(z[:, 0]) + a * np.sin(z[:, 1]) ** 2 + b * z[:, 2] ** 4 * np.sin(z[:, 0])


def ishigami_analytic(a: float = 7.0, b: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    pi4 = np.pi ** 4
    V = a * a / 8 + b * pi4 / 5 + b * b * np.pi ** 8 / 18 + 0.5
    V1 = 0.5 * (1 + b * pi4 / 5) ** 2
    V2 = a * a / 8
    V13 = b * b * np.pi ** 8 * (1 / 18 - 1 / 50)
    return np.array([V1 / V, V2 / V, 0.0]), np.array([(V1 + V13) / V, V2 / V, V13 / V])


def sobol_g(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Sobol' G function prod_i (|4 x_i - 2| + a_i)/(1 + a_i) (Saltelli & Sobol' 1995); inputs on [0,1]^k."""
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    return np.prod((np.abs(4 * x - 2) + a) / (1 + a), axis=1)


def sobol_g_analytic(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    Vi = 1.0 / (3.0 * (1 + a) ** 2)
    V = np.prod(1 + Vi) - 1
    ST = np.array([Vi[i] * np.prod(np.delete(1 + Vi, i)) for i in range(a.size)]) / V
    return Vi / V, ST


# ------------------------------------------------------------------------------------------ Sobol'
def saltelli_matrices(n: int, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """A, B (n, k) and [AB_0..AB_{k-1}] from a scrambled 2k-dimensional Sobol' sequence (same as the study design)."""
    from .study import _sobol_points
    ab = _sobol_points(n, 2 * k, seed)
    A, B = ab[:, :k], ab[:, k:]
    out = []
    for i in range(k):
        M = A.copy()
        M[:, i] = B[:, i]
        out.append(M)
    return A, B, out


def _sobol_point(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fAB (k, n) -> (S1, ST) for one (re)sample."""
    var = np.var(np.concatenate([fA, fB]), ddof=1)
    if not var > 0:
        return np.full(fAB.shape[0], np.nan), np.full(fAB.shape[0], np.nan)
    s1 = np.mean(fB * (fAB - fA), axis=1) / var
    st = 0.5 * np.mean((fA - fAB) ** 2, axis=1) / var
    return s1, st


def sobol_indices(Y: np.ndarray, n: int, k: int, *, n_boot: int = 300, seed: int = 0, conf: float = 0.95) -> dict:
    """Sobol' first-order ``S1`` and total ``ST`` indices with bootstrap CIs from outputs ``Y`` ordered like the
    Saltelli design: A (n rows), B (n rows), AB_0 (n rows), ..., AB_{k-1} (n rows).

    Returns arrays ``S1, S1_conf, ST, ST_conf`` (conf = half-width of the percentile interval is NOT used: the
    bounds are returned as ``S1_ci`` / ``ST_ci`` of shape (k, 2)), plus ``variance`` and ``n``."""
    Y = np.asarray(Y, dtype=float)
    if Y.size != n * (k + 2):
        raise ValueError(f"Y has {Y.size} values, expected n*(k+2) = {n * (k + 2)}")
    fA, fB = Y[:n], Y[n:2 * n]
    fAB = Y[2 * n:].reshape(k, n)
    s1, st = _sobol_point(fA, fB, fAB)
    rng = np.random.default_rng(seed)
    b1, bt = np.empty((n_boot, k)), np.empty((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        b1[b], bt[b] = _sobol_point(fA[idx], fB[idx], fAB[:, idx])
    lo, hi = 50 * (1 - conf), 100 - 50 * (1 - conf)
    with np.errstate(all="ignore"):
        return {"S1": s1, "ST": st, "S1_ci": np.stack(np.nanpercentile(b1, [lo, hi], axis=0), axis=1),
                "ST_ci": np.stack(np.nanpercentile(bt, [lo, hi], axis=0), axis=1),
                "S1_se": np.nanstd(b1, axis=0, ddof=1), "ST_se": np.nanstd(bt, axis=0, ddof=1),
                "variance": float(np.var(np.concatenate([fA, fB]), ddof=1)), "n": n, "k": k}


def sobol_analyse_function(f: Callable[[np.ndarray], np.ndarray], k: int, n: int = 2 ** 13, *, seed: int = 0,
                           n_boot: int = 300) -> dict:
    """Run the whole Saltelli scheme on a cheap analytic function ``f(X: (m, k)) -> (m,)`` (for validation)."""
    A, B, ABs = saltelli_matrices(n, k, seed)
    Y = np.concatenate([f(A), f(B)] + [f(M) for M in ABs])
    return sobol_indices(Y, n, k, n_boot=n_boot, seed=seed)


# ------------------------------------------------------------------------------------------ Morris
def morris_effects(Y: np.ndarray, traj: np.ndarray, step: np.ndarray, step_param: np.ndarray,
                   direction: np.ndarray, k: int, delta: float) -> dict[str, np.ndarray]:
    """Per-parameter elementary effects from trajectory-ordered outputs.  Returns ``ee`` as a list of arrays."""
    Y = np.asarray(Y, dtype=float)
    ee: list[list[float]] = [[] for _ in range(k)]
    for t in np.unique(traj):
        idx = np.where(traj == t)[0]
        idx = idx[np.argsort(step[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            i = int(step_param[b])
            if i >= 0 and np.isfinite(Y[a]) and np.isfinite(Y[b]):
                ee[i].append((Y[b] - Y[a]) / (direction[b] * delta))
    return ee  # type: ignore[return-value]


def morris_indices(Y, traj, step, step_param, direction, k: int, delta: float, *, n_boot: int = 300,
                   seed: int = 0) -> dict:
    """Morris mu, mu*, sigma (and mu* bootstrap 95 % CI over elementary effects) per input."""
    ee = morris_effects(Y, np.asarray(traj), np.asarray(step), np.asarray(step_param), np.asarray(direction), k, delta)
    rng = np.random.default_rng(seed)
    mu, mus, sd, ci, n = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros((k, 2)), np.zeros(k, dtype=int)
    for i, e in enumerate(ee):
        e = np.asarray(e)
        n[i] = e.size
        if e.size == 0:
            mu[i] = mus[i] = sd[i] = np.nan
            ci[i] = np.nan
            continue
        mu[i], mus[i] = e.mean(), np.abs(e).mean()
        sd[i] = e.std(ddof=1) if e.size > 1 else 0.0
        bs = np.abs(e[rng.integers(0, e.size, (n_boot, e.size))]).mean(axis=1)
        ci[i] = np.percentile(bs, [2.5, 97.5])
    return {"mu": mu, "mu_star": mus, "sigma": sd, "mu_star_ci": ci, "n_effects": n}


# ------------------------------------------------------------------------------------------ studies
def noise_floor(df: pd.DataFrame, col: str) -> dict:
    """Replicate-noise diagnostics of metric column ``col`` of a runs table (see module docstring)."""
    g = df.groupby("point")[col]
    R = int(g.size().median())
    if R < 2:
        return {"replicates": R, "within_var": float("nan"), "noise_floor": float("nan")}
    wv = float(g.var(ddof=1).mean())
    tv = float(g.mean().var(ddof=1))
    return {"replicates": R, "within_var": wv, "noise_floor": (wv / R) / tv if tv > 0 else float("nan")}


def _point_means(df: pd.DataFrame, col: str) -> pd.DataFrame:
    keep = ["point", "block", "row", "traj", "step", "step_param", "direction"]
    m = df.groupby("point", sort=True)[col].mean().rename("y")
    first = df.drop_duplicates("point").set_index("point")[keep[1:]]
    return first.join(m).reset_index()


def analyse_runs(df: pd.DataFrame, manifest: dict, *, n_boot: int = 300, seed: int = 0) -> dict:
    """Sensitivity tables for every metric of a study (Saltelli -> Sobol', Morris -> Morris; other designs raise)."""
    spec = manifest["spec"]
    kind = spec["design"]["kind"]
    if kind not in ("saltelli", "morris"):
        raise ValueError(f"sensitivity analysis needs a 'saltelli' or 'morris' design, this study used {kind!r}")
    labels = [p["label"] for p in spec["parameters"]]
    k = len(labels)
    out: dict[str, Any] = {"design": kind, "parameters": labels, "metrics": {}}
    ok = df[df["status"] == "ok"] if "status" in df else df
    for m in spec["metrics"]:
        col = f"m:{m['name']}"
        pm = _point_means(ok, col)
        entry: dict[str, Any] = {"noise": noise_floor(ok, col)}
        if kind == "saltelli":
            n = int(manifest["design"]["n_base"])
            pm = pm.sort_values(["block", "row"], key=lambda s: s.map(_block_order) if s.name == "block" else s)
            blocks = ["A", "B"] + [f"AB:{i}" for i in range(k)]
            Y = np.full((len(blocks), n), np.nan)
            for bi, b in enumerate(blocks):
                sub = pm[pm["block"] == b]
                Y[bi, sub["row"].to_numpy()] = sub["y"].to_numpy()
            good = np.isfinite(Y).all(axis=0)
            if good.sum() < max(8, n // 4):
                raise ValueError(f"metric {m['name']!r}: too few complete Saltelli rows ({int(good.sum())}/{n})")
            n_eff = int(good.sum())
            res = sobol_indices(Y[:, good].reshape(-1), n_eff, k, n_boot=n_boot, seed=seed)
            entry.update(method="sobol", n_rows=n_eff, variance=res["variance"],
                         indices={lab: {"S1": float(res["S1"][i]), "S1_ci": [float(v) for v in res["S1_ci"][i]],
                                        "ST": float(res["ST"][i]), "ST_ci": [float(v) for v in res["ST_ci"][i]]}
                                  for i, lab in enumerate(labels)},
                         sum_S1=float(np.nansum(res["S1"])))
        else:
            p_lv = int(spec["design"]["levels"])
            delta = p_lv / (2.0 * (p_lv - 1))
            pm = pm.sort_values("point")
            res = morris_indices(pm["y"].to_numpy(), pm["traj"].to_numpy(), pm["step"].to_numpy(),
                                 pm["step_param"].to_numpy(), pm["direction"].to_numpy(), k, delta,
                                 n_boot=n_boot, seed=seed)
            entry.update(method="morris", delta=delta,
                         indices={lab: {"mu": float(res["mu"][i]), "mu_star": float(res["mu_star"][i]),
                                        "sigma": float(res["sigma"][i]),
                                        "mu_star_ci": [float(v) for v in res["mu_star_ci"][i]],
                                        "n_effects": int(res["n_effects"][i])} for i, lab in enumerate(labels)})
        out["metrics"][m["name"]] = entry
    return out


def _block_order(b: str) -> int:
    if b == "A":
        return -2
    if b == "B":
        return -1
    return int(b.split(":")[1]) if b.startswith("AB:") else 0


def analyse_study(out_dir: str | Path, *, n_boot: int = 300, seed: int = 0, save: bool = True) -> dict:
    """Load ``runs.parquet`` + ``manifest.json`` of a study directory, compute the indices of every metric and
    (``save=True``) write ``sensitivity.json`` and ``sensitivity.csv`` next to them."""
    from .study import load_manifest, load_runs
    res = analyse_runs(load_runs(out_dir), load_manifest(out_dir), n_boot=n_boot, seed=seed)
    if save:
        p = Path(out_dir)
        (p / "sensitivity.json").write_text(json.dumps(res, indent=2, default=float))
        tab_to_frame(res).to_csv(p / "sensitivity.csv", index=False)
    return res


def tab_to_frame(res: dict) -> pd.DataFrame:
    rows = []
    for mname, e in res["metrics"].items():
        for lab, v in e["indices"].items():
            row = {"metric": mname, "parameter": lab, "method": e["method"], **{k: x for k, x in v.items()
                                                                              if not isinstance(x, list)}}
            for k, x in v.items():
                if isinstance(x, list):
                    row[f"{k}_lo"], row[f"{k}_hi"] = x
            row["noise_floor"] = e["noise"]["noise_floor"]
            rows.append(row)
    return pd.DataFrame(rows)


def format_table(res: dict) -> str:
    """Plain-text rendering of :func:`analyse_study` results (what the CLI prints)."""
    lines = []
    for mname, e in res["metrics"].items():
        nf = e["noise"]["noise_floor"]
        lines.append(f"metric {mname!r}  ({e['method']}, noise floor "
                     f"{'n/a' if not np.isfinite(nf) else f'{nf:.3f}'})")
        if e["method"] == "sobol":
            lines.append(f"  {'parameter':<28}{'S1':>8}{'[95% CI]':>20}{'ST':>8}{'[95% CI]':>20}")
            for lab, v in e["indices"].items():
                lines.append(f"  {lab:<28}{v['S1']:>8.3f}{'[%.3f, %.3f]' % tuple(v['S1_ci']):>20}"
                             f"{v['ST']:>8.3f}{'[%.3f, %.3f]' % tuple(v['ST_ci']):>20}")
            lines.append(f"  sum S1 = {e['sum_S1']:.3f}")
        else:
            lines.append(f"  {'parameter':<28}{'mu*':>12}{'[95% CI]':>26}{'mu':>12}{'sigma':>12}")
            for lab, v in e["indices"].items():
                lines.append(f"  {lab:<28}{v['mu_star']:>12.4g}{'[%.3g, %.3g]' % tuple(v['mu_star_ci']):>26}"
                             f"{v['mu']:>12.4g}{v['sigma']:>12.4g}")
    return "\n".join(lines)
