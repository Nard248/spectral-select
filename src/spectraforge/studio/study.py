"""Computational studies: design of experiments, metric extraction and a resumable, deterministic runner.

Pipeline::

    StudySpec --make_design()--> Design (unit-cube points U, parameter values, block/trajectory labels)
              --run_study()----> out_dir/{runs.parquet, runs.csv, manifest.json, runs.jsonl [, captures/*.npz]}

Designs (all seeded; ``scipy.stats.qmc`` provides the low-discrepancy generators)
    grid       full factorial; continuous/integer parameters get ``levels`` equispaced levels (or their explicit
               ``values``), choices use all values.
    random     i.i.d. uniform on the unit cube.
    lhs        Latin hypercube (McKay, Beckman & Conover 1979; scipy ``LatinHypercube``, ``optimization=None``).
    sobol      scrambled Sobol' sequence (Owen 1998 scrambling, scipy ``Sobol``; use n = 2^m for balance).
    saltelli   A, B, AB_1..AB_k blocks for Sobol' indices (Saltelli 2010, Comput. Phys. Commun. 181:259,
               doi:10.1016/j.cpc.2009.09.018).  A and B are the two halves of a 2k-dimensional scrambled Sobol'
               sequence; AB_i is A with column i taken from B.  Rows = n (k + 2).
    morris     r random trajectories with k+1 points on a p-level grid, step Delta = p / (2 (p - 1)) (Morris 1991,
               Technometrics 33:161, doi:10.1080/00401706.1991.10484804; Saltelli et al. 2008, Global Sensitivity
               Analysis: The Primer, sec. 3.4).  Rows = r (k + 1).

Runs
    One run = one (design point, replicate).  A replicate re-runs the capture with the Monte Carlo / sensor seed
    ``base_seed + replicate`` (``acquisition.seed`` of the base experiment is ``base_seed``).  A run is identified
    by ``run_id = sha256(experiment hash | mode | excitations | metric definitions)[:20]``; completed runs are
    journalled to ``runs.jsonl`` as they finish, so an interrupted study resumes (``resume=True``) skipping every
    run id already present.  The final tables are always rebuilt from the journal in design order, so the output
    is independent of the number of workers and of interruptions.
"""
from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import ValidationError

from . import analysis
from .provenance import canonical_json, experiment_hash, provenance, sha256_hex
from .schema import Experiment
from .study_schema import MetricSpec, ParameterSpec, StudySpec
from .sweeps import SweepError, parse_pointer, set_pointer


class StudyError(ValueError):
    """Invalid study specification (reported to API clients as HTTP 422)."""


# ------------------------------------------------------------------------------------------ parameters
def param_value(p: ParameterSpec, u: float) -> Any:
    """Map a unit-interval coordinate ``u`` in [0, 1] to the parameter value."""
    u = min(max(float(u), 0.0), 1.0)
    if p.kind == "choice":
        n = len(p.values)
        return p.values[min(int(math.floor(u * n)), n - 1)]
    lo, hi = float(p.low), float(p.high)
    if p.kind == "integer":
        n = int(round(hi)) - int(round(lo)) + 1
        return int(round(lo)) + min(int(math.floor(u * n)), n - 1)
    if p.scale == "log":
        return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
    return float(lo + u * (hi - lo))


def param_unit(p: ParameterSpec, v: Any) -> float:
    """Inverse of :func:`param_value` (centre of the cell for choice / integer parameters)."""
    if p.kind == "choice":
        return (p.values.index(v) + 0.5) / len(p.values)
    lo, hi = float(p.low), float(p.high)
    if p.kind == "integer":
        n = int(round(hi)) - int(round(lo)) + 1
        return (int(v) - int(round(lo)) + 0.5) / n
    if p.scale == "log":
        return (math.log(float(v)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (float(v) - lo) / (hi - lo)


# ------------------------------------------------------------------------------------------ designs
@dataclass
class Design:
    kind: str
    U: np.ndarray                       # (n, k) unit-cube coordinates
    values: list[list[Any]]             # (n, k) parameter values
    table: pd.DataFrame                 # n rows: block, row, traj, step, step_param, direction
    n_base: int = 0                     # saltelli / morris base size

    def __len__(self) -> int:
        return len(self.values)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _sobol_points(n: int, dim: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc
    s = qmc.Sobol(d=dim, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # "n is not a power of 2" balance warning
        return s.random(n)


def _meta(n: int, **cols) -> pd.DataFrame:
    base = {"block": [""] * n, "row": list(range(n)), "traj": [-1] * n, "step": [-1] * n,
            "step_param": [-1] * n, "direction": [0] * n}
    base.update(cols)
    return pd.DataFrame(base)


def _grid_axes(spec: StudySpec) -> list[list[Any]]:
    lv = spec.design.levels
    axes = []
    for p in spec.parameters:
        if p.kind == "choice":
            axes.append(list(p.values))
        elif p.values:
            axes.append(list(p.values))
        elif p.kind == "integer":
            vals = sorted({param_value(p, u) for u in np.linspace(0, 1, lv)}) if lv else []
            axes.append(vals)
        else:
            axes.append([param_value(p, u) for u in np.linspace(0.0, 1.0, lv)])
    return axes


def make_design(spec: StudySpec) -> Design:
    """Generate the design of ``spec`` (deterministic in ``spec.design.seed``)."""
    d, ps = spec.design, spec.parameters
    k = len(ps)
    rng = np.random.default_rng(d.seed)
    kind = d.kind
    if kind == "grid":
        axes = _grid_axes(spec)
        combos = list(itertools.product(*axes))
        vals = [list(c) for c in combos]
        U = np.array([[param_unit(p, v) for p, v in zip(ps, row)] for row in vals], dtype=float)
        return Design(kind, U, vals, _meta(len(vals)))
    if kind in ("random", "lhs", "sobol"):
        if kind == "random":
            U = rng.random((d.n, k))
        elif kind == "lhs":
            from scipy.stats import qmc
            U = qmc.LatinHypercube(d=k, seed=d.seed).random(d.n)
        else:
            U = _sobol_points(d.n, k, d.seed)
        vals = [[param_value(p, u) for p, u in zip(ps, row)] for row in U]
        return Design(kind, U, vals, _meta(len(vals)))
    if kind == "saltelli":
        n = d.n
        AB = _sobol_points(n, 2 * k, d.seed)
        A, B = AB[:, :k], AB[:, k:]
        blocks, mats = ["A", "B"], [A, B]
        for i in range(k):
            M = A.copy()
            M[:, i] = B[:, i]
            blocks.append(f"AB:{i}")
            mats.append(M)
        U = np.vstack(mats)
        meta = pd.DataFrame({"block": np.repeat(blocks, n), "row": np.tile(np.arange(n), len(blocks)),
                             "traj": -1, "step": -1, "step_param": -1, "direction": 0})
        vals = [[param_value(p, u) for p, u in zip(ps, row)] for row in U]
        return Design(kind, U, vals, meta, n_base=n)
    if kind == "morris":
        p_lv, r = d.levels, d.n
        if p_lv % 2:
            raise StudyError("morris needs an even number of levels (Morris 1991: Delta = p/(2(p-1)))")
        delta = p_lv / (2.0 * (p_lv - 1))
        grid = np.arange(p_lv) / (p_lv - 1)
        low_levels = grid[grid <= 1.0 - delta + 1e-9]
        rows, meta = [], []
        for t in range(r):
            x = rng.choice(low_levels, size=k)
            rows.append(x.copy())
            meta.append((f"T{t}", 0, t, 0, -1, 0))
            for s, i in enumerate(rng.permutation(k), start=1):
                up, down = x[i] + delta <= 1 + 1e-9, x[i] - delta >= -1e-9
                sign = 1 if up and (not down or rng.random() < 0.5) else -1
                x = x.copy()
                x[i] = x[i] + sign * delta
                rows.append(x)
                meta.append((f"T{t}", s, t, s, int(i), sign))
        U = np.clip(np.array(rows), 0.0, 1.0)
        mt = pd.DataFrame(meta, columns=["block", "row", "traj", "step", "step_param", "direction"])
        vals = [[param_value(p, u) for p, u in zip(ps, row)] for row in U]
        return Design(kind, U, vals, mt, n_base=r)
    raise StudyError(f"unknown design kind {kind!r}")


# ------------------------------------------------------------------------------------------ experiments
def _tree_values(p: ParameterSpec, v: Any) -> dict[str, Any]:
    return {ptr: v for ptr in [p.pointer, *p.also_set]}


def build_experiment(base_doc: dict, params: list[ParameterSpec], values: list[Any], seed: int | None = None) -> Experiment:
    """Deep-copy ``base_doc`` (the base experiment as a JSON dict), set every parameter (and linked pointers) and
    optionally the acquisition seed, then validate."""
    import copy
    doc = copy.deepcopy(base_doc)
    try:
        for p, v in zip(params, values):
            for ptr, val in _tree_values(p, v).items():
                set_pointer(doc, ptr, val)
        if seed is not None:
            doc["acquisition"]["seed"] = int(seed)
        return Experiment.model_validate(doc)
    except SweepError as exc:
        raise StudyError(str(exc)) from exc
    except ValidationError as exc:
        e = exc.errors()[0]
        raise StudyError(f"{'/'.join(str(x) for x in e['loc'])}: {e['msg']} (parameter values {values})") from exc


def validate_spec(spec: StudySpec, base: Experiment | None = None) -> Experiment:
    """Check pointers / ranges apply and validate to a legal experiment; returns the base experiment."""
    base = base or spec.resolve_base()
    doc = base.model_dump(mode="json")
    for i, p in enumerate(spec.parameters):
        cands = list(p.values) if p.kind == "choice" else [param_value(p, 0.0), param_value(p, 1.0)]
        if p.kind != "choice" and p.values:
            cands += list(p.values)
        for v in cands:
            vals = [param_value(q, 0.5) if j != i else v for j, q in enumerate(spec.parameters)]
            build_experiment(doc, spec.parameters, vals)
    return base


# ------------------------------------------------------------------------------------------ metrics
def _roi_px(roi, units: str, width: int, height: int) -> dict:
    if roi is None:
        return {"shape": "rect", "x0": 0, "y0": 0, "x1": width - 1, "y1": height - 1}
    d = roi.model_dump() if hasattr(roi, "model_dump") else dict(roi)
    if units == "pixels":
        return d
    fx, fy = max(width - 1, 1), max(height - 1, 1)
    s = d["shape"]
    if s == "point":
        d.update(x=d["x"] * fx, y=d["y"] * fy)
    elif s == "disk":
        d.update(x=d["x"] * fx, y=d["y"] * fy, radius_px=d["radius_px"] * fx)
    elif s == "rect":
        d.update(x0=d["x0"] * fx, x1=d["x1"] * fx, y0=d["y0"] * fy, y1=d["y1"] * fy)
    elif s == "ellipse":
        d.update(cx=d["cx"] * fx, cy=d["cy"] * fy, rx=d["rx"] * fx, ry=d["ry"] * fy)
    else:
        d["points"] = [(x * fx, y * fy) for x, y in d["points"]]
    return d


def _reduce(v: np.ndarray, stat: str) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float({"mean": np.mean, "median": np.median, "std": np.std, "min": np.min, "max": np.max}[stat](v))


def compute_metric(res, m: MetricSpec) -> float:
    """Evaluate one :class:`MetricSpec` on a capture (see its docstring for the definitions)."""
    norm = m.norm
    roi = _roi_px(m.roi, m.roi_units, res.width, res.height)
    ex_default = float(res.excitations_nm[0])
    ex = float(m.ex) if m.ex is not None else ex_default
    k = m.kind

    def pix(roi_px):
        return analysis._roi_pixels(res, roi_px)[2]

    if k == "roi_band":
        v, _ = analysis.metric_values(res, {"kind": "band", "ex": ex, "band_nm": m.band_nm}, norm, pix(roi))
        return _reduce(v, m.stat)
    if k == "band_ratio":
        sel = pix(roi)
        va, _ = analysis.metric_values(res, {"kind": "band", "ex": m.a.ex if m.a.ex is not None else ex_default,
                                             "band_nm": m.a.band_nm}, norm, sel)
        vb, _ = analysis.metric_values(res, {"kind": "band", "ex": m.b.ex if m.b.ex is not None else ex_default,
                                             "band_nm": m.b.band_nm}, norm, sel)
        den = _reduce(vb, m.stat)
        return _reduce(va, m.stat) / den if den not in (0.0,) and np.isfinite(den) else float("nan")
    if k == "band_sum":
        v, _ = analysis.metric_values(res, {"kind": "band_sum", "ex": ex, "from_nm": m.from_nm, "to_nm": m.to_nm},
                                      norm, pix(roi))
        return _reduce(v, m.stat)
    if k == "peak_wavelength":
        e = analysis.eem(res, roi, norm)
        fex = analysis.find_ex(res, ex)
        return float(analysis.peak_wavelength(np.asarray(e.value[fex])[None, :], e.emission_nm[fex])[0])
    if k == "eem_integral":
        e = analysis.eem(res, roi, norm)
        tot = 0.0
        for fex in e.excitations_nm:
            y = np.nan_to_num(np.asarray(e.value[fex], dtype=float))
            tot += float(np.trapezoid(y, np.asarray(e.emission_nm[fex], dtype=float))) if y.size > 1 else float(y.sum())
        return tot
    if k == "gt_stat":
        g = analysis._gt_map(res, m.gt)
        rows, cols = pix(roi) if m.roi is not None else np.nonzero(np.isfinite(g))
        return _reduce(g[rows, cols], m.stat)
    inner = m.inner or {"kind": "band_sum", "ex": ex}
    va, _ = analysis.metric_values(res, inner, norm, pix(roi))
    roi_b = _roi_px(m.roi_b, m.roi_units, res.width, res.height)
    vb, _ = analysis.metric_values(res, inner, norm, pix(roi_b))
    if k == "contrast":
        a, b = _reduce(va, m.stat), _reduce(vb, m.stat)
        if m.contrast == "weber":
            return (a - b) / b if b != 0 and np.isfinite(b) else float("nan")
        return (a - b) / (a + b) if (a + b) != 0 and np.isfinite(a + b) else float("nan")
    # cnr
    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
    if va.size == 0 or vb.size == 0:
        return float("nan")
    sd = math.sqrt(0.5 * (va.var() + vb.var()))
    return float(abs(va.mean() - vb.mean()) / sd) if sd > 0 else float("nan")


def compute_metrics(res, metrics: list[MetricSpec]) -> dict[str, float]:
    return {m.name: compute_metric(res, m) for m in metrics}


# ------------------------------------------------------------------------------------------ one run
CaptureFn = Callable[..., Any]


def _default_capture(exp: Experiment, mode: str, excitations):
    from . import api
    return api.capture(exp, mode=mode if mode in ("preview", "fast") else "full",
                       excitations=excitations)


def run_one(exp_json: str, mode: str, excitations, metrics_json: list[dict], keep_path: str | None = None,
            capture_fn: CaptureFn | None = None) -> dict:
    """Capture + metrics of one experiment (module-level so a process pool can run it)."""
    exp = Experiment.model_validate_json(exp_json)
    metrics = [MetricSpec.model_validate(m) for m in metrics_json]
    out: dict[str, Any] = {"status": "ok", "error": "", "metrics": {m.name: float("nan") for m in metrics}}
    t0 = time.perf_counter()
    try:
        res = (capture_fn or _default_capture)(exp, mode, excitations)
        t1 = time.perf_counter()
        out["metrics"] = compute_metrics(res, metrics)
        out["t_capture_s"], out["t_metrics_s"] = t1 - t0, time.perf_counter() - t1
        out["warnings"] = len(getattr(res, "warnings", []) or [])
        if keep_path:
            from . import api
            api.save(res, keep_path)
    except Exception as exc:  # noqa: BLE001 - a failing design point must not kill the study
        out.update(status="error", error=f"{type(exc).__name__}: {exc}", t_capture_s=time.perf_counter() - t0,
                   t_metrics_s=0.0, warnings=0)
    return out


def _pool_init(threads: int) -> None:
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, threads))


# ------------------------------------------------------------------------------------------ runner
@dataclass
class StudyResult:
    out_dir: Path
    state: str
    runs: pd.DataFrame
    manifest: dict
    design: Design | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"StudyResult(state={self.state!r}, runs={len(self.runs)}, out_dir={str(self.out_dir)!r})"


def _spec_hash(spec: StudySpec) -> str:
    return sha256_hex(canonical_json(spec.model_dump(mode="json")))


def _metrics_hash(spec: StudySpec) -> str:
    return sha256_hex(canonical_json([m.model_dump(mode="json") for m in spec.metrics]))[:12]


def _read_journal(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # torn last line of an interrupted write
                continue
            if r.get("status") == "ok":
                done[r["run_id"]] = r
    return done


def study_columns(spec: StudySpec) -> dict[str, list[str]]:
    return {"parameters": [f"p:{p.label}" for p in spec.parameters],
            "units": [f"u:{p.label}" for p in spec.parameters],
            "metrics": [f"m:{m.name}" for m in spec.metrics]}


def plan_runs(spec: StudySpec, base: Experiment | None = None) -> tuple[Experiment, Design, list[dict]]:
    """Validate ``spec`` and expand it to the list of run descriptors (without running anything)."""
    base = validate_spec(spec, base)
    design = make_design(spec)
    total = len(design) * spec.replicates
    if total > spec.max_runs:
        raise StudyError(f"the study has {total} runs (> max_runs={spec.max_runs})")
    base_doc = base.model_dump(mode="json")
    seed0 = int(base.acquisition.seed)
    mh = _metrics_hash(spec)
    exc = None if spec.excitations is None else [float(e) for e in spec.excitations]
    plan = []
    for i in range(len(design)):
        for r in range(spec.replicates):
            exp = build_experiment(base_doc, spec.parameters, design.values[i], seed0 + r)
            eh = experiment_hash(exp)
            rid = sha256_hex(f"{eh}|{spec.mode}|{exc}|{mh}")[:20]
            plan.append({"point": i, "replicate": r, "seed": seed0 + r, "exp": exp, "exp_hash": eh, "run_id": rid})
    return base, design, plan


def run_study(spec: StudySpec, out_dir: str | Path, *, workers: int = 1, resume: bool = True,
              progress: Callable[[float, str], None] | None = None, cancel: Callable[[], bool] | None = None,
              keep_captures: bool = False, capture_fn: CaptureFn | None = None) -> StudyResult:
    """Run (or resume) a study and write ``runs.parquet``, ``runs.csv``, ``manifest.json`` into ``out_dir``.

    ``workers > 1`` runs captures in a spawn process pool (``capture_fn`` must then be None).  ``cancel()`` is polled
    between runs; a cancelled study keeps its journal, writes partial tables and can be resumed.  The result does
    not depend on ``workers``, on interruptions, or on wall-clock time (only the timing columns do)."""
    t_start = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "runs.jsonl"
    if not resume and journal.exists():
        journal.unlink()
    base, design, plan = plan_runs(spec)
    cols = study_columns(spec)
    done = _read_journal(journal) if resume else {}
    if workers > 1 and capture_fn is not None:
        raise StudyError("capture_fn is only supported with workers=1")
    metrics_json = [m.model_dump(mode="json") for m in spec.metrics]
    exc = None if spec.excitations is None else [float(e) for e in spec.excitations]
    cap_dir = out / "captures"
    if keep_captures:
        cap_dir.mkdir(exist_ok=True)

    def report(f: float, msg: str) -> None:
        if progress is not None:
            progress(float(min(max(f, 0.0), 1.0)), msg)

    todo: dict[str, dict] = {}                      # unique run ids still to compute
    for item in plan:
        if item["run_id"] not in done and item["run_id"] not in todo:
            todo[item["run_id"]] = item
    n_total, n_new = len(plan), len(todo)
    n_resumed = len({i["run_id"] for i in plan}) - n_new
    state = "done"
    results = dict(done)
    jf = open(journal, "a", buffering=1)

    def record(item: dict, r: dict) -> None:
        row = {"run_id": item["run_id"], "status": r["status"], "error": r["error"], "metrics": r["metrics"],
               "t_capture_s": r.get("t_capture_s", 0.0), "t_metrics_s": r.get("t_metrics_s", 0.0),
               "warnings": r.get("warnings", 0)}
        jf.write(json.dumps(row, default=float) + "\n")
        results[item["run_id"]] = row

    def keep(item) -> str | None:
        return str(cap_dir / f"{item['run_id']}.npz") if keep_captures else None

    completed = 0
    try:
        if workers <= 1:
            for item in todo.values():
                if cancel is not None and cancel():
                    state = "cancelled"
                    break
                r = run_one(item["exp"].model_dump_json(), spec.mode, exc, metrics_json, keep(item), capture_fn)
                record(item, r)
                completed += 1
                report(completed / max(n_new, 1), f"run {completed}/{n_new}")
        else:
            ctx = mp.get_context("spawn")
            per = max(1, (os.cpu_count() or 1) // workers)
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_pool_init,
                                     initargs=(per,)) as pool:
                pending, it = {}, iter(todo.values())
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < workers * 2 and not (cancel is not None and cancel()):
                        item = next(it, None)
                        if item is None:
                            exhausted = True
                            break
                        pending[pool.submit(run_one, item["exp"].model_dump_json(), spec.mode, exc,
                                            metrics_json, keep(item))] = item
                    if cancel is not None and cancel():
                        state = "cancelled"
                        exhausted = True
                    if not pending:
                        break
                    fin, _ = wait(list(pending), timeout=0.2, return_when=FIRST_COMPLETED)
                    for fut in fin:
                        item = pending.pop(fut)
                        record(item, fut.result())
                        completed += 1
                        report(completed / max(n_new, 1), f"run {completed}/{n_new}")
    finally:
        jf.close()

    rows = []
    for item in plan:
        r = results.get(item["run_id"])
        if r is None:
            continue
        row = {"run_id": item["run_id"], "point": item["point"], "replicate": item["replicate"], "seed": item["seed"],
               "exp_hash": item["exp_hash"], **{c: design.table.iloc[item["point"]][c] for c in
                                                ("block", "row", "traj", "step", "step_param", "direction")}}
        for j, p in enumerate(spec.parameters):
            row[f"p:{p.label}"] = design.values[item["point"]][j]
            row[f"u:{p.label}"] = float(design.U[item["point"], j])
        for m in spec.metrics:
            row[f"m:{m.name}"] = r["metrics"].get(m.name, float("nan"))
        row.update(status=r["status"], error=r["error"], t_capture_s=r["t_capture_s"],
                   t_metrics_s=r["t_metrics_s"], warnings=r.get("warnings", 0))
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in cols["parameters"]:                    # keep choice columns as strings / numbers consistently
        if c in df and df[c].map(lambda v: isinstance(v, (str, bool))).any():
            df[c] = df[c].astype(str)
    df.to_csv(out / "runs.csv", index=False)
    df.to_parquet(out / "runs.parquet", index=False)
    n_err = int((df["status"] != "ok").sum()) if len(df) else 0
    if state == "done" and len(df) < n_total:
        state = "error"
    manifest = {
        "name": spec.name, "state": state, "spec_hash": _spec_hash(spec), "spec": spec.model_dump(mode="json"),
        "base_experiment_hash": experiment_hash(base), "design": {"kind": design.kind, "n_points": len(design),
                                                                   "n_base": design.n_base},
        "n_runs": n_total, "n_completed": int(len(df)), "n_failed": n_err, "n_resumed": n_resumed,
        "n_unique_runs": len({i["run_id"] for i in plan}), "columns": cols, "mode": spec.mode,
        "workers": workers, "wall_time_s": time.perf_counter() - t_start,
        "provenance": provenance(), "files": {"runs_parquet": "runs.parquet", "runs_csv": "runs.csv",
                                              "journal": "runs.jsonl"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    report(1.0 if state == "done" else completed / max(n_new, 1), state)
    return StudyResult(out, state, df, manifest, design)


def load_runs(out_dir: str | Path) -> pd.DataFrame:
    """The ``runs.parquet`` (or csv) table of a finished / partial study directory."""
    p = Path(out_dir)
    if (p / "runs.parquet").exists():
        return pd.read_parquet(p / "runs.parquet")
    return pd.read_csv(p / "runs.csv")


def load_manifest(out_dir: str | Path) -> dict:
    return json.loads((Path(out_dir) / "manifest.json").read_text())
