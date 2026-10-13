"""HTTP API of the inverse-problem tools (v1.2, W4): unmixing, surrogates, parameter fits, Fisher information, design.

``router`` is a FastAPI ``APIRouter``; the server mounts it with ``app.include_router(router)``.  It reads the shared
state ``app.state.store`` (capture store), ``app.state.jobs`` (the single-worker ``JobManager``) and optionally
``app.state.sweeps`` (sweep captures are found by id too).  Anything that runs the simulator (surrogates, finite-difference
Fisher information) is a **job** on that worker (same ``/api/jobs/{id}`` polling / WebSocket as captures; the ``done`` event's
``capture_id`` field carries the *inverse result id*, fetch it with ``GET /api/inverse/results/{id}``).  Everything that only
evaluates a surrogate or unmixes a stored capture is synchronous (runs in a thread, bounded sizes).

Full request / response documentation: ``docs/studio/INVERSE_API.md``.  JSON only (binary maps: ``GET /api/unmix/{id}/map/{name}``).
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import ValidationError

from . import analysis, information, inverse, unmix
from .schema import Experiment
from .surrogate import ChannelTable, InverseError, NoiseModel, Surrogate, build_surrogate, coerce_params
from .sweeps import SweepError

router = APIRouter()

MAX_SIM_RUNS = 128            # simulator evaluations in one surrogate / Fisher job
MAX_UNMIX_PIXELS = 200_000
MAX_MAP_TILES = 20_000
LRU = 12


class _Registry:
    """Thread-safe LRU of heterogeneous inverse objects (surrogates, results, unmix results)."""

    def __init__(self, capacity: int = LRU):
        self.capacity = capacity
        self._d: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, obj: Any, kind: str) -> str:
        key = f"{kind[:3]}-{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._d[key] = obj
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)
        return key

    def get(self, key: str, kind: str | None = None) -> Any:
        with self._lock:
            obj = self._d.get(key)
            if obj is not None:
                self._d.move_to_end(key)
        if obj is None or (kind and not key.startswith(kind[:3])):
            raise HTTPException(404, f"unknown {kind or 'object'} '{key}'")
        return obj

    def keys(self, kind: str) -> list[str]:
        with self._lock:
            return [k for k in reversed(self._d) if k.startswith(kind[:3])]


def _reg(request: Request) -> _Registry:
    st = request.app.state
    if not hasattr(st, "inverse"):
        st.inverse = _Registry()
    return st.inverse


def _capture_fn(request: Request) -> Callable | None:
    return getattr(request.app.state, "inverse_capture_fn", None)       # tests inject fakes; None -> real engine


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "request body is not valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(422, "request body must be a JSON object")
    return body


def _experiment(body: dict) -> Experiment:
    if "experiment" not in body:
        raise HTTPException(422, detail=[{"loc": ["experiment"], "msg": "field required"}])
    try:
        return Experiment.model_validate(body["experiment"])
    except ValidationError as exc:
        raise HTTPException(422, detail=[{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()[:20]])


def _stored_capture(request: Request, cid: str):
    st = request.app.state
    c = st.store.get(cid) if hasattr(st, "store") else None
    if c is None and getattr(st, "sweeps", None) is not None:
        c = st.sweeps.find_capture(cid)
    if c is None:
        raise HTTPException(404, f"unknown capture '{cid}'")
    return c


async def _run(fn: Callable[[], Any]) -> Any:
    try:
        return await asyncio.to_thread(fn)
    except HTTPException:
        raise
    except analysis.AnalysisError as exc:
        raise HTTPException(exc.status, str(exc))
    except (InverseError, SweepError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")


def _jsonable(x: Any) -> Any:
    return analysis.to_jsonable(x)


def _submit(request: Request, fn: Callable[[Any], Any], kind: str) -> dict:
    """Queue ``fn(job) -> result object`` on the shared worker; the job result is the registered id."""
    reg, jobs = _reg(request), request.app.state.jobs

    def run(job) -> str:
        obj = fn(job)
        return reg.put(obj, kind)
    return {"job_id": jobs.submit(run).id}


# --------------------------------------------------------------------------------------------
# unmixing
# --------------------------------------------------------------------------------------------
def _endmembers(c, spec: dict, mode_signal: str, ex: list[float] | None) -> unmix.Endmembers:
    src = spec.get("source", "forward_model")
    signal = spec.get("signal", mode_signal)
    if src == "forward_model":
        return unmix.endmembers_from_forward_model(
            c.experiment, c.result, scale=c.scale, groups=spec.get("groups"), n_candidates=int(spec.get("n_candidates", 48)),
            n_average=int(spec.get("n_average", 3)), seed=int(spec.get("seed", 0)), signal=signal)
    if src == "rois":
        if not isinstance(spec.get("rois"), dict) or not spec["rois"]:
            raise InverseError("endmembers.rois must be an object {name: roi}")
        return unmix.endmembers_from_rois(c.result, spec["rois"], signal=signal, excitations=ex)
    if src in ("nmf", "vca"):
        Y, table = unmix.capture_matrix(c.result, signal, ex)
        k = int(spec.get("n_components", 3))
        mask = unmix.object_mask(c.result)
        fn = unmix.endmembers_nmf if src == "nmf" else unmix.endmembers_vca
        return fn(Y, table, k, seed=int(spec.get("seed", 0)), mask=mask, signal=signal)
    raise InverseError("endmembers.source must be 'forward_model', 'rois', 'nmf' or 'vca'")


@router.post("/api/unmix")
async def unmix_endpoint(request: Request):
    """Body: ``{capture_id, method?: 'nnls'|'fcls'|'ls', endmembers?: {source, ...}, weighting?, signal?, excitations?,
    compare_gt?: {n_pixels, seed}, include_maps?: bool}``.  Returns the endmembers, summary statistics, optional GT
    comparison and an ``unmix_id`` for the binary maps."""
    body = await _json(request)
    c = _stored_capture(request, str(body.get("capture_id", "")))
    reg = _reg(request)

    def work() -> dict:
        res = c.result
        if res.width * res.height > MAX_UNMIX_PIXELS:
            raise InverseError(f"capture has {res.width * res.height} pixels; the limit is {MAX_UNMIX_PIXELS}")
        ex = body.get("excitations")
        em = _endmembers(c, body.get("endmembers") or {}, body.get("signal", "adu_net"), ex)
        noise = None
        if body.get("weighting") == "poisson":
            from .surrogate import noise_model_for
            noise = noise_model_for(c.experiment, em.channels, res.exposure_ms, 1.0)
        mask = unmix.object_mask(res)
        r = unmix.unmix(res, em, body.get("method", "nnls"), mask=mask, weighting=body.get("weighting", "none"),
                        noise=noise, excitations=ex)
        out: dict = {"names": r.names, "method": r.method, "endmembers": em.to_dict(), "meta": r.meta,
                     "shape": [res.height, res.width],
                     "mean_residual_fraction": float(np.nanmean(r.residual_fraction[mask])) if mask.any() else None,
                     "mean_abundance": {n: float(np.mean(r.abundances[..., i][mask])) for i, n in enumerate(r.names)}}
        gt = body.get("compare_gt")
        if gt:
            if not em.meta.get("groups"):
                raise InverseError("compare_gt needs forward-model endmembers (it uses their component groups)")
            rng = np.random.default_rng(int(gt.get("seed", 0)))
            ys, xs = np.nonzero(mask)
            pick = rng.choice(xs.size, size=min(int(gt.get("n_pixels", 40)), xs.size), replace=False)
            px = [(int(xs[i]), int(ys[i])) for i in pick]
            amounts = unmix.gt_component_amounts(c.experiment, res, em, px, scale=c.scale)
            est = np.array([r.abundances[y, x] for x, y in px])
            out["gt_comparison"] = unmix.compare_with_gt(est, amounts, em.names)
        if body.get("include_maps"):
            out["maps"] = r.to_dict(with_maps=True)
        out["unmix_id"] = reg.put(r, "unmix")
        return _jsonable(out)
    return await _run(work)


@router.get("/api/unmix/{uid}/map/{name}")
async def unmix_map(uid: str, name: str, request: Request):
    """float32 little-endian ``H*W`` map: an endmember abundance, ``rmse`` or ``residual_fraction``."""
    r: unmix.UnmixResult = _reg(request).get(uid, "unmix")
    if name in r.names:
        a = r.abundances[..., r.names.index(name)]
    elif name == "rmse":
        a = r.rmse
    elif name == "residual_fraction":
        a = r.residual_fraction
    else:
        raise HTTPException(404, f"unknown map '{name}' (have {r.names + ['rmse', 'residual_fraction']})")
    a = np.ascontiguousarray(a, dtype="<f4")
    return Response(a.tobytes(), media_type="application/octet-stream",
                    headers={"X-Width": str(a.shape[1]), "X-Height": str(a.shape[0])})


# --------------------------------------------------------------------------------------------
# surrogates and fits
# --------------------------------------------------------------------------------------------
def _grid_size(grid: Any, params: list, mode: str, n_points: Any) -> int:
    if mode == "sobol":
        return int(n_points or 0)
    if not isinstance(grid, dict):
        raise InverseError("grid must be an object {parameter name: [values]}")
    n = 1
    for p in params:
        v = grid.get(p.name)
        if not isinstance(v, list):
            raise InverseError(f"grid is missing the list of values for '{p.name}'")
        n *= len(set(map(float, v)))
    return n


def _surrogate_summary(s: Surrogate, sid: str | None = None) -> dict:
    d = s.to_dict(include_values=False)
    d.update(loo=s.loo_error(), n_channels=s.channels.n)
    if sid:
        d["surrogate_id"] = sid
    return d


@router.post("/api/surrogates")
async def create_surrogate(request: Request):
    """Job.  Body: ``{experiment, parameters:[{name, path|paths, mode?, lo?, hi?}], grid, mode?:'grid'|'sobol', n_points?,
    roi?, scale?, excitations?, overrides?, smooth?, interp?, n_pixels?}``.  Result: ``{surrogate_id, ...summary}``."""
    body = await _json(request)
    exp = _experiment(body)
    try:
        params = coerce_params(body.get("parameters") or [])
        mode = body.get("mode", "grid")
        n = _grid_size(body.get("grid"), params, mode, body.get("n_points"))
    except (InverseError, ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    if not 2 <= n <= MAX_SIM_RUNS:
        raise HTTPException(422, f"a surrogate needs 2..{MAX_SIM_RUNS} simulator evaluations (got {n})")
    cache = body.get("cache", True)
    if not isinstance(cache, bool):          # a string would become a filesystem path: server uses its own directory
        raise HTTPException(422, "cache must be a boolean")
    cap = _capture_fn(request)
    kw = {k: body[k] for k in ("roi", "scale", "excitations", "overrides", "smooth", "interp", "n_pixels", "n_points",
                               "seed") if k in body}

    reg = _reg(request)

    def run(job) -> str:
        s = build_surrogate(exp, params, body["grid"], mode=mode, capture_fn=cap, cache=cache,
                            progress=job.progress, cancel=job.cancelled, **kw)
        sid = reg.put(s, "surrogate")
        s.meta["surrogate_id"] = sid
        return sid
    return {"job_id": request.app.state.jobs.submit(run).id, "n_simulator_runs": n}


@router.get("/api/surrogates")
async def list_surrogates(request: Request):
    reg = _reg(request)
    return [{"surrogate_id": k, "names": reg.get(k).names, "n_points": int(reg.get(k).theta.shape[0])}
            for k in reg.keys("surrogate")]


@router.get("/api/surrogates/{sid}")
async def get_surrogate(sid: str, request: Request, values: bool = False):
    s: Surrogate = _reg(request).get(sid, "surrogate")
    return _jsonable(s.to_dict(include_values=values) | {"loo": s.loo_error(), "surrogate_id": sid})


@router.post("/api/surrogates/{sid}/predict")
async def predict_surrogate(sid: str, request: Request):
    """Body ``{theta: [[...], ...]}`` -> expected signal electrons ``(N, C)`` (cheap)."""
    s: Surrogate = _reg(request).get(sid, "surrogate")
    body = await _json(request)
    th = np.asarray(body.get("theta"), float)
    if th.size == 0 or th.shape[-1] != s.n_params and not (th.ndim == 1 and s.n_params == 1):
        raise HTTPException(422, f"theta must have {s.n_params} column(s)")
    return _jsonable({"channels": s.channels.to_dict(), "mean_e": s.predict_e(th)})


def _observation(request: Request, s: Surrogate, spec: dict) -> tuple[np.ndarray, int | None]:
    if "values_e" in spec:
        y = np.asarray(spec["values_e"], float)
        if y.size != s.channels.n:
            raise InverseError(f"values_e has {y.size} entries, the surrogate has {s.channels.n} channels")
        return y, None
    c = _stored_capture(request, str(spec.get("capture_id", "")))
    return inverse.observation_from_capture(c.result, s, spec.get("roi"), kind=spec.get("kind", "adu"))


@router.post("/api/fit")
async def fit_endpoint(request: Request):
    """Synchronous.  Body: ``{surrogate_id, observation: {capture_id, roi?, kind?} | {values_e}, x0?, bounds?, profile?,
    level?}`` -> :meth:`FitResult.to_dict` (estimate, covariance, profile-likelihood CIs).
    With ``map: {tile, gt_name?, min_signal_e?}`` and ``observation.capture_id`` it retrieves a parameter map (limit
    ``MAX_MAP_TILES`` tiles) and returns bias / RMSE against ``gt_name``; maps are returned as nested lists."""
    body = await _json(request)
    reg = _reg(request)
    s: Surrogate = reg.get(str(body.get("surrogate_id", "")), "surrogate")

    def work() -> dict:
        if body.get("map"):
            m = body["map"]
            c = _stored_capture(request, str((body.get("observation") or {}).get("capture_id", "")))
            tile = int(m.get("tile", 4))
            n_tiles = (c.result.height // max(tile, 1)) * (c.result.width // max(tile, 1))
            if n_tiles > MAX_MAP_TILES:
                raise InverseError(f"{n_tiles} tiles exceeds the limit {MAX_MAP_TILES}; increase 'tile'")
            r = inverse.retrieve_map(c.result, s, tile=tile, gt_name=m.get("gt_name"),
                                     min_signal_e=float(m.get("min_signal_e", 0.0)), kind=m.get("kind", "adu"))
            out = r.to_dict(with_maps=True)
            out["map_id"] = reg.put(r, "map")
            return _jsonable(out)
        y, n = _observation(request, s, body.get("observation") or {})
        noise = s.noise.with_n_pix(n) if (n and s.noise is not None and body.get("observation", {}).get("roi") is not None
                                          and body.get("observation", {}).get("rescale_noise")) else None
        r = inverse.fit(y, s, noise, body.get("x0"), bounds=body.get("bounds"), profile=bool(body.get("profile")),
                        level=float(body.get("level", 0.95)))
        return _jsonable(r.to_dict() | {"n_roi_pixels": n})
    return await _run(work)


# --------------------------------------------------------------------------------------------
# Fisher information and design
# --------------------------------------------------------------------------------------------
def _fisher_from_body(request: Request, body: dict, job=None) -> information.FisherResult:
    reg = _reg(request)
    if body.get("surrogate_id"):
        s: Surrogate = reg.get(str(body["surrogate_id"]), "surrogate")
        theta = body.get("theta")
        return information.fisher_from_surrogate(s, theta, bands=_bands(body.get("bands")),
                                                 excitations=body.get("excitations") or "all")
    exp = _experiment(body)
    params = coerce_params(body.get("parameters") or [])
    runs = len(params) * (2 if body.get("scheme", "central") == "central" else 1) + 1
    if runs > MAX_SIM_RUNS:
        raise InverseError("too many parameters")
    kw = {k: body[k] for k in ("roi", "step", "theta0", "scale", "overrides", "n_pixels", "scheme") if k in body}
    return information.fisher(exp, params, bands=_bands(body.get("bands")), excitations=body.get("excitations") or "all",
                              capture_fn=_capture_fn(request), progress=None if job is None else job.progress, **kw)


def _bands(b: Any):
    if b is None or b == "all":
        return "all"
    if isinstance(b, dict) and "from_nm" in b:
        return (float(b["from_nm"]), float(b["to_nm"]))
    return [float(v) for v in b]


@router.post("/api/fisher")
async def fisher_endpoint(request: Request):
    """Body: either ``{surrogate_id, theta?, bands?, excitations?}`` (synchronous, microseconds) or
    ``{experiment, parameters, bands?, excitations?, roi?, step?, theta0?, scale?, overrides?, scheme?}`` (a job: finite
    differences of the simulator, 2P+1 runs).  Result: names, theta, Fisher matrix, CRLB covariance / sd, per-channel
    information.  Synchronous responses also carry ``fisher_id`` (usable in ``/api/design``)."""
    body = await _json(request)
    reg = _reg(request)
    if body.get("surrogate_id"):
        def work():
            fi = _fisher_from_body(request, body)
            return _jsonable(fi.to_dict() | {"fisher_id": reg.put(fi, "fisher")})
        return await _run(work)
    _experiment(body)
    try:
        coerce_params(body.get("parameters") or [])
    except InverseError as exc:
        raise HTTPException(422, str(exc))

    def fn(job):
        return _fisher_from_body(request, body, job)
    return _submit(request, fn, "fisher")


@router.get("/api/inverse/results/{rid}")
async def get_result(rid: str, request: Request):
    """Fetch a finished job's result (``fisher`` / ``map`` / ``unmix``: JSON; ``surrogate`` summary)."""
    obj = _reg(request).get(rid)
    if isinstance(obj, information.FisherResult):
        return _jsonable(obj.to_dict() | {"fisher_id": rid})
    if isinstance(obj, Surrogate):
        return _jsonable(_surrogate_summary(obj, rid))
    if isinstance(obj, inverse.MapResult):
        return _jsonable(obj.to_dict(with_maps=True))
    if isinstance(obj, unmix.UnmixResult):
        return _jsonable(obj.to_dict(with_maps=False))
    if isinstance(obj, _DesignBundle):
        return _jsonable(obj.to_dict() | {"design_id": rid})
    raise HTTPException(404, "result has no JSON form")


@router.post("/api/design")
async def design_endpoint(request: Request):
    """Body: ``{fisher_id | surrogate_id (+theta) | experiment + parameters (job), k, criterion?: 'D'|'A'|'target:<p>',
    budget?: 'channels'|'exposure_time', total_time_s?, bands?, excitations?}``.  Greedy channel selection (with the
    exchange refinement and an exhaustive check when feasible), the CRLB after 1..k channels and, for
    ``exposure_time``, the optimal allocation.  Synchronous unless the Fisher information must be computed from the
    simulator (then a job; fetch the result with ``GET /api/inverse/results/{id}``)."""
    body = await _json(request)
    reg = _reg(request)
    k = int(body.get("k", 0))
    if k < 1:
        raise HTTPException(422, "k must be >= 1")

    def design(fi):
        d = information.optimal_design(fi, k, body.get("criterion", "D"), body.get("budget", "channels"),
                                       total_time_s=body.get("total_time_s"))
        return d

    if body.get("fisher_id") or body.get("surrogate_id"):
        def work():
            fi = reg.get(str(body["fisher_id"]), "fisher") if body.get("fisher_id") else _fisher_from_body(request, body)
            d = design(fi)
            return _jsonable(d.to_dict() | {"all_channels_sd": fi.sd, "names": fi.names})
        return await _run(work)
    _experiment(body)

    def fn(job):
        fi = _fisher_from_body(request, body, job)
        d = design(fi)
        return _DesignBundle(fi, d)
    return _submit(request, fn, "design")


class _DesignBundle:
    def __init__(self, fi, d):
        self.fisher, self.design = fi, d

    def to_dict(self) -> dict:
        return self.design.to_dict() | {"all_channels_sd": self.fisher.sd, "names": self.fisher.names}
