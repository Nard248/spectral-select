"""SpectralForge Studio engine service (FastAPI). Contract: ``apps/studio-web/src/api/types.ts``.

``create_app`` takes the engine entry points as injectable callables (defaults import the real
``capture`` / ``tracer`` modules lazily) so the server can be tested with fakes.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError

from . import analysis, exports, render_png
from .jobs import Cancelled, JobManager
from .presets import PRESETS, builtin_fluorophores, builtin_materials
from .schema import EXTINCTION_REFERENCE_M_CM, SCHEMA_VERSION, Experiment
from .sweeps import SweepError, SweepManager

ENGINE_VERSION = "studio-1.0"
DEFAULT_DIST = Path(__file__).resolve().parents[3] / "apps" / "studio-web" / "dist"
_EXPOSE = "X-Width, X-Height, X-Bands, X-Dims, Content-Disposition"


# --------------------------------------------------------------------------------------------
# Capture store
# --------------------------------------------------------------------------------------------
@dataclass
class StoredCapture:
    meta: dict
    result: Any            # contracts.CaptureResult
    experiment: Experiment
    scale: float = 1.0     # resolution scale relative to the sensor (explain_pixel coordinates)
    pixel_cache: dict = field(default_factory=dict)   # (x, y) -> explained components (analysis ROI stats)


class CaptureStore:
    """Thread-safe in-memory LRU of the last ``capacity`` captures."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._d: "OrderedDict[str, StoredCapture]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, item: StoredCapture) -> None:
        with self._lock:
            self._d[item.meta["id"]] = item
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def get(self, cid: str) -> StoredCapture | None:
        with self._lock:
            item = self._d.get(cid)
            if item is not None:
                self._d.move_to_end(cid)
            return item

    def list_meta(self) -> list[dict]:
        with self._lock:
            return [it.meta for it in reversed(self._d.values())]


_plain = analysis.to_jsonable


def build_meta(result, experiment: Experiment, cid: str, mode: str) -> dict:
    """``CaptureMeta`` (types.ts) for a ``CaptureResult``."""
    cam = _active(experiment.cameras, experiment.active_camera_id)
    src = _active(experiment.sources, experiment.active_source_id)
    full_scale = float(2 ** cam.sensor.bit_depth - 1) if cam else 65535.0
    stats, sat, emission, expo, power = {}, {}, {}, {}, {}
    for ex in result.excitations_nm:
        k = f"{ex:g}"
        adu = np.asarray(result.adu[ex], dtype=np.float32)
        flat = adu.reshape(-1, adu.shape[2])
        p01, p99 = np.percentile(flat, [1, 99], axis=0)
        stats[k] = [{"min": float(mn), "max": float(mx), "p01": float(a), "p99": float(b), "mean": float(m)}
                    for mn, mx, a, b, m in zip(flat.min(0), flat.max(0), p01, p99, flat.mean(0))]
        sat[k] = float((adu >= full_scale - 0.5).mean())
        emission[k] = [float(w) for w in np.asarray(result.emission_nm[ex])]
        expo[k] = float(result.exposure_ms[ex])
        power[k] = float(result.power_mW[ex])
    return {
        "id": cid, "created": datetime.now(timezone.utc).isoformat(), "name": experiment.name, "mode": mode,
        "width": int(result.width), "height": int(result.height),
        "excitations_nm": [float(e) for e in result.excitations_nm],
        "emission_nm": emission, "exposure_ms": expo, "power_mW": power, "stats": stats,
        "saturated_fraction": sat, "ground_truth_layers": sorted(result.ground_truth.keys()),
        "warnings": [str(w) for w in result.warnings],
        "timings_s": {k: float(v) for k, v in result.timings_s.items()},
        "camera_id": cam.id if cam else "", "source_id": src.id if src else "",
        "volumes": analysis.volume_infos(result),
        "blocks": [{"object_id": str(b["object_id"]), "layer_names": [str(n) for n in b["layer_names"]],
                    "footprint_mm": [float(v) for v in b["footprint_mm"]]}
                   for b in (getattr(result, "blocks", None) or [])],
    }


def _active(items, active_id):
    if not items:
        return None
    for it in items:
        if it.id == active_id:
            return it
    return items[0]


# --------------------------------------------------------------------------------------------
# Validation / physics warnings
# --------------------------------------------------------------------------------------------
def _issues(exc: ValidationError) -> list[dict]:
    return [{"loc": list(e.get("loc", ())), "msg": str(e.get("msg", ""))} for e in exc.errors()]


def band_centers(cam, ex_nm: float) -> np.ndarray:
    f = cam.filter
    start = f.band_start_nm if f.band_start_policy == "fixed" else max(f.band_start_nm, ex_nm + f.start_offset_nm)
    return start + f.band_step_nm * np.arange(f.n_bands)


def static_warnings(exp: Experiment) -> list[dict]:
    """Cheap checks that need no ray tracing."""
    w: list[dict] = []
    add = lambda code, sev, msg, tid=None: w.append(  # noqa: E731
        {"code": code, "severity": sev, "message": msg, **({"target_id": tid} if tid else {})})
    if not exp.objects:
        add("no_objects", "info", "The scene has no objects; only the background will be imaged.")
    if not exp.sources:
        add("no_source", "warn", "No light source: nothing excites the sample.")
    if not exp.cameras:
        add("no_camera", "warn", "No camera: nothing can be captured.")
    src = _active(exp.sources, exp.active_source_id)
    cam = _active(exp.cameras, exp.active_camera_id)
    if src is not None:
        steps = [s for s in src.sequence if s.enabled]
        if not steps:
            add("no_excitation_steps", "warn", "The excitation sequence has no enabled steps.", src.id)
        for s in steps:
            if s.power_mW <= 0:
                add("zero_power", "warn", f"Excitation {s.wavelength_nm:g} nm has zero power.", src.id)
    if src is not None and cam is not None:
        lp = cam.filter.emission_longpass_nm
        leaking, beyond = [], []
        for s in (s for s in src.sequence if s.enabled):
            centers = band_centers(cam, s.wavelength_nm)
            half = cam.filter.fwhm_nm / 2
            if not centers.size:
                continue
            if centers[0] - half <= s.wavelength_nm <= centers[-1] + half and (lp is None or s.wavelength_nm >= lp):
                leaking.append(s.wavelength_nm)
            elif centers[-1] + half < s.wavelength_nm:
                beyond.append(s.wavelength_nm)
        if leaking:
            add("excitation_in_band", "warn",
                f"Excitation {', '.join(f'{v:g}' for v in leaking)} nm lies inside the camera band range and is "
                "not blocked by the emission long-pass filter; scattered excitation light will leak into the cube.",
                cam.id)
        if beyond:
            add("excitation_beyond_bands", "info",
                f"Excitation {', '.join(f'{v:g}' for v in beyond)} nm is beyond the red end of the camera bands.", cam.id)
    return w


def trace_warnings(exp: Experiment, hit) -> list[dict]:
    """Warnings derived from a coarse ``HitBuffer``."""
    w: list[dict] = []
    add = lambda code, sev, msg, tid=None: w.append(  # noqa: E731
        {"code": code, "severity": sev, "message": msg, **({"target_id": tid} if tid else {})})
    obj0 = np.asarray(hit.object_index)[0]
    on_obj = obj0 >= 0
    if not on_obj.any():
        add("camera_sees_no_objects", "warn", "The camera does not see any object (check pose and field of view).")
        return w
    irr = np.asarray(hit.irradiance_per_mW)[0]
    if float(np.max(irr[on_obj])) <= 0 or float(np.mean(irr[on_obj] > 1e-6 * max(irr.max(), 1e-30))) < 0.01:
        add("beam_misses_objects", "warn", "The excitation beam hardly illuminates any visible object.")
    cam = _active(exp.cameras, exp.active_camera_id)
    if cam is not None:
        depth = np.asarray(hit.depth_mm)[on_obj]
        depth = depth[np.isfinite(depth)]
        if depth.size:
            f, N, s = cam.lens.focal_length_mm, cam.lens.f_number, float(hit.focus_depth_mm or cam.lens.focus_distance_mm)
            c = cam.sensor.pixel_pitch_um * cam.sensor.binning * 1e-3
            H = f * f / (N * c) + f
            near = H * s / (H + (s - f)) if H + (s - f) > 0 else 0.0
            far = H * s / (H - (s - f)) if H - (s - f) > 0 else np.inf
            med = float(np.median(depth))
            if med < near or med > far:
                add("out_of_depth_of_field", "info",
                    f"Objects at ~{med:.0f} mm are outside the depth of field ({near:.0f}-{far:.0f} mm); expect blur.", cam.id)
    glint = np.asarray(hit.glint_per_mW)
    if glint.size and float(np.mean(glint > 0.01)) > 0.0005:
        add("glint_in_view", "info", "A specular reflection of the source enters the camera.")
    return w


# --------------------------------------------------------------------------------------------
# Default (lazy) engine adapters
# --------------------------------------------------------------------------------------------
def _default_capture(experiment, **kw):
    from .capture import capture
    return capture(experiment, **kw)


def _default_preview(experiment, excitation_nm=None, scale=None):
    from .capture import preview
    return preview(experiment, excitation_nm=excitation_nm, scale=scale)


def _default_explain(experiment, x, y, *, scale=1.0):
    from .capture import explain_pixel
    return explain_pixel(experiment, x, y, scale=scale)


def _default_trace(experiment, scale):
    from .tracer import trace_scene
    cam = experiment.camera()
    return trace_scene(experiment, max(8, round(cam.sensor.width_px * scale)),
                       max(8, round(cam.sensor.height_px * scale)))



# --------------------------------------------------------------------------------------------
# Browser-origin hardening (CSRF, cross-origin reads, DNS rebinding)
# --------------------------------------------------------------------------------------------
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")
_STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}


def _host_only(hostport: str) -> str:
    h = hostport.strip().lower()
    if h.startswith("["):                       # [::1]:8765
        return h.split("]")[0] + "]"
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


class SecurityMiddleware:
    """Pure-ASGI guard for HTTP *and* WebSocket scopes.

    * ``Host`` must be a loopback name (or one explicitly allowed) -> blocks DNS rebinding.
    * ``Origin`` (``Referer`` fallback) on state-changing requests and on WebSocket handshakes must be the
      served app itself (Origin host:port == Host) or a configured dev origin.  A request with neither header
      (curl, scripts) is a non-browser client and is allowed.  Any request that *does* carry a disallowed
      ``Origin`` is refused, whatever its method.
    * State-changing requests must be ``Content-Type: application/json`` (415 otherwise), so "simple"
      cross-origin form / text/plain posts cannot reach a handler.
    """

    def __init__(self, app, allowed_origins=DEV_ORIGINS, allowed_hosts=LOOPBACK_HOSTS):
        self.app = app
        self.origins = {o.lower().rstrip("/") for o in allowed_origins}
        self.hosts = {h.lower() for h in allowed_hosts}

    @staticmethod
    def _hdr(scope, name: bytes) -> str | None:
        for k, v in scope.get("headers", ()):
            if k == name:
                return v.decode("latin-1")
        return None

    def _origin_ok(self, scope, host: str | None) -> bool:
        origin = self._hdr(scope, b"origin")
        if origin is None or origin == "":
            referer = self._hdr(scope, b"referer")
            if not referer:
                return True                      # non-browser client (curl, python): nothing to forge
            from urllib.parse import urlsplit
            try:
                u = urlsplit(referer)
                origin = f"{u.scheme}://{u.netloc}"
            except ValueError:
                return False
        origin = origin.lower().rstrip("/")
        if origin in self.origins:
            return True
        scheme, sep, netloc = origin.partition("://")
        return bool(sep and scheme in ("http", "https") and host is not None and netloc == host.lower())

    async def _deny(self, scope, receive, send, status: int, msg: str):
        if scope["type"] == "websocket":
            await receive()                      # consume websocket.connect, then refuse the handshake
            await send({"type": "websocket.close", "code": 1008})
            return
        await JSONResponse({"detail": msg}, status_code=status)(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        host = self._hdr(scope, b"host")
        if host is None or _host_only(host) not in self.hosts:
            return await self._deny(scope, receive, send, 403, "forbidden host")
        method = scope.get("method", "GET")
        guarded = scope["type"] == "websocket" or method in _STATE_CHANGING or self._hdr(scope, b"origin")
        if guarded and not self._origin_ok(scope, host):
            return await self._deny(scope, receive, send, 403, "forbidden origin")
        if scope["type"] == "http" and method in _STATE_CHANGING:
            ctype = (self._hdr(scope, b"content-type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return await self._deny(scope, receive, send, 415, "Content-Type must be application/json")
        return await self.app(scope, receive, send)

# --------------------------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------------------------
def create_app(capture_fn: Callable | None = None, preview_fn: Callable | None = None,
               explain_fn: Callable | None = None, trace_fn: Callable | None = None,
               dist_dir: Path | str | None = None, store_size: int = 10,
               warmup_start_fn: Callable | None = None, warmup_status_fn: Callable | None = None,
               allowed_hosts: tuple[str, ...] = ()) -> FastAPI:
    """``warmup_start_fn`` / ``warmup_status_fn``: the transport warm-up hooks (``transport.start_warmup`` /
    ``transport.warmup_status``).  Off by default (so tests never trigger Monte Carlo bakes); ``main()``
    passes the real ones, and ``/api/health`` then carries an extra ``warmup`` field."""
    capture_fn = capture_fn or _default_capture
    preview_fn = preview_fn or _default_preview
    explain_fn = explain_fn or _default_explain
    trace_fn = trace_fn or _default_trace
    store = CaptureStore(store_size)
    jobs = JobManager()
    sweeps: SweepManager | None = None   # created below, after the capture helpers

    @asynccontextmanager
    async def lifespan(_app):
        if warmup_start_fn is not None:        # non-blocking daemon thread; previews/captures work (slower) meanwhile
            try:
                warmup_start_fn()
            except Exception:  # noqa: BLE001 - never block start-up on a warm-up problem
                pass
        yield
        jobs.shutdown()

    app = FastAPI(title="SpectralForge Studio engine", version=ENGINE_VERSION, lifespan=lifespan)
    app.state.store, app.state.jobs = store, jobs
    app.add_middleware(CORSMiddleware, allow_origins=list(DEV_ORIGINS), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], expose_headers=_EXPOSE.split(", "))
    app.add_middleware(SecurityMiddleware, allowed_origins=DEV_ORIGINS,   # added last = outermost: runs first
                       allowed_hosts=LOOPBACK_HOSTS + tuple(allowed_hosts))

    async def _json(request: Request) -> Any:
        try:
            return await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "request body is not valid JSON")

    def _parse_experiment(raw: Any) -> Experiment:
        try:
            return Experiment.model_validate(raw)
        except ValidationError as exc:
            raise HTTPException(422, detail=_issues(exc))

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        out = {"ok": True, "engine_version": ENGINE_VERSION, "schema_version": SCHEMA_VERSION}
        if warmup_status_fn is not None:
            out["warmup"] = warmup_status_fn()      # {state, fraction, message, result}
        return out

    @app.get("/api/presets")
    def presets():
        return {"schema_version": SCHEMA_VERSION, "extinction_reference_M_cm": EXTINCTION_REFERENCE_M_CM,
                "fluorophores": [f.model_dump() for f in builtin_fluorophores()],
                "materials": [m.model_dump() for m in builtin_materials()],
                "experiments": {k: f().model_dump() for k, f in PRESETS.items()}}

    @app.post("/api/validate")
    async def validate(request: Request):
        raw = await _json(request)
        try:
            exp = Experiment.model_validate(raw)
        except ValidationError as exc:
            return {"ok": False, "errors": _issues(exc), "warnings": []}
        except Exception as exc:  # noqa: BLE001 - never crash on bad input
            return {"ok": False, "errors": [{"loc": [], "msg": f"{type(exc).__name__}: {exc}"}], "warnings": []}
        warnings = await asyncio.to_thread(_all_warnings, exp)
        return {"ok": True, "errors": [], "warnings": warnings}

    def _all_warnings(exp: Experiment) -> list[dict]:
        try:
            out = static_warnings(exp)
        except Exception as exc:  # noqa: BLE001
            out = [{"code": "validator_error", "severity": "info", "message": f"static checks failed: {exc}"}]
        if exp.objects and exp.sources and exp.cameras:
            try:
                hit = trace_fn(exp, 0.125)
                out += trace_warnings(exp, hit)
            except ImportError:
                pass   # tracer not installed yet: static checks only
            except Exception as exc:  # noqa: BLE001
                out.append({"code": "trace_failed", "severity": "info",
                            "message": f"Preview trace failed: {type(exc).__name__}: {exc}"})
        return out

    # ------------------------------------------------------------------ preview / capture
    def _store_result(result, exp: Experiment, mode: str, scale: float, into: CaptureStore | None = None) -> dict:
        cid = uuid.uuid4().hex[:12]
        if mode in ("sample", "dark", "white_reference") and exp.acquisition.mode != mode:
            # explain_pixel() reads the mode from the experiment: keep it consistent with the capture
            exp = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"mode": mode})})
        meta = build_meta(result, exp, cid, mode)
        (into or store).put(StoredCapture(meta=meta, result=result, experiment=exp, scale=scale))
        return meta

    @app.post("/api/preview")
    async def preview(request: Request):
        body = await _json(request)
        if not isinstance(body, dict) or "experiment" not in body:
            raise HTTPException(422, detail=[{"loc": ["experiment"], "msg": "field required"}])
        exp = _parse_experiment(body["experiment"])
        scale = body.get("scale") or exp.acquisition.preview_scale
        ex = body.get("excitation_nm")

        def run():
            result = preview_fn(exp, excitation_nm=ex, scale=scale)
            return _store_result(result, exp, "preview", float(scale))
        try:
            return await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, detail=f"{type(exc).__name__}: {exc}")

    @app.post("/api/captures")
    async def start_capture(request: Request):
        body = await _json(request)
        if not isinstance(body, dict) or "experiment" not in body:
            raise HTTPException(422, detail=[{"loc": ["experiment"], "msg": "field required"}])
        exp = _parse_experiment(body["experiment"])
        mode = body.get("mode") or exp.acquisition.mode
        if mode not in ("sample", "dark", "white_reference"):
            raise HTTPException(422, detail=[{"loc": ["mode"], "msg": f"invalid mode '{mode}'"}])

        def run(job) -> str:
            result = capture_fn(exp, mode=mode, scale=1.0, progress=job.progress, cancel=job.cancelled)
            if job.cancelled():
                raise Cancelled()
            job.progress(0.99, "storing")
            return _store_result(result, exp, mode, 1.0)["id"]
        return {"job_id": jobs.submit(run).id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.status()

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.status()

    @app.websocket("/api/jobs/{job_id}")
    async def job_ws(ws: WebSocket, job_id: str):
        job = jobs.get(job_id)
        if job is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        i = 0
        try:
            while True:
                new = job.events_since(i)
                for ev in new:
                    await ws.send_json(ev)
                    i += 1
                    if ev["type"] in ("done", "error", "cancelled"):
                        await ws.close()
                        return
                if not new:
                    await asyncio.sleep(0.03)
        except WebSocketDisconnect:
            return

    # ------------------------------------------------------------------ capture access
    def _cap(cid: str) -> StoredCapture:
        c = store.get(cid) or (sweeps.find_capture(cid) if sweeps is not None else None)
        if c is None:
            raise HTTPException(404, "unknown capture (the server keeps only the most recent ones)")
        return c

    def _ex(c: StoredCapture, ex: float | None) -> float:
        exs = c.result.excitations_nm
        if ex is None:
            return exs[0]
        best = min(exs, key=lambda e: abs(e - ex))
        if abs(best - ex) > 0.5:
            raise HTTPException(404, f"excitation {ex:g} nm not in capture ({[f'{e:g}' for e in exs]})")
        return best

    def _cube(c: StoredCapture, ex: float, kind: str) -> np.ndarray:
        src = {"adu": c.result.adu, "radiance": c.result.radiance, "electrons": c.result.electrons_mean}.get(kind)
        if src is None:
            raise HTTPException(422, f"kind must be adu|radiance|electrons, got '{kind}'")
        return src[ex]

    def _binary(arr: np.ndarray, w: int, h: int, bands: int | None = None) -> Response:
        headers = {"X-Width": str(w), "X-Height": str(h), "Access-Control-Expose-Headers": _EXPOSE}
        if bands is not None:
            headers["X-Bands"] = str(bands)
        return Response(np.ascontiguousarray(arr, dtype="<f4").tobytes(), media_type="application/octet-stream",
                        headers=headers)

    @app.get("/api/captures")
    def list_captures():
        return store.list_meta()

    @app.get("/api/captures/{cid}")
    def get_capture(cid: str):
        return _cap(cid).meta

    @app.get("/api/captures/{cid}/band")
    def band(cid: str, ex: float | None = None, band: int = 0, kind: str = "adu"):
        c = _cap(cid)
        cube = _cube(c, _ex(c, ex), kind)
        if not 0 <= band < cube.shape[2]:
            raise HTTPException(404, f"band {band} out of range 0..{cube.shape[2] - 1}")
        return _binary(cube[:, :, band], c.result.width, c.result.height, 1)

    @app.get("/api/captures/{cid}/cube")
    def cube(cid: str, ex: float | None = None, kind: str = "adu"):
        c = _cap(cid)
        a = _cube(c, _ex(c, ex), kind)
        return _binary(a, c.result.width, c.result.height, a.shape[2])

    @app.get("/api/captures/{cid}/gt/{name}")
    def gt(cid: str, name: str):
        c = _cap(cid)
        if name not in c.result.ground_truth:
            raise HTTPException(404, f"unknown ground-truth layer '{name}'")
        try:
            a = np.asarray(c.result.ground_truth[name]).astype(np.float32)
        except (TypeError, ValueError):
            raise HTTPException(422, f"layer '{name}' is not numeric")
        return _binary(a, c.result.width, c.result.height)

    @app.get("/api/captures/{cid}/image.png")
    def image(cid: str, mode: str = "true_color", ex: float | None = None, band: int | None = None,
              gain: float = 1.0, bands: str | None = None):
        c = _cap(cid)
        e = _ex(c, ex)
        try:
            if mode == "true_color":
                rgb = render_png.true_color(c.result.radiance[e], c.result.emission_nm[e], gain)
            elif mode == "false_color":
                trip = tuple(int(v) for v in bands.split(",")) if bands else None
                if trip is not None and len(trip) != 3:
                    raise ValueError("bands must be 3 comma-separated indices (R,G,B)")
                rgb = render_png.false_color(c.result.adu[e], trip, gain)
            elif mode == "band":
                cube_ = c.result.adu[e]
                b = 0 if band is None else band
                if not 0 <= b < cube_.shape[2]:
                    raise HTTPException(404, f"band {b} out of range 0..{cube_.shape[2] - 1}")
                rgb = render_png.band_image(cube_[:, :, b], gain)
            else:
                raise HTTPException(422, "mode must be true_color|false_color|band")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return Response(render_png.encode_png(rgb), media_type="image/png")

    @app.get("/api/captures/{cid}/pixel")
    def pixel(cid: str, x: int = Query(...), y: int = Query(...)):
        c = _cap(cid)
        if not (0 <= x < c.result.width and 0 <= y < c.result.height):
            raise HTTPException(404, f"pixel ({x}, {y}) outside {c.result.width}x{c.result.height}")
        try:
            d = explain_fn(c.experiment, x, y, scale=c.scale)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
        out = _plain(d)
        out.setdefault("x", x)
        out.setdefault("y", y)
        out.setdefault("emission_nm", {f"{e:g}": [float(v) for v in c.result.emission_nm[e]]
                                       for e in c.result.excitations_nm})
        out.setdefault("adu", {f"{e:g}": [float(v) for v in c.result.adu[e][y, x]] for e in c.result.excitations_nm})
        out.setdefault("radiance", {f"{e:g}": [float(v) for v in c.result.radiance[e][y, x]]
                                    for e in c.result.excitations_nm})
        out.setdefault("components", {})
        out.setdefault("ground_truth", {})
        return out

    @app.get("/api/captures/{cid}/download")
    def download(cid: str, format: str = "spectradata_pkl"):
        c = _cap(cid)
        try:
            payload, fname, media = exports.build_export(c.result, cid, c.meta["name"], format)
        except KeyError:
            raise HTTPException(422, f"format must be one of {sorted(exports.FORMATS)}")
        return Response(payload, media_type=media, headers={
            "Content-Disposition": exports.content_disposition(fname), "Access-Control-Expose-Headers": _EXPOSE})

    # ------------------------------------------------------------------ v1.1: sweeps
    def _sweep_point(exp: Experiment, mode: str, progress, cancel, into: CaptureStore) -> str:
        if mode == "preview":
            scale = exp.acquisition.preview_scale
            progress(0.0, "preview")
            result = preview_fn(exp, excitation_nm=None, scale=scale)
            if cancel():
                raise Cancelled()
            return _store_result(result, exp, "preview", float(scale), into)["id"]
        result = capture_fn(exp, mode="sample", scale=1.0, progress=progress, cancel=cancel)
        if cancel():
            raise Cancelled()
        return _store_result(result, exp, "sample", 1.0, into)["id"]

    sweeps = SweepManager(jobs, _sweep_point, lambda n: CaptureStore(max(1, n)))
    app.state.sweeps = sweeps

    @app.post("/api/sweeps")
    async def start_sweep(request: Request):
        body = await _json(request)
        if not isinstance(body, dict) or "experiment" not in body:
            raise HTTPException(422, detail=[{"loc": ["experiment"], "msg": "field required"}])
        exp = _parse_experiment(body["experiment"])
        param = body.get("parameter")
        if not isinstance(param, dict) or "path" not in param or "values" not in param:
            raise HTTPException(422, detail=[{"loc": ["parameter"], "msg": "parameter.path and parameter.values are required"}])
        try:
            sw = await asyncio.to_thread(sweeps.start, exp, param, body.get("mode") or "preview")
        except SweepError as exc:
            raise HTTPException(422, str(exc))
        return {"sweep_id": sw.id}

    def _sweep(sweep_id: str):
        sw = sweeps.get(sweep_id)
        if sw is None:
            raise HTTPException(404, "unknown sweep")
        return sw

    @app.get("/api/sweeps/{sweep_id}")
    def sweep_status(sweep_id: str):
        return _sweep(sweep_id).status()

    @app.post("/api/sweeps/{sweep_id}/cancel")
    def sweep_cancel(sweep_id: str):
        _sweep(sweep_id)
        return sweeps.cancel(sweep_id).status()

    @app.websocket("/api/sweeps/{sweep_id}")
    async def sweep_ws(ws: WebSocket, sweep_id: str):
        sw = sweeps.get(sweep_id)
        if sw is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        i = 0
        try:
            while True:
                new = sw.events_since(i)
                for ev in new:
                    await ws.send_json(_plain(ev))
                    i += 1
                    if ev["type"] in ("done", "error", "cancelled"):
                        await ws.close()
                        return
                if not new:
                    await asyncio.sleep(0.03)
        except WebSocketDisconnect:
            return

    # ------------------------------------------------------------------ v1.1: analysis
    def _norm_inputs(norm_raw):
        try:
            norm = analysis.coerce_norm(norm_raw)
        except analysis.AnalysisError as exc:
            raise HTTPException(exc.status, str(exc))
        dark = white = None
        if norm.dark_capture_id:
            dark = store_get(norm.dark_capture_id, "dark").result
        if norm.white_capture_id:
            white = store_get(norm.white_capture_id, "white").result
        return norm, dark, white

    def store_get(cid: str, role: str) -> StoredCapture:
        c = store.get(cid) or sweeps.find_capture(cid)
        if c is None:
            raise HTTPException(404, f"unknown {role} capture '{cid}'")
        return c

    async def _body(request: Request) -> dict:
        body = await _json(request)
        if not isinstance(body, dict):
            raise HTTPException(422, "request body must be a JSON object")
        return body

    async def _analyse(fn: Callable[[], Any]) -> Any:
        try:
            return await asyncio.to_thread(fn)
        except analysis.AnalysisError as exc:
            raise HTTPException(exc.status, str(exc))
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(422, f"{type(exc).__name__}: {exc}")
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")

    @app.post("/api/captures/{cid}/eem")
    async def eem_endpoint(cid: str, request: Request):
        c, body = _cap(cid), await _body(request)
        if "roi" not in body:
            raise HTTPException(422, detail=[{"loc": ["roi"], "msg": "field required"}])
        norm, dark, white = _norm_inputs(body.get("norm"))
        return await _analyse(lambda: analysis.eem(c.result, body["roi"], norm, body.get("stat") or "mean",
                                                   dark=dark, white=white).to_dict())

    @app.post("/api/captures/{cid}/roi")
    async def roi_endpoint(cid: str, request: Request):
        c, body = _cap(cid), await _body(request)
        if "roi" not in body:
            raise HTTPException(422, detail=[{"loc": ["roi"], "msg": "field required"}])
        norm, dark, white = _norm_inputs(body.get("norm"))
        want = bool(body.get("components"))
        explain = (lambda x, y: explain_fn(c.experiment, x, y, scale=c.scale)) if want else None
        return await _analyse(lambda: analysis.roi_stats(
            c.result, body["roi"], norm, body.get("stat") or "mean", components=want, explain=explain,
            cache=c.pixel_cache, dark=dark, white=white).to_dict())

    @app.post("/api/captures/{cid}/line")
    async def line_endpoint(cid: str, request: Request):
        c, body = _cap(cid), await _body(request)
        for k in ("p0", "p1"):
            if k not in body:
                raise HTTPException(422, detail=[{"loc": [k], "msg": "field required"}])
        norm, dark, white = _norm_inputs(body.get("norm"))
        return await _analyse(lambda: analysis.line_profile(
            c.result, body["p0"], body["p1"], width_px=float(body.get("width_px") or 1.0), norm=norm,
            excitations_nm=body.get("excitations_nm"), ground_truth=body.get("ground_truth"),
            dark=dark, white=white).to_dict())

    @app.post("/api/captures/{cid}/response")
    async def response_endpoint(cid: str, request: Request):
        c, body = _cap(cid), await _body(request)
        for k in ("x_ground_truth", "metric"):
            if k not in body:
                raise HTTPException(422, detail=[{"loc": [k], "msg": "field required"}])
        norm, dark, white = _norm_inputs(body.get("norm"))
        return await _analyse(lambda: analysis.response_curve(
            c.result, body["x_ground_truth"], body["metric"], body.get("roi"), int(body.get("bins") or 24), norm,
            dark=dark, white=white).to_dict())

    @app.get("/api/captures/{cid}/volumes")
    def volumes_endpoint(cid: str):
        return _plain(analysis.volume_infos(_cap(cid).result))

    @app.get("/api/captures/{cid}/volume")
    async def volume_endpoint(cid: str, object: str = Query(...), kind: str = Query(...),
                              ex: float | None = None, channel: str | None = None):
        c = _cap(cid)
        a = await _analyse(lambda: analysis.volume(c.result, object, kind, ex, channel))
        return Response(np.ascontiguousarray(a, dtype="<f4").tobytes(), media_type="application/octet-stream",
                        headers={"X-Dims": ",".join(str(d) for d in a.shape), "Access-Control-Expose-Headers": _EXPOSE})

    # ------------------------------------------------------------------ v1.2: studies, methods text, inverse
    from . import routes_study
    app.include_router(routes_study.router)
    import importlib.util
    if importlib.util.find_spec(f"{__package__}.routes_inverse") is not None:   # W4's router is optional
        try:
            from . import routes_inverse
            app.include_router(routes_inverse.router)
        except Exception as exc:  # noqa: BLE001 - a broken optional module must not take the server down
            print(f"warning: routes_inverse not loaded: {type(exc).__name__}: {exc}", flush=True)

    # ------------------------------------------------------------------ SPA
    dist = Path(dist_dir) if dist_dir is not None else DEFAULT_DIST
    if (dist / "index.html").is_file():
        root = dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            f = (root / path).resolve()
            if path and f.is_file() and root in f.parents:
                return FileResponse(f)
            return FileResponse(root / "index.html")
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse
    import webbrowser

    import uvicorn

    p = argparse.ArgumentParser(prog="spectralforge-studio", description="SpectralForge Studio engine + web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open the browser once the server is up")
    a = p.parse_args(argv)
    extra_hosts: tuple[str, ...] = ()
    if a.host not in LOOPBACK_HOSTS:
        print(f"\n*** SECURITY WARNING: binding to {a.host!r}, which is not loopback. The engine has no "
              "authentication: anyone who can reach this port can run captures and download data. "
              "Use only on a trusted network, or keep the default 127.0.0.1. ***\n", flush=True)
        if a.host not in ("0.0.0.0", "::"):
            extra_hosts = (a.host.lower(),)
    if a.open:
        url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}/"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    from spectraforge.studio import transport
    app = create_app(warmup_start_fn=transport.start_warmup, warmup_status_fn=transport.warmup_status,
                     allowed_hosts=extra_hosts)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
