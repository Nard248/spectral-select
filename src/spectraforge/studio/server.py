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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError

from . import exports, render_png
from .jobs import Cancelled, JobManager
from .presets import PRESETS, builtin_fluorophores, builtin_materials
from .schema import EXTINCTION_REFERENCE_M_CM, SCHEMA_VERSION, Experiment

ENGINE_VERSION = "studio-1.0"
DEFAULT_DIST = Path(__file__).resolve().parents[3] / "apps" / "studio-web" / "dist"
_EXPOSE = "X-Width, X-Height, X-Bands, Content-Disposition"


# --------------------------------------------------------------------------------------------
# Capture store
# --------------------------------------------------------------------------------------------
@dataclass
class StoredCapture:
    meta: dict
    result: Any            # contracts.CaptureResult
    experiment: Experiment
    scale: float = 1.0     # resolution scale relative to the sensor (explain_pixel coordinates)


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


def _plain(x: Any) -> Any:
    """Recursively convert numpy / non-JSON values to JSON-safe Python."""
    if isinstance(x, dict):
        return {(f"{k:g}" if isinstance(k, (float, np.floating)) else str(k)): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


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
# App factory
# --------------------------------------------------------------------------------------------
def create_app(capture_fn: Callable | None = None, preview_fn: Callable | None = None,
               explain_fn: Callable | None = None, trace_fn: Callable | None = None,
               dist_dir: Path | str | None = None, store_size: int = 10,
               warmup_start_fn: Callable | None = None, warmup_status_fn: Callable | None = None) -> FastAPI:
    """``warmup_start_fn`` / ``warmup_status_fn``: the transport warm-up hooks (``transport.start_warmup`` /
    ``transport.warmup_status``).  Off by default (so tests never trigger Monte Carlo bakes); ``main()``
    passes the real ones, and ``/api/health`` then carries an extra ``warmup`` field."""
    capture_fn = capture_fn or _default_capture
    preview_fn = preview_fn or _default_preview
    explain_fn = explain_fn or _default_explain
    trace_fn = trace_fn or _default_trace
    store = CaptureStore(store_size)
    jobs = JobManager()

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
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"], expose_headers=_EXPOSE.split(", "))

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
    def _store_result(result, exp: Experiment, mode: str, scale: float) -> dict:
        cid = uuid.uuid4().hex[:12]
        if mode in ("sample", "dark", "white_reference") and exp.acquisition.mode != mode:
            # explain_pixel() reads the mode from the experiment: keep it consistent with the capture
            exp = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"mode": mode})})
        meta = build_meta(result, exp, cid, mode)
        store.put(StoredCapture(meta=meta, result=result, experiment=exp, scale=scale))
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
        c = store.get(cid)
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
            "Content-Disposition": f'attachment; filename="{fname}"', "Access-Control-Expose-Headers": _EXPOSE})

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
    if a.open:
        url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}/"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    from spectraforge.studio import transport
    app = create_app(warmup_start_fn=transport.start_warmup, warmup_status_fn=transport.warmup_status)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
