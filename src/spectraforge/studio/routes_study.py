"""HTTP routes for computational studies, Methods text and provenance (``docs/studio/STUDY_API.md``).

The router is state-free: it reads ``request.app.state.jobs`` (the shared single heavy-worker ``JobManager``, so a study
never runs concurrently with a capture or a sweep) and keeps its registry of studies in ``app.state.studies``.
Studies are written to ``$SPECTRAFORGE_STUDY_DIR`` (default: a ``spectraforge_studies`` folder in the system temp dir).
``app.state.study_capture_fn`` may hold an injected capture function (tests).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from . import analysis
from .jobs import TERMINAL, Cancelled
from .provenance import provenance
from .schema import Experiment

router = APIRouter()
MAX_ROWS = 5000


def studies_root() -> Path:
    p = Path(os.environ.get("SPECTRAFORGE_STUDY_DIR") or Path(tempfile.gettempdir()) / "spectraforge_studies")
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class StudyRecord:
    id: str
    name: str
    out_dir: Path
    n_runs: int
    state: str = "queued"
    fraction: float = 0.0
    message: str = "queued"
    error: str | None = None
    job_id: str | None = None
    events: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    sens_cache: dict = field(default_factory=dict)

    def emit(self, ev: dict) -> None:
        with self.lock:
            self.events.append(ev)

    def events_since(self, i: int) -> list[dict]:
        with self.lock:
            return list(self.events[i:])

    def finish(self, state: str, ev: dict, error: str | None = None) -> None:
        with self.lock:
            if self.state in TERMINAL:
                return
            self.state, self.error = state, error
            if state == "done":
                self.fraction = 1.0
            self.events.append(ev)

    def status(self) -> dict:
        with self.lock:
            out = {"study_id": self.id, "name": self.name, "state": self.state, "fraction": self.fraction,
                   "message": self.message, "n_runs": self.n_runs, "error": self.error}
        man = self.out_dir / "manifest.json"
        if self.state in TERMINAL and man.exists():
            out["manifest"] = json.loads(man.read_text())
        return out


def _registry(request_or_ws) -> dict[str, StudyRecord]:
    st = request_or_ws.app.state
    if not hasattr(st, "studies"):
        st.studies = {}
    return st.studies


def _get(request_or_ws, sid: str) -> StudyRecord:
    rec = _registry(request_or_ws).get(sid)
    if rec is None:
        raise HTTPException(404, "unknown study")
    return rec


def _issues(exc: ValidationError) -> list[dict]:
    return [{"loc": list(e.get("loc", ())), "msg": str(e.get("msg", ""))} for e in exc.errors()]


async def _json(request: Request) -> Any:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "request body is not valid JSON")


@router.post("/api/studies")
async def start_study(request: Request):
    """Body: a ``StudySpec`` (JSON).  Query ``keep_captures=1`` stores every capture as npz.  -> 200 ``{study_id, n_runs, n_points}``."""
    from .study import StudyError, plan_runs, run_study
    from .study_schema import StudySpec
    body = await _json(request)
    try:
        spec = StudySpec.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(422, detail=_issues(exc))
    keep = request.query_params.get("keep_captures", "0") in ("1", "true", "yes")
    try:
        _base, design, plan = await asyncio.to_thread(plan_runs, spec)
    except StudyError as exc:
        raise HTTPException(422, str(exc))
    sid = uuid.uuid4().hex[:12]
    rec = StudyRecord(id=sid, name=spec.name, out_dir=studies_root() / sid, n_runs=len(plan))
    _registry(request)[sid] = rec
    capture_fn = getattr(request.app.state, "study_capture_fn", None)

    def job_fn(job) -> str:
        with rec.lock:
            rec.state = "running"

        def progress(f: float, msg: str) -> None:
            with rec.lock:
                rec.fraction, rec.message = max(rec.fraction, min(f, 0.999)), msg
                rec.events.append({"type": "progress", "fraction": rec.fraction, "message": msg})
            job.progress(f, msg)

        try:
            res = run_study(spec, rec.out_dir, workers=1, resume=True, progress=progress, cancel=job.cancelled,
                            keep_captures=keep, capture_fn=capture_fn)
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            rec.finish("error", {"type": "error", "message": msg}, msg)
            return sid
        if res.state == "cancelled" or job.cancelled():
            rec.finish("cancelled", {"type": "cancelled"})
            raise Cancelled()
        if res.state == "error":
            rec.finish("error", {"type": "error", "message": "some runs failed"}, "some runs failed")
        else:
            rec.finish("done", {"type": "done", "study_id": sid, "n_failed": res.manifest["n_failed"]})
        return sid

    rec.job_id = request.app.state.jobs.submit(job_fn).id
    return {"study_id": sid, "n_runs": len(plan), "n_points": len(design)}


@router.get("/api/studies")
def list_studies(request: Request):
    return [r.status() for r in reversed(list(_registry(request).values()))]


@router.get("/api/studies/{sid}")
def study_status(sid: str, request: Request):
    return _get(request, sid).status()


@router.post("/api/studies/{sid}/cancel")
def study_cancel(sid: str, request: Request):
    rec = _get(request, sid)
    if rec.job_id:
        request.app.state.jobs.cancel(rec.job_id)
    with rec.lock:
        queued = rec.state == "queued"
    if queued:
        rec.finish("cancelled", {"type": "cancelled"})
    return rec.status()


@router.websocket("/api/studies/{sid}/events")
async def study_events(ws: WebSocket, sid: str):
    rec = _registry(ws).get(sid)
    if rec is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    i = 0
    try:
        while True:
            new = rec.events_since(i)
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


@router.get("/api/studies/{sid}/runs")
def study_runs(sid: str, request: Request, offset: int = 0, limit: int = 1000):
    """Rows of ``runs.parquet`` as JSON ``{columns, rows, total}`` (NaN -> null; at most 5000 rows per call)."""
    from .study import load_runs
    rec = _get(request, sid)
    if not (rec.out_dir / "runs.parquet").exists():
        raise HTTPException(409, "the study has not written results yet")
    df = load_runs(rec.out_dir)
    part = df.iloc[max(0, offset): max(0, offset) + max(1, min(limit, MAX_ROWS))]
    return {"columns": list(df.columns), "total": int(len(df)),
            "rows": analysis.to_jsonable(part.astype(object).where(part.notna(), None).to_dict(orient="records"))}


@router.get("/api/studies/{sid}/sensitivity")
async def study_sensitivity(sid: str, request: Request, boot: int = 300):
    """Sobol' (saltelli design) or Morris (morris design) indices of every metric; 422 for other designs."""
    from .sensitivity import analyse_runs
    from .study import load_manifest, load_runs
    rec = _get(request, sid)
    if not (rec.out_dir / "runs.parquet").exists():
        raise HTTPException(409, "the study has not written results yet")
    boot = max(10, min(int(boot), 5000))
    key = (boot, (rec.out_dir / "runs.parquet").stat().st_mtime_ns)
    if key in rec.sens_cache:
        return rec.sens_cache[key]

    def work():
        return analyse_runs(load_runs(rec.out_dir), load_manifest(rec.out_dir), n_boot=boot)
    try:
        out = analysis.to_jsonable(await asyncio.to_thread(work))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    rec.sens_cache = {key: out}
    return out


@router.post("/api/methods_text")
async def methods_text_endpoint(request: Request):
    """Body ``{experiment, study?, style?: 'markdown'|'latex'}`` -> ``{style, text}``."""
    from . import methods_text
    from .study_schema import StudySpec
    body = await _json(request)
    if not isinstance(body, dict) or "experiment" not in body:
        raise HTTPException(422, detail=[{"loc": ["experiment"], "msg": "field required"}])
    try:
        exp = Experiment.model_validate(body["experiment"])
        study = StudySpec.model_validate(body["study"]) if body.get("study") else None
    except ValidationError as exc:
        raise HTTPException(422, detail=_issues(exc))
    style = body.get("style") or "markdown"
    if style not in ("markdown", "latex"):
        raise HTTPException(422, detail=[{"loc": ["style"], "msg": "style must be markdown or latex"}])
    text = await asyncio.to_thread(methods_text.methods_text, exp, study, style=style)
    return {"style": style, "text": text}


@router.get("/api/provenance")
def provenance_endpoint():
    return provenance()
