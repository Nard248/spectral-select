"""Parameter sweeps: re-run a preview / capture for a list of values of one experiment parameter.

``SweepManager.start`` deep-copies the experiment for every value, sets the value at an RFC 6901 JSON pointer
(e.g. ``/objects/0/shape/layers/2/thickness/thickness_mm``), re-validates with ``Experiment.model_validate`` and
runs the injected ``evaluate`` function.  Sweeps run on the *same* single worker as ordinary capture jobs
(``JobManager``), so two heavy Monte Carlo jobs never run at once; a sweep is one queued job.

Failure policy (documented decision)
    * All values are validated *before* anything is queued: a bad value (pointer does not exist, or the
      experiment no longer validates) raises :class:`SweepError` naming the offending indices and nothing runs.
    * A failure while running (engine error) emits an ``error`` event and **stops** the sweep; the points
      completed so far stay available (``capture_ids`` has ``None`` for the rest).

Captures of a sweep live in a store of their own, sized to the sweep, so the ordinary capture LRU cannot evict
them; the newest ``keep`` sweeps are retained (older sweeps' captures are dropped).
"""
from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from .jobs import TERMINAL, Cancelled
from .schema import Experiment

MAX_SWEEP_POINTS = 64


class SweepError(ValueError):
    """Invalid sweep request (reported to the client as HTTP 422)."""


# --------------------------------------------------------------------------------------------
# RFC 6901 JSON pointer
# --------------------------------------------------------------------------------------------
def parse_pointer(path: str) -> list[str]:
    if not isinstance(path, str) or not path.startswith("/"):
        raise SweepError(f"parameter path must be a JSON pointer starting with '/', got {path!r}")
    return [t.replace("~1", "/").replace("~0", "~") for t in path[1:].split("/")]


def set_pointer(doc: Any, path: str, value: Any) -> None:
    """Set an EXISTING location of ``doc`` (dict / list tree) in place; never creates keys."""
    tokens = parse_pointer(path)
    cur = doc
    for i, t in enumerate(tokens):
        last = i == len(tokens) - 1
        if isinstance(cur, dict):
            if t not in cur:
                raise SweepError(f"path {path!r}: no key '{t}' (have {sorted(cur)[:12]})")
            if last:
                cur[t] = value
            else:
                cur = cur[t]
        elif isinstance(cur, list):
            if not t.isdigit() or int(t) >= len(cur):
                raise SweepError(f"path {path!r}: bad list index '{t}' (length {len(cur)})")
            if last:
                cur[int(t)] = value
            else:
                cur = cur[int(t)]
        else:
            raise SweepError(f"path {path!r}: cannot descend into a scalar at '{t}'")


def apply_value(base: dict, path: str, value: Any) -> Experiment:
    """Deep-copy ``base`` (a validated experiment as a JSON dict), set the value, re-validate."""
    doc = copy.deepcopy(base)
    set_pointer(doc, path, value)
    try:
        return Experiment.model_validate(doc)
    except ValidationError as exc:
        e = exc.errors()[0]
        raise SweepError(f"{'/'.join(str(p) for p in e['loc'])}: {e['msg']}") from exc


# --------------------------------------------------------------------------------------------
# Sweep records
# --------------------------------------------------------------------------------------------
@dataclass
class Sweep:
    id: str
    parameter: dict
    mode: str
    n: int
    store: Any
    state: str = "queued"
    fraction: float = 0.0
    capture_ids: list = field(default_factory=list)
    error: str | None = None
    events: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    job_id: str | None = None

    def emit(self, event: dict) -> None:
        with self.lock:
            self.events.append(event)

    def status(self) -> dict:
        with self.lock:
            return {"sweep_id": self.id, "state": self.state, "fraction": self.fraction,
                    "parameter": self.parameter, "capture_ids": list(self.capture_ids), "error": self.error}

    def events_since(self, index: int) -> list[dict]:
        with self.lock:
            return list(self.events[index:])

    def _finish(self, state: str, event: dict, error: str | None = None) -> None:
        with self.lock:
            if self.state in TERMINAL:
                return
            self.state, self.error = state, error
            if state == "done":
                self.fraction = 1.0
            self.events.append(event)


EvaluateFn = Callable[..., str]   # (experiment, mode, progress, cancel, store) -> capture_id


class SweepManager:
    """``evaluate(exp, mode, progress, cancel, store) -> capture_id`` runs one point and stores the capture in
    ``store`` (progress(fraction 0..1, message)); raise ``Cancelled`` (or return after ``cancel()``) to abort.
    ``store_factory(capacity)`` creates a capture store; ``jobs`` is the shared ``JobManager``."""

    def __init__(self, jobs, evaluate: EvaluateFn, store_factory: Callable[[int], Any],
                 max_points: int = MAX_SWEEP_POINTS, keep: int = 4, max_history: int = 50):
        self.jobs, self.evaluate, self.store_factory = jobs, evaluate, store_factory
        self.max_points, self.keep, self.max_history = max_points, keep, max_history
        self._sweeps: dict[str, Sweep] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- API
    def start(self, experiment: Experiment, parameter: dict, mode: str = "preview") -> Sweep:
        if mode not in ("preview", "sample"):
            raise SweepError(f"mode must be 'preview' or 'sample', got {mode!r}")
        path, values = parameter.get("path"), parameter.get("values")
        if not isinstance(values, list) or not values:
            raise SweepError("parameter.values must be a non-empty list")
        if len(values) > self.max_points:
            raise SweepError(f"a sweep may have at most {self.max_points} values (got {len(values)})")
        for i, v in enumerate(values):
            if isinstance(v, bool) or isinstance(v, (int, float, str)):
                if isinstance(v, float) and v != v:
                    raise SweepError(f"values[{i}] is NaN")
            else:
                raise SweepError(f"values[{i}] must be a number, string or boolean")
        base = experiment.model_dump(mode="json")
        bad = []
        for i, v in enumerate(values):
            try:
                apply_value(base, path, v)
            except SweepError as exc:
                bad.append(f"values[{i}]={v!r}: {exc}")
                if len(bad) >= 5:
                    break
        if bad:
            raise SweepError("invalid sweep value(s) - nothing was run. " + " | ".join(bad))

        sweep = Sweep(id=uuid.uuid4().hex[:12], parameter={"path": path, "values": list(values),
                                                             **({"label": parameter["label"]} if parameter.get("label") else {})},
                      mode=mode, n=len(values), store=self.store_factory(len(values)),
                      capture_ids=[None] * len(values))
        with self._lock:
            self._sweeps[sweep.id] = sweep
            self._order.append(sweep.id)
            self._trim()
        sweep.job_id = self.jobs.submit(lambda job: self._run(sweep, base, job)).id
        return sweep

    def get(self, sweep_id: str) -> Sweep | None:
        with self._lock:
            return self._sweeps.get(sweep_id)

    def cancel(self, sweep_id: str) -> Sweep | None:
        sw = self.get(sweep_id)
        if sw is None:
            return None
        if sw.job_id:
            self.jobs.cancel(sw.job_id)                       # cooperative flag, read by the running loop
        with sw.lock:
            queued = sw.state == "queued"
        if queued:                                            # never started: finish it right away
            sw._finish("cancelled", {"type": "cancelled"})
        return sw

    def find_capture(self, cid: str):
        with self._lock:
            sweeps = list(self._sweeps.values())
        for sw in reversed(sweeps):
            item = sw.store.get(cid)
            if item is not None:
                return item
        return None

    def list_meta(self) -> list[dict]:
        with self._lock:
            sweeps = list(self._sweeps.values())
        return [m for sw in reversed(sweeps) for m in sw.store.list_meta()]

    # ---------------------------------------------------------------- worker
    def _trim(self) -> None:
        """Drop the oldest finished sweeps beyond ``keep`` (captures) / ``max_history`` (records)."""
        finished = [sid for sid in self._order if self._sweeps[sid].state in TERMINAL]
        while len(self._order) > self.max_history and finished:
            sid = finished.pop(0)
            self._order.remove(sid)
            self._sweeps.pop(sid, None)
        with_store = [sid for sid in self._order if sid in self._sweeps and self._sweeps[sid].state in TERMINAL]
        for sid in with_store[:-self.keep] if self.keep else with_store:
            sw = self._sweeps[sid]
            sw.store = self.store_factory(1)

    def _run(self, sw: Sweep, base: dict, job) -> str:
        with sw.lock:
            if sw.state in TERMINAL:
                return sw.id
            sw.state = "running"
        values = sw.parameter["values"]

        def cancelled() -> bool:
            return job.cancelled()

        try:
            for i, v in enumerate(values):
                if cancelled():
                    raise Cancelled()
                exp = apply_value(base, sw.parameter["path"], v)

                def progress(frac: float, msg: str = "", _i=i) -> None:
                    f = (_i + min(max(float(frac), 0.0), 1.0)) / sw.n
                    with sw.lock:
                        sw.fraction = max(sw.fraction, min(f, 0.999))
                        ev = {"type": "progress", "fraction": sw.fraction,
                              "message": f"[{_i + 1}/{sw.n}] {msg}".rstrip(), "index": _i}
                        sw.events.append(ev)
                    job.progress(sw.fraction, ev["message"])

                cid = self.evaluate(exp, sw.mode, progress, cancelled, sw.store)
                if cancelled():
                    raise Cancelled()
                with sw.lock:
                    sw.capture_ids[i] = cid
                    sw.fraction = max(sw.fraction, (i + 1) / sw.n * 0.999)
                sw.emit({"type": "point", "index": i, "value": v, "capture_id": cid})
        except Cancelled:
            sw._finish("cancelled", {"type": "cancelled"})
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the client
            if cancelled():
                sw._finish("cancelled", {"type": "cancelled"})
                raise Cancelled() from exc
            msg = f"point {i + 1}/{sw.n} (value {v!r}): {type(exc).__name__}: {exc}"
            sw._finish("error", {"type": "error", "message": msg}, error=msg)
            return sw.id
        sw._finish("done", {"type": "done", "sweep_id": sw.id})
        return sw.id
