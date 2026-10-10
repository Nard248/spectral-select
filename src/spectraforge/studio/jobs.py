"""Single-worker job manager for captures (captures are CPU heavy: one at a time, the rest queue)."""
from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

TERMINAL = ("done", "error", "cancelled")


class Cancelled(Exception):
    """Raise from a job function to signal cooperative cancellation."""


@dataclass
class Job:
    id: str
    fn: Callable[["Job"], str]
    state: str = "queued"
    fraction: float = 0.0
    message: str = "queued"
    capture_id: str | None = None
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def cancelled(self) -> bool:
        return self.cancel_flag.is_set()

    def progress(self, fraction: float, message: str = "") -> None:
        with self.lock:
            if self.state in TERMINAL:
                return
            fraction = float(min(max(fraction, self.fraction, 0.0), 1.0))   # monotonic
            if self.events and self.events[-1]["type"] == "progress" and \
                    self.events[-1]["fraction"] == fraction and self.events[-1]["message"] == message:
                return
            self.fraction, self.message = fraction, message
            self.events.append({"type": "progress", "fraction": fraction, "message": message})

    def _finish(self, state: str, event: dict, **kw) -> None:
        with self.lock:
            self.state = state
            for k, v in kw.items():
                setattr(self, k, v)
            self.events.append(event)

    def status(self) -> dict:
        with self.lock:
            return {"job_id": self.id, "state": self.state, "fraction": self.fraction,
                    "message": self.message, "capture_id": self.capture_id, "error": self.error}

    def events_since(self, index: int) -> list[dict]:
        with self.lock:
            return list(self.events[index:])


class JobManager:
    def __init__(self, max_history: int = 50):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: "queue.Queue[Job | None]" = queue.Queue()
        self._max_history = max_history
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="studio-capture-worker", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[Job], str]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], fn=fn)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._max_history:
                old = self._order[0]
                if self._jobs[old].state not in TERMINAL:
                    break
                self._order.pop(0)
                self._jobs.pop(old, None)
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        job.cancel_flag.set()
        if job.state == "queued":  # never started: the worker will mark it cancelled when it dequeues it
            pass
        return job

    def shutdown(self) -> None:
        self._queue.put(None)

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            if job.cancelled():
                job._finish("cancelled", {"type": "cancelled"}, message="cancelled")
                continue
            with job.lock:
                job.state, job.message = "running", "starting"
            try:
                capture_id = job.fn(job)
                if job.cancelled():
                    job._finish("cancelled", {"type": "cancelled"}, message="cancelled")
                else:
                    job._finish("done", {"type": "done", "capture_id": capture_id},
                                capture_id=capture_id, fraction=1.0, message="done")
            except Cancelled:
                job._finish("cancelled", {"type": "cancelled"}, message="cancelled")
            except Exception as exc:  # noqa: BLE001 - reported to the client, never swallowed
                if job.cancelled():
                    job._finish("cancelled", {"type": "cancelled"}, message="cancelled")
                else:
                    msg = f"{type(exc).__name__}: {exc}"
                    job._finish("error", {"type": "error", "message": msg}, error=msg, message=msg)
