import threading
import time

import pytest

from spectraforge.studio.jobs import Cancelled, JobManager
from spectraforge.studio.presets import preset
from spectraforge.studio.server import CaptureStore
from spectraforge.studio.sweeps import SweepError, SweepManager, apply_value, parse_pointer, set_pointer

EXP = preset("generic_macro_bench")
PATH = "/acquisition/preview_scale"          # a simple numeric field with validation (gt 0)


def wait(sw, timeout=5.0):
    t0 = time.time()
    while sw.status()["state"] in ("queued", "running"):
        assert time.time() - t0 < timeout, sw.status()
        time.sleep(0.01)
    return sw.status()


@pytest.fixture
def jobs():
    j = JobManager()
    yield j
    j.shutdown()


def make(jobs, calls=None, gate=None, fail_at=None):
    def evaluate(exp, mode, progress, cancel, store):
        idx = len(calls) if calls is not None else 0
        if calls is not None:
            calls.append((exp.acquisition.preview_scale, mode))
        if gate is not None:
            gate.wait(5)
        progress(0.5, "half")
        if fail_at is not None and idx == fail_at:
            raise RuntimeError("engine exploded")
        if cancel():
            raise Cancelled()
        cid = f"cap{idx}"
        store.put(type("S", (), {"meta": {"id": cid}, "result": None})())
        return cid
    return SweepManager(jobs, evaluate, lambda n: CaptureStore(n))


def test_pointer_helpers():
    assert parse_pointer("/a~1b/c~0d/0") == ["a/b", "c~d", "0"]
    doc = {"a": [{"b": 1}, {"b": 2}]}
    set_pointer(doc, "/a/1/b", 9)
    assert doc["a"][1]["b"] == 9
    for bad in ("a/b", "/a/5/b", "/a/x", "/a/0/zzz", "/a/0/b/c"):
        with pytest.raises(SweepError):
            set_pointer(doc, bad, 0)
    base = EXP.model_dump(mode="json")
    assert apply_value(base, PATH, 0.4).acquisition.preview_scale == 0.4
    assert base["acquisition"]["preview_scale"] == EXP.acquisition.preview_scale        # deep copy: base untouched
    with pytest.raises(SweepError, match="preview_scale"):
        apply_value(base, PATH, -1)


def test_sweep_end_to_end(jobs):
    calls = []
    m = make(jobs, calls)
    sw = m.start(EXP, {"path": PATH, "values": [0.2, 0.4, 0.6], "label": "scale"}, "preview")
    st = wait(sw)
    assert st["state"] == "done" and st["fraction"] == 1.0 and st["capture_ids"] == ["cap0", "cap1", "cap2"]
    assert st["parameter"]["label"] == "scale"
    assert calls == [(0.2, "preview"), (0.4, "preview"), (0.6, "preview")]
    ev = sw.events_since(0)
    pts = [e for e in ev if e["type"] == "point"]
    assert [(e["index"], e["value"], e["capture_id"]) for e in pts] == [(0, 0.2, "cap0"), (1, 0.4, "cap1"), (2, 0.6, "cap2")]
    assert ev[-1] == {"type": "done", "sweep_id": sw.id}
    fr = [e["fraction"] for e in ev if e["type"] == "progress"]
    assert fr == sorted(fr) and all(0 <= f < 1 for f in fr)
    assert all("index" in e for e in ev if e["type"] == "progress")
    assert m.find_capture("cap1") is not None and m.find_capture("zzz") is None      # pinned, not LRU-evicted
    assert len(sw.events_since(2)) == len(ev) - 2                                    # history replay from any index


def test_invalid_values_rejected_up_front(jobs):
    calls = []
    m = make(jobs, calls)
    with pytest.raises(SweepError, match=r"values\[1\]"):
        m.start(EXP, {"path": PATH, "values": [0.3, -5, 0.5]})
    assert calls == []                                                                  # nothing ran
    for params in ({"path": PATH, "values": []}, {"path": "/nope/x", "values": [1]}, {"path": "no-slash", "values": [1]},
                   {"path": PATH, "values": [[1]]}, {"path": PATH, "values": list(range(1, 66))}):
        with pytest.raises(SweepError):
            m.start(EXP, params)
    with pytest.raises(SweepError, match="mode"):
        m.start(EXP, {"path": PATH, "values": [0.5]}, "bogus")


def test_engine_failure_stops_sweep(jobs):
    m = make(jobs, [], fail_at=1)
    sw = m.start(EXP, {"path": PATH, "values": [0.2, 0.3, 0.4]})
    st = wait(sw)
    assert st["state"] == "error" and "engine exploded" in st["error"] and "2/3" in st["error"]
    assert st["capture_ids"] == ["cap0", None, None]
    assert sw.events_since(0)[-1]["type"] == "error"


def test_cancel_running_and_queued(jobs):
    gate = threading.Event()
    calls = []
    m = make(jobs, calls, gate=gate)
    first = m.start(EXP, {"path": PATH, "values": [0.2, 0.3, 0.4, 0.5]})
    second = m.start(EXP, {"path": PATH, "values": [0.2]})                  # queued behind the first (one worker)
    time.sleep(0.1)
    assert first.status()["state"] == "running" and second.status()["state"] == "queued"
    m.cancel(second.id)
    assert second.status()["state"] == "cancelled" and second.events_since(0)[-1] == {"type": "cancelled"}
    m.cancel(first.id)
    gate.set()
    st = wait(first)
    assert st["state"] == "cancelled" and len(calls) < 4
    assert first.events_since(0)[-1] == {"type": "cancelled"}
    # the worker is free again and ignored the cancelled queued sweep
    third = m.start(EXP, {"path": PATH, "values": [0.5]})
    assert wait(third)["state"] == "done"


def test_old_sweeps_are_trimmed(jobs):
    m = make(jobs, [])
    m.keep = 1
    a = m.start(EXP, {"path": PATH, "values": [0.3]})
    wait(a)
    b = m.start(EXP, {"path": PATH, "values": [0.4]})
    wait(b)
    c = m.start(EXP, {"path": PATH, "values": [0.5]})
    wait(c)
    assert m.find_capture("cap2") is not None          # newest sweep retained ...
    assert m.find_capture("cap0") is None              # ... the oldest one's captures dropped (trim runs at start)
    assert m.get(a.id) is not None and m.get(a.id).status()["state"] == "done"   # status record kept
