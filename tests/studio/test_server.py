"""Server tests with fake engine callables (no real capture/tracer needed)."""
import io
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from spectraforge.studio.contracts import CaptureResult
from spectraforge.studio.presets import PRESETS, preset
from spectraforge.studio.schema import Experiment
from spectraforge.studio.server import create_app

H, W, B = 12, 16, 5


def fake_result(exp: Experiment, mode="sample", exs=None, scale=1.0) -> CaptureResult:
    rng = np.random.default_rng(1)
    src = exp.source()
    exs = exs or [s.wavelength_nm for s in src.sequence if s.enabled][:3] or [365.0]
    em = {e: 420.0 + 10.0 * np.arange(B) for e in exs}
    yy, xx = np.mgrid[0:H, 0:W]
    adu, rad, el = {}, {}, {}
    for e in exs:
        base = (xx / W + 0.2)[..., None] * np.linspace(1, 2, B)
        rad[e] = (base * 0.5).astype(np.float32)
        el[e] = (base * 1000).astype(np.float32)
        a = 100 + base * 1000 + rng.normal(0, 1, (H, W, B))
        a[0, 0, :] = 65535.0   # saturated pixel
        adu[e] = a.astype(np.float32)
    return CaptureResult(
        experiment_json=exp.model_dump_json(), mode=mode, width=W, height=H, excitations_nm=list(exs),
        emission_nm=em, adu=adu, radiance=rad, electrons_mean=el,
        exposure_ms={e: 50.0 for e in exs}, power_mW={e: 2.0 for e in exs},
        ground_truth={"material": (xx > W // 2).astype(np.int32), "depth_mm": np.full((H, W), 300.0)},
        warnings=["fake warning"], timings_s={"total": 0.1})


class Fakes:
    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.calls = []
        self.explained = []

    def capture(self, exp, *, mode=None, scale=1.0, excitations=None, progress=None, cancel=None, **kw):
        self.calls.append(mode)
        for i in range(5):
            self.gate.wait(5)
            if cancel and cancel():
                return fake_result(exp, mode or "sample")
            progress((i + 1) / 5, f"step {i}")
            time.sleep(0.01)
        return fake_result(exp, mode or "sample")

    def preview(self, exp, excitation_nm=None, scale=None):
        return fake_result(exp, "preview", exs=[excitation_nm or 365.0], scale=scale)

    def explain(self, exp, x, y, *, scale=1.0):
        self.explained.append((x, y, scale))
        return {"components": {"365": {"dye:EGFP": np.ones(B), "elastic": np.zeros(B)}},
                "ground_truth": {"material": "m", "depth_mm": np.float64(300.0), "object": None}}

    def trace(self, exp, scale):
        raise ImportError("no tracer")


@pytest.fixture
def fakes():
    return Fakes()


@pytest.fixture
def client(fakes):
    app = create_app(capture_fn=fakes.capture, preview_fn=fakes.preview, explain_fn=fakes.explain,
                     trace_fn=fakes.trace, dist_dir="/nonexistent")
    with TestClient(app) as c:
        yield c


EXP = preset("generic_macro_bench").model_dump()


def run_capture(client, mode=None, wait=True):
    body = {"experiment": EXP}
    if mode:
        body["mode"] = mode
    jid = client.post("/api/captures", json=body).json()["job_id"]
    if wait:
        for _ in range(300):
            st = client.get(f"/api/jobs/{jid}").json()
            if st["state"] in ("done", "error", "cancelled"):
                return jid, st
            time.sleep(0.02)
        raise AssertionError("job did not finish")
    return jid, None


def test_health_and_presets(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["engine_version"] and h["schema_version"]
    p = client.get("/api/presets").json()
    assert set(p["experiments"]) == set(PRESETS)
    for name, e in p["experiments"].items():
        assert Experiment.model_validate(e).model_dump(mode="json") == e
    assert p["fluorophores"] and p["materials"]


def test_presets_json_in_sync(client):
    from pathlib import Path
    f = Path(__file__).resolve().parents[2] / "apps/studio-web/src/generated/presets.json"
    if f.exists():
        assert json.loads(f.read_text()) == client.get("/api/presets").json()


def test_validate_ok_and_errors(client):
    r = client.post("/api/validate", json=EXP).json()
    assert r["ok"] and r["errors"] == []
    bad = json.loads(json.dumps(EXP))
    bad["objects"][0]["material_id"] = "nonexistent"
    bad["sources"][0]["beam"]["aperture_diameter_mm"] = -1
    r = client.post("/api/validate", json=bad).json()
    assert not r["ok"] and r["errors"]
    assert all(isinstance(e["loc"], list) and isinstance(e["msg"], str) for e in r["errors"])
    assert client.post("/api/validate", json={"junk": 1}).json()["ok"] is False
    assert client.post("/api/validate", json=[1, 2]).json()["ok"] is False
    assert client.post("/api/validate", content=b"not json").status_code == 400


def test_validate_warnings(client):
    e = json.loads(json.dumps(EXP))
    e["sources"][0]["sequence"] = []
    r = client.post("/api/validate", json=e).json()
    assert any(w["code"] == "no_excitation_steps" for w in r["warnings"])
    e = json.loads(json.dumps(EXP))
    cam = e["cameras"][0]
    cam["filter"]["emission_longpass_nm"] = None
    cam["filter"]["band_start_policy"] = "fixed"
    cam["filter"]["band_start_nm"] = 400.0
    e["sources"][0]["sequence"] = [{"wavelength_nm": 450.0, "power_mW": 1, "exposure_ms": 10, "enabled": True}]
    r = client.post("/api/validate", json=e).json()
    assert any(w["code"] == "excitation_in_band" for w in r["warnings"])
    assert all(w["severity"] in ("info", "warn") for w in r["warnings"])


def test_validate_trace_warnings():
    from types import SimpleNamespace as NS
    P = 40
    hit = NS(object_index=np.full((4, P), -1), irradiance_per_mW=np.zeros((4, P)), depth_mm=np.full(P, np.inf),
             glint_per_mW=np.zeros(P), focus_depth_mm=300.0)
    with TestClient(create_app(trace_fn=lambda e, s: hit, dist_dir="/x")) as c:
        r = c.post("/api/validate", json=EXP).json()
    assert any(w["code"] == "camera_sees_no_objects" for w in r["warnings"])
    hit.object_index = np.zeros((4, P), int)
    hit.depth_mm = np.full(P, 5000.0)
    hit.glint_per_mW = np.ones(P)
    with TestClient(create_app(trace_fn=lambda e, s: hit, dist_dir="/x")) as c:
        codes = {w["code"] for w in c.post("/api/validate", json=EXP).json()["warnings"]}
    assert {"beam_misses_objects", "out_of_depth_of_field", "glint_in_view"} <= codes


def test_validate_survives_tracer_crash():
    def bad(e, s):
        raise RuntimeError("boom")
    with TestClient(create_app(trace_fn=bad, dist_dir="/x")) as c:
        r = c.post("/api/validate", json=EXP).json()
    assert r["ok"] and any(w["code"] == "trace_failed" for w in r["warnings"])


def test_preview(client):
    r = client.post("/api/preview", json={"experiment": EXP, "excitation_nm": 405.0, "scale": 0.5})
    assert r.status_code == 200
    m = r.json()
    assert m["mode"] == "preview" and m["excitations_nm"] == [405.0]
    assert client.get(f"/api/captures/{m['id']}").json()["id"] == m["id"]
    assert client.post("/api/preview", json={"experiment": {"x": 1}}).status_code == 422
    assert client.post("/api/preview", json={}).status_code == 422


def test_capture_job_and_meta(client, fakes):
    jid, st = run_capture(client, "dark")
    assert st["state"] == "done" and st["fraction"] == 1.0 and st["capture_id"]
    assert fakes.calls == ["dark"]
    m = client.get(f"/api/captures/{st['capture_id']}").json()
    assert m["mode"] == "dark" and m["width"] == W and m["height"] == H
    key = f"{m['excitations_nm'][0]:g}"
    assert len(m["stats"][key]) == B and set(m["stats"][key][0]) == {"min", "max", "p01", "p99", "mean"}
    assert m["emission_nm"][key][0] == 420.0 and m["exposure_ms"][key] == 50.0 and m["power_mW"][key] == 2.0
    assert m["saturated_fraction"][key] == pytest.approx(1 / (H * W))
    assert sorted(m["ground_truth_layers"]) == ["depth_mm", "material"]
    assert m["warnings"] == ["fake warning"] and m["camera_id"] and m["source_id"]
    assert client.get("/api/captures").json()[0]["id"] == m["id"]
    assert client.get("/api/captures/nope").status_code == 404


def test_websocket_progress(client, fakes):
    fakes.gate.clear()
    jid, _ = run_capture(client, wait=False)
    fakes.gate.set()
    with client.websocket_connect(f"/api/jobs/{jid}") as ws:
        events = []
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] in ("done", "error", "cancelled"):
                break
    assert events[-1]["type"] == "done" and events[-1]["capture_id"]
    prog = [e["fraction"] for e in events if e["type"] == "progress"]
    assert prog and prog == sorted(prog)
    with client.websocket_connect(f"/api/jobs/{jid}") as ws:      # late subscriber gets the history
        assert ws.receive_json()["type"] == "progress"


def test_cancel_and_queue(client, fakes):
    fakes.gate.clear()
    j1, _ = run_capture(client, wait=False)
    j2, _ = run_capture(client, wait=False)       # queued behind j1
    assert client.get(f"/api/jobs/{j2}").json()["state"] == "queued"
    assert client.post(f"/api/jobs/{j1}/cancel").status_code == 200
    fakes.gate.set()
    for j in (j1, j2):
        for _ in range(300):
            if client.get(f"/api/jobs/{j}").json()["state"] in ("done", "cancelled", "error"):
                break
            time.sleep(0.02)
    assert client.get(f"/api/jobs/{j1}").json()["state"] == "cancelled"
    assert client.get(f"/api/jobs/{j2}").json()["state"] == "done"
    assert client.get("/api/captures").json().__len__() == 1       # cancelled capture is not stored
    with client.websocket_connect(f"/api/jobs/{j1}") as ws:
        while True:
            ev = ws.receive_json()
            if ev["type"] != "progress":
                break
    assert ev["type"] == "cancelled"
    assert client.post("/api/jobs/zzz/cancel").status_code == 404


def test_cancel_queued_job_never_runs(client, fakes):
    fakes.gate.clear()
    j1, _ = run_capture(client, wait=False)
    j2, _ = run_capture(client, wait=False)
    client.post(f"/api/jobs/{j2}/cancel")
    fakes.gate.set()
    for _ in range(300):
        if client.get(f"/api/jobs/{j2}").json()["state"] == "cancelled":
            break
        time.sleep(0.02)
    assert client.get(f"/api/jobs/{j2}").json()["state"] == "cancelled"
    assert len(fakes.calls) == 1


def test_job_error():
    def boom(exp, **kw):
        raise RuntimeError("kaboom")
    with TestClient(create_app(capture_fn=boom, dist_dir="/x")) as c:
        _, st = run_capture(c)
        assert st["state"] == "error" and "kaboom" in st["error"]
        jid = c.post("/api/captures", json={"experiment": EXP}).json()["job_id"]
        with c.websocket_connect(f"/api/jobs/{jid}") as ws:
            ev = ws.receive_json()
        assert ev["type"] == "error" and "kaboom" in ev["message"]


def test_invalid_capture_request(client):
    assert client.post("/api/captures", json={"experiment": {"bad": 1}}).status_code == 422
    assert client.post("/api/captures", json={"experiment": EXP, "mode": "wat"}).status_code == 422
    assert client.get("/api/jobs/nope").status_code == 404
    with pytest.raises(Exception):
        with client.websocket_connect("/api/jobs/nope"):
            pass


def test_binary_endpoints(client):
    _, st = run_capture(client)
    cid = st["capture_id"]
    r = client.get(f"/api/captures/{cid}/band", params={"band": 2, "kind": "radiance"})
    assert r.status_code == 200
    assert (r.headers["x-width"], r.headers["x-height"]) == (str(W), str(H))
    assert np.frombuffer(r.content, "<f4").reshape(H, W).shape == (H, W)
    r = client.get(f"/api/captures/{cid}/cube")
    assert r.headers["x-bands"] == str(B)
    cube = np.frombuffer(r.content, "<f4").reshape(H, W, B)
    assert cube[0, 0, 0] == 65535.0
    assert "x-width" in r.headers["access-control-expose-headers"].lower()
    g = np.frombuffer(client.get(f"/api/captures/{cid}/gt/material").content, "<f4").reshape(H, W)
    assert g[0, W - 1] == 1 and g[0, 0] == 0
    assert client.get(f"/api/captures/{cid}/gt/zzz").status_code == 404
    assert client.get(f"/api/captures/{cid}/band", params={"band": 99}).status_code == 404
    assert client.get(f"/api/captures/{cid}/band", params={"ex": 999}).status_code == 404
    assert client.get(f"/api/captures/{cid}/band", params={"kind": "x"}).status_code == 422
    ex = client.get(f"/api/captures/{cid}").json()["excitations_nm"][0]
    assert client.get(f"/api/captures/{cid}/cube", params={"ex": ex, "kind": "electrons"}).status_code == 200


def test_png_modes(client):
    _, st = run_capture(client)
    cid = st["capture_id"]
    for q in ({"mode": "true_color"}, {"mode": "true_color", "gain": 2.0}, {"mode": "false_color"},
              {"mode": "false_color", "bands": "4,2,0"}, {"mode": "band", "band": 3}):
        r = client.get(f"/api/captures/{cid}/image.png", params=q)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", q
        im = Image.open(io.BytesIO(r.content))
        assert im.size == (W, H) and im.mode == "RGB"
    assert client.get(f"/api/captures/{cid}/image.png", params={"mode": "zzz"}).status_code == 422
    assert client.get(f"/api/captures/{cid}/image.png", params={"mode": "band", "band": 50}).status_code == 404
    assert client.get(f"/api/captures/{cid}/image.png", params={"mode": "false_color", "bands": "1,2"}).status_code == 422


def test_pixel_explain(client, fakes):
    cid = client.post("/api/preview", json={"experiment": EXP, "scale": 0.5}).json()["id"]
    p = client.get(f"/api/captures/{cid}/pixel", params={"x": 3, "y": 4})
    assert p.status_code == 200
    d = p.json()
    assert d["x"] == 3 and d["y"] == 4 and len(next(iter(d["adu"].values()))) == B
    assert d["components"]["365"]["dye:EGFP"] == [1.0] * B
    assert d["ground_truth"]["depth_mm"] == 300.0
    assert fakes.explained[-1] == (3, 4, 0.5)
    assert client.get(f"/api/captures/{cid}/pixel", params={"x": W, "y": 0}).status_code == 404


def test_lru_store():
    fk = Fakes()
    app = create_app(capture_fn=fk.capture, preview_fn=fk.preview, explain_fn=fk.explain,
                     trace_fn=fk.trace, dist_dir="/x", store_size=3)
    with TestClient(app) as c:
        ids = [c.post("/api/preview", json={"experiment": EXP}).json()["id"] for _ in range(5)]
        assert [m["id"] for m in c.get("/api/captures").json()] == ids[:1:-1]   # newest first, last three
        assert c.get(f"/api/captures/{ids[0]}").status_code == 404


def test_cors_and_spa(tmp_path, fakes):
    (tmp_path / "index.html").write_text("<html>app</html>")
    (tmp_path / "a.js").write_text("1")
    with TestClient(create_app(trace_fn=fakes.trace, dist_dir=tmp_path, warmup_start_fn=lambda: None,
                               warmup_status_fn=lambda: {"state": "idle"})) as c:
        assert c.get("/").text == "<html>app</html>"
        assert c.get("/a.js").text == "1"
        assert c.get("/some/route").text == "<html>app</html>"
        assert c.get("/api/nothing").status_code == 404
        assert c.get("/api/health").json()["ok"]
        r = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_health_exposes_warmup_and_starts_it_non_blocking(fakes):
    calls = []
    state = {"state": "running", "fraction": 0.4, "message": "baking", "result": None}
    app = create_app(capture_fn=fakes.capture, preview_fn=fakes.preview, explain_fn=fakes.explain,
                     trace_fn=fakes.trace, dist_dir="/nonexistent",
                     warmup_start_fn=lambda: calls.append("start"), warmup_status_fn=lambda: dict(state))
    with TestClient(app) as c:
        assert calls == ["start"]                      # started once, from the lifespan hook
        h = c.get("/api/health").json()
        assert h["ok"] and h["warmup"]["state"] == "running" and h["warmup"]["fraction"] == 0.4
        # preview still works while the warm-up is "running"
        assert c.post("/api/preview", json={"experiment": EXP}).status_code == 200
        state.update(state="done", fraction=1.0)
        assert c.get("/api/health").json()["warmup"]["state"] == "done"


def test_health_without_warmup_hooks_has_no_warmup_key(client):
    assert "warmup" not in client.get("/api/health").json()
