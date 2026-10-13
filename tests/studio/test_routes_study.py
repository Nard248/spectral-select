import json
import time

import pytest
from fastapi.testclient import TestClient

from spectraforge.studio.server import create_app

from .fakes_study import COVER, FLUOR, base_experiment, fake_capture

BASE = "http://127.0.0.1:8765"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRAFORGE_STUDY_DIR", str(tmp_path))
    app = create_app(dist_dir=tmp_path / "nodist")
    app.state.study_capture_fn = fake_capture
    with TestClient(app, base_url=BASE) as c:
        yield c


def body(kind="saltelli", n=8):
    return {"name": "r", "base_experiment": base_experiment().model_dump(mode="json"), "replicates": 1,
            "parameters": [{"pointer": COVER, "label": "cover", "low": 0.1, "high": 1.0},
                           {"pointer": FLUOR, "label": "fluor", "low": 0.2, "high": 1.0}],
            "design": {"kind": kind, "n": n, "seed": 1},
            "metrics": [{"name": "signal", "kind": "band_sum"}]}


def wait(client, sid, timeout=30):
    t0 = time.time()
    while True:
        s = client.get(f"/api/studies/{sid}").json()
        if s["state"] in ("done", "error", "cancelled"):
            return s
        assert time.time() - t0 < timeout, s
        time.sleep(0.05)


def test_study_lifecycle(client):
    r = client.post("/api/studies", json=body())
    assert r.status_code == 200, r.text
    sid = r.json()["study_id"]
    assert r.json()["n_runs"] == 32 and r.json()["n_points"] == 32
    s = wait(client, sid)
    assert s["state"] == "done" and s["manifest"]["n_completed"] == 32
    runs = client.get(f"/api/studies/{sid}/runs?limit=10").json()
    assert runs["total"] == 32 and len(runs["rows"]) == 10 and "m:signal" in runs["columns"]
    json.dumps(runs)
    sens = client.get(f"/api/studies/{sid}/sensitivity?boot=20").json()
    assert sens["metrics"]["signal"]["method"] == "sobol"
    assert set(sens["metrics"]["signal"]["indices"]) == {"cover", "fluor"}
    assert client.get(f"/api/studies/{sid}/sensitivity?boot=20").json() == sens           # cached
    assert any(x["study_id"] == sid for x in client.get("/api/studies").json())


def test_websocket_events(client):
    sid = client.post("/api/studies", json=body("lhs", 4)).json()["study_id"]
    with client.websocket_connect(f"ws://127.0.0.1:8765/api/studies/{sid}/events") as ws:
        evs = []
        while True:
            ev = ws.receive_json()
            evs.append(ev)
            if ev["type"] in ("done", "error", "cancelled"):
                break
    assert evs[-1]["type"] == "done" and any(e["type"] == "progress" for e in evs)


def test_validation_errors(client):
    b = body()
    b["parameters"][0]["pointer"] = "/objects/0/shape/layers/2/thickness/nope"
    assert client.post("/api/studies", json=b).status_code == 422
    b = body()
    b["design"]["kind"] = "bogus"
    assert client.post("/api/studies", json=b).status_code == 422
    assert client.get("/api/studies/nope").status_code == 404
    sid = client.post("/api/studies", json=body("lhs", 3)).json()["study_id"]
    wait(client, sid)
    assert client.get(f"/api/studies/{sid}/sensitivity").status_code == 422             # lhs: no indices


def test_cancel(client):
    sid = client.post("/api/studies", json=body("random", 40)).json()["study_id"]
    client.post(f"/api/studies/{sid}/cancel")
    s = wait(client, sid)
    assert s["state"] in ("cancelled", "done")


def test_methods_text_and_provenance(client):
    exp = base_experiment().model_dump(mode="json")
    r = client.post("/api/methods_text", json={"experiment": exp, "style": "latex", "study": body()})
    assert r.status_code == 200 and r.json()["style"] == "latex" and "Study design" in r.json()["text"]
    assert client.post("/api/methods_text", json={"experiment": exp, "style": "x"}).status_code == 422
    assert client.post("/api/methods_text", json={}).status_code == 422
    p = client.get("/api/provenance").json()
    assert "git" in p and "numpy" in p


def test_security_rules_still_apply(client):
    r = client.post("/api/studies", content=json.dumps(body()), headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    r = client.post("/api/studies", json=body(), headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
