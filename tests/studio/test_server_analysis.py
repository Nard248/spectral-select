"""v1.1 endpoints: analysis (eem / roi / line / response), volumes, sweeps — with fake engine callables."""
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from spectraforge.studio.presets import preset
from spectraforge.studio.server import create_app

from .fakes_v11 import B, H, OFFSET, W, fake_block_result

BASE = "http://127.0.0.1:8765"
EXP = preset("generic_macro_bench").model_dump()
JSON = {"Content-Type": "application/json"}


class Fakes:
    def __init__(self):
        self.explained = []
        self.irradiance = False

    def capture(self, exp, *, mode=None, scale=1.0, excitations=None, progress=None, cancel=None, **kw):
        if progress:
            progress(0.5, "half")
        r = fake_block_result(mode or "sample", scale_adu=0.0 if mode == "dark" else (10.0 if mode == "white_reference" else 1.0),
                              power_mW=1.0 if mode == "dark" else 2.0)
        if self.irradiance:
            for e in r.excitations_nm:
                r.ground_truth[f"irradiance@{e:g}"] = np.full((H, W), 4.0)
        return r

    def preview(self, exp, excitation_nm=None, scale=None):
        r = fake_block_result("sample")
        t = exp.acquisition.preview_scale                      # make sweep points distinguishable
        r.adu = {e: (a * np.float32(t)) for e, a in r.adu.items()}
        return r

    def explain(self, exp, x, y, *, scale=1.0):
        self.explained.append((x, y))
        return {"components": {"365": {"dye:A@tissue": np.full(B, float(x)), "elastic": np.ones(B)}}}

    def trace(self, exp, scale):
        raise ImportError


@pytest.fixture
def fk():
    return Fakes()


@pytest.fixture
def client(fk):
    app = create_app(capture_fn=fk.capture, preview_fn=fk.preview, explain_fn=fk.explain, trace_fn=fk.trace,
                     dist_dir="/nonexistent")
    with TestClient(app, base_url=BASE) as c:
        yield c


def cap(client, mode="sample"):
    jid = client.post("/api/captures", json={"experiment": EXP, "mode": mode}).json()["job_id"]
    for _ in range(300):
        st = client.get(f"/api/jobs/{jid}").json()
        if st["state"] in ("done", "error", "cancelled"):
            assert st["state"] == "done", st
            return st["capture_id"]
        time.sleep(0.02)
    raise AssertionError("timeout")


DISK = {"shape": "disk", "x": 12, "y": 8, "radius_px": 3}


def test_meta_has_volumes_and_blocks(client):
    cid = cap(client)
    m = client.get(f"/api/captures/{cid}").json()
    assert m["blocks"] == [{"object_id": "block1", "layer_names": ["substrate", "tissue", "cover"], "footprint_mm": [4.0, 3.0]}]
    assert {v["kind"] for v in m["volumes"]} == {"fluence", "absorbed", "source"}
    assert "layer_thickness:cover" in m["ground_truth_layers"]
    assert client.get(f"/api/captures/{cid}/volumes").json() == m["volumes"]


def test_eem_endpoint(client):
    cid = cap(client)
    r = client.post(f"/api/captures/{cid}/eem", json={"roi": DISK, "norm": {"mode": "exposure_power"}})
    assert r.status_code == 200
    d = r.json()
    assert d["n_pixels"] == 29 and d["unit"].startswith("ADU/") and d["norm"]["mode"] == "exposure_power"
    assert set(d["value"]) == {"365", "405"} and len(d["value"]["365"]) == B
    assert d["grid_value"][0][-1] is None                      # NaN beyond the 365 nm band span -> null
    d2 = client.post(f"/api/captures/{cid}/eem", json={"roi": DISK}).json()
    assert d2["norm"]["mode"] == "adu" and d2["unit"] == "ADU"
    assert client.post(f"/api/captures/{cid}/eem", json={"roi": DISK, "stat": "median"}).status_code == 200


def test_eem_dark_white_norms(client):
    cid, dark, white = cap(client), cap(client, "dark"), cap(client, "white_reference")
    r = client.post(f"/api/captures/{cid}/eem", json={"roi": DISK, "norm": {
        "mode": "white_normalized", "dark_capture_id": dark, "white_capture_id": white}})
    assert r.status_code == 200 and r.json()["unit"] == "ratio"
    s = client.post(f"/api/captures/{cid}/eem", json={"roi": DISK, "norm": {"mode": "dark_subtracted", "dark_capture_id": dark}})
    assert s.status_code == 200


def test_eem_irradiance_normalized(client, fk):
    cid = cap(client)
    r = client.post(f"/api/captures/{cid}/eem", json={"roi": DISK, "norm": {"mode": "irradiance_normalized"}})
    assert r.status_code == 422 and "irradiance@365" in r.json()["detail"]     # no irradiance map in this capture
    fk.irradiance = True
    cid = cap(client)
    r = client.post(f"/api/captures/{cid}/eem", json={"roi": DISK, "norm": {"mode": "irradiance_normalized"}})
    assert r.status_code == 200
    d = r.json()
    assert d["norm"]["mode"] == "irradiance_normalized" and "mW/mm" in d["unit"]
    raw = client.post(f"/api/captures/{cid}/eem", json={"roi": DISK, "norm": {"mode": "adu"}}).json()
    # (mean ADU - offset) / (0.05 s * 4 mW/mm^2)
    assert d["value"]["365"][0] == pytest.approx((raw["value"]["365"][0] - OFFSET) / (0.05 * 4.0), rel=1e-4)
    assert client.post(f"/api/captures/{cid}/eem", json={"roi": DISK, "norm": {"mode": "nope"}}).status_code == 422


def test_errors_are_4xx_with_messages(client):
    cid = cap(client)
    url = f"/api/captures/{cid}/eem"
    r = client.post(url, json={"roi": {"shape": "disk", "x": 1, "y": 1, "radius_px": -2}})
    assert r.status_code == 422 and "invalid ROI" in r.json()["detail"]
    r = client.post(url, json={"roi": {"shape": "disk", "x": 900, "y": 1, "radius_px": 2}})
    assert r.status_code == 422 and "no pixels" in r.json()["detail"]
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json=[1]).status_code == 422
    assert client.post(url, content=b"{nope", headers=JSON).status_code == 400
    r = client.post(url, json={"roi": DISK, "norm": {"mode": "dark_subtracted"}})
    assert r.status_code == 422 and "dark" in r.json()["detail"]
    r = client.post(url, json={"roi": DISK, "norm": {"mode": "dark_subtracted", "dark_capture_id": "nope"}})
    assert r.status_code == 404 and "dark" in r.json()["detail"]
    r = client.post(url, json={"roi": DISK, "stat": "mode"})
    assert r.status_code == 422
    assert client.post("/api/captures/nope/eem", json={"roi": DISK}).status_code == 404
    # mismatched dark: another size
    bad = fake_block_result("dark")
    bad.width, bad.adu = W - 4, {e: a[:, :W - 4] for e, a in bad.adu.items()}
    app = client.app
    from spectraforge.studio.server import StoredCapture
    app.state.store.put(StoredCapture(meta={"id": "baddark"}, result=bad, experiment=preset("generic_macro_bench")))
    r = client.post(url, json={"roi": DISK, "norm": {"mode": "dark_subtracted", "dark_capture_id": "baddark"}})
    assert r.status_code == 422 and "px" in r.json()["detail"]


def test_roi_endpoint_with_components(client, fk):
    cid = cap(client)
    r = client.post(f"/api/captures/{cid}/roi", json={"roi": {"shape": "rect", "x0": 6, "y0": 0, "x1": 20, "y1": 9}})
    d = r.json()
    assert r.status_code == 200 and d["components"] is None
    assert d["ground_truth"]["layer_thickness:cover"]["min"] == pytest.approx(2.0 * 6 / (W - 1))
    assert "label" not in d["ground_truth"]
    r = client.post(f"/api/captures/{cid}/roi", json={"roi": DISK, "components": True})
    comps = r.json()["components"]["365"]
    assert set(comps) == {"dye:A@tissue", "elastic"} and comps["elastic"] == [1.0] * B
    n = len(fk.explained)
    client.post(f"/api/captures/{cid}/roi", json={"roi": DISK, "components": True})
    assert len(fk.explained) == n                              # per-capture pixel cache


def test_line_and_response_endpoints(client):
    cid = cap(client)
    r = client.post(f"/api/captures/{cid}/line", json={"p0": [2, 5], "p1": [22, 5], "width_px": 3,
                                                       "ground_truth": ["layer_thickness:cover"]})
    d = r.json()
    assert r.status_code == 200 and len(d["distance_px"]) == 21 and d["distance_mm"][-1] > 0
    assert set(d["value"]) == {"365", "405"} and len(d["value"]["365"]) == 21 and len(d["value"]["365"][0]) == B
    assert d["ground_truth"]["layer_thickness:cover"][0] is None and d["unit"] == "ADU"
    assert client.post(f"/api/captures/{cid}/line", json={"p0": [2, 5]}).status_code == 422
    assert client.post(f"/api/captures/{cid}/line", json={"p0": [2, 5], "p1": [9, 9], "ground_truth": ["zz"]}).status_code == 422
    r = client.post(f"/api/captures/{cid}/response", json={
        "x_ground_truth": "layer_thickness:cover", "metric": {"kind": "band", "ex": 365, "band_nm": 460}, "bins": 8,
        "norm": {"mode": "exposure_power"}})
    d = r.json()
    assert r.status_code == 200 and len(d["bin_centres"]) == 8 and sum(d["count"]) == H * (W - 5)
    assert d["metric"]["kind"] == "band" and len(d["scatter_x"]) <= 4000
    r = client.post(f"/api/captures/{cid}/response", json={"x_ground_truth": "nope", "metric": {"kind": "peak_wavelength", "ex": 365}})
    assert r.status_code == 422 and "nope" in r.json()["detail"]
    r = client.post(f"/api/captures/{cid}/response", json={"x_ground_truth": "depth_mm", "metric": {"kind": "x"}})
    assert r.status_code == 422 and "invalid metric" in r.json()["detail"]


def test_volume_endpoint(client):
    cid = cap(client)
    r = client.get(f"/api/captures/{cid}/volume", params={"object": "block1", "kind": "fluence", "ex": 365})
    assert r.status_code == 200 and r.headers["x-dims"] == "4,3,6"
    assert "x-dims" in r.headers["access-control-expose-headers"].lower()
    a = np.frombuffer(r.content, dtype="<f4").reshape(4, 3, 6)
    assert a.shape == (4, 3, 6)
    r = client.get(f"/api/captures/{cid}/volume", params={"object": "block1", "kind": "source", "ex": 365, "channel": "tissue|NADH"})
    assert r.status_code == 200 and len(r.content) == 4 * 3 * 6 * 4
    for params, code in (({"object": "zz", "kind": "fluence"}, 404), ({"object": "block1", "kind": "bogus"}, 422),
                         ({"object": "block1", "kind": "fluence", "ex": 500}, 404),
                         ({"object": "block1", "kind": "source", "channel": "q"}, 404)):
        assert client.get(f"/api/captures/{cid}/volume", params=params).status_code == code
    assert client.get(f"/api/captures/{cid}/volume").status_code == 422


def wait_sweep(client, sid):
    for _ in range(500):
        st = client.get(f"/api/sweeps/{sid}").json()
        if st["state"] in ("done", "error", "cancelled"):
            return st
        time.sleep(0.02)
    raise AssertionError("sweep timeout")


def test_sweep_endpoints_and_ws(client):
    body = {"experiment": EXP, "parameter": {"path": "/acquisition/preview_scale", "values": [0.2, 0.4, 0.6]}}
    r = client.post("/api/sweeps", json=body)
    assert r.status_code == 200
    sid = r.json()["sweep_id"]
    with client.websocket_connect(f"ws://127.0.0.1:8765/api/sweeps/{sid}") as ws:
        evs = []
        while True:
            e = ws.receive_json()
            evs.append(e)
            if e["type"] in ("done", "error", "cancelled"):
                break
    assert evs[-1] == {"type": "done", "sweep_id": sid}
    assert [e["index"] for e in evs if e["type"] == "point"] == [0, 1, 2]
    st = wait_sweep(client, sid)
    assert st["state"] == "done" and len(st["capture_ids"]) == 3 and all(st["capture_ids"])
    assert st["parameter"]["values"] == [0.2, 0.4, 0.6]
    # sweep captures are reachable through the normal capture endpoints (and not evicted by the main LRU)
    a, b = st["capture_ids"][0], st["capture_ids"][2]
    ra = client.post(f"/api/captures/{a}/eem", json={"roi": {"shape": "point", "x": 5, "y": 5}}).json()
    rb = client.post(f"/api/captures/{b}/eem", json={"roi": {"shape": "point", "x": 5, "y": 5}}).json()
    assert rb["value"]["365"][2] / ra["value"]["365"][2] == pytest.approx(3.0, rel=1e-4)       # 0.6 / 0.2 scaling
    assert client.get(f"/api/captures/{a}").json()["mode"] == "preview"
    for _ in range(12):                                         # churn the main store (capacity 10)
        cap(client)
    assert client.get(f"/api/captures/{a}").status_code == 200


def test_sweep_errors_and_cancel(client):
    ok = {"path": "/acquisition/preview_scale", "values": [0.3]}
    assert client.post("/api/sweeps", json={"parameter": ok}).status_code == 422
    assert client.post("/api/sweeps", json={"experiment": EXP}).status_code == 422
    r = client.post("/api/sweeps", json={"experiment": EXP, "parameter": {"path": "/acquisition/preview_scale", "values": [0.3, -1]}})
    assert r.status_code == 422 and "values[1]" in r.json()["detail"]
    r = client.post("/api/sweeps", json={"experiment": EXP, "parameter": {"path": "/nope", "values": [1]}})
    assert r.status_code == 422 and "no key" in r.json()["detail"]
    assert client.post("/api/sweeps", json={"experiment": EXP, "parameter": ok, "mode": "zz"}).status_code == 422
    assert client.get("/api/sweeps/nope").status_code == 404
    assert client.post("/api/sweeps/nope/cancel", json={}).status_code == 404
    sid = client.post("/api/sweeps", json={"experiment": EXP, "parameter": ok}).json()["sweep_id"]
    assert client.post(f"/api/sweeps/{sid}/cancel", json={}).status_code == 200
    assert wait_sweep(client, sid)["state"] in ("done", "cancelled")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("ws://127.0.0.1:8765/api/sweeps/nope"):
            pass


def test_security_applies_to_new_endpoints(client):
    cid = cap(client)
    for url, body in ((f"/api/captures/{cid}/eem", {"roi": DISK}), ("/api/sweeps", {"experiment": EXP, "parameter": {}})):
        assert client.post(url, content=json.dumps(body), headers={**JSON, "Origin": "https://evil.example"}).status_code == 403
        assert client.post(url, content=json.dumps(body), headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.get(f"/api/captures/{cid}/volumes", headers={"Origin": "https://evil.example"}).status_code == 403
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("ws://127.0.0.1:8765/api/sweeps/x", headers={"Origin": "https://evil.example"}):
            pass
    assert client.post(f"/api/captures/{cid}/eem", content=json.dumps({"roi": DISK}),
                       headers={**JSON, "Origin": "http://localhost:5173"}).status_code == 200


def test_sweep_sample_mode_uses_capture_fn(client):
    sid = client.post("/api/sweeps", json={"experiment": EXP, "mode": "sample", "parameter": {
        "path": "/acquisition/preview_scale", "values": [0.3, 0.5]}}).json()["sweep_id"]
    st = wait_sweep(client, sid)
    assert st["state"] == "done"
    assert client.get(f"/api/captures/{st['capture_ids'][1]}").json()["mode"] == "sample"
