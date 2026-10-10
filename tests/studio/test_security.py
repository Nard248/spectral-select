"""CSRF / Origin / Host / Content-Type hardening and archive-name sanitizing."""
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from spectraforge.studio import exports
from spectraforge.studio.presets import preset
from spectraforge.studio.server import create_app

from .test_server import BASE, Fakes, fake_result, run_capture

EVIL = {"Origin": "https://evil.example"}
JSON = {"Content-Type": "application/json"}


@pytest.fixture
def client():
    fk = Fakes()
    with TestClient(create_app(capture_fn=fk.capture, preview_fn=fk.preview, trace_fn=fk.trace, dist_dir="/x"),
                    base_url=BASE) as c:
        yield c


def _body():
    return json.dumps({"experiment": preset("generic_macro_bench").model_dump()})


# ------------------------------------------------------------------ content type
@pytest.mark.parametrize("ct", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"])
def test_non_json_content_type_is_415(client, ct):
    for path in ("/api/captures", "/api/preview", "/api/validate", "/api/jobs/abc/cancel"):
        r = client.post(path, content=_body(), headers={"Content-Type": ct})
        assert r.status_code == 415, (path, ct)
    r = client.post("/api/captures", content=_body())          # no content-type at all
    assert r.status_code in (415,)


def test_json_content_type_with_charset_ok(client):
    r = client.post("/api/validate", content=_body(), headers={"Content-Type": "application/json; charset=utf-8"})
    assert r.status_code == 200


def test_empty_body_json_cancel_reaches_handler(client):
    # the web client POSTs cancel with a JSON content-type and no body
    assert client.post("/api/jobs/nope/cancel", headers=JSON).status_code == 404


# ------------------------------------------------------------------ origin
@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:6666", "null", "http://127.0.0.1:5174"])
def test_cross_origin_post_rejected(client, origin):
    r = client.post("/api/captures", content=_body(), headers={**JSON, "Origin": origin})
    assert r.status_code == 403
    assert client.get("/api/captures").status_code == 200       # nothing was started / server fine


def test_cross_origin_referer_fallback_rejected(client):
    r = client.post("/api/captures", content=_body(), headers={**JSON, "Referer": "https://evil.example/x"})
    assert r.status_code == 403
    r = client.post("/api/validate", content=_body(), headers={**JSON, "Referer": "http://localhost:5173/app"})
    assert r.status_code == 200


def test_cross_origin_get_with_origin_rejected(client):
    assert client.get("/api/health", headers=EVIL).status_code == 403


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:5173", BASE])
def test_allowed_origins(client, origin):
    r = client.post("/api/validate", content=_body(), headers={**JSON, "Origin": origin})
    assert r.status_code == 200
    assert client.get("/api/health", headers={"Origin": origin}).status_code == 200


def test_missing_origin_allowed_for_non_browser_clients(client):
    assert client.post("/api/validate", content=_body(), headers=JSON).status_code == 200
    _, st = run_capture(client)
    assert st["state"] == "done"


def test_cors_is_minimal(client):
    r = client.options("/api/captures", headers={"Origin": "http://localhost:5173",
                                                 "Access-Control-Request-Method": "POST",
                                                 "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "DELETE" not in r.headers["access-control-allow-methods"]
    r = client.options("/api/captures", headers={"Origin": "https://evil.example",
                                                 "Access-Control-Request-Method": "POST"})
    assert r.status_code == 403 or "access-control-allow-origin" not in r.headers


# ------------------------------------------------------------------ host (DNS rebinding)
@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "attacker.localhost.evil.com"])
def test_bad_host_rejected(host):
    with TestClient(create_app(trace_fn=Fakes().trace, dist_dir="/x"), base_url=f"http://{host}") as c:
        assert c.get("/api/health").status_code == 403
        assert c.post("/api/validate", content=_body(), headers=JSON).status_code == 403


@pytest.mark.parametrize("host", ["localhost:8765", "127.0.0.1", "[::1]:8765"])
def test_loopback_hosts_ok(host):
    with TestClient(create_app(trace_fn=Fakes().trace, dist_dir="/x"), base_url=f"http://{host}") as c:
        assert c.get("/api/health").status_code == 200


def test_explicitly_allowed_host():
    with TestClient(create_app(trace_fn=Fakes().trace, dist_dir="/x", allowed_hosts=("lab-pc.local",)),
                    base_url="http://lab-pc.local:8765") as c:
        assert c.get("/api/health").status_code == 200


# ------------------------------------------------------------------ websocket
def test_websocket_origin_checked(client):
    job_id = client.post("/api/captures", content=_body(), headers=JSON).json()["job_id"]
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"ws://127.0.0.1:8765/api/jobs/{job_id}", headers=EVIL):
            pass
    # same-origin / dev origin / no origin are accepted
    for hdr in ({}, {"Origin": "http://localhost:5173"}, {"Origin": BASE}):
        with client.websocket_connect(f"ws://127.0.0.1:8765/api/jobs/{job_id}", headers=hdr) as ws:
            while ws.receive_json()["type"] not in ("done", "error", "cancelled"):
                pass


# ------------------------------------------------------------------ --host warning
def test_non_loopback_host_warns(capsys, monkeypatch):
    import uvicorn
    from spectraforge.studio import server, transport
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(transport, "start_warmup", lambda *a, **k: None)
    server.main(["--host", "0.0.0.0", "--port", "9"])
    assert "SECURITY WARNING" in capsys.readouterr().out
    server.main(["--port", "9"])
    assert "SECURITY WARNING" not in capsys.readouterr().out


# ------------------------------------------------------------------ names
EVIL_NAMES = ["../../evil", "/etc/passwd", "a\r\nX-Injected: 1", "..", "...", "", "naïve/日本語\x00", 'q"uote\\',
              "conc:../../x", "irradiance@365.5", "a" * 500, "C:\\Windows\\x"]


@pytest.mark.parametrize("n", EVIL_NAMES)
def test_safe_name_properties(n):
    s = exports.safe_name(n)
    assert s and len(s) <= 80
    assert all(c.isascii() and (c.isalnum() or c in "._@:+-") for c in s)
    assert not s.startswith(".") and not s.endswith(".") and ".." not in s
    assert "/" not in s and "\\" not in s and "\r" not in s and "\n" not in s


def test_safe_name_keeps_legit_layer_names():
    assert exports.safe_name("conc:EGFP") == "conc:EGFP"
    assert exports.safe_name("irradiance@365") == "irradiance@365"


@pytest.mark.parametrize("n", EVIL_NAMES)
def test_content_disposition_is_safe(n):
    cd = exports.content_disposition(f"{n}_abc.pkl")
    assert "\r" not in cd and "\n" not in cd and cd.isascii()
    assert cd.startswith('attachment; filename="')
    fn = cd.split('filename="')[1].split('"')[0]
    assert "/" not in fn and "\\" not in fn and not fn.startswith(".")


def _unsafe(names):
    return [m for m in names if m.startswith("/") or ".." in m.split("/") or "\\" in m or "\x00" in m
            or m.count("/") > 0]


def test_ground_truth_and_tiff_zip_members_are_flat_and_safe():
    r = fake_result(preset("generic_macro_bench"), exs=[365.0, 405.0])
    gt = dict(r.ground_truth)
    r.ground_truth.clear()
    for n in ("../../evil", "/etc/passwd", "a\r\nX: 1", "conc:EGFP", "irradiance@365", "naïve日本"):
        r.ground_truth[n] = gt["depth_mm"]
    r.ground_truth["x/y"] = gt["depth_mm"]
    r.ground_truth["x_y"] = gt["depth_mm"]                      # collides with the sanitized "x/y"
    payload, _, _ = exports.build_export(r, "id1", "../../evil", "ground_truth_zip")
    names = zipfile.ZipFile(io.BytesIO(payload)).namelist()
    assert not _unsafe(names), names
    assert len(names) == len(set(names)) == len(r.ground_truth) + 1
    assert "conc:EGFP.npy" in names and "irradiance@365.npy" in names
    index = json.loads(zipfile.ZipFile(io.BytesIO(payload)).read("ground_truth.json"))
    assert {v["file"] for v in index.values()} == {n for n in names if n.endswith(".npy")}
    t, _, _ = exports.build_export(r, "id1", "../../evil", "tiff_zip")
    assert not _unsafe(zipfile.ZipFile(io.BytesIO(t)).namelist())
    z, fname, _ = exports.build_export(r, "id1", "a\r\nX-Injected: 1/../..", "npz")
    assert "\r" not in fname and "/" not in fname and ".." not in fname


def test_download_endpoint_header_not_injectable():
    fk = Fakes()
    exp = preset("generic_macro_bench").model_copy(update={"name": 'a\r\nX-Injected: 1/../../"evil'})
    with TestClient(create_app(capture_fn=fk.capture, trace_fn=fk.trace, dist_dir="/x"), base_url=BASE) as c:
        jid = c.post("/api/captures", json={"experiment": exp.model_dump()}).json()["job_id"]
        st = None
        while not st or st["state"] not in ("done", "error"):
            st = c.get(f"/api/jobs/{jid}").json()
        r = c.get(f"/api/captures/{st['capture_id']}/download", params={"format": "tiff_zip"})
        assert r.status_code == 200
        assert "x-injected" not in {k.lower() for k in r.headers}
        cd = r.headers["content-disposition"]
        assert "\n" not in cd and ".." not in cd and "/" not in cd.split('filename="')[1].split('"')[0]
