"""``POST /api/material_optics`` and ``GET /api/calibrated_fluorophores``: engine-backed optics for the web inspector."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spectraforge.studio import optics
from spectraforge.studio.presets import builtin_fluorophores, builtin_materials
from spectraforge.studio.presets_tissue import tissue_materials
from spectraforge.studio.server import create_app

BASE = "http://127.0.0.1:8765"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(dist_dir=tmp_path_factory.mktemp("nodist"))
    with TestClient(app, base_url=BASE) as c:
        yield c


def _tissue():
    return next(m for m in tissue_materials() if m.tissue is not None and m.tissue.blood_volume_fraction > 0 and m.tissue.water_fraction > 0)


def test_matches_engine_exactly(client):
    mat = _tissue()
    r = client.post("/api/material_optics", json={"material": mat.model_dump(mode="json"), "nm_min": 400, "nm_max": 800, "step": 10})
    assert r.status_code == 200, r.text
    d = r.json()
    nm = np.arange(400, 801, 10.0)
    ref = optics.material_optics(mat, {f.name: f for f in builtin_fluorophores()}, grid=nm)
    assert np.allclose(d["nm"], nm)
    assert np.allclose(d["mua_per_mm"], ref.mua_per_mm, rtol=1e-12)
    assert np.allclose(d["mus_prime_per_mm"], optics.reduced_scattering_per_mm(mat.scattering, nm), rtol=1e-12)
    assert d["g"] == pytest.approx(mat.scattering.g) and d["n"] == pytest.approx(mat.refractive_index)
    assert {"hbo2", "hb", "water"} <= set(d["components"])


def test_components_sum_to_total(client):
    mat = _tissue()
    d = client.post("/api/material_optics", json={"material": mat.model_dump(mode="json"), "step": 25}).json()
    tot = np.sum([np.asarray(v) for v in d["components"].values()], axis=0)
    assert np.allclose(tot, d["mua_per_mm"], rtol=1e-9, atol=1e-9)


def test_dyes_are_components(client):
    mat = next(m for m in builtin_materials() if m.dyes)
    d = client.post("/api/material_optics", json={"material": mat.model_dump(mode="json"), "nm_min": 300, "nm_max": 600, "step": 10}).json()
    assert any(k.startswith("dye:") for k in d["components"])
    assert max(d["mua_per_mm"]) > 0


def test_validation_and_limits(client):
    mat = _tissue().model_dump(mode="json")
    assert client.post("/api/material_optics", json={}).status_code == 422
    assert client.post("/api/material_optics", json={"material": {"id": 1}}).status_code == 422
    assert client.post("/api/material_optics", json={"material": mat, "step": 0}).status_code == 422
    assert client.post("/api/material_optics", json={"material": mat, "nm_min": 900, "nm_max": 350}).status_code == 422
    assert client.post("/api/material_optics", json={"material": mat, "nm_min": 350, "nm_max": 900, "step": 0.01}).status_code == 422
    assert client.post("/api/material_optics", json={"material": mat, "step": "x"}).status_code == 422
    r = client.post("/api/material_optics", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    big = client.post("/api/material_optics", content=json.dumps({"material": mat, "pad": "x" * 1_100_000}),
                      headers={"Content-Type": "application/json"})
    assert big.status_code == 413
    bad = dict(mat, dyes=[{"fluorophore": "no_such_dye", "concentration_uM": 1}])
    assert client.post("/api/material_optics", json={"material": bad}).status_code == 422


def test_json_only_post(client):
    r = client.post("/api/material_optics", content="material=1", headers={"Content-Type": "text/plain"})
    assert r.status_code in (400, 415)


def test_calibrated_library_is_strict_json(client):
    r = client.get("/api/calibrated_fluorophores")
    assert r.status_code == 200
    items = json.loads(r.text, parse_constant=lambda c: pytest.fail(f"bare {c} in response"))
    for it in items:
        assert it["spec"]["name"] and "provenance" in it
