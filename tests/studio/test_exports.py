import io
import json
import zipfile

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from spectraforge.studio import exports, render_png
from spectraforge.studio.presets import preset
from spectraforge.studio.server import create_app

from .test_server import BASE, B, H, W, Fakes, fake_result, run_capture


@pytest.fixture
def result():
    return fake_result(preset("generic_macro_bench"), exs=[365.0, 405.0])


def test_spectradata_pkl_reloads_with_existing_loader(tmp_path, result):
    from spectral_select.types import SpectraData
    payload, fname, _ = exports.build_export(result, "abc123", "My Exp/1", "spectradata_pkl")
    assert fname == "My_Exp_1_abc123.pkl"
    f = tmp_path / fname
    f.write_bytes(payload)
    sd = SpectraData.from_pickle(f)
    assert sd.excitation_wavelengths == [365.0, 405.0]
    ed = sd.get_excitation(365.0)
    assert ed.shape == (H, W, B) and ed.cube.dtype == np.float32
    off = preset("generic_macro_bench").camera().sensor.offset_adu
    np.testing.assert_allclose(ed.cube, result.adu[365.0] - off, rtol=1e-6)
    assert ed.emission_wavelengths == [float(w) for w in result.emission_nm[365.0]]
    assert ed.exposure_time == pytest.approx(0.05) and ed.laser_power == 2.0
    assert sd.metadata["engine_version"] == result.engine_version
    assert sd.metadata["ground_truth_layers"] == ["depth_mm", "material"]
    assert json.loads(sd.metadata["experiment_json"])["name"] == preset("generic_macro_bench").name
    assert sd.metadata["seed"] == 0 and sd.mask.shape == (H, W)
    sd.to_pickle(tmp_path / "again.pkl")      # round trip through the loader's own writer
    assert SpectraData.from_pickle(tmp_path / "again.pkl").excitation_wavelengths == [365.0, 405.0]


def test_npz(result):
    payload, _, _ = exports.build_export(result, "id1", "x", "npz")
    z = np.load(io.BytesIO(payload))
    assert z["adu_365"].shape == (H, W, B) and z["radiance_405"].shape == (H, W, B)
    assert z["emission_nm_365"].shape == (B,) and "gt_material" in z.files
    assert json.loads(str(z["experiment_json"]))["name"]


def test_tiff_zip(result):
    payload, _, _ = exports.build_export(result, "id1", "x", "tiff_zip")
    zf = zipfile.ZipFile(io.BytesIO(payload))
    assert sorted(zf.namelist()) == ["adu_ex365nm.tif", "adu_ex405nm.tif", "experiment.json"]
    arr = tifffile.imread(io.BytesIO(zf.read("adu_ex365nm.tif")))
    assert arr.dtype == np.uint16 and arr.shape == (B, H, W)
    np.testing.assert_array_equal(arr[2], np.rint(result.adu[365.0][:, :, 2]).clip(0, 65535).astype(np.uint16))
    with tifffile.TiffFile(io.BytesIO(zf.read("adu_ex365nm.tif"))) as t:
        assert json.loads(t.pages[0].description)["excitation_nm"] == 365.0


def test_ground_truth_zip(result):
    payload, _, _ = exports.build_export(result, "id1", "x", "ground_truth_zip")
    zf = zipfile.ZipFile(io.BytesIO(payload))
    assert {"material.npy", "depth_mm.npy", "ground_truth.json"} <= set(zf.namelist())
    np.testing.assert_array_equal(np.load(io.BytesIO(zf.read("material.npy"))), result.ground_truth["material"])


def test_experiment_json_and_unknown(result):
    payload, _, media = exports.build_export(result, "i", "n", "experiment_json")
    assert media == "application/json" and json.loads(payload)["schema_version"]
    with pytest.raises(KeyError):
        exports.build_export(result, "i", "n", "bogus")


def test_download_endpoint_all_formats():
    fk = Fakes()
    with TestClient(create_app(capture_fn=fk.capture, trace_fn=fk.trace, dist_dir="/x"), base_url=BASE) as c:
        _, st = run_capture(c)
        cid = st["capture_id"]
        for fmt in exports.FORMATS:
            r = c.get(f"/api/captures/{cid}/download", params={"format": fmt})
            assert r.status_code == 200 and len(r.content) > 100, fmt
            cd = r.headers["content-disposition"]
            assert cd.startswith("attachment") and cid in cd
        assert c.get(f"/api/captures/{cid}/download", params={"format": "x"}).status_code == 422


def test_render_png_colour_science():
    em = np.arange(400.0, 701.0, 10.0)
    cube = np.zeros((2, 3, em.size))
    for j, centre in enumerate((620.0, 540.0, 450.0)):          # red, green, blue emitters
        cube[0, j] = np.exp(-0.5 * ((em - centre) / 12) ** 2)
    rgb = render_png.true_color(cube, em)
    assert rgb.dtype == np.uint8 and rgb.shape == (2, 3, 3)
    assert rgb[0, 0, 0] > rgb[0, 0, 2] and rgb[0, 0, 0] > rgb[0, 0, 1]
    assert rgb[0, 1, 1] > rgb[0, 1, 0] and rgb[0, 1, 1] > rgb[0, 1, 2]
    assert rgb[0, 2, 2] > rgb[0, 2, 0] and rgb[0, 2, 2] > rgb[0, 2, 1]
    assert render_png.true_color(np.zeros((2, 2, 4)), em[:4]).max() == 0
    assert render_png.cie_cmf([555.0])[0][1] == pytest.approx(1.0, abs=0.02)
    v = render_png.viridis(np.array([0.0, 1.0]))
    assert v[0, 2] > v[0, 0] and v[1, 0] > 0.9
