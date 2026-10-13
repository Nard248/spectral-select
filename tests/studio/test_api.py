import json

import numpy as np
import pytest

from spectraforge.studio import analysis, api
from spectraforge.studio.sweeps import SweepError

from .fakes_study import COVER, base_experiment, fake_capture


def test_presets_and_set_returns_validated_copy():
    assert "layered_wedge_phantom" in api.presets()
    e = api.preset("layered_wedge_phantom")
    e2 = api.set(e, COVER, 0.3)
    assert api.get(e2, COVER) == 0.3 and api.get(e, COVER) != 0.3      # original untouched
    with pytest.raises(SweepError):
        api.set(e, "/objects/0/shape/layers/9/thickness", 1)
    with pytest.raises(SweepError):
        api.set(e, "/acquisition/preview_scale", -1)                    # fails validation
    with pytest.raises(KeyError):
        api.preset("nope")


def test_set_many_and_load_dump(tmp_path):
    e = api.preset("generic_macro_bench")
    e2 = api.set_many(e, {"/acquisition/seed": 7, "/acquisition/preview_scale": 0.5})
    assert e2.acquisition.seed == 7 and e2.acquisition.preview_scale == 0.5
    p = api.dump(e2, tmp_path / "e.json")
    assert api.load(p) == e2


def test_capture_modes_validated():
    with pytest.raises(ValueError):
        api.capture(api.preset("generic_macro_bench"), mode="bogus")


def test_save_load_roundtrip_and_provenance(tmp_path):
    res = fake_capture(base_experiment())
    p = api.save(res, tmp_path / "out.npz")
    back = api.load_capture(p)
    ex = res.excitations_nm[0]
    assert np.allclose(back.adu[ex], res.adu[ex])
    prov = back.meta["provenance"]
    assert prov["experiment_hash"] == api.experiment_hash(base_experiment())
    roi = {"shape": "rect", "x0": 2, "y0": 2, "x1": 10, "y1": 10}
    e = api.eem(back, roi)
    assert e.n_pixels > 0
    r = api.response(back, "layer_thickness:cover", {"kind": "band_sum", "ex": ex}, roi=roi, bins=2)
    assert r.unit


def test_to_spectradata_roundtrip(tmp_path):
    from spectral_select.types import SpectraData
    res = fake_capture(base_experiment())
    labels = {"cls": np.zeros((res.height, res.width), np.int16)}
    sd = api.to_spectradata(res, "phantom", labels=labels)
    ex = res.excitations_nm[0]
    assert sd.sample_name == "phantom" and sd.excitation_wavelengths == [ex]
    assert sd.excitations[ex].cube.shape == res.adu[ex].shape
    off = 100.0
    assert np.allclose(sd.excitations[ex].cube, res.adu[ex] - off)
    assert sd.excitations[ex].exposure_time == pytest.approx(0.1)
    assert "provenance" in sd.metadata and "labels" in sd.metadata
    pk = tmp_path / "s.pkl"
    sd.to_pickle(pk)
    assert SpectraData.from_pickle(pk).excitations[ex].cube.shape == res.adu[ex].shape


def test_real_preview_attaches_provenance_and_is_deterministic():
    e = api.preset("generic_macro_bench")
    a = api.capture(e, "preview", seed=3)
    b = api.capture(e, "preview", seed=3)
    c = api.capture(e, "preview", seed=4)
    ex = a.excitations_nm[0]
    assert np.array_equal(a.adu[ex], b.adu[ex]) and not np.array_equal(a.adu[ex], c.adu[ex])
    assert a.meta["provenance"]["seed"] == 3 and a.meta["provenance"]["requested_mode"] == "preview"
