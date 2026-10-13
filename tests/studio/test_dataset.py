import json

import numpy as np
import pytest

from spectraforge.studio.dataset import generate_dataset, iter_dataset, label_map, load_index, load_sample
from spectraforge.studio.study import StudyError
from spectraforge.studio.study_schema import DatasetSpec

from .fakes_study import COVER, FLUOR, base_experiment, fake_capture


def dspec(n=4, **kw):
    return DatasetSpec(name="ds", base_experiment=base_experiment(), n=n, design="lhs", seed=3,
                       parameters=[{"pointer": COVER, "label": "cover", "low": 0.1, "high": 1.0},
                                   {"pointer": FLUOR, "label": "fluor", "low": 0.2, "high": 1.0}],
                       targets=[{"name": "cover_mean", "gt": "layer_thickness:cover"}],
                       maps=["layer_thickness:cover"], **kw)


def cap(exp, mode, ex):
    return fake_capture(exp, mode, ex)


def test_generate_and_roundtrip_through_spectral_select(tmp_path):
    from spectral_select.types import SpectraData
    idx = generate_dataset(dspec(), tmp_path, capture_fn=cap)
    assert idx["n"] == 4 and (tmp_path / "datasets.json").exists()
    assert idx["class_names"] == {"0": "background", "1": "substrate", "2": "fluorescent layer", "3": "cover"}
    assert len({e["exp_hash"] for e in idx["samples"]}) == 4
    for e in idx["samples"]:
        sd = SpectraData.from_pickle(tmp_path / e["files"]["spectradata"])           # the repo's own loader
        assert isinstance(sd, SpectraData) and sd.excitation_wavelengths == [340.0]
        assert sd.metadata["labels"].shape == (16, 20) and sd.metadata["class_names"]["2"] == "fluorescent layer"
        assert sd.metadata["targets"]["cover_mean"] == pytest.approx(e["params"]["cover"], rel=1e-5)
    sd, labels, extra = load_sample(tmp_path, "s00001")
    assert labels.dtype == np.int16 and set(np.unique(labels)) <= {0, 1, 2, 3}
    assert extra["maps"]["layer_thickness:cover"].shape == labels.shape
    assert extra["targets"]["cover_mean"] == pytest.approx(extra["params"]["cover"], rel=1e-5)
    assert len(list(iter_dataset(tmp_path))) == 4


def test_labels_origin_layer_matches_fraction_argmax():
    res = fake_capture(base_experiment())
    lab, names = label_map(res, "origin_layer")
    assert names[2] == "fluorescent layer"
    assert (lab[:5, :5] == 2).all()                                  # patch: all signal from layer 2 (fraction 1)
    assert (lab[8:, 8:] == 2).all()                                  # elsewhere 0.7 fluorescent > 0.3 substrate
    top, _ = label_map(res, "top_layer")
    assert (top == 3).all()                                          # cover thickness > 0 everywhere
    obj, on = label_map(res, "object")
    assert on[0] == "background" and (obj == 1).all()
    g, gn = label_map(res, "gt_key:object")
    assert (g == 0).all()
    with pytest.raises(StudyError):
        label_map(res, "nope")


def test_resume_skips_finished_samples(tmp_path):
    calls = []
    generate_dataset(dspec(3), tmp_path, capture_fn=lambda *a: (calls.append(1), cap(*a))[1])
    generate_dataset(dspec(3), tmp_path, capture_fn=lambda *a: (calls.append(1), cap(*a))[1])
    assert len(calls) == 3


def test_deterministic(tmp_path):
    a = generate_dataset(dspec(), tmp_path / "a", capture_fn=cap)
    b = generate_dataset(dspec(), tmp_path / "b", capture_fn=cap)
    assert [e["exp_hash"] for e in a["samples"]] == [e["exp_hash"] for e in b["samples"]]
    sa, la, _ = load_sample(tmp_path / "a", "s00002")
    sb, lb, _ = load_sample(tmp_path / "b", "s00002")
    assert np.array_equal(sa.excitations[340.0].cube, sb.excitations[340.0].cube) and np.array_equal(la, lb)


def test_no_parameters_varies_only_noise_seed(tmp_path):
    idx = generate_dataset(DatasetSpec(base_experiment=base_experiment(), n=3), tmp_path, capture_fn=cap)
    assert [e["seed"] for e in idx["samples"]] == [0, 1, 2]
