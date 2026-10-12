import io
import json

import numpy as np
import pytest

from spectraforge.studio import analysis as an
from spectraforge.studio import exports

from .fakes_v11 import B, H, OFFSET, W, fake_block_result


@pytest.fixture
def cap():
    return fake_block_result()


# ----------------------------------------------------------------------------- normalisation
def test_norm_modes_hand_computed(cap):
    dark = fake_block_result("dark", scale_adu=0.0)            # all pixels == offset
    white = fake_block_result("white_reference", scale_adu=10.0)
    s = cap.adu[365.0]
    d = dark.adu[365.0]
    w = white.adu[365.0]
    out, unit = an.normalize(cap, {"mode": "adu"})
    assert unit == "ADU" and out[365.0].dtype == np.float32
    np.testing.assert_allclose(out[365.0], s)
    out, unit = an.normalize(cap, {"mode": "radiance"})
    np.testing.assert_allclose(out[405.0], cap.radiance[405.0]) and unit.startswith("mW")
    out, unit = an.normalize(cap, {"mode": "electrons"})
    np.testing.assert_allclose(out[365.0], cap.electrons_mean[365.0])
    out, _ = an.normalize(cap, {"mode": "dark_subtracted"}, dark=dark)
    np.testing.assert_allclose(out[365.0], s - d)
    out, unit = an.normalize(cap, {"mode": "exposure_power"})
    np.testing.assert_allclose(out[365.0], (s - OFFSET) / (0.05 * 2.0), rtol=1e-5)
    assert "s" in unit and "mW" in unit
    out, unit = an.normalize(cap, {"mode": "white_normalized"}, dark=dark, white=white)
    np.testing.assert_allclose(out[365.0], (s - d) / (w - d), rtol=1e-5)
    assert unit == "ratio"
    # without a dark capture the sensor offset plays the dark's role
    out, _ = an.normalize(cap, {"mode": "white_normalized"}, white=white)
    np.testing.assert_allclose(out[365.0], (s - OFFSET) / (w - OFFSET), rtol=1e-5)


def test_irradiance_normalized_hand_computed(cap):
    """(ADU - offset) / (t_exp[s] * E_ex); NaN where E_ex = 0 (lamp gradient divided out)."""
    e365 = np.tile(np.linspace(0.0, 2.0, W), (H, 1))          # col 0 has E = 0
    cap.ground_truth["irradiance@365"] = e365
    cap.ground_truth["irradiance@405"] = np.ones((H, W))
    out, unit = an.normalize(cap, {"mode": "irradiance_normalized"})
    o = out[365.0]
    assert unit == "ADU/(s·mW/mm²)"
    np.testing.assert_allclose(o[:, 1:], (cap.adu[365.0][:, 1:] - OFFSET) / (0.05 * e365[:, 1:, None]), rtol=1e-5)
    assert np.isnan(o[:, 0]).all() and np.isfinite(o[:, 1:]).all()
    # a vanishing tail (< 1e-3 of the maximum, e.g. a narrow beam's wings) is blank, not noise / ~0
    cap.ground_truth["irradiance@365"] = np.where(np.arange(W)[None, :] < 5, 1e-9, e365 + 1.0) * np.ones((H, 1))
    t = an.normalize(cap, {"mode": "irradiance_normalized"})[0][365.0]
    assert np.isnan(t[:, :5]).all() and np.isfinite(t[:, 5:]).all()
    cap.ground_truth["irradiance@365"] = e365
    r = an.eem(cap, {"shape": "point", "x": 10, "y": 3}, {"mode": "irradiance_normalized"})
    np.testing.assert_allclose(r.value[365.0], o[3, 10], rtol=1e-5)
    # a uniform irradiance map reduces to exposure-only scaling
    np.testing.assert_allclose(r.value[405.0], (cap.adu[405.0][3, 10] - OFFSET) / 0.05, rtol=1e-5)


def test_irradiance_normalized_needs_irradiance_map(cap):
    with pytest.raises(an.AnalysisError, match="irradiance@365"):
        an.normalize(cap, {"mode": "irradiance_normalized"})


def test_white_safe_division_and_missing_inputs(cap):
    white = fake_block_result("white_reference", scale_adu=0.0)       # W - D == 0 everywhere
    out, _ = an.normalize(cap, {"mode": "white_normalized"}, white=white)
    assert np.isnan(out[365.0]).all()
    with pytest.raises(an.AnalysisError, match="dark"):
        an.normalize(cap, {"mode": "dark_subtracted"})
    with pytest.raises(an.AnalysisError, match="white"):
        an.normalize(cap, {"mode": "white_normalized"})
    dark = fake_block_result("dark", power_mW=0.0)
    with pytest.raises(an.AnalysisError, match="power"):
        an.normalize(dark, {"mode": "exposure_power"})
    with pytest.raises(an.AnalysisError, match="invalid norm"):
        an.normalize(cap, {"mode": "bogus"})


def test_dark_white_must_match_geometry(cap):
    bad_size = fake_block_result("dark")
    bad_size.width = W + 1
    with pytest.raises(an.AnalysisError, match="px"):
        an.normalize(cap, {"mode": "dark_subtracted"}, dark=bad_size)
    bad_ex = fake_block_result("dark")
    bad_ex.excitations_nm = [365.0]
    with pytest.raises(an.AnalysisError, match="excitations"):
        an.normalize(cap, {"mode": "dark_subtracted"}, dark=bad_ex)
    bad_bands = fake_block_result("dark")
    bad_bands.emission_nm = {365.0: np.arange(B) * 1.0, 405.0: bad_bands.emission_nm[405.0]}
    with pytest.raises(an.AnalysisError, match="bands"):
        an.normalize(cap, {"mode": "dark_subtracted"}, dark=bad_bands)
    with pytest.raises(an.AnalysisError, match="bands"):
        an.normalize(cap, {"mode": "white_normalized"}, white=bad_bands)


# ----------------------------------------------------------------------------- ROI masks
def test_roi_masks():
    m = an.roi_mask({"shape": "point", "x": 3, "y": 4}, H, W)
    assert m.sum() == 1 and m[4, 3]
    assert an.roi_mask({"shape": "point", "x": -3, "y": 4}, H, W).sum() == 0
    d1 = an.roi_mask({"shape": "disk", "x": 10, "y": 10, "radius_px": 1}, H, W)
    assert d1.sum() == 5 and d1[10, 10] and d1[10, 11] and not d1[11, 11]
    assert an.roi_mask({"shape": "disk", "x": 10, "y": 10, "radius_px": 2}, H, W).sum() == 13
    assert an.roi_mask({"shape": "disk", "x": 10, "y": 10, "radius_px": 0}, H, W).sum() == 1
    assert an.roi_mask({"shape": "disk", "x": 0, "y": 0, "radius_px": 2}, H, W).sum() == 6      # clipped
    r = an.roi_mask({"shape": "rect", "x0": 12, "y0": 8, "x1": 4, "y1": 2}, H, W)                # unordered corners
    assert r.sum() == 9 * 7 and r[2, 4] and r[8, 12] and not r[9, 12]
    e = an.roi_mask({"shape": "ellipse", "cx": 15, "cy": 10, "rx": 5, "ry": 2}, H, W)
    assert e[10, 10] and e[10, 20] and not e[10, 21] and e[12, 15] and not e[13, 15]
    poly = an.roi_mask({"shape": "polygon", "points": [[2, 2], [8, 2], [8, 8], [2, 8]]}, H, W)
    assert poly[5, 5] and not poly[5, 9] and not poly[9, 5]
    # even-odd: the centre of a pentagram (drawn as a self-intersecting 5-point star) is a hole
    ang = np.deg2rad(90 + 144 * np.arange(5))
    star = [[15 + 8 * np.cos(a), 10 - 8 * np.sin(a)] for a in ang]
    sm = an.roi_mask({"shape": "polygon", "points": star}, 21, 31)
    assert not sm[10, 15] and sm[6, 15]
    # the unit-square corners as pixel centres: concave "L"
    L = an.roi_mask({"shape": "polygon", "points": [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]]}, H, W)
    assert L[1, 4] and L[4, 1] and not L[4, 4]


def test_bad_rois(cap):
    for bad in ({"shape": "disk", "x": 1, "y": 1, "radius_px": -1}, {"shape": "polygon", "points": [[0, 0], [1, 1]]},
                {"shape": "blob"}, {"shape": "rect", "x0": 0}, {"shape": "point", "x": float("nan"), "y": 0}):
        with pytest.raises(an.AnalysisError, match="invalid ROI"):
            an.eem(cap, bad)
    with pytest.raises(an.AnalysisError, match="no pixels"):
        an.eem(cap, {"shape": "disk", "x": 500, "y": 5, "radius_px": 3})


# ----------------------------------------------------------------------------- EEM
def test_eem_mean_median_std_and_grid(cap):
    roi = {"shape": "rect", "x0": 10, "y0": 3, "x1": 14, "y1": 5}                      # cols 10..14, rows 3..5
    r = an.eem(cap, roi, {"mode": "adu"})
    assert r.n_pixels == 15
    cols = np.arange(10, 15)
    exp365 = OFFSET + (cols + 1).mean() * (np.arange(B) + 1) * 10.0
    np.testing.assert_allclose(r.value[365.0], exp365, rtol=1e-5)
    np.testing.assert_allclose(r.std[365.0], np.std(cols + 1) * (np.arange(B) + 1) * 10.0, rtol=1e-4)
    np.testing.assert_allclose(r.value[405.0], OFFSET + 5.0 * (np.arange(B) + 1) * 10.0, rtol=1e-5)  # rows 3..5 -> 4,5,6
    med = an.eem(cap, {"shape": "rect", "x0": 10, "y0": 3, "x1": 14, "y1": 3}, stat="median")
    np.testing.assert_allclose(med.value[365.0], OFFSET + 13.0 * (np.arange(B) + 1) * 10.0, rtol=1e-5)
    # common grid: union 420..490 in 2 nm steps; 365 spans 420-470, 405 spans 440-490
    g = r.grid_emission_nm
    assert g[0] == 420 and g[-1] == 490 and np.allclose(np.diff(g), 2.0)
    assert np.isfinite(r.grid_value[0][g <= 470]).all() and np.isnan(r.grid_value[0][g > 470]).all()
    assert np.isnan(r.grid_value[1][g < 440]).all() and np.isfinite(r.grid_value[1][g >= 440]).all()
    # linear interpolation at a point between bands 430 and 440
    k = int(np.where(g == 436)[0][0])
    assert r.grid_value[0][k] == pytest.approx(0.4 * r.value[365.0][1] + 0.6 * r.value[365.0][2])
    d = r.to_dict()
    assert d["value"]["365"] and d["grid_value"][0][-1] is None and d["roi"]["shape"] == "rect"
    json.dumps(d, allow_nan=False)
    with pytest.raises(an.AnalysisError):
        an.eem(cap, roi, stat="mode")


def test_eem_single_pixel_and_norm(cap):
    r = an.eem(cap, {"shape": "point", "x": 7, "y": 2}, {"mode": "exposure_power"})
    assert r.n_pixels == 1 and r.unit.startswith("ADU/")
    np.testing.assert_allclose(r.value[365.0], 8 * (np.arange(B) + 1) * 10.0 / (0.05 * 2.0), rtol=1e-5)
    assert np.allclose(r.std[365.0], 0)


# ----------------------------------------------------------------------------- ROI stats
def test_roi_stats_ground_truth_and_components(cap):
    roi = {"shape": "rect", "x0": 0, "y0": 0, "x1": 9, "y1": 3}                       # cols 0..9 (0..4 are NaN GT)
    st = an.roi_stats(cap, roi)
    t = st.ground_truth["layer_thickness:cover"]
    cols = np.arange(5, 10)
    assert t["mean"] == pytest.approx((2.0 * cols / (W - 1)).mean())
    assert t["min"] == pytest.approx(2.0 * 5 / (W - 1)) and t["max"] == pytest.approx(2.0 * 9 / (W - 1))
    assert "label" not in st.ground_truth and "depth_mm" in st.ground_truth
    only_nan = an.roi_stats(cap, {"shape": "rect", "x0": 0, "y0": 0, "x1": 3, "y1": 3})
    assert "layer_thickness:cover" not in only_nan.ground_truth
    assert st.components is None

    calls = []

    def explain(x, y):
        calls.append((x, y))
        return {"components": {"365": {"dye:A@tissue": [x] * B, "elastic": [1.0] * B}, "405.0": {"raman@cover": [2.0] * B}}}
    big = {"shape": "rect", "x0": 0, "y0": 0, "x1": 29, "y1": 19}                      # 600 px -> 64 sampled
    cache: dict = {}
    s2 = an.roi_stats(cap, big, components=True, explain=explain, cache=cache)
    assert len(calls) == 64 and len(cache) == 64
    assert set(s2.components) == {"365", "405"}
    assert s2.components["365"]["elastic"] == pytest.approx([1.0] * B)
    xs = np.array([c[0] for c in calls])
    assert s2.components["365"]["dye:A@tissue"][0] == pytest.approx(xs.mean())
    an.roi_stats(cap, big, components=True, explain=explain, cache=cache)
    assert len(calls) == 64                                                           # served from cache
    s3 = an.roi_stats(cap, {"shape": "point", "x": 3, "y": 3}, components=True, explain=explain)
    assert s3.components["405"]["raman@cover"] == pytest.approx([2.0] * B)
    with pytest.raises(an.AnalysisError):
        an.roi_stats(cap, big, components=True)
    json.dumps(s2.to_dict(), allow_nan=False)


# ----------------------------------------------------------------------------- line profile
def test_line_profile_gradient(cap):
    p = an.line_profile(cap, (2, 5), (22, 5), norm={"mode": "adu"}, ground_truth=["layer_thickness:cover"])
    assert p.distance_px[0] == 0 and p.distance_px[-1] == pytest.approx(20) and len(p.distance_px) == 21
    # adu365 = offset + (col+1)*(b+1)*10 : bilinear along x reproduces it exactly at sub-pixel positions
    xs = 2 + p.distance_px
    np.testing.assert_allclose(p.value[365.0][:, 2], OFFSET + (xs + 1) * 30.0, rtol=1e-5)
    gtv = p.ground_truth["layer_thickness:cover"]
    assert np.isnan(gtv[xs < 4]).all() and np.isnan(gtv[(xs > 4) & (xs < 5)]).all()      # NaN neighbour -> NaN
    np.testing.assert_allclose(gtv[xs >= 5], 2.0 * xs[xs >= 5] / (W - 1), rtol=1e-6)
    assert p.value[405.0].shape == (21, B)
    # off-grid line: half-pixel offsets in y interpolate the row gradient of ex 405
    q = an.line_profile(cap, (3.5, 4.5), (3.5, 8.5), excitations_nm=[405])
    ys = 4.5 + q.distance_px
    np.testing.assert_allclose(q.value[405.0][:, 0], OFFSET + (ys + 1) * 10.0, rtol=1e-5)
    assert list(q.value) == [405.0]


def test_line_profile_width_outside_and_distance(cap):
    # width averaging of the column gradient across rows is a no-op for ex 365, and averages ex 405 rows
    w1 = an.line_profile(cap, (2, 10), (20, 10), width_px=5)
    exp = OFFSET + (10 + 1) * 10.0 * (np.arange(B) + 1)             # mean row index over rows 8..12 is 10
    np.testing.assert_allclose(w1.value[405.0][5], exp, rtol=1e-5)
    # samples past the image edge are NaN, inside ones valid
    out = an.line_profile(cap, (25, 5), (35, 5))
    assert np.isfinite(out.value[365.0][:5]).all() and np.isnan(out.value[365.0][6:]).all()
    # distance in mm: depth 300 mm x (pitch*binning*...)/focal
    cam = json.loads(cap.experiment_json)["cameras"][0]
    s = cam["sensor"]
    fp = 300.0 * (s["pixel_pitch_um"] * s["binning"] * 1e-3 * s["width_px"] / W) / cam["lens"]["focal_length_mm"]
    p = an.line_profile(cap, (0, 0), (10, 0))
    assert p.distance_mm[-1] == pytest.approx(10 * fp, rel=1e-6) and p.distance_mm[0] == 0
    cap.ground_truth["depth_mm"] = np.full((H, W), np.inf)
    assert an.line_profile(cap, (0, 0), (10, 0)).distance_mm is None
    with pytest.raises(an.AnalysisError, match="coincide"):
        an.line_profile(cap, (1, 1), (1, 1))
    with pytest.raises(an.AnalysisError, match="unknown ground-truth"):
        an.line_profile(cap, (0, 0), (5, 5), ground_truth=["nope"])
    with pytest.raises(an.AnalysisError, match="not a numeric"):
        an.line_profile(cap, (0, 0), (5, 5), ground_truth=["label"])


# ----------------------------------------------------------------------------- response curve
def test_response_curve_recovers_known_dependence():
    cap = fake_block_result()
    thick = cap.ground_truth["layer_thickness:cover"]
    # band 460 nm of ex 365 (index 4) := 1000 * exp(-1.3 * thickness) on the block
    f = 1000.0 * np.exp(-1.3 * np.nan_to_num(thick))
    a = cap.adu[365.0].astype(np.float64)
    a[..., 4] = f + OFFSET
    cap.adu[365.0] = a.astype(np.float32)
    rc = an.response_curve(cap, "layer_thickness:cover", {"kind": "band", "ex": 365, "band_nm": 458}, bins=10,
                           norm={"mode": "exposure_power"})
    assert rc.unit.startswith("ADU/")
    k = 0.05 * 2.0
    good = rc.count > 0
    assert good.all() and rc.count.sum() == (H * (W - 5))
    # bin mean is the mean of the function over the thicknesses in that bin; with 25 distinct columns
    # per-bin std is small and the log-slope recovers the decay constant
    slope = np.polyfit(rc.bin_centres, np.log(rc.mean * k), 1)[0]
    assert slope == pytest.approx(-1.3, rel=0.05)
    assert len(rc.scatter_x) <= 4000 and len(rc.scatter_x) == len(rc.scatter_y)
    d = rc.to_dict()
    assert d["x_ground_truth"] == "layer_thickness:cover" and d["metric"]["kind"] == "band"
    json.dumps(d, allow_nan=False)


def test_response_curve_metrics_and_roi(cap):
    roi = {"shape": "rect", "x0": 10, "y0": 0, "x1": 19, "y1": 19}
    r = an.response_curve(cap, "layer_thickness:cover", {"kind": "band_ratio", "a": {"ex": 365, "band_nm": 440},
                                                         "b": {"ex": 365, "band_nm": 420}}, roi, bins=5)
    assert np.isfinite(r.scatter_y).all()
    assert r.unit == "ratio" and r.count.sum() == 10 * H
    s = an.response_curve(cap, "layer_thickness:cover", {"kind": "band_sum", "ex": 365, "from_nm": 430, "to_nm": 450},
                          roi, norm={"mode": "adu"})
    cols = np.array(s.scatter_x) * (W - 1) / 2.0
    np.testing.assert_allclose(s.scatter_y, 3 * OFFSET + (cols + 1) * 10.0 * (2 + 3 + 4), rtol=1e-5)
    with pytest.raises(an.AnalysisError, match="no band"):
        an.response_curve(cap, "layer_thickness:cover", {"kind": "band_sum", "ex": 365, "from_nm": 900, "to_nm": 950})
    with pytest.raises(an.AnalysisError, match="invalid metric"):
        an.response_curve(cap, "layer_thickness:cover", {"kind": "nope"})
    with pytest.raises(an.AnalysisError, match="unknown ground-truth"):
        an.response_curve(cap, "bogus", {"kind": "peak_wavelength", "ex": 365})
    with pytest.raises(an.AnalysisError, match="no pixel with a finite"):
        an.response_curve(cap, "layer_thickness:cover", {"kind": "peak_wavelength", "ex": 365},
                          {"shape": "rect", "x0": 0, "y0": 0, "x1": 3, "y1": 3})
    with pytest.raises(an.AnalysisError, match="not in the capture"):
        an.response_curve(cap, "layer_thickness:cover", {"kind": "band", "ex": 300, "band_nm": 440})


def test_peak_wavelength_parabolic():
    c = np.arange(420.0, 520.0, 10.0)
    for true in (453.0, 461.7, 438.2):
        spec = np.exp(-0.5 * ((c - true) / 25.0) ** 2)[None, :]
        # parabola through a gaussian's top is approximate; the error must stay well below the band step
        assert abs(an.peak_wavelength(spec, c)[0] - true) < 1.0
    flat_edge = np.arange(10.0)[None, :]
    assert an.peak_wavelength(flat_edge, c)[0] == c[-1]                                   # end band: no refinement
    assert np.isnan(an.peak_wavelength(np.full((1, 10), np.nan), c)[0])
    quad = (-(c - 447.3) ** 2)[None, :]                                                   # exact parabola
    assert an.peak_wavelength(quad, c)[0] == pytest.approx(447.3)


# ----------------------------------------------------------------------------- volumes
def test_volume_infos_and_volume(cap):
    infos = an.volume_infos(cap)
    kinds = {i["kind"]: i for i in infos}
    assert set(kinds) == {"fluence", "absorbed", "source"}
    f = kinds["fluence"]
    assert f["object_id"] == "block1" and f["dims"] == [4, 3, 6] and f["sublayers"] == 2
    assert f["layer_names"] == ["substrate", "tissue", "cover"] and f["unit"] == "mW/mm^2"
    assert len(f["x_edges_mm"]) == 5 and len(f["z_edges_mm"]) == 4 and f["excitations_nm"] == [365.0, 405.0]
    assert kinds["source"]["channels"] == ["tissue|FAD", "tissue|NADH"] and kinds["source"]["excitations_nm"] == [365.0]
    json.dumps(infos, allow_nan=False)
    v = an.volume(cap, "block1", "fluence", 405)
    assert v.shape == (4, 3, 6) and v.dtype == np.float32 and v.flags.c_contiguous
    np.testing.assert_array_equal(v.reshape(4, 3, 3, 2), cap.volumes["block1"]["fluence"][405.0])
    s = an.volume(cap, "block1", "source", 365, "tissue|NADH")
    np.testing.assert_array_equal(s.reshape(4, 3, 3, 2), cap.volumes["block1"]["source"][(365.0, "tissue|NADH")])
    for args in (("nope", "fluence"), ("block1", "bogus"), ("block1", "fluence", 500), ("block1", "source", 365, "x")):
        with pytest.raises(an.AnalysisError):
            an.volume(cap, *args)
    plain = fake_block_result()
    plain.volumes, plain.blocks = {}, []
    assert an.volume_infos(plain) == []


# ----------------------------------------------------------------------------- npz round trip
def test_npz_roundtrip_and_notebook_example(tmp_path, cap):
    payload = exports.to_npz(cap)
    path = tmp_path / "cap.npz"
    path.write_bytes(payload)
    back = an.load_npz_capture(path)
    assert (back.width, back.height, back.mode) == (W, H, "sample") and back.excitations_nm == [365.0, 405.0]
    for ex in cap.excitations_nm:
        np.testing.assert_array_equal(back.adu[ex], cap.adu[ex])
        np.testing.assert_array_equal(back.electrons_mean[ex], cap.electrons_mean[ex])
        np.testing.assert_array_equal(back.emission_nm[ex], cap.emission_nm[ex])
        assert back.exposure_ms[ex] == 50.0 and back.power_mW[ex] == 2.0
    assert set(back.ground_truth) == set(cap.ground_truth)                    # incl. 'layer_thickness:cover'
    np.testing.assert_array_equal(back.ground_truth["layer_thickness:cover"], cap.ground_truth["layer_thickness:cover"])
    assert back.blocks[0]["layer_names"] == ["substrate", "tissue", "cover"]
    # volumes: same keys / values, grid reconstructed
    assert an.volume_infos(back) == an.volume_infos(cap)
    np.testing.assert_array_equal(an.volume(back, "block1", "source", 365, "tissue|NADH"),
                                  an.volume(cap, "block1", "source", 365, "tissue|NADH"))
    z = np.load(io.BytesIO(payload))
    assert "volume/block1/fluence/365" in z.files and "volume/block1/source/365/tissue_NADH" in z.files
    # every analysis function gives identical answers offline
    roi = {"shape": "disk", "x": 12, "y": 8, "radius_px": 3}
    a, b = an.eem(cap, roi, {"mode": "exposure_power"}), an.eem(back, roi, {"mode": "exposure_power"})
    np.testing.assert_allclose(a.value[365.0], b.value[365.0])
    with pytest.raises(an.AnalysisError):
        np.savez(tmp_path / "other.npz", x=np.zeros(2))
        an.load_npz_capture(tmp_path / "other.npz")

    # the module-docstring example, run verbatim (apart from the path)
    ns: dict = {}
    code = an.__doc__.split("::\n\n", 1)[1].split("\n\nConventions", 1)[0]
    lines = [ln[4:] for ln in code.splitlines()]
    src = "\n".join(lines).replace('"my_capture.npz"', repr(str(path)))
    src = src.replace("(10, 80), (230, 80)", "(2, 8), (25, 8)").replace('"x": 120, "y": 80', '"x": 12, "y": 8')
    exec(src, ns)                                                          # noqa: S102 - test of documented example
    assert ns["vol"].dtype == np.float32 and len(ns["curve"].mean) == 16


def test_ground_truth_zip_includes_block_layers(cap):
    import zipfile
    payload, _, _ = exports.build_export(cap, "id", "x", "ground_truth_zip")
    zf = zipfile.ZipFile(io.BytesIO(payload))
    assert "layer_thickness:cover.npy" in zf.namelist() and "origin_depth_mm.npy" in zf.namelist()
    t = np.load(io.BytesIO(zf.read("layer_thickness:cover.npy")))
    np.testing.assert_array_equal(t, cap.ground_truth["layer_thickness:cover"])


def test_user_guide_notebook_snippet_runs(tmp_path):
    """The ``load_npz_capture`` snippet of docs/studio/USER_GUIDE.md, executed verbatim on a fake export."""
    import re
    from pathlib import Path
    md = (Path(__file__).resolve().parents[2] / "docs" / "studio" / "USER_GUIDE.md").read_text()
    m = re.search(r"<!-- notebook-snippet -->\s*```python\n(.*?)```", md, re.S)
    assert m, "snippet marker not found in USER_GUIDE.md"
    cap = fake_block_result()
    for e in cap.excitations_nm:
        cap.ground_truth[f"irradiance@{e:g}"] = np.full((H, W), 2.0)
    (tmp_path / "my_capture.npz").write_bytes(exports.to_npz(cap))
    ns: dict = {}
    code = m.group(1).replace('"my_capture.npz"', repr(str(tmp_path / "my_capture.npz")))
    exec(compile(code, "USER_GUIDE snippet", "exec"), ns)
    assert len(ns["curve"].bin_centres) > 0 and np.isfinite(ns["curve"].mean).any()
