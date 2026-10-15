"""Spectral unmixing (v1.2 W4): exact recovery of synthetic linear mixtures, constraints, unsupervised endmembers,
and agreement with the simulator's ground truth on a 2-dye layered capture."""
from __future__ import annotations

import numpy as np
import pytest

from spectraforge.studio import unmix as um
from spectraforge.studio.surrogate import ChannelTable, NoiseModel


def _gauss(c, w, x):
    return np.exp(-0.5 * ((x - c) / w) ** 2)


def _synthetic(H=24, W=30, K=3, noise=0.0, seed=0, pure=True):
    rng = np.random.default_rng(seed)
    ex = np.repeat([365.0, 405.0, 430.0], 12)
    em = np.tile(450.0 + 10.0 * np.arange(12), 3)
    tab = ChannelTable(ex, em)
    centres = [480.0, 540.0, 590.0][:K]
    E = np.stack([_gauss(c, 25.0, em) * (1.0 + 0.5 * (ex == 405.0) * (k == 1)) for k, c in enumerate(centres)], axis=1)
    E = E / E.sum(axis=0)
    A = rng.dirichlet(np.ones(K), size=H * W) * rng.uniform(200, 2000, size=(H * W, 1))
    if pure:
        for k in range(K):
            A[k] = 0.0
            A[k, k] = 1000.0
    Y = A @ E.T + rng.normal(0, noise, size=(H * W, tab.n)) if noise else A @ E.T
    em_obj = um.Endmembers([f"e{k}" for k in range(K)], E, tab, "test", "electrons")
    return Y.reshape(H, W, -1), A.reshape(H, W, K), em_obj, tab


@pytest.mark.parametrize("method", ["ls", "nnls"])
def test_perfect_recovery_on_noise_free_linear_mixture(method):
    Y, A, em, _ = _synthetic()
    r = um.unmix(Y, em, method, mask=np.ones(Y.shape[:2], bool))
    assert np.max(np.abs(r.abundances - A)) < 1e-6 * A.max()
    assert np.nanmax(r.rmse) < 1e-6 * Y.max()


def test_fcls_fractions_sum_to_one_and_recover_fractions():
    Y, A, em, _ = _synthetic()
    r = um.unmix(Y, em, "fcls", mask=np.ones(Y.shape[:2], bool))
    f = A / A.sum(axis=2, keepdims=True)
    fr = r.fractions()
    assert np.allclose(fr.sum(axis=2), 1.0, atol=1e-4)
    assert np.max(np.abs(fr - f)) < 2e-3                         # delta-augmentation tolerance
    assert np.all(r.abundances >= -1e-12)
    assert np.allclose(r.scale, A.sum(axis=2), rtol=1e-6)


def test_nnls_is_nonnegative_and_beats_ls_on_noisy_sparse_data():
    Y, A, em, _ = _synthetic(noise=0.02, pure=False, seed=3)
    m = np.ones(Y.shape[:2], bool)
    ls, nn = um.unmix(Y, em, "ls", mask=m), um.unmix(Y, em, "nnls", mask=m)
    assert nn.abundances.min() >= 0
    assert np.sqrt(np.mean((nn.abundances - A) ** 2)) <= np.sqrt(np.mean((ls.abundances - A) ** 2)) * 1.0001


def test_poisson_weighting_reduces_error_for_heteroscedastic_noise():
    rng = np.random.default_rng(1)
    Y0, A, em, tab = _synthetic(K=2, pure=False, seed=2)
    nm = NoiseModel(1.5, 8.0, np.zeros(tab.n), n_pix=1.0)
    Y = Y0 + rng.normal(size=Y0.shape) * np.sqrt(nm.variance_e(np.maximum(Y0, 0)))
    m = np.ones(Y.shape[:2], bool)
    e_u = np.sqrt(np.mean((um.unmix(Y, em, "nnls", mask=m).abundances - A) ** 2))
    e_w = np.sqrt(np.mean((um.unmix(Y, em, "nnls", mask=m, weighting="poisson", noise=nm).abundances - A) ** 2))
    assert e_w < e_u


def test_joint_eem_beats_single_excitation():
    """Endmembers that overlap in one excitation but differ in another: joint unmixing is more accurate."""
    rng = np.random.default_rng(0)
    H, W = 20, 20
    ex = np.repeat([365.0, 430.0], 10)
    em = np.tile(450.0 + 10.0 * np.arange(10), 2)
    tab = ChannelTable(ex, em)
    a = _gauss(520, 30, em) * np.where(ex == 365.0, 1.0, 0.2)
    b = _gauss(530, 30, em) * np.where(ex == 365.0, 1.0, 1.0)         # identical at 365 up to a small shift
    E = np.stack([a / a.sum(), b / b.sum()], axis=1)
    A = rng.uniform(0, 1000, size=(H * W, 2))
    Y = (A @ E.T + rng.normal(0, 3.0, size=(H * W, tab.n))).reshape(H, W, -1)
    em_obj = um.Endmembers(["a", "b"], E, tab, "test", "electrons")
    m = np.ones((H, W), bool)
    joint = um.unmix(Y, em_obj, "nnls", mask=m)
    one = um.unmix(Y[..., :10], um._subset_endmembers(em_obj, ex == 365.0), "nnls", mask=m)
    err = lambda r: np.sqrt(np.mean((r.abundances.reshape(-1, 2) - A) ** 2))  # noqa: E731
    assert err(joint) < 0.5 * err(one)


def test_vca_and_nmf_recover_endmembers_when_pure_pixels_exist():
    Y, A, em, tab = _synthetic(K=3, noise=0.5, seed=5, pure=True)
    for fn in (um.endmembers_vca, um.endmembers_nmf):
        est = fn(Y, tab, 3, seed=0)
        idx, ang = um.match_endmembers(est.matrix, em.matrix)
        assert np.max(ang) < 6.0, (fn.__name__, ang)
        # unsupervised endmembers + NNLS reproduce the data
        r = um.unmix(Y, est, "nnls", mask=np.ones(Y.shape[:2], bool))
        assert np.nanmean(r.residual_fraction) < 0.05


def test_spectral_angle_and_matching():
    a = np.array([1.0, 0.0, 0.0])
    assert um.spectral_angle_deg(a, a) == pytest.approx(0.0, abs=1e-6)
    assert um.spectral_angle_deg(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(90.0)
    ref = np.eye(3)
    est = ref[:, [2, 0, 1]]
    idx, ang = um.match_endmembers(est, ref)
    assert list(idx) == [1, 2, 0] and np.allclose(ang, 0.0, atol=1e-6)


def test_forward_model_endmembers_use_components_with_fake_explain():
    """Endmembers from explain_pixel components: purest pixels, electron conversion, unit sum."""
    tab = ChannelTable(np.repeat([365.0], 6), 450.0 + 10 * np.arange(6))
    cube = np.ones((4, 5, 6), np.float32)
    from spectraforge.studio.contracts import CaptureResult
    from spectraforge.studio.presets import preset
    exp = preset("tissue_under_collagen")
    exp = exp.model_copy(deep=True)
    res = CaptureResult(experiment_json=exp.model_dump_json(), mode="sample", width=5, height=4, excitations_nm=[365.0],
                        emission_nm={365.0: tab.em_nm}, adu={365.0: cube}, radiance={365.0: cube},
                        electrons_mean={365.0: cube * 100}, exposure_ms={365.0: 250.0}, power_mW={365.0: 2.6},
                        ground_truth={"object": np.zeros((4, 5))})
    shapes = {"dye:A": _gauss(480, 20, tab.em_nm), "dye:B@top": _gauss(520, 20, tab.em_nm), "elastic": np.full(6, 0.3)}
    cam = exp.camera().sensor

    def fake_explain(e, x, y, scale=1.0, excitations=None):
        mix = {k: v * (0.2 + 0.1 * x) if k == "dye:A" else v * (1.0 + 0.1 * y) for k, v in shapes.items()}
        rad = sum(mix.values())
        e_tot = rad * 40.0                                     # 40 electrons per radiance unit
        dark = cam.dark_current_e_per_s * 0.25
        return {"x": x, "y": y, "emission_nm": {"365": list(tab.em_nm)}, "radiance": {"365": list(rad)},
                "adu": {"365": list((e_tot + dark) / cam.gain_e_per_adu + cam.offset_adu)},
                "components": {"365": {k: list(v) for k, v in mix.items()}}, "ground_truth": {}}

    em = um.endmembers_from_forward_model(exp, res, scale=1.0, pixels=[(x, y) for x in range(5) for y in range(4)],
                                          explain_fn=fake_explain, signal="electrons")
    assert em.names == ["A", "B", "scatter"] and em.matrix.shape == (6, 3)
    assert np.allclose(em.matrix.sum(axis=0), 1.0)
    A = shapes["dye:A"]
    assert um.spectral_angle_deg(em.matrix[:, 0], A) < 1e-6      # pure component shape recovered exactly
    gt = um.gt_component_amounts(exp, res, em, [(4, 0)], explain_fn=fake_explain)
    assert gt[0, 0] == pytest.approx((A * (0.2 + 0.4) * 40.0).sum(), rel=1e-6)


def test_compare_with_gt_statistics():
    rng = np.random.default_rng(0)
    g = rng.uniform(0, 100, size=(200, 2))
    est = g * np.array([1.0, 0.5]) + rng.normal(0, 1, size=g.shape)
    c = um.compare_with_gt(est, g, ["a", "b"])
    assert c["a"]["pearson_r"] > 0.99 and c["b"]["slope"] == pytest.approx(0.5, abs=0.02)


def _two_dye_experiment():
    from spectraforge.studio.presets import preset
    from spectraforge.studio.schema import Experiment
    d = preset("tissue_under_collagen").model_dump(mode="json")
    for m in d["materials"]:
        if m["id"] == "tissue_fluor_layer":
            m["dyes"] = [{"fluorophore": "NADH", "concentration_uM": 120.0}]
        if m["id"] == "collagen_cover":
            m["dyes"] = [{"fluorophore": "collagen", "concentration_uM": 600.0}]
        if m["id"] == "tissue_substrate":
            m["dyes"] = []
    d["sources"][0]["sequence"] = [dict(s, exposure_ms=s["exposure_ms"] * 8) for s in d["sources"][0]["sequence"]
                                   if s["wavelength_nm"] in (340.0, 365.0, 385.0, 415.0)]
    return Experiment.model_validate(d)


def test_two_dye_layered_capture_abundance_correlates_with_ground_truth():
    """NADH (tissue layer) + collagen (cover) under a wedge, high SNR: joint-EEM NNLS abundances correlate with the
    explain_pixel ground truth at r > 0.95 for both dyes."""
    from spectraforge.studio.capture import capture
    exp = _two_dye_experiment()
    res = capture(exp, scale=0.25)
    em = um.endmembers_from_forward_model(exp, res, scale=0.25, n_candidates=24)
    assert set(em.names) == {"NADH", "collagen", "scatter"}
    om = um.object_mask(res)
    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(om)
    pick = rng.choice(xs.size, 50, replace=False)
    px = [(int(xs[i]), int(ys[i])) for i in pick]
    gt = um.gt_component_amounts(exp, res, em, px, scale=0.25)
    for method in ("nnls", "fcls"):
        r = um.unmix(res, em, method, mask=om)
        est = np.array([r.abundances[y, x] for x, y in px])
        cmp = um.compare_with_gt(est, gt, em.names)
        assert cmp["NADH"]["pearson_r"] > 0.95 and cmp["collagen"]["pearson_r"] > 0.95, (method, cmp)
