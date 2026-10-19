import numpy as np
import pytest

from me_attention import data as D
from tests.me_attention.conftest import GRIDS, write_dataset


def test_wavelength_alignment_gaps_and_offset(syn_dir):
    d = D.load("collagen_acetic", data_root=syn_dir, min_specimen_pixels=10)
    union = sorted({w for g in GRIDS.values() for w in g})
    assert d.emissions.tolist() == union                       # real nm axis (5-nm offset grid included)
    assert d.excitations.tolist() == sorted(GRIDS)
    assert d.n_features == sum(len(g) for g in GRIDS.values())
    # every feature value encodes its own (ex, em): proves lookup by wavelength, not index
    assert np.allclose(d.X, d.ex_nm * 1000 + d.em_nm)
    assert (d.emissions[d.grid_index[:, 1]] == d.em_nm).all()
    assert (d.excitations[d.grid_index[:, 0]] == d.ex_nm).all()
    # sorted by (ex, em)
    key = d.ex_nm * 1e4 + d.em_nm
    assert (np.diff(key) > 0).all()


def test_specimens_and_dropped(syn_dir):
    d = D.load("collagen_acetic", data_root=syn_dir, min_specimen_pixels=10)
    assert d.specimen_classes.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert d.info["dropped_px"] == 1 and d.info["dropped_components"] == 1
    assert len(d.y) == 9 * 16
    assert (d.specimen_map[d.coords[:, 0], d.coords[:, 1]] == d.groups).all()


def test_to_grid(syn_dir):
    d = D.load("collagen_acetic", data_root=syn_dir, min_specimen_pixels=10)
    grid, mask = D.to_grid(d.X, d)
    assert grid.shape == (len(d.y), 3, len(d.emissions)) and mask.sum() == d.n_features
    assert (grid[:, ~mask] == 0).all()
    e, m = d.grid_index[5]
    assert np.allclose(grid[:, e, m], d.X[:, 5])


def test_missing_wavelengths_raises(tmp_path):
    root = tmp_path / "x"
    root.mkdir()
    write_dataset(root, with_wl=False)
    with pytest.raises(ValueError, match="wavelength"):
        D.load("collagen_acetic", data_root=root, min_specimen_pixels=10)


def test_cube_mask_shape_mismatch_raises(tmp_path):
    import pickle
    root = tmp_path / "y"
    root.mkdir()
    d = write_dataset(root)
    p = d / "spectra_masked.pkl"
    raw = pickle.load(open(p, "rb"))
    k = str(float(310.0))
    raw["data"][k]["cube"] = raw["data"][k]["cube"][:-1]
    raw.pop("mask")
    pickle.dump(raw, open(p, "wb"))
    with pytest.raises(ValueError, match="mask"):
        D.load("collagen_acetic", data_root=root, min_specimen_pixels=10)


# ---- Gevorg's fixed loader
def test_legacy_loader_aligns_by_wavelength(syn_dir):
    from spectral_select.me_smartattention.data import load_dataset
    r = load_dataset(syn_dir / "Collagen_Acetic_Acid", min_specimen_pixels=10)
    union = sorted({w for g in GRIDS.values() for w in g})
    assert r["emissions"].tolist() == union
    for e, (ex, wl) in enumerate(GRIDS.items()):
        pos = [union.index(w) for w in wl]
        assert np.allclose(r["X_eem"][:, e, pos], ex * 1000 + np.asarray(wl))
        assert r["eem_mask"][e].sum() == len(wl)


def test_legacy_loader_missing_wavelengths_raises(tmp_path):
    from spectral_select.me_smartattention.data import load_dataset
    root = tmp_path / "z"
    root.mkdir()
    d = write_dataset(root, with_wl=False)
    with pytest.raises(ValueError):
        load_dataset(d, min_specimen_pixels=10)


@pytest.mark.slow
def test_real_collagen_acetic_smoke():
    d = D.load("collagen_acetic")
    assert len(d.emissions) == 31 and d.emissions[0] == 420 and d.emissions[-1] == 720
    assert d.n_features == 149 and d.n_classes == 3
