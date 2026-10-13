import os
from pathlib import Path

import pytest

from spectraforge.studio import api
from spectraforge.studio.methods_text import methods_text
from spectraforge.studio.study_schema import StudySpec

SNAP = Path(__file__).parent / "snapshots"
SW = {"package_version": "0.0.0-test", "git": {"commit": "0123456789abcdef", "dirty": False}, "python": "3.11.0",
      "numpy": "2.1.3", "numba": {"version": "0.63.1"}, "platform": "test-platform"}


def check_snapshot(name: str, text: str):
    p = SNAP / name
    if os.environ.get("UPDATE_SNAPSHOTS") or not p.exists():
        p.write_text(text)
    assert text == p.read_text(), f"snapshot {name} differs (UPDATE_SNAPSHOTS=1 to refresh)"


def study():
    return StudySpec(name="s", base_experiment="layered_wedge_phantom", replicates=3,
                     parameters=[{"pointer": "/objects/0/shape/layers/2/thickness/thickness_mm", "label": "cover_mm",
                                  "low": 0.1, "high": 1.0},
                                 {"pointer": "/acquisition/preview_scale", "label": "scale", "kind": "choice",
                                  "values": [0.2, 0.3]}],
                     design={"kind": "saltelli", "n": 64, "seed": 2},
                     metrics=[{"name": "signal", "kind": "band_sum"}, {"name": "ratio", "kind": "band_ratio",
                              "a": {"ex": 340, "band_nm": 500}, "b": {"ex": 340, "band_nm": 600}}])


def test_markdown_snapshot_wedge():
    check_snapshot("methods_layered_wedge.md", methods_text(api.preset("layered_wedge_phantom"), software=SW))


def test_latex_snapshot_with_study():
    check_snapshot("methods_tissue_study.tex",
                   methods_text(api.preset("tissue_under_collagen"), study(), style="latex", software=SW))


def test_content_facts():
    exp = api.preset("layered_wedge_phantom")
    t = methods_text(exp, study(), software=SW)
    for needle in ("0.0.0-test", "0123456789", "intralipid_cover", "fluorescein 5 µM", "340 nm", "indirect_bounce",
                   "2,000,000 excitation", "seeded with 0", "Saltelli", "N(k + 2) = 256", "Jansen", "WHISPERS",
                   exp.camera().name):
        assert needle in t, needle
    assert "μa" in t and "μs'" in t
    assert methods_text(exp, software=SW) == methods_text(exp, software=SW)            # deterministic


def test_off_toggles_listed():
    exp = api.set(api.preset("layered_wedge_phantom"), "/acquisition/physics/raman", False)
    exp = api.set(exp, "/acquisition/physics/shot_noise", False)
    t = methods_text(exp, software=SW)
    assert "`raman`" in t and "`shot_noise`" in t
    exp2 = api.set_many(exp, {f"/acquisition/physics/{k}": True for k in ("raman", "shot_noise", "indirect_bounce")})
    assert "All physics modules" in methods_text(exp2, software=SW)


def test_optics_values_match_engine():
    import numpy as np
    from spectraforge.studio.optics import material_optics, merged_fluorophores
    exp = api.preset("layered_wedge_phantom")
    o = material_optics(exp.material("intralipid_cover"), merged_fluorophores(exp), toggles=exp.acquisition.physics)
    musp = float(np.interp(340, o.wavelengths_nm, o.mus_per_mm)) * (1 - o.g)
    t = methods_text(exp, software=SW)
    row = next(r for r in t.splitlines() if r.startswith("| intralipid_cover"))
    assert f"{musp:.3g}" in row and f"{o.n:.3g}" in row


def test_latex_escapes_and_bad_style():
    t = methods_text(api.preset("layered_wedge_phantom"), style="latex", software=SW)
    assert r"black\_anodized" in t and r"$\mu_a$" in t and "_anodized" not in t.replace(r"\_", "")
    with pytest.raises(ValueError):
        methods_text(api.preset("layered_wedge_phantom"), style="html")
