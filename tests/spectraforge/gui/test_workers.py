import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spectraforge.material import Material
from spectraforge.gui.state import ForgeState
from spectraforge.gui.workers import run_render_job


def test_run_render_job_returns_pair():
    st = ForgeState(height=10, width=10)
    layer = st.add_layer("g", Material("g", {"EGFP": 1.0}))
    layer.amount_map[:] = 1.0
    spectra, gt = run_render_job(st)
    assert spectra.n_excitations == 3
