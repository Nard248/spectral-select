"""Runtime-stability tests: numba threading policy and concurrent entry into the parallel kernels.

Background (see ``spectraforge.studio._numba_env`` and DEVELOPER_GUIDE "Troubleshooting"): numba's
``workqueue`` layer is not thread-safe and aborts the process on concurrent entry; the server runs
previews in a thread pool while the job worker runs a capture, so this must be safe."""
from __future__ import annotations

import os
import threading

import numpy as np
import numba
import pytest

from spectraforge.studio import _numba_env, block_capture, capture
from spectraforge.studio.presets import PRESETS

from tests.studio.test_block_capture import _small, _steps


def _wedge_exp():
    return _steps(_small(PRESETS["layered_wedge_phantom"](), photons=40_000, nodes=60.0, cell=1.5), 2)


def _launched() -> bool:
    try:
        numba.threading_layer()
        return True
    except ValueError:
        return False


def _run_threads(fns, timeout=240):
    out, errs = [None] * len(fns), []

    def wrap(i, f):
        try:
            out[i] = f()
        except BaseException as e:   # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=wrap, args=(i, f)) for i, f in enumerate(fns)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout)
    assert not any(t.is_alive() for t in ts), "deadlock: a worker thread never finished"
    assert not errs, errs
    return out


def test_layer_policy_never_picks_omp_automatically():
    if os.environ.get("NUMBA_THREADING_LAYER"):
        pytest.skip("layer forced by the user")
    from spectraforge.studio import geometry_nb  # noqa: F401  (declares the parallel kernels)
    prio = list(numba.config.THREADING_LAYER_PRIORITY)
    assert prio.index("workqueue") < prio.index("omp")
    assert prio[0] in ("tbb", "workqueue")
    assert numba.threading_layer() in ("tbb", "workqueue") if _launched() else True


def test_parallel_njit_serialises_concurrent_entry():
    @_numba_env.parallel_njit()
    def kern(a):
        s = 0.0
        for i in numba.prange(a.shape[0]):
            s += a[i]
        return s

    a = np.ones(1000)
    res = _run_threads([lambda: [kern(a) for _ in range(200)] for _ in range(4)])
    assert all(r == [1000.0] * 200 for r in res)


def test_two_block_previews_run_concurrently_and_match_serial():
    """Two threads, two different excitations, real 3D Monte Carlo (parallel kernels) -- must not abort and
    must reproduce the serial result bit-for-bit (determinism is independent of scheduling)."""
    exp = _wedge_exp()
    exs = [s.wavelength_nm for s in exp.source().sequence if s.enabled][:2]
    assert len(exs) == 2

    def preview(ex):
        return capture.preview(exp, excitation_nm=ex)

    block_capture.clear_cache()
    serial = [preview(ex) for ex in exs]
    block_capture.clear_cache()
    conc = _run_threads([lambda ex=ex: preview(ex) for ex in exs])
    for s, c in zip(serial, conc):
        assert s.excitations_nm == c.excitations_nm
        for ex in s.adu:
            np.testing.assert_array_equal(s.adu[ex], c.adu[ex])


def test_preview_while_capture_runs():
    """The server pattern: a capture on the job worker while previews arrive on the request thread pool."""
    exp = _wedge_exp()
    block_capture.clear_cache()
    res = _run_threads([lambda: capture.capture(exp, scale=0.25),
                        lambda: capture.preview(exp), lambda: capture.preview(exp)])
    assert all(r is not None and np.all(np.isfinite(next(iter(r.adu.values())))) for r in res)
