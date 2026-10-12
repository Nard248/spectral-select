"""Process-wide numba threading policy for the Studio engine (import this BEFORE any ``parallel=True`` kernel).

Why this module exists
----------------------
The Studio engine uses ``@njit(parallel=True)`` kernels (``geometry_nb``, ``transport_mc``,
``transport3d_mc``).  numba picks its parallel *threading layer* lazily, at the first parallel launch,
with the default priority ``tbb > omp > workqueue``.  On the supported macOS/arm64 wheels this had two
product-relevant consequences (diagnosed 2026-10, see ``docs/studio/DEVELOPER_GUIDE.md``):

* ``tbb`` is not shipped, so the layer silently depended on *import order*: numba's ``omp`` pool is linked
  against ``@rpath/libomp.dylib`` with no ``LC_RPATH``; it only loads if torch's copy of LLVM ``libomp``
  was already mapped (``spectraforge`` -> ``spectral_select`` -> torch always does this).  The process then
  holds up to three different LLVM OpenMP runtimes (torch, sklearn, skimage) plus Qt/Accelerate threads, and
  prints ``OMP: Info #276``.  Without torch the layer fell back to ``workqueue``.
* ``workqueue`` is **not thread-safe**: two Python threads entering parallel kernels concurrently (the server
  runs previews in a thread pool while the job worker runs a capture) abort the interpreter with
  ``Terminating: Concurrent access has been detected``.

Policy implemented here
-----------------------
1. **Layer**: prefer ``tbb`` (thread-safe, own runtime) if present, else ``workqueue`` (numba's own
   pthread pool: no external OpenMP runtime, so nothing to clash with torch / sklearn / PyQt6 / Accelerate,
   identical speed to ``omp`` for these coarse-grained ``prange`` loops -- measured).  ``omp`` is never
   selected automatically.  An explicit ``NUMBA_THREADING_LAYER`` set by the user is respected.
2. **Serialisation**: whatever the layer, every parallel kernel is entered under one re-entrant lock
   (:data:`PARALLEL_LOCK`, applied with :func:`parallel_njit`).  Kernels already use all cores, so
   running two at once gains nothing; the lock only makes concurrent entry from several Python threads
   (server preview + capture, Jupyter + background thread) safe and deterministic.  Kernels are launched in
   bounded chunks (``transport3d``) so a waiting preview is delayed by at most one chunk.

Nothing here hides undefined behaviour (``KMP_DUPLICATE_LIB_OK`` is deliberately NOT used).
"""
from __future__ import annotations

import functools
import os
import threading

import numba

#: Serialises entry into numba ``parallel=True`` kernels across Python threads.
PARALLEL_LOCK = threading.RLock()


def _tbb_available() -> bool:
    try:
        import numba.np.ufunc.tbbpool  # noqa: F401
        return True
    except Exception:           # ImportError or a broken TBB runtime
        return False


def configure() -> str:
    """Set numba's threading-layer priority (idempotent, a no-op once a layer is launched). Returns the policy.

    Resulting priority: ``tbb`` (only if it imports) > ``workqueue`` > ``omp``."""
    if os.environ.get("NUMBA_THREADING_LAYER"):
        return "user:" + os.environ["NUMBA_THREADING_LAYER"]
    # numba requires a permutation of all three names; "omp" is last, i.e. reached only if the always-present
    # workqueue pool failed to load.
    prio = ["tbb", "workqueue", "omp"] if _tbb_available() else ["workqueue", "tbb", "omp"]
    try:
        numba.config.THREADING_LAYER_PRIORITY = prio
    except Exception:           # pragma: no cover - older numba without the knob
        pass
    return ">".join(prio)


POLICY = configure()


def parallel_njit(*args, **kwargs):
    """``numba.njit(parallel=True, ...)`` whose calls are serialised under :data:`PARALLEL_LOCK`.

    Use as ``@parallel_njit(cache=True)``.  The compiled dispatcher stays reachable as ``.dispatcher``
    (and ``.py_func``).  Do not call a ``parallel_njit`` function from inside another jitted function."""
    def deco(fn):
        disp = numba.njit(*args, parallel=True, **kwargs)(fn)

        @functools.wraps(fn)
        def call(*a, **k):
            with PARALLEL_LOCK:
                return disp(*a, **k)

        call.dispatcher = disp
        call.py_func = fn
        return call
    return deco
