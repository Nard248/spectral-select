r"""Spectral unmixing of simulated (or measured) multi-excitation hyperspectral captures (v1.2, W4).

Linear mixing model
-------------------
For a pixel, the net signal in the ``C`` channels (all emission bands of all excitations concatenated, the *joint EEM*)
is modelled as ``y = E a + noise`` where the columns of ``E`` (``C x K``) are the *endmembers* (spectral signatures
through the instrument) and ``a`` the abundances.  Fluorescence adds linearly in photons, so the model is exact for the
sum of independent emitters seen through the same optical path; it is approximate when the optical path differs between
emitters (a dye under a scattering cover has a filtered, shifted signature, which is the physics the forward-model
endmembers expose).  Endmembers are normalised to unit total signal, so ``a_k`` is the *signal of component k summed over all
channels* (electrons) and abundances of different components are directly comparable.

Endmember sources
-----------------
(a) **forward model** (:func:`endmembers_from_forward_model`): the simulator's per-pixel component decomposition
    (:func:`spectraforge.studio.capture.explain_pixel`, ``dye:<f>[@layer]`` components, the dye's emission through the same
    optics, LCTF bands, QE and exposure as the capture) is evaluated at candidate pixels; for each requested component
    group the *purest* pixels (largest share of the group in the total signal) provide the signature.  This is the
    in-silico equivalent of a pure-component capture through the same instrument (a single-dye material in the same
    geometry) at the cost of a handful of single-pixel evaluations instead of one full capture per dye.  Radiance
    components are converted to electrons band by band with the ratio ``electrons / radiance`` of the same pixel's total
    signal (the band-response weighting differs from the total only through the spectral slope inside a 20 nm band).
(b) **user ROIs** (:func:`endmembers_from_rois`): mean spectrum of regions drawn on the capture.
(c) **unsupervised**: non-negative matrix factorisation (:func:`endmembers_nmf`, Lee & Seung, Nature 401, 788, 1999;
    sklearn implementation) or **vertex component analysis** (:func:`endmembers_vca`, Nascimento & Dias, IEEE TGRS 43(4),
    898-910, 2005, doi:10.1109/TGRS.2005.844293).

Solvers (:func:`unmix`)
-----------------------
``ls`` unconstrained least squares; ``nnls`` non-negative (Lawson & Hanson, *Solving Least Squares Problems*, SIAM 1995,
scipy ``nnls``); ``fcls`` fully constrained (``a >= 0``, ``sum a = 1``) by the augmented-NNLS trick of Heinz & Chang, IEEE TGRS
39(3), 529-545, 2001, doi:10.1109/36.911111, applied to the pixel normalised to unit sum (the returned ``scale`` map restores
the amplitude).  ``weighting='poisson'`` weights the channels of each pixel by the inverse Poisson-Gaussian variance.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls

from .surrogate import ChannelTable, InverseError, resolve_roi

SIGNALS = ("adu_net", "electrons", "radiance")


# --------------------------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------------------------
def _experiment_of(result):
    from .schema import Experiment
    return Experiment.model_validate_json(result.experiment_json)


def capture_matrix(result, signal: str = "adu_net", excitations: Sequence[float] | None = None
                   ) -> tuple[np.ndarray, ChannelTable]:
    """Joint-EEM cube ``(H, W, C)`` of a capture and its channel table.

    ``signal``: ``'adu_net'`` raw counts converted to net signal electrons ``(ADU - offset) * gain - dark`` (what a
    measurement provides; noisy), ``'electrons'`` the noise-free expected electrons, ``'radiance'`` noise-free band radiance."""
    if signal not in SIGNALS:
        raise InverseError(f"signal must be one of {SIGNALS}")
    table = ChannelTable.from_result(result, excitations)
    exp = _experiment_of(result)
    s, tg = exp.camera().sensor, exp.acquisition.physics
    cubes = []
    for ex in dict.fromkeys(table.ex_nm.tolist()):
        if signal == "radiance":
            c = np.asarray(result.radiance[ex], np.float32)
        elif signal == "electrons":
            c = np.asarray(result.electrons_mean[ex], np.float32)
        else:
            dark = s.dark_current_e_per_s * result.exposure_ms[ex] * 1e-3 if tg.dark_current else 0.0
            c = ((np.asarray(result.adu[ex], np.float32) - s.offset_adu) * s.gain_e_per_adu - dark).astype(np.float32)
        cubes.append(c)
    return np.concatenate(cubes, axis=-1), table


def object_mask(result) -> np.ndarray:
    """Pixels that see an object (not background / empty): ground-truth ``object >= 0``."""
    g = result.ground_truth.get("object")
    if g is None:
        return np.ones((result.height, result.width), bool)
    return np.asarray(g) >= 0


# --------------------------------------------------------------------------------------------
# endmembers
# --------------------------------------------------------------------------------------------
@dataclass
class Endmembers:
    names: list[str]
    matrix: np.ndarray               # (C, K) unit-sum columns (signal units per unit abundance)
    channels: ChannelTable
    source: str
    signal: str = "adu_net"
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict:
        return {"names": self.names, "source": self.source, "signal": self.signal, "channels": self.channels.to_dict(),
                "matrix": self.matrix.T.tolist(), "meta": self.meta}


def _unit_sum(v: np.ndarray) -> np.ndarray:
    s = float(np.sum(v))
    if not s > 0:
        raise InverseError("an endmember has no signal")
    return v / s


def spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral angle mapper distance between two signatures."""
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _components_of(expl: dict, table: ChannelTable) -> dict[str, np.ndarray]:
    """Flatten ``explain_pixel`` components to ``name -> (C,)`` radiance over the channels ``table``."""
    comps: dict[str, np.ndarray] = {}
    for ex in dict.fromkeys(table.ex_nm.tolist()):
        key = f"{ex:g}"
        sel = np.abs(table.ex_nm - ex) < 1e-6
        n = int(sel.sum())
        for name, arr in (expl["components"].get(key) or {}).items():
            comps.setdefault(name, np.zeros(table.n))[sel] = np.asarray(arr, float)[:n]
    return comps


def _electron_ratio(expl: dict, table: ChannelTable, exp) -> np.ndarray:
    """Per channel electrons / radiance of the explained pixel (total signal), 0 where there is no radiance."""
    s = exp.camera().sensor
    tg = exp.acquisition.physics
    out = np.zeros(table.n)
    for ex in dict.fromkeys(table.ex_nm.tolist()):
        key = f"{ex:g}"
        sel = np.abs(table.ex_nm - ex) < 1e-6
        rad = np.asarray(expl["radiance"][key], float)
        adu = np.asarray(expl["adu"][key], float)
        dark = s.dark_current_e_per_s * _exposure_ms(exp, ex) * 1e-3 if tg.dark_current else 0.0
        e = np.maximum((adu - s.offset_adu) * s.gain_e_per_adu - dark, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(rad > 0, e / rad, 0.0)
        out[sel] = r[:int(sel.sum())]
    return out


def _exposure_ms(exp, ex: float) -> float:
    st = [s for s in exp.source().sequence if abs(s.wavelength_nm - ex) < 1e-6]
    return float(st[0].exposure_ms)


def default_groups(component_names: Sequence[str]) -> dict[str, list[str]]:
    """One group per dye (``dye:<f>`` and ``dye:<f>@*``), plus ``scatter`` (elastic remission, glint and filter
    leakage: the excitation light that reaches the bands through the LCTF tails), ``raman`` and ``room_light`` when
    those components exist.  Without the non-fluorescent groups the elastic ridge (often > 50 % of the signal in the
    bands next to the excitation) is projected onto the dye signatures."""
    names = list(component_names)
    dyes = sorted({n.split(":", 1)[1].split("@", 1)[0] for n in names if n.startswith("dye:")})
    out = {d: [f"dye:{d}", f"dye:{d}@*"] for d in dyes}
    sc = [p for p in ("elastic", "elastic@*", "glint", "glint@*", "leak_source", "leak_source@*") if
          any(fnmatch.fnmatchcase(n, p) for n in names)]
    if sc:
        out["scatter"] = sc
    if any(n.startswith("raman") for n in names):
        out["raman"] = ["raman", "raman@*"]
    if any(n.startswith("room_light") for n in names):
        out["room_light"] = ["room_light"]
    return out


def _group_sum(comps: dict[str, np.ndarray], patterns: Sequence[str], C: int) -> np.ndarray:
    out = np.zeros(C)
    for n, v in comps.items():
        if any(fnmatch.fnmatchcase(n, p) for p in patterns):
            out += v
    return out


def explain_pixels(exp, pixels: Sequence[tuple[int, int]], *, scale: float, excitations=None,
                   explain_fn: Callable | None = None) -> list[dict]:
    """``explain_pixel`` for several pixels (``explain_fn`` is injectable for tests)."""
    if explain_fn is None:
        from .capture import explain_pixel as explain_fn  # noqa: PLC0415
    return [explain_fn(exp, int(x), int(y), scale=scale, excitations=excitations) for x, y in pixels]


def endmembers_from_forward_model(exp, result, *, scale: float = 1.0, groups: dict[str, list[str]] | None = None,
                                  pixels: Sequence[tuple[int, int]] | None = None, n_candidates: int = 48,
                                  n_average: int = 3, seed: int = 0, signal: str = "adu_net",
                                  explain_fn: Callable | None = None) -> Endmembers:
    """Endmembers of the component groups ``groups`` (default: one per dye) through the capture's own instrument.

    ``pixels``: candidate pixels ``(x, y)`` in the capture's resolution (default: ``n_candidates`` random object pixels
    from the brighter half of the image, seeded).  The ``n_average`` purest candidates of each group are averaged."""
    if signal not in SIGNALS:
        raise InverseError(f"signal must be one of {SIGNALS}")
    Y, table = capture_matrix(result, "electrons")
    if pixels is None:
        om = object_mask(result)
        tot = Y.sum(axis=2)
        thr = np.median(tot[om]) if om.any() else 0.0
        ys, xs = np.nonzero(om & (tot >= thr))
        if xs.size == 0:
            raise InverseError("no object pixels to take endmembers from")
        rng = np.random.default_rng(seed)
        pick = rng.choice(xs.size, size=min(n_candidates, xs.size), replace=False)
        pixels = [(int(xs[i]), int(ys[i])) for i in pick]
    expls = explain_pixels(exp, pixels, scale=scale, explain_fn=explain_fn,
                           excitations=sorted(set(table.ex_nm.tolist())))
    comps_all = [_components_of(e, table) for e in expls]
    names_all = sorted({n for c in comps_all for n in c})
    groups = groups or default_groups(names_all)
    if not groups:
        raise InverseError("no fluorescent components found; pass groups=")
    ratios = [_electron_ratio(e, table, exp) for e in expls]
    cols, picked = [], {}
    for g, pats in groups.items():
        sigs, shares = [], []
        for comps, ratio, e in zip(comps_all, ratios, expls):
            sig_rad = _group_sum(comps, pats, table.n)
            tot_rad = sum(comps.values())
            sig = sig_rad * ratio if signal != "radiance" else sig_rad
            tot = tot_rad * ratio if signal != "radiance" else tot_rad
            sigs.append(sig)
            shares.append(float(sig.sum() / max(tot.sum(), 1e-300)))
        order = np.argsort(shares)[::-1][:max(1, n_average)]
        if shares[order[0]] <= 0:
            raise InverseError(f"group '{g}' has no signal at any candidate pixel")
        cols.append(np.mean([_unit_sum(sigs[i]) for i in order], axis=0))
        picked[g] = {"pixels": [pixels[i] for i in order], "share": [shares[i] for i in order]}
    return Endmembers(list(groups), np.stack(cols, axis=1), table, "forward_model", signal,
                      {"groups": groups, "picked": picked, "n_candidates": len(pixels)})


def endmembers_from_rois(result, rois: dict[str, dict], *, signal: str = "adu_net",
                         excitations: Sequence[float] | None = None) -> Endmembers:
    """Mean spectrum of each ROI (``{name: roi spec}``, analysis ROI dicts, ``normalized`` allowed)."""
    from . import analysis
    Y, table = capture_matrix(result, signal, excitations)
    cols = []
    for name, spec in rois.items():
        m = analysis.roi_mask(resolve_roi(spec, result.width, result.height), result.height, result.width)
        if not m.any():
            raise InverseError(f"ROI '{name}' is empty")
        cols.append(_unit_sum(np.maximum(Y[m].mean(axis=0), 0.0)))
    return Endmembers(list(rois), np.stack(cols, axis=1), table, "rois", signal)


def endmembers_nmf(Y: np.ndarray, table: ChannelTable, k: int, *, seed: int = 0, mask: np.ndarray | None = None,
                   max_iter: int = 500, signal: str = "adu_net") -> Endmembers:
    """Unsupervised endmembers by NMF of the (negative-clipped) pixel-by-channel matrix."""
    from sklearn.decomposition import NMF
    X = Y.reshape(-1, Y.shape[-1])
    if mask is not None:
        X = X[mask.reshape(-1)]
    X = np.maximum(X, 0.0)
    if X.shape[0] < k:
        raise InverseError("fewer pixels than components")
    sub = X if X.shape[0] <= 20000 else X[np.random.default_rng(seed).choice(X.shape[0], 20000, replace=False)]
    m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=max_iter)
    m.fit(sub / max(float(sub.mean()), 1e-12))
    H = np.maximum(m.components_, 0.0)
    return Endmembers([f"nmf{i}" for i in range(k)], np.stack([_unit_sum(h) for h in H], axis=1), table, "nmf", signal,
                      {"reconstruction_err": float(m.reconstruction_err_)})


def endmembers_vca(Y: np.ndarray, table: ChannelTable, k: int, *, seed: int = 0, mask: np.ndarray | None = None,
                   signal: str = "adu_net") -> Endmembers:
    """Vertex component analysis (Nascimento & Dias 2005): iteratively project the data onto a direction orthogonal to
    the span of the endmembers found so far and take the extreme pixel.  Requires pure pixels in the data."""
    X = Y.reshape(-1, Y.shape[-1])
    if mask is not None:
        X = X[mask.reshape(-1)]
    X = np.maximum(X, 0.0).T                       # (C, N)
    C, N = X.shape
    if k > min(C, N):
        raise InverseError("too many components for VCA")
    rng = np.random.default_rng(seed)
    # project onto the k-dimensional signal subspace (SVD of the uncentred data, as in the noiseless branch of VCA)
    U, _, _ = np.linalg.svd(X @ X.T / N)
    Ud = U[:, :k]
    Xp = Ud.T @ X                                   # (k, N)
    u = Xp.mean(axis=1, keepdims=True)
    Yp = Xp / np.maximum((Xp * u).sum(axis=0, keepdims=True), 1e-300)   # scale so that u . y = 1
    A = np.zeros((k, k))
    A[k - 1, 0] = 1.0
    idx = np.zeros(k, dtype=int)
    for i in range(k):
        w = rng.random(k)
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / (np.linalg.norm(f) + 1e-300)
        v = f @ Yp
        idx[i] = int(np.argmax(np.abs(v)))
        A[:, i] = Yp[:, idx[i]]
    E = X[:, idx]
    return Endmembers([f"vca{i}" for i in range(k)], np.stack([_unit_sum(E[:, i]) for i in range(k)], axis=1), table,
                      "vca", signal, {"pixel_indices": idx.tolist()})


def match_endmembers(est: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one assignment (Hungarian) of estimated to reference signatures by spectral angle.
    Returns ``(assignment, angles_deg)`` where ``assignment[i]`` is the estimated column matched to reference ``i``."""
    K1, K2 = ref.shape[1], est.shape[1]
    cost = np.array([[spectral_angle_deg(ref[:, i], est[:, j]) for j in range(K2)] for i in range(K1)])
    r, c = linear_sum_assignment(cost)
    return c, cost[r, c]


# --------------------------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------------------------
@dataclass
class UnmixResult:
    names: list[str]
    abundances: np.ndarray            # (H, W, K)
    rmse: np.ndarray                  # (H, W) residual RMS over channels (signal units)
    residual_fraction: np.ndarray     # (H, W) ||residual|| / ||y||
    scale: np.ndarray | None          # (H, W) total signal (fcls)
    method: str
    mask: np.ndarray
    channels: ChannelTable
    meta: dict = field(default_factory=dict)

    def fractions(self) -> np.ndarray:
        s = self.abundances.sum(axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, self.abundances / s, 0.0)

    def to_dict(self, with_maps: bool = True) -> dict:
        d = {"names": self.names, "method": self.method, "shape": list(self.abundances.shape[:2]),
             "mean_rmse": float(np.nanmean(self.rmse[self.mask])) if self.mask.any() else None, "meta": self.meta}
        if with_maps:
            d["abundances"] = {n: np.round(self.abundances[..., i], 4).tolist() for i, n in enumerate(self.names)}
            d["rmse"] = np.round(self.rmse, 4).tolist()
        return d


def _solve_pixels(Y: np.ndarray, E: np.ndarray, method: str, w: np.ndarray | None, weighting: str,
                  noise_var: Callable[[np.ndarray], np.ndarray] | None, progress) -> tuple[np.ndarray, np.ndarray]:
    N, C = Y.shape
    K = E.shape[1]
    A = np.zeros((N, K))
    scale = np.ones(N)
    if method == "ls" and weighting == "none" and w is None:
        A = np.linalg.lstsq(E, Y.T, rcond=None)[0].T
        return A, scale
    delta = 1e3
    for i in range(N):
        y = Y[i]
        wi = np.ones(C) if w is None else w.copy()
        if weighting == "poisson":
            # two-pass IRLS: weights from the first-pass MODEL prediction (weights taken from the noisy data itself
            # correlate with the noise and bias the fit towards low-signal channels)
            a0 = nnls(E, y, maxiter=50 * K)[0] if method != "ls" else np.linalg.lstsq(E, y, rcond=None)[0]
            wi = wi / noise_var(np.maximum(E @ a0, 0.0))
        sw = np.sqrt(wi)
        if method == "fcls":
            s = float(y.sum())
            if not s > 0:
                continue
            scale[i] = s
            yn = y / s
            Ea = np.vstack([E * sw[:, None], delta * np.ones((1, K)) * sw.mean()])
            ya = np.concatenate([yn * sw, [delta * sw.mean()]])
            A[i], _ = nnls(Ea, ya, maxiter=50 * K)
        elif method == "nnls":
            A[i], _ = nnls(E * sw[:, None], y * sw, maxiter=50 * K)
        else:
            A[i] = np.linalg.lstsq(E * sw[:, None], y * sw, rcond=None)[0]
        if progress is not None and i % 4096 == 0:
            progress(i / N, f"unmixing pixel {i}/{N}")
    return A, scale


def unmix(data, endmembers: Endmembers, method: str = "nnls", *, mask: np.ndarray | None = None,
          weighting: str = "none", noise=None, excitations: Sequence[float] | None = None,
          progress: Callable[[float, str], None] | None = None) -> UnmixResult:
    """Unmix ``data`` with ``endmembers``.

    ``data``: a ``CaptureResult`` (converted with ``endmembers.signal``) or an array ``(H, W, C)`` already in the
    endmembers' signal units with channel order ``endmembers.channels``.  ``method``: ``'ls'`` | ``'nnls'`` | ``'fcls'``.
    ``weighting='poisson'`` needs ``noise`` (a :class:`~spectraforge.studio.surrogate.NoiseModel` for these channels,
    n_pix=1).  ``mask``: only these pixels are solved (default: pixels with positive total signal)."""
    if method not in ("ls", "nnls", "fcls"):
        raise InverseError("method must be 'ls', 'nnls' or 'fcls'")
    if weighting not in ("none", "poisson"):
        raise InverseError("weighting must be 'none' or 'poisson'")
    if weighting == "poisson" and noise is None:
        raise InverseError("poisson weighting needs a noise model")
    ex_list = excitations if excitations is not None else sorted(set(endmembers.channels.ex_nm.tolist()))
    if isinstance(data, np.ndarray):
        cube, table = np.asarray(data, float), endmembers.channels
    else:
        cube, table = capture_matrix(data, endmembers.signal, ex_list)
    E = endmembers.matrix
    if cube.shape[-1] != table.n:
        raise InverseError("data and endmember channels differ")
    # restrict endmembers to the channels of the data (joint vs per-excitation unmixing)
    if table.n != E.shape[0] or not (np.allclose(table.ex_nm, endmembers.channels.ex_nm) and
                                     np.allclose(table.em_nm, endmembers.channels.em_nm)):
        E = _align(endmembers, table)
    H, W, C = cube.shape
    Y = cube.reshape(-1, C).astype(np.float64)
    if mask is None:
        mask = Y.sum(axis=1).reshape(H, W) > 0
    sel = np.flatnonzero(np.asarray(mask).reshape(-1))
    nv = noise.variance_e if noise is not None else None
    A_sel, s_sel = _solve_pixels(Y[sel], E, method, None, weighting, nv, progress)
    A = np.zeros((H * W, E.shape[1]))
    scale = np.zeros(H * W)
    A[sel], scale[sel] = A_sel, s_sel
    if method == "fcls":
        A[sel] = A_sel * s_sel[:, None]                            # back to signal units: a_k = fraction * total
    with np.errstate(all="ignore"):          # Accelerate BLAS raises spurious FP flags on matmul
        resid = Y[sel] - A[sel] @ E.T
    rm = np.full(H * W, np.nan)
    rf = np.full(H * W, np.nan)
    rm[sel] = np.sqrt(np.mean(resid ** 2, axis=1))
    rf[sel] = np.linalg.norm(resid, axis=1) / np.maximum(np.linalg.norm(Y[sel], axis=1), 1e-300)
    return UnmixResult(endmembers.names, A.reshape(H, W, -1), rm.reshape(H, W), rf.reshape(H, W),
                       scale.reshape(H, W) if method == "fcls" else None, method,
                       np.asarray(mask, bool).reshape(H, W), table,
                       {"weighting": weighting, "signal": endmembers.signal, "endmember_source": endmembers.source,
                        "n_channels": int(C), "n_solved": int(sel.size)})


def _align(em: Endmembers, table: ChannelTable) -> np.ndarray:
    """Endmember rows for the channels of ``table`` (subset / reorder of the endmembers' channels)."""
    rows = []
    for e, b in zip(table.ex_nm, table.em_nm):
        d = np.abs(em.channels.ex_nm - e) + np.abs(em.channels.em_nm - b)
        j = int(np.argmin(d))
        if d[j] > 0.5:
            raise InverseError(f"channel {e:g}->{b:g} nm is not in the endmember set")
        rows.append(j)
    return em.matrix[rows]


def unmix_per_excitation(result, endmembers: Endmembers, method: str = "nnls", **kw) -> dict[float, UnmixResult]:
    """Unmix every excitation separately (bands of that excitation only) for comparison with the joint EEM unmixing."""
    out = {}
    for ex in sorted(set(endmembers.channels.ex_nm.tolist())):
        em = _subset_endmembers(endmembers, np.abs(endmembers.channels.ex_nm - ex) < 1e-6)
        out[ex] = unmix(result, em, method, excitations=[ex], **kw)
    return out


def _subset_endmembers(em: Endmembers, mask: np.ndarray) -> Endmembers:
    return Endmembers(em.names, em.matrix[mask], ChannelTable(em.channels.ex_nm[mask], em.channels.em_nm[mask]),
                      em.source, em.signal, em.meta)


# --------------------------------------------------------------------------------------------
# comparison with the simulator's ground truth
# --------------------------------------------------------------------------------------------
def gt_component_amounts(exp, result, em: Endmembers, pixels: Sequence[tuple[int, int]], *, scale: float = 1.0,
                         groups: dict[str, list[str]] | None = None, explain_fn: Callable | None = None
                         ) -> np.ndarray:
    """Ground-truth signal of each endmember group at ``pixels`` ``(n, K)`` (same units as the abundances: signal summed
    over all channels), from ``explain_pixel`` components."""
    groups = groups or em.meta.get("groups") or {}
    table = em.channels
    expls = explain_pixels(exp, pixels, scale=scale, explain_fn=explain_fn, excitations=sorted(set(table.ex_nm.tolist())))
    out = np.zeros((len(pixels), em.k))
    for i, e in enumerate(expls):
        comps = _components_of(e, table)
        ratio = _electron_ratio(e, table, exp) if em.signal != "radiance" else np.ones(table.n)
        for k, name in enumerate(em.names):
            pats = groups.get(name)
            if pats is None:
                raise InverseError(f"no component pattern for endmember '{name}'")
            out[i, k] = float((_group_sum(comps, pats, table.n) * ratio).sum())
    return out


def compare_with_gt(est: np.ndarray, gt: np.ndarray, names: Sequence[str]) -> dict:
    """Per-endmember agreement between estimated and ground-truth amounts at the same pixels ``(n, K)``:
    Pearson r, regression slope, RMSE and relative RMSE (to the mean ground truth)."""
    out = {}
    for k, nm in enumerate(names):
        a, g = np.asarray(est)[:, k], np.asarray(gt)[:, k]
        r = float(np.corrcoef(a, g)[0, 1]) if a.std() > 1e-12 and g.std() > 1e-12 else float("nan")
        slope = float(np.polyfit(g, a, 1)[0]) if g.std() > 1e-12 else float("nan")
        rmse = float(np.sqrt(np.mean((a - g) ** 2)))
        out[nm] = {"pearson_r": r, "slope": slope, "rmse": rmse, "rel_rmse": rmse / max(float(np.mean(np.abs(g))), 1e-300),
                   "n": int(a.size)}
    return out


def compare_to_layer_fraction(res: UnmixResult, result, layer_groups: dict[str, Sequence[str]],
                              mask: np.ndarray | None = None) -> dict:
    """Agreement with the ground-truth maps ``origin_layer_fraction:<layer>`` of a layered block.

    ``layer_groups = {layer: [endmember names emitted by that layer]}``.  Both sides are renormalised over the listed
    layers (the share of the fluorescence of the listed layers that comes from each), because the unmixing also
    returns non-fluorescent endmembers (scatter) that the ground truth does not contain."""
    m = res.mask if mask is None else mask
    est, gt = {}, {}
    for layer, members in layer_groups.items():
        g = result.ground_truth.get(f"origin_layer_fraction:{layer}")
        if g is None:
            raise InverseError(f"capture has no ground truth origin_layer_fraction:{layer}")
        est[layer] = res.abundances[..., [res.names.index(n) for n in members]].sum(axis=2)
        gt[layer] = np.asarray(g, float)
    e_tot = sum(est.values())
    g_tot = sum(gt.values())
    out = {}
    for layer in layer_groups:
        ok = m & np.isfinite(g_tot) & (g_tot > 0) & (e_tot > 0)
        a, b = (est[layer] / np.where(e_tot > 0, e_tot, 1))[ok], (gt[layer] / np.where(g_tot > 0, g_tot, 1))[ok]
        out[layer] = {"pearson_r": float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else float("nan"),
                      "rmse": float(np.sqrt(np.mean((a - b) ** 2))), "n": int(ok.sum())}
    return out
