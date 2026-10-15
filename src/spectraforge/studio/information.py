r"""Fisher information, Cramer-Rao bounds and optimal acquisition design for the simulated ME-HSI instrument (v1.2, W4).

Question answered: *how well can a parameter theta (cover thickness, dye concentration, ...) be estimated from a
given camera, and which (excitation, emission band) channels / exposure times should be acquired?*

Model
-----
The data are the ROI-mean signals ``y_c`` of ``C`` channels (excitation i, emission band j).  Each is
independent with mean ``mu_c(theta)`` (noise-free expected signal electrons from the simulator) and the
Poisson-Gaussian variance of :class:`~spectraforge.studio.surrogate.NoiseModel`,
``var_c = [(mu_c + D_c) + sigma_r^2 + g^2/12] / n``.  The Fisher information matrix is

    F_ij = sum_c  (d mu_c / d theta_i)(d mu_c / d theta_j) / var_c(mu_c)

(Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*, Prentice Hall 1993, eq. 3.31 for
Gaussian data with parameter-independent covariance; it is exactly the Poisson information ``mu'^2/mu`` when the
read noise vanishes and exactly the Gaussian one when the signal is read-noise limited.  The term of the
Gaussian information that comes from the dependence of the *variance* on theta, ``(1/2)(var'/var)^2``, is **not**
included: it does not exist for the Poisson part and is negligible at the photon counts of interest.)  The
Cramer-Rao lower bound on the covariance of any unbiased estimator is ``F^{-1}`` (Rao 1945, Bull. Calcutta Math.
Soc. 37, 81; Cramer 1946).  Finite-difference Jacobians use *common random numbers* (identical MC seed), so the
differences are smooth; the Monte Carlo noise left in a difference is removed by building a
:class:`~spectraforge.studio.surrogate.Surrogate` with ``smooth='gcv'`` (:func:`fisher_from_surrogate`).

Optimal design
--------------
Because the channels are independent, F is a sum of per-channel matrices ``I_c = J_c J_c^T / var_c``.  Choosing the
best ``k`` channels is the classical optimal design problem (Pukelsheim, *Optimal Design of Experiments*, SIAM 2006):

* **D-optimality** maximises ``log det F`` (volume of the confidence ellipsoid); **A-optimality** minimises
  ``tr F^{-1}``; **target** minimises ``[F^{-1}]_{pp}``, the CRLB of one parameter (c-optimality).
* :func:`optimal_design` selects channels greedily (``log det(F + eps I)`` is monotone and submodular, so the
  greedy set is within ``1 - 1/e`` of the optimum, Nemhauser et al., Math. Prog. 14, 265, 1978) and checks it
  against an exhaustive search when ``C choose k`` is small.
* :func:`allocate_exposure` distributes a total exposure-time budget over the selected channels.  Information vs
  exposure time ``t`` of a channel is exactly ``I_c(t) = J_c J_c^T (t/t0)^2 / var_c(t)`` with
  ``var_c(t) = [shot (S + D)(t/t0) + sigma_r^2 + g^2/12]/n``; in the shot-limited regime this is linear in t and the
  problem is convex (maximise ``log det sum_c t_c F_c``, Pukelsheim ch. 7); with read noise it is solved by SLSQP
  from the shot-limit solution.  Per-channel exposure is limited by the full well.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from .schema import Experiment
from .surrogate import (ChannelTable, InverseError, NoiseModel, ParamSpec, Surrogate, apply_params, coerce_params,
                        simulate_channels)

#: relative ridge added to F inside log-det / inverse criteria so that k < P channels are comparable
RIDGE = 1e-9


# --------------------------------------------------------------------------------------------
# core algebra
# --------------------------------------------------------------------------------------------
def channel_information(jac: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-channel information matrices ``(C, P, P)``: ``J_c J_c^T / var_c``."""
    jac, var = np.asarray(jac, float), np.asarray(var, float)
    if jac.shape[0] != var.shape[0]:
        raise InverseError("jacobian and variance disagree on the number of channels")
    return jac[:, :, None] * jac[:, None, :] / var[:, None, None]


def fisher_matrix(jac: np.ndarray, var: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``F = J^T diag(1/var) J`` (optionally over the channels ``mask`` only)."""
    jac, var = np.asarray(jac, float), np.asarray(var, float)
    if mask is not None:
        jac, var = jac[mask], var[mask]
    return jac.T @ (jac / var[:, None])


def crlb_covariance(F: np.ndarray, rcond: float = 1e-12) -> tuple[np.ndarray, bool]:
    """``(F^{-1}, identifiable)``.  When F is numerically singular (smallest eigenvalue below ``rcond`` times the
    largest) the covariance bound is infinite along the null directions: the diagonal of the result is ``inf`` for
    every parameter that has weight there, and ``identifiable`` is False."""
    F = np.atleast_2d(np.asarray(F, float))
    F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)
    wmax = max(float(w.max()), 0.0)
    if wmax <= 0:
        return np.full(F.shape, np.inf), False
    ok = w > rcond * wmax
    if ok.all():
        return (V / w) @ V.T, True
    P = F.shape[0]
    cov = np.full((P, P), np.nan)
    sd2 = np.zeros(P)
    for i in range(P):
        # variance along coordinate i is infinite iff e_i has a component in the null space
        sd2[i] = np.inf if np.any(V[i, ~ok] ** 2 > 1e-12) else float((V[i, ok] ** 2 / w[ok]).sum())
    cov[np.diag_indices(P)] = sd2
    return cov, False


@dataclass
class FisherResult:
    """Fisher information of an experiment for parameters ``names`` at ``theta`` over the channel set ``channels``."""

    names: list[str]
    theta: np.ndarray                    # (P,)
    channels: ChannelTable
    mean_e: np.ndarray                   # (C,) noise-free signal electrons
    jac_e: np.ndarray                    # (C, P) d mean_e / d theta
    var_e: np.ndarray                    # (C,) variance of the ROI mean, e-^2
    fisher: np.ndarray                   # (P, P)
    covariance: np.ndarray               # (P, P) CRLB
    identifiable: bool
    noise: NoiseModel | None = None
    exposure_s: np.ndarray | None = None  # (C,) per-channel reference exposure
    channel_info: np.ndarray | None = field(default=None, repr=False)   # (C, P, P)
    meta: dict = field(default_factory=dict)

    @property
    def sd(self) -> np.ndarray:
        """CRLB standard deviation per parameter."""
        return np.sqrt(np.diag(self.covariance))

    @property
    def rel_sd(self) -> np.ndarray:
        return self.sd / np.maximum(np.abs(self.theta), 1e-300)

    def info_per_channel(self) -> np.ndarray:
        """Diagonal information ``J_cp^2 / var_c`` per channel and parameter, ``(C, P)``."""
        return self.jac_e ** 2 / self.var_e[:, None]

    def subset(self, mask: np.ndarray) -> "FisherResult":
        mask = np.asarray(mask, bool)
        return _assemble(self.names, self.theta, ChannelTable(self.channels.ex_nm[mask], self.channels.em_nm[mask]),
                         self.mean_e[mask], self.jac_e[mask], self.var_e[mask],
                         self.noise.subset(mask) if self.noise is not None else None,
                         None if self.exposure_s is None else self.exposure_s[mask], self.meta)

    def to_dict(self, per_channel: bool = True) -> dict:
        d = {"names": self.names, "theta": self.theta.tolist(), "fisher": self.fisher.tolist(),
             "crlb_covariance": _jsonf(self.covariance), "crlb_sd": _jsonf(self.sd),
             "crlb_rel_sd": _jsonf(self.rel_sd), "identifiable": self.identifiable,
             "n_channels": int(self.channels.n), "meta": self.meta}
        if per_channel:
            d.update(channels=self.channels.to_dict(), mean_e=self.mean_e.tolist(), jac_e=self.jac_e.tolist(),
                     var_e=self.var_e.tolist(), info_per_channel=self.info_per_channel().tolist())
        return d


def _jsonf(a) -> list:
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in np.atleast_2d(a)] \
        if np.ndim(a) == 2 else [None if not np.isfinite(v) else float(v) for v in np.asarray(a)]


def _assemble(names, theta, channels, mean_e, jac_e, var_e, noise, exposure_s, meta) -> FisherResult:
    F = fisher_matrix(jac_e, var_e)
    cov, ok = crlb_covariance(F)
    return FisherResult(list(names), np.asarray(theta, float), channels, np.asarray(mean_e, float),
                        np.asarray(jac_e, float), np.asarray(var_e, float), F, cov, ok, noise, exposure_s,
                        channel_info_cached(jac_e, var_e), dict(meta or {}))


def channel_info_cached(jac, var):
    return channel_information(jac, var)


def fisher_from_jacobian(names: Sequence[str], theta, jac_e: np.ndarray, mean_e: np.ndarray, noise: NoiseModel,
                         channels: ChannelTable | None = None, exposure_s: np.ndarray | None = None,
                         meta: dict | None = None) -> FisherResult:
    """Assemble a :class:`FisherResult` from an electron-space Jacobian ``(C, P)`` and mean signal ``(C,)``."""
    mean_e = np.asarray(mean_e, float)
    ch = channels or ChannelTable(np.zeros(mean_e.size), np.arange(mean_e.size, dtype=float))
    return _assemble(names, theta, ch, mean_e, jac_e, noise.variance_e(mean_e), noise, exposure_s, meta)


# --------------------------------------------------------------------------------------------
# from a surrogate (smooth Jacobian, microseconds)
# --------------------------------------------------------------------------------------------
def fisher_from_surrogate(sur: Surrogate, theta=None, *, noise: NoiseModel | None = None, bands="all",
                          excitations="all", exposure_scale: float = 1.0,
                          model_error_rel: float | None = None) -> FisherResult:
    """Fisher information at ``theta`` from the surrogate's interpolant.

    ``noise`` defaults to the surrogate's own instrument noise model.  ``exposure_scale`` multiplies every
    exposure (signal and dark scale linearly, read noise does not).  ``model_error_rel`` (eps) overrides the noise
    model's systematic relative error: per-channel variance gains ``(eps * S_c)^2`` (see
    :class:`~spectraforge.studio.surrogate.NoiseModel`); default keeps the noise model's own value (0)."""
    nm = noise or sur.noise
    if nm is None:
        raise InverseError("this surrogate has no noise model; pass noise=")
    th = np.asarray(theta if theta is not None else sur.bounds.mean(axis=1), float).reshape(-1)
    mask = sur.channels.select(excitations, bands)
    if not mask.any():
        raise InverseError("no channel matches the band / excitation selection")
    mean = sur.predict_e(th)[mask]
    jac = sur.jac_e(th)[mask]
    nm = nm.subset(mask)
    if exposure_scale != 1.0:
        mean, jac = mean * exposure_scale, jac * exposure_scale
        nm = replace(nm, dark_e=nm.dark_e * exposure_scale)
    if model_error_rel is not None:
        nm = nm.with_model_error(model_error_rel)
    ch = ChannelTable(sur.channels.ex_nm[mask], sur.channels.em_nm[mask])
    expo = None
    if sur.meta.get("exposure_ms"):
        expo = np.array([sur.meta["exposure_ms"][f"{e:g}"] * 1e-3 for e in ch.ex_nm]) * exposure_scale
    return fisher_from_jacobian(sur.names, th, jac, mean, nm, ch, expo, {"source": "surrogate"})


# --------------------------------------------------------------------------------------------
# from the simulator (finite differences, common random numbers)
# --------------------------------------------------------------------------------------------
def fisher(base_exp: Experiment, params: Sequence, *, bands="all", excitations="all", roi: dict | None = None,
           step: float | dict | None = None, theta0: Sequence[float] | None = None, scale: float = 0.25,
           overrides: dict | None = None, n_pixels: float | None = None, scheme: str = "central",
           capture_fn: Callable | None = None, progress: Callable[[float, str], None] | None = None) -> FisherResult:
    """Fisher information / CRLB of ``params`` at ``theta0`` (default: their values in ``base_exp``) from
    finite-difference Jacobians of the noise-free ROI-mean signal of the simulator.

    ``step``: relative step (float, default 0.05 of ``|theta|``, or 0.05 absolute if theta = 0) or dict name -> absolute
    step.  ``scheme``: ``'central'`` (2P + 1 simulator runs) or ``'forward'`` (P + 1).  Every run uses the experiment's
    own seed (common random numbers).  ``bands`` / ``excitations`` select channels (see
    :meth:`ChannelTable.select`); excitations not selected are not simulated.  The Poisson-Gaussian variance per
    channel comes from the camera of the experiment, ROI averaging divides it by the number of ROI pixels
    (expressed at full detector resolution, ``n_pixels`` overrides)."""
    specs = coerce_params(params)
    base_doc = base_exp.model_dump(mode="json")
    th0 = np.array([s.reference(_with_overrides(base_doc, overrides)) for s in specs], float) if theta0 is None \
        else np.asarray(theta0, float).reshape(-1)
    if th0.size != len(specs):
        raise InverseError("theta0 must have one entry per parameter")
    ex_sel = None if (isinstance(excitations, str) and excitations == "all") else [float(e) for e in excitations]
    if isinstance(step, dict):
        h = np.array([float(step[s.name]) for s in specs])
    else:
        rel = 0.05 if step is None else float(step)
        h = np.where(np.abs(th0) > 0, rel * np.abs(th0), rel)
    if np.any(h <= 0):
        raise InverseError("steps must be positive")
    n_runs = len(specs) * (2 if scheme == "central" else 1) + 1
    done = [0]

    def run(theta):
        exp = apply_params(base_doc, specs, theta, overrides)
        cm = simulate_channels(exp, roi, scale=scale, excitations=ex_sel, capture_fn=capture_fn, n_pixels=n_pixels)
        done[0] += 1
        if progress is not None:
            progress(done[0] / n_runs, f"simulator run {done[0]}/{n_runs}")
        return cm

    c0 = run(th0)
    jac = np.empty((c0.channels.n, len(specs)))
    for j in range(len(specs)):
        up = th0.copy(); up[j] += h[j]
        cu = run(up)
        if scheme == "central":
            dn = th0.copy(); dn[j] -= h[j]
            if specs[j].name and th0[j] - h[j] < 0 and th0[j] >= 0:
                dn[j] = max(th0[j] - h[j], 0.0)
            cd = run(dn)
            jac[:, j] = (cu.mean_e - cd.mean_e) / (up[j] - dn[j])
        else:
            jac[:, j] = (cu.mean_e - c0.mean_e) / h[j]
    mask = c0.channels.select("all", bands)
    if not mask.any():
        raise InverseError("no channel matches the band selection")
    expo = np.array([c0.exposure_ms[float(e)] * 1e-3 for e in c0.channels.ex_nm])
    full = fisher_from_jacobian([s.name for s in specs], th0, jac, c0.mean_e, c0.noise, c0.channels, expo,
                                {"source": "simulator", "scheme": scheme, "step": h.tolist(), "n_runs": n_runs,
                                 "scale": scale, "n_roi_pixels": c0.n_roi_pixels, "n_pix_effective": c0.noise.n_pix})
    return full if mask.all() else full.subset(mask)


def _with_overrides(doc: dict, overrides: dict | None) -> dict:
    if not overrides:
        return doc
    import copy
    from .sweeps import set_pointer
    d = copy.deepcopy(doc)
    for p, v in overrides.items():
        set_pointer(d, p, v)
    return d


# --------------------------------------------------------------------------------------------
# optimal design
# --------------------------------------------------------------------------------------------
def parse_criterion(criterion: str, names: Sequence[str]) -> tuple[str, int | None]:
    c = criterion.strip()
    if c in ("D", "A"):
        return c, None
    if c.startswith("target:"):
        nm = c.split(":", 1)[1]
        if nm not in names:
            raise InverseError(f"criterion target '{nm}' is not one of the parameters {list(names)}")
        return "target", list(names).index(nm)
    raise InverseError("criterion must be 'D', 'A' or 'target:<param>'")


def _objective(F: np.ndarray, kind: str, idx: int | None, ridge: float) -> np.ndarray:
    """Criterion value (larger is better) for a stack of information matrices ``F (..., P, P)``."""
    P = F.shape[-1]
    Fr = F + ridge * np.eye(P)
    if kind == "D":
        return np.linalg.slogdet(Fr)[1]
    inv = np.linalg.inv(Fr)
    if kind == "A":
        return -np.trace(inv, axis1=-2, axis2=-1)
    return -inv[..., idx, idx]


@dataclass
class DesignResult:
    criterion: str
    k: int
    selected: list[int]                  # channel indices in order of selection
    channels: ChannelTable               # of the selected channels, in order
    objective: list[float]               # criterion after 1..k channels
    crlb_sd: np.ndarray                  # (k, P) CRLB sd after 1..k channels
    names: list[str]
    fisher: np.ndarray                   # (P, P) of the selected set
    exhaustive: dict | None = None
    exposure: dict | None = None

    def to_dict(self) -> dict:
        return {"criterion": self.criterion, "k": self.k, "selected": self.selected,
                "channels": self.channels.to_dict(), "labels": self.channels.labels(), "objective": self.objective,
                "names": self.names, "crlb_sd_vs_k": _jsonf(self.crlb_sd), "fisher": self.fisher.tolist(),
                "exhaustive": self.exhaustive, "exposure": self.exposure}


def _sd_from(F: np.ndarray) -> np.ndarray:
    cov, _ = crlb_covariance(F)
    return np.sqrt(np.diag(cov))


def greedy_select(info: np.ndarray, k: int, kind: str, idx: int | None, candidates: np.ndarray | None = None
                  ) -> tuple[list[int], list[float]]:
    """Greedy forward selection of ``k`` of the ``C`` matrices ``info (C, P, P)``."""
    C, P, _ = info.shape
    cand = np.arange(C) if candidates is None else np.asarray(candidates, int)
    if not 1 <= k <= cand.size:
        raise InverseError(f"k must be between 1 and the number of candidate channels ({cand.size})")
    ridge = RIDGE * max(float(np.trace(info.sum(axis=0))) / P, 1e-300)
    F = np.zeros((P, P))
    chosen: list[int] = []
    path: list[float] = []
    free = list(cand)
    for _ in range(k):
        vals = _objective(F[None] + info[free], kind, idx, ridge)
        j = int(np.argmax(vals))
        chosen.append(int(free[j]))
        F = F + info[free[j]]
        free.pop(j)
        path.append(float(_objective(F, kind, idx, ridge)))
    return chosen, path


def exchange_refine(info: np.ndarray, chosen: Sequence[int], kind: str, idx: int | None,
                    candidates: np.ndarray | None = None, max_sweeps: int = 50) -> list[int]:
    """Fedorov-style exchange: swap one selected channel for one unselected channel while the criterion improves
    (Fedorov, *Theory of Optimal Experiments*, Academic Press 1972).  Greedy forward selection is exact-guaranteed
    only up to ``1 - 1/e`` for D-optimality and has no guarantee for A / target criteria, whose first greedy picks are
    ridge-dominated; the exchange removes that start-up artefact."""
    C, P, _ = info.shape
    cand = np.arange(C) if candidates is None else np.asarray(candidates, int)
    ridge = RIDGE * max(float(np.trace(info.sum(axis=0))) / P, 1e-300)
    cur = list(chosen)
    for _ in range(max_sweeps):
        improved = False
        for pos in range(len(cur)):
            rest = cur[:pos] + cur[pos + 1:]
            F0 = info[rest].sum(axis=0) if rest else np.zeros((P, P))
            pool = np.array([c for c in cand if c not in rest])
            vals = _objective(F0[None] + info[pool], kind, idx, ridge)
            j = int(np.argmax(vals))
            now = float(_objective(F0 + info[cur[pos]], kind, idx, ridge))
            if vals[j] > now + 1e-12 * max(1.0, abs(now)) and pool[j] != cur[pos]:
                cur[pos] = int(pool[j])
                improved = True
        if not improved:
            break
    return cur


def exhaustive_select(info: np.ndarray, k: int, kind: str, idx: int | None, candidates: np.ndarray | None = None,
                      max_combinations: int = 2_000_000) -> tuple[list[int], float] | None:
    """Exact optimum over all ``C choose k`` subsets, or None when that exceeds ``max_combinations``."""
    C, P, _ = info.shape
    cand = np.arange(C) if candidates is None else np.asarray(candidates, int)
    if math.comb(cand.size, k) > max_combinations:
        return None
    ridge = RIDGE * max(float(np.trace(info.sum(axis=0))) / P, 1e-300)
    best, best_set = -np.inf, None
    it = itertools.combinations(cand.tolist(), k)
    while True:
        chunk = list(itertools.islice(it, 50_000))
        if not chunk:
            break
        idxs = np.array(chunk)
        F = info[idxs].sum(axis=1)
        vals = _objective(F, kind, idx, ridge)
        j = int(np.argmax(vals))
        if vals[j] > best:
            best, best_set = float(vals[j]), chunk[j]
    return list(best_set), best


def optimal_design(fi: FisherResult, k: int, criterion: str = "D", budget: str = "channels", *,
                   candidates: np.ndarray | None = None, exhaustive_check: bool = True, refine: bool = True,
                   total_time_s: float | None = None, max_combinations: int = 2_000_000) -> DesignResult:
    """Choose the ``k`` most informative (excitation, band) channels of ``fi``.

    ``criterion``: ``'D'`` | ``'A'`` | ``'target:<param>'``.  ``budget='channels'`` (default) only selects;
    ``budget='exposure_time'`` additionally allocates ``total_time_s`` (default: the reference exposures of the
    selected channels) across them with :func:`allocate_exposure`.  Returns the ranked channels (selection order),
    the criterion and the CRLB sd after 1..k channels, and (when feasible) the exhaustive optimum for comparison."""
    kind, idx = parse_criterion(criterion, fi.names)
    if budget not in ("channels", "exposure_time"):
        raise InverseError("budget must be 'channels' or 'exposure_time'")
    info = fi.channel_info if fi.channel_info is not None else channel_information(fi.jac_e, fi.var_e)
    sel, path = greedy_select(info, k, kind, idx, candidates)
    greedy_set = list(sel)
    if refine and len(sel) < (info.shape[0] if candidates is None else len(candidates)):
        better = exchange_refine(info, sel, kind, idx, candidates)
        if set(better) != set(sel):
            sel, path = greedy_select(info, k, kind, idx, np.asarray(better))      # rank inside the refined set
    sds = np.stack([_sd_from(info[sel[:i + 1]].sum(axis=0)) for i in range(len(sel))])
    ex = None
    if exhaustive_check:
        r = exhaustive_select(info, k, kind, idx, candidates, max_combinations)
        if r is not None:
            ridge = RIDGE * max(float(np.trace(info.sum(axis=0))) / info.shape[1], 1e-300)
            g = float(_objective(info[sel].sum(axis=0), kind, idx, ridge))
            gg = float(_objective(info[greedy_set].sum(axis=0), kind, idx, ridge))
            ex = {"set": r[0], "objective": r[1], "greedy_objective": gg, "refined_objective": g,
                  "greedy_same_set": set(r[0]) == set(greedy_set), "same_set": set(r[0]) == set(sel),
                  "greedy_gap": float(r[1] - gg), "refined_gap": float(r[1] - g), "n_combinations": math.comb(
                      info.shape[0] if candidates is None else len(candidates), k)}
    res = DesignResult(criterion, k, sel, ChannelTable(fi.channels.ex_nm[sel], fi.channels.em_nm[sel]), path, sds,
                       list(fi.names), info[sel].sum(axis=0), ex)
    if budget == "exposure_time":
        res.exposure = allocate_exposure(fi, sel, criterion, total_time_s)
    return res


def information_at_exposure(fi: FisherResult, sel: Sequence[int], t_s: np.ndarray) -> np.ndarray:
    """Per-channel information matrices ``(len(sel), P, P)`` at exposure times ``t_s`` (seconds)."""
    if fi.noise is None or fi.exposure_s is None:
        raise InverseError("exposure allocation needs a noise model and reference exposures")
    sel = np.asarray(sel, int)
    t0 = fi.exposure_s[sel]
    r = np.asarray(t_s, float) / t0
    nm = fi.noise
    J = fi.jac_e[sel] * r[:, None]
    S = fi.mean_e[sel] * r
    D = nm.dark_e[sel] * r
    v = ((S + D) if nm.shot else 0.0 * S)
    v = v + (nm.read_e ** 2 if nm.read else 0.0) + (nm.gain_e_per_adu ** 2 / 12.0 if nm.quantization else 0.0)
    v = v / nm.n_pix + (nm.model_error_rel * S) ** 2
    return channel_information(J, v)


def max_exposure_s(fi: FisherResult, sel: Sequence[int], fill: float = 0.9) -> np.ndarray:
    """Exposure at which the signal + dark of each channel reaches ``fill`` of the full well."""
    sel = np.asarray(sel, int)
    s = fi.mean_e[sel] + (fi.noise.dark_e[sel] if fi.noise is not None else 0.0)
    fw = fi.noise.full_well_e if fi.noise is not None else 1e30
    return fi.exposure_s[sel] * fill * fw / np.maximum(s, 1e-12)


def allocate_exposure(fi: FisherResult, sel: Sequence[int], criterion: str = "D", total_time_s: float | None = None,
                      t_min_frac: float = 1e-3) -> dict:
    """Allocate a total exposure time over channels ``sel`` to optimise ``criterion`` (exact information model, see
    the module docstring).  Returns the optimal times, the reference (as acquired) and equal allocations, and the CRLB sd for
    each; the optimum is by construction at least as good as both (the reference allocation is rescaled to the same
    total time)."""
    from scipy.optimize import minimize
    kind, idx = parse_criterion(criterion, fi.names)
    sel = np.asarray(sel, int)
    t_ref = fi.exposure_s[sel]
    T = float(total_time_s) if total_time_s else float(t_ref.sum())
    n = sel.size
    tmax = max_exposure_s(fi, sel)
    tmin = np.full(n, t_min_frac * T / n)
    P = len(fi.names)
    ridge = RIDGE * max(float(np.trace(fi.channel_info[sel].sum(axis=0))) / P, 1e-300)

    def total_info(t):
        return information_at_exposure(fi, sel, t).sum(axis=0)

    def neg(t):
        return -float(_objective(total_info(t), kind, idx, ridge))

    def clipT(t):
        t = np.clip(t, tmin, tmax)
        return t

    t_scaled = t_ref * (T / t_ref.sum())
    equal = np.full(n, T / n)
    best_x, best_v = None, np.inf
    for x0 in (t_scaled, equal):
        x0 = clipT(x0)
        res = minimize(neg, x0, method="SLSQP", bounds=list(zip(tmin, tmax)),
                       constraints=[{"type": "eq", "fun": lambda t: t.sum() - T, "jac": lambda t: np.ones_like(t)}],
                       options={"maxiter": 300, "ftol": 1e-12})
        if res.fun < best_v and abs(res.x.sum() - T) < 1e-6 * T:
            best_x, best_v = res.x, res.fun
    if best_x is None:                       # infeasible budget (T above the saturation limit): report clipped
        best_x = clipT(t_scaled)
    t_opt = best_x

    def sd(t):
        return _sd_from(total_info(t))

    return {"total_time_s": T, "channels": ChannelTable(fi.channels.ex_nm[sel], fi.channels.em_nm[sel]).to_dict(),
            "t_opt_s": t_opt.tolist(), "t_reference_s": t_scaled.tolist(), "t_equal_s": equal.tolist(),
            "crlb_sd_opt": sd(t_opt).tolist(), "crlb_sd_reference": sd(t_scaled).tolist(),
            "crlb_sd_equal": sd(equal).tolist(), "t_max_s": tmax.tolist(), "names": list(fi.names),
            "objective_opt": -float(neg(t_opt)), "objective_reference": float(_objective(total_info(t_scaled), kind, idx, ridge)),
            "objective_equal": float(_objective(total_info(equal), kind, idx, ridge))}


def crlb_for_channels(fi: FisherResult, channel_idx: Sequence[int]) -> np.ndarray:
    """CRLB standard deviation per parameter if only ``channel_idx`` were acquired."""
    info = fi.channel_info if fi.channel_info is not None else channel_information(fi.jac_e, fi.var_e)
    return _sd_from(info[list(channel_idx)].sum(axis=0))


# --------------------------------------------------------------------------------------------
# link to the repository's unsupervised channel-selection research
# --------------------------------------------------------------------------------------------
def spectradata_from_capture(result, name: str = "sim"):
    """``spectral_select.types.SpectraData`` of a ``CaptureResult`` (offset-subtracted ADU cubes)."""
    from spectral_select.types import ExcitationData, SpectraData  # noqa: PLC0415

    from .exports import spectradata_payload
    pl = spectradata_payload(result, name)
    exc = {float(k): ExcitationData(excitation_nm=float(k), cube=v["cube"], emission_wavelengths=v["wavelengths"],
                                    exposure_time=pl["exposure_times"][k], laser_power=pl["laser_powers"][k])
           for k, v in pl["data"].items()}
    return SpectraData(excitations=exc, mask=pl["mask"], sample_name=name)


def compare_with_channel_selection(fi: FisherResult, spectra, *, k: int = 8, method: str = "spectral_ae",
                                   criterion: str = "D", seed: int = 0, epochs: int = 60, **kw) -> dict:
    """CRLB achieved by the channels chosen by one of the repo's *unsupervised* channel-selection methods versus the
    information-optimal channels and random channels.

    ``spectra``: a :class:`spectral_select.types.SpectraData` of simulated data (e.g. from
    :func:`spectraforge.studio.exports.spectradata_payload` of a capture) whose (excitation, band) grid matches the
    channels of ``fi``; ``method``: ``'spectral_ae'`` (:class:`spectral_select.architectures.SpectralAE`, per-pixel
    spectral autoencoder + perturbation influence).  Selected ``(excitation, emission)`` pairs are mapped to the nearest
    channel of ``fi``.  If the selection library (torch) is unavailable or fails, the function raises
    :class:`ImportError` / the original error, and callers (the router) report it instead of aborting."""
    info = fi.channel_info if fi.channel_info is not None else channel_information(fi.jac_e, fi.var_e)
    kind, idx = parse_criterion(criterion, fi.names)
    if method != "spectral_ae":
        raise InverseError("method must be 'spectral_ae'")
    from spectral_select.architectures.spectral_ae import SpectralAE  # noqa: PLC0415 - torch import is heavy
    model = SpectralAE(epochs=epochs, seed=seed, **kw).fit(spectra)
    picks = model.select(k)
    ch = fi.channels
    chosen: list[int] = []
    for ex, em in picks:
        d = (ch.ex_nm - ex) ** 2 + (ch.em_nm - em) ** 2
        for j in np.argsort(d):
            if int(j) not in chosen:
                chosen.append(int(j))
                break
    greedy, _ = greedy_select(info, k, kind, idx)
    rng = np.random.default_rng(seed)
    rand_sd = np.stack([_sd_from(info[rng.choice(ch.n, k, replace=False)].sum(axis=0)) for _ in range(200)])
    return {"method": method, "k": k, "criterion": criterion, "names": list(fi.names),
            "unsupervised": {"channels": [list(map(float, p)) for p in picks], "indices": chosen,
                             "crlb_sd": crlb_for_channels(fi, chosen).tolist()},
            "information_optimal": {"indices": greedy, "crlb_sd": crlb_for_channels(fi, greedy).tolist()},
            "random_median": {"crlb_sd": np.median(rand_sd, axis=0).tolist()},
            "all_channels": {"crlb_sd": fi.sd.tolist()}}
