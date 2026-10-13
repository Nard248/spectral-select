r"""Maximum-likelihood parameter retrieval from simulated (or measured) ME-HSI data using a surrogate (v1.2, W4).

Estimator
---------
Given the ROI-mean signals ``y_c`` (electrons) of all channels and a :class:`~spectraforge.studio.surrogate.Surrogate`
``mu_c(theta)``, the estimator minimises the weighted least-squares criterion

    chi2(theta) = sum_c (y_c - mu_c(theta))^2 / var_c ,      var_c = [(mu_c + D_c) + sigma_r^2 + g^2/12] / n

(the Poisson-Gaussian variance of :class:`~spectraforge.studio.surrogate.NoiseModel`).  The weights are evaluated at the
model prediction and held fixed inside each Levenberg-Marquardt / trust-region solve (scipy ``least_squares``), then
refreshed (iteratively re-weighted least squares, 3-8 outer iterations).  The fixed point is the quasi-likelihood
estimator: it solves the same estimating equation ``J^T W (y - mu) = 0`` as the Gaussian maximum likelihood and, for
Poisson data, the exact Poisson likelihood equation (McCullagh & Nelder, *Generalized Linear Models*, 2nd ed.,
Chapman & Hall 1989, ch. 9, 10).  Its covariance is the sandwich-free ``(J^T W J)^-1`` evaluated at the estimate,
which equals the Cramer-Rao bound of :mod:`spectraforge.studio.information` when the estimator is efficient (checked in
the tests).  Starting points come from a scan of a dense surrogate table (global search), so multi-modal problems
(e.g. a signal that is non-monotonic in the parameter) do not depend on the user's guess.

Confidence intervals
--------------------
Profile likelihood (Venzon & Moolgavkar, Appl. Stat. 37, 87, 1988; Meeker & Escobar, Am. Stat. 52, 593, 1998):
``{theta_p : min_{others} chi2(theta) - chi2_min <= q_chi2(level, 1)}`` with the weights held at their best-fit
values.  Open intervals (the profile does not cross the threshold inside the surrogate's bounds) are returned as the
bound and flagged.

Thickness maps
--------------
:func:`retrieve_map` fits one parameter vector per tile of a noisy capture and compares the map with the ground truth
of the simulator (e.g. ``layer_thickness:collagen``), reporting bias and RMSE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import optimize, stats

from .surrogate import ChannelTable, InverseError, NoiseModel, Surrogate


# --------------------------------------------------------------------------------------------
# data in electrons
# --------------------------------------------------------------------------------------------
def adu_to_electrons(adu: np.ndarray, noise: NoiseModel, offset_adu: float, dark_e: np.ndarray | float = 0.0
                     ) -> np.ndarray:
    """Net signal electrons ``(adu - offset) * g - D`` (dark mean removed)."""
    return (np.asarray(adu, float) - offset_adu) * noise.gain_e_per_adu - dark_e


def observation_from_capture(result, sur: Surrogate, roi: dict | None = None, *, kind: str = "adu"
                             ) -> tuple[np.ndarray, int]:
    """ROI-mean net-signal electrons per surrogate channel from a (noisy) ``CaptureResult`` -> ``(y_e (C,), n_pixels)``.

    ``kind='adu'`` uses the raw counts (offset and dark removed), ``'electrons'`` the noise-free ``electrons_mean``."""
    from . import analysis
    from .surrogate import resolve_roi
    if sur.noise is None:
        raise InverseError("the surrogate has no noise model (custom metric)")
    mask = analysis.roi_mask(resolve_roi(roi or sur.meta.get("roi"), result.width, result.height),
                             result.height, result.width)
    n = int(mask.sum())
    if n == 0:
        raise InverseError("the ROI contains no pixels")
    exp = _experiment_of(result)
    off = exp.camera().sensor.offset_adu
    out = []
    ch = sur.channels
    for ex in dict.fromkeys(ch.ex_nm.tolist()):
        rex = _find_ex(result, ex)
        sel = np.abs(ch.ex_nm - ex) < 1e-6
        bands = ch.em_nm[sel]
        have = np.asarray(result.emission_nm[rex], float)
        cols = [int(np.argmin(np.abs(have - b))) for b in bands]
        if np.any(np.abs(have[cols] - bands) > 0.5):
            raise InverseError(f"the capture's bands at {ex:g} nm do not match the surrogate's channels")
        if kind == "electrons":
            out.append(np.asarray(result.electrons_mean[rex], float)[mask].mean(axis=0)[cols])
        else:
            cube = np.asarray(result.adu[rex], float)[mask].mean(axis=0)[cols]
            out.append(adu_to_electrons(cube, sur.noise, off, sur.noise.dark_e[sel]))
    return np.concatenate(out), n


def _find_ex(result, ex: float) -> float:
    for e in result.excitations_nm:
        if abs(float(e) - ex) < 1e-6:
            return float(e)
    raise InverseError(f"the capture has no {ex:g} nm excitation")


def _experiment_of(result):
    from .schema import Experiment
    return Experiment.model_validate_json(result.experiment_json)


def simulate_observation(mean_e: np.ndarray, noise: NoiseModel, rng: np.random.Generator,
                         n_pix: float | None = None) -> np.ndarray:
    """One noisy ROI-mean observation (net signal electrons) of the channels with expected signal ``mean_e``.

    The mean of ``n`` independent pixels of ``Poisson(S + D) + N(0, sigma_r^2) + uniform rounding`` is drawn exactly
    for the shot term (a sum of Poissons is a Poisson of ``n (S + D)``) and as a Gaussian for the read and
    quantisation terms.  ``n_pix`` defaults to ``noise.n_pix``."""
    n = float(noise.n_pix if n_pix is None else n_pix)
    S = np.maximum(np.asarray(mean_e, float), 0.0)
    if noise.shot:
        lam = n * (S + noise.dark_e)
        y = rng.poisson(lam) / n - noise.dark_e
    else:
        y = S.copy()
    extra = (noise.read_e ** 2 if noise.read else 0.0) + (noise.gain_e_per_adu ** 2 / 12.0 if noise.quantization else 0.0)
    if extra > 0:
        y = y + rng.normal(0.0, np.sqrt(extra / n), size=y.shape)
    return y


# --------------------------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------------------------
@dataclass
class FitResult:
    names: list[str]
    theta: np.ndarray
    cov: np.ndarray
    se: np.ndarray
    chi2: float
    dof: int
    success: bool
    n_iter: int
    at_bound: list[bool]
    ci: dict = field(default_factory=dict)       # name -> {"lo","hi","open_lo","open_hi","level"}
    profile: dict = field(default_factory=dict)  # name -> {"values": [...], "delta_chi2": [...]}
    mean_e: np.ndarray | None = None
    var_e: np.ndarray | None = None

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.dof, 1)

    def to_dict(self) -> dict:
        return {"names": self.names, "theta": self.theta.tolist(), "se": self.se.tolist(),
                "cov": self.cov.tolist(), "chi2": self.chi2, "dof": self.dof, "reduced_chi2": self.reduced_chi2,
                "success": self.success, "n_iter": self.n_iter, "at_bound": self.at_bound, "ci": self.ci,
                "profile": self.profile}


def _weights(noise: NoiseModel, mu: np.ndarray) -> np.ndarray:
    return 1.0 / noise.variance_e(mu)


def _solve(y, sur: Surrogate, noise: NoiseModel, x0, lo, hi, irls: int, free: np.ndarray, fixed_vals: np.ndarray,
           tol: float = 1e-7):
    """IRLS-LM over the free parameters; returns (theta_full, n_iter, success)."""
    P = sur.n_params
    span = np.where(hi > lo, hi - lo, 1.0)
    th = np.array(x0, float)
    th[~free] = fixed_vals[~free]
    ok = True
    n_it = 0
    w = _weights(noise, sur.predict_e(th))
    for it in range(irls):
        sw = np.sqrt(w)

        def full(z):
            t = th.copy()
            t[free] = z
            return t

        def fun(z):
            return (y - sur.predict_e(full(z))) * sw

        def jac(z):
            return -(sur.jac_e(full(z)))[:, free] * sw[:, None]

        z0 = np.clip(th[free], lo[free], hi[free])
        r = optimize.least_squares(fun, z0, jac=jac, bounds=(lo[free], hi[free]), x_scale=span[free],
                                   method="trf", xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
        n_it += r.nfev
        ok = ok and r.success
        new = full(r.x)
        w_new = _weights(noise, sur.predict_e(new))
        moved = np.max(np.abs(new - th) / span)
        th, w = new, w_new
        if moved < tol:
            break
    return th, n_it, ok, w


def _chi2(y, sur, theta, w) -> float:
    r = y - sur.predict_e(theta)
    return float(np.sum(w * r * r))


def fit(observation_e: np.ndarray, sur: Surrogate, noise_model: NoiseModel | None = None, x0=None, *,
        bounds=None, irls: int = 8, n_start: int = 3, profile: bool = False, level: float = 0.95,
        profile_points: int = 41, mask: np.ndarray | None = None) -> FitResult:
    """Maximum-likelihood estimate of all parameters of ``sur`` from the net signal electrons ``observation_e``
    (one value per channel; see :func:`observation_from_capture`).

    ``noise_model``: defaults to the surrogate's.  ``x0``: start (default: the best of a dense surrogate scan, refined
    from the ``n_start`` best candidates).  ``bounds``: ``[(lo, hi), ...]`` inside the surrogate's range.  ``mask``:
    use only these channels.  ``profile=True`` adds profile-likelihood confidence intervals at ``level``."""
    nm = noise_model or sur.noise
    if nm is None:
        raise InverseError("no noise model: pass noise_model=")
    y = np.asarray(observation_e, float).reshape(-1)
    if mask is not None:
        sur = sur.subset(np.asarray(mask, bool))
        nm = nm.subset(np.asarray(mask, bool)) if nm.dark_e.size == len(mask) else nm
        y = y[np.asarray(mask, bool)]
    if y.size != sur.channels.n:
        raise InverseError(f"observation has {y.size} values, the surrogate {sur.channels.n} channels")
    if nm.dark_e.size != y.size:
        raise InverseError("the noise model's dark vector does not match the channels")
    P = sur.n_params
    lo, hi = sur.bounds[:, 0].copy(), sur.bounds[:, 1].copy()
    if bounds is not None:
        b = np.asarray(bounds, float).reshape(P, 2)
        lo, hi = np.maximum(lo, b[:, 0]), np.minimum(hi, b[:, 1])
        if np.any(hi <= lo):
            raise InverseError("empty parameter bounds")
    free = np.ones(P, dtype=bool)
    fixed = np.zeros(P)
    if x0 is None:
        T, M = sur.dense_table(400 if P == 1 else 2000)
        inside = np.all((T >= lo) & (T <= hi), axis=1)
        T, M = T[inside], M[inside]
        w = 1.0 / nm.variance_e(M)
        c = np.sum(w * (y[None, :] - M) ** 2, axis=1)
        starts = T[np.argsort(c)[:max(1, n_start)]]
    else:
        starts = np.atleast_2d(np.asarray(x0, float)).reshape(-1, P)
    best = None
    for s in starts:
        th, nit, ok, w = _solve(y, sur, nm, s, lo, hi, irls, free, fixed)
        c2 = _chi2(y, sur, th, w)
        if best is None or c2 < best[0] - 1e-12:
            best = (c2, th, nit, ok, w)
    c2, th, nit, ok, w = best
    J = sur.jac_e(th)
    A = J.T @ (J * w[:, None])
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov = np.full((P, P), np.inf)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    span = np.where(hi > lo, hi - lo, 1.0)
    at_b = [bool(th[i] <= lo[i] + 1e-6 * span[i] or th[i] >= hi[i] - 1e-6 * span[i]) for i in range(P)]
    res = FitResult(sur.names, th, cov, se, c2, y.size - P, bool(ok), int(nit), at_b,
                    mean_e=sur.predict_e(th), var_e=nm.variance_e(sur.predict_e(th)))
    if profile:
        res.ci, res.profile = profile_ci(y, sur, nm, res, w, lo, hi, level, profile_points)
    return res


def fit_scaled(observation_e: np.ndarray, sur: Surrogate, noise_model: NoiseModel | None = None, x0=None, *,
               irls: int = 4, n_start: int = 3) -> FitResult:
    """Like :func:`fit` with a free multiplicative amplitude ``s``: ``y_c ~ s * mu_c(theta)``.

    The amplitude absorbs everything that scales all channels of a pixel equally and is not a property of the sample
    layers: the illumination irradiance across the field (Gaussian beam profile), pixel position, exposure drift.  It is
    eliminated analytically (variable projection, Golub & Pereyra, SIAM J. Numer. Anal. 10, 413, 1973):
    ``s(theta) = sum w y mu / sum w mu^2``.  The returned covariance is that of ``(theta, s)`` marginalised to ``theta``
    (``(J_a^T W J_a)^-1`` with ``J_a = [s J, mu]``); ``theta`` is then determined by the *spectral shape* (band and
    excitation ratios), so it needs a surrogate whose channels differ in their thickness dependence."""
    nm = noise_model or sur.noise
    y = np.asarray(observation_e, float).reshape(-1)
    if nm is None or y.size != sur.channels.n:
        raise InverseError("need a noise model and one observation per channel")
    P = sur.n_params
    lo, hi = sur.bounds[:, 0], sur.bounds[:, 1]
    span = np.where(hi > lo, hi - lo, 1.0)

    def amp(mu, w):
        return float(np.sum(w * y * mu) / max(np.sum(w * mu * mu), 1e-300))

    if x0 is None:
        T, M = sur.dense_table(400 if P == 1 else 2000)
        sc = (M * y).sum(1) / np.maximum((M * M).sum(1), 1e-300)
        w = 1.0 / nm.variance_e(np.maximum(sc[:, None] * M, 0))
        sc = (w * M * y).sum(1) / np.maximum((w * M * M).sum(1), 1e-300)
        c = np.sum(w * (y[None] - sc[:, None] * M) ** 2, axis=1)
        starts = T[np.argsort(c)[:max(1, n_start)]]
    else:
        starts = np.atleast_2d(np.asarray(x0, float)).reshape(-1, P)
    best = None
    for st in starts:
        th = np.array(st, float)
        mu = sur.predict_e(th)
        w = 1.0 / nm.variance_e(np.maximum(amp(mu, np.ones_like(mu)) * mu, 0))
        for _ in range(irls):
            sw = np.sqrt(w)

            def fun(z):
                m = sur.predict_e(z)
                return (y - amp(m, w) * m) * sw

            r = optimize.least_squares(fun, np.clip(th, lo, hi), bounds=(lo, hi), x_scale=span, xtol=1e-10, ftol=1e-10,
                                       gtol=1e-10, max_nfev=100)
            th = r.x
            m = sur.predict_e(th)
            w = 1.0 / nm.variance_e(np.maximum(amp(m, w) * m, 0))
        m = sur.predict_e(th)
        c2 = float(np.sum(w * (y - amp(m, w) * m) ** 2))
        if best is None or c2 < best[0]:
            best = (c2, th, w)
    c2, th, w = best
    m = sur.predict_e(th)
    sc = amp(m, w)
    Ja = np.hstack([sc * sur.jac_e(th), m[:, None]])
    try:
        cov_a = np.linalg.inv(Ja.T @ (Ja * w[:, None]))
    except np.linalg.LinAlgError:
        cov_a = np.full((P + 1, P + 1), np.inf)
    cov = cov_a[:P, :P]
    at_b = [bool(th[i] <= lo[i] + 1e-6 * span[i] or th[i] >= hi[i] - 1e-6 * span[i]) for i in range(P)]
    res = FitResult(sur.names, th, cov, np.sqrt(np.maximum(np.diag(cov), 0)), c2, y.size - P - 1, True, 0, at_b,
                    mean_e=sc * m, var_e=nm.variance_e(sc * m))
    res.scale = sc
    return res


def profile_ci(y, sur: Surrogate, nm: NoiseModel, res: FitResult, w: np.ndarray, lo, hi, level: float = 0.95,
               n_points: int = 41) -> tuple[dict, dict]:
    """Profile-likelihood intervals and curves for every parameter (weights ``w`` fixed at the best fit)."""
    P = sur.n_params
    q = float(stats.chi2.ppf(level, 1))
    span = np.where(hi > lo, hi - lo, 1.0)
    ci, prof = {}, {}
    chi_min = res.chi2

    def prof_chi2(j: int, v: float, warm: np.ndarray) -> tuple[float, np.ndarray]:
        if P == 1:
            t = np.array([v])
            return _chi2(y, sur, t, w), t
        free = np.ones(P, bool)
        free[j] = False
        fixed = np.zeros(P)
        fixed[j] = v
        x0 = warm.copy()
        x0[j] = v
        t, _, _, _ = _solve_fixed_w(y, sur, w, x0, lo, hi, free, fixed)
        return _chi2(y, sur, t, w), t

    for j in range(P):
        # curve over theta_j in the surrogate's range
        vals = np.linspace(lo[j], hi[j], n_points)
        vals = np.unique(np.concatenate([vals, [res.theta[j]]]))
        warm, d = res.theta.copy(), []
        order = np.argsort(np.abs(vals - res.theta[j]))        # warm-start outward from the optimum
        out = np.zeros(vals.size)
        left = {}
        for idx in order:
            c, t = prof_chi2(j, vals[idx], warm)
            out[idx] = c - chi_min
        prof[sur.names[j]] = {"values": vals.tolist(), "delta_chi2": out.tolist(), "threshold": q}

        def g(v, _j=j):
            c, _ = prof_chi2(_j, v, res.theta)
            return c - chi_min - q

        t0 = res.theta[j]
        bounds_ij = []
        for side, edge in ((-1, lo[j]), (1, hi[j])):
            if g(edge) < 0:                       # still inside the threshold at the surrogate's boundary: open
                bounds_ij.append((float(edge), True))
            else:
                try:
                    r = optimize.brentq(g, min(t0, edge), max(t0, edge), xtol=1e-10 * span[j], maxiter=100)
                except ValueError:
                    r = float("nan")
                bounds_ij.append((float(r), False))
        ci[sur.names[j]] = {"lo": bounds_ij[0][0], "hi": bounds_ij[1][0], "open_lo": bounds_ij[0][1],
                            "open_hi": bounds_ij[1][1], "level": level, "estimate": float(t0)}
    return ci, prof


def _solve_fixed_w(y, sur, w, x0, lo, hi, free, fixed):
    """LM with fixed weights over the free parameters (used by the profile)."""
    span = np.where(hi > lo, hi - lo, 1.0)
    th = np.array(x0, float)
    th[~free] = fixed[~free]
    sw = np.sqrt(w)

    def full(z):
        t = th.copy()
        t[free] = z
        return t

    r = optimize.least_squares(lambda z: (y - sur.predict_e(full(z))) * sw, np.clip(th[free], lo[free], hi[free]),
                               jac=lambda z: -(sur.jac_e(full(z)))[:, free] * sw[:, None],
                               bounds=(lo[free], hi[free]), x_scale=span[free], xtol=1e-10, ftol=1e-10, gtol=1e-10,
                               max_nfev=100)
    return full(r.x), r.nfev, r.success, w


# --------------------------------------------------------------------------------------------
# Monte Carlo efficiency check
# --------------------------------------------------------------------------------------------
def monte_carlo_fit(sur: Surrogate, theta_true, n_real: int = 200, seed: int = 0, *, noise: NoiseModel | None = None,
                    truth_fn: Callable[[np.ndarray], np.ndarray] | None = None, irls: int = 6) -> dict:
    """Fit ``n_real`` noisy realisations of the data at ``theta_true`` and compare the empirical covariance of the
    estimates with the CRLB and the fit's own covariance.

    ``truth_fn``: optional ``theta -> mean signal electrons (C,)`` used to generate the data (default: the surrogate,
    which makes the test self-consistent; pass an independent simulator evaluation to include interpolation error).
    Returns arrays of estimates and ``{'empirical_sd','crlb_sd','mean_fit_se','bias','ratio'}``."""
    from .information import fisher_from_surrogate
    nm = noise or sur.noise
    th = np.asarray(theta_true, float).reshape(-1)
    mean = truth_fn(th) if truth_fn is not None else sur.predict_e(th)
    rng = np.random.default_rng(seed)
    est = np.empty((n_real, sur.n_params))
    ses = np.empty_like(est)
    for i in range(n_real):
        y = simulate_observation(mean, nm, rng)
        r = fit(y, sur, nm, x0=th, irls=irls)
        est[i], ses[i] = r.theta, r.se
    fi = fisher_from_surrogate(sur, th, noise=nm)
    emp = est.std(axis=0, ddof=1)
    return {"estimates": est, "empirical_sd": emp, "crlb_sd": fi.sd, "mean_fit_se": ses.mean(axis=0),
            "bias": est.mean(axis=0) - th, "ratio": emp / fi.sd, "n_real": n_real}


# --------------------------------------------------------------------------------------------
# parameter maps
# --------------------------------------------------------------------------------------------
@dataclass
class MapResult:
    names: list[str]
    theta: np.ndarray              # (ty, tx, P)
    se: np.ndarray                 # (ty, tx, P)
    chi2_red: np.ndarray           # (ty, tx)
    tile: int
    gt: np.ndarray | None          # (ty, tx) tile-mean ground truth of the first parameter (or chosen)
    stats: dict
    valid: np.ndarray              # (ty, tx) bool: tile fitted

    def to_dict(self, with_maps: bool = True) -> dict:
        d = {"names": self.names, "tile": self.tile, "stats": self.stats, "shape": list(self.theta.shape[:2])}
        if with_maps:
            nan = lambda a: np.where(np.isfinite(a), a, None).tolist()  # noqa: E731
            d.update(theta=nan(self.theta), se=nan(self.se), chi2_red=nan(self.chi2_red),
                     gt=None if self.gt is None else nan(self.gt))
        return d


def retrieve_map(result, sur: Surrogate, *, tile: int = 4, gt_name: str | None = None, gt_param: int = 0,
                 min_signal_e: float = 0.0, progress: Callable[[float, str], None] | None = None,
                 kind: str = "adu", pixel_independent: bool = True, free_amplitude: bool = True) -> MapResult:
    """Per-tile parameter retrieval on a (noisy) capture, with the simulator's ground truth for validation.

    Tiles are ``tile x tile`` pixels (``tile=1``: per pixel); tiles whose mean total signal is below
    ``min_signal_e`` (e.g. background) or whose ground truth is NaN are skipped.  ``gt_name``: ground-truth map of
    the first parameter (e.g. ``'layer_thickness:collagen'``); bias and RMSE are computed over fitted tiles whose
    ground truth lies inside the surrogate's parameter range.  ``pixel_independent``: treat the capture's pixels as
    independent detector pixels (true for the simulator, whose noise is drawn per output pixel at every scale), so
    the noise variance of a tile is the per-pixel variance divided by ``tile**2``."""
    if sur.noise is None:
        raise InverseError("the surrogate has no noise model")
    H, W = result.height, result.width
    ty, tx = H // tile, W // tile
    if ty < 1 or tx < 1:
        raise InverseError("tile larger than the image")
    exp = _experiment_of(result)
    off = exp.camera().sensor.offset_adu
    ch = sur.channels
    nm1 = sur.noise.with_n_pix(float(tile * tile))
    cubes = []
    for ex in dict.fromkeys(ch.ex_nm.tolist()):
        rex = _find_ex(result, ex)
        sel = np.abs(ch.ex_nm - ex) < 1e-6
        have = np.asarray(result.emission_nm[rex], float)
        cols = [int(np.argmin(np.abs(have - b))) for b in ch.em_nm[sel]]
        src = np.asarray(result.electrons_mean[rex] if kind == "electrons" else result.adu[rex], float)[:, :, cols]
        src = src[:ty * tile, :tx * tile].reshape(ty, tile, tx, tile, -1).mean(axis=(1, 3))      # tile means
        if kind != "electrons":
            src = adu_to_electrons(src, sur.noise, off, sur.noise.dark_e[sel][None, None, :])
        cubes.append(src)
    Y = np.concatenate(cubes, axis=-1)                        # (ty, tx, C)
    gt_tile = None
    if gt_name is not None:
        g = np.asarray(result.ground_truth[gt_name], float)[:ty * tile, :tx * tile]
        with np.errstate(all="ignore"):
            gt_tile = np.nanmean(g.reshape(ty, tile, tx, tile), axis=(1, 3))
    theta = np.full((ty, tx, sur.n_params), np.nan)
    se = np.full_like(theta, np.nan)
    chi = np.full((ty, tx), np.nan)
    valid = np.zeros((ty, tx), bool)
    total = ty * tx
    prev = None
    for i in range(ty):
        for j in range(tx):
            if gt_tile is not None and not np.isfinite(gt_tile[i, j]):
                continue
            y = Y[i, j]
            if float(y.sum()) < min_signal_e:
                continue
            if free_amplitude:
                r = fit_scaled(y, sur, nm1, x0=prev) if prev is not None else fit_scaled(y, sur, nm1)
            else:
                r = fit(y, sur, nm1, x0=prev, irls=5) if prev is not None else fit(y, sur, nm1, irls=5)
            theta[i, j], se[i, j], chi[i, j] = r.theta, r.se, r.reduced_chi2
            valid[i, j] = True
            prev = r.theta
        if progress is not None:
            progress((i + 1) / ty, f"fitted row {i + 1}/{ty}")
    st: dict = {"n_tiles": int(valid.sum()), "tile": tile, "median_reduced_chi2":
                float(np.nanmedian(chi)) if valid.any() else None}
    if gt_tile is not None and valid.any():
        lo, hi = sur.bounds[gt_param]
        use = valid & np.isfinite(gt_tile) & (gt_tile >= lo) & (gt_tile <= hi)
        if use.any():
            err = theta[..., gt_param][use] - gt_tile[use]
            z = err / np.maximum(se[..., gt_param][use], 1e-30)
            st.update(n_scored=int(use.sum()), bias=float(err.mean()), rmse=float(np.sqrt(np.mean(err ** 2))),
                      sd=float(err.std()), median_abs_err=float(np.median(np.abs(err))),
                      mean_predicted_se=float(se[..., gt_param][use].mean()),
                      coverage_1p96se=float(np.mean(np.abs(z) < 1.96)),
                      pearson_r=float(np.corrcoef(theta[..., gt_param][use], gt_tile[use])[0, 1])
                      if use.sum() > 2 else None)
    return MapResult(sur.names, theta, se, chi, tile, gt_tile, st, valid)
