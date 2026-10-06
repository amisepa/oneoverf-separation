"""How a periodic-power result depends on the assumed coupling exponent.

Under an assumed lambda, periodic power is measured as y = ln a - lambda ln b,
with a the periodic and b the aperiodic power in a band (lambda = 0:
background subtracted in linear power, as IRASA does; lambda = 1: divided
out, as specparam does). Any linear effect on y is therefore linear in
lambda, s(lambda) = s_a - lambda s_b, where s_a and s_b are the same effect
computed on ln a and on ln b, and it changes sign at the crossover
lambda* = s_a / s_b. An effect whose crossover lies outside [0, 1] does not
depend on the assumption; one with a crossover inside [0, 1] does.

band_power()     total, aperiodic and periodic power in a band, one spectrum
peak_frequency() individual peak frequency from the flattened spectrum
effect_curve()   s_a, s_b, s(lambda) over a grid and lambda*, for a
                 regression coefficient (between subjects: y on x, plus
                 covariates) or a mean paired difference (within subjects),
                 with pairs-bootstrap intervals and the Bayesian-bootstrap
                 highest-density interval of lambda*
"""
import numpy as np

from ap_models import ap_band_power, fit_aperiodic


def fit_mask(f, censor):
    """True for frequencies used in the aperiodic fit (outside every censor window)."""
    keep = np.ones(f.size, bool)
    for lo, hi in censor:
        keep &= ~((f >= lo) & (f <= hi))
    return keep


def peak_frequency(P, f, search=(6.0, 14.0), censor=((6.0, 16.0),)):
    """Frequency of the largest positive residual above a censored log-log line."""
    keep = fit_mask(f, censor)
    X = np.column_stack([np.ones(keep.sum()), np.log10(f[keep])])
    beta = np.linalg.lstsq(X, np.log10(P[keep]), rcond=None)[0]
    resid = P - 10.0 ** (beta[0] + beta[1] * np.log10(f))
    s = (f >= search[0]) & (f <= search[1])
    if not np.any(resid[s] > 0):
        return np.nan
    return float(f[s][np.argmax(resid[s])])


def band_power(P, f, band, fit_range=(2.0, 55.0), censor=((6.0, 16.0),),
               model="fixed"):
    """Total (tot), aperiodic (b) and periodic (a = tot - b) power in a band.

    P, f: one spectrum (linear power) and its frequencies. The aperiodic
    model is fitted over fit_range with the censor windows left out, by
    Whittle deviance (ap_models.fit_aperiodic). Returns None if the fit fails.
    """
    sel = (f >= fit_range[0]) & (f <= fit_range[1])
    ff = f[sel]
    PP = np.clip(np.nan_to_num(np.asarray(P, float)[sel], nan=1e-12), 1e-12, None)
    fit = fit_aperiodic(PP, ff, fit_mask(ff, censor), model)
    if fit is None:
        return None
    lo, hi = band
    b = ap_band_power(fit, ff, lo, hi)
    tot = float(np.mean(PP[(ff >= lo) & (ff <= hi)]))
    return dict(tot=tot, b=b, a=tot - b, exponent=fit["exponent"], dev=fit["dev"])


def hdi(x, mass=0.95):
    x = np.sort(np.asarray(x)[np.isfinite(x)])
    n = int(np.floor(mass * x.size))
    i = int(np.argmin(x[n:] - x[:x.size - n]))
    return float(x[i]), float(x[i + n])


def _coef(Y, X, w=None):
    """Weighted least-squares coefficient of column 1 of X for each column of Y."""
    if w is None:
        return np.linalg.lstsq(X, Y, rcond=None)[0][1]
    WX = X * w[:, None]
    return np.linalg.solve(X.T @ WX, WX.T @ Y)[1]


def effect_curve(la, lb, x=None, covariates=None, grid=None, nboot=2000, rng=None):
    """Effect on ln a - lambda ln b as a function of lambda.

    la, lb: ln a and ln b per subject (between-subject design, with x the
    predictor of interest and optional covariates, n x k), or the paired
    differences of ln a and ln b per subject (within-subject design, x None:
    the effect is the mean difference). Rows with non-finite values are
    dropped. Returns a dict with s_a, s_b, lam_star, its pairs-bootstrap
    percentile CI and Bayesian-bootstrap HDI, the share of Bayesian draws with
    a crossover inside [0, 1], the draws themselves (lam_star_draws), and the
    curve (grid, estimate, 2.5%, 97.5%).
    s_b_ci is the pairs-bootstrap CI of s_b; when it includes 0 the crossover
    is unbounded (lam_star_bounded False) and its intervals mean nothing.
    """
    rng = rng or np.random.default_rng(0)
    grid = np.round(np.arange(0, 1.0001, 0.05), 2) if grid is None else np.asarray(grid)
    la, lb = np.asarray(la, float), np.asarray(lb, float)
    cols = [np.ones_like(la)]
    if x is not None:
        cols.append(np.asarray(x, float))
        if covariates is not None:
            C = np.asarray(covariates, float)
            cols += list(C.T if C.ndim == 2 else [C])
    X = np.column_stack(cols)
    ok = np.isfinite(la) & np.isfinite(lb) & np.all(np.isfinite(X), 1)
    la, lb, X = la[ok], lb[ok], X[ok]
    Y = np.column_stack([la, lb])
    n = la.size

    def coefs(w=None):
        if x is None:
            ww = np.ones(n) / n if w is None else w
            return ww @ Y
        return _coef(Y, X, w)

    s_a, s_b = coefs()
    boot, bayes = [], []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        if x is None:
            boot.append(Y[i].mean(0))
        else:
            boot.append(_coef(Y[i], X[i]))
        bayes.append(coefs(rng.dirichlet(np.ones(n))))
    boot, bayes = np.array(boot), np.array(bayes)
    with np.errstate(divide="ignore", invalid="ignore"):
        ls_boot = boot[:, 0] / boot[:, 1]
        ls_bayes = bayes[:, 0] / bayes[:, 1]
    curve = np.array([[s_a - g * s_b,
                       *np.percentile(boot[:, 0] - g * boot[:, 1], [2.5, 97.5])]
                      for g in grid])
    s_b_ci = tuple(np.percentile(boot[:, 1], [2.5, 97.5]))
    return dict(n=n, s_a=float(s_a), s_b=float(s_b), s_b_ci=s_b_ci,
                lam_star_bounded=bool(np.sign(s_b_ci[0]) == np.sign(s_b_ci[1])),
                lam_star=float(s_a / s_b) if s_b != 0 else np.nan,
                ci=tuple(np.nanpercentile(ls_boot, [2.5, 97.5])),
                hdi=hdi(ls_bayes),
                p_cross_in_01=float(np.mean((ls_bayes >= 0) & (ls_bayes <= 1))),
                lam_star_draws=ls_bayes, grid=grid, curve=curve)
