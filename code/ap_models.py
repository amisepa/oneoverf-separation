"""Aperiodic spectral models fitted by Whittle deviance.

Models (L is linear power):
    fixed         L(f) = 10^b / f^chi
    plateau       L(f) = 10^b / f^chi + p
    knee          L(f) = 10^b / (k + f^chi)
    knee_plateau  L(f) = 10^b / (k + f^chi) + p

The plateau is a flat high-frequency floor (amplifier and line noise), so the
neural part of the background is 10^b / (k + f^chi). In knee mode chi is the
asymptotic high-frequency slope and is not comparable with fixed-mode values.
Matches code/lib/oof_est_kneeplateau.m.

loglog_fit is the least-squares fit of a model linear in ln L to the single
spectra (epochs, tapers) of one recording; flank_level uses it on the two
flanks of a gap to give the background level in a band inside the gap.
"""
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import digamma, polygamma

MODELS = ("fixed", "plateau", "knee", "knee_plateau")


def n_params(model):
    return {"fixed": 2, "plateau": 3, "knee": 3, "knee_plateau": 4}[model]


def ap_eval(theta, f, model="knee_plateau"):
    """theta = [b, chi, (log10 knee), (log10 plateau)]; returns linear power."""
    b, chi = theta[0], theta[1]
    i = 2
    has_k = model in ("knee", "knee_plateau")
    has_p = model in ("plateau", "knee_plateau")
    k = 10.0 ** theta[i] if has_k else 0.0
    if has_k:
        i += 1
    p = 10.0 ** theta[i] if has_p else 0.0
    return 10.0 ** b / (k + f ** chi) + p


def whittle_dev(theta, f, P, model):
    mu = np.maximum(ap_eval(theta, f, model), 1e-300)
    return float(np.sum(np.log(mu) + P / mu))


def loglog_fit(P, X):
    """Least squares of ln P on X for many spectra of one recording, with the
    intercept corrected for the log of a gamma-distributed spectral estimate.

    P: linear power at the fit bins, shape (n_bins, n_spectra), at least two
    spectra; X: design, shape (n_bins, n_params), first column ones (e.g.
    ones and ln f for a power law). If P is gamma-distributed with shape K
    around its mean, E[ln P] lies ln K - psi(K) below the log of that mean:
    Euler's constant for one periodogram (K = 1), less for an average over
    tapers or channels. K is taken from the variance of the residuals,
    trigamma(K), computed about each bin's mean residual over the spectra so
    that a fixed departure of the recording's spectrum from the model is not
    counted as noise. Returns beta, shape (n_params, n_spectra), with
    exp(X beta) an estimate of the mean of P.
    """
    Y = np.log(np.maximum(P, 1e-30))
    beta = np.linalg.pinv(X) @ Y
    R = Y - X @ beta
    R -= R.mean(axis=1, keepdims=True)
    v = np.sum(R ** 2) / ((R.shape[1] - 1) * (R.shape[0] - X.shape[1]))
    v = np.clip(v, polygamma(1, 1e3), polygamma(1, 0.05))
    K = brentq(lambda k: polygamma(1, k) - v, 0.05, 1e3)
    beta[0] += np.log(K) - digamma(K)
    return beta


def flank_bins(f, flanks):
    """Mask of the bins on the two flanks of a gap; flanks = ((lo, gap_lo),
    (gap_hi, hi)), the edges of the gap themselves left out."""
    (a, b), (c, d) = flanks
    return ((f >= a) & (f < b)) | ((f > c) & (f <= d))


def flank_level(P, f, bins, band, flanks, Pm=None):
    """Background level in a band inside a gap, per spectrum, from a power
    law fitted to the flanks of the gap only.

    P: spectra of one recording, shape (n_spectra, n_f); bins: indices of the
    fit bins (all bins of flank_bins(f, flanks), or a subset of them); band:
    mask of the band. Each spectrum gets its own loglog_fit line through the
    fit bins. The lines are then rescaled, by one factor per band bin, so that
    their mean over the spectra equals the recording's mean spectrum
    interpolated across the gap: a straight line in ln f through ln Pm at all
    flank bins (Pm: mean spectrum, default the mean of P). The level and
    slope of the mean background thus come from the mean spectrum next to the
    gap, and the fit only supplies each spectrum's departure from it. Returns
    the band mean, shape (n_spectra,).
    """
    X = np.column_stack([np.ones(bins.size), np.log(f[bins])])
    beta = loglog_fit(P[:, bins].T, X)
    lf = np.log(np.maximum(f, 1e-9))
    L = np.exp(beta[0][:, None] + beta[1][:, None] * lf[None, band])
    fl = flank_bins(f, flanks)
    Pm = P.mean(0) if Pm is None else Pm
    target = np.exp(np.polyval(np.polyfit(lf[fl], np.log(Pm[fl]), 1), lf[band]))
    return (L * (target / L.mean(0))[None, :]).mean(1)


def ols_start(P, f):
    lf, lp = np.log10(f), np.log10(P)
    X = np.column_stack([np.ones(f.size), lf])
    beta, *_ = np.linalg.lstsq(X, lp, rcond=None)
    return float(beta[0]), float(-beta[1])


def fit_aperiodic(P, f, keep=None, model="knee_plateau", restarts=2):
    """Fit an aperiodic model to the kept frequencies of one spectrum.

    Returns dict with theta, the model name, the Whittle deviance, and
    convenience fields offset/exponent/knee/plateau. `keep` is a boolean mask
    over f selecting the frequencies used (i.e. the censored fit).
    """
    P = np.asarray(P, float)
    f = np.asarray(f, float)
    if keep is None:
        keep = np.ones_like(f, dtype=bool)
    ff, PP = f[keep], P[keep]
    b0, chi0 = ols_start(PP, ff)

    lo_p = np.log10(max(PP.min() * 1e-4, 1e-18))
    hi_p = np.log10(max(PP.min() * 2.0, 1e-17))
    bounds = {"fixed": [(b0 - 5, b0 + 5), (0.05, 6.0)],
              "plateau": [(b0 - 5, b0 + 5), (0.05, 6.0), (lo_p, hi_p)],
              "knee": [(b0 - 5, b0 + 8), (0.05, 6.0), (-5.0, 8.0)],
              "knee_plateau": [(b0 - 5, b0 + 8), (0.05, 6.0), (-5.0, 8.0),
                               (lo_p, hi_p)]}[model]

    # starting points: no knee, and a knee near the low edge of the fit range
    starts = []
    base = [b0, chi0]
    if model == "fixed":
        starts = [base]
    elif model == "plateau":
        starts = [base + [np.log10(max(PP.min() * 0.3, 1e-18))]]
    else:
        for lk in (-4.0, chi0 * np.log10(max(ff[0] * 1.5, 1.5))):
            th = base + [lk]
            if model == "knee_plateau":
                th = th + [np.log10(max(PP.min() * 0.3, 1e-18))]
            # a knee raises the fitted level, so lift the offset to match
            th[0] = b0 + (lk if lk > 0 else 0.0)
            starts.append(th)
        starts = starts[:max(restarts, 1)]

    best = None
    for th0 in starts:
        th0 = np.clip(th0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            r = minimize(whittle_dev, th0, args=(ff, PP, model),
                         method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=800, ftol=1e-12))
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        return None
    th = best.x
    out = dict(theta=th, model=model, dev=float(best.fun),
               offset=float(th[0]), exponent=float(th[1]),
               knee=(10.0 ** th[2] if model in ("knee", "knee_plateau") else 0.0),
               plateau=(10.0 ** th[-1] if model in ("plateau", "knee_plateau")
                        else 0.0),
               n_fit=int(keep.sum()))
    out["knee_freq"] = (out["knee"] ** (1.0 / out["exponent"])
                        if out["knee"] > 0 else 0.0)
    return out


def ap_band_power(fit, f, lo, hi):
    """Mean fitted aperiodic power inside a band."""
    m = (f >= lo) & (f <= hi)
    return float(np.mean(ap_eval(fit["theta"], f[m], fit["model"])))


def compare_models(P, f, keep=None, models=MODELS):
    """Fit each model and return deviances with a BIC-style penalty.

    2*(D_simple - D_complex) is the likelihood-ratio statistic scaled by the
    Gamma shape K; K is unknown here, so report raw deviance differences and
    let the caller scale them.
    """
    out = {}
    for m in models:
        fit = fit_aperiodic(P, f, keep, m)
        if fit is not None:
            out[m] = fit
    return out
