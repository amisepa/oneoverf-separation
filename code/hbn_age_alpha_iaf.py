"""How much of the alpha-vs-age result is the band moving with alpha frequency?

Periodic (a) and aperiodic (b) power are taken in the individual band, the
eyes-closed alpha frequency (IAF) +/- 2 Hz. IAF rises with age, so the band
moves down the 1/f slope and the background under it falls for that reason
alone. On the quality-controlled sample (results/hbn_qc_flags.csv), for eyes
closed and eyes open and for the power-law and knee+plateau backgrounds
(6-16 Hz censored; results/hbn_kp_fits.csv), with a > 0:

  individual band    the reference: age slopes of ln a (lambda = 0), ln b and
                     ln a - ln b (lambda = 1), crossover lambda* = s_a / s_b
  + IAF covariate    the same at fixed IAF (IAF added to the regression)
  background 8-12    a in the individual band, b from the same fit averaged
                     over a fixed 8-12 Hz band, so that the background no
                     longer moves with IAF; s_a is unchanged
  band-shift share   1 - s_b(8-12 Hz) / s_b(IAF +/- 2 Hz): the share of the
                     background's age slope that is due to the band moving,
                     with the age slope of ln b(IAF +/- 2) - ln b(8-12)

The background of each fit is evaluated in the fixed band from its stored
parameters: the exponent, knee frequency and plateau are in hbn_kp_fits.csv,
and the offset follows from the stored band mean b on the 0.25 Hz grid of the
spectra. This reproduces the stored 30-38 Hz band mean of the same fit to
rounding error (checked at run time). Periodic power in a fixed band needs
the spectra themselves and is in hbn_age_alpha_bands.py.

Slopes are OLS on age with HC3 intervals, in natural-log units per year
(100 (exp(s) - 1) is % per year); the crossover has a Bayesian-bootstrap HDI
(lambda_curve.effect_curve) and the share a pairs-bootstrap interval.
Writes results/hbn_age_alpha_iaf.csv.

Usage: python hbn_age_alpha_iaf.py [--psd-dir DIR] [--nboot 2000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hbn_age_alpha_robust import RES, meta
from lambda_curve import effect_curve

FREQS = np.arange(2.0, 55.0 + 0.125, 0.25)     # grid of the fitted spectra (hbn_kp.py)
FIXED_BAND = (8.0, 12.0)
CONTROL_BAND = (30.0, 38.0)


def band_shape(chi, knee_freq, lo, hi):
    """Band mean of 1 / (k + f^chi), the background without offset and plateau."""
    f = FREQS[(FREQS >= lo) & (FREQS <= hi)]
    return np.mean(1.0 / (knee_freq ** chi + f ** chi))


def fits(model):
    """Per participant and condition: a, b and IAF of the individual band, and
    the same fit's background over the fixed band (b_fixed)."""
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    k = k[(k.model == model) & (k.window == "censor 6-16") & (k.split == "full")]
    w = k.pivot_table(index=["subject", "cond"], columns="band",
                      values=["a", "b", "tot", "iaf", "exponent", "knee_freq", "plateau"])
    g = lambda v, band="alpha": w[(v, band)].to_numpy()
    chi, plateau, iaf = g("exponent"), g("plateau"), g("iaf")
    kf = g("knee_freq") if model == "knee_plateau" else np.zeros(len(w))
    shape = lambda lo, hi: np.array([band_shape(c, q, l, h)
                                     for c, q, l, h in zip(chi, kf, lo, hi)])
    # b = 10^offset * shape + plateau in every band of one fit
    amp = (g("b") - plateau) / shape(iaf - 2, iaf + 2)
    n = len(w)
    check = amp * shape(np.full(n, CONTROL_BAND[0]), np.full(n, CONTROL_BAND[1])) + plateau
    err = np.nanmax(np.abs(check / g("b", "control") - 1))
    assert err < 1e-9, f"background not reproduced from the stored parameters ({err:.2g})"
    out = pd.DataFrame(dict(a=g("a"), b=g("b"), tot=g("tot"), iaf=iaf,
                            b_fixed=amp * shape(np.full(n, FIXED_BAND[0]),
                                                np.full(n, FIXED_BAND[1])) + plateau),
                       index=w.index)
    return out.reset_index()


def slope(y, x, cov=None):
    """OLS slope on x (optionally with covariates), HC3 interval and p."""
    X = np.column_stack([np.ones(x.size), x] + ([] if cov is None else [cov]))
    r = sm.OLS(y, X).fit(cov_type="HC3")
    return dict(est=r.params[1], lo=r.conf_int()[1][0], hi=r.conf_int()[1][1], p=r.pvalues[1])


def curve_rows(key, analysis, la, lb, x, cov, nboot, seed):
    """Rows for lambda = 0, the background, lambda = 1 and the crossover."""
    r = effect_curve(la, lb, x=x, covariates=cov, nboot=nboot, rng=np.random.default_rng(seed))
    base = dict(**key, analysis=analysis, n=r["n"])
    rows = [dict(**base, measure=m, **slope(y, x, cov))
            for m, y in (("lambda=0", la), ("background ln b", lb), ("lambda=1", la - lb))]
    rows.append(dict(**base, measure="crossover lambda*", est=r["lam_star"], lo=r["hdi"][0],
                     hi=r["hdi"][1], p=np.nan, p_cross_in_01=r["p_cross_in_01"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    M = meta(a.psd_dir)
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    rows = []
    for model in ("fixed", "knee_plateau"):
        F = fits(model)
        F = F[F.subject.isin(Q.index[Q.qc_ok])].join(M[["age"]], on="subject", how="inner")
        for cond in ("ec", "eo"):
            d = F[(F.cond == cond) & np.isfinite(F.a) & np.isfinite(F.b) & np.isfinite(F.tot)
                  & (F.a > 0)]
            x, iaf = d.age.to_numpy(), d.iaf.to_numpy()
            la, lb, lbf = (np.log(d[c]).to_numpy() for c in ("a", "b", "b_fixed"))
            key = dict(model=model, cond=cond)
            rows.append(dict(**key, analysis="individual band", measure="IAF (Hz/year)",
                             n=len(d), **slope(iaf, x)))
            rows += curve_rows(key, "individual band", la, lb, x, None, a.nboot, 0)
            rows += curve_rows(key, "+ IAF covariate", la, lb, x, iaf, a.nboot, 0)
            rows += curve_rows(key, "background 8-12 Hz", la, lbf, x, None, a.nboot, 0)
            # share of the background slope that is the band moving
            rows.append(dict(**key, analysis="band shift", n=len(d),
                             measure="ln b(IAF +/- 2) - ln b(8-12)", **slope(lb - lbf, x)))
            rng = np.random.default_rng(0)
            n = len(d)
            share = lambda i: 1 - np.polyfit(x[i], lbf[i], 1)[0] / np.polyfit(x[i], lb[i], 1)[0]
            bs = [share(rng.integers(0, n, n)) for _ in range(a.nboot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(**key, analysis="band shift", n=n,
                             measure="share of background slope", est=share(np.arange(n)),
                             lo=lo, hi=hi, p=np.nan))
    O = pd.DataFrame(rows)[["model", "cond", "analysis", "measure", "est", "lo", "hi", "p",
                            "p_cross_in_01", "n"]]
    O.to_csv(os.path.join(RES, "hbn_age_alpha_iaf.csv"), index=False)
    for (model, cond), G in O.groupby(["model", "cond"], sort=False):
        print(f"\n=== {model}, {cond.upper()}: n = {G.n.iloc[0]}")
        for _, r in G.iterrows():
            raw = "lambda*" in r.measure or "share" in r.measure or "Hz/year" in r.measure
            f = (lambda v: v) if raw else (lambda v: 100 * (np.exp(v) - 1))
            print(f"  {r.analysis:20s} {r.measure:30s} {f(r.est):+7.3f}{'' if raw else '%/y'} "
                  f"[{f(r.lo):+.3f}, {f(r.hi):+.3f}]"
                  + (f"  p = {r.p:.2g}" if np.isfinite(r.p) else "")
                  + (f"  P(in [0,1]) {r.p_cross_in_01:.3f}" if np.isfinite(r.p_cross_in_01) else ""))


if __name__ == "__main__":
    main()
