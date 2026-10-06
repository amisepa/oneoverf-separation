"""Is the alpha-vs-age reversal the same in childhood and in adolescence?

The age slope of ln a - lambda ln b over the whole 5-22 y range averages two
developmental phases. Here the quality-controlled sample
(results/hbn_qc_flags.csv) is split at an age cut (11, 12 and 13 years; 12 is
the reference) and, in each age range, for eyes closed and eyes open and for
the power-law and knee+plateau backgrounds (6-16 Hz censored;
results/hbn_kp_fits.csv):

  lam0, lam1   age slope of ln a (background subtracted) and of ln a - ln b
               (divided out): OLS with HC3 interval and p, and the
               pairs-bootstrap interval of lambda_curve.effect_curve
  s_b          age slope of ln b with its pairs-bootstrap interval; when the
               interval includes 0 the crossover is unbounded
  lam_star     crossover s_a / s_b with its Bayesian-bootstrap HDI and the
               share of draws inside [0, 1]

and, for each cut, a "difference" row (older minus younger): the difference of
the two slopes with a normal interval from the HC3 standard errors, the
difference of the two crossovers with the HDI of the difference of their
(independent) Bayesian-bootstrap draws, and p_older_gt, the share of draws in
which the older crossover exceeds the younger one.

Slopes are in natural-log units per year; 100 (exp(s) - 1) is % per year.
Writes results/hbn_age_alpha_phases.csv.

Usage: python hbn_age_alpha_phases.py [--psd-dir DIR] [--nboot 2000]
                                      [--cuts 11,12,13]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hbn_age_alpha_robust import RES, meta, ols_hc3, wide
from lambda_curve import effect_curve, hdi

MODELS = ("fixed", "knee_plateau")
CONDS = ("ec", "eo")


def fit_range(d, cond, nboot, seed):
    """Slopes and crossover of one age range. Returns the table row and the
    Bayesian-bootstrap draws of the crossover."""
    x = d.age.to_numpy()
    la, lb = np.log(d[f"a_{cond}_full"]).to_numpy(), np.log(d[f"b_{cond}_full"]).to_numpy()
    r = effect_curve(la, lb, x=x, nboot=nboot, rng=np.random.default_rng(seed))
    row = dict(n=r["n"], age_min=x.min(), age_max=x.max(), age_median=np.median(x))
    X = sm.add_constant(x)
    for lam, c in ((0, r["curve"][0]), (1, r["curve"][-1])):
        s, ci, p, se = ols_hc3(la - lam * lb, X)
        row.update({f"lam{lam}": s, f"lam{lam}_lo": ci[0], f"lam{lam}_hi": ci[1],
                    f"lam{lam}_p": p, f"lam{lam}_se": se,
                    f"lam{lam}_boot_lo": c[1], f"lam{lam}_boot_hi": c[2]})
    row.update(s_b=r["s_b"], s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
               lam_star_bounded=r["lam_star_bounded"], lam_star=r["lam_star"],
               hdi_lo=r["hdi"][0], hdi_hi=r["hdi"][1], p_cross_in_01=r["p_cross_in_01"])
    return row, r["lam_star_draws"]


def difference(young, old, dy, do):
    """Older minus younger: slopes (independent samples, HC3 standard errors)
    and crossovers (difference of independent Bayesian-bootstrap draws)."""
    row = dict(n=young["n"] + old["n"])
    z = stats.norm.ppf(0.975)
    for lam in (0, 1):
        diff = old[f"lam{lam}"] - young[f"lam{lam}"]
        se = np.hypot(old[f"lam{lam}_se"], young[f"lam{lam}_se"])
        row.update({f"lam{lam}": diff, f"lam{lam}_lo": diff - z * se,
                    f"lam{lam}_hi": diff + z * se, f"lam{lam}_se": se,
                    f"lam{lam}_p": 2 * stats.norm.sf(abs(diff / se))})
    dd = do - dy
    row.update(lam_star_bounded=bool(young["lam_star_bounded"] and old["lam_star_bounded"]),
               lam_star=old["lam_star"] - young["lam_star"],
               hdi_lo=hdi(dd)[0], hdi_hi=hdi(dd)[1], p_older_gt=float(np.mean(dd > 0)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--cuts", default="11,12,13")
    a = ap.parse_args()
    cuts = [float(c) for c in a.cuts.split(",")]
    M = meta(a.psd_dir)
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    rows = []
    for im, model in enumerate(MODELS):
        W = wide(model).join(M, how="inner")
        W = W.loc[W.index.intersection(Q.index[Q.qc_ok])]
        for ic, cond in enumerate(CONDS):
            A, B, T = (W[f"{v}_{cond}_full"] for v in ("a", "b", "tot"))
            d = W[np.isfinite(A) & np.isfinite(B) & np.isfinite(T) & (A > 0)]
            key = dict(model=model, cond=cond)
            row, _ = fit_range(d, cond, a.nboot, [0, im, ic, 0, 0])
            rows.append(dict(**key, cut=np.nan, range="all", **row))
            for cut in cuts:
                # one random stream per cell, so that a cell does not depend on
                # which other cells are run
                young, dy = fit_range(d[d.age < cut], cond, a.nboot, [0, im, ic, int(cut), 1])
                old, do = fit_range(d[d.age >= cut], cond, a.nboot, [0, im, ic, int(cut), 2])
                rows.append(dict(**key, cut=cut, range="younger", **young))
                rows.append(dict(**key, cut=cut, range="older", **old))
                rows.append(dict(**key, cut=cut, range="difference",
                                 **difference(young, old, dy, do)))
    O = pd.DataFrame(rows).drop(columns=["lam0_se", "lam1_se"])
    O.to_csv(os.path.join(RES, "hbn_age_alpha_phases.csv"), index=False)

    pct = lambda v: 100 * (np.exp(v) - 1)
    for (model, cond), G in O.groupby(["model", "cond"], sort=False):
        print(f"\n=== {model}, {cond.upper()} (% per year, HC3 95% CI)")
        for _, r in G.iterrows():
            if r.range == "difference":
                print(f"  cut {r.cut:.0f}  older - younger: lambda=0 {r.lam0*100:+.1f} "
                      f"[{r.lam0_lo*100:+.1f}, {r.lam0_hi*100:+.1f}], lambda=1 {r.lam1*100:+.1f} "
                      f"[{r.lam1_lo*100:+.1f}, {r.lam1_hi*100:+.1f}] (ln x 100); lambda* "
                      f"{r.lam_star:+.2f} HDI [{r.hdi_lo:+.2f}, {r.hdi_hi:+.2f}], "
                      f"P(older > younger) {r.p_older_gt:.3f}")
                continue
            lab = "all ages" if r.range == "all" else \
                f"cut {r.cut:.0f}  {'<' if r.range == 'younger' else '>='} {r.cut:.0f} y"
            print(f"  {lab:18s} n {r.n:4.0f}: lambda=0 {pct(r.lam0):+.1f} [{pct(r.lam0_lo):+.1f}, "
                  f"{pct(r.lam0_hi):+.1f}], lambda=1 {pct(r.lam1):+.1f} [{pct(r.lam1_lo):+.1f}, "
                  f"{pct(r.lam1_hi):+.1f}]; ln b {pct(r.s_b):+.1f} [{pct(r.s_b_lo):+.1f}, "
                  f"{pct(r.s_b_hi):+.1f}]; lambda* {r.lam_star:.2f} HDI [{r.hdi_lo:.2f}, "
                  f"{r.hdi_hi:.2f}], P(in [0,1]) {r.p_cross_in_01:.3f}"
                  + ("" if r.lam_star_bounded else "  (unbounded)"))


if __name__ == "__main__":
    main()
