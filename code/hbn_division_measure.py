"""Does the lambda = 1 result depend on which ratio is taken?

The lambda = 1 measure of the paper is ln a - ln b = ln(T/b - 1), with
a = T - b the band power above the background. A division rule as usually
reported (specparam's peak height) is ln(T/b) = ln(1 + a/b). The two agree for
peaks well above the background and differ where a/b < 1.

For the quality-controlled HBN sample (results/hbn_qc_flags.csv), eyes closed
and eyes open, power-law and knee+plateau backgrounds (6-16 Hz censored;
results/hbn_kp_fits.csv), the age slope (OLS, HC3 interval) of:

  ln a          background subtracted (lambda = 0), participants with a > 0
  ln a - ln b   divided out (lambda = 1), participants with a > 0
  ln(T/b)       the ratio a division rule reports, with a > 0 and for everyone
  ln b, ln T    background and total band power
  a             T - b without the logarithm, as a share of its mean per year

Slopes are in natural-log units per year; 100 (exp(s) - 1) is % per year.
Writes results/hbn_division_measure.csv.

Usage: python hbn_division_measure.py
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def age_slope(y, age):
    ok = np.isfinite(y) & np.isfinite(age)
    r = sm.OLS(y[ok], sm.add_constant(age[ok])).fit(cov_type="HC3")
    lo, hi = r.conf_int()[1]
    return r.params[1], lo, hi, r.pvalues[1], int(ok.sum())


def main():
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    k = k[(k.window == "censor 6-16") & (k.band == "alpha") & (k.split == "full")]
    qc = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv")).set_index("subject")
    age = (pd.read_csv(os.path.join(RES, "hbn_roi.csv"), usecols=["subject", "age"])
           .drop_duplicates("subject").set_index("subject"))
    pct = lambda s: 100 * (np.exp(s) - 1)

    rows = []
    for model in ("fixed", "knee_plateau"):
        for cond in ("ec", "eo"):
            d = k[(k.model == model) & (k.cond == cond)].set_index("subject")
            d = d[["a", "b", "tot"]].join(age).join(qc)
            d = d[d.qc_ok]
            x = d.age.to_numpy(float)
            a, b, T = (d[c].to_numpy(float) for c in ("a", "b", "tot"))
            pos = a > 0
            ln_a = np.full(a.size, np.nan)
            ln_a[pos] = np.log(a[pos])
            ln_ratio = np.log(T / b)
            measures = {
                "ln a": ln_a,
                "ln a - ln b": ln_a - np.log(b),
                "ln(T/b), a > 0": np.where(pos, ln_ratio, np.nan),
                "ln(T/b), all": ln_ratio,
                "ln b": np.log(b),
                "ln T": np.log(T),
            }
            for name, y in measures.items():
                s, lo, hi, p, n = age_slope(y, x)
                rows.append(dict(model=model, cond=cond, measure=name, n=n,
                                 pct_per_year=pct(s), lo=pct(lo), hi=pct(hi), p=p))
            # without the logarithm the slope has units of power; give it as a
            # share of the sample mean so that it can be read next to the others
            s, lo, hi, p, n = age_slope(a, x)
            rows.append(dict(model=model, cond=cond, measure="a, share of mean", n=n,
                             pct_per_year=100 * s / a.mean(), lo=100 * lo / a.mean(),
                             hi=100 * hi / a.mean(), p=p,
                             spearman=pd.Series(a).corr(pd.Series(x), method="spearman"),
                             median_a_over_b=float(np.median(a[pos] / b[pos])),
                             share_below_1=float(np.mean(a[pos] / b[pos] < 1))))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "hbn_division_measure.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
