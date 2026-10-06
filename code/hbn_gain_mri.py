"""How much of the HBN age effect on the background is scalp and skull gain?

Uses the scalp-to-cortex distance (SCD) under the posterior EEG region from
hbn_scalp_distance.py (participants passing its checks) and the eyes-closed
and eyes-open alpha-band periodic (a) and aperiodic (b) power of the same
participants (power-law background, 6-16 Hz censored; hbn_kp_fits.csv).

  1. SCD against age (+ sex, site).
  2. ln a and ln b on age and SCD (+ sex, site). A pure gain lowers both by
     the same amount per mm, so the SCD coefficients should agree; their
     difference is tested with a participant bootstrap.
  3. phi, the share of the background's age slope carried by SCD:
     beta_SCD(ln b) * dSCD/dage / s_b, with a bootstrap interval. phi enters
     the effective coupling 1 - (1 - lambda)(1 - phi) of the main text.
  4. The crossover lambda* in this subsample without and with SCD as a
     covariate (lambda_curve.effect_curve), and the age slopes of ln a
     (lambda = 0) and ln a - ln b (lambda = 1) without and with it, each with
     a pairs-bootstrap interval.
  5. phi_a, the share of the age slope of ln a (the lambda = 0 decline)
     carried by SCD: beta_SCD(ln a) * dSCD/dage / s_a, which is
     1 - (slope with SCD) / (slope without), with a bootstrap interval.

Writes results/hbn_gain_mri_<region>.csv.

Usage: python hbn_gain_mri.py [--psd-dir DIR] [--nboot 2000] [--region post]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hbn_age_alpha_robust import RES, meta, wide
from lambda_curve import effect_curve


def ols(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def design(d, *cols):
    base = [np.ones(len(d)), d.age.to_numpy()] + [d[c].to_numpy() for c in cols]
    base += [(d.sex == "M").astype(float).to_numpy(), (d.site == "Site-RU").astype(float).to_numpy()]
    return np.column_stack(base)


def stats(d, region):
    la, lb, scd = np.log(d.a).to_numpy(), np.log(d.b).to_numpy(), d[f"scd_{region}"].to_numpy()
    g = ols(scd, design(d))[1]                               # mm per year
    Xs = design(d, f"scd_{region}")
    ba, bb = ols(la, Xs)[2], ols(lb, Xs)[2]                 # per mm, at fixed age
    s_a, s_b = ols(la, design(d))[1], ols(lb, design(d))[1]  # total age slopes
    return dict(scd_per_year=g, beta_a=ba, beta_b=bb, beta_diff=ba - bb, s_b=s_b,
                phi=bb * g / s_b, s_a=s_a, phi_a=ba * g / s_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--region", default="post")
    a = ap.parse_args()
    S = pd.read_csv(os.path.join(RES, "hbn_scalp_distance.csv"))
    S = S[S.mri_ok].set_index("subject")[[f"scd_{a.region}", "site", "brain_ml"]]
    M = meta(a.psd_dir)
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    W = wide("fixed").join(M, how="inner").join(S, how="inner")
    W = W.loc[W.index.intersection(Q.index[Q.qc_ok])]
    rng = np.random.default_rng(0)
    rows = []
    for cond in ("ec", "eo"):
        d = W[(W[f"a_{cond}_full"] > 0)].rename(columns={f"a_{cond}_full": "a",
                                                          f"b_{cond}_full": "b"})
        d = d[np.isfinite(d.a) & np.isfinite(d.b) & np.isfinite(d[f"scd_{a.region}"])]
        est = stats(d, a.region)
        n = len(d)
        bs = pd.DataFrame([stats(d.iloc[rng.integers(0, n, n)], a.region)
                           for _ in range(a.nboot)])
        row = dict(cond=cond, region=a.region, n=n)
        for k, v in est.items():
            row[k] = v
            row[k + "_lo"], row[k + "_hi"] = np.percentile(bs[k], [2.5, 97.5])
        cov = np.column_stack([(d.sex == "M").astype(float), (d.site == "Site-RU").astype(float)])
        for tag, C in (("", cov), ("_scd", np.column_stack([cov, d[f"scd_{a.region}"]]))):
            r = effect_curve(np.log(d.a), np.log(d.b), x=d.age.to_numpy(), covariates=C,
                             rng=np.random.default_rng(0))
            row[f"lam_star{tag}"] = r["lam_star"]
            row[f"hdi_lo{tag}"], row[f"hdi_hi{tag}"] = r["hdi"]
            row[f"lam0{tag}"], row[f"lam1{tag}"] = r["curve"][0][0], r["curve"][-1][0]
            for lam, c in ((0, r["curve"][0]), (1, r["curve"][-1])):
                row[f"lam{lam}{tag}_lo"], row[f"lam{lam}{tag}_hi"] = c[1], c[2]
        rows.append(row)
        print(f"{cond}: n {n}; SCD {row['scd_per_year']:+.3f} mm/y "
              f"[{row['scd_per_year_lo']:+.3f}, {row['scd_per_year_hi']:+.3f}]; "
              f"per mm at fixed age: ln a {row['beta_a']:+.3f} [{row['beta_a_lo']:+.3f}, "
              f"{row['beta_a_hi']:+.3f}], ln b {row['beta_b']:+.3f} [{row['beta_b_lo']:+.3f}, "
              f"{row['beta_b_hi']:+.3f}], difference {row['beta_diff']:+.3f} "
              f"[{row['beta_diff_lo']:+.3f}, {row['beta_diff_hi']:+.3f}]; s_b {row['s_b']:+.3f}; "
              f"phi {row['phi']:.2f} [{row['phi_lo']:.2f}, {row['phi_hi']:.2f}]; lambda* "
              f"{row['lam_star']:.2f} [{row['hdi_lo']:.2f}, {row['hdi_hi']:.2f}] -> with SCD "
              f"{row['lam_star_scd']:.2f} [{row['hdi_lo_scd']:.2f}, {row['hdi_hi_scd']:.2f}]",
              flush=True)
        pct = lambda v: 100 * (np.exp(v) - 1)
        print("    % per year: " + "; ".join(
            f"lambda={lam}{' with SCD' if tag else ''} {pct(row[f'lam{lam}{tag}']):+.1f} "
            f"[{pct(row[f'lam{lam}{tag}_lo']):+.1f}, {pct(row[f'lam{lam}{tag}_hi']):+.1f}]"
            for tag in ("", "_scd") for lam in (0, 1))
            + f"; share of the lambda=0 slope carried by SCD {row['phi_a']:.2f} "
              f"[{row['phi_a_lo']:.2f}, {row['phi_a_hi']:.2f}]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"hbn_gain_mri_{a.region}.csv"), index=False)


if __name__ == "__main__":
    main()
