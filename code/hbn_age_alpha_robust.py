"""Robustness of the alpha-vs-age result to data quality and statistics.

Quality control (a participant is excluded from the primary "qc" sample if
any flag is set; the "all" sample keeps everyone for comparison):
  flat      alpha-band total power in either condition more than 300 times
            below or above the sample median (flat or mis-scaled recording)
  fallback  every Welch segment kept in a condition (35 eyes open or 85 eyes
            closed): the extractor keeps all segments when fewer than four
            pass rejection, so these recordings were never cleaned
  edge_iaf  individual alpha frequency at the edge of the 6-14 Hz search
            window (<= 6.25 or >= 13.75 Hz), i.e. no detectable alpha peak

For each sample, background model (power law, knee+plateau) and condition,
the age slope of y = ln a - lambda ln b (lambda = 0: background subtracted in
linear power; lambda = 1: divided out) is estimated with:
  ols       OLS, HC3 CI                 boot      pairs bootstrap CI
  huber     Huber M-estimator           theilsen  Theil-Sen slope and CI
  skipped   Spearman rho after removing bivariate outliers (minimum
            covariance determinant, chi-square 0.975 cut-off)
  wls       weights 1 / (s_i^2 + tau^2), s_i^2 from the split-half difference
  covar     + sex, recording release, ln(number of segments)
  release_re  random-effects meta-analysis of per-release slopes
            (DerSimonian-Laird tau^2, Hartung-Knapp-Sidik-Jonkman CI)
  age5_16   OLS restricted to ages 5-16
  quad      quadratic age term; slope at the median age
plus: truncation-free measures (ln(tot/b) for everyone; linear a, Theil-Sen);
the crossover lambda* = s_a / s_b with bootstrap CI and Bayesian-bootstrap HDI
(unadjusted and covariate-adjusted); alpha reactivity (eyes closed minus eyes
open) under both lambdas; age-resolved slopes from a natural cubic spline
(df = 4) with pointwise bootstrap CIs; Holm correction over the eight primary
OLS tests of each sample; and age dependence of exclusion for a <= 0.

Writes results/hbn_age_alpha_robust.csv, hbn_age_alpha_crossover.csv,
hbn_age_alpha_spline.csv, hbn_age_alpha_reactivity.csv, hbn_qc_flags.csv.

Usage: python hbn_age_alpha_robust.py [--psd-dir DIR] [--nboot 2000]
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from sklearn.covariance import MinCovDet

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
warnings.filterwarnings("ignore")
FULL_SEG = {"eo": 35, "ec": 85}


def meta(psd_dir):
    """Age, sex, release and segment counts per participant, from the PSD
    files, or from results/hbn_roi.csv (which carries the same fields) when
    the PSD directory is not available."""
    cols = ["subject", "age", "sex", "release", "n_seg_eo", "n_seg_ec"]
    files = glob.glob(os.path.join(psd_dir, "*.npz"))
    if not files:
        R = pd.read_csv(os.path.join(RES, "hbn_roi.csv"), usecols=cols)
        return R.drop_duplicates("subject").set_index("subject")
    rows = []
    for p in files:
        d = np.load(p, allow_pickle=True)
        rows.append(dict(subject=str(d["subject"]), age=float(d["age"]),
                         sex=str(d["sex"]), release=str(d["release"]),
                         n_seg_eo=int(d["n_seg_eo"]), n_seg_ec=int(d["n_seg_ec"])))
    return pd.DataFrame(rows).drop_duplicates("subject").set_index("subject")


def wide(model, window="censor 6-16"):
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    k = k[(k.model == model) & (k.window == window) & (k.band == "alpha")]
    w = k.pivot_table(index="subject", columns=["cond", "split"],
                      values=["a", "b", "tot", "iaf"])
    w.columns = [f"{v}_{c}_{s}" for v, c, s in w.columns]
    return w


def qc_flags(M):
    w = wide("fixed")
    t_ec, t_eo = w["tot_ec_full"], w["tot_eo_full"]
    flat = ((t_ec < t_ec.median() / 300) | (t_ec > t_ec.median() * 300) |
            (t_eo < t_eo.median() / 300) | (t_eo > t_eo.median() * 300))
    iaf = w["iaf_ec_full"]
    edge = (iaf <= 6.25) | (iaf >= 13.75)
    Q = pd.DataFrame(dict(flat=flat, edge_iaf=edge)).join(M, how="inner")
    Q["fallback"] = (Q.n_seg_eo >= FULL_SEG["eo"]) | (Q.n_seg_ec >= FULL_SEG["ec"])
    Q["qc_ok"] = ~(Q.flat | Q.edge_iaf | Q.fallback)
    return Q[["flat", "edge_iaf", "fallback", "qc_ok"]]


def hdi(x, mass=0.95):
    x = np.sort(np.asarray(x))
    n = int(np.floor(mass * x.size))
    i = int(np.argmin(x[n:] - x[:x.size - n]))
    return x[i], x[i + n]


def ols_hc3(y, X):
    r = sm.OLS(y, X).fit(cov_type="HC3")
    return r.params[1], r.conf_int()[1], r.pvalues[1], r.bse[1]


def release_re(d, y):
    """Random-effects meta-analysis of per-release OLS slopes (DL + HKSJ)."""
    b, v = [], []
    for rel, g in d.groupby("release"):
        if len(g) < 20:
            continue
        s, _, _, se = ols_hc3(y[g.index].to_numpy(), sm.add_constant(g.age.to_numpy()))
        b.append(s)
        v.append(se ** 2)
    b, v = np.array(b), np.array(v)
    w = 1 / v
    fixed = np.sum(w * b) / np.sum(w)
    Qs = np.sum(w * (b - fixed) ** 2)
    k = b.size
    tau2 = max(0.0, (Qs - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (v + tau2)
    mu = np.sum(ws * b) / np.sum(ws)
    q = np.sum(ws * (b - mu) ** 2) / (k - 1)          # HKSJ scaling
    se = np.sqrt(q / np.sum(ws))
    t = stats.t.ppf(0.975, k - 1)
    i2 = max(0.0, (Qs - (k - 1)) / Qs) if Qs > 0 else 0.0
    p = 2 * stats.t.sf(abs(mu / se), k - 1)
    return mu, mu - t * se, mu + t * se, p, i2, k


def meas_var(d, cond, lam):
    ao, ae = d[f"a_{cond}_odd"], d[f"a_{cond}_even"]
    bo, be = d[f"b_{cond}_odd"], d[f"b_{cond}_even"]
    ok = (ao > 0) & (ae > 0)
    yo = np.log(ao.where(ok)) - lam * np.log(bo)
    ye = np.log(ae.where(ok)) - lam * np.log(be)
    raw = (yo - ye) ** 2 / 4.0
    snr = np.log(d[f"a_{cond}_full"] / d[f"b_{cond}_full"])
    nseg = np.log(d[f"n_seg_{cond}"])
    m = np.isfinite(raw) & (raw > 0) & np.isfinite(snr)
    X = sm.add_constant(np.column_stack([snr, snr ** 2, nseg]))
    fit = sm.OLS(np.log(raw[m]), X[m]).fit()
    return pd.Series(np.exp(X @ fit.params + 1.27), index=d.index)


def slopes(d, y, cond, lam, rng, nboot):
    x, yv = d.age.to_numpy(), y.to_numpy()
    X = sm.add_constant(x)
    out = {}
    s, ci, p, _ = ols_hc3(yv, X)
    out["ols"] = (s, ci[0], ci[1], p)
    n = x.size
    bs = np.array([np.polyfit(x[i], yv[i], 1)[0]
                   for i in (rng.integers(0, n, n) for _ in range(nboot))])
    out["boot"] = (s, *np.percentile(bs, [2.5, 97.5]), np.nan)
    hub = sm.RLM(yv, X, M=sm.robust.norms.HuberT()).fit()
    out["huber"] = (hub.params[1], *hub.conf_int()[1], hub.pvalues[1])
    ts = stats.theilslopes(yv, x, alpha=0.95)
    out["theilsen"] = (ts.slope, ts.low_slope, ts.high_slope, np.nan)
    Z = np.column_stack([(x - np.median(x)) / stats.median_abs_deviation(x),
                         (yv - np.median(yv)) / stats.median_abs_deviation(yv)])
    keep = MinCovDet(random_state=0).fit(Z).mahalanobis(Z) <= stats.chi2.ppf(0.975, 2)
    rho, prho = stats.spearmanr(x[keep], yv[keep])
    xk, yk = x[keep], yv[keep]
    rb = [stats.spearmanr(xk[i], yk[i])[0]
          for i in (rng.integers(0, xk.size, xk.size) for _ in range(nboot // 4))]
    out["skipped"] = (rho, *np.percentile(rb, [2.5, 97.5]), prho)
    s2 = meas_var(d, cond, lam)
    ok = np.isfinite(s2).to_numpy()
    Xw = sm.add_constant(x[ok])
    r0 = sm.OLS(yv[ok], Xw).fit()
    s2v = s2.to_numpy()[ok]
    tau2 = max(np.var(r0.resid, ddof=2) - np.mean(s2v), 0.05 * np.var(r0.resid))
    rw = sm.WLS(yv[ok], Xw, weights=1 / (s2v + tau2)).fit(cov_type="HC3")
    out["wls"] = (rw.params[1], *rw.conf_int()[1], rw.pvalues[1])
    C = pd.get_dummies(d[["sex", "release"]], drop_first=True, dtype=float)
    Xc = sm.add_constant(pd.concat([d.age, C, np.log(d[f"n_seg_{cond}"]).rename("lnseg")],
                                   axis=1))
    rc = sm.OLS(y, Xc).fit(cov_type="HC3")
    out["covar"] = (rc.params["age"], *rc.conf_int().loc["age"], rc.pvalues["age"])
    mu, lo, hi, p, i2, k = release_re(d, y)
    out["release_re"] = (mu, lo, hi, p)
    out["release_I2"] = i2
    r5 = (d.age <= 16).to_numpy()
    s, ci, p, _ = ols_hc3(yv[r5], sm.add_constant(x[r5]))
    out["age5_16"] = (s, ci[0], ci[1], p)
    ac = x - np.median(x)
    rq = sm.OLS(yv, sm.add_constant(np.column_stack([ac, ac ** 2]))).fit(cov_type="HC3")
    out["quad"] = (rq.params[1], *rq.conf_int()[1], rq.pvalues[1])
    return out


def spline_derivative(d, y, rng, nboot=1000, ages=np.arange(6, 18.01, 1.0)):
    x, yv = d.age.to_numpy(), y.to_numpy()
    lo_, hi_ = np.percentile(x, [1, 99])
    def deriv(xx, yy):
        D = patsy.dmatrix("cr(x, df=4, lower_bound=lb, upper_bound=ub)",
                          {"x": np.clip(xx, lo_, hi_), "lb": lo_, "ub": hi_},
                          return_type="dataframe")
        beta = np.linalg.lstsq(D.to_numpy(), yy, rcond=None)[0]
        info = D.design_info
        def pred(a):
            return np.asarray(patsy.build_design_matrices(
                [info], {"x": a, "lb": lo_, "ub": hi_})[0]) @ beta
        h = 0.05
        return (pred(ages + h) - pred(ages - h)) / (2 * h)
    est = deriv(x, yv)
    n = x.size
    bs = np.array([deriv(x[i], yv[i]) for i in (rng.integers(0, n, n) for _ in range(nboot))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return pd.DataFrame(dict(age=ages, deriv=est, lo=lo, hi=hi))


def crossover(d, la, lb, cond, rng, nboot):
    x, n = d.age.to_numpy(), len(d)
    X = np.column_stack([np.ones(n), x])
    sa, sb = np.polyfit(x, la, 1)[0], np.polyfit(x, lb, 1)[0]
    pb, bb = [], []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        pb.append(np.polyfit(x[i], la[i], 1)[0] / np.polyfit(x[i], lb[i], 1)[0])
        w = rng.dirichlet(np.ones(n))
        WX = X * w[:, None]
        G = X.T @ WX
        bb.append(np.linalg.solve(G, WX.T @ la)[1] / np.linalg.solve(G, WX.T @ lb)[1])
    C = pd.get_dummies(d[["sex", "release"]], drop_first=True, dtype=float)
    Xa = np.column_stack([np.ones(n), x, C.to_numpy(), np.log(d[f"n_seg_{cond}"].to_numpy())])
    def coef(w):
        WX = Xa * w[:, None]
        G = Xa.T @ WX
        return (np.linalg.lstsq(G, WX.T @ la, rcond=None)[0][1],
                np.linalg.lstsq(G, WX.T @ lb, rcond=None)[0][1])
    ca, cb = coef(np.ones(n))
    ba = []
    for _ in range(nboot):
        u, v = coef(rng.dirichlet(np.ones(n)))
        ba.append(u / v)
    return dict(lam_star=sa / sb, ci_lo=np.percentile(pb, 2.5), ci_hi=np.percentile(pb, 97.5),
                hdi_lo=hdi(bb)[0], hdi_hi=hdi(bb)[1], slope_ln_a=sa, slope_ln_b=sb,
                lam_star_adj=ca / cb, hdi_adj_lo=hdi(ba)[0], hdi_adj_hi=hdi(ba)[1])


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (p.size - rank) * p[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    M = meta(a.psd_dir)
    Q = qc_flags(M)
    Q.to_csv(os.path.join(RES, "hbn_qc_flags.csv"))
    print(f"QC: {len(Q)} participants; flat {int(Q.flat.sum())}, edge IAF "
          f"{int(Q.edge_iaf.sum())}, fallback (uncleaned) {int(Q.fallback.sum())}; "
          f"pass {int(Q.qc_ok.sum())}")

    rows, cross, spl, react = [], [], [], []
    for sample in ("qc", "all"):
        keep_ids = Q.index[Q.qc_ok] if sample == "qc" else Q.index
        for model in ("fixed", "knee_plateau"):
            W = wide(model).join(M, how="inner")
            W = W.loc[W.index.intersection(keep_ids)]
            for cond in ("ec", "eo"):
                A, B, T = (W[f"{v}_{cond}_full"] for v in ("a", "b", "tot"))
                base = W[np.isfinite(A) & np.isfinite(B) & np.isfinite(T)]
                excl = (base[f"a_{cond}_full"] <= 0).astype(float)
                lg = sm.Logit(excl, sm.add_constant(base.age)).fit(disp=0)
                d = base[base[f"a_{cond}_full"] > 0].copy()
                la = np.log(d[f"a_{cond}_full"])
                lb = np.log(d[f"b_{cond}_full"])
                print(f"\n=== {sample}, {model}, {cond.upper()}: n = {len(d)} of {len(base)} "
                      f"(a<=0 excluded {excl.mean()*100:.1f}%, OR/yr "
                      f"{np.exp(lg.params['age']):.3f}, p = {lg.pvalues['age']:.2g})")
                for lam in (0.0, 1.0):
                    y = la - lam * lb
                    res = slopes(d, y, cond, lam, rng, a.nboot)
                    for est in ("ols", "boot", "huber", "theilsen", "skipped", "wls",
                                "covar", "release_re", "age5_16", "quad"):
                        s, lo, hi, p = res[est]
                        rows.append(dict(sample=sample, model=model, cond=cond, lam=lam,
                                         estimator=est, est=s, lo=lo, hi=hi, p=p,
                                         n=len(d),
                                         I2=res["release_I2"] if est == "release_re" else np.nan))
                    print(f"  lambda={lam:.0f}: " + "; ".join(
                        f"{e} {res[e][0]*100:+.1f} [{res[e][1]*100:+.1f},{res[e][2]*100:+.1f}]"
                        for e in ("ols", "huber", "theilsen", "wls", "covar", "release_re",
                                  "age5_16", "quad"))
                        + f"; I2 {res['release_I2']:.2f}; skipped rho "
                          f"{res['skipped'][0]:+.2f} [{res['skipped'][1]:+.2f},"
                          f"{res['skipped'][2]:+.2f}]")
                    if sample == "qc" and model == "fixed":
                        sd = spline_derivative(d, y, rng)
                        sd["cond"], sd["lam"] = cond, lam
                        spl.append(sd)
                ratio = np.log(base[f"tot_{cond}_full"] / base[f"b_{cond}_full"])
                s, ci, p, _ = ols_hc3(ratio.to_numpy(), sm.add_constant(base.age.to_numpy()))
                rows.append(dict(sample=sample, model=model, cond=cond, lam=1.0,
                                 estimator="ln(tot/b), all", est=s, lo=ci[0], hi=ci[1], p=p,
                                 n=len(base), I2=np.nan))
                lin = base[f"a_{cond}_full"].to_numpy()
                ts = stats.theilslopes(lin, base.age.to_numpy())
                med = np.median(lin)
                rows.append(dict(sample=sample, model=model, cond=cond, lam=0.0,
                                 estimator="linear a, Theil-Sen, %/y", est=ts.slope / med,
                                 lo=ts.low_slope / med, hi=ts.high_slope / med, p=np.nan,
                                 n=len(base), I2=np.nan))
                cr = crossover(d, la.to_numpy(), lb.to_numpy(), cond, rng, a.nboot)
                cr.update(sample=sample, model=model, cond=cond, n=len(d))
                cross.append(cr)
                print(f"  ln(tot/b) {s*100:+.1f} [{ci[0]*100:+.1f},{ci[1]*100:+.1f}]; "
                      f"lambda* {cr['lam_star']:.2f} HDI [{cr['hdi_lo']:.2f},{cr['hdi_hi']:.2f}]"
                      f", adjusted {cr['lam_star_adj']:.2f} [{cr['hdi_adj_lo']:.2f},"
                      f"{cr['hdi_adj_hi']:.2f}]; slope ln b {cr['slope_ln_b']*100:+.1f}")
            # reactivity (eyes closed minus eyes open), gain-invariant
            ok = (W.a_ec_full > 0) & (W.a_eo_full > 0)
            d = W[ok]
            for lam in (0.0, 1.0):
                y = (np.log(d.a_ec_full) - lam * np.log(d.b_ec_full)) - \
                    (np.log(d.a_eo_full) - lam * np.log(d.b_eo_full))
                s, ci, p, _ = ols_hc3(y.to_numpy(), sm.add_constant(d.age.to_numpy()))
                mu, lo, hi, pr, i2, k = release_re(d, y)
                react.append(dict(sample=sample, model=model, lam=lam, est=s, lo=ci[0],
                                  hi=ci[1], p=p, re_est=mu, re_lo=lo, re_hi=hi, I2=i2,
                                  n=len(d)))
                print(f"  reactivity (EC-EO) {model} lambda={lam:.0f}: {s*100:+.1f} "
                      f"[{ci[0]*100:+.1f},{ci[1]*100:+.1f}] per year; release RE "
                      f"{mu*100:+.1f} [{lo*100:+.1f},{hi*100:+.1f}]")

    R = pd.DataFrame(rows)
    R["p_holm"] = np.nan
    for sample in ("qc", "all"):
        m = (R["sample"] == sample) & (R.estimator == "ols")
        R.loc[m, "p_holm"] = holm(R.loc[m, "p"].to_numpy())
    print("\nPrimary OLS tests, Holm-adjusted over 8 per sample:")
    print(R[R.estimator == "ols"][["sample", "model", "cond", "lam", "est", "lo", "hi",
                                    "p", "p_holm", "n"]].round(4).to_string(index=False))
    R.to_csv(os.path.join(RES, "hbn_age_alpha_robust.csv"), index=False)
    pd.DataFrame(cross).to_csv(os.path.join(RES, "hbn_age_alpha_crossover.csv"), index=False)
    pd.concat(spl).to_csv(os.path.join(RES, "hbn_age_alpha_spline.csv"), index=False)
    pd.DataFrame(react).to_csv(os.path.join(RES, "hbn_age_alpha_reactivity.csv"), index=False)


if __name__ == "__main__":
    main()
