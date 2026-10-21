"""Does the maturation of organoid low-frequency oscillatory power depend on lambda?

Trujillo et al. (2019, Cell Stem Cell 25:558) report that 2.5-4.5 Hz
oscillatory power of cortical organoids rises up to about the 25th week in
culture and then tapers off. Their measure, per channel, is the mean log PSD
minus the mean log of a robust straight-line fit over 0.5-20 Hz in log-log
space: log(T / b), a division rule (lambda = 1 family; it equals
log(1 + a / b), not the lambda = 1 measure log(a / b)). Here the same weekly
recordings (Zenodo 4751759, wells 5-12; spectra from
trujillo_extract_psd.py) are separated under several aperiodic
specifications and the maturation slope of ln a - lambda ln b is computed
with its crossover lambda*.

Timeline: recording 1 (2016-07-31) is at 8 weeks of differentiation, so
weeks since the first recording + 8 = weeks in culture. The paper's turning
point (25th week in culture) is week 17 here, and the nested 2-3 Hz
oscillation appears "from 4 months" (week ~9). Windows (--windows) default to
9-17 (rise), 17-33 (after the turning point) and 9-33 (one line through both).

Per well and recording, for the model-based specifications: the mean
spectrum over channels whose standard deviation lies within 0.2-5 x the
well median; an aperiodic model fitted by Whittle deviance to the fit bins
with the low-frequency hump censored; in the band, tot (mean observed
power), b (fitted background without the plateau), p (plateau) and
a = tot - b - p. The plateau is the flat high-frequency floor; here it rises
with maturation, so it is probably not only amplifier noise.

Specifications:
    kp_flanks        10^o/(k + f^chi) + p, bins 0.25-1 and 6-45 Hz
    plateau_flanks   10^o/f^chi + p, same bins
    *_h              the same with 6-9 Hz also censored (harmonic of ~3.5 Hz)
    fixed_0.5-20     10^o/f^chi by Whittle, 0.5-20 Hz with 1-6 Hz censored
    published        the paper's fit: per channel, bisquare robust line
                     through log10 power over 0.5-20 Hz (nothing censored);
                     b = channel mean of the fitted power in the band,
                     tot = channel mean of the observed power, and
                     osc_pub = channel mean of (mean log10 P - mean log10 fit),
                     the published oscillatory power

Maturation slopes, per well, over the recordings in a window:
    ols      least-squares slope of ln a and of ln b on weeks; recordings with
             a <= 0 are dropped (the share is reported)
    logfree  for each lambda on a grid, the slope s of a / b^lambda =
             exp(alpha + s week), from the Poisson score equations
             sum (y - mu) [1, week] = 0, which accept single values <= 0. A
             well whose mean of a / b^lambda is <= 0 has no slope at that
             lambda (n_wells_lam0, n_wells_lam1 count the wells used); roots
             of the mean curve are searched only where every well has a slope
             (grid_lo..grid_hi), all are listed, lambda* is given only when
             there is one, and its HDI only when >= 95% of the draws have one
Across wells: lambda_curve (ols: mean over wells, wells resampled; 95%
bootstrap intervals and the Bayesian-bootstrap HDI of lambda*), a t interval
with n = 8 wells, a two-way bootstrap that resamples wells and recording days
together (all wells share each recording session, and their week-to-week
fluctuations correlate at 0.3-0.9, so a bootstrap over wells alone is too
narrow; verdict_2w uses it), and, because the 8 wells are "four independent
experiments in duplicates" with the pairing undocumented (week-to-week
co-fluctuation does not reveal it), the share of the 105 ways of pairing the
wells for which the t interval on the 4 pair means excludes 0.

Checks: the empty band (a / b in the band before the rhythm, weeks < --pre,
default 6: inside events the rhythm is already measurable from week 6.3),
which should be near 0, and a control band, 10-14 Hz, left out of every fit.

Writes results/trujillo_bandpower_<band>.csv and results/trujillo_maturation[_tag].csv
(one row per specification, window and slope method).

Usage: python trujillo_maturation.py [--psd-dir DIR] [--band 2.5,4.5]
       [--windows 9-17,17-33,9-33] [--pre 9] [--min-rec 6] [--nboot 4000]
       [--tag T] [--reuse]
"""
import argparse
import datetime as dt
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "lambdacurve", "src"))
from lambdacurve import ap_eval, fit_aperiodic, hdi, lambda_curve, verdict  # noqa: E402

RES = os.path.join(ROOT, "results")
FIRST = dt.date(2016, 7, 31)
CULTURE_WEEKS_AT_FIRST = 8
SPECS = {
    "kp_flanks": dict(model="knee_plateau", bins=((0.25, 1.0), (6.0, 45.0))),
    "kp_flanks_h": dict(model="knee_plateau", bins=((0.25, 1.0), (9.0, 45.0))),
    "plateau_flanks": dict(model="plateau", bins=((0.25, 1.0), (6.0, 45.0))),
    "plateau_flanks_h": dict(model="plateau", bins=((0.25, 1.0), (9.0, 45.0))),
    "fixed_0.5-20": dict(model="fixed", bins=((0.5, 1.0), (6.0, 20.0))),
}
CONTROL = (10.0, 14.0)
# regular recordings made on the days of the drug sessions (all 8 wells were
# treated those days; the regular file may be the baseline or a washout)
DRUG_DAYS = ("2016-12-17", "2017-02-07", "2017-03-03")
LAM_GRID = np.round(np.arange(-4.0, 12.0001, 0.05), 2)


def good_channels(d, w):
    rms = d[f"rms{w}"]
    med = np.median(rms[rms > 0])
    return (rms > 0.2 * med) & (rms < 5 * med)


def well_spectrum(d, w):
    good = good_channels(d, w)
    return d[f"w{w}"][good].mean(0), int(good.sum())


def band_terms(P, f, fit, band, model):
    sel = (f >= band[0]) & (f <= band[1])
    full = ap_eval(fit["theta"], f[sel], model)
    neural = ap_eval(fit["theta"], f[sel], model, neural=True)
    tot = float(P[sel].mean())
    b = float(neural.mean())
    plat = float((full - neural).mean())
    return tot, b, plat, tot - b - plat


def published_terms(p, f, band):
    """The paper's per-channel robust line over 0.5-20 Hz (MATLAB robustfit:
    bisquare, tuning 4.685). Returns channel means of tot and b in the band,
    the published oscillatory power, and the same quantities in CONTROL."""
    fit_sel = (f >= 0.5) & (f <= 20.0)
    X = sm.add_constant(np.log10(f[fit_sel]))
    out = {k: [] for k in ("tot", "b", "osc", "ctl_tot", "ctl_b")}
    for pc in p:
        y = np.log10(np.clip(pc[fit_sel], 1e-30, None))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight(4.685)).fit().params
        for lab, (lo, hi) in (("", band), ("ctl_", CONTROL)):
            sel = (f >= lo) & (f <= hi)
            line = beta[0] + beta[1] * np.log10(f[sel])
            out[f"{lab}tot"].append(pc[sel].mean())
            out[f"{lab}b"].append(np.mean(10.0 ** line))
            if lab == "":
                out["osc"].append(np.mean(np.log10(pc[sel])) - np.mean(line))
    return {k: float(np.mean(v)) for k, v in out.items()}


def bandpower_table(psd_dir, band):
    rows = []
    for fn in sorted(glob.glob(os.path.join(psd_dir, "[0-9]*.npz"))):
        tag = os.path.basename(fn)[:-4]
        date = dt.datetime.strptime(tag[:6], "%y%m%d").date()
        d = np.load(fn)
        f = d["f"]
        wells = sorted(int(k[1:]) for k in d.files if k.startswith("w"))
        for w in wells:
            P, n_good = well_spectrum(d, w)
            base = dict(date=date, week=(date - FIRST).days / 7.0, well=w + 1, n_chan=n_good)
            for name, sp in SPECS.items():
                keep = np.zeros(f.size, bool)
                for lo, hi in sp["bins"]:
                    keep |= (f >= lo) & (f <= hi)
                fit = fit_aperiodic(P[keep], f[keep], None, sp["model"])
                if fit is None:
                    continue
                row = dict(base, spec=name, exponent=fit["exponent"],
                           knee_freq=fit["knee_freq"], plateau=fit["plateau"])
                for lab, bd in (("", band), ("ctl_", CONTROL)):
                    tot, b, plat, a = band_terms(P, f, fit, bd, sp["model"])
                    row.update({f"{lab}tot": tot, f"{lab}b": b, f"{lab}p": plat, f"{lab}a": a})
                rows.append(row)
            pub = published_terms(d[f"w{w}"][good_channels(d, w)], f, band)
            rows.append(dict(base, spec="published", tot=pub["tot"], b=pub["b"], p=0.0,
                             a=pub["tot"] - pub["b"], osc_pub=pub["osc"],
                             ctl_tot=pub["ctl_tot"], ctl_b=pub["ctl_b"], ctl_p=0.0,
                             ctl_a=pub["ctl_tot"] - pub["ctl_b"]))
    return pd.DataFrame(rows)


def ols_slopes(t, min_rec):
    """Per-well least-squares slopes of ln a and ln b; drops a <= 0."""
    out = []
    for w, g in t.groupby("well"):
        ok = (g.a > 0).to_numpy()
        if ok.sum() < min_rec:
            continue
        x = g.week.to_numpy()[ok]
        X = np.column_stack([np.ones(x.size), x])
        Y = np.column_stack([np.log(g.a.to_numpy()[ok]), np.log(g.b.to_numpy()[ok])])
        sa, sb = np.linalg.lstsq(X, Y, rcond=None)[0][1]
        out.append(dict(well=w, n_rec=int(ok.sum()), n_nonpos=int((~ok).sum()), s_a=sa, s_b=sb))
    return pd.DataFrame(out)


def poisson_slope(x, y, iters=100):
    """Slope s of y = exp(alpha + s x) from the Poisson score equations;
    y may contain values <= 0. nan if the iterations fail."""
    X = np.column_stack([np.ones(x.size), x - x.mean()])
    pos = y > 0
    if pos.sum() < 2 or y.mean() <= 0:
        return np.nan
    y = y / y.mean()                               # the slope is scale-free
    beta = np.array([0.0, 0.0])
    for _ in range(iters):
        mu = np.exp(np.clip(X @ beta, -50, 50))
        try:
            step = np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return np.nan
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            return float(beta[1])
    return np.nan


def logfree_curves(t, min_rec):
    """Per-well slope of a / b^lambda over LAM_GRID (rows: wells)."""
    curves, wells = [], []
    for w, g in t.groupby("well"):
        if len(g) < min_rec:
            continue
        x, a, b = g.week.to_numpy(), g.a.to_numpy(), g.b.to_numpy()
        b = b / np.exp(np.mean(np.log(b)))          # constant factors leave slopes unchanged
        curves.append([poisson_slope(x, a / b ** lam) for lam in LAM_GRID])
        wells.append(w)
    return np.array(curves), wells


def roots(curve, grid=None):
    """All sign changes of a curve over grid (linear interpolation)."""
    grid = LAM_GRID if grid is None else grid
    s = np.sign(curve)
    idx = np.where(np.isfinite(curve[:-1]) & np.isfinite(curve[1:]) & (s[:-1] * s[1:] < 0))[0]
    return [float(grid[i] - curve[i] * (grid[i + 1] - grid[i]) / (curve[i + 1] - curve[i]))
            for i in idx]


def crossing(curve, grid=None):
    """The single sign change of a curve; nan when there is none or several."""
    r = roots(curve, grid)
    return r[0] if len(r) == 1 else np.nan


def t_interval(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
    q = stats.t.ppf(0.975, v.size - 1)
    return m, m - q * se, m + q * se


def pairings(items):
    """All partitions of an even-sized list into pairs."""
    if not items:
        yield []
        return
    first = items[0]
    for i in range(1, len(items)):
        rest = items[1:i] + items[i + 1:]
        for p in pairings(rest):
            yield [(first, items[i])] + p


def pairing_share(values, wells):
    """Share of the pairings of the wells for which the t interval of the
    pair means excludes 0, and the share with a positive and a negative mean."""
    if len(wells) != 8:
        return np.nan, np.nan
    v = dict(zip(wells, values))
    excl, pos = [], []
    for p in pairings(list(wells)):
        m = [(v[i] + v[j]) / 2 for i, j in p]
        _, lo, hi = t_interval(m)
        excl.append(lo > 0 or hi < 0)
        pos.append(np.mean(m) > 0)
    return float(np.mean(excl)), float(np.mean(pos))


def twoway_interval(t, lam, method, nboot=2000, seed=2):
    """95% interval of the mean per-well slope of ln a - lam ln b (ols) or of
    a / b^lam (logfree), resampling wells and recording days together. All
    wells share each recording session, so day-level shocks are common to
    them and a bootstrap over wells alone misses that part of the error."""
    rng = np.random.default_rng(seed)
    wells = np.array(sorted(t.well.unique()))
    days = np.array(sorted(t.week.unique()))
    tab = {(w, d): r for w, d, r in zip(t.well, t.week, t[["a", "b"]].to_numpy())}
    est = []
    for _ in range(nboot):
        ws, ds = rng.choice(wells, wells.size), rng.choice(days, days.size)
        sl = []
        for w in ws:
            rows = [(d, *tab[(w, d)]) for d in ds if (w, d) in tab]
            if len(rows) < 4:
                continue
            x, a, b = (np.array(v) for v in zip(*rows))
            if np.unique(x).size < 3:
                continue
            if method == "ols":
                ok = a > 0
                if ok.sum() < 4 or np.unique(x[ok]).size < 3:
                    continue
                sl.append(np.polyfit(x[ok], np.log(a[ok]) - lam * np.log(b[ok]), 1)[0])
            else:
                b = b / np.exp(np.mean(np.log(b)))
                sl.append(poisson_slope(x, a / b ** lam))
        if sl:
            est.append(np.nanmean(sl))
    return tuple(float(v) for v in np.nanpercentile(est, [2.5, 97.5]))


def analyse(t, pre, spec, window, min_rec, nboot):
    rows = []
    ctl = t.ctl_a / t.ctl_b
    common = dict(spec=spec, window=window, n_rec=len(t), share_a_nonpos=float((t.a <= 0).mean()),
                  ctl_a_over_b_median=float(ctl.median()),
                  n_pre=len(pre), pre_a_over_b_median=float((pre.a / pre.b).median()))
    # ols on ln a, ln b
    sl = ols_slopes(t, min_rec)
    if len(sl) >= 3:
        r = lambda_curve(sl.s_a, sl.s_b, nboot=nboot, seed=0)
        row = dict(common, method="ols", n_wells=len(sl), wells_dropped_rec=int(sl.n_nonpos.sum()))
        row.update(r.to_dict())
        for lam in (0, 1):
            v = sl.s_a - lam * sl.s_b
            m, lo, hi = t_interval(v)
            row.update({f"lam{lam}_t_lo": lo, f"lam{lam}_t_hi": hi})
            row[f"lam{lam}_pairs_excl0"], row[f"lam{lam}_pairs_pos"] = pairing_share(v, list(sl.well))
            row[f"lam{lam}_2w_lo"], row[f"lam{lam}_2w_hi"] = twoway_interval(t, lam, "ols", nboot // 2)
        row["s_b_pairs_excl0"], _ = pairing_share(sl.s_b, list(sl.well))
        rows.append(row)
    # log-free. A well's slope is undefined where the mean of a / b^lambda is
    # <= 0 (it is then left out at that lambda) or where the fit diverges; the
    # crossover is searched only where every well has a slope, and all roots
    # there are reported.
    C, wells = logfree_curves(t, min_rec)
    if len(wells) >= 3:
        rng = np.random.default_rng(1)
        i0, i1 = np.searchsorted(LAM_GRID, 0.0), np.searchsorted(LAM_GRID, 1.0)
        full = np.all(np.isfinite(C), 0)
        grid = LAM_GRID[full]
        Cf = C[:, full]
        r_all = roots(Cf.mean(0), grid)
        boot = np.array([np.nanmean(C[rng.integers(0, len(wells), len(wells))], 0)
                         for _ in range(nboot)])
        draws = [roots(rng.dirichlet(np.ones(len(wells))) @ Cf, grid) for _ in range(nboot)]
        one = np.array([len(d) == 1 for d in draws])
        first = np.array([d[0] if len(d) == 1 else np.nan for d in draws])
        row = dict(common, method="logfree", n_wells=len(wells), wells_dropped_rec=0,
                   grid_lo=float(grid.min()) if grid.size else np.nan,
                   grid_hi=float(grid.max()) if grid.size else np.nan,
                   lam_star=r_all[0] if len(r_all) == 1 else np.nan,
                   roots=";".join(f"{v:.2f}" for v in r_all),
                   share_single_root=float(one.mean()),
                   p_cross_in_01=float(np.mean([any(0 <= v <= 1 for v in d) for d in draws])))
        row["hdi_lo"], row["hdi_hi"] = hdi(first) if one.mean() >= 0.95 else (np.nan, np.nan)
        for lam, i in ((0, i0), (1, i1)):
            ok = np.isfinite(C[:, i])
            lo, hi = np.nanpercentile(boot[:, i], [2.5, 97.5])
            m, tlo, thi = t_interval(C[:, i])
            row.update({f"lam{lam}": float(np.nanmean(C[:, i])), f"lam{lam}_lo": lo,
                        f"lam{lam}_hi": hi, f"lam{lam}_t_lo": tlo, f"lam{lam}_t_hi": thi,
                        f"n_wells_lam{lam}": int(ok.sum())})
            if ok.all():
                row[f"lam{lam}_pairs_excl0"], row[f"lam{lam}_pairs_pos"] = pairing_share(C[:, i], wells)
            row[f"lam{lam}_2w_lo"], row[f"lam{lam}_2w_hi"] = twoway_interval(t, lam, "logfree",
                                                                             nboot // 2)
        # s_b: slope of b alone (lambda = 1 minus lambda = 0 is not s_b off the log scale)
        sb = np.array([poisson_slope(g.week.to_numpy(), g.b.to_numpy())
                       for _, g in t.groupby("well") if len(g) >= min_rec])
        row["s_b"], row["s_b_lo"], row["s_b_hi"] = t_interval(sb)
        row["verdict"] = verdict(row)
        rows.append(row)
    for row in rows:
        row["verdict_2w"] = verdict({f"lam{k}_{e}": row[f"lam{k}_2w_{e}"]
                                     for k in (0, 1) for e in ("lo", "hi")})
    return rows


def published_measure(bp, windows):
    """The paper's own oscillatory power (channel mean of mean log10 PSD minus
    mean log10 robust line, 2.5-4.5 Hz; 4-s windows here, 2-s in the paper):
    per-well least-squares slope per week, mean with a t interval over wells."""
    t = bp[bp.spec == "published"]
    print("Published measure (log10 ratio), per-well slope per week:")
    for win in windows.split(","):
        w0, w1 = (float(v) for v in win.split("-"))
        x = t[(t.week >= w0) & (t.week <= w1)]
        sl = [np.polyfit(g.week, g.osc_pub, 1)[0] for _, g in x.groupby("well") if len(g) >= 4]
        m, lo, hi = t_interval(sl)
        print(f"  weeks {win}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  wells > 0: "
              f"{int(np.sum(np.array(sl) > 0))}/{len(sl)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.path.join(RES, "trujillo_psd"))
    ap.add_argument("--band", default="2.5,4.5")
    ap.add_argument("--windows", default="9-17,17-33,9-33")
    ap.add_argument("--pre", type=float, default=6.0)
    ap.add_argument("--drop-drug-days", action="store_true",
                    help="leave out the regular recordings made on drug-session days")
    ap.add_argument("--min-rec", type=int, default=6)
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--tag", default="", help="suffix of the output table")
    ap.add_argument("--reuse", action="store_true",
                    help="read results/trujillo_bandpower_<band>.csv instead of refitting")
    a = ap.parse_args()
    band = tuple(float(v) for v in a.band.split(","))
    bp_file = os.path.join(RES, f"trujillo_bandpower_{a.band.replace(',', '-')}.csv")
    if a.reuse:
        bp = pd.read_csv(bp_file, parse_dates=["date"])
    else:
        bp = bandpower_table(a.psd_dir, band)
        bp.to_csv(bp_file, index=False)

    if a.drop_drug_days:
        bp = bp[~pd.to_datetime(bp.date).dt.strftime("%Y-%m-%d").isin(DRUG_DAYS)]
    published_measure(bp, a.windows)
    rows = []
    for spec, t_all in bp.groupby("spec", sort=False):
        pre = t_all[t_all.week < a.pre]
        for win in a.windows.split(","):
            w0, w1 = (float(v) for v in win.split("-"))
            t = t_all[(t_all.week >= w0) & (t_all.week <= w1)]
            rows += analyse(t, pre, spec, win, a.min_rec, a.nboot)
    out = pd.DataFrame(rows)
    tag = f"_{a.tag}" if a.tag else ""
    out.to_csv(os.path.join(RES, f"trujillo_maturation{tag}.csv"), index=False)
    cols = ["spec", "window", "method", "n_wells", "share_a_nonpos", "pre_a_over_b_median",
            "s_b", "lam0", "lam0_lo", "lam0_hi", "lam1", "lam1_lo", "lam1_hi", "lam_star",
            "hdi_lo", "hdi_hi", "lam0_2w_lo", "lam0_2w_hi", "lam1_2w_lo", "lam1_2w_hi",
            "lam0_pairs_excl0", "lam1_pairs_excl0", "verdict", "verdict_2w"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
