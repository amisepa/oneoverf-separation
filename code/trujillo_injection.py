"""Injected-rhythm checks of the organoid maturation pipeline.

The recordings before the nested oscillation appears (weeks < 6 since the
first recording, 6 dates x 8 wells; inside events the rhythm is measurable
from week 6.3) have no rhythm in 2.5-4.5 Hz, so their band power is
background. A rhythm of known size added to them gives a
known truth, with the real spectral shape and recording-to-recording
variability (cf. inject_calibration.py).

1. Recovery. Into each pre-rhythm spectrum, add a Gaussian in linear power,
   narrow (centre 3.25 Hz, s.d. 0.4 Hz) or broad (centre 2.5 Hz, s.d. 0.9 Hz,
   reaching into the low flank like the real hump), scaled so that its mean
   over the band is r times the band power, r in R. Per specification:
   recovered share (a_hat - a_hat0) / a_true, change of the fitted
   background (b_hat - b_hat0) / b_hat0, and the log transfer between
   neighbouring sizes, d ln a_hat / d ln a_true (1 when a log-slope of a is
   recovered without bias).

2. End to end. Per well, the 11 pre-rhythm spectra in random order serve as
   the backgrounds of a synthetic series over the weeks of a window, each
   multiplied by exp(s_b t) (a background change by gain, shape fixed), plus
   a broad rhythm with a prescribed a/b. Two regimes, after the observed
   ones: "rise" (weeks 9-17, s_b = 0.035/week, a/b from 0.05 to 0.45: true
   lambda = 0 slope 0.31, lambda = 1 slope 0.27 per week, lambda* 8.9) and
   "after" (weeks 17-33, s_b = 0.035/week, a/b held at 0.3: true slopes
   0.035 and 0, lambda* = 1).
   The pipeline's per-well slopes (ols and logfree) are compared with the
   truth, over --reps random orders.

The published per-channel robust fit is run on --pub-chan channels per well
to keep the run short.

Writes results/trujillo_injection_recovery.csv and results/trujillo_injection_e2e.csv.

Usage: python trujillo_injection.py [--reps 40] [--pub-chan 16]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trujillo_maturation import (FIRST, LAM_GRID, RES, SPECS, band_terms, crossing,  # noqa: E402
                                 fit_aperiodic, good_channels, logfree_curves, ols_slopes,
                                 published_terms)
import datetime as dt  # noqa: E402

BAND = (2.5, 4.5)
SHAPES = {"narrow": (3.25, 0.4), "broad": (2.5, 0.9)}
R = (0.1, 0.25, 0.5, 1.0, 2.0)


def load_pre(psd_dir, pre_week=6.0):
    out = []
    for fn in sorted(glob.glob(os.path.join(psd_dir, "[0-9]*.npz"))):
        date = dt.datetime.strptime(os.path.basename(fn)[:6], "%y%m%d").date()
        week = (date - FIRST).days / 7.0
        if week >= pre_week:
            continue
        d = np.load(fn)
        for w in range(4, 12):
            g = good_channels(d, w)
            out.append(dict(week=week, well=w + 1, f=d["f"], chans=d[f"w{w}"][g]))
    return out


def gauss(f, shape):
    cf, sd = SHAPES[shape]
    return np.exp(-0.5 * ((f - cf) / sd) ** 2)


def estimate(spec, f, P=None, chans=None):
    """(tot, b, a) in BAND for one specification."""
    if spec == "published":
        r = published_terms(chans, f, BAND)
        return r["tot"], r["b"], r["tot"] - r["b"]
    sp = SPECS[spec]
    keep = np.zeros(f.size, bool)
    for lo, hi in sp["bins"]:
        keep |= (f >= lo) & (f <= hi)
    fit = fit_aperiodic(P[keep], f[keep], None, sp["model"])
    tot, b, plat, a = band_terms(P, f, fit, BAND, sp["model"])
    return tot, b, a


def inject(f, base_band, r, shape):
    g = gauss(f, shape)
    sel = (f >= BAND[0]) & (f <= BAND[1])
    return g * (r * base_band / g[sel].mean())


def recovery(pre, specs, pub_chan):
    rows = []
    for k, s in enumerate(pre):
        f, ch = s["f"], s["chans"]
        sub = ch[np.linspace(0, len(ch) - 1, min(pub_chan, len(ch))).astype(int)]
        P = ch.mean(0)
        sel = (f >= BAND[0]) & (f <= BAND[1])
        for spec in specs:
            base = estimate(spec, f, P, sub)
            for shape in SHAPES:
                for r in R:
                    if spec == "published":
                        add = np.array([inject(f, c[sel].mean(), r, shape) for c in sub])
                        est = estimate(spec, f, None, sub + add)
                        a_true = float(add[:, sel].mean())
                    else:
                        add = inject(f, P[sel].mean(), r, shape)
                        est = estimate(spec, f, P + add)
                        a_true = float(add[sel].mean())
                    rows.append(dict(spec=spec, shape=shape, r=r, well=s["well"], week=s["week"],
                                     a0=base[2], b0=base[1], a=est[2], b=est[1], a_true=a_true))
    return pd.DataFrame(rows)


def summarise_recovery(df):
    df = df.assign(recovered=(df.a - df.a0) / df.a_true, db=(df.b - df.b0) / df.b0,
                   ratio=df.a / df.a_true)
    out = df.groupby(["spec", "shape", "r"]).agg(
        recovered=("recovered", "median"), db=("db", "median"),
        a_hat_over_a_true=("ratio", "median")).reset_index()
    trans = []
    for (spec, shape), x in df.groupby(["spec", "shape"]):
        med = x.groupby("r")[["a", "a_true", "b"]].median()
        la, lt, lb = np.log(med.a.clip(lower=1e-30)), np.log(med.a_true), np.log(med.b)
        for i in range(len(med) - 1):
            trans.append(dict(spec=spec, shape=shape, r=med.index[i + 1],
                              transfer_a=(la.iloc[i + 1] - la.iloc[i]) / (lt.iloc[i + 1] - lt.iloc[i]),
                              transfer_b=(lb.iloc[i + 1] - lb.iloc[i]) / (lt.iloc[i + 1] - lt.iloc[i])))
    return out.merge(pd.DataFrame(trans), on=["spec", "shape", "r"], how="left")


def e2e(pre, specs, regime, reps, pub_chan, seed):
    rng = np.random.default_rng(seed)
    if regime == "rise":
        weeks = np.arange(9.0, 17.01, 1.0)
        ratio = np.exp(np.linspace(np.log(0.05), np.log(0.45), weeks.size))
    else:
        weeks = np.arange(17.0, 33.01, 2.0)
        ratio = np.full(weeks.size, 0.3)
    s_b = 0.035
    gain = np.exp(s_b * (weeks - weeks[0]))
    # a(t) = c(t) b(t)^lam with a/b = ratio: the truth for ln a is ln ratio + ln b
    s_a_true = np.polyfit(weeks, np.log(ratio * gain), 1)[0]
    by_well = {}
    for s in pre:
        by_well.setdefault(s["well"], []).append(s)
    rows = []
    for rep in range(reps):
        recs = []
        for well, items in by_well.items():
            order = rng.permutation(len(items))
            for j, wk in enumerate(weeks):
                s = items[order[j % len(items)]]
                f, ch = s["f"], s["chans"]
                sel = (f >= BAND[0]) & (f <= BAND[1])
                for spec in specs:
                    if spec == "published":
                        sub = ch[np.linspace(0, len(ch) - 1, min(pub_chan, len(ch))).astype(int)] * gain[j]
                        add = np.array([inject(f, c[sel].mean(), ratio[j], "broad") for c in sub])
                        tot, b, a = estimate(spec, f, None, sub + add)
                    else:
                        P = ch.mean(0) * gain[j]
                        tot, b, a = estimate(spec, f, P + inject(f, P[sel].mean(), ratio[j], "broad"))
                    recs.append(dict(spec=spec, well=well, week=wk, a=a, b=b,
                                     ctl_a=np.nan, ctl_b=np.nan))
        t_all = pd.DataFrame(recs)
        for spec, t in t_all.groupby("spec"):
            sl = ols_slopes(t, 5)
            C, wells = logfree_curves(t, 5)
            mean = np.nanmean(C, 0)
            i0, i1 = np.searchsorted(LAM_GRID, 0.0), np.searchsorted(LAM_GRID, 1.0)
            rows.append(dict(regime=regime, rep=rep, spec=spec,
                             s_a_true=s_a_true, s_b_true=s_b,
                             ols_s_a=sl.s_a.mean() if len(sl) else np.nan,
                             ols_s_b=sl.s_b.mean() if len(sl) else np.nan,
                             ols_dropped=int(sl.n_nonpos.sum()) if len(sl) else np.nan,
                             lf_lam0=mean[i0], lf_lam1=mean[i1], lf_lam_star=crossing(mean)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.path.join(RES, "trujillo_psd"))
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--pub-chan", type=int, default=16)
    ap.add_argument("--skip-recovery", action="store_true")
    a = ap.parse_args()
    pre = load_pre(a.psd_dir)
    specs = list(SPECS) + ["published"]
    print(len(pre), "pre-rhythm spectra")
    if not a.skip_recovery:
        rec = recovery(pre, specs, a.pub_chan)
        summ = summarise_recovery(rec)
        summ.to_csv(os.path.join(RES, "trujillo_injection_recovery.csv"), index=False)
        with pd.option_context("display.width", 200):
            print(summ.round(3).to_string(index=False))
    out = pd.concat([e2e(pre, specs, reg, a.reps, a.pub_chan, seed)
                     for seed, reg in enumerate(("rise", "after"))])
    out.to_csv(os.path.join(RES, "trujillo_injection_e2e.csv"), index=False)
    g = out.groupby(["regime", "spec"])
    summ = g[["s_a_true", "s_b_true", "ols_s_a", "ols_s_b", "ols_dropped", "lf_lam0",
              "lf_lam1", "lf_lam_star"]].median()
    with pd.option_context("display.width", 200):
        print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
