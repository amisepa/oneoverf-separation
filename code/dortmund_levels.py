"""Coupling exponent from segment-to-segment fluctuations within a recording
(Dortmund Vital Study), before and after a 2-hour task battery.

Input: dortmund_extract_psd.py output (per-segment posterior ROI spectra,
4-s Hann, 0.25 Hz; kept segments only). Within each segment three nearly
independent estimates come from disjoint frequency bins: the band total T
from the bins of the alpha band (individual alpha frequency +/- 2 Hz, from
the recording's mean spectrum); the background B inside a = T - B from a
power law fitted to the two flanks of the band only, 2-6 and 26-32 Hz (the
gap leaves out the alpha peak, its harmonic and the beta range), on every
sixth bin, corrected for the log of a gamma-distributed estimate
(ap_models.loglog_fit) and rescaled per participant so that its mean over
segments equals the participant's mean spectrum interpolated across the gap
(ap_models.flank_level); the instrument and weights from the same fit on the
interleaved bins half-way between. The two sets are 3 bins apart because
Hann-windowed power is correlated between bins 2 apart (1/36), which would
add to the covariance of the background with the instrument. Both
assignments of the two bin sets are pooled. lambda is estimated with
lambda_gmm.estimate_levels (participant-specific intrinsic strength), with a
participant-cluster bootstrap, for the eyes-closed recordings before and
after the task battery (session 1): a stable property should give the same
lambda-hat in both, although arousal differs.

--background power-law gives the earlier background instead: one censored
log-log fit over 2-45 Hz with 6-16 Hz (or 6 to --censor-hi) left out, the
shape of the gamma distribution estimated per recording, on every fourth bin
with the instrument on the bins 2 further on; its output files carry the
upper edge of the censored band (_c16, _c26).

With the flank background and the real data, each recording's rows are also
written to results/within_recording_variants.csv (other choices of flanks)
and results/within_recording_controls.csv (two bands without a rhythm), as
described in ds003690_lambda.py.

--simulate L builds synthetic segments from each participant's own mean
background, alpha peak and segment-to-segment background fluctuation, with
coupling L, exponential noise per bin and the same analysis.

Usage: python dortmund_levels.py [--data DIR] [--nboot 200] [--simulate L]
       [--cond ec_pre] [--max-subjects N] [--background flanks|power-law]
       [--censor-hi 16]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lambda_gmm as G
from ap_models import flank_bins, flank_level, loglog_fit
from ds003690_lambda import (CONTROLS, FLANKS, VARIANTS, control_row, merge_rows, tracking,
                             variant_row)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIT = (2.0, 45.0)
CENSOR = (6.0, 16.0)


def fit_sets(f, flanks=None):
    """Two interleaved sets of fit bins: every 6th bin of the flanks, offset
    by 3; with flanks None, every 4th bin of the censored fit range, offset
    by 2."""
    if flanks is not None:
        idx = np.where(flank_bins(f, flanks))[0]
        return idx[idx % 6 == 0], idx[idx % 6 == 3]
    idx = np.where((f >= FIT[0]) & (f <= FIT[1]) & ~((f >= CENSOR[0]) & (f <= CENSOR[1])))[0]
    return idx[idx % 4 == 0], idx[idx % 4 == 2]


def lnL(P, f, bins):
    X = np.column_stack([np.ones(bins.size), np.log(f[bins])])
    beta = loglog_fit(P[:, bins].T, X)
    return beta[0][:, None] + beta[1][:, None] * np.log(np.maximum(f, 1e-9))[None, :]


def iaf(Pm, f):
    b = np.where((f >= FIT[0]) & (f <= FIT[1]) & ~((f >= CENSOR[0]) & (f <= CENSOR[1])))[0]
    X = np.column_stack([np.ones(b.size), np.log(f[b])])
    beta = np.linalg.lstsq(X, np.log(Pm[b]), rcond=None)[0]
    s = (f >= 7) & (f <= 13)
    resid = np.log(Pm[s]) - (beta[0] + beta[1] * np.log(f[s]))
    return float(f[s][np.argmax(resid)]) if np.max(resid) > 0.1 else np.nan


def load(data, cond, max_subjects):
    units = []
    for p in sorted(glob.glob(os.path.join(data, "*_ses-1.npz"))):
        d = np.load(p, allow_pickle=True)
        if f"{cond}_seg_post" not in d:
            continue
        keep = d[f"{cond}_keep"].astype(bool)
        if keep.sum() < 20:
            continue
        units.append(dict(subject=str(d["subject"]), age=float(d["age"]),
                          f=d["freqs"].astype(float), P=d[f"{cond}_seg_post"][keep].astype(float),
                          fp=d[f"{cond}_seg_fp"][keep].astype(float),
                          temp=d[f"{cond}_seg_temp"][keep].astype(float)))
        if max_subjects and len(units) >= max_subjects:
            break
    return units


def simulate(units, lam, rng):
    out = []
    for u in units:
        f, P = u["f"], u["P"]
        cf = iaf(P.mean(0), f)
        if not np.isfinite(cf):
            continue
        s0, s2 = fit_sets(f)
        L0 = lnL(P, f, s0)
        i = np.argmin(np.abs(f - cf))
        lev = L0[:, i]
        L2 = lnL(P, f, s2)[:, i]
        noise_var = np.var(lev - L2) / 2
        sd = np.sqrt(max(np.var(lev) - noise_var, 1e-4))
        mean_lnL = L0.mean(0)
        Lcf = np.exp(mean_lnL[i])
        a_rel = max(P.mean(0)[i] / Lcf - 1, 0.1)
        c = a_rel * Lcf ** (1 - lam)
        g = np.exp(-0.5 * ((f - cf) / 1.5) ** 2)
        n = P.shape[0]
        eta = rng.normal(0, sd, n)
        eps = rng.normal(0, 0.3, n)
        L = np.exp(mean_lnL[None, :] + eta[:, None])
        S = L + c * np.exp(eps)[:, None] * g[None, :] * L ** lam
        v = dict(u)
        v["P"] = S * rng.exponential(1.0, S.shape)
        out.append(v)
    return out


def arrays(units, use_cov, flanks=None, fixed=None):
    """T, B, Bz, participant labels and covariates, both assignments of the
    two bin sets stacked. flanks: background from these two flanks
    (ap_models.flank_level on each bin set); None: the censored power law.
    fixed: band (lo, hi) in Hz; None: individual alpha frequency +/- 2 Hz.
    Participants are those with an alpha peak in either case."""
    T, B, Bz, grp, X = [], [], [], [], []
    for u in units:
        f, P = u["f"], u["P"]
        cf = iaf(P.mean(0), f)
        if not np.isfinite(cf):
            continue
        lo, hi = (cf - 2, cf + 2) if fixed is None else fixed
        band = (f >= lo) & (f <= hi)
        if flanks is None:
            Lb = [np.exp(lnL(P, f, s))[:, band].mean(1) for s in fit_sets(f)]
        else:
            Lb = [flank_level(P, f, s, band, flanks) for s in fit_sets(f, flanks)]
        t = P[:, band].mean(1)
        C = None
        if use_cov:
            C = np.column_stack([np.log(u["fp"][:, (f >= 1) & (f <= 4)].mean(1)),
                                 np.log(u["temp"][:, (f >= 60) & (f <= 95)].mean(1))])
            C = (C - C.mean(0)) / np.where(C.std(0) > 0, C.std(0), 1)
        for bi, zi in ((0, 1), (1, 0)):
            T.append(t); B.append(Lb[bi]); Bz.append(Lb[zi])
            grp.append(np.full(t.size, u["subject"]))
            if use_cov:
                X.append(C)
    return (np.concatenate(T), np.concatenate(B), np.concatenate(Bz), np.concatenate(grp),
            np.concatenate(X) if use_cov else None)


GRID = np.linspace(-1.0, 3.0, 161)


def run(units, use_cov, nboot, rng, flanks=None, fixed=None):
    T, B, Bz, grp, X = arrays(units, use_cov, flanks, fixed)
    est = G.estimate_levels(T, B, Bz, grp, X, grid=GRID)
    subs = np.unique(grp)
    idx = {s: np.where(grp == s)[0] for s in subs}
    bs = []
    for _ in range(nboot):
        pick = rng.choice(subs, subs.size, replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        g2 = np.concatenate([np.full(idx[s].size, f"{s}_{j}") for j, s in enumerate(pick)])
        e = G.estimate_levels(T[rows], B[rows], Bz[rows], g2, None if X is None else X[rows],
                              grid=GRID)
        r = e["roots"]
        bs.append(min(r, key=lambda x: abs(x - est["lam"])) if r and np.isfinite(est["lam"])
                  else e["lam"])
    bs = np.array(bs, float)
    ok = np.isfinite(bs)
    lo, hi = np.percentile(bs[ok], [2.5, 97.5]) if ok.sum() > 10 else (np.nan, np.nan)
    rel = np.mean([np.corrcoef(np.log(B[grp == s][: idx[s].size // 2]),
                               np.log(Bz[grp == s][: idx[s].size // 2]))[0, 1] for s in subs])
    return dict(lam=est["lam"], lo=lo, hi=hi, fail=float(1 - ok.mean()), n_units=int(subs.size),
                n_segments=int(T.size // 2), reliability=float(rel),
                gamma=np.round(est["gamma"], 3).tolist(), n_roots=len(est["roots"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DORTMUND_OUT", "dortmund_psd"))
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--simulate", type=float, default=None)
    ap.add_argument("--cond", default="ec_pre,ec_post")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--background", choices=("flanks", "power-law"), default="flanks",
                    help="flanks: power law through 2-6 and 26-32 Hz, rescaled to the mean "
                         "spectrum; power-law: one censored fit over 2-45 Hz")
    ap.add_argument("--censor-hi", type=float, default=16.0,
                    help="upper edge of the censored band of the power-law background")
    ap.add_argument("--no-cov", action="store_true")
    a = ap.parse_args()
    flanks = FLANKS if a.background == "flanks" else None
    global CENSOR
    if flanks is None:
        CENSOR = (6.0, a.censor_hi)
    rng = np.random.default_rng(a.seed)
    out, var, ctl = [], [], []
    for cond in a.cond.split(","):
        units = load(a.data, cond, a.max_subjects)
        if a.simulate is not None:
            units = simulate(units, a.simulate, rng)
        for use_cov in ((False,) if (a.simulate is not None or a.no_cov) else (False, True)):
            r = run(units, use_cov, a.nboot, rng, flanks)
            r.update(cond=cond, covariates=use_cov,
                     sample="data" if a.simulate is None else f"simulated {a.simulate}")
            if flanks is None:
                r.update(censor_hi=a.censor_hi)
            out.append(r)
            print(f"{r['sample']} {cond} covariates={use_cov}: lambda {r['lam']:+.3f} "
                  f"[{r['lo']:+.2f}, {r['hi']:+.2f}] fail {r['fail']:.2f} "
                  f"n {r['n_units']}/{r['n_segments']} bin-set reliability {r['reliability']:.2f} "
                  f"gamma {r['gamma']}", flush=True)
        if flanks is None or a.simulate is not None or a.max_subjects:
            continue
        # other flanks, and bands without a rhythm; no covariates
        key = dict(dataset="Dortmund", recording=cond)
        first = next(r for r in out if r["cond"] == cond and not r["covariates"])
        var.append(variant_row(key, FLANKS, first, first["reliability"]))
        for fl in VARIANTS:
            r = run(units, False, a.nboot, rng, fl)
            var.append(variant_row(key, fl, r, r["reliability"]))
        for band, fl in CONTROLS:
            r = run(units, False, a.nboot, rng, fl, band)
            t = tracking(*arrays(units, False, fl, band)[:4], a.nboot, rng)
            ctl.append(control_row(key, band, fl, r, t, r["reliability"]))
    if var:
        merge_rows("within_recording_variants.csv", var)
        merge_rows("within_recording_controls.csv", ctl)
    tag = "" if flanks else f"_c{a.censor_hi:.0f}"
    name = (f"dortmund_levels{tag}.csv" if a.simulate is None
            else f"dortmund_levels_sim{a.simulate}{tag}.csv")
    pd.DataFrame(out).to_csv(os.path.join(RES, name), index=False)


if __name__ == "__main__":
    main()
