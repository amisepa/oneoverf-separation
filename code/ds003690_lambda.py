"""Coupling exponent from epoch-to-epoch fluctuations within a session.

Data: ds003690_epochs.py output (2-s eyes-open epochs, posterior ROI, three
DPSS tapers per epoch, pupil, EOG and temporal high-frequency power).

Alpha band = individual alpha frequency +/- 2 Hz (from the participant's
mean spectrum). Per epoch and taper the aperiodic background in the band
comes from a power law fitted by least squares on ln power to the two flanks
of the band only, 2-6 and 26-32 Hz (the gap leaves out the alpha peak, its
harmonic and the beta range), corrected for the log of a gamma-distributed
spectral estimate (ap_models.loglog_fit) and rescaled per participant so
that its mean over epochs equals the participant's mean spectrum
interpolated across the gap (ap_models.flank_level). --background power-law
gives the earlier background instead: one power law over 2-40 Hz with 6-16 Hz
left out, the shape of the gamma distribution (1 for one periodogram, more
for the mean of several channels) estimated per participant from the
residual variance.
Band total T comes from one taper, the background B inside a = T - B from a
second and the instrument/weights from a third; all six assignments are
pooled. lambda is estimated with lambda_gmm.estimate_levels (participant-
specific intrinsic strength, no a > 0 needed), without and with the measured
confounders (pupil diameter, VEOG and HEOG power, temporal 60-95 Hz power),
with a participant-cluster bootstrap. For comparison: the within-participant
slope of ln a on ln b over epochs with a > 0, instrumented by the third
taper's ln b.

With the flank background and the real data, two further tables get this
recording's rows (those of other recordings are kept):
results/within_recording_variants.csv, the estimate without covariates for
the primary flanks and for four other choices of flanks, with the
reliability of the background across tapers; and
results/within_recording_controls.csv, the same analysis in two bands
without a rhythm (31-35 and 36-40 Hz, flanks 4 Hz below and 6 Hz above a
12-Hz gap, 4 Hz above for the upper band): the relative residual
sum(T - B) / sum(B), the slope with which ln T follows ln B (tracking), and
the root of the moment condition, if any, with the share of bootstrap
samples without one.

--simulate L replaces the data by synthetic epochs built from each
participant's own mean background, alpha peak and epoch-to-epoch background
fluctuation, with coupling L and the same three-taper noise, to check that
the analysis recovers L on data like these.

Usage: python ds003690_lambda.py [--data DIR] [--nboot 200] [--simulate L]
       [--background flanks|power-law]
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lambda_gmm as G
from ap_models import flank_bins, flank_level, loglog_fit

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIT = (2.0, 40.0)
CENSOR = (6.0, 16.0)
EULER = 0.5772156649
GRID = np.linspace(-1.0, 3.0, 161)
FLANKS = ((2.0, 6.0), (26.0, 32.0))        # fit bins of the alpha background
VARIANTS = (((2.0, 6.0), (26.0, 36.0)), ((3.0, 6.0), (26.0, 32.0)),
            ((2.0, 5.0), (26.0, 32.0)), ((2.0, 6.0), (28.0, 36.0)))
# bands without a rhythm, with their flanks
CONTROLS = (((31.0, 35.0), ((23.0, 27.0), (39.0, 45.0))),
            ((36.0, 40.0), ((28.0, 32.0), (44.0, 48.0))))


def fit_background(P, f):
    """Vectorised censored log-log fit to the spectra of one participant;
    P (..., n_f). Returns ln L (..., n_f)."""
    keep = (f >= FIT[0]) & (f <= FIT[1]) & ~((f >= CENSOR[0]) & (f <= CENSOR[1]))
    X = np.column_stack([np.ones(keep.sum()), np.log(f[keep])])
    beta = loglog_fit(P[..., keep].reshape(-1, keep.sum()).T, X)        # 2 x N
    lf = np.log(np.maximum(f, 1e-9))
    lnL = beta[0][:, None] + beta[1][:, None] * lf[None, :]
    return lnL.reshape(P.shape)


def iaf(Pm, f):
    """Peak frequency in 7-13 Hz of the mean spectrum, if its ln power exceeds
    the censored least-squares log-log line + Euler's constant by 0.1."""
    keep = (f >= FIT[0]) & (f <= FIT[1]) & ~((f >= CENSOR[0]) & (f <= CENSOR[1]))
    X = np.column_stack([np.ones(keep.sum()), np.log(f[keep])])
    beta = np.linalg.lstsq(X, np.log(np.maximum(Pm[keep], 1e-30)), rcond=None)[0]
    s = (f >= 7) & (f <= 13)
    resid = np.log(Pm[s]) - (beta[0] + EULER + beta[1] * np.log(f[s]))
    return float(f[s][np.argmax(resid)]) if np.max(resid) > 0.1 else np.nan


def load(data):
    units = []
    for p in sorted(glob.glob(os.path.join(data, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        units.append(dict(subject=str(d["subject"]), group=str(d["group"]), age=float(d["age"]),
                          f=d["freqs"].astype(float), P=d["post"].astype(float),
                          cov=np.column_stack([d["pupil"], d["veog"], d["heog"], d["emg"]])))
    return units


def simulate(units, lam, rng):
    """Synthetic three-taper spectra matched to each participant."""
    out = []
    for u in units:
        f, P = u["f"], u["P"]
        Pm = P.mean(axis=(0, 1))
        cf = iaf(Pm, f)
        if not np.isfinite(cf):
            continue
        lnL = fit_background(P, f)                      # epochs x tapers x f
        mean_lnL = lnL.mean(axis=(0, 1))
        # epoch fluctuation of the background: mean over tapers of the level at cf
        i = np.argmin(np.abs(f - cf))
        lev = lnL[:, :, i].mean(1)
        noise_var = np.var(lnL[:, 0, i] - lnL[:, 1, i]) / 2
        sd = np.sqrt(max(np.var(lev) - noise_var / 3, 1e-4))
        Lcf = np.exp(mean_lnL[i])
        a_rel = max(Pm[i] / Lcf - 1, 0.1)                # peak height relative to background
        c = a_rel * Lcf ** (1 - lam)
        g = np.exp(-0.5 * ((f - cf) / 1.5) ** 2)
        n_ep = P.shape[0]
        eta = rng.normal(0, sd, n_ep)
        eps = rng.normal(0, 0.3, n_ep)                  # intrinsic fluctuation, independent
        L = np.exp(mean_lnL[None, :] + eta[:, None])
        S = L + c * np.exp(eps)[:, None] * g[None, :] * L ** lam
        Pk = S[:, None, :] * rng.exponential(1.0, (n_ep, 3, f.size))
        v = dict(u)
        v["P"] = Pk
        v["cov"] = np.full_like(u["cov"], np.nan)
        out.append(v)
    return out


def band_arrays(units, flanks=None, fixed=None):
    """Per epoch and taper: T (band total), B (fitted background, band mean).
    flanks: fit the background to these two flanks (ap_models.flank_level,
    each taper on its own, rescaled to the mean spectrum over epochs and
    tapers); None: the censored power law of fit_background. fixed: band
    (lo, hi) in Hz; None: individual alpha frequency +/- 2 Hz. Participants
    are those with an alpha peak in either case."""
    rows = []
    for u in units:
        f, P = u["f"], u["P"]
        cf = iaf(P.mean(axis=(0, 1)), f)
        if not np.isfinite(cf):
            continue
        lo, hi = (cf - 2, cf + 2) if fixed is None else fixed
        band = (f >= lo) & (f <= hi)
        T = P[..., band].mean(-1)                        # epochs x tapers
        if flanks is None:
            B = np.exp(fit_background(P, f))[..., band].mean(-1)
        else:
            bins = np.where(flank_bins(f, flanks))[0]
            B = np.column_stack([flank_level(P[:, a], f, bins, band, flanks, P.mean(axis=(0, 1)))
                                 for a in range(P.shape[1])])
        rows.append(dict(subject=u["subject"], group=u["group"], T=T, B=B, cov=u["cov"], iaf=cf))
    return rows


COVSETS = {"none": [], "pupil": [0], "pupil + EOG": [0, 1, 2],
           "pupil + EOG + EMG": [0, 1, 2, 3]}


def stack(rows, use_cov):
    """use_cov: list of covariate columns (pupil, VEOG, HEOG, EMG), or empty."""
    T, B, Bz, grp, X = [], [], [], [], []
    for r in rows:
        C = r["cov"][:, use_cov].copy() if use_cov else r["cov"][:, :0]
        if use_cov:
            for k in range(C.shape[1]):
                col = C[:, k]
                m = np.nanmean(col) if np.any(np.isfinite(col)) else 0.0
                col[~np.isfinite(col)] = m
                sd = np.std(col)
                C[:, k] = (col - m) / sd if sd > 0 else 0.0
        for a, b, c in itertools.permutations(range(3)):
            T.append(r["T"][:, a]); B.append(r["B"][:, b]); Bz.append(r["B"][:, c])
            grp.append(np.full(r["T"].shape[0], r["subject"]))
            X.append(C)
    return (np.concatenate(T), np.concatenate(B), np.concatenate(Bz),
            np.concatenate(grp), np.concatenate(X) if use_cov else None)


def instrument_strength(rows, use_cov):
    """Within-participant correlation of the instrument (third taper's ln b,
    residualised on the covariates) with the second taper's ln b."""
    rs = []
    for r in rows:
        zb = np.log(r["B"][:, 2]); xb = np.log(r["B"][:, 1])
        zb, xb = zb - zb.mean(), xb - xb.mean()
        if use_cov:
            C = r["cov"][:, use_cov].copy()
            for k in range(C.shape[1]):
                col = C[:, k]
                m = np.nanmean(col) if np.any(np.isfinite(col)) else 0.0
                col[~np.isfinite(col)] = m
                C[:, k] = col - col.mean()
            zb = zb - C @ np.linalg.lstsq(C, zb, rcond=None)[0]
        if np.std(zb) > 0:
            rs.append(np.corrcoef(zb, xb)[0, 1])
    return float(np.mean(rs))


def reliability(rows):
    """Within-participant correlation of ln B between two tapers, mean over
    participants."""
    return float(np.nanmean([np.corrcoef(np.log(r["B"][:, 0]), np.log(r["B"][:, 1]))[0, 1]
                             for r in rows]))


def tracking(T, B, Bz, grp, nboot, rng):
    """For a band without a rhythm, where T should equal B up to noise: the
    relative residual sum(T - B) / sum(B), and the within-participant slope
    cov(ln T, z) / cov(ln B, z) against the instrument z = ln Bz (1 if the
    band follows the fitted background), with a participant-bootstrap
    interval."""
    _, inv = np.unique(grp, return_inverse=True)
    w = lambda x: x - (np.bincount(inv, x) / np.bincount(inv))[inv]
    z = w(np.log(Bz))
    num = np.bincount(inv, w(np.log(np.maximum(T, 1e-30))) * z)
    den = np.bincount(inv, w(np.log(B)) * z)
    n = num.size
    bs = [num[i].sum() / den[i].sum() for i in (rng.integers(0, n, n) for _ in range(nboot))]
    lo, hi = np.percentile(bs, [2.5, 97.5]) if nboot else (np.nan, np.nan)
    return dict(resid=float(np.sum(T - B) / np.sum(B)), track=float(num.sum() / den.sum()),
                track_lo=float(lo), track_hi=float(hi))


def flank_label(flanks):
    return ", ".join(f"{a:g}-{b:g}" for a, b in flanks)


def merge_rows(name, rows):
    """Write rows to results/name, replacing the rows of the same dataset and
    recording and keeping those of the others."""
    D = pd.DataFrame(rows)
    path = os.path.join(RES, name)
    if os.path.exists(path):
        old = pd.read_csv(path)
        key = lambda d: d.dataset + "/" + d.recording
        D = pd.concat([old[~key(old).isin(key(D))], D])
    D.to_csv(path, index=False)


def variant_row(key, flanks, r, rel):
    """Row of within_recording_variants.csv from a result of run()."""
    row = dict(key, flanks=flank_label(flanks), primary=flanks == FLANKS, n_units=r["n_units"],
               lam=r["lam"], lo=r["lo"], hi=r["hi"], fail=r["fail"], r=rel)
    print(f"  flanks {row['flanks']:12s} lambda {r['lam']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
          f"r {rel:.2f}", flush=True)
    return row


def control_row(key, band, flanks, r, t, rel):
    """Row of within_recording_controls.csv from the results of run() and
    tracking() in a band without a rhythm."""
    row = dict(key, band=f"{band[0]:g}-{band[1]:g}", flanks=flank_label(flanks),
               n_units=r["n_units"], resid_pct=100 * t["resid"], track=t["track"],
               track_lo=t["track_lo"], track_hi=t["track_hi"], lam=r["lam"], n_roots=r["n_roots"],
               lo=r["lo"], hi=r["hi"], fail=r["fail"], r=rel)
    print(f"  control {row['band']} Hz: residual {row['resid_pct']:+.1f}%, tracking "
          f"{t['track']:.2f} [{t['track_lo']:.2f}, {t['track_hi']:.2f}], lambda {r['lam']:+.3f} "
          f"({r['n_roots']} roots, bootstrap without a root {r['fail']:.2f}), r {rel:.2f}",
          flush=True)
    return row


def iv_log(rows):
    """Within-participant IV slope of ln a (taper A) on ln b (B), instrument ln b (C)."""
    num = den = 0.0
    n = 0
    for r in rows:
        for a, b, c in itertools.permutations(range(3)):
            A = r["T"][:, a] - r["B"][:, b]
            ok = A > 0
            if ok.sum() < 10:
                continue
            y, x, z = np.log(A[ok]), np.log(r["B"][ok, b]), np.log(r["B"][ok, c])
            z = z - z.mean()
            num += np.sum(z * (y - y.mean()))
            den += np.sum(z * (x - x.mean()))
            n += ok.sum()
    return num / den if den else np.nan, n


def run(rows, use_cov, nboot, rng):
    T, B, Bz, grp, X = stack(rows, use_cov)
    est = G.estimate_levels(T, B, Bz, grp, X, grid=GRID)
    subs = [r["subject"] for r in rows]
    bs = []
    for _ in range(nboot):
        pick = rng.choice(len(rows), len(rows), replace=True)
        rr = []
        for j, i in enumerate(pick):
            r = dict(rows[i])
            r["subject"] = f"{r['subject']}_{j}"
            rr.append(r)
        T2, B2, Bz2, g2, X2 = stack(rr, use_cov)
        e = G.estimate_levels(T2, B2, Bz2, g2, X2, grid=GRID)
        if e["roots"]:
            bs.append(min(e["roots"], key=lambda x: abs(x - est["lam"])) if np.isfinite(est["lam"])
                      else e["lam"])
        else:
            bs.append(np.nan)
    bs = np.array(bs)
    ok = np.isfinite(bs)
    lo, hi = (np.percentile(bs[ok], [2.5, 97.5]) if ok.sum() > 10 else (np.nan, np.nan))
    return dict(lam=est["lam"], lo=lo, hi=hi, fail=float(1 - ok.mean()), n_units=len(subs),
                n_epochs=int(T.size / 6), gamma=np.round(est["gamma"], 3).tolist(),
                n_roots=len(est["roots"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DS003690_OUT", "ds003690_epochs"))
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--simulate", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--background", choices=("flanks", "power-law"), default="flanks",
                    help="flanks: power law through 2-6 and 26-32 Hz, rescaled to the mean "
                         "spectrum; power-law: one fit over 2-40 Hz without 6-16 Hz")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    units = load(a.data)
    if a.simulate is not None:
        units = simulate(units, a.simulate, rng)
    flanks = FLANKS if a.background == "flanks" else None
    rows = band_arrays(units, flanks)
    # instrument strength: within-participant correlation of ln B across two tapers
    rel = reliability(rows)
    lam_iv, n_iv = iv_log(rows)
    tag = "data" if a.simulate is None else f"simulated lambda = {a.simulate}"
    print(f"{tag}: {len(rows)} participants with an alpha peak; within-participant "
          f"reliability of ln b across tapers r = {rel:.2f}; IV-log (a > 0) {lam_iv:.3f} (n = {n_iv})")
    out = []
    for grp in ("all", "Young", "older"):
        rr = rows if grp == "all" else [r for r in rows if r["group"].lower() == grp.lower()]
        for cname, use_cov in (COVSETS.items() if a.simulate is None else [("none", [])]):
            r = run(rr, use_cov, a.nboot, rng)
            r.update(sample=tag, group=grp, covariates=cname, reliability=rel,
                     instrument_r=instrument_strength(rr, use_cov))
            out.append(r)
            print(f"  {grp:6s} covariates={cname:18s} instrument r {r['instrument_r']:.2f} lambda {r['lam']:+.3f} "
                  f"[{r['lo']:+.2f}, {r['hi']:+.2f}] fail {r['fail']:.2f} "
                  f"n {r['n_units']}/{r['n_epochs']} gamma {r['gamma']}", flush=True)
    suffix = "" if flanks else "_c16"
    name = (f"ds003690_lambda{suffix}.csv" if a.simulate is None
            else f"ds003690_lambda_sim{a.simulate}{suffix}.csv")
    pd.DataFrame(out).to_csv(os.path.join(RES, name), index=False)
    if flanks is None or a.simulate is not None:
        return
    # other flanks, and bands without a rhythm; no covariates, all participants
    key = dict(dataset="ds003690", recording="eyes open")
    var = [variant_row(key, FLANKS, out[0], rel)]
    for fl in VARIANTS:
        rr = band_arrays(units, fl)
        var.append(variant_row(key, fl, run(rr, [], a.nboot, rng), reliability(rr)))
    ctl = []
    for band, fl in CONTROLS:
        rr = band_arrays(units, fl, band)
        ctl.append(control_row(key, band, fl, run(rr, [], a.nboot, rng),
                               tracking(*stack(rr, [])[:4], a.nboot, rng), reliability(rr)))
    merge_rows("within_recording_variants.csv", var)
    merge_rows("within_recording_controls.csv", ctl)


if __name__ == "__main__":
    main()
