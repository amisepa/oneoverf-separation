"""Separation with the recording floor modelled explicitly.

In the Trujillo et al. (2019) organoids the quiet-window spectrum (no network
activity) equals the spectrum of a pharmacologically silenced well
(trujillo_drugs.py), so it is the recording floor, and it makes up most of a
fitted aperiodic background (trujillo_floor.py). Subtracting a band-power
floor from a fitted background leaves a small difference of two noisy
estimates, sometimes <= 0. Here the floor enters the model instead:

    E[P(f)] = s N(f) + L(f) + rhythm,
    L(f) = 10^o / (fk^chi + f^chi) + q   (knee_plateau, knee frequency fk <= 1.5 Hz)
        or 10^o / f^chi + q                (fixed_plateau)

N is the quiet-window spectrum of the same well and recording (3-s windows,
as the spectra fitted), s = 1 (primary) or free in [0.8, 1.25] (sensitivity:
session drift of the floor was within about +-20%). L is fitted by Whittle
deviance to the flanks (0.5-1 and 6-45 Hz; the hump is left out), with s N
as a fixed (or scaled) offset. q is a neural plateau: inside events spiking
adds a flat high-frequency floor on top of the recording floor (without it
the knee fit of event spectra runs to its bounds). In 2.5-4.5 Hz:

    b = mean L (the neural background, plateau included; > 0 by construction)
    a = mean (P - s N - L) (periodic power)

With a free floor scale the fit trades s N against L (both are 1/f-shaped):
a neural background a few percent of the floor is then not identifiable, and
that case is a sensitivity result, not an estimate. The knee is kept below the
band (fk <= 1.5 Hz): once the floor is removed, the neural part is a steep
low-frequency hump, and a free knee settles inside 2.5-4.5 Hz and absorbs the
rhythm (exponent at its bound, a / b ~ 0 in test fits).

for the time-averaged spectrum (A, all 3-s windows) and the event spectrum
(E, windows at network events; trujillo_events.py).

Calibration (split floor). Each recording's quiet windows are split by time
into two halves, N1 and N2. A synthetic spectrum N1 + L_true + R, with L_true
a knee-shaped background scaled to a fraction r_L of the floor in the band
and R a broad rhythm scaled to r_a times L_true in the band, is fitted with
N2 as the floor. Reported: median recovered / true b and a, and the s.d. of
ln b_hat - ln b_true, per (r_L, r_a): how well a neural background that is a
few percent of the floor can be recovered when the floor itself is
estimated.

Then, as in trujillo_maturation.py: per-well slopes over weeks of ln a and
ln b (ols and log-free), two-way bootstrap over wells and days, the 105
pairings of duplicate wells, and the re-tested-claims row
(trujillo_breadth.claim_row) for the rise (weeks 9-17); weeks 20.5-33 (after
the first drug session) are reported as description only.

Writes results/trujillo_floor_model_bandpower.csv, _calibration.csv,
_maturation.csv and _breadth.csv.

Usage: python trujillo_floor_model.py [--nboot 4000] [--reuse] [--workers 6]
"""
import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trujillo_maturation import FIRST, RES, analyse  # noqa: E402
from trujillo_events import classify  # noqa: E402
from trujillo_breadth import claim_row, verdict as row_verdict  # noqa: E402

BAND, CONTROL = (2.5, 4.5), (10.0, 14.0)
BINS = ((0.5, 1.0), (6.0, 45.0))
MODELS = ("knee_plateau", "fixed_plateau")


def L_eval(theta, f, model, plateau=True):
    o, chi = theta[0], theta[1]
    if model == "knee_plateau":
        L = 10.0 ** o / ((10.0 ** theta[2]) ** chi + f ** chi)
    else:
        L = 10.0 ** o / f ** chi
    return L + (10.0 ** theta[-1] if plateau else 0.0)


def fit_floor_model(P, f, N, model, keep, free_scale=False):
    """Whittle fit of s N + L on the kept frequencies. Returns theta, s."""
    Pk, fk, Nk = P[keep], f[keep], N[keep]
    resid = np.clip(Pk - Nk, Pk * 1e-3, None)
    o0 = float(np.log10(np.median(resid * fk ** 2)))
    lo_o, hi_o = o0 - 6.0, o0 + 6.0

    def dev(th):
        s = th[-1] if free_scale else 1.0
        core = th[:-1] if free_scale else th
        mu = s * Nk + L_eval(core, fk, model)
        return float(np.sum(np.log(mu) + Pk / mu))

    q_hi = float(np.log10(np.max(Pk)))
    bounds = [(lo_o, hi_o), (0.05, 6.0)] + ([(-2.0, np.log10(1.5))] if model == "knee_plateau"
                                             else [])
    bounds.append((q_hi - 10.0, q_hi))
    q0 = float(np.log10(np.median(resid[fk >= 30]) if np.any(fk >= 30) else np.min(resid)))
    if free_scale:
        bounds.append((0.8, 1.25))
    best = None
    for chi0 in (1.0, 2.0, 3.0):
        for k0 in ((-1.0, 0.0) if model == "knee_plateau" else (None,)):
            for do, dq in ((0.0, 0.0), (-2.0, -1.0), (0.0, -3.0)):
                th0 = [o0 + do, chi0]
                if model == "knee_plateau":
                    th0.append(k0)
                th0.append(q0 + dq)
                if free_scale:
                    th0.append(1.0)
                th0 = np.clip(th0, [b[0] for b in bounds], [b[1] for b in bounds])
                r = minimize(dev, th0, method="L-BFGS-B", bounds=bounds,
                             options=dict(maxiter=2000, ftol=1e-13))
                if best is None or r.fun < best.fun:
                    best = r
    th = best.x
    return (th[:-1], float(th[-1])) if free_scale else (th, 1.0)


def terms(P, f, N, model, free_scale=False):
    keep = np.zeros(f.size, bool)
    for lo, hi in BINS:
        keep |= (f >= lo) & (f <= hi)
    theta, s = fit_floor_model(P, f, N, model, keep, free_scale)
    out = {}
    for lab, (lo, hi) in (("", BAND), ("ctl_", CONTROL)):
        sel = (f >= lo) & (f <= hi)
        L = L_eval(theta, f[sel], model)
        out[f"{lab}tot"] = float(P[sel].mean())
        out[f"{lab}floor"] = float((s * N[sel]).mean())
        out[f"{lab}b"] = float(L.mean())
        out[f"{lab}b_noplat"] = float(L_eval(theta, f[sel], model, plateau=False).mean())
        out[f"{lab}a"] = float((P[sel] - s * N[sel] - L).mean())
    out.update(scale=s, exponent=float(theta[1]), plateau=float(10.0 ** theta[-1]),
               knee_freq=float(10.0 ** theta[2]) if model == "knee_plateau" else 0.0)
    return out


def one_file(fn):
    date = dt.datetime.strptime(os.path.basename(fn)[:6], "%y%m%d").date()
    week = (date - FIRST).days / 7.0
    d = np.load(fn)
    f = d["f"]
    rows = []
    for w in sorted(int(k[3:]) for k in d.files if k.startswith("pop")):
        S, starts = d[f"S{w}"], d[f"win{w}"]
        ev, quiet, _, _ = classify(d[f"pop{w}"], starts)
        if quiet.sum() < 20:
            continue
        qi = np.where(quiet)[0]
        N = S[qi].mean(0)
        half = starts[qi] < np.median(starts[qi])
        N1, N2 = S[qi[half]].mean(0), S[qi[~half]].mean(0)
        spectra = {"A": S.mean(0)}
        if len(ev) >= 2:
            spectra["E"] = S[ev].mean(0)
        base = dict(date=date, week=week, well=w + 1, n_quiet=int(quiet.sum()), n_events=len(ev))
        for src, P in spectra.items():
            for model in MODELS:
                for free in (False, True):
                    r = terms(P, f, N, model, free)
                    rows.append(dict(base, source=src, model=model,
                                     floor_scale="free" if free else "fixed", **r))
        rows.append(dict(base, source="split", model="", floor_scale="", N1=N1, N2=N2, f=f))
    return rows


def split_only(fn):
    """The two half-recording floors of each well (for the calibration)."""
    date = dt.datetime.strptime(os.path.basename(fn)[:6], "%y%m%d").date()
    d = np.load(fn)
    out = []
    for w in sorted(int(k[3:]) for k in d.files if k.startswith("pop")):
        S, starts = d[f"S{w}"], d[f"win{w}"]
        _, quiet, _, _ = classify(d[f"pop{w}"], starts)
        if quiet.sum() < 20:
            continue
        qi = np.where(quiet)[0]
        half = starts[qi] < np.median(starts[qi])
        out.append(dict(week=(date - FIRST).days / 7.0, well=w + 1, f=d["f"],
                        N1=S[qi[half]].mean(0), N2=S[qi[~half]].mean(0)))
    return out


def calibration(split_rows, r_L=(0.02, 0.03, 0.05, 0.1, 0.2, 0.5), r_a=(0.5, 2.0)):
    """Synthetic N1 + L_true + R fitted with N2 as the floor (knee model)."""
    out = []
    for r in split_rows:
        f, N1, N2 = r["f"], r["N1"], r["N2"]
        sel = (f >= BAND[0]) & (f <= BAND[1])
        shape = 1.0 / (1.0 + f ** 2.0)
        shape[0] = shape[1]
        g = np.exp(-0.5 * ((f - 2.5) / 0.9) ** 2)
        for rl in r_L:
            L_true = shape * (rl * N1[sel].mean() / shape[sel].mean())
            for ra in r_a:
                R = g * (ra * L_true[sel].mean() / g[sel].mean())
                P = N1 + L_true + R
                t = terms(P, f, N2, "knee_plateau", False)
                out.append(dict(week=r["week"], well=r["well"], r_L=rl, r_a=ra,
                                b_true=L_true[sel].mean(), a_true=R[sel].mean(),
                                b_hat=t["b"], a_hat=t["a"]))
    c = pd.DataFrame(out)
    c["lnb_err"] = np.log(c.b_hat) - np.log(c.b_true)
    c["a_ratio"] = c.a_hat / c.a_true
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev-dir", default=os.path.join(RES, "trujillo_events"))
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--recalibrate", action="store_true",
                    help="with --reuse, recompute only the split-floor calibration")
    ap.add_argument("--windows", default="9-17,20.5-33")
    a = ap.parse_args()
    f_bp = os.path.join(RES, "trujillo_floor_model_bandpower.csv")
    f_cal = os.path.join(RES, "trujillo_floor_model_calibration.csv")
    if a.reuse and a.recalibrate:
        bp = pd.read_csv(f_bp)
        files = sorted(glob.glob(os.path.join(a.ev_dir, "[0-9]*.npz")))
        files = [fn for fn in files
                 if 9 <= (dt.datetime.strptime(os.path.basename(fn)[:6], "%y%m%d").date()
                          - FIRST).days / 7.0 <= 17]
        res = Parallel(n_jobs=a.workers)(delayed(split_only)(fn) for fn in files)
        cal = calibration([r for rr in res for r in rr])
        cal.to_csv(f_cal, index=False)
    elif a.reuse:
        bp, cal = pd.read_csv(f_bp), pd.read_csv(f_cal)
    else:
        files = sorted(glob.glob(os.path.join(a.ev_dir, "[0-9]*.npz")))
        res = Parallel(n_jobs=a.workers)(delayed(one_file)(fn) for fn in files)
        rows = [r for rr in res for r in rr]
        bp = pd.DataFrame([r for r in rows if r["source"] != "split"])
        bp.to_csv(f_bp, index=False)
        split = [r for r in rows if r["source"] == "split" and 9 <= r["week"] <= 17]
        cal = calibration(split)
        cal.to_csv(f_cal, index=False)

    print("Split-floor calibration (weeks 9-17, knee + plateau): median b_hat/b_true, s.d. of ln b error,"
          " median a_hat/a_true")
    print(cal.groupby(["r_L", "r_a"]).agg(
        b_ratio=("lnb_err", lambda v: float(np.exp(np.median(v)))),
        lnb_sd=("lnb_err", "std"), a_ratio=("a_ratio", "median")).round(3).to_string())

    print("\nEmpty band, weeks < 6 (median a / (floor + b), and a / b):")
    pre = bp[bp.week < 6]
    print(pre.assign(rel=pre.a / (pre.floor + pre.b), ab=pre.a / pre.b)
          .groupby(["source", "model", "floor_scale"])[["rel", "ab"]].median().round(3).to_string())
    print("\nNeural background share of the total background, b / (floor + b), median:")
    ph = pd.cut(bp.week, [-1, 9, 17, 20.5, 34], labels=["0-9", "9-17", "17-20.5", "20.5-33"])
    print(bp.assign(share=bp.b / (bp.floor + bp.b), ph=ph)
          .pivot_table(index=["source", "model", "floor_scale"], columns="ph", values="share",
                       aggfunc="median", observed=True).round(3).to_string())

    rows, brows = [], []
    for (src, model, fs), t_all in bp.groupby(["source", "model", "floor_scale"]):
        t_all = t_all.assign(p=0.0, ctl_p=0.0)
        pre = t_all[t_all.week < 6]
        for win in a.windows.split(","):
            w0, w1 = (float(v) for v in win.split("-"))
            t = t_all[(t_all.week >= w0) & (t_all.week <= w1)]
            for r in analyse(t, pre, model, win, 5, a.nboot):
                r.update(source=src, model=model, floor_scale=fs)
                rows.append(r)
            if src == "A" and win == "9-17":
                wells = sorted(t.well.unique())
                r = claim_row(t, [[w] for w in wells], a.nboot, seed=11)
                brows.append(dict(source=src, model=model, floor_scale=fs, window=win,
                                  share_a_nonpos=float((t.a <= 0).mean()), **r,
                                  verdict=row_verdict(r), verdict_t=row_verdict(r, "_t")))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "trujillo_floor_model_maturation.csv"), index=False)
    B = pd.DataFrame(brows)
    B.to_csv(os.path.join(RES, "trujillo_floor_model_breadth.csv"), index=False)
    cols = ["source", "model", "floor_scale", "window", "method", "share_a_nonpos", "s_b", "lam0",
            "lam0_2w_lo", "lam0_2w_hi", "lam1", "lam1_2w_lo", "lam1_2w_hi", "lam_star", "roots",
            "verdict_2w"]
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print("\n" + out[[c for c in cols if c in out]].round(3).to_string(index=False))
        print("\nRe-tested-claims row (time-averaged, weeks 9-17):")
        print(B[["model", "floor_scale", "share_a_nonpos", "lam0", "lam0_lo", "lam0_hi", "lam1",
                 "lam1_lo", "lam1_hi", "lam1_t_lo", "lam1_t_hi", "s_b_lo", "s_b_hi", "lam_star",
                 "hdi_lo", "hdi_hi", "p_cross_in_01", "verdict", "verdict_t"]].round(3)
              .to_string(index=False))


if __name__ == "__main__":
    main()
