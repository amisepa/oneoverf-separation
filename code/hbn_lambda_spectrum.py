"""Crossover lambda* of the HBN age effect as a function of frequency.

For each frequency band, eyes closed and eyes open, and each participant
passing quality control (results/hbn_qc_flags.csv): total power T and
background b in the band, from the posterior-ROI mean spectrum and the
background of the primary analysis (Whittle fit over 2-55 Hz with 6-16 Hz
censored; power law, and knee+plateau as the alternative, as in hbn_kp.py),
and periodic power a = T - b. Then, over the participants with a > 0,
s_a = OLS age slope of ln a, s_b = age slope of ln b, the age slope
s(lambda) = s_a - lambda s_b at lambda = 0 and 1, and lambda* = s_a / s_b,
with pairs-bootstrap intervals, the Bayesian-bootstrap HDI of lambda* and
the share of its draws inside [0, 1] (lambda_curve.effect_curve).

Bands (closed intervals on the 0.25 Hz grid of the PSDs):
  absolute   1 Hz wide, 2-3 Hz to 39-40 Hz (centre 2.5 to 39.5)
  iaf_rel    1 Hz wide, centred on IAF + k for k = -4..4, IAF the
             eyes-closed individual alpha frequency (hbn_kp.find_iaf)
  iaf_band   IAF +/- 2 Hz, the primary band; reproduces
             results/hbn_age_alpha_crossover.csv (sample "qc")

Where few participants have a > 0, a is centred on 0 and lambda* is noise
over noise; reliable = share_pos >= 0.8, reliable_both = reliable under
both backgrounds. Outside the 6-16 Hz censor window the band's bins were
part of the background fit, so a there is a fit residual (in_fit = share of
the band's bins used in the fit, median over participants). The iaf_rel
bands move with IAF across the 1/f slope, so their s_b differs from that of
any fixed band. lambda* has no interval when the bootstrap
interval of s_b includes 0 (lam_star_bounded False). crossover classifies
the HDI as in breadth_summary.py: inside [0, 1], outside, overlaps, or
unbounded.

Slopes are in natural-log units per year; share_pos_age is the
linear-probability age slope of 1[a > 0] (per year).

Writes results/hbn_lambda_spectrum.csv and the per-participant band powers
results/hbn_lambda_spectrum_fits.csv (kept out of the repository).

Usage: python hbn_lambda_spectrum.py [--psd-dir DIR] [--workers 12]
                                     [--nboot 2000] [--from-fits]
"""
import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ap_models import ap_eval, fit_aperiodic
from hbn_kp import CENSOR, FIT_RANGE, ROI, find_iaf, mask_for
from lambda_curve import effect_curve

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
MODELS = ("fixed", "knee_plateau")
ABS_LO = np.arange(2.0, 40.0, 1.0)
IAF_OFFSETS = np.arange(-4, 5)
RELIABLE = 0.8
warnings.filterwarnings("ignore")


def bands(iaf):
    """(axis, centre, lo, hi) of every band for one participant."""
    out = [("absolute", lo + 0.5, lo, lo + 1.0) for lo in ABS_LO]
    out += [("iaf_rel", float(k), iaf + k - 0.5, iaf + k + 0.5) for k in IAF_OFFSETS]
    out.append(("iaf_band", 0.0, iaf - 2.0, iaf + 2.0))
    return out


def subject_rows(path):
    """Band totals and backgrounds of one participant; (rows, error or None)."""
    rows = []
    try:
        d = np.load(path, allow_pickle=True)
        f0 = d["freqs"].astype(float)
        sel = (f0 >= FIT_RANGE[0]) & (f0 <= FIT_RANGE[1])
        f = f0[sel]
        ch = [str(c) for c in d["ch_names"]]
        ix = [ch.index(c) for c in ROI if c in ch]
        if len(ix) < 3:
            return rows, None
        sub, age = str(d["subject"]), float(d["age"])
        iaf = find_iaf(d["ec"][ix][:, sel].astype(float).mean(0), f)
        if not np.isfinite(iaf):
            return rows, None
        keep = mask_for(f, "censor", [CENSOR])
        B = bands(iaf)
        for cond in ("ec", "eo"):
            P = d[cond][ix][:, sel].astype(float).mean(0)
            P = np.clip(np.nan_to_num(P, nan=1e-12), 1e-12, None)
            for model in MODELS:
                fit = fit_aperiodic(P, f, keep, model)
                if fit is None:
                    continue
                L = ap_eval(fit["theta"], f, model)
                for axis, centre, lo, hi in B:
                    m = (f >= lo) & (f <= hi)
                    rows.append(dict(subject=sub, age=age, iaf=iaf, cond=cond, model=model,
                                     axis=axis, centre=centre, tot=float(P[m].mean()),
                                     b=float(L[m].mean()), in_fit=float(keep[m].mean())))
    except Exception as e:
        return [], f"{os.path.basename(path)}: {type(e).__name__}: {e}"
    return rows, None


def lp_slope(y, x):
    """OLS slope of y on x."""
    return float(np.polyfit(x, y, 1)[0])


def summarise(g, nboot):
    """One row of the output table from the participants of one band."""
    a = (g.tot - g.b).to_numpy()
    lb = np.log(g.b.to_numpy())
    x = g.age.to_numpy()
    pos = a > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        la = np.where(pos, np.log(np.where(pos, a, 1.0)), np.nan)
    r = effect_curve(la, lb, x=x, grid=(0.0, 1.0), nboot=nboot,
                     rng=np.random.default_rng(0))
    (s0, s0_lo, s0_hi), (s1, s1_lo, s1_hi) = r["curve"]
    hdi_lo, hdi_hi = r["hdi"]
    if not r["lam_star_bounded"]:
        verdict = "unbounded"
    elif hdi_lo >= 0 and hdi_hi <= 1:
        verdict = "inside"
    elif hdi_hi < 0 or hdi_lo > 1:
        verdict = "outside"
    else:
        verdict = "overlaps"
    share = float(pos.mean())
    return dict(n_all=len(g), n=r["n"], share_pos=share, reliable=share >= RELIABLE,
                share_pos_age=lp_slope(pos.astype(float), x),
                median_a_over_b=float(np.median(a / g.b.to_numpy())),
                in_fit=float(np.median(g.in_fit)), s_b_all=lp_slope(lb, x),
                s_a=r["s_a"], s_b=r["s_b"], s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
                s0=s0, s0_lo=s0_lo, s0_hi=s0_hi, s1=s1, s1_lo=s1_lo, s1_hi=s1_hi,
                lam_star=r["lam_star"], lam_star_bounded=r["lam_star_bounded"],
                ci_lo=r["ci"][0], ci_hi=r["ci"][1], hdi_lo=hdi_lo, hdi_hi=hdi_hi,
                p_cross_in_01=r["p_cross_in_01"], crossover=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--from-fits", action="store_true",
                    help="read the per-participant band powers instead of refitting")
    a = ap.parse_args()
    fits_path = os.path.join(RES, "hbn_lambda_spectrum_fits.csv")
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    ok_ids = set(Q.index[Q.qc_ok])

    if a.from_fits:
        T = pd.read_csv(fits_path)
    else:
        files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
        jobs = [p for p in files
                if os.path.basename(p).split("_", 1)[1].replace(".npz", "") in ok_ids]
        print(f"{len(jobs)} PSD files of participants passing QC", flush=True)
        rows = []
        with Pool(a.workers) as pool:
            for k, (r, err) in enumerate(pool.imap(subject_rows, jobs, chunksize=8)):
                if err:
                    print(f"  {err}", flush=True)
                rows += r
                if (k + 1) % 500 == 0:
                    print(f"  {k + 1}/{len(jobs)}", flush=True)
        T = pd.DataFrame(rows)
        T.to_csv(fits_path, index=False)
    T = T[T.subject.isin(ok_ids)]
    T = T[np.isfinite(T.tot) & np.isfinite(T.b) & (T.b > 0)]
    print(f"{T.subject.nunique()} participants", flush=True)

    out = []
    for (cond, model, axis, centre), g in T.groupby(["cond", "model", "axis", "centre"]):
        half = 2.0 if axis == "iaf_band" else 0.5
        row = dict(cond=cond, model=model, axis=axis, centre=centre,
                   f_lo=centre - half, f_hi=centre + half)
        row.update(summarise(g, a.nboot))
        out.append(row)
    O = pd.DataFrame(out).sort_values(["axis", "model", "cond", "centre"])
    O["reliable_both"] = O.groupby(["axis", "cond", "centre"]).reliable.transform("all")
    O.to_csv(os.path.join(RES, "hbn_lambda_spectrum.csv"), index=False)

    pct = lambda s: 100 * (np.exp(s) - 1)
    for (axis, model, cond), g in O.groupby(["axis", "model", "cond"], sort=False):
        print(f"\n=== {axis}, {model}, {cond.upper()} ===")
        print(f"{'band':>9s} {'a>0':>5s} {'a/b':>6s} {'s(0)%/y':>8s} {'s(1)%/y':>8s} "
              f"{'s_b%/y':>7s} {'lam*':>7s} {'HDI':>17s} {'P01':>5s} {'n':>5s}  crossover")
        for _, r in g.iterrows():
            lab = (f"{r.f_lo:g}-{r.f_hi:g}" if axis == "absolute"
                   else f"IAF{r.centre:+.0f}" if axis == "iaf_rel" else "IAF+/-2")
            hdi = (f"[{r.hdi_lo:6.2f},{r.hdi_hi:6.2f}]" if r.lam_star_bounded
                   else "      --       ")
            flag = "" if r.reliable else "  (a>0 < 80%)"
            print(f"{lab:>9s} {r.share_pos:5.2f} {r.median_a_over_b:6.2f} {pct(r.s0):+8.2f} "
                  f"{pct(r.s1):+8.2f} {pct(r.s_b):+7.2f} {r.lam_star:7.2f} {hdi:>17s} "
                  f"{r.p_cross_in_01:5.2f} {r.n:5d}  {r.crossover}{flag}")
    print(f"\nwrote {os.path.join(RES, 'hbn_lambda_spectrum.csv')}")


if __name__ == "__main__":
    main()
