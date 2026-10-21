"""Inside versus between network events in the Trujillo et al. (2019) organoids.

The nested 2-3 Hz oscillation of these organoids occurs inside network events
(population bursts every ~20-90 s). Time-averaged spectra mix the bursts,
their decaying tails and quiet periods, and the background of such a
spectrum has to be modelled. Here the windows are classified first, from
multi-unit spiking rather than from LFP power:

    events  the paper's rule: population spiking above 80% of its maximum over
            the recording, peaks at least 1 s apart; window -0.5 to +2.5 s
            around the peak
    quiet   3-s windows in which population spiking stays below
            max(median + 10 x MAD, 1% of its maximum), with at least 1 s to any
            sample above it

and per well and recording (spectra from trujillo_extract_events.py: 3-s
Hann windows, mean over channels):

    E  mean spectrum of the event windows      (2 or more events required)
    Q  mean spectrum of the quiet windows      (the background of the
                                                quiescent network, measured)
    A  mean spectrum of all windows             (time-averaged)

Per spectrum and background specification (as in trujillo_maturation.py,
with the low flank at 0.5-1 Hz because 3-s windows smear lower frequencies;
"robust_line" is the paper's robust straight line over 0.5-20 Hz applied to
the channel-mean spectrum), band terms in 2.5-4.5 Hz and in the 10-14 Hz
control band.

Checks and measures:
  - Q as an empty band at every age: a / b of Q should be near 0
  - rhythm per event, a_E, against the modelled event background b_E, and
    against the measured quiet background (b = Q band power)
  - event rate, and the band power of E, Q and A over weeks, to split the
    time-averaged rise into more events and a stronger rhythm per event
  - the event peak (largest residual of E over its knee+plateau background in
    1.5-6 Hz) and the centroid of the positive residual, over weeks, and the
    rhythm in a band tracking the peak (+-1 Hz), to tell a weaker rhythm from
    one sliding out of 2.5-4.5 Hz; and the slopes of Q (2.5-4.5 and 6-20 Hz)
  - maturation lambda-curves as in trujillo_maturation.py (per-well slopes,
    ols and log-free, two-way bootstrap over wells and recording days, the 105
    pairings of duplicate wells), in weeks 0-9, 9-17, 17-33

Writes results/trujillo_events_bandpower.csv, results/trujillo_events_counts.csv,
results/trujillo_events_peakfreq.csv and results/trujillo_events_maturation.csv.

Usage: python trujillo_events.py [--ev-dir DIR] [--windows 0-9,9-17,17-33] [--nboot 2000]
       [--reuse]
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
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trujillo_maturation import FIRST, RES, analyse, ap_eval, band_terms, fit_aperiodic  # noqa: E402

BAND, CONTROL = (2.5, 4.5), (10.0, 14.0)
SPECS = {
    "kp_flanks": dict(model="knee_plateau", bins=((0.5, 1.0), (6.0, 45.0))),
    "plateau_flanks": dict(model="plateau", bins=((0.5, 1.0), (6.0, 45.0))),
    "fixed_0.5-20": dict(model="fixed", bins=((0.5, 1.0), (6.0, 20.0))),
}
MIN_EVENTS = 2


def classify(pop, starts, step_pop=10):
    """Event windows (indices into starts) and quiet windows (boolean)."""
    peaks, _ = find_peaks(pop, height=max(0.8 * pop.max(), 1.0), distance=1000 // step_pop)
    ev_idx = []
    for p in peaks:
        s0 = p * step_pop - 500
        j = int(np.argmin(np.abs(starts - s0)))
        if abs(starts[j] - s0) <= 125 and s0 >= 0:
            ev_idx.append(j)
    med = np.median(pop)
    # floor at 1% of the peak: early recordings have median and MAD ~0, and a
    # single sporadic spike would otherwise rule a window out
    thr = max(med + 10 * 1.4826 * np.median(np.abs(pop - med)), 0.01 * pop.max())
    burst = pop > thr
    pad = 1000 // step_pop
    near = np.convolve(burst.astype(float), np.ones(2 * pad + 1), mode="same") > 0
    win = 3000 // step_pop
    quiet = np.array([not near[s // step_pop:s // step_pop + win].any() for s in starts])
    return np.array(sorted(set(ev_idx)), int), quiet, float(thr), len(peaks)


def robust_line_terms(P, f, band):
    sel = (f >= 0.5) & (f <= 20.0)
    X = sm.add_constant(np.log10(f[sel]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beta = sm.RLM(np.log10(P[sel]), X, M=sm.robust.norms.TukeyBiweight(4.685)).fit().params
    out = []
    for lo, hi in (band, CONTROL):
        s = (f >= lo) & (f <= hi)
        tot = float(P[s].mean())
        b = float(np.mean(10.0 ** (beta[0] + beta[1] * np.log10(f[s]))))
        out.append((tot, b, 0.0, tot - b))
    return out


def spec_terms(P, f):
    rows = {}
    for name, sp in SPECS.items():
        keep = np.zeros(f.size, bool)
        for lo, hi in sp["bins"]:
            keep |= (f >= lo) & (f <= hi)
        fit = fit_aperiodic(P[keep], f[keep], None, sp["model"])
        if fit is None:
            continue
        rows[name] = [band_terms(P, f, fit, bd, sp["model"]) for bd in (BAND, CONTROL)]
        rows[name].append(dict(exponent=fit["exponent"], knee_freq=fit["knee_freq"]))
    rows["robust_line"] = robust_line_terms(P, f, BAND) + [dict(exponent=np.nan, knee_freq=np.nan)]
    return rows


def tables(ev_dir):
    bp, counts = [], []
    for fn in sorted(glob.glob(os.path.join(ev_dir, "[0-9]*.npz"))):
        tag = os.path.basename(fn)[:-4]
        date = dt.datetime.strptime(tag[:6], "%y%m%d").date()
        week = (date - FIRST).days / 7.0
        d = np.load(fn)
        f = d["f"]
        for w in sorted(int(k[3:]) for k in d.files if k.startswith("pop")):
            pop, S, starts = d[f"pop{w}"], d[f"S{w}"], d[f"win{w}"]
            ev, quiet, thr, n_peaks = classify(pop, starts)
            n_min = d[f"n{w}"] / 60000.0
            counts.append(dict(date=date, week=week, well=w + 1, n_events=len(ev),
                               event_rate=len(ev) / n_min, quiet_share=float(quiet.mean()),
                               burst_share=float((pop > thr).mean()), n_spikes=int(d[f"nspk{w}"]),
                               peak_pop=float(pop.max())))
            spectra = {"A": S.mean(0)}
            if quiet.sum() >= 10:
                spectra["Q"] = S[quiet].mean(0)
            if len(ev) >= MIN_EVENTS:
                spectra["E"] = S[ev].mean(0)
            for src, P in spectra.items():
                for spec, (bt, ct, extra) in spec_terms(P, f).items():
                    bp.append(dict(date=date, week=week, well=w + 1, source=src, spec=spec,
                                   tot=bt[0], b=bt[1], p=bt[2], a=bt[3],
                                   ctl_tot=ct[0], ctl_b=ct[1], ctl_p=ct[2], ctl_a=ct[3], **extra))
    return pd.DataFrame(bp), pd.DataFrame(counts)


def peak_table(ev_dir):
    rows = []
    sp = SPECS["kp_flanks"]
    for fn in sorted(glob.glob(os.path.join(ev_dir, "[0-9]*.npz"))):
        date = dt.datetime.strptime(os.path.basename(fn)[:6], "%y%m%d").date()
        d = np.load(fn)
        f = d["f"]
        keep = np.zeros(f.size, bool)
        for lo, hi in sp["bins"]:
            keep |= (f >= lo) & (f <= hi)
        for w in sorted(int(k[3:]) for k in d.files if k.startswith("pop")):
            ev, quiet, _, _ = classify(d[f"pop{w}"], d[f"win{w}"])
            if len(ev) < MIN_EVENTS or quiet.sum() < 10:
                continue
            S = d[f"S{w}"]
            E, Q = S[ev].mean(0), S[quiet].mean(0)
            fit = fit_aperiodic(E[keep], f[keep], None, sp["model"])
            s = (f >= 1.5) & (f <= 6.0)
            r = E[s] - ap_eval(fit["theta"], f[s], sp["model"])
            rp = np.clip(r, 0, None)
            pk = float(f[s][np.argmax(r)]) if r.max() > 0 else np.nan
            row = dict(date=date, week=(date - FIRST).days / 7.0, well=w + 1, peak=pk,
                       centroid=float((f[s] * rp).sum() / rp.sum()) if rp.sum() > 0 else np.nan,
                       q_band=float(Q[(f >= BAND[0]) & (f <= BAND[1])].mean()),
                       q_6_20=float(Q[(f >= 6) & (f <= 20)].mean()),
                       e_6_20=float(E[(f >= 6) & (f <= 20)].mean()))
            if np.isfinite(pk):
                tb = (f >= pk - 1) & (f <= pk + 1)
                bg = ap_eval(fit["theta"], f[tb], sp["model"], neural=True)
                bg_all = ap_eval(fit["theta"], f[tb], sp["model"])
                row.update(a_track=float((E[tb] - bg_all).mean()), b_track=float(bg.mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev-dir", default=os.path.join(RES, "trujillo_events"))
    ap.add_argument("--windows", default="0-9,9-17,17-33")
    ap.add_argument("--min-rec", type=int, default=5)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    f_bp = os.path.join(RES, "trujillo_events_bandpower.csv")
    f_ct = os.path.join(RES, "trujillo_events_counts.csv")
    if a.reuse:
        bp, ct = pd.read_csv(f_bp), pd.read_csv(f_ct)
    else:
        bp, ct = tables(a.ev_dir)
        bp.to_csv(f_bp, index=False)
        ct.to_csv(f_ct, index=False)

    # empty band on Q, every age
    q = bp[bp.source == "Q"].assign(ab=lambda x: x.a / x.b)
    q["phase"] = pd.cut(q.week, [-1, 9, 17, 34], labels=["0-9", "9-17", "17-33"])
    print("Quiet windows, a/b in 2.5-4.5 Hz (median; should be 0):")
    print(q.pivot_table(index="spec", columns="phase", values="ab", aggfunc="median",
                        observed=True).round(3).to_string())

    rows = []
    for spec in bp.spec.unique():
        E = bp[(bp.source == "E") & (bp.spec == spec)]
        Q = bp[(bp.source == "Q") & (bp.spec == spec)]
        A = bp[(bp.source == "A") & (bp.spec == spec)]
        Eq = E.merge(Q[["date", "well", "tot"]].rename(columns={"tot": "q_tot"}), on=["date", "well"])
        variants = {"event_vs_event_bg": E,
                    "event_vs_quiet": Eq.assign(b=Eq.q_tot, p=0.0, a=Eq.a),
                    "timeavg": A}
        for var, t_all in variants.items():
            for win in a.windows.split(","):
                w0, w1 = (float(v) for v in win.split("-"))
                t = t_all[(t_all.week >= w0) & (t_all.week <= w1)]
                if t.well.nunique() < 3:
                    continue
                for r in analyse(t, Q, spec, win, a.min_rec, a.nboot):
                    r["variant"] = var
                    rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "trujillo_events_maturation.csv"), index=False)
    cols = ["variant", "spec", "window", "method", "n_wells", "share_a_nonpos", "s_b", "lam0",
            "lam0_2w_lo", "lam0_2w_hi", "lam1", "lam1_2w_lo", "lam1_2w_hi", "lam_star",
            "verdict_2w"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[cols].round(3).to_string(index=False))

    pk = peak_table(a.ev_dir)
    pk.to_csv(os.path.join(RES, "trujillo_events_peakfreq.csv"), index=False)
    print()
    print("Per-well slopes over weeks (mean; Hz/week for frequencies, ln/week otherwise):")
    for win in a.windows.split(","):
        w0, w1 = (float(v) for v in win.split("-"))
        g = pk[(pk.week >= w0) & (pk.week <= w1)]
        for col, log in (("peak", False), ("centroid", False), ("q_band", True), ("q_6_20", True),
                         ("e_6_20", True), ("a_track", True)):
            sl = []
            for _, x in g.dropna(subset=[col]).groupby("well"):
                if len(x) < a.min_rec or (log and (x[col] <= 0).any()):
                    continue
                sl.append(np.polyfit(x.week, np.log(x[col]) if log else x[col], 1)[0])
            if sl:
                print(f"  weeks {win} {col:9s} {np.mean(sl):+.3f} (wells > 0: {np.sum(np.array(sl) > 0)}/{len(sl)})")

    # decomposition of the time-averaged rise: events vs rhythm per event
    print("\nPer-well slopes over weeks (mean; ln per week, quiet_share in share per week):")
    for win in a.windows.split(","):
        w0, w1 = (float(v) for v in win.split("-"))
        c = ct[(ct.week >= w0) & (ct.week <= w1)]
        for col in ("event_rate", "quiet_share", "n_spikes"):
            y = (lambda v: v) if col == "quiet_share" else (lambda v: np.log(v.clip(lower=1e-3)))
            sl = [np.polyfit(g.week, y(g[col]), 1)[0]
                  for _, g in c.groupby("well") if len(g) >= a.min_rec]
            print(f"  weeks {win} {col:12s} {np.mean(sl):+.3f} (wells > 0: {np.sum(np.array(sl) > 0)}/{len(sl)})")


if __name__ == "__main__":
    main()
