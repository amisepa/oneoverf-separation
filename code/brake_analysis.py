"""Propofol loss of consciousness (Brake et al. 2024 Nat Commun): do the
band-power changes before and after LOC depend on lambda?

Data: Cz spectrograms of 14 patients, aligned to loss of consciousness
(object drop), -300 to +60 s, 0.5-Hz resolution (figshare 24777990, CC BY
4.0). Brake et al. detrended each spectrum by dividing out a fitted
aperiodic model (their "eq6": a synaptic filter with a noise floor, plus up
to three Gaussian peaks for delta, alpha and beta, fitted in log10 power by
least absolute deviation; code at github.com/niklasbrake/EEG_modelling),
i.e. lambda = 1, and reported that detrended delta rises only at LOC while
alpha and beta rise earlier.

Here each spectrogram is averaged into 5-s bins, the same model is fitted to
each bin over 0.5-100 Hz (55-65 Hz excluded), and band power is split into
the fitted aperiodic part b and the rest a = total - b for delta (1-4 Hz),
alpha (8-15 Hz) and beta (15-30 Hz). Changes from baseline to a pre-LOC
window (-60 to -10 s) and a post-LOC window (+10 to +60 s) are reported as a
function of lambda, with the crossover lambda*, and, for comparison, Brake's
own detrended measure, the mean of 10 log10(P / L) over the band.

Two baselines: "pre-infusion", the minute before each patient's infusion
onset (to 5 s before it; from the authors' timing table in the manuscript
source data, figshare file 43599408, exported from MATLAB as
data_time_information.csv; the spectrograms start 300 s before LOC, so one
patient whose infusion began 285 s before LOC has 15 s), and "first 60 s"
of the spectrogram, the approximation used when the timing was not
available.

The fit minimises the mean of sqrt(r^2 + EPS^2) over the residuals r (least
absolute deviation, smoothed at EPS = 0.01 log10 units) with L-BFGS-B from
several starts: the authors' start, the previous bin's solution and random
points in the bounds. The objective has several local minima, and the
authors' call (the scalar mean absolute deviation handed to least_squares
as a single residual) stops before it reaches one. Each bin keeps the
start with the smallest absolute deviation (fit_lad), whether that run
converged (fit_ok) and how many starts came within 1% of it (fit_agree).

Solutions within 2% of the best absolute deviation fit a bin equally well
but can put very different power in the background. Each bin keeps the
background of every such solution (their range is {band}_b_lo, {band}_b_hi),
and results/brake_fit_sensitivity.csv carries that multiplicity into the
effects in two ways: the envelope, with every bin of the window at its
lowest or highest near-optimal background and every baseline bin at its
lowest or highest (four assignments, all patients pushed the same way), and
intervals from draws in which each bin's background is taken at random from
its near-optimal set while the patients are reweighted (Bayesian bootstrap),
with the standard deviation of the lambda = 1 effect from the fit alone
(lam1_sd_fit) and from the patients alone (lam1_sd_patients).

Usage: python brake_analysis.py [--data ZIP] [--timing CSV] [--bin 5] [--jobs 1]
Writes results/brake_bins.csv (per patient and bin), results/brake_lambda.csv
and results/brake_fit_sensitivity.csv.
"""
import argparse
import io
import multiprocessing
import os
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lambda_curve import effect_curve, hdi

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
BANDS = {"delta": (1.0, 4.0), "alpha": (8.0, 15.0), "beta": (15.0, 30.0)}
WINDOWS = {"baseline": (-300.0, -240.0), "pre-LOC": (-60.0, -10.0), "post-LOC": (10.0, 60.0)}

# ---- Brake et al. eq6 model, as in their detrending.py -------------------
LB_AP, UB_AP = [7e-3, 3.9e-3, -21, 1], [75e-3, 4.1e-3, -7, 5]
SP_AP = [17e-3, 4e-3, -10.5, 4.2]
LB_P = [0, 0, 0.2, 6, 0, 0.6, 15, 0, 1]
UB_P = [4, 4, 3, 15, 4, 4, 40, 3, 10]
SP_P = [0.5, 2, 1.5, 8, 0.3, 1, 22, 0.1, 4]
EPS, N_RANDOM, NEAR = 0.01, 30, 0.02


def eq6(f, tau1, tau2, offset, mag):
    w2 = (2 * np.pi * f) ** 2
    tau2 = 4e-3
    x2 = (tau1 - tau2) ** 2 / ((1 + tau1 ** 2 * w2) * (1 + tau2 ** 2 * w2))
    return mag + np.log10(np.exp(offset) + x2)


def peaks(f, p):
    y = np.zeros_like(f)
    for i in range(0, len(p), 3):
        c, h, w = p[i:i + 3]
        y = y + h * np.exp(-(f - c) ** 2 / (2 * w ** 2))
    return y


def model_jac(f, x):
    """Derivatives of eq6 + peaks with respect to the parameters (tau2 is fixed)."""
    tau1, off = x[0], x[2]
    w2 = (2 * np.pi * f) ** 2
    x2 = (tau1 - 4e-3) ** 2 / ((1 + tau1 ** 2 * w2) * (1 + 4e-3 ** 2 * w2))
    den = (np.exp(off) + x2) * np.log(10)
    J = np.zeros((f.size, len(x)))
    J[:, 0] = x2 * (2 / (tau1 - 4e-3) - 2 * tau1 * w2 / (1 + tau1 ** 2 * w2)) / den
    J[:, 2] = np.exp(off) / den
    J[:, 3] = 1.0
    for i in range(4, len(x), 3):
        c, h, w = x[i:i + 3]
        g = np.exp(-(f - c) ** 2 / (2 * w ** 2))
        J[:, i], J[:, i + 1] = h * g * (f - c) / w ** 2, g
        J[:, i + 2] = h * g * (f - c) ** 2 / w ** 3
    return J


def fit_bin(f, logP, start=None, rng=None):
    """LAD fit of eq6 + 3 peaks in log10 power from several starts.

    Returns the parameter vector, its mean absolute deviation, whether the
    best run converged, the number of starts within 1% of the best, and the
    parameter vectors of all starts within NEAR of the best."""
    rng = rng or np.random.default_rng(0)
    lb, ub = np.array(LB_AP + LB_P, float), np.array(UB_AP + UB_P, float)
    span = ub - lb
    resid = lambda x: logP - eq6(f, *x[:4]) - peaks(f, x[4:])

    def obj(u):                                           # parameters scaled to [0, 1]
        x = lb + u * span
        r = resid(x)
        s = np.sqrt(r * r + EPS * EPS)
        return s.mean(), -(model_jac(f, x).T @ (r / s)) / r.size * span

    starts = [SP_AP + SP_P] + ([start] if start is not None else [])
    starts = [(np.clip(np.array(x0, float), lb, ub) - lb) / span for x0 in starts]
    starts += [rng.uniform(size=lb.size) for _ in range(N_RANDOM)]
    runs = []
    for u0 in starts:
        r = optimize.minimize(obj, u0, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * lb.size,
                              options=dict(maxiter=3000, maxfun=6000, ftol=1e-12, gtol=1e-7))
        x = lb + r.x * span
        runs.append((float(np.mean(np.abs(resid(x)))), x, bool(r.success)))
    lad, x, ok = min(runs, key=lambda q: q[0])
    near = np.array([q[1] for q in runs if q[0] <= (1 + NEAR) * lad])
    return x, lad, ok, sum(q[0] <= 1.01 * lad for q in runs), near


def fit_patient(args):
    """Bin one patient's spectrogram and fit every bin. Returns the rows of
    brake_bins.csv and, per row, the background band power of each
    near-optimal solution (dict band -> array, the best fit first)."""
    pt, data, binw = args
    z = zipfile.ZipFile(data)
    f = pd.read_csv(io.BytesIO(z.read("csv/frequency.csv"))).iloc[:, 0].to_numpy(float)
    t = pd.read_csv(io.BytesIO(z.read("csv/time.csv"))).iloc[:, 0].to_numpy(float)
    fsel = (f >= 0.5) & (f <= 100) & ~((f > 55) & (f < 65))
    edges = np.arange(-300, 60.0001, binw)
    S = pd.read_csv(io.BytesIO(z.read(f"csv/pt_{pt:02d}.csv")), header=None).to_numpy(float)
    start, rng, rows, sets = None, np.random.default_rng(pt), [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi)
        if m.sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # bins with no valid frame
            P = np.nanmean(S[:, m], axis=1)
        good = np.isfinite(P) & (P > 0)
        if good[fsel].mean() < 0.9:
            continue
        ff, lP = f[fsel & good], np.log10(P[fsel & good])
        x, lad, ok, agree, near = fit_bin(ff, lP, start, rng)
        start = x
        L = 10 ** eq6(f, *x[:4])                          # aperiodic, linear
        Ln = np.array([10 ** eq6(f, *q[:4]) for q in near])
        row = dict(patient=pt, t=(lo + hi) / 2, n_frames=int(m.sum()),
                   fit_lad=lad, fit_ok=ok, fit_agree=agree)
        cand = {}
        for name, (b0, b1) in BANDS.items():
            bm = (f >= b0) & (f < b1) & good
            tot, b = float(np.mean(P[bm])), float(np.mean(L[bm]))
            # distinct near-optimal backgrounds (to 1%), after the best fit
            cand[name] = np.r_[b, np.exp(np.unique(np.round(np.log(Ln[:, bm].mean(1)), 2)))]
            row.update({f"{name}_tot": tot, f"{name}_b": b, f"{name}_a": tot - b,
                        f"{name}_brake_db": float(np.mean(10 * np.log10(P[bm] / L[bm]))),
                        f"{name}_b_lo": float(cand[name][1:].min()),
                        f"{name}_b_hi": float(cand[name][1:].max())})
        rows.append(row)
        sets.append(cand)
    print(f"patient {pt}: {len(rows)} bins", flush=True)
    return rows, sets


def fit_sensitivity(tot, sets, patient, m0, m1, rng, ndraw=2000):
    """Effects from baseline bins (m0) to window bins (m1) of one band when the
    background of each bin is any of its near-optimal solutions.

    tot: total band power per bin; sets: per bin, the candidate backgrounds
    (the best fit, then the distinct near-optimal ones). Returns a dict: the
    envelope over the four extreme assignments, and intervals over draws that
    take each bin's background at random from its set and reweight the
    patients."""
    pts = np.unique(patient[m0 | m1])
    n = np.array([len(c) - 1 for c in sets])
    C = np.array([np.r_[c, np.full(n.max() + 1 - len(c), np.nan)] for c in sets])
    lo, hi = np.nanmin(C[:, 1:], axis=1), np.nanmax(C[:, 1:], axis=1)

    def change(b):
        """d ln a and d ln b per patient; patients with a <= 0 are left out."""
        D = []
        for p in pts:
            i0, i1 = m0 & (patient == p), m1 & (patient == p)
            T0, B0, T1, B1 = tot[i0].mean(), b[i0].mean(), tot[i1].mean(), b[i1].mean()
            if T0 > B0 and T1 > B1:
                D.append((np.log((T1 - B1) / (T0 - B0)), np.log(B1 / B0)))
        return np.array(D)

    env = np.array([change(np.where(m1, bw, bb)).mean(0)
                    for bw in (lo, hi) for bb in (lo, hi)])
    env = np.column_stack([env[:, 0], env[:, 0] - env[:, 1], env[:, 0] / env[:, 1]])
    best = change(C[:, 0])
    fit, pat, both = [], [], []
    for _ in range(ndraw):
        D = change(C[np.arange(n.size), 1 + (rng.uniform(size=n.size) * n).astype(int)])
        fit.append(D.mean(0))
        both.append(rng.dirichlet(np.ones(len(D))) @ D)
        pat.append(rng.dirichlet(np.ones(len(best))) @ best)
    fit, pat, both = np.array(fit), np.array(pat), np.array(both)
    ls = both[:, 0] / both[:, 1]
    pc = lambda v: [float(q) for q in np.percentile(v, [2.5, 97.5])]
    out = dict(near_multiple=float(np.mean(n[m0 | m1] > 1)),
               near_twofold=float(np.mean((hi / lo)[m0 | m1] > 2)))
    for k, name in enumerate(("lam0", "lam1", "lam_star")):
        out[f"{name}_env_lo"], out[f"{name}_env_hi"] = float(env[:, k].min()), float(env[:, k].max())
    (out["lam0_lo"], out["lam0_hi"]), (out["lam1_lo"], out["lam1_hi"]) = \
        pc(both[:, 0]), pc(both[:, 0] - both[:, 1])
    out["s_b_lo"], out["s_b_hi"] = pc(both[:, 1])
    out["hdi_lo"], out["hdi_hi"] = hdi(ls)
    out["p_cross_in_01"] = float(np.mean((ls >= 0) & (ls <= 1)))
    out["lam1_sd_fit"] = float(np.std(fit[:, 0] - fit[:, 1]))
    out["lam1_sd_patients"] = float(np.std(pat[:, 0] - pat[:, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.environ.get("EEG_DATA", "eeg_data"),
                                                   "brake2024", "spectrogram_Cz_all_subjects.zip"))
    ap.add_argument("--timing", default=os.path.join(
        os.environ.get("EEG_DATA", "eeg_data"), "brake2024", "source", "_data", "EEG_data",
        "data_time_information.csv"))
    ap.add_argument("--bin", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=1, help="patients fitted in parallel")
    a = ap.parse_args()
    jobs = [(pt, a.data, a.bin) for pt in range(1, 15)]
    if a.jobs > 1:
        with multiprocessing.Pool(a.jobs) as pool:
            R = pool.map(fit_patient, jobs, chunksize=1)
    else:
        R = [fit_patient(j) for j in jobs]
    B = pd.DataFrame([row for rows, _ in R for row in rows])
    sets = [c for _, cs in R for c in cs]
    B.to_csv(os.path.join(RES, "brake_bins.csv"), index=False)

    out = []
    W = {w: B[(B.t >= lo) & (B.t < hi)].groupby("patient").mean(numeric_only=True)
         for w, (lo, hi) in WINDOWS.items()}
    bases = {"first 60 s": W["baseline"]}
    lo, hi = WINDOWS["baseline"]
    rows0 = {"first 60 s": ((B.t >= lo) & (B.t < hi)).to_numpy()}     # baseline bins
    if os.path.exists(a.timing):
        T = pd.read_csv(a.timing)
        T.index = np.arange(1, len(T) + 1)               # row k = patient k
        onset = (T.infusion_onset - T.object_drop).reindex(B.patient).to_numpy()
        pre = B[(B.t >= onset - 60) & (B.t < onset - 5)]
        bases["pre-infusion"] = pre.groupby("patient").mean(numeric_only=True)
        rows0["pre-infusion"] = ((B.t >= onset - 60) & (B.t < onset - 5)).to_numpy()
        dur = pre.groupby("patient").size() * a.bin
        print("pre-infusion baseline, s per patient:", dur.to_dict())
    for base, W0 in bases.items():
        for band in BANDS:
            for win in ("pre-LOC", "post-LOC"):
                d = W[win].join(W0, rsuffix="_0", how="inner")
                with np.errstate(invalid="ignore", divide="ignore"):
                    dA = np.log(d[f"{band}_a"].where(d[f"{band}_a"] > 0)) - \
                        np.log(d[f"{band}_a_0"].where(d[f"{band}_a_0"] > 0))
                dB = np.log(d[f"{band}_b"]) - np.log(d[f"{band}_b_0"])
                dT = np.log(d[f"{band}_tot"]) - np.log(d[f"{band}_tot_0"])
                dBrake = d[f"{band}_brake_db"] - d[f"{band}_brake_db_0"]
                r = effect_curve(dA.to_numpy(), dB.to_numpy(), rng=np.random.default_rng(0))
                tt, tb = stats.ttest_1samp(dT, 0), stats.ttest_1samp(dBrake, 0)
                c = r["curve"]
                out.append(dict(baseline=base, band=band, window=win, n=r["n"], n_total=len(d),
                                total_dB=float(10 / np.log(10) * dT.mean()),
                                total_p=float(tt.pvalue), brake_dB=float(dBrake.mean()),
                                brake_p=float(tb.pvalue), s_a=r["s_a"], s_b=r["s_b"],
                                s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
                                lam_star_bounded=r["lam_star_bounded"],
                                lam_star=r["lam_star"], hdi_lo=r["hdi"][0], hdi_hi=r["hdi"][1],
                                lam0=c[0][0], lam0_lo=c[0][1], lam0_hi=c[0][2],
                                lam1=c[-1][0], lam1_lo=c[-1][1], lam1_hi=c[-1][2],
                                p_cross_in_01=r["p_cross_in_01"]))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(RES, "brake_lambda.csv"), index=False)
    pd.set_option("display.width", 220)
    print(O.round(3).to_string(index=False))

    # the same effects when each bin's background is any near-optimal solution
    sens = []
    for r in O.itertuples():
        lo, hi = WINDOWS[r.window]
        m1 = ((B.t >= lo) & (B.t < hi)).to_numpy()
        q = fit_sensitivity(B[f"{r.band}_tot"].to_numpy(), [c[r.band] for c in sets],
                            B.patient.to_numpy(), rows0[r.baseline], m1, np.random.default_rng(0))
        sens.append(dict(baseline=r.baseline, band=r.band, window=r.window, n=r.n,
                         lam0=r.lam0, lam1=r.lam1, lam_star=r.lam_star, **q))
    Q = pd.DataFrame(sens)
    Q.to_csv(os.path.join(RES, "brake_fit_sensitivity.csv"), index=False)
    print(Q.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
