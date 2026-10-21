"""Adult age and posterior alpha power in MEG, with the recording floor
measured from empty-room recordings (NIMH Healthy Research Volunteer
Dataset, OpenNeuro ds005752).

The claim. Alpha power falls with adult age. Quinn et al. (2025, eLife
RP108829) report, in MEG, lower 9.5-12.5 Hz and higher 7-8.5 Hz relative
magnitude with age; the EEG re-tests of this repo (aging_lambda.py) use the
periodic power in the individual alpha band. Here: periodic power in
IAF +/- 2 Hz (IAF from the flattened spectrum, 7-14 Hz), regressed on age
with sex as covariate between participants, as a function of the assumed
coupling exponent (lambdacurve.lambda_curve: effects at lambda = 0 and 1
with pairs-bootstrap intervals, lambda* with its Bayesian-bootstrap HDI and
P(lambda* in [0, 1]), verdict). The two fixed bands of Quinn et al. are run
the same way as secondary rows, and the IAF-age slope is reported.

Why MEG. A fitted background contains every aperiodic component, neural or
not; a stable non-neural floor that is a large share of b makes lambda = 1
look robust (coupling-exponent skill, "The recording floor"). MEG has a
measured floor: the empty-room recording, here the 100 s task-noise run
shared by the participants of a month (meg_nimh_extract_psd.py).

Input: results/meg_nimh_psd/ (meg_nimh_extract_psd.py). Spectra: mean over
the posterior axial gradiometers present in both the rest and the
empty-room recording, fT^2/Hz, leaving out broken channels (mean log power
over 2-45 Hz more than 1 decade from the recording's median channel; one
such channel, 7 decades high, otherwise decides a participant's ROI mean).

Backgrounds, fitted by Whittle deviance over 2-45 Hz with 6-16 Hz censored
(as aging_lambda.py):
  (i)   power_law  10^o / f^chi; b its band mean, a = tot - b
  (ii)  kp_out     knee + plateau, b = 10^o / (k + f^chi) without the plateau
                   p, a = tot - b - p (lambdacurve.band_power)
  (iii) kp_in      the same fit with b + p, a unchanged
  (iv)  floor      E[P] = N + L + rhythm with N the empty-room spectrum (scale
                   fixed at 1: it trades with a 1/f L when free),
                   L = 10^o / (fk^chi + f^chi) + q with the knee frequency fk
                   kept at or below the band's low edge (once the floor is out
                   a free knee settles on the rhythm) and q a plateau of the
                   participant (residual muscle, subject noise), fitted on the
                   same bins (2-6 and 16-45 Hz) by Whittle deviance, as
                   trujillo_floor_model.terms; b = band mean of L (q
                   included, as there), a = mean (P - N - L)
  (iv') floor_qout as (iv) with q left out of b (a unchanged)
  (iv'') floor_pl  as (iv) with a power law, L = 10^o / f^chi + q: the floor
                   added to (i), so that the effect of the floor is not
                   confounded with that of the knee
Rows with a <= 0 cannot enter ln a and are dropped; their number is reported
(n_dropped). Quality control as aging_lambda.py: fewer than 10 clean
segments, total band power more than 300x from the median, or IAF at the
edge of 7-14 Hz (no alpha peak) excludes a participant; so does missing sex.

Floor share. Empty-room power over total power, and over the fitted
background (power law; knee + plateau with the plateau in b), in the alpha
band and in a control band (30-40 Hz), and N / (N + L) of (iv).

Calibration (split floor, as trujillo_floor_model.calibration). Each
empty-room recording is split into its first and second half (no shared
data). For each participant a synthetic spectrum N1 + L + R, with L the
participant's own fitted neural background from (iv) (scaled by 0.1, 0.3, 1
or 3) and R a Gaussian at the IAF (s.d. 1 Hz) with the participant's own
periodic power, is fitted with N2 as the floor. Reported: median b_hat /
b_true, s.d. of ln b_hat - ln b_true next to the between-participant s.d.
of ln b, median a_hat / a_true, and the age slopes of ln b and ln a
recovered from the synthetic set against their true values. The same with
N2 replaced by the empty room of another, randomly chosen session
(cross-session: the floor of the rest recording comes from another day).

Gain. Between participants, MEG gain depends on the distance between the
head and the sensors (not the skull). The head coils (.hc, dewar
coordinates) give the head origin (midpoint of the ear coils); its mean
distance to the posterior sensors is correlated with age, and added as a
covariate in a sensitivity row.

Writes results/meg_nimh_age_alpha.csv (claims), results/meg_nimh_floor.csv
(floor shares, calibration, floor drift across sessions, gain proxy) and,
locally, results/meg_nimh_psd/bandpower.csv (per participant).

Usage: python meg_nimh_age_alpha.py [--psd DIR] [--nboot 4000] [--workers 12]
"""
import argparse
import csv
import glob
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lambdacurve", "src"))
from lambdacurve import band_power, lambda_curve, peak_frequency  # noqa: E402
from trujillo_floor_model import L_eval  # noqa: E402
from aging_lambda import slope_ci  # noqa: E402

RES = os.path.join(HERE, "..", "results")
FIT = (2.0, 45.0)
CENSOR = ((6.0, 16.0),)
FLOOR_BINS = ((2.0, 6.0), (16.0, 45.0))     # the bins of the censored fits
CONTROL = (30.0, 40.0)
SEARCH = (7.0, 14.0)
QUINN = {"7-8.5 Hz": (7.0, 8.5), "9.5-12.5 Hz": (9.5, 12.5)}
SCALES = (0.1, 0.3, 1.0, 3.0)
BAD_DEC = 1.0
BACKGROUNDS = ("power_law", "kp_out", "kp_in", "floor", "floor_qout", "floor_pl")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- loading

def roi_mean(d, key, names):
    ix = [list(d["ch_names"]).index(c) for c in names]
    return d[key][ix].mean(0).astype(float)


def bad_channels(d):
    """Channels whose mean log10 power over 2-45 Hz lies more than BAD_DEC
    decades from the median channel of the recording (a broken SQUID; the
    channels.tsv files mark every channel good). Normal posterior topography
    and head position move channels by up to ~0.7 decades."""
    f = d["f"]
    m = (f >= FIT[0]) & (f <= FIT[1])
    lp = np.log10(np.maximum(d["P"][:, m], 1e-12)).mean(1)
    return {c for c, v in zip(d["ch_names"], lp - np.median(lp)) if abs(v) > BAD_DEC}


def load(psd):
    """Per participant: ROI-mean rest and empty-room spectra over the channels
    present and not broken (bad_channels) in both, plus metadata. Returns a
    list of dicts."""
    files = pd.read_csv(os.path.join(psd, "files.csv"))
    noise = {}
    out = []
    for _, r in files.iterrows():
        fn = os.path.join(psd, f"rest_{r.subject}.npz")
        nf = os.path.join(psd, f"noise_{r.noise_etag}.npz")
        if not (os.path.exists(fn) and os.path.exists(nf)):
            continue
        d = np.load(fn)
        if r.noise_etag not in noise:
            noise[r.noise_etag] = dict(np.load(nf))
        n = noise[r.noise_etag]
        bad = bad_channels(d) | bad_channels(n)
        names = [c for c in d["ch_names"] if c in set(n["ch_names"]) and c not in bad]
        rec = dict(subject=r.subject, age=float(r.age), sex=r.sex, noise=r.noise_etag,
                   names=names, n_bad=len(bad), f=d["f"].astype(float), n_ch=len(names),
                   n_ch_rest=len(d["ch_names"]),
                   n_seg=int(d["n_seg"]), n_seg_total=int(d["n_seg_total"]),
                   n_seg_noise=int(n["n_seg"]))
        for k in ("P", "P_all", "P_h1", "P_h2"):
            if k in d:
                rec[k] = roi_mean(d, k, names)
        for k, nk in (("P", "N"), ("P_all", "N_all"), ("P_h1", "N1"), ("P_h2", "N2")):
            if k in n:
                rec[nk] = roi_mean(n, k, names)
        if "hc_lpa" in d:
            ix = [list(d["ch_names"]).index(c) for c in names]
            origin = (d["hc_lpa"] + d["hc_rpa"]) / 2.0
            rec["dist_cm"] = float(np.linalg.norm(d["sensor_pos_cm"][ix] - origin, axis=1).mean())
            rec["head_width_cm"] = float(np.linalg.norm(d["hc_lpa"] - d["hc_rpa"]))
        out.append(rec)
    return out, noise


# ---------------------------------------------------------------- the floor model

def fit_floor(P, f, N, keep, knee_max, model="knee_plateau"):
    """Whittle fit of N + L on the kept bins; trujillo_floor_model.
    fit_floor_model with the knee bound as a parameter (it is 1.5 Hz there)
    and the floor scale fixed at 1. model "knee_plateau": L = 10^o /
    (fk^chi + f^chi) + q, fk <= knee_max, theta = [o, chi, log10 fk, log10 q];
    "fixed_plateau": L = 10^o / f^chi + q, theta = [o, chi, log10 q]."""
    Pk, fk, Nk = P[keep], f[keep], N[keep]
    resid = np.clip(Pk - Nk, Pk * 1e-3, None)
    o0 = float(np.log10(np.median(resid * fk ** 2)))
    q_hi = float(np.log10(np.max(Pk)))
    knee = model == "knee_plateau"
    bounds = ([(o0 - 6.0, o0 + 6.0), (0.05, 6.0)]
              + ([(-2.0, np.log10(knee_max))] if knee else []) + [(q_hi - 10.0, q_hi)])
    lo, hi = [b[0] for b in bounds], [b[1] for b in bounds]
    q0 = float(np.log10(np.median(resid[fk >= 30]) if np.any(fk >= 30) else np.min(resid)))

    def dev(th):
        mu = Nk + L_eval(th, fk, model)
        return float(np.sum(np.log(mu) + Pk / mu))
    best = None
    for chi0 in (1.0, 2.0, 3.0):
        for k0 in ((-1.0, 0.0, np.log10(min(5.0, knee_max))) if knee else (0.0,)):
            for do, dq in ((0.0, 0.0), (-2.0, -1.0), (0.0, -3.0), (chi0 * k0, 0.0)):
                th0 = [o0 + do, chi0] + ([k0] if knee else []) + [q0 + dq]
                th0 = np.clip(th0, lo, hi)
                r = minimize(dev, th0, method="L-BFGS-B", bounds=bounds,
                             options=dict(maxiter=2000, ftol=1e-13))
                if best is None or r.fun < best.fun:
                    best = r
    return best.x


def floor_terms(P, f, N, band, model="knee_plateau", theta=None):
    """tot, floor, b (L incl. q), b_noplat and a = mean(P - N - L) in band,
    and the same in CONTROL, from the floor model (see top); the knee is kept
    at or below the band's low edge."""
    if theta is None:
        keep = np.zeros(f.size, bool)
        for lo, hi in FLOOR_BINS:
            keep |= (f >= lo) & (f <= hi)
        theta = fit_floor(P, f, N, keep, band[0], model)
    out = dict(theta=theta, exponent=float(theta[1]),
               knee_freq=float(10.0 ** theta[2]) if model == "knee_plateau" else 0.0,
               q=float(10.0 ** theta[-1]))
    for lab, (lo, hi) in (("", band), ("ctl_", CONTROL)):
        sel = (f >= lo) & (f <= hi)
        L = L_eval(theta, f[sel], model)
        out[f"{lab}tot"] = float(P[sel].mean())
        out[f"{lab}floor"] = float(N[sel].mean())
        out[f"{lab}b"] = float(L.mean())
        out[f"{lab}b_noplat"] = float(L_eval(theta, f[sel], model, plateau=False).mean())
        out[f"{lab}a"] = float((P[sel] - N[sel] - L).mean())
    return out


# ---------------------------------------------------------------- per participant

def bands_of(iaf):
    return {"IAF +/- 2 Hz": (iaf - 2.0, iaf + 2.0), **QUINN}


def measures(P, N, f, band, tag):
    """Rows (one per background) of tot, a, b and floor shares for one band."""
    rows = []
    ctl = (f >= CONTROL[0]) & (f <= CONTROL[1])
    inb = (f >= band[0]) & (f <= band[1])
    Nb, Nc = float(N[inb].mean()), float(N[ctl].mean())
    for model in ("fixed", "knee_plateau"):
        bp = band_power(P, f, band, FIT, CENSOR, model)
        bc = band_power(P, f, CONTROL, FIT, CENSOR, model)
        if bp is None or bc is None:
            continue
        flag = bool(bp["exponent"] >= 5.9 or bp["knee_freq"] >= band[0])
        common = dict(tot=bp["tot"], floor=Nb, exponent=bp["exponent"],
                      knee_freq=bp["knee_freq"], plateau=bp["plateau"], knee_flag=flag,
                      ctl_tot=bc["tot"], ctl_floor=Nc)
        if model == "fixed":
            rows.append(dict(common, background="power_law", a=bp["a"], b=bp["b"],
                             ctl_b=bc["b"]))
        else:
            rows.append(dict(common, background="kp_out", a=bp["a"], b=bp["b"], ctl_b=bc["b"]))
            rows.append(dict(common, background="kp_in", a=bp["a"], b=bp["b"] + bp["plateau"],
                             ctl_b=bc["b"] + bc["plateau"]))
    for model in ("fixed_plateau", "knee_plateau"):
        tt = floor_terms(P, f, N, band, model)
        common = dict(tot=tt["tot"], floor=tt["floor"], exponent=tt["exponent"],
                      knee_freq=tt["knee_freq"], plateau=tt["q"],
                      knee_flag=bool(tt["exponent"] >= 5.9), ctl_tot=tt["ctl_tot"],
                      ctl_floor=tt["ctl_floor"])
        if model == "fixed_plateau":
            rows.append(dict(common, background="floor_pl", a=tt["a"], b=tt["b"],
                             ctl_b=tt["ctl_b"]))
            continue
        t = tt
        rows.append(dict(common, background="floor", a=t["a"], b=t["b"], ctl_b=t["ctl_b"]))
        rows.append(dict(common, background="floor_qout", a=t["a"], b=t["b_noplat"],
                         ctl_b=t["ctl_b_noplat"]))
    for r in rows:
        r["spectra"] = tag
    return rows, t


def one(args):
    rec, other_N = args
    f, P, N = rec["f"], rec["P"], rec["N"]
    iaf = peak_frequency(P, f, search=SEARCH, censor=CENSOR, fit_range=FIT)
    edge = (not np.isfinite(iaf)) or iaf <= SEARCH[0] + 0.25 or iaf >= SEARCH[1] - 0.25
    iaf_use = iaf if np.isfinite(iaf) else 10.0
    base = {k: rec[k] for k in ("subject", "age", "sex", "noise", "n_ch", "n_ch_rest", "n_bad",
                                "n_seg", "n_seg_total", "n_seg_noise") if k in rec}
    base.update(iaf=iaf, iaf_edge=edge, dist_cm=rec.get("dist_cm", np.nan),
                head_width_cm=rec.get("head_width_cm", np.nan))
    rows, cal = [], []
    for bname, band in bands_of(iaf_use).items():
        r, t = measures(P, N, f, band, "rejected")
        rows += [dict(base, band=bname, lo=band[0], hi=band[1], **x) for x in r]
        if "P_all" in rec and "N_all" in rec and bname == "IAF +/- 2 Hz":
            r2, _ = measures(rec["P_all"], rec["N_all"], f, band, "all")
            rows += [dict(base, band=bname, lo=band[0], hi=band[1], **x) for x in r2]
        if bname != "IAF +/- 2 Hz" or "N1" not in rec:
            continue
        # split-floor calibration with this participant's own L and rhythm
        Lf = L_eval(t["theta"], f, "knee_plateau")
        inb = (f >= band[0]) & (f <= band[1])
        g = np.exp(-0.5 * ((f - iaf_use) / 1.0) ** 2)
        a_obs = max(t["a"], 0.0)
        for variant, N1, N2 in (("split", rec["N1"], rec["N2"]),
                                ("cross-session", rec["N"], other_N)):
            if N2 is None or not np.all(np.isfinite(N1)) or not np.all(np.isfinite(N2)):
                continue
            for s in SCALES:
                L_true = s * Lf
                R = g * (s * a_obs / g[inb].mean()) if a_obs > 0 else 0.0 * g
                Psyn = N1 + L_true + R
                ts = floor_terms(Psyn, f, N2, band)
                cal.append(dict(subject=rec["subject"], age=rec["age"], sex=rec["sex"],
                                variant=variant, scale=s, b_true=float(L_true[inb].mean()),
                                a_true=float(R[inb].mean()) if a_obs > 0 else 0.0,
                                b_hat=ts["b"], a_hat=ts["a"], floor=float(N1[inb].mean()),
                                knee_true=t["knee_freq"], knee_hat=ts["knee_freq"]))
    return rows, cal


# ---------------------------------------------------------------- group level

def qc(T):
    F = T[(T.background == "power_law") & (T.band == "IAF +/- 2 Hz") & (T.spectra == "rejected")]
    med = F.tot.median()
    why = pd.Series("", index=F.index)
    why[F.n_seg < 10] += "few clean segments; "
    why[(F.tot < med / 300) | (F.tot > med * 300)] += "band power 300x from median; "
    why[F.iaf_edge] += "no alpha peak (IAF at edge); "
    why[~F.sex.isin(["male", "female"])] += "sex missing; "
    flags = F.assign(why=why.str.rstrip("; "))[["subject", "why"]]
    return T.merge(flags, on="subject", how="left")


def claim_row(D, claim, band, background, spectra, nboot, extra_cov=None, note=""):
    sex = (D.sex == "male").astype(float).to_numpy()
    cov = sex if extra_cov is None else np.column_stack([sex, D[extra_cov].to_numpy(float)])
    with np.errstate(divide="ignore", invalid="ignore"):
        la, lb = np.log(D.a.to_numpy(float)), np.log(D.b.to_numpy(float))
    r = lambda_curve(la, lb, x=D.age.to_numpy(float), covariates=cov, nboot=nboot, seed=0,
                     dropna=True)
    row = dict(claim=claim, band=band, background=background, spectra=spectra,
               covariates="sex" + (f" + {extra_cov}" if extra_cov else ""), per="year",
               **r.to_dict(), s_a_lo=r.s_a_ci[0], s_a_hi=r.s_a_ci[1],
               lam_star_bounded=r.lam_star_bounded,
               floor_over_b_median=float(np.median(D.floor / D.b)),
               floor_over_tot_median=float(np.median(D.floor / D.tot)),
               share_knee_flag=float(D.knee_flag.mean()), note=note)
    return row


def share_rows(T):
    """Floor shares in the alpha and the control band (rejected spectra)."""
    rows = []
    A = T[(T.band == "IAF +/- 2 Hz") & (T.spectra == "rejected") & (T.why == "")]
    for bg, G in A.groupby("background"):
        for lab, num, den in (("alpha: N / total", G.floor, G.tot),
                              ("alpha: N / b", G.floor, G.b),
                              ("alpha: N / (N + b)", G.floor, G.floor + G.b),
                              ("control 30-40 Hz: N / total", G.ctl_floor, G.ctl_tot),
                              ("control 30-40 Hz: N / b", G.ctl_floor, G.ctl_b),
                              ("control 30-40 Hz: N / (N + b)", G.ctl_floor,
                               G.ctl_floor + G.ctl_b)):
            v = (num / den).to_numpy(float)
            if "total" in lab and bg != "power_law":
                continue                      # the same for every background
            rows.append(dict(kind="floor share", quantity=lab,
                             background="(any)" if "total" in lab else bg,
                             median=np.median(v), q25=np.percentile(v, 25),
                             q75=np.percentile(v, 75), q90=np.percentile(v, 90), n=len(v)))
    return rows


def calibration_rows(C, keep_subjects):
    rows = []
    C = C[C.subject.isin(keep_subjects)]
    for (variant, scale), G in C.groupby(["variant", "scale"]):
        lbh = np.log(G.b_hat.clip(lower=1e-30))
        e = lbh - np.log(G.b_true)
        G = G.assign(lnb_err=e)
        sex = (G.sex == "male").astype(float).to_numpy()
        X = np.column_stack([np.ones(len(G)), G.age, sex])

        def slope(y):
            ok = np.isfinite(y)
            return float(np.linalg.lstsq(X[ok], y[ok], rcond=None)[0][1])
        okA = (G.a_hat > 0) & (G.a_true > 0)
        rows.append(dict(kind="calibration", quantity=f"{variant}, L x {scale:g}",
                         background="floor", n=len(G),
                         floor_share_median=float(np.median(G.floor / (G.floor + G.b_true))),
                         b_ratio_median=float(np.exp(np.median(e))), lnb_err_sd=float(e.std()),
                         lnb_between_sd=float(np.log(G.b_true).std()),
                         r_lnb=float(np.corrcoef(np.log(G.b_true), lbh)[0, 1]),
                         a_ratio_median=float(np.median(G.a_hat[okA] / G.a_true[okA])),
                         share_a_hat_nonpos=float((G.a_hat <= 0).mean()),
                         s_b_true=slope(np.log(G.b_true.to_numpy())),
                         s_b_hat=slope(lbh.to_numpy()),
                         s_a_true=slope(np.log(G.a_true.where(G.a_true > 0).to_numpy(float))),
                         s_a_hat=slope(np.log(G.a_hat.where(okA).to_numpy(float)))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd", default=os.path.join(RES, "meg_nimh_psd"))
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    recs, noise = load(a.psd)
    print(f"{len(recs)} participants with rest and empty-room spectra, "
          f"{len(noise)} empty-room recordings")
    rng = np.random.default_rng(1)
    jobs = []
    for r in recs:
        others = [k for k in noise if k != r["noise"]]
        other = None
        if others:
            n = noise[others[rng.integers(len(others))]]
            names = [c for c in r["names"] if c in set(n["ch_names"]) - bad_channels(n)]
            other = roi_mean(n, "P", names)
        jobs.append((r, other))
    with Pool(a.workers) as pool:
        res = pool.map(one, jobs, chunksize=2)
    T = pd.DataFrame([x for rows, _ in res for x in rows])
    C = pd.DataFrame([x for _, cal in res for x in cal])
    T = qc(T)
    T.to_csv(os.path.join(a.psd, "bandpower.csv"), index=False)
    C.to_csv(os.path.join(a.psd, "calibration.csv"), index=False)
    excl = T[(T.background == "power_law") & (T.band == "IAF +/- 2 Hz")
             & (T.spectra == "rejected") & (T.why != "")]
    print(f"excluded by QC: {len(excl)}")
    for _, r in excl.iterrows():
        print(f"  {r.subject}: {r.why}")
    good = T[T.why == ""]
    keep_subjects = sorted(good.subject.unique())

    # ---- claims
    rows = []
    for (band, bg, spectra), D in good.groupby(["band", "background", "spectra"], sort=False):
        if spectra == "all" and bg not in ("power_law", "floor"):
            continue
        claim = "alpha periodic power falls with adult age"
        rows.append(claim_row(D, claim, band, bg, spectra, a.nboot,
                              note="" if spectra == "rejected" else "no segment rejection"))
        if band == "IAF +/- 2 Hz" and spectra == "rejected" and bg in ("power_law", "floor"):
            Dd = D[np.isfinite(D.dist_cm)]
            rows.append(claim_row(Dd, claim, band, bg, spectra, a.nboot, extra_cov="dist_cm",
                                  note="head-sensor distance as covariate"))
    O = pd.DataFrame(rows)
    O["_b"] = O.band.map({b: k for k, b in enumerate(bands_of(10.0))})
    O["_g"] = O.background.map({b: k for k, b in enumerate(BACKGROUNDS)})
    O = O.sort_values(["_b", "spectra", "covariates", "_g"],
                      ascending=[True, False, True, True]).drop(columns=["_b", "_g"])

    # published-style measures and the IAF
    D = good[(good.band == "IAF +/- 2 Hz") & (good.background == "power_law")
             & (good.spectra == "rejected")]
    sex = (D.sex == "male").astype(float).to_numpy()
    pub = []
    for meas, y in (("IAF (Hz)", D.iaf), ("total alpha power, IAF +/- 2 (ln)", np.log(D.tot)),
                    ("flattened alpha, ln(tot / b), power law", np.log(D.tot / D.b))):
        s, lo, hi, p, n = slope_ci(np.asarray(y, float), D.age.to_numpy(float), sex)
        pub.append(dict(claim=meas, band="IAF +/- 2 Hz", background="power_law",
                        spectra="rejected", covariates="sex", per="year", n=n, s_a=s,
                        s_a_lo=lo, s_a_hi=hi, note=f"OLS, HC3 interval, p = {p:.3g}"))
    O = pd.concat([O, pd.DataFrame(pub)], ignore_index=True)
    O.to_csv(os.path.join(RES, "meg_nimh_age_alpha.csv"), index=False)

    # ---- floor shares, calibration, drift of the floor, gain proxy
    F = share_rows(T)
    G = good[(good.background == "floor") & (good.band == "IAF +/- 2 Hz")
             & (good.spectra == "rejected")]
    F += calibration_rows(C, keep_subjects)
    f0 = recs[0]["f"]
    # the same recording can carry two ETags (single- and multipart uploads):
    # sessions are told apart by their spectra
    sess, seen = [], set()
    for n in noise.values():
        good = [i for i, c in enumerate(n["ch_names"]) if c not in bad_channels(n)]
        key = round(float(n["P"][good].mean()), 6)
        if key not in seen:
            seen.add(key)
            sess.append({k: n[k][good] for k in ("P", "P_h1", "P_h2")})
    for lab, (lo, hi) in (("alpha 8-12 Hz", (8.0, 12.0)), ("control 30-40 Hz", CONTROL)):
        sel = (f0 >= lo) & (f0 <= hi)
        v = np.array([np.log(n["P"][:, sel].mean()) for n in sess])
        h = np.array([np.log(n["P_h2"][:, sel].mean() / n["P_h1"][:, sel].mean())
                      for n in sess])
        F.append(dict(kind="floor drift", quantity=f"{lab}: s.d. of ln N across sessions",
                      median=float(v.std(ddof=1)), q25=float(np.exp(np.median(v))),
                      q75=float(np.exp(v.max())), n=len(v),
                      note="q25: median N, q75: largest N (fT^2/Hz)"))
        F.append(dict(kind="floor drift", quantity=f"{lab}: s.d. of ln N2/N1 within session",
                      median=float(h.std(ddof=1)), n=len(h)))
    Gd = G[np.isfinite(G.dist_cm)]
    for xname in ("dist_cm", "head_width_cm"):
        r, p = stats.pearsonr(Gd[xname], Gd.age)
        rs, ps = stats.spearmanr(Gd[xname], Gd.age)
        F.append(dict(kind="gain", quantity=f"{xname} vs age", median=float(Gd[xname].median()),
                      q25=float(Gd[xname].quantile(0.25)), q75=float(Gd[xname].quantile(0.75)),
                      r=float(r), p=float(p), rho=float(rs), p_rho=float(ps), n=len(Gd)))
    for yname, y in (("ln b (floor model)", np.log(Gd.b)), ("ln tot", np.log(Gd.tot))):
        r, p = stats.pearsonr(Gd.dist_cm, y)
        F.append(dict(kind="gain", quantity=f"dist_cm vs {yname}", r=float(r), p=float(p),
                      n=len(Gd)))
    FF = pd.DataFrame(F)
    FF.to_csv(os.path.join(RES, "meg_nimh_floor.csv"), index=False)

    # ---- print
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"\nn analysed {len(keep_subjects)}; ROI channels per participant: "
          f"{G.n_ch.min()}-{G.n_ch.max()} (median {int(G.n_ch.median())}); "
          f"participants with broken channels left out: {int((G.n_bad > 0).sum())}")
    print(FF[FF.kind == "floor share"][["quantity", "background", "median", "q25", "q75", "q90",
                                       "n"]]
          .round(4).to_string(index=False))
    print(FF[FF.kind == "calibration"][["quantity", "n", "floor_share_median", "b_ratio_median",
                                       "lnb_err_sd", "lnb_between_sd", "r_lnb", "a_ratio_median",
                                       "share_a_hat_nonpos", "s_b_true", "s_b_hat", "s_a_true",
                                       "s_a_hat"]].round(4).to_string(index=False))
    print(FF[FF.kind.isin(["floor drift", "gain"])].dropna(axis=1, how="all").round(4)
          .to_string(index=False))
    cols = ["band", "background", "spectra", "covariates", "n", "n_dropped", "s_a", "s_b",
            "s_b_lo", "s_b_hi", "lam0", "lam0_lo", "lam0_hi", "lam1", "lam1_lo", "lam1_hi",
            "lam_star", "hdi_lo", "hdi_hi", "p_cross_in_01", "verdict", "floor_over_b_median",
            "share_knee_flag"]
    print(O[O.claim.str.startswith("alpha")][cols].round(4).to_string(index=False))
    print(O[~O.claim.str.startswith("alpha")][["claim", "n", "s_a", "s_a_lo", "s_a_hi", "note"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
