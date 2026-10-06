"""Calibration of the within-session coupling estimate on real spectra.

The existing calibrations (ds003690_lambda.py and dortmund_levels.py with
--simulate) replace the data by synthetic spectra. Here the real epoch
spectra are kept, with their noise, and a synthetic rhythm with a known
coupling exponent is added in a band where they have no peak: a Gaussian
centred at 33 Hz with s.d. 2 Hz (the grand-mean residual from a smooth
background is flat over 27-39 Hz in both datasets, eyes open and closed).
The band is analysed as the alpha band is: band total T over 31-35 Hz;
background from a power law fitted to the flanks of the 27-39 Hz gap only,
23-27 and 39-45 Hz, and rescaled per participant to the mean spectrum
interpolated across the gap (ap_models.flank_level); three independent
estimates (ds003690: the three DPSS tapers, all six assignments through
ds003690_lambda.stack; Dortmund: the band bins and two interleaved sets of
fit bins 3 bins apart, both assignments, as in dortmund_levels.arrays); and
lambda_gmm.estimate_levels on the grid -1 to 3. Participants are those of the
real alpha analysis (an alpha peak in the mean spectrum).
--background power-law gives the earlier background instead: the censored
log-log fit (ap_models.loglog_fit) over 2-45 Hz with 6-16 Hz and 27-39 Hz
left out (ds003690's own power-law fit stops at 40 Hz, which would leave
almost no bins above the band), with Dortmund's two sets of fit bins 2 bins
apart. That fit also gives, with either background, the references 'own'
and 'pooled', the synthetic spectra and the alpha peak heights below.

Injected power in epoch j of participant i:

    A_ij = c_i Bref_ij^lambda exp(eps_ij) [exp(kappa v_ij - kappa^2 / 2)]

spread over frequency as the periodogram of an independent segment of a
Gaussian rhythm taken through the same window or tapers, so that its own
estimation noise is exponential per bin and correlated across bins and
tapers as in the data. Bref_ij ('inband') is the epoch's real power in the
band before injection, averaged over tapers: it follows the real background
fluctuation, and its estimation noise is independent of the out-of-band fits
that give B and the instrument, as the noise of a true background would be.
'own' (ds003690 only) makes the rhythm in each taper follow that taper's own
out-of-band fit, which in every assignment is neither the B nor the
instrument taper: coupling to the fitted broadband level rather than to the
band's own power, still with independent noise. 'pooled' uses the most
precise background estimate, the out-of-band fit to all tapers (ds003690) or
all fit bins (Dortmund); it shares estimation noise with B and the
instrument and so adds coupling to the truth.
'synthetic' replaces the data by idealised spectra matched to each
participant (power-law background from the same fit, log-normal epoch
fluctuation, exponential noise), as the existing calibrations do, with the
same rhythm, to isolate what the real noise does.
c_i = h_i median_j(Bref_ij)^(1 - lambda), log10 h_i ~ N(log10 h, 0.25), with
h = 1/3, 1 or 3 times the median relative alpha peak height in the same data
(computed as in the existing calibrations); eps ~ N(0, 0.3), independent.
Replicates redraw h_i, eps and the rhythm's noise; the real data are fixed,
so a participant bootstrap (first replicate, middle height) gives the
sampling spread.

Common drive: v_ij, standardised within participant, is either correlated
rho with ln Bref_ij or the participant's real alpha-band power (individual
alpha frequency +/- 2 Hz). To first order it shifts lambda-hat by
kappa cov(v, ln b) / var(ln b); both moments are taken against the
instrument z = ln Bz, cov(v, z) / cov(ln B, z), whose noise is independent.

Checks: the real alpha-band estimates are recomputed with the original
functions and set beside those in results/, and the generalised fit functions
used here are compared with the originals on the alpha configuration.

Outputs (group level): results/inject_calibration_runs.csv (one row per
run), results/inject_calibration_summary.csv (mean and s.d. over
replicates), results/inject_calibration_drive.csv (drive runs, with the
analytic shift) and results/inject_calibration_checks.csv.

Usage: python inject_calibration.py [--ds003690 DIR] [--dortmund DIR]
       [--conds ds003690,ec_pre,eo_pre,ec_post,eo_post] [--reps 5]
       [--nboot 50] [--workers 5] [--max-subjects N] [--seed 0] [--out DIR]
       [--background flanks|power-law] [--tag SUFFIX]
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker on a shared machine

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal.windows import dpss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dortmund_levels as DV
import ds003690_lambda as L3
import lambda_gmm as G
from ap_models import flank_bins, flank_level, loglog_fit

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.environ.get("EEG_DATA", "eeg_data")

CF, SD, HALF = 33.0, 2.0, 2.0              # injected rhythm; analysis band CF +/- HALF
FIT = (2.0, 45.0)
CENSORS = ((6.0, 16.0), (27.0, 39.0))      # alpha, and the injected band
FLANKS = ((23.0, 27.0), (39.0, 45.0))      # fit bins of the flank background
GRID = DV.GRID
SR = 250.0
CONDS = ("ds003690", "ec_pre", "eo_pre", "ec_post", "eo_post")
LAMS = (0.0, 0.5, 1.0, 1.5)
H_MULT = (1 / 3, 1.0, 3.0)
H_SD = 0.25                                # log10 s.d. of h_i across participants
EPS_SD = 0.3                               # intrinsic fluctuation, as in the existing simulations
RHOS = (0.3, 0.6)
KAPPAS = (0.1, 0.25, 0.5, 0.75, 1.0)
PERMS = list(itertools.permutations(range(3)))   # taper assignments, as ds003690_lambda.stack


def fit_mask(f, fit=FIT, censors=CENSORS):
    keep = (f >= fit[0]) & (f <= fit[1])
    for lo, hi in censors:
        keep &= ~((f >= lo) & (f <= hi))
    return keep


def ln_background(P, f, keep):
    """ds003690_lambda.fit_background with any set of fit bins."""
    X = np.column_stack([np.ones(keep.sum()), np.log(f[keep])])
    beta = loglog_fit(P[..., keep].reshape(-1, keep.sum()).T, X)
    lf = np.log(np.maximum(f, 1e-9))
    lnL = beta[0][:, None] + beta[1][:, None] * lf[None, :]
    return lnL.reshape(P.shape)


def interleaved(f, keep, step=6):
    """dortmund_levels.fit_sets with any set of fit bins: every step-th bin,
    and the bins half-way between (step 6 for the flank background, 4 for the
    power law)."""
    idx = np.where(keep)[0]
    return idx[idx % step == 0], idx[idx % step == step // 2]


def within(x, inv):
    """Centre x within participants (inv: participant index per row)."""
    m = np.bincount(inv, x) / np.bincount(inv)
    return x - m[inv]


def standardise(x):
    s = np.std(x)
    return (x - x.mean()) / s if s > 0 else x * 0.0


# ---------------------------------------------------------------- data

def check_alpha(cond, units, args):
    """Real alpha-band estimate through the original functions, and the
    generalised fit functions against the originals."""
    f = units[0]["f"]
    fl = L3.FLANKS if args.background == "flanks" else None
    tag = "" if fl else "_c16"
    read = lambda name: pd.read_csv(name) if os.path.exists(name) else pd.DataFrame(
        columns=["group", "covariates", "cond", "lam"])
    if cond == "ds003690":
        rows = L3.band_arrays(units, fl)
        T, B, Bz, grp, _ = L3.stack(rows, [])
        lam = G.estimate_levels(T, B, Bz, grp, None, grid=GRID)["lam"]
        inst = L3.instrument_strength(rows, [])
        keep0 = fit_mask(f, L3.FIT, (L3.CENSOR,))
        same = all(np.array_equal(ln_background(u["P"], f, keep0), L3.fit_background(u["P"], f))
                   for u in units[:5])
        ref = read(os.path.join(RES, f"ds003690_lambda{tag}.csv")).query(
            "group == 'all' and covariates == 'none'")["lam"]
    else:
        T, B, Bz, grp, _ = DV.arrays(units, False, fl)
        lam = G.estimate_levels(T, B, Bz, grp, None, grid=GRID)["lam"]
        # first half of each participant's rows = first assignment, as in dortmund_levels.run
        inst = float(np.nanmean([np.corrcoef(np.log(B[grp == s][: (grp == s).sum() // 2]),
                                             np.log(Bz[grp == s][: (grp == s).sum() // 2]))[0, 1]
                                 for s in np.unique(grp)]))
        keep0 = fit_mask(f, DV.FIT, (DV.CENSOR,))
        same = all(np.array_equal(a, b) for a, b in zip(
            interleaved(f, keep0, 4) + interleaved(f, flank_bins(f, L3.FLANKS)),
            DV.fit_sets(f) + DV.fit_sets(f, L3.FLANKS)))
        ref = read(os.path.join(RES, f"dortmund_levels{tag}.csv")).query(
            "cond == @cond and covariates == False")["lam"]
    return dict(alpha_lam=lam, alpha_lam_published=ref.iloc[0] if len(ref) else np.nan,
                alpha_instrument_r=inst, fit_functions_identical=bool(same))


def prepare(cond, args):
    ds = cond == "ds003690"
    if ds:
        units = L3.load(args.ds003690)
        units = units[:args.max_subjects] if args.max_subjects else units
    else:
        units = DV.load(args.dortmund, cond, args.max_subjects)
    chk = check_alpha(cond, units, args)
    f = units[0]["f"]
    keep = fit_mask(f)
    band = (f >= CF - HALF) & (f <= CF + HALF)
    s0, s2 = interleaved(f, keep, 4)
    a0, _ = DV.fit_sets(f)
    out = []
    for u in units:
        P = u["P"]
        Pm = P.mean(axis=(0, 1)) if ds else P.mean(0)
        cf = L3.iaf(Pm, f) if ds else DV.iaf(Pm, f)
        if not np.isfinite(cf):
            continue
        i = np.argmin(np.abs(f - cf))
        ab = (f >= cf - 2) & (f <= cf + 2)
        if ds:
            Lcf = np.exp(L3.fit_background(P, f).mean(axis=(0, 1))[i])
            t_real = P[..., band].mean(-1)                               # epochs x tapers
            ref_in = t_real.mean(1)
            ref_pool = np.exp(ln_background(P.mean(1), f, keep))[:, band].mean(1)
            ref_own = np.exp(ln_background(P, f, keep))[..., band].mean(-1)  # epochs x tapers
            t_alpha = P[..., ab].mean(-1).mean(1)
        else:
            Lcf = np.exp(DV.lnL(P, f, a0).mean(0)[i])
            ref_in = P[:, band].mean(1)
            ref_pool = np.exp(DV.lnL(P, f, np.where(keep)[0]))[:, band].mean(1)
            ref_own = None
            t_alpha = P[:, ab].mean(1)
        # relative alpha peak height as in the existing simulate() functions
        out.append(dict(subject=u["subject"], P=P, iaf=cf, h_alpha=max(Pm[i] / Lcf - 1, 0.1),
                        ref_in=ref_in, ref_pool=ref_pool, ref_own=ref_own, t_alpha=t_alpha))
    D = dict(ds=ds, cond=cond, f=f, keep=keep, band=band, s0=s0, s2=s2, units=out,
             flanks=FLANKS if args.background == "flanks" else None,
             sets=interleaved(f, flank_bins(f, FLANKS)))
    chk.update(n_units=len(out), n_epochs=int(sum(u["P"].shape[0] for u in out)),
               iaf_median=float(np.median([u["iaf"] for u in out])),
               h_alpha_median=float(np.median([u["h_alpha"] for u in out])),
               h_alpha_q25=float(np.percentile([u["h_alpha"] for u in out], 25)),
               h_alpha_q75=float(np.percentile([u["h_alpha"] for u in out], 75)))
    return D, chk


# ---------------------------------------------------------------- rhythm

def rhythm_setup(D):
    """Window(s), frequency bins and normalisation of the rhythm periodogram."""
    n = int((2.0 if D["ds"] else 4.0) * SR)
    win = dpss(n, 2.0, 3) if D["ds"] else np.hanning(n + 1)[:n][None, :]
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    fsel = np.array([np.argmin(np.abs(fr - x)) for x in D["f"]])
    assert np.allclose(fr[fsel], D["f"], atol=1e-3)
    k = np.argmin(np.abs(D["f"] - CF))
    rng = np.random.default_rng(12345)
    acc, m = np.zeros(win.shape[0]), 0
    for _ in range(10):
        acc += rhythm(2000, win, fsel, rng)[:, :, k].sum(0)
        m += 2000
    return win, fsel, acc / m


def rhythm(n_ep, win, fsel, rng, norm=None):
    """Periodograms (epochs x windows x bins) of independent segments of a
    Gaussian process with spectrum exp(-(f - CF)^2 / (2 SD^2)); with norm,
    scaled to an expected value of 1 at the peak bin."""
    n = win.shape[1]
    m = 2 * n
    fm = np.fft.rfftfreq(m, 1.0 / SR)
    amp = np.exp(-0.25 * ((fm - CF) / SD) ** 2)
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal((n_ep, m)), axis=1) * amp, n=m, axis=1)
    x = x[:, n // 2:n // 2 + n]
    P = (np.abs(np.fft.rfft(x[:, None, :] * win[None], axis=2)) ** 2)[:, :, fsel]
    return P if norm is None else P / norm[None, :, None]


def amplitude(ref, lam, h, eps, v=None, kappa=0.0):
    """A = c ref^lambda e^eps, with c = h median(ref)^(1 - lambda)."""
    med = np.median(ref)
    A = h * med * (ref / med) ** lam * np.exp(eps)
    if kappa:
        A = A * np.exp(kappa * v - kappa ** 2 / 2)
    return A


# ---------------------------------------------------------------- analysis

def analyse(D, spectra):
    """Band T, B, Bz and participant labels, stacked as the original scripts
    stack them, plus the instrument strength."""
    f, keep, band, fl = D["f"], D["keep"], D["band"], D["flanks"]
    if D["ds"]:
        rows = []
        bins = np.where(flank_bins(f, FLANKS))[0]
        for u, P in zip(D["units"], spectra):
            if fl is None:
                B = np.exp(ln_background(P, f, keep))[..., band].mean(-1)
            else:
                B = np.column_stack([flank_level(P[:, a], f, bins, band, fl, P.mean(axis=(0, 1)))
                                     for a in range(3)])
            rows.append(dict(subject=u["subject"], T=P[..., band].mean(-1), B=B,
                             cov=np.zeros((P.shape[0], 4))))
        T, B, Bz, grp, _ = L3.stack(rows, [])
        return T, B, Bz, grp, L3.instrument_strength(rows, []), 6
    T, B, Bz, grp, rs = [], [], [], [], []
    for u, P in zip(D["units"], spectra):
        if fl is None:
            Lb = [np.exp(DV.lnL(P, f, s))[:, band].mean(1) for s in (D["s0"], D["s2"])]
        else:
            Lb = [flank_level(P, f, s, band, fl) for s in D["sets"]]
        t = P[:, band].mean(1)
        for bi, zi in ((0, 1), (1, 0)):
            T.append(t); B.append(Lb[bi]); Bz.append(Lb[zi])
            grp.append(np.full(t.size, u["subject"]))
        rs.append(np.corrcoef(np.log(Lb[0]), np.log(Lb[1]))[0, 1])
    T, B, Bz, grp = (np.concatenate(v) for v in (T, B, Bz, grp))
    return T, B, Bz, grp, float(np.nanmean(rs)), 2


def stacked(D, per_epoch, reps):
    """Per-epoch values repeated in the row order of analyse()."""
    return np.concatenate([np.tile(x, reps) for x in per_epoch])


def moments(T, B, Bz, grp, v=None):
    """Within-participant diagnostics: s.d. of ln b (from cov(ln B, z)), the
    mean relative band excess (T - B) / B, and cov(v, z) / cov(ln B, z)."""
    _, inv = np.unique(grp, return_inverse=True)
    z, lb = within(np.log(Bz), inv), within(np.log(B), inv)
    cbz = np.sum(lb * z)
    out = dict(sd_lnb=float(np.sqrt(max(cbz / T.size, 0))), rel_excess=float(np.sum(T - B) / np.sum(B)))
    if v is not None:
        out["beta_vb"] = float(np.sum(within(v, inv) * z) / cbz)
    return out


def estimate(T, B, Bz, grp):
    e = G.estimate_levels(T, B, Bz, grp, None, grid=GRID)
    return e["lam"], len(e["roots"])


def bootstrap(T, B, Bz, grp, lam0, nboot, rng):
    """Participant bootstrap, root nearest the full-sample estimate (as dortmund_levels.run)."""
    subs = np.unique(grp)
    idx = {s: np.where(grp == s)[0] for s in subs}
    out = []
    for _ in range(nboot):
        pick = rng.choice(subs, subs.size, replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        g2 = np.concatenate([np.full(idx[s].size, f"{s}_{j}") for j, s in enumerate(pick)])
        e = G.estimate_levels(T[rows], B[rows], Bz[rows], g2, None, grid=GRID)
        r = e["roots"]
        out.append(min(r, key=lambda x: abs(x - lam0)) if r and np.isfinite(lam0) else e["lam"])
    return np.array(out, float)


# ---------------------------------------------------------------- synthetic twin

def twin_model(D):
    """Per participant: mean ln background (power law from the same fit) and
    the s.d. of the epoch fluctuation at CF net of estimation noise, as in the
    existing simulate() functions."""
    f, i = D["f"], np.argmin(np.abs(D["f"] - CF))
    for u in D["units"]:
        P = u["P"]
        if D["ds"]:
            lnL = ln_background(P, f, D["keep"])
            lev = lnL[:, :, i].mean(1)
            noise_var = np.var(lnL[:, 0, i] - lnL[:, 1, i]) / 2
            u["twin"] = (lnL.mean(axis=(0, 1)), np.sqrt(max(np.var(lev) - noise_var / 3, 1e-4)))
        else:
            L0 = DV.lnL(P, f, D["s0"])
            noise_var = np.var(L0[:, i] - DV.lnL(P, f, D["s2"])[:, i]) / 2
            u["twin"] = (L0.mean(0), np.sqrt(max(np.var(L0[:, i]) - noise_var, 1e-4)))


def twin_draws(D, rng):
    out = []
    for u in D["units"]:
        mean_lnL, sd = u["twin"]
        n = u["P"].shape[0]
        L = np.exp(mean_lnL[None, :] + rng.normal(0, sd, n)[:, None])
        E = rng.exponential(1.0, (n, 3, L.shape[1]) if D["ds"] else L.shape)
        out.append(dict(L=L, E=E, ref=L[:, D["band"]].mean(1)))
    return out


def twin_spectra(D, tw, A_list):
    g = np.exp(-0.5 * ((D["f"] - CF) / SD) ** 2)
    out = []
    for t, A in zip(tw, A_list):
        S = t["L"] + A[:, None] * g[None, :]
        out.append(S[:, None, :] * t["E"] if D["ds"] else S * t["E"])
    return out


# ---------------------------------------------------------------- runs

def inject(D, A_list, draws):
    """Real spectra plus A times the rhythm periodogram; A per epoch, or per
    epoch and taper ('own')."""
    if D["ds"]:
        return [u["P"] + (A[:, None] if A.ndim == 1 else A)[:, :, None] * d["Pr"]
                for u, A, d in zip(D["units"], A_list, draws)]
    return [u["P"] + A[:, None] * d["Pr"] for u, A, d in zip(D["units"], A_list, draws)]


def run_condition(cond, args):
    t0 = time.time()
    ci = CONDS.index(cond)
    D, chk = prepare(cond, args)
    win, fsel, norm = rhythm_setup(D)
    twin_model(D)
    units = D["units"]
    h0 = chk["h_alpha_median"]
    log = lambda s: print(f"[{cond} {time.time() - t0:6.0f}s] {s}", flush=True)
    log(f"{chk['n_units']} participants, {chk['n_epochs']} epochs; alpha lambda "
        f"{chk['alpha_lam']:.4f} (published {chk['alpha_lam_published']:.4f}); "
        f"median alpha height {h0:.2f}")

    # no injection: the real residual in the band, and what the estimator makes of it
    T, B, Bz, grp, inst, reps = analyse(D, [u["P"] for u in units])
    lam, nr = estimate(T, B, Bz, grp)
    mo = moments(T, B, Bz, grp, stacked(D, [np.log(u["t_alpha"]) for u in units], reps))
    # how each reference moves with the instrumented background: to first order
    # lambda-hat / lambda for a rhythm that follows it
    b_in, b_pool = (moments(T, B, Bz, grp, stacked(D, [np.log(u[k]) for u in units], reps))["beta_vb"]
                    for k in ("ref_in", "ref_pool"))
    b_own = np.nan
    if D["ds"]:
        # the rhythm in the T taper follows that taper's own fit (taper order as in stack())
        own = np.concatenate([np.concatenate([np.log(u["ref_own"][:, a]) for a, _, _ in PERMS])
                              for u in units])
        b_own = moments(T, B, Bz, grp, own)["beta_vb"]
    inv_ep = np.concatenate([np.full(u["P"].shape[0], k) for k, u in enumerate(units)])
    sd_w = lambda x: float(np.sqrt(np.mean(within(np.concatenate(x), inv_ep) ** 2)))
    chk.update(inject_instrument_r=inst, inject_control_lam=lam, inject_control_roots=nr,
               inject_rel_resid=mo["rel_excess"], inject_sd_lnb=mo["sd_lnb"],
               sd_ln_ref_in=sd_w([np.log(u["ref_in"]) for u in units]),
               sd_ln_ref_pool=sd_w([np.log(u["ref_pool"]) for u in units]),
               sd_ln_alpha=sd_w([np.log(u["t_alpha"]) for u in units]),
               beta_ref_in_on_b=b_in, beta_ref_pool_on_b=b_pool, beta_ref_own_on_b=b_own,
               beta_alpha_on_b=mo["beta_vb"], rhythm_norm=";".join(f"{x:.4g}" for x in norm))
    chk["corr_alpha_b"] = chk["beta_alpha_on_b"] * chk["inject_sd_lnb"] / chk["sd_ln_alpha"]
    log(f"control (no injection): lambda {lam:+.3f} ({nr} roots), relative residual "
        f"{mo['rel_excess']:+.3f}, instrument r {inst:.2f}")

    runs, drive = [], []
    brng = np.random.default_rng([args.seed, ci, 999])
    for rep in range(args.reps):
        rng = np.random.default_rng([args.seed, ci, rep])
        dr = []
        for u in units:
            n = u["P"].shape[0]
            Pr = rhythm(n, win, fsel, rng, norm)
            dr.append(dict(Pr=Pr if D["ds"] else Pr[:, 0], eps=rng.normal(0, EPS_SD, n),
                           xi=rng.standard_normal(n), zh=rng.standard_normal()))
        tw = twin_draws(D, np.random.default_rng([args.seed, ci, rep, 1]))
        base = {}
        for ref in (("inband", "own", "pooled", "synthetic") if D["ds"] else ("inband", "pooled", "synthetic")):
            for hm in H_MULT:
                for lam_true in LAMS:
                    hs = [hm * h0 * 10 ** (H_SD * d["zh"]) for d in dr]
                    if ref == "synthetic":
                        A = [amplitude(t["ref"], lam_true, h, d["eps"]) for t, h, d in zip(tw, hs, dr)]
                        spectra, bref = twin_spectra(D, tw, A), [t["ref"] for t in tw]
                    elif ref == "own":
                        A = [amplitude(u["ref_own"], lam_true, h, d["eps"][:, None])
                             for u, h, d in zip(units, hs, dr)]
                        spectra, bref = inject(D, A, dr), [u["ref_in"][:, None] for u in units]
                    else:
                        key = "ref_in" if ref == "inband" else "ref_pool"
                        A = [amplitude(u[key], lam_true, h, d["eps"]) for u, h, d in zip(units, hs, dr)]
                        spectra, bref = inject(D, A, dr), [u["ref_in"] for u in units]
                    T, B, Bz, grp, inst, reps = analyse(D, spectra)
                    lam, nr = estimate(T, B, Bz, grp)
                    row = dict(cond=cond, ref=ref, h_mult=round(hm, 3), h=hm * h0, lam_true=lam_true,
                               rep=rep, lam_hat=lam, n_roots=nr, instrument_r=inst,
                               rel_height=float(np.median(np.concatenate([(a / r).ravel()
                                                                           for a, r in zip(A, bref)]))),
                               **moments(T, B, Bz, grp))
                    if rep == 0 and ref == "inband" and hm == 1.0 and args.nboot:
                        bs = bootstrap(T, B, Bz, grp, lam, args.nboot, brng)
                        ok = np.isfinite(bs)
                        row.update(boot_sd=float(np.std(bs[ok])) if ok.sum() > 2 else np.nan,
                                   boot_lo=float(np.percentile(bs[ok], 2.5)) if ok.sum() > 2 else np.nan,
                                   boot_hi=float(np.percentile(bs[ok], 97.5)) if ok.sum() > 2 else np.nan,
                                   boot_fail=float(1 - ok.mean()))
                    runs.append(row)
                    if ref == "inband" and hm == 1.0:
                        base[lam_true] = lam
                    log(f"rep {rep} {ref:9s} h x{hm:.2f} lambda {lam_true:.1f} -> {lam:+.3f} "
                        f"({nr} roots)")
        # common drive, in-band reference, middle height
        hs = [h0 * 10 ** (H_SD * d["zh"]) for d in dr]
        drives = [("synthetic", rho) for rho in RHOS] + [("alpha", np.nan)]
        for kind, rho in drives:
            if kind == "synthetic":
                vs = [standardise(rho * standardise(np.log(u["ref_in"])) + np.sqrt(1 - rho ** 2) * d["xi"])
                      for u, d in zip(units, dr)]
            else:
                vs = [standardise(np.log(u["t_alpha"])) for u in units]
            for lam_true in (0.0, 1.0):
                for kappa in KAPPAS:
                    A = [amplitude(u["ref_in"], lam_true, h, d["eps"], v, kappa)
                         for u, h, d, v in zip(units, hs, dr, vs)]
                    T, B, Bz, grp, inst, reps = analyse(D, inject(D, A, dr))
                    lam, nr = estimate(T, B, Bz, grp)
                    mo = moments(T, B, Bz, grp, stacked(D, vs, reps))
                    cvr = np.mean([np.corrcoef(v, np.log(u["ref_in"]))[0, 1] for u, v in zip(units, vs)])
                    drive.append(dict(cond=cond, drive=kind, rho=rho, kappa=kappa, lam_true=lam_true,
                                      rep=rep, lam_hat=lam, n_roots=nr, lam_hat_kappa0=base[lam_true],
                                      shift=lam - base[lam_true], beta_vb=mo["beta_vb"],
                                      analytic_shift=kappa * mo["beta_vb"],
                                      corr_v_lnb=mo["beta_vb"] * mo["sd_lnb"], corr_v_lnref=cvr,
                                      rel_excess=mo["rel_excess"]))
                    log(f"rep {rep} drive {kind} rho {rho} kappa {kappa} lambda {lam_true:.0f} -> "
                        f"{lam:+.3f} (shift {lam - base[lam_true]:+.3f}, analytic "
                        f"{kappa * mo['beta_vb']:+.3f})")
    log("done")
    return runs, drive, [dict(cond=cond, **chk)]


def summarise(runs):
    R = pd.DataFrame(runs)
    g = R.groupby(["cond", "ref", "h_mult", "lam_true"])
    S = g.agg(h=("h", "first"), lam_mean=("lam_hat", "mean"), lam_sd=("lam_hat", "std"),
              n_ok=("lam_hat", lambda x: int(np.isfinite(x).sum())), n_reps=("lam_hat", "size"),
              max_roots=("n_roots", "max"), rel_height=("rel_height", "median"),
              instrument_r=("instrument_r", "mean"), sd_lnb=("sd_lnb", "mean"),
              rel_excess=("rel_excess", "mean")).reset_index()
    if "boot_sd" in R:
        b = R.dropna(subset=["boot_sd"])[["cond", "ref", "h_mult", "lam_true", "boot_sd", "boot_lo", "boot_hi"]]
        S = S.merge(b, on=["cond", "ref", "h_mult", "lam_true"], how="left")
    curve = []
    for k, d in S.sort_values("lam_true").groupby(["cond", "ref", "h_mult"]):
        y = d.lam_mean.to_numpy()
        ok = np.all(np.isfinite(y))
        curve.append(dict(zip(["cond", "ref", "h_mult"], k), monotone=bool(ok and np.all(np.diff(y) > 0)),
                          slope=np.polyfit(d.lam_true, y, 1)[0] if ok else np.nan))
    return S.merge(pd.DataFrame(curve), on=["cond", "ref", "h_mult"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds003690", default=os.environ.get("DS003690_OUT", os.path.join(DATA, "ds003690_epochs")))
    ap.add_argument("--dortmund", default=os.environ.get("DORTMUND_OUT", os.path.join(DATA, "dortmund_psd")))
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--nboot", type=int, default=50)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=RES, help="output directory")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--background", choices=("flanks", "power-law"), default="flanks",
                    help="flanks: power law through 23-27 and 39-45 Hz, rescaled to the mean "
                         "spectrum; power-law: one fit over 2-45 Hz without 6-16 and 27-39 Hz")
    a = ap.parse_args()
    conds = a.conds.split(",")
    runs, drive, checks = [], [], []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(conds))) as ex:
        futs = {ex.submit(run_condition, c, a): c for c in conds}
        for fu in as_completed(futs):
            r, d, c = fu.result()
            runs += r; drive += d; checks += c
    order = {c: k for k, c in enumerate(CONDS)}
    key = lambda df: df.assign(_o=df.cond.map(order)).sort_values(["_o"], kind="stable").drop(columns="_o")
    name = lambda s: os.path.join(a.out, f"inject_calibration_{s}{a.tag}.csv")
    R = key(pd.DataFrame(runs))
    R.to_csv(name("runs"), index=False)
    key(summarise(runs)).to_csv(name("summary"), index=False)
    key(pd.DataFrame(drive)).to_csv(name("drive"), index=False)
    key(pd.DataFrame(checks)).to_csv(name("checks"), index=False)
    print("written", name("*"), flush=True)


if __name__ == "__main__":
    main()
