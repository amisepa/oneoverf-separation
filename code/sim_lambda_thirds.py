"""Log-free coupling exponent from halves and from thirds of the segments,
when the background is not the same in every segment of a recording.

lambda_gmm takes the band total t, the background b and the instrument from
separate sets of Welch segments. With two sets (odd and even segments) the
instrument has to share a set with t or with b. If the true background
differs between the sets, the estimate then reads high (instrument from the
set of t) or low (from the set of b). With three sets (segment index mod 3)
the three come from three different sets (instrument="third").

Simulated participants have two conditions with a background change that
varies across participants, a rhythm a = c G(f) L(f)^lambda with a common
intrinsic change plus a participant-specific part, and segment counts like
those kept in HBN (median 11 eyes open, 28 eyes closed). Each 4 s segment is
synthesised in the time domain from its own spectrum:

    background  L(f) exp(g),              g ~ N(0, sigma_g) per segment
    rhythm      a(f) exp(kappa g + v),    v ~ N(0, sigma_v) per segment

g is the non-stationarity of the background. kappa lets the rhythm follow
it within a recording; it is a nuisance of the recording, set independently
of lambda, which is defined by the condition-level backgrounds. The
scenarios are

    stationary   sigma_g = 0
    hbn          sigma_g = 0.36, kappa = 0.4: close to HBN (power law, 6-16 Hz
                 censor) in the correlation between the half-differences of
                 ln t and of fitted ln b (0.55 eyes open and 0.39 eyes
                 closed here, 0.50 and 0.45 in HBN), in the s.d. of the
                 half-difference of fitted ln b (0.25 and 0.16; 0.26 and
                 0.16) and in the split-half reliability of the instrument
                 (0.82; 0.81)
    background   sigma_g = 0.36, kappa = 0 (only the background moves)
    common       sigma_g = 0.36, kappa = 1 (rhythm and background move as one)

The spectra are averaged over two channels per segment and fitted as in
hbn_kp.py (power law by Whittle deviance, 2-55 Hz without 6-16 Hz; band =
IAF +/- 2 Hz, IAF from the eyes-closed spectrum).

Writes results/sim_lambda_thirds.csv (one row per replicate) and prints the
mean and s.d. over replicates of the three estimators, with the calibration
statistics of each scenario.

Usage: python sim_lambda_thirds.py [--n 2400] [--reps 6] [--workers 4]
                                   [--scenarios stationary,hbn,background,common]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lambda_gmm as G
from ap_models import fit_aperiodic, ap_band_power
from hbn_kp import FIT_RANGE, find_iaf, mask_for

SRATE = 250.0
NPER = 1000                       # 4 s
N_CH = 2                          # channels averaged per segment
GRID = np.linspace(-1, 3, 161)
SIGMA_V = 0.5                     # s.d. of the rhythm's own fluctuation (ln)
SCENARIOS = {"stationary": (0.0, 0.0), "hbn": (0.36, 0.4),
             "background": (0.36, 0.0), "common": (0.36, 1.0)}   # sigma_g, kappa


def segments(L, A, g, v, kappa, rng):
    """Welch spectra (n_seg, n_freq) of segments with their own spectra.

    L, A: background and rhythm on the frequency grid of an 8 s stretch;
    each segment is the first 4 s of an 8 s Gaussian signal with spectrum
    L exp(g) + A exp(kappa g + v), Hann-windowed, averaged over N_CH draws.
    """
    n = 2 * NPER
    S = L[None, :] * np.exp(g)[:, None] + A[None, :] * np.exp(kappa * g + v)[:, None]
    S[:, 0] = 0.0
    S = np.repeat(S, N_CH, axis=0)
    sigma = np.sqrt(S * SRATE * n / 2.0)
    X = sigma * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape)) / np.sqrt(2)
    X[:, -1] = sigma[:, -1] * rng.standard_normal(S.shape[0])
    x = np.fft.irfft(X, n, axis=1)[:, :NPER]
    win = np.hanning(NPER + 1)[:NPER]
    x = (x - x.mean(1, keepdims=True)) * win
    P = 2.0 * np.abs(np.fft.rfft(x, axis=1)) ** 2 / (SRATE * np.sum(win ** 2))
    return P.reshape(g.size, N_CH, -1).mean(1)


def simulate(lam_true, n, sigma_g, kappa, rng):
    """Band totals and fitted backgrounds per participant.

    Returns dict split -> (t, b), arrays (n, 2 conditions, n_sets), for
    split "halves" (2 sets) and "thirds" (3 sets).
    """
    fg = np.fft.rfftfreq(2 * NPER, 1 / SRATE)
    ff = np.fft.rfftfreq(NPER, 1 / SRATE)
    sel = (ff >= FIT_RANGE[0]) & (ff <= FIT_RANGE[1])
    f = ff[sel]
    keep = mask_for(f, "censor", [(6, 16)])
    out = {"halves": [np.full((n, 2, 2), np.nan) for _ in range(2)],
           "thirds": [np.full((n, 2, 3), np.nan) for _ in range(2)]}
    Lref = 10.0 ** 1.0 / 10.0 ** 1.5         # median eyes-open background at 10 Hz
    for i in range(n):
        off_eo = 1.0 + 0.33 * rng.standard_normal()
        expo_eo = 1.5 + 0.25 * rng.standard_normal()
        off_ec = off_eo + 0.20 + 0.17 * rng.standard_normal()
        expo_ec = expo_eo + 0.10 + 0.10 * rng.standard_normal()
        cf = float(np.clip(9.5 + 0.9 * rng.standard_normal(), 7.5, 12.0))
        # rhythm strength scaled by one reference level, not by the
        # participant's own background
        c_eo = 1.4 * Lref ** (1 - lam_true) * 10 ** (0.25 * rng.standard_normal())
        c_ec = c_eo * 5.0 * np.exp(0.5 * rng.standard_normal())
        z1, z2 = rng.standard_normal(2)
        n_ec = int(np.clip(round(28 * np.exp(0.48 * z1)), 6, 84))
        n_eo = int(np.clip(round(11 * np.exp(0.42 * (0.7 * z1 + 0.714 * z2))), 4, 34))
        gauss = np.exp(-0.5 * ((fg - cf) / 1.5) ** 2)
        spec = []
        for off, ex, c, ns in ((off_eo, expo_eo, c_eo, n_eo), (off_ec, expo_ec, c_ec, n_ec)):
            L = 10.0 ** off / np.maximum(fg, 1e-9) ** ex
            g = sigma_g * rng.standard_normal(ns) - 0.5 * sigma_g ** 2
            v = SIGMA_V * rng.standard_normal(ns) - 0.5 * SIGMA_V ** 2
            spec.append(segments(L, c * gauss * L ** lam_true, g, v, kappa, rng)[:, sel])
        iaf = find_iaf(spec[1].mean(0), f)
        if not np.isfinite(iaf):
            continue
        band = (f >= iaf - 2) & (f <= iaf + 2)
        for k, P in enumerate(spec):
            for name, nset in (("halves", 2), ("thirds", 3)):
                for s in range(nset):
                    p = np.clip(P[s::nset].mean(0), 1e-12, None)
                    fit = fit_aperiodic(p, f, keep, "fixed")
                    if fit is None:
                        continue
                    out[name][0][i, k, s] = np.mean(p[band])
                    out[name][1][i, k, s] = ap_band_power(fit, f, iaf - 2, iaf + 2)
    return out


def one(job):
    scen, lam_true, rep, n = job
    sigma_g, kappa = SCENARIOS[scen]
    seed = [list(SCENARIOS).index(scen), int(round(100 * lam_true)), rep]
    S = simulate(lam_true, n, sigma_g, kappa, np.random.default_rng(seed))
    t, b = S["halves"]
    ok = np.all(np.isfinite(t), (1, 2)) & np.all(b > 0, (1, 2))
    lt, lb = np.log(t[ok]), np.log(b[ok])
    z = lb[:, 1] - lb[:, 0]
    r = np.corrcoef
    row = dict(scenario=scen, lambda_true=lam_true, rep=rep, n=n,
               r_half_eo=r(lt[:, 0, 0] - lt[:, 0, 1], lb[:, 0, 0] - lb[:, 0, 1])[0, 1],
               r_half_ec=r(lt[:, 1, 0] - lt[:, 1, 1], lb[:, 1, 0] - lb[:, 1, 1])[0, 1],
               sd_dlnb_eo=np.std(lb[:, 0, 0] - lb[:, 0, 1]),
               sd_dlnb_ec=np.std(lb[:, 1, 0] - lb[:, 1, 1]),
               sd_dlnt_eo=np.std(lt[:, 0, 0] - lt[:, 0, 1]),
               sd_dlnt_ec=np.std(lt[:, 1, 0] - lt[:, 1, 1]),
               sd_z=np.std(z[:, 0]), rel_z_halves=r(z[:, 0], z[:, 1])[0, 1])
    for inst in ("total", "background"):
        row[f"halves_{inst}"] = G.estimate(t[:, 0], t[:, 1], b[:, 0], b[:, 1], grid=GRID,
                                           instrument=inst)["lam"]
    t, b = S["thirds"]
    ok = np.all(b > 0, (1, 2))
    z = np.log(b[ok, 1]) - np.log(b[ok, 0])
    row["rel_z_thirds"] = np.mean([r(z[:, j], z[:, k])[0, 1] for j, k in ((0, 1), (0, 2), (1, 2))])
    row["thirds"] = G.estimate(t[:, 0], t[:, 1], b[:, 0], b[:, 1], grid=GRID,
                               instrument="third")["lam"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--grid", default="0,0.5,1")
    ap.add_argument("--scenarios", default="stationary,hbn,background,common")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                  "sim_lambda_thirds.csv"))
    a = ap.parse_args()
    jobs = [(s, float(g), rep, a.n) for s in a.scenarios.split(",")
            for g in a.grid.split(",") for rep in range(a.reps)]
    rows = []
    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(one, jobs):
            rows.append(row)
            print(f"{row['scenario']:11s} lambda {row['lambda_true']:.1f} rep {row['rep']}  "
                  f"halves, instrument with t {row['halves_total']:+.3f}  with b "
                  f"{row['halves_background']:+.3f}  thirds {row['thirds']:+.3f}", flush=True)
    D = pd.DataFrame(rows).sort_values(["scenario", "lambda_true", "rep"])
    D.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    est = ["halves_total", "halves_background", "thirds"]
    g = D.groupby(["scenario", "lambda_true"], sort=False)
    print(f"\nestimates, mean (s.d.) over {a.reps} replicates of n = {a.n}:")
    M, S = g[est].mean(), g[est].std()
    print(pd.DataFrame({c: [f"{m:+.3f} ({s:.3f})" for m, s in zip(M[c], S[c])] for c in est},
                       index=M.index).to_string())
    print("\ncalibration statistics (mean over replicates):")
    print(g[["r_half_eo", "r_half_ec", "sd_dlnb_eo", "sd_dlnb_ec", "sd_dlnt_eo", "sd_dlnt_ec",
             "sd_z", "rel_z_halves", "rel_z_thirds"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
