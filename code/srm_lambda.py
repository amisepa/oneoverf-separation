"""SRM resting-state EEG (ds003775): the adult age slope of alpha under each
separation rule, and the coupling exponent from test-retest change.

Posterior ROI (O1, Oz, O2, PO3, POz, PO4), eyes closed; background fitted over
2-45 Hz with 6-16 Hz censored (power law or knee + plateau); alpha band =
individual alpha frequency +/- 2 Hz; a = total - b. Quality control as for
the other datasets (fewer than 10 clean segments, alpha power more than 300
times from the median, or no alpha peak).

  age        session 1 age slope of ln a - lambda ln b, adjusted for sex
             (lambda_curve.effect_curve)
  retest     session 2 minus session 1 within person: the lambda-curve of
             the change, and the log-free estimate of lambda
             (lambda_gmm.bootstrap, sessions as conditions, odd/even halves)

The same log-free test-retest estimate is computed for the Dortmund study
(session 1 vs session 2, about five years apart, before the task battery)
from results/aging_bandpower.csv. Between sessions, electrode gain changes
(cap position, impedance); in a synaptic model a pure gain change has
lambda = 1 (sim_mechanisms.py), so test-retest variation is expected to pull
the estimate towards 1.

Writes results/srm_bandpower.csv (per recording; local) and
results/srm_lambda.csv; --tag TAG is appended to these names and to the
aging_bandpower.csv it reads (_iclabel for the cleaned spectra).

Usage: python srm_lambda.py [--srm DIR] [--workers 6] [--tag TAG]
"""
import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lambda_gmm
from lambda_curve import band_power, effect_curve, peak_frequency

RES = os.path.join(HERE, "..", "results")
ROI = ["O1", "Oz", "O2", "PO3", "POz", "PO4"]
FIT = (2.0, 45.0)
CENSOR = ((6.0, 16.0),)
warnings.filterwarnings("ignore")


def one(path):
    d = np.load(path, allow_pickle=True)
    f = d["freqs"].astype(float)
    ch = [str(c) for c in d["ch_names"]]
    ix = [ch.index(c) for c in ROI if c in ch]
    spec = {s: d[f"ec_{s}"][ix].mean(0).astype(float) for s in ("full", "odd", "even")}
    iaf = peak_frequency(spec["full"], f)
    edge = (not np.isfinite(iaf)) or iaf <= 6.25 or iaf >= 13.75
    iaf_use = 10.0 if not np.isfinite(iaf) else iaf
    out = []
    for split, P in spec.items():
        for model in ("fixed", "knee_plateau"):
            bp = band_power(P, f, (iaf_use - 2, iaf_use + 2), FIT, CENSOR, model)
            if bp is None:
                continue
            out.append(dict(subject=str(d["subject"]), session=str(d["session"]),
                            age=float(d["age"]), sex=str(d["sex"]), split=split, model=model,
                            iaf=iaf, iaf_edge=edge, n_seg=int(d["ec_n_seg"]),
                            few=bool(d["ec_flag_few"]), **bp))
    return out


def qc(T):
    F = T[(T.split == "full") & (T.model == "fixed")]
    med = F.groupby("session").tot.transform("median")
    bad = (F.tot < med / 300) | (F.tot > med * 300) | F.few | F.iaf_edge
    return T.merge(F.assign(bad=bad)[["subject", "session", "bad"]],
                   on=["subject", "session"], how="left")


def curve_row(name, model, r, extra=None):
    c = r["curve"]
    row = dict(analysis=name, model=model, n=r["n"], s_a=r["s_a"], s_b=r["s_b"],
               s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
               lam_star_bounded=r["lam_star_bounded"], lam_star=r["lam_star"], hdi_lo=r["hdi"][0], hdi_hi=r["hdi"][1],
               lam0=c[0][0], lam0_lo=c[0][1], lam0_hi=c[0][2],
               lam1=c[-1][0], lam1_lo=c[-1][1], lam1_hi=c[-1][2])
    row.update(extra or {})
    return row


def retest_gmm(T, key, s1, s2, name, model, nboot=300):
    """Log-free lambda with sessions s1, s2 as the two conditions."""
    D = T[(T.model == model) & (T.split != "full")]
    w = D.pivot_table(index=key, columns=["session", "split"], values=["tot", "b"]).dropna()
    arr = [w[v][s][["odd", "even"]].to_numpy() for v, s in
           (("tot", s1), ("tot", s2), ("b", s1), ("b", s2))]
    r = lambda_gmm.bootstrap(*arr, nboot=nboot, rng=np.random.default_rng(0),
                             grid=np.linspace(-1, 3, 161))
    z = [np.log(arr[3][:, h]) - np.log(arr[2][:, h]) for h in (0, 1)]
    return dict(analysis=name, model=model, n=len(w), lam_gmm=r["lam"], gmm_lo=r["ci"][0],
                gmm_hi=r["ci"][1], gmm_fail=r["boot_fail"], roots=str(r["roots"]),
                sd_dlnb=float(np.std(z[0])), instrument_r=float(np.corrcoef(*z)[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srm", default=os.environ.get("SRM_OUT", "srm_psd"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        T = pd.DataFrame([r for rs in pool.map(one, sorted(glob.glob(os.path.join(a.srm, "*.npz"))))
                          for r in rs])
    T = qc(T)
    T.to_csv(os.path.join(RES, f"srm_bandpower{a.tag}.csv"), index=False)
    print(f"SRM: {T.subject.nunique()} participants, "
          f"{T[(T.split == 'full') & (T.model == 'fixed')].groupby('session').size().to_dict()} "
          f"recordings; excluded {int(T[(T.split == 'full') & (T.model == 'fixed')].bad.sum())}")
    rows = []
    for model in ("fixed", "knee_plateau"):
        F = T[(T.split == "full") & (T.model == model) & ~T.bad & (T.a > 0)]
        D = F[F.session == "t1"]
        sex = (D.sex.str.lower() == "m").astype(float).to_numpy()
        r = effect_curve(np.log(D.a), np.log(D.b), x=D.age.to_numpy(), covariates=sex,
                         rng=np.random.default_rng(0))
        rows.append(curve_row("srm age slope, ses-t1 (+ sex)", model, r, dict(per="year")))
        w = F.pivot_table(index="subject", columns="session", values=["a", "b"]).dropna()
        r = effect_curve(np.log(w[("a", "t2")]) - np.log(w[("a", "t1")]),
                         np.log(w[("b", "t2")]) - np.log(w[("b", "t1")]),
                         rng=np.random.default_rng(0))
        rows.append(curve_row("srm retest change (t2 - t1)", model, r, dict(per="session")))
        ok = T[~T.bad.fillna(True)]
        rows.append(retest_gmm(ok, "subject", "t1", "t2", "srm retest, log-free lambda", model))
    # Dortmund, session 1 vs 2 (about five years), eyes closed and open before the tasks
    ag = pd.read_csv(os.path.join(RES, f"aging_bandpower{a.tag}.csv"))
    ag = ag[(ag.study == "dortmund") & ~ag.bad.fillna(True).astype(bool)]
    for cond in ("ec_pre", "eo_pre"):
        for model in ("fixed", "knee_plateau"):
            rows.append(retest_gmm(ag[ag.cond == cond], "subject", 1, 2,
                                   f"dortmund retest {cond}, log-free lambda", model))
    O = pd.DataFrame(rows)
    O.to_csv(os.path.join(RES, f"srm_lambda{a.tag}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
