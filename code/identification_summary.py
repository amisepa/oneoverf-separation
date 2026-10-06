"""Collect every attempt to estimate lambda into one table.

- HBN eyes closed vs eyes open, log-free two-condition estimator
  (lambda_gmm.bootstrap, root-tracking) for the four specifications of
  hbn_kp.py; computed here from results/hbn_kp_fits.csv (local) and saved to
  results/hbn_lambda_gmm.csv.
- ds003690 within-session eyes-open fluctuations (ds003690_lambda.py), with
  its data-matched calibration.
- Dortmund within-session fluctuations, eyes closed and open, before and
  after the task battery (dortmund_levels.py), with calibration.
  For these two, the background is the power law through the flanks 2-6 and
  26-32 Hz; lam_min and lam_max give the range of the estimate without
  covariates over that and four other choices of flanks
  (results/within_recording_variants.csv).
- Chennu graded propofol, baseline vs moderate (chennu_lambda_gmm.csv).
- Test-retest: SRM (later session) and Dortmund (5 years), log-free
  estimator with sessions as conditions (srm_lambda.py).
- Intracranial within-session fluctuations (ieeg_lambda.py), with its
  data-matched calibration; n_patients is the number of patients behind the
  channels of each row.

When hbn_half_assignment.py has been run, the HBN rows carry both half
assignments of the instrument: lam, lo, hi with the instrument from the half
of the band total (the default of lambda_gmm), lam_alt, lo_alt, hi_alt with
the instrument from the half of the background, both with roots searched on
-1 to 3. The Chennu rows then take their intervals from the same search
range (results/twocond_lambda_gmm_halves.csv, which also holds the second
assignment for Chennu and for the test-retest rows).

Writes results/identification_summary.csv.

Usage: python identification_summary.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
import lambda_gmm as G


def hbn_gmm():
    path = os.path.join(RES, "hbn_lambda_gmm.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    k = k[k.subject.isin(Q.index[Q.qc_ok]) & (k.band == "alpha") & (k.split != "full")]
    rows = []
    for model in ("fixed", "knee_plateau"):
        for win in ("censor 6-16", "flanks 3-6, 26-36"):
            g = k[(k.model == model) & (k.window == win)]
            w = g.pivot_table(index="subject", columns=["cond", "split"], values=["tot", "b"]).dropna()
            arr = [w[v][c][["odd", "even"]].to_numpy() for v, c in
                   (("tot", "eo"), ("tot", "ec"), ("b", "eo"), ("b", "ec"))]
            r = G.bootstrap(*arr, nboot=300, rng=np.random.default_rng(0))
            rows.append(dict(model=model, window=win, n=len(w), lam=r["lam"], lo=r["ci"][0],
                             hi=r["ci"][1], fail=r["boot_fail"]))
    D = pd.DataFrame(rows)
    D.to_csv(path, index=False)
    return D


def both_assignments(name, keys):
    """Estimates under the two half assignments, indexed by keys; empty if
    hbn_half_assignment.py has not written the file."""
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return {}
    D = pd.read_csv(path)
    w = D.pivot(index=keys, columns="instrument", values=["lam", "lo", "hi"])
    return {k: dict(lam=r["lam", "total"], lo=r["lo", "total"], hi=r["hi", "total"],
                    lam_alt=r["lam", "background"], lo_alt=r["lo", "background"],
                    hi_alt=r["hi", "background"]) for k, r in w.iterrows()}


def flank_range():
    """Smallest and largest estimate over the choices of flanks, per dataset
    and recording; empty if the within-recording scripts have not been run."""
    path = os.path.join(RES, "within_recording_variants.csv")
    if not os.path.exists(path):
        return {}
    g = pd.read_csv(path).groupby(["dataset", "recording"]).lam
    return {k: dict(lam_min=lo, lam_max=hi) for (k, lo), hi in zip(g.min().items(), g.max())}


def calibration(pattern, query):
    """'true -> estimate' from the data-matched simulations that exist."""
    cal = []
    for L in ("0.0", "0.5", "1.0"):
        q = os.path.join(RES, pattern.format(L))
        if os.path.exists(q):
            c = pd.read_csv(q).query(query)
            if len(c):
                cal.append(f"{float(L):g} -> {c.lam.iloc[0]:.2f}")
    return ", ".join(cal)


def main():
    out = []
    H = hbn_gmm()
    B = both_assignments("hbn_lambda_gmm_halves.csv", ["model", "window"])
    T = both_assignments("twocond_lambda_gmm_halves.csv", ["dataset", "spec", "model"])
    for r in H.itertuples():
        out.append(dict(design="between conditions: eyes closed vs open", dataset="HBN, 5-22 y",
                        spec=f"{'power law' if r.model == 'fixed' else 'knee + plateau'}, {r.window}",
                        n=r.n, lam=r.lam, lo=r.lo, hi=r.hi, calibration=""))
        out[-1].update(B.get((r.model, r.window), {}))
    V = flank_range()
    d = pd.read_csv(os.path.join(RES, "ds003690_lambda.csv"))
    for r in d[(d.covariates.isin(["none", "pupil"])) & (d.group == "all")].itertuples():
        out.append(dict(design="within session: epoch fluctuations, eyes open",
                        dataset="ds003690, 20-75 y",
                        spec=f"flanks 2-6, 26-32, covariates: {r.covariates}",
                        n=r.n_units, lam=r.lam, lo=r.lo, hi=r.hi,
                        calibration=calibration("ds003690_lambda_sim{}.csv", "group == 'all'")))
        if r.covariates == "none":
            out[-1].update(V.get(("ds003690", "eyes open"), {}))
    p = os.path.join(RES, "dortmund_levels.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).query("covariates == False").itertuples():
            state = {"ec": "eyes closed", "eo": "eyes open"}[r.cond[:2]]
            when = "before" if r.cond.endswith("pre") else "after"
            out.append(dict(design=f"within session: segment fluctuations, {state}",
                            dataset="Dortmund, 20-70 y",
                            spec=f"flanks 2-6, 26-32, {when} 2-h tasks",
                            n=r.n_units, lam=r.lam, lo=r.lo, hi=r.hi,
                            calibration=calibration("dortmund_levels_sim{}.csv",
                                                    f"cond == '{r.cond}'")))
            out[-1].update(V.get(("Dortmund", r.cond), {}))
    x = pd.read_csv(os.path.join(RES, "chennu_lambda_gmm.csv"))
    for r in x[x.contrast == "baseline -> moderate"].itertuples():
        out.append(dict(design="between doses: propofol baseline vs moderate",
                        dataset="Chennu 2016", spec=f"{r.roi}, {r.window}", n=r.n, lam=r.lam,
                        lo=r.lo, hi=r.hi, calibration=""))
        t = T.get(("Chennu 2016", f"{r.roi}, {r.window}, {r.contrast}", "fixed"), {})
        out[-1].update({k: t[k] for k in ("lam", "lo", "hi") if k in t})
    p = os.path.join(RES, "srm_lambda.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        s = s[s.analysis.str.contains("log-free") & (s.model == "fixed")]
        for r in s.itertuples():
            srm = r.analysis.startswith("srm")
            out.append(dict(design="between sessions: test-retest",
                            dataset="SRM, 17-71 y" if srm else "Dortmund, 20-70 y",
                            spec=("SRM, eyes closed, later session" if srm else
                                  f"Dortmund, {'eyes closed' if 'ec_pre' in r.analysis else 'eyes open'}"
                                  ", 5 years"),
                            n=r.n, lam=r.lam_gmm, lo=r.gmm_lo, hi=r.gmm_hi, calibration=""))
    p = os.path.join(RES, "ieeg_lambda.csv")
    if os.path.exists(p):
        cal = []
        for L in ("0.0", "0.5", "1.0"):
            q = os.path.join(RES, f"ieeg_lambda_sim{L}.csv")
            if os.path.exists(q):
                c = pd.read_csv(q)
                c = c[c.subset == "all"]
                cal.append(f"{float(L):g} -> {c.lam.iloc[0]:.2f}")
        for r in pd.read_csv(p).query("covariates in ['none', 'EOG + EMG']").itertuples():
            out.append(dict(design="within session: epoch fluctuations, intracranial",
                            dataset="ds003688 iEEG", n=r.n_channels, lam=r.lam, lo=r.lo, hi=r.hi,
                            spec=f"{r.subset} channels, covariates: {r.covariates}",
                            calibration=", ".join(cal), n_patients=r.n_patients))
    cols = ["design", "dataset", "spec", "n", "lam", "lo", "hi", "calibration",
            "lam_alt", "lo_alt", "hi_alt", "n_patients", "lam_min", "lam_max"]
    O = pd.DataFrame(out).reindex(columns=cols)
    O["n_patients"] = O.n_patients.astype("Int64")
    O.to_csv(os.path.join(RES, "identification_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
