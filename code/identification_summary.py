"""Collect every attempt to estimate lambda into one table.

- HBN eyes closed vs eyes open, log-free two-condition estimator with the
  band total, the background and the instrument from three interleaved
  thirds of each recording, for the four specifications of hbn_kp.py
  (hbn_thirds.py, results/hbn_lambda_gmm_thirds.csv): the selected root, its
  interval, and every root of the moment condition (n_roots, roots).
- Dortmund within-session fluctuations, eyes closed and open, before and
  after the task battery (dortmund_levels.py), with calibration. The
  background is the power law through the flanks 2-6 and 26-32 Hz; lam_min
  and lam_max give the range of the estimate without covariates over that and
  four other choices of flanks (results/within_recording_variants.csv).
- Chennu graded propofol, baseline vs moderate (chennu_lambda_gmm.csv).
- Test-retest: SRM (later session) and Dortmund (5 years), log-free
  estimator with sessions as conditions (srm_lambda.py).
- Intracranial within-session fluctuations (ieeg_lambda.py), with its
  data-matched calibration; n_patients is the number of patients behind the
  channels of each row. The background is the same flank fit, and the row
  of all channels without covariates carries the range over the five
  choices of flanks.

When hbn_half_assignment.py has been run, the Chennu rows take their
intervals from roots searched on -1 to 3 (results/twocond_lambda_gmm_halves.csv).
The Dortmund and SRM rows are repeated after artefact cleaning (cleaning =
"ICA"; the *_iclabel.csv outputs of dortmund_levels.py and srm_lambda.py run
with --tag _iclabel); every other row has cleaning = "none".

Writes results/identification_summary.csv (the rows of Supplementary Table 6).

Usage: python identification_summary.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CLEANING = (("", "none"), ("_iclabel", "ICA"))


def hbn_thirds():
    """HBN estimates from three interleaved thirds (written by hbn_thirds.py)."""
    T = pd.read_csv(os.path.join(RES, "hbn_lambda_gmm_thirds.csv"))
    return T[(T.band == "alpha") & (T.estimator == "thirds")]


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


def flank_range(tag=""):
    """Smallest and largest estimate over the choices of flanks, per dataset
    and recording; empty if the within-recording scripts have not been run."""
    path = os.path.join(RES, f"within_recording_variants{tag}.csv")
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
    T = both_assignments("twocond_lambda_gmm_halves.csv", ["dataset", "spec", "model"])
    for r in hbn_thirds().itertuples():
        out.append(dict(design="between conditions: eyes closed vs open", dataset="HBN, 5-22 y",
                        spec=f"{'power law' if r.model == 'fixed' else 'knee + plateau'}, {r.window}",
                        n=r.n, lam=r.lam, lo=r.lo, hi=r.hi, calibration="",
                        n_roots=r.n_roots, roots=r.roots))
    V = flank_range()
    for tag, cleaning in CLEANING:
        p = os.path.join(RES, f"dortmund_levels{tag}.csv")
        if not os.path.exists(p):
            continue
        Vt = flank_range(tag)
        for r in pd.read_csv(p).query("covariates == False").itertuples():
            state = {"ec": "eyes closed", "eo": "eyes open"}[r.cond[:2]]
            when = "before" if r.cond.endswith("pre") else "after"
            out.append(dict(design=f"within session: segment fluctuations, {state}",
                            dataset="Dortmund, 20-70 y", cleaning=cleaning,
                            spec=f"flanks 2-6, 26-32, {when} 2-h tasks",
                            n=r.n_units, lam=r.lam, lo=r.lo, hi=r.hi,
                            calibration=calibration("dortmund_levels_sim{}" + tag + ".csv",
                                                    f"cond == '{r.cond}'")))
            out[-1].update(Vt.get(("Dortmund", r.cond), {}))
    x = pd.read_csv(os.path.join(RES, "chennu_lambda_gmm.csv"))
    for r in x[x.contrast == "baseline -> moderate"].itertuples():
        out.append(dict(design="between doses: propofol baseline vs moderate",
                        dataset="Chennu 2016", spec=f"{r.roi}, {r.window}", n=r.n, lam=r.lam,
                        lo=r.lo, hi=r.hi, calibration=""))
        t = T.get(("Chennu 2016", f"{r.roi}, {r.window}, {r.contrast}", "fixed"), {})
        out[-1].update({k: t[k] for k in ("lam", "lo", "hi") if k in t})
    for tag, cleaning in CLEANING:
        p = os.path.join(RES, f"srm_lambda{tag}.csv")
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p)
        s = s[s.analysis.str.contains("log-free") & (s.model == "fixed")]
        for r in s.itertuples():
            srm = r.analysis.startswith("srm")
            roots = [x for x in str(r.roots).strip("[]").split(",") if x.strip()]
            out.append(dict(design="between sessions: test-retest",
                            dataset="SRM, 17-71 y" if srm else "Dortmund, 20-70 y",
                            cleaning=cleaning,
                            spec=("SRM, eyes closed, later session" if srm else
                                  f"Dortmund, {'eyes closed' if 'ec_pre' in r.analysis else 'eyes open'}"
                                  ", 5 years"),
                            n=r.n, lam=r.lam_gmm, lo=r.gmm_lo, hi=r.gmm_hi, calibration="",
                            n_roots=len(roots), roots=str(r.roots)))
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
            if r.subset == "all" and r.covariates == "none":
                out[-1].update(V.get(("ds003688 iEEG", "rest"), {}))
    cols = ["design", "dataset", "spec", "cleaning", "n", "lam", "lo", "hi", "calibration",
            "n_patients", "lam_min", "lam_max", "n_roots", "roots"]
    O = pd.DataFrame(out).reindex(columns=cols)
    O["cleaning"] = O.cleaning.fillna("none")
    O["n_patients"] = O.n_patients.astype("Int64")
    O["n_roots"] = O.n_roots.astype("Int64")
    O.to_csv(os.path.join(RES, "identification_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
