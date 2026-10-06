"""Log-free coupling exponent under both half assignments of the instrument.

lambda_gmm takes the band total t from one half of a recording's Welch
segments and the background b from the other half. The instrument and the
weights come from a background fit as well, and either half can supply it:

  total        the half of t (the default of lambda_gmm)
  background   the half of b

Both are valid when the two halves share one true background. When they do
not (the background drifts between the odd and the even segments), the first
reads high and the second reads low, so the two are reported side by side
and the gap between them is the sensitivity of the estimate to that choice.

HBN, eyes open -> eyes closed, alpha band, participants passing quality
control, for the four specifications of hbn_kp.py (power law or knee +
plateau; censor 6-16 Hz or flanks 3-6 and 26-36 Hz), from the per-participant
fits in results/hbn_kp_fits.csv (local). The same comparison is made for the
other two-condition designs whose half spectra are in results/: test-retest
in SRM (results/srm_bandpower.csv) and Dortmund (results/aging_bandpower.csv),
as in srm_lambda.py, and propofol baseline -> moderate
(results/chennu_spectra.csv), as in chennu_analysis.py.

Roots are searched on -1 to 3 in every case; intervals are percentile
intervals over a participant bootstrap that follows the root nearest the
full-sample estimate, with the same resamples for both assignments.

Writes results/hbn_lambda_gmm_halves.csv and
results/twocond_lambda_gmm_halves.csv (one row per specification and
assignment).

Usage: python hbn_half_assignment.py [--nboot 300]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
import lambda_gmm as G
from hbn_kp import WINDOWS, halves

GRID = np.linspace(-1, 3, 161)
ASSIGNMENTS = ("total", "background")


def both(t1, t2, b1, b2, nboot):
    """One row per assignment: estimate, interval, roots."""
    rows = []
    for inst in ASSIGNMENTS:
        r = G.bootstrap(t1, t2, b1, b2, nboot=nboot, rng=np.random.default_rng(0),
                        grid=GRID, instrument=inst)
        rows.append(dict(instrument=inst, n=t1.shape[0], lam=r["lam"], lo=r["ci"][0],
                         hi=r["ci"][1], boot_sd=r["boot_sd"], boot_fail=r["boot_fail"],
                         n_roots=len(r["roots"]),
                         roots=" ".join(f"{x:.3f}" for x in r["roots"])))
    return rows


def hbn(nboot):
    T = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    T = T[T.subject.isin(Q.index[Q.qc_ok])]
    if "b_neural" not in T:
        T = T.assign(b_neural=T.b - T.plateau)
    out = []
    for model in ("fixed", "knee_plateau"):
        for win in WINDOWS:
            H, _ = halves(T, "alpha", model, win)
            for r in both(H["tot", "eo"], H["tot", "ec"], H["b", "eo"], H["b", "ec"], nboot):
                out.append(dict(model=model, window=win, **r))
    return pd.DataFrame(out)


def paired(D, key, by, c1, c2):
    """Odd/even totals and backgrounds for conditions c1, c2 of column `by`."""
    w = D[D.split != "full"].pivot_table(index=key, columns=[by, "split"],
                                         values=["tot", "b"]).dropna()
    return [w[v][c][["odd", "even"]].to_numpy() for v, c in
            (("tot", c1), ("tot", c2), ("b", c1), ("b", c2))]


def others(nboot):
    out = []

    def add(dataset, spec, model, arr):
        for r in both(*arr, nboot):
            out.append(dict(dataset=dataset, spec=spec, model=model, **r))

    p = os.path.join(RES, "srm_bandpower.csv")
    if os.path.exists(p):
        S = pd.read_csv(p)
        S = S[~S.bad.fillna(True).astype(bool)]
        for model in ("fixed", "knee_plateau"):
            add("SRM", "eyes closed, later session", model,
                paired(S[S.model == model], "subject", "session", "t1", "t2"))
    p = os.path.join(RES, "aging_bandpower.csv")
    if os.path.exists(p):
        A = pd.read_csv(p)
        A = A[(A.study == "dortmund") & ~A.bad.fillna(True).astype(bool)]
        for cond, state in (("ec_pre", "eyes closed"), ("eo_pre", "eyes open")):
            for model in ("fixed", "knee_plateau"):
                add("Dortmund", f"{state}, 5 years", model,
                    paired(A[(A.cond == cond) & (A.model == model)], "subject", "session", 1, 2))
    p = os.path.join(RES, "chennu_spectra.csv")
    if os.path.exists(p):
        C = pd.read_csv(p)
        for win in ("censor 6-16", "censor 6-25"):
            for roi in ("posterior", "frontal"):
                add("Chennu 2016", f"{roi}, {win}, baseline -> moderate", "fixed",
                    paired(C[(C.window == win) & (C.roi == roi)], "subject", "level", 1, 3))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=300)
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    H = hbn(a.nboot)
    H.to_csv(os.path.join(RES, "hbn_lambda_gmm_halves.csv"), index=False)
    print("HBN, eyes open -> eyes closed")
    print(H.round(3).to_string(index=False))
    O = others(a.nboot)
    if len(O):
        O.to_csv(os.path.join(RES, "twocond_lambda_gmm_halves.csv"), index=False)
        print("\nother two-condition designs")
        print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
