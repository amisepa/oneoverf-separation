"""Coupling exponent from epoch-to-epoch fluctuations in intracranial rest
recordings (ds003688), without skull or scalp artefacts.

Input: ieeg_epochs.py output (2-s epochs, bipolar pairs, three DPSS tapers).
Each bipolar channel with a peak in 7-13 Hz (residual above the censored
background fit > 0.1 in ln power, as in ds003690_lambda.py) is a unit with
its own intrinsic strength; epochs flagged for gross artefacts are dropped.
The analysis is that of ds003690_lambda.py (per-taper censored log-log fit
over 2-40 Hz, 6-16 Hz left out, ap_models.loglog_fit; band = peak +/- 2 Hz;
T, B and the instrument from three different tapers, all six assignments
pooled; lambda_gmm.estimate_levels), with EOG (0.5-4 Hz) and EMG
(60-95 Hz) log power as optional covariates and a bootstrap over patients
(all of a patient's channels together). Subsets: all channels with a peak,
and posterior channels (pair midpoint y < -40 mm, ACPC).

--simulate L replaces each channel by synthetic epochs built from its own
mean background, peak and background fluctuation with coupling L
(ds003690_lambda.simulate), as a check on data like these.

Usage: python ieeg_lambda.py [--data DIR] [--nboot 200] [--simulate L]
       [--subsets all,posterior] [--covsets none,EOG,"EOG + EMG"] [--tag _part]
Writes results/ieeg_lambda.csv (or ieeg_lambda_sim<L>.csv).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lambda_gmm as G
from ds003690_lambda import GRID, band_arrays, iv_log, simulate, stack

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
COVSETS = {"none": [], "EOG": [0], "EOG + EMG": [0, 1]}


def load(data):
    units = []
    for p in sorted(glob.glob(os.path.join(data, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        keep = ~d["bad"]
        f = d["freqs"].astype(float)
        P = d["P"][keep].astype(float)                 # epochs x tapers x pairs x f
        cov = d["cov"][keep]
        sub = str(d["subject"])
        for j, name in enumerate(d["pairs"]):
            units.append(dict(subject=f"{sub}:{name}", group=sub, age=np.nan, f=f,
                              P=P[:, :, j, :], cov=cov, y=float(d["pos"][j][1])))
    return units


def run(rows, use_cov, nboot, rng):
    T, B, Bz, grp, X = stack(rows, use_cov)
    est = G.estimate_levels(T, B, Bz, grp, X, grid=GRID)
    pats = sorted({r["group"] for r in rows})
    by = {p: [r for r in rows if r["group"] == p] for p in pats}
    bs = []
    for _ in range(nboot):
        rr = []
        for j, p in enumerate(rng.choice(pats, len(pats), replace=True)):
            for r in by[p]:
                r2 = dict(r)
                r2["subject"] = f"{r['subject']}_{j}"
                rr.append(r2)
        e = G.estimate_levels(*stack(rr, use_cov), grid=GRID)
        if e["roots"] and np.isfinite(est["lam"]):
            bs.append(min(e["roots"], key=lambda x: abs(x - est["lam"])))
        else:
            bs.append(e["lam"] if e["roots"] else np.nan)
    bs = np.array(bs)
    ok = np.isfinite(bs)
    lo, hi = np.percentile(bs[ok], [2.5, 97.5]) if ok.sum() > 10 else (np.nan, np.nan)
    return dict(lam=est["lam"], lo=lo, hi=hi, fail=float(1 - ok.mean()), roots=str(est["roots"]),
                n_channels=len(rows), n_patients=len(pats), n_epochs=int(T.size / 6),
                gamma=np.round(est["gamma"], 3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("IEEG_OUT", "ieeg_epochs"))
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--simulate", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsets", default="all,posterior")
    ap.add_argument("--covsets", default=",".join(COVSETS))
    ap.add_argument("--tag", default="", help="suffix for the output file (parallel runs)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    units = load(a.data)
    if a.simulate is not None:
        units = simulate(units, a.simulate, rng)
    rows = band_arrays(units)
    ypos = {u["subject"]: u["y"] for u in units}
    for r in rows:
        r["y"] = ypos[r["subject"]]
    tag = "data" if a.simulate is None else f"simulated lambda = {a.simulate}"
    rel = np.nanmean([np.corrcoef(np.log(r["B"][:, 0]), np.log(r["B"][:, 1]))[0, 1] for r in rows])
    lam_iv, n_iv = iv_log(rows)
    print(f"{tag}: {len(units)} bipolar channels, {len(rows)} with a 7-13 Hz peak in "
          f"{len({r['group'] for r in rows})} patients; reliability of ln b across tapers "
          f"{rel:.2f}; IV-log (a > 0) {lam_iv:.3f} (n {n_iv})", flush=True)
    out = []
    for subset in a.subsets.split(","):
        rr = rows if subset == "all" else [r for r in rows if r["y"] < -40]
        sets = [(c, COVSETS[c]) for c in a.covsets.split(",")]
        for cname, use_cov in (sets if a.simulate is None else [("none", [])]):
            r = run(rr, use_cov, a.nboot, rng)
            r.update(sample=tag, subset=subset, covariates=cname, reliability=rel)
            out.append(r)
            print(f"  {subset:9s} covariates={cname:9s} lambda {r['lam']:+.3f} "
                  f"[{r['lo']:+.2f}, {r['hi']:+.2f}] fail {r['fail']:.2f} "
                  f"channels {r['n_channels']} patients {r['n_patients']} roots {r['roots']}",
                  flush=True)
    name = ("ieeg_lambda" if a.simulate is None else f"ieeg_lambda_sim{a.simulate}") + a.tag + ".csv"
    pd.DataFrame(out).to_csv(os.path.join(RES, name), index=False)


if __name__ == "__main__":
    main()
