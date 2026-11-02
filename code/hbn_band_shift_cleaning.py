"""Band-shift share of the HBN background's age slope across five backgrounds,
without and with artefact cleaning.

The share is 1 - s_b(8-12 Hz) / s_b(IAF +/- 2 Hz) (hbn_age_alpha_iaf.py): the
part of the background's age slope that comes from the individual band moving
up the 1/f slope as alpha frequency rises. Backgrounds:
  power law, knee + plateau  the censored fits of hbn_clean_check.py compare
                             (WORK/fits.csv, 2-55 Hz, 6-16 Hz censored), the
                             8-12 Hz band mean rebuilt from the stored parameters
  specparam fixed, knee      specparam 2.0 as in hbn_band_shift_specparam.py
                             (2-40 Hz, peak widths 1-8 Hz, up to 6 peaks)
  IRASA                      the fractal spectrum stored by hbn_extract_psd.py
                             (<cond>_irasa), mean over the region of interest
Pipelines: no cleaning (the uncleaned re-extraction WORK/psd_primary_seg, which
also carries IRASA; its Welch spectra equal the stored primary ones) and ICA +
ICLabel (WORK/psd_iclabel). Each pipeline uses its own eyes-closed alpha
frequency from the fits. Sample: quality-controlled participants with positive
narrowband power under the power law in both pipelines and every background
positive in both, so the backgrounds and pipelines are compared on the same
people. Intervals: pairs bootstrap over participants.
Writes results/hbn_band_shift_cleaning.csv.
Usage: python hbn_band_shift_cleaning.py [--work DIR] [--workers 8] [--nboot 2000]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
FIXED_BAND = (8.0, 12.0)
PIPES = {"primary": "psd_primary_seg", "iclabel": "psd_iclabel"}


def censored(F):
    """b(IAF +/- 2) and b(8-12) from the censored fits, per pipeline, model,
    participant and condition."""
    from hbn_age_alpha_iaf import band_shape
    F = F[(F.window == "censor 6-16") & (F.split == "full") & (F.band == "alpha")].copy()
    kf = np.where(F.model == "knee_plateau", F.knee_freq, 0.0)
    shape = lambda lo, hi: np.array([band_shape(c, q, l, h) for c, q, l, h in
                                     zip(F.exponent, kf, lo, hi)])
    amp = (F.b - F.plateau) / shape(F.iaf - 2, F.iaf + 2)
    n = len(F)
    F["b_fixed"] = amp * shape(np.full(n, FIXED_BAND[0]), np.full(n, FIXED_BAND[1])) + F.plateau
    F["background"] = F.model.map({"fixed": "power law", "knee_plateau": "knee + plateau"})
    return F[["pipeline", "background", "subject", "cond", "age", "a", "b", "b_fixed", "iaf"]]


def spectral(args):
    """specparam and IRASA backgrounds of one file at the given IAF."""
    path, pipeline, iaf = args
    from hbn_band_shift_specparam import one
    r = one((path, iaf))
    if r is None:
        return []
    d = np.load(path, allow_pickle=True)
    rows = []
    for cond in ("ec", "eo"):
        if not np.isfinite(iaf.get(cond, np.nan)):
            continue
        for mode in ("fixed", "knee"):
            rows.append(dict(pipeline=pipeline, background=f"specparam {mode}",
                             subject=r["subject"], cond=cond,
                             b=r.get(f"b_iaf_{mode}_{cond}", np.nan),
                             b_fixed=r.get(f"b812_{mode}_{cond}", np.nan)))
        if str(d["irasa_status"]) == "ok" and f"{cond}_irasa" in d.files:
            f = d["freqs"].astype(float)
            seg = [str(c) for c in d["seg_names"]]
            ix = [seg.index(str(c)) for c in d["roi_names"]]
            frac = d[f"{cond}_irasa"][ix].astype(float).mean(0)      # (n_freqs,)
            band = lambda lo, hi: float(np.mean(frac[(f >= lo) & (f <= hi)]))
            rows.append(dict(pipeline=pipeline, background="IRASA", subject=r["subject"],
                             cond=cond, b=band(iaf[cond] - 2, iaf[cond] + 2),
                             b_fixed=band(*FIXED_BAND)))
    return rows


def share(x, lb, lbf):
    return 1 - np.polyfit(x, lbf, 1)[0] / np.polyfit(x, lb, 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("HBN_CLEAN_WORK", "hbn_clean_all"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    C = censored(pd.read_csv(os.path.join(a.work, "fits.csv")))
    C = C[C.subject.isin(Q.index[Q.qc_ok])]
    iaf = C[C.background == "power law"].pivot_table(index=["pipeline", "subject"],
                                                     columns="cond", values="iaf")
    jobs = []
    for pipeline, sub_dir in PIPES.items():
        for p in glob.glob(os.path.join(a.work, sub_dir, "*.npz")):
            sub = os.path.basename(p)[:-4].split("_", 1)[1]
            if (pipeline, sub) in iaf.index:
                jobs.append((p, pipeline, iaf.loc[(pipeline, sub)].to_dict()))
    with Pool(a.workers) as pool:
        S = pd.DataFrame([r for rs in pool.imap(spectral, jobs, chunksize=8) for r in rs])
    S = S.drop_duplicates(["pipeline", "background", "subject", "cond"])
    D = pd.concat([C.drop(columns=["iaf"]), S], ignore_index=True)
    ages = C.drop_duplicates("subject").set_index("subject").age
    D["age"] = D.subject.map(ages)
    rows = []
    for cond in ("ec", "eo"):
        apos = C[(C.background == "power law") & (C.cond == cond)].pivot_table(
            index="subject", columns="pipeline", values="a")
        keep = set(apos.index[(apos > 0).all(axis=1)])
        G = D[(D.cond == cond) & D.subject.isin(keep)]
        ok = G[(G.b > 0) & (G.b_fixed > 0)].groupby("subject").size()
        n_cells = G.groupby(["pipeline", "background"]).ngroups
        both = set(ok.index[ok == n_cells])
        for (pipeline, bg), g in G[G.subject.isin(both)].groupby(["pipeline", "background"]):
            g = g.sort_values("subject")
            x, lb, lbf = g.age.to_numpy(), np.log(g.b.to_numpy()), np.log(g.b_fixed.to_numpy())
            rng = np.random.default_rng(0)
            n = len(g)
            bs = [share(x[i], lb[i], lbf[i]) for i in (rng.integers(0, n, n)
                                                       for _ in range(a.nboot))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(cond=cond, pipeline=pipeline, background=bg, n=n,
                             share=share(x, lb, lbf), lo=lo, hi=hi,
                             s_b_iaf=np.polyfit(x, lb, 1)[0], s_b_812=np.polyfit(x, lbf, 1)[0]))
    O = pd.DataFrame(rows)
    O.to_csv(os.path.join(RES, "hbn_band_shift_cleaning.csv"), index=False)
    for _, r in O.iterrows():
        print(f"{r.cond.upper()} {r.pipeline:8s} {r.background:16s} n = {r.n:5d}  share "
              f"{r.share:.3f} [{r.lo:.3f}, {r.hi:.3f}]  s_b(IAF) "
              f"{100 * (np.exp(r.s_b_iaf) - 1):+.2f}%/y  s_b(8-12) "
              f"{100 * (np.exp(r.s_b_812) - 1):+.2f}%/y")


if __name__ == "__main__":
    main()
