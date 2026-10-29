"""Share of the background's age slope that is the band moving, under specparam.

hbn_age_alpha_iaf.py reports the band-shift share 1 - s_b(8-12 Hz) /
s_b(IAF +/- 2 Hz) for the censored power-law and knee+plateau fits. This
recomputes it with specparam 2.0's own aperiodic fit (2-40 Hz, peak width
1-8 Hz, up to 6 peaks), in fixed and knee mode, on the same posterior-ROI
spectra, quality-controlled sample and individual alpha frequency (each
condition's IAF from the censored power-law fit, results/hbn_kp_fits.csv).
Band means are taken on the 0.25 Hz grid of hbn_age_alpha_iaf.py. Two samples:
"a > 0" is the power-law row's sample in hbn_age_alpha_iaf.csv (so only the
background estimator differs), "all qc" keeps every quality-controlled
participant.

Writes results/hbn_band_shift_specparam.csv.

Usage: python hbn_band_shift_specparam.py [--psd-dir DIR] [--workers 12] [--nboot 2000]
"""
import argparse
import glob
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROI = ["E62", "E65", "E70", "E75", "E83", "E90"]
FREQS = np.arange(2.0, 55.0 + 0.125, 0.25)     # grid of hbn_age_alpha_iaf.py
FIXED_BAND = (8.0, 12.0)
warnings.filterwarnings("ignore")


def band_mean(sm, lo, hi):
    """Band mean of specparam's aperiodic fit in linear power."""
    f = FREQS[(FREQS >= lo) & (FREQS <= hi)]
    return np.mean(10 ** sm.modes.aperiodic.func(f, *sm.get_params("aperiodic")))


def one(args):
    path, iaf = args
    try:
        from specparam import SpectralModel
        d = np.load(path, allow_pickle=True)
        f = d["freqs"].astype(float)
        ch = [str(c) for c in d["ch_names"]]
        ix = [ch.index(c) for c in ROI if c in ch]
        out = dict(subject=str(d["subject"]), age=float(d["age"]))
        for cond in ("ec", "eo"):
            if not np.isfinite(iaf.get(cond, np.nan)):
                continue
            P = d[cond][ix].astype(float).mean(0)
            for mode in ("fixed", "knee"):
                sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                                   aperiodic_mode=mode, verbose=False)
                sm.fit(f, P, [2, 40])
                out[f"b_iaf_{mode}_{cond}"] = band_mean(sm, iaf[cond] - 2, iaf[cond] + 2)
                out[f"b812_{mode}_{cond}"] = band_mean(sm, *FIXED_BAND)
                out[f"exp_{mode}_{cond}"] = sm.get_params("aperiodic")[-1]
        return out
    except Exception:
        return None


def share(x, lb, lbf):
    return 1 - np.polyfit(x, lbf, 1)[0] / np.polyfit(x, lb, 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    k = k[(k.model == "fixed") & (k.window == "censor 6-16") & (k.band == "alpha")
          & (k.split == "full")]
    iaf = k.pivot_table(index="subject", columns="cond", values="iaf")
    apos = k.assign(pos=k.a > 0).pivot_table(index="subject", columns="cond", values="pos")
    jobs = []
    for p in glob.glob(os.path.join(a.psd_dir, "*.npz")):
        sub = os.path.basename(p).split("_")[1].replace(".npz", "")
        if sub in Q.index and Q.loc[sub, "qc_ok"] and sub in iaf.index:
            jobs.append((p, iaf.loc[sub].to_dict()))
    with Pool(a.workers) as pool:
        R = pd.DataFrame([r for r in pool.imap(one, jobs, chunksize=8) if r is not None])
    R = R.drop_duplicates("subject").set_index("subject")
    rows = []
    for cond in ("ec", "eo"):
        for sample in ("a > 0", "all qc"):
            for mode in ("fixed", "knee"):
                c_b, c_f = f"b_iaf_{mode}_{cond}", f"b812_{mode}_{cond}"
                d = R[np.isfinite(R[c_b]) & np.isfinite(R[c_f]) & (R[c_b] > 0) & (R[c_f] > 0)]
                if sample == "a > 0":
                    d = d[d.index.isin(apos.index[apos[cond] == 1])]
                x, lb, lbf = d.age.to_numpy(), np.log(d[c_b]).to_numpy(), np.log(d[c_f]).to_numpy()
                rng = np.random.default_rng(0)
                n = len(d)
                bs = []
                for _ in range(a.nboot):
                    i = rng.integers(0, n, n)
                    bs.append(share(x[i], lb[i], lbf[i]))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append(dict(cond=cond, sample=sample, model=f"specparam {mode}", n=n,
                                 share=share(x, lb, lbf), lo=lo, hi=hi,
                                 s_b_iaf=np.polyfit(x, lb, 1)[0],
                                 s_b_812=np.polyfit(x, lbf, 1)[0],
                                 median_exponent=np.median(d[f"exp_{mode}_{cond}"])))
    O = pd.DataFrame(rows)
    O.to_csv(os.path.join(RES, "hbn_band_shift_specparam.csv"), index=False)
    for _, r in O.iterrows():
        print(f"{r.cond.upper()} {r['sample']:7s} {r.model:16s} n = {r.n:5d}  share {r.share:.3f} "
              f"[{r.lo:.3f}, {r.hi:.3f}]  s_b(IAF) {100 * (np.exp(r.s_b_iaf) - 1):+.2f}%/y  "
              f"s_b(8-12) {100 * (np.exp(r.s_b_812) - 1):+.2f}%/y  median exponent {r.median_exponent:.2f}")


if __name__ == "__main__":
    main()
