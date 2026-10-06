"""HBN log-free coupling exponent with the band total, the background and the
instrument each taken from a different third of the Welch segments.

What. For every participant and condition (eyes open, eyes closed) the kept
4 s segments are split into three interleaved thirds (segment index mod 3,
stored by hbn_extract_psd.py). The aperiodic background is fitted to the
posterior ROI spectrum of each third, and lambda_gmm is run with
instrument="third": the band total t, the background b in a = t - b and in
b^lambda, and the instrument with the weights come from three different
thirds, pooled over the six assignments.

Why. With two halves the instrument has to share a half with t or with b.
In HBN the true background differs between the halves of a recording (the
half-differences of ln t and fitted ln b correlate about +0.45 to +0.50), so
the estimate depends on that choice: it reads high with the instrument from
the half of t and low from the half of b (hbn_half_assignment.py;
simulation in sim_lambda_thirds.py). With thirds neither is shared. A third
holds about 4 eyes-open segments at the median, so the estimate is noisier.

Specifications as in hbn_kp.py: power law or knee + plateau, fitted on
2-55 Hz without 6-16 Hz or on the flanks 3-6 and 26-36 Hz; alpha band
(IAF +/- 2 Hz) and the peak-free control band (30-38 Hz). In the control
band the fitting windows contain bins of the band, but t and the fits now
come from different segments. Roots are searched on -1 to 3; intervals are
percentile intervals over a participant bootstrap that follows the root
nearest the full-sample estimate. The two half assignments are computed
from the odd/even spectra of the same files for comparison.

Quality control follows hbn_age_alpha_robust.py: flat (alpha-band total
more than 300 times off the sample median in either condition), edge IAF
(<= 6.25 or >= 13.75 Hz) and fallback (every segment kept: 35 eyes open or
85 eyes closed). By default the flags are recomputed from the fits made here
and compared with results/hbn_qc_flags.csv; --qc file applies the stored
flags instead (use it on a subset of the sample, where the sample median is
not meaningful).

Writes results/hbn_thirds_fits.csv (per participant, kept out of the
repository) and results/hbn_lambda_gmm_thirds.csv.

Usage: python hbn_thirds.py [--psd-dir DIR] [--workers N] [--nboot 300]
                            [--qc recompute|file|none] [--from-fits]
                            [--limit N] [--out-dir DIR]
  --psd-dir    directory of the .npz files (default $HBN_OUT or ./hbn_psd)
  --from-fits  skip the fits and read hbn_thirds_fits.csv from the output
               directory
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
from ap_models import fit_aperiodic, ap_band_power
import lambda_gmm as G
from hbn_kp import (CONTROL_BAND, FIT_RANGE, ROI, WINDOWS, find_iaf, mask_for,
                    neural_band_power)

GRID = np.linspace(-1, 3, 161)
MODELS = ("fixed", "knee_plateau")
THIRDS = ("t0", "t1", "t2")
HALVES = ("odd", "even")
FULL_SEG = {"eo": 35, "ec": 85}     # segments before rejection


def spectra(d, cond):
    """split -> (channel spectra, number of segments) for one condition."""
    n = int(d[f"n_seg_{cond}"])
    out = {"full": (d[cond], n), "odd": (d[cond + "_odd"], (n + 1) // 2),
           "even": (d[cond + "_even"], n // 2)}
    for k, name in enumerate(THIRDS):
        out[name] = (d[cond + "_thirds"][k], int(d[f"n_seg_{cond}_thirds"][k]))
    return out


def subject_rows(path):
    """Fit rows for one PSD file; returns (rows, error message or None)."""
    rows = []
    try:
        d = np.load(path, allow_pickle=True)
        f0 = d["freqs"].astype(float)
        sel = (f0 >= FIT_RANGE[0]) & (f0 <= FIT_RANGE[1])
        f = f0[sel]
        ch = [str(c) for c in d["ch_names"]]
        ix = [ch.index(c) for c in ROI if c in ch]
        if len(ix) < 3:
            return rows, None
        iaf = find_iaf(d["ec"][ix][:, sel].astype(float).mean(0), f)
        if not np.isfinite(iaf):
            return rows, None
        meta = dict(subject=str(d["subject"]), dataset=str(d["dataset"]),
                    n_seg_eo=int(d["n_seg_eo"]), n_seg_ec=int(d["n_seg_ec"]), iaf=iaf)
        bands = {"alpha": (iaf - 2, iaf + 2), "control": CONTROL_BAND}
        for cond in ("eo", "ec"):
            for split, (S, nseg) in spectra(d, cond).items():
                if nseg == 0:
                    continue
                P = S[ix][:, sel].astype(float).mean(0)
                P = np.clip(np.nan_to_num(P, nan=1e-12), 1e-12, None)
                for wname, (kind, spec) in WINDOWS.items():
                    keep = mask_for(f, kind, spec)
                    for model in MODELS:
                        fit = fit_aperiodic(P, f, keep, model)
                        if fit is None:
                            continue
                        for bname, (lo, hi) in bands.items():
                            rows.append(dict(
                                **meta, cond=cond, split=split, n_seg=nseg,
                                window=wname, model=model, band=bname,
                                tot=float(np.mean(P[(f >= lo) & (f <= hi)])),
                                b=ap_band_power(fit, f, lo, hi),
                                b_neural=neural_band_power(fit, f, lo, hi),
                                exponent=fit["exponent"], knee_freq=fit["knee_freq"],
                                plateau=fit["plateau"], dev=fit["dev"]))
    except Exception as e:
        return [], f"{os.path.basename(path)}: {type(e).__name__}: {e}"
    return rows, None


def qc_flags(T):
    """flat, edge_iaf, fallback and qc_ok per participant, from the fits."""
    F = T[(T.model == "fixed") & (T.window == "censor 6-16") & (T.band == "alpha")
          & (T.split == "full")]
    t = F.pivot_table(index="subject", columns="cond", values="tot")
    M = F.drop_duplicates("subject").set_index("subject").loc[t.index]
    flat = ((t.ec < t.ec.median() / 300) | (t.ec > t.ec.median() * 300) |
            (t.eo < t.eo.median() / 300) | (t.eo > t.eo.median() * 300))
    Q = pd.DataFrame(dict(flat=flat, edge_iaf=(M.iaf <= 6.25) | (M.iaf >= 13.75),
                          fallback=(M.n_seg_eo >= FULL_SEG["eo"]) |
                                   (M.n_seg_ec >= FULL_SEG["ec"])))
    Q["qc_ok"] = ~(Q.flat | Q.edge_iaf | Q.fallback)
    return Q


def replicates(T, band, model, window, splits):
    """Band totals and backgrounds per condition, arrays (n, len(splits)),
    for the participants with every split in both conditions."""
    F = T[(T.band == band) & (T.model == model) & (T.window == window)
          & T.split.isin(splits)]
    w = F.pivot_table(index="subject", columns=["cond", "split"],
                      values=["tot", "b", "n_seg"]).dropna()
    H = {(v, c): w[v][c][list(splits)].to_numpy()
         for v in ("tot", "b", "n_seg") for c in ("eo", "ec")}
    return H, len(w)


def reliability(H):
    """Mean correlation of the instrument between the replicates."""
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.log(H["b", "ec"]) - np.log(H["b", "eo"])
    z = z[np.all(np.isfinite(z), 1)]
    k = z.shape[1]
    if z.shape[0] < 3:
        return np.nan
    return float(np.mean([np.corrcoef(z[:, i], z[:, j])[0, 1]
                          for i in range(k) for j in range(i + 1, k)]))


def lambda_rows(T, nboot):
    """One row per band, specification and estimator."""
    out = []
    for band in ("alpha", "control"):
        for model in MODELS:
            for win in WINDOWS:
                for est, splits, inst in (("thirds", THIRDS, "third"),
                                          ("halves, instrument with t", HALVES, "total"),
                                          ("halves, instrument with b", HALVES, "background")):
                    H, n = replicates(T, band, model, win, splits)
                    if n < 3:
                        continue
                    r = G.bootstrap(H["tot", "eo"], H["tot", "ec"], H["b", "eo"], H["b", "ec"],
                                    nboot=nboot, rng=np.random.default_rng(0), grid=GRID,
                                    instrument=inst)
                    out.append(dict(
                        band=band, model=model, window=win, estimator=est, n=n,
                        lam=r["lam"], lo=r["ci"][0], hi=r["ci"][1], boot_sd=r["boot_sd"],
                        boot_fail=r["boot_fail"], n_roots=len(r["roots"]),
                        roots=" ".join(f"{x:.3f}" for x in r["roots"]),
                        r_instrument=reliability(H),
                        med_seg_eo=float(np.median(H["n_seg", "eo"])),
                        med_seg_ec=float(np.median(H["n_seg", "ec"]))))
    return pd.DataFrame(out)


def coverage(T, psd_dir):
    """Participants per dataset here and in the earlier extraction
    (results/hbn_roi.csv), so that a release that came back short shows."""
    now = T.drop_duplicates("subject").groupby("dataset").size().rename("fitted")
    C = pd.DataFrame(now)
    p = os.path.join(RES, "hbn_roi.csv")
    if os.path.exists(p):
        R = pd.read_csv(p, usecols=["subject", "dataset"]).drop_duplicates("subject")
        C = C.join(R.groupby("dataset").size().rename("earlier"), how="outer")
    print(C.fillna(0).astype(int).to_string())
    p = os.path.join(psd_dir, "extract_status.csv")
    if os.path.exists(p):
        S = pd.read_csv(p)
        bad = S.status.str.startswith(("fetch-failed", "error"))
        print(f"extract_status.csv: {int(bad.sum())} of {len(S)} participants failed "
              "to download or read" + (" -- run the extraction again" if bad.any() else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--from-fits", action="store_true")
    ap.add_argument("--qc", default="recompute", choices=["recompute", "file", "none"])
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--out-dir", default=RES)
    a = ap.parse_args()
    fits_path = os.path.join(a.out_dir, "hbn_thirds_fits.csv")
    pd.set_option("display.width", 220)

    if a.from_fits:
        T = pd.read_csv(fits_path)
        print(f"read {fits_path}", flush=True)
    else:
        files = sorted(glob.glob(os.path.join(a.psd_dir, "*.npz")))
        if a.limit:
            files = files[:a.limit]
        if not files:
            sys.exit(f"no .npz files in {a.psd_dir}")
        print(f"{len(files)} PSD files", flush=True)
        pool = Pool(a.workers) if a.workers > 1 else None
        results = pool.imap(subject_rows, files, chunksize=4) if pool else map(subject_rows, files)
        rows = []
        for k, (r, err) in enumerate(results):
            if err:
                print(f"  {err}", flush=True)
            rows += r
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(files)}", flush=True)
        if pool:
            pool.close()
        T = pd.DataFrame(rows)
        T.to_csv(fits_path, index=False)
    print(f"{T.subject.nunique()} participants fitted; per dataset:")
    coverage(T, a.psd_dir)

    Q = qc_flags(T)
    print(f"\nQC from these fits: {len(Q)} participants; flat {int(Q.flat.sum())}, edge IAF "
          f"{int(Q.edge_iaf.sum())}, fallback {int(Q.fallback.sum())}; "
          f"{int(Q.qc_ok.sum())} pass")
    p = os.path.join(RES, "hbn_qc_flags.csv")
    if os.path.exists(p):
        Q0 = pd.read_csv(p, index_col=0)
        both = Q.index.intersection(Q0.index)
        print(f"stored flags (hbn_qc_flags.csv): {len(both)} participants in common, "
              f"{int((Q.loc[both, 'qc_ok'] != Q0.loc[both, 'qc_ok']).sum())} differ in qc_ok; "
              f"{len(Q.index.difference(Q0.index))} only here, "
              f"{len(Q0.index.difference(Q.index))} only stored")
        if a.qc == "file":
            Q = Q0
    if a.qc != "none":
        T = T[T.subject.isin(Q.index[Q.qc_ok])]
    print(f"{T.subject.nunique()} participants analysed (qc: {a.qc})")

    n3 = T[T.split.isin(THIRDS)].drop_duplicates(["subject", "cond", "split"])
    for cond in ("eo", "ec"):
        v = n3[n3.cond == cond].n_seg
        if len(v):
            print(f"segments per third, {cond}: median {v.median():.0f}, "
                  f"5th-95th percentile {v.quantile(.05):.0f}-{v.quantile(.95):.0f}, "
                  f"{100 * np.mean(v < 2):.1f}% with one segment")

    L = lambda_rows(T, a.nboot)
    out = os.path.join(a.out_dir, "hbn_lambda_gmm_thirds.csv")
    L.to_csv(out, index=False)
    print("\nHBN, eyes open -> eyes closed")
    print(L.round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
