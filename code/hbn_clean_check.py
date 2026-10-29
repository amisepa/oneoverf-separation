"""Does artefact cleaning change the HBN age result? A paired subset check.

The primary HBN spectra (hbn_extract_psd.py) come from band-passed,
average-referenced recordings with outlier segments rejected, but no ICA,
ASR or channel repair. This script takes an age-stratified subset of the
quality-controlled sample, cleans the same recordings in EEGLAB in two ways
(hbn_clean_eeglab.m: 60/120 Hz notch, average reference with the zero Cz
included, bad channels by clean_rawdata's channel criteria, then GEDAI 'auto'
or Picard ICA at the effective rank with ICLabel eye/muscle >= 0.8 removed,
then spherical-spline interpolation) and recomputes the spectra with the same
extractor, so the pipelines differ only by the cleaning (the notch lies above
the 55 Hz fit range).

  select    pick up to --per-bin participants per one-year age bin (5-21 y)
            from the quality-controlled sample; writes the subject list
  download  fetch their RestingState .set files into WORK/raw
  (then hbn_clean_eeglab.m on WORK/raw into WORK/clean_gedai and
   WORK/clean_iclabel, one MATLAB process per core; hbn_clean_all.py runs
   those processes, the extraction and the clean-up for a whole list)
  extract   run hbn_extract_psd.py on WORK/clean_<arm> into WORK/psd_<arm>;
            --arm primary re-extracts the uncleaned WORK/raw into
            WORK/psd_primary_seg, with the per-segment spectra the stored
            primary files predate (for segmatch)
  compare   fit both sets of spectra with hbn_kp.py's subject_rows (power
            law and knee+plateau, 2-55 Hz, 6-16 Hz censored, band IAF +/- 2 Hz,
            IAF from each pipeline's own eyes-closed spectrum), then per
            condition the age slopes of ln a, ln b, ln a - ln b and the
            crossover lambda* on the participants present in every pipeline,
            plus the paired per-participant differences from the primary
            spectra; writes results/hbn_clean_check.csv
  segmatch  the same, per arm, on the segments kept by both that arm and the
            primary pipeline: cleaning changes which segments pass the
            extractor's outlier rejection, so compare alone mixes the effect
            of cleaning with a change of data. Rebuilds each participant's
            ROI spectrum in both pipelines from the shared segments (WORK/
            segmatch/<arm>), drops a condition with fewer than 4; writes
            results/hbn_clean_check_segmatched.csv

Usage: python hbn_clean_check.py select|download|extract|compare|segmatch
           [--work DIR] [--psd-dir DIR] [--arm gedai|iclabel|primary]
           [--per-bin 24] [--workers 8]
Extract with --workers 3 or fewer while the MATLAB cleaning runs; 8 broke the
process pool.
"""
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")


def subject_list(work):
    return os.path.join(work, "subjects.txt")


def select(a):
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    rows = []
    for p in glob.glob(os.path.join(a.psd_dir, "*.npz")):
        ds, sub = os.path.basename(p)[:-4].split("_", 1)
        if sub in Q.index and Q.loc[sub, "qc_ok"]:
            age = float(np.load(p, allow_pickle=True)["age"])
            rows.append(dict(ds=ds, sub=sub, age=age))
    D = pd.DataFrame(rows).drop_duplicates("sub")
    D["bin"] = np.clip(np.floor(D.age), 5, 21).astype(int)
    pick = (D.groupby("bin", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), a.per_bin), random_state=0)))
    os.makedirs(a.work, exist_ok=True)
    pick[["ds", "sub"]].to_csv(subject_list(a.work), sep=" ", header=False, index=False)
    print(f"{len(pick)} participants, ages {pick.age.min():.1f}-{pick.age.max():.1f}")
    print(pick.groupby("bin").size().to_string())


def fetch_raw(ds, sub, raw):
    """Fetch one RestingState .set to raw/<ds>_<sub>.set (via a .part file,
    so a partial download never looks finished); returns a status."""
    import hbn_extract_psd as X
    dest = os.path.join(raw, f"{ds}_{sub}.set")
    if os.path.exists(dest):
        return "cached"
    part = dest + ".part"
    base = f"{X.S3}/{ds}/{sub}/eeg/{sub}_task-RestingState_eeg.set"
    try:
        ok = X.fetch(base, part)
        if not ok:
            url = X.snapshot_url(ds, sub, f"{sub}_task-RestingState_eeg.set")
            ok = bool(url) and X.fetch(url, part)
        if not ok:
            return f"missing {ds} {sub}"
        os.replace(part, dest)
        return "ok"
    except Exception as e:
        return f"failed {ds} {sub}: {e}"


def download(a):
    raw = os.path.join(a.work, "raw")
    os.makedirs(raw, exist_ok=True)
    jobs = [l.split() for l in open(subject_list(a.work)) if l.strip()]
    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(lambda j: fetch_raw(*j, raw), jobs))
    for s in sorted(set(out)):
        print(f"{out.count(s):4d}  {s}")


ARMS = ("gedai", "iclabel")
SEG_PRIMARY = "psd_primary_seg"


def extract(a):
    if a.arm == "primary":
        # hbn_extract_psd.py reads local files only as <ds>_<sub>_clean.set
        src = os.path.join(a.work, "raw_as_clean")
        os.makedirs(src, exist_ok=True)
        for p in glob.glob(os.path.join(a.work, "raw", "*.set")):
            dest = os.path.join(src, os.path.basename(p)[:-4] + "_clean.set")
            if not os.path.exists(dest):
                os.link(p, dest)
        out = os.path.join(a.work, SEG_PRIMARY)
    else:
        src, out = os.path.join(a.work, f"clean_{a.arm}"), os.path.join(a.work, f"psd_{a.arm}")
    env = dict(os.environ, HBN_OUT=out, HBN_LOCAL_SET=src)
    dss = sorted({l.split()[0] for l in open(subject_list(a.work)) if l.strip()})
    subprocess.run([sys.executable, os.path.join(HERE, "hbn_extract_psd.py"), *dss,
                    "--workers", str(a.workers), "--subjects", subject_list(a.work)],
                   env=env, check=True)


def fit_one(args):
    """hbn_kp.py's own fits (full recording, 6-16 Hz censored, IAF +/- 2 Hz) of
    one spectrum file, so both pipelines go through identical code."""
    path, pipeline = args
    from hbn_kp import subject_rows
    rows, err = subject_rows((path, ("fixed", "knee_plateau")))
    d = np.load(path, allow_pickle=True)
    age = float(d["age"])
    return [dict(r, age=age, pipeline=pipeline,
                 n_seg=int(d[f"n_seg_{r['cond']}"])) for r in rows
            if r["window"] == "censor 6-16" and r["split"] == "full" and r["band"] == "alpha"]


def slope(y, x):
    r = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    return r.params[1], r.conf_int()[1][0], r.conf_int()[1][1]


def compare(a):
    subs = {l.split()[1] for l in open(subject_list(a.work)) if l.strip()}
    jobs = []
    dirs = [("primary", a.psd_dir)] + [(arm, os.path.join(a.work, f"psd_{arm}")) for arm in ARMS]
    for pipeline, d in dirs:
        for p in glob.glob(os.path.join(d, "*.npz")):
            if os.path.basename(p)[:-4].split("_", 1)[1] in subs:
                jobs.append((p, pipeline))
    with Pool(a.workers) as pool:
        F = pd.DataFrame([r for rs in pool.imap(fit_one, jobs, chunksize=4) for r in rs])
    F.to_csv(os.path.join(a.work, "fits.csv"), index=False)
    O = summarise(F, [p for p, _ in dirs], a.nboot)
    O.to_csv(os.path.join(RES, "hbn_clean_check.csv"), index=False)
    report(O)


def matched(p, q, out_p, out_q):
    """Write the ROI spectra of the segments kept in both p and q (two
    pipelines on one recording) to out_p and out_q, as spectrum files
    fit_one reads; returns the kept counts per condition (shared -1 when the
    segmentations differ)."""
    dp, dq = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
    roi = [str(c) for c in dp["roi_names"]]
    six = [list(dp["seg_names"]).index(c) for c in roi]
    cix = [list(dp["ch_names"]).index(c) for c in roi]
    outs = [dict(freqs=d["freqs"], ch_names=np.array(roi), subject=d["subject"], age=d["age"])
            for d in (dp, dq)]
    counts = {}
    for c in ("ec", "eo"):
        up, uq = dp[f"{c}_seg_used"], dq[f"{c}_seg_used"]
        same = np.array_equal(dp[f"{c}_seg_block"], dq[f"{c}_seg_block"])
        both = up & uq if same else None
        n = int(both.sum()) if same else -1
        counts[c] = (int(up.sum()), int(uq.sum()), n)
        for d, o in zip((dp, dq), outs):
            # under 4 shared segments: the file's own spectrum keeps the fit
            # (and the IAF) running, n_seg 0 drops the condition's rows
            P = d[f"{c}_seg"][both][:, six].mean(0) if n >= 4 else d[c][cix]
            for k in ("", "_odd", "_even"):
                o[c + k] = P
            o[f"n_seg_{c}"] = n if n >= 4 else 0
    np.savez(out_p, **outs[0])
    np.savez(out_q, **outs[1])
    return counts


def segmatch(a):
    ref = os.path.join(a.work, SEG_PRIMARY)
    jobs, counts, check = [], [], {}
    for arm in ARMS:
        src = os.path.join(a.work, f"psd_{arm}")
        if not os.path.isdir(src):
            continue
        for pipe in ("primary", arm):
            os.makedirs(os.path.join(a.work, "segmatch", arm, pipe), exist_ok=True)
        for q in sorted(glob.glob(os.path.join(src, "*.npz"))):
            name = os.path.basename(q)
            p, stored = os.path.join(ref, name), os.path.join(a.psd_dir, name)
            if not os.path.exists(p):
                continue
            if name not in check and os.path.exists(stored):
                # the re-extraction should reproduce the stored primary spectra
                dp, ds = np.load(p, allow_pickle=True), np.load(stored, allow_pickle=True)
                ix = [list(ds["ch_names"]).index(c) for c in dp["roi_names"]]
                check[name] = float(np.nanmax(np.abs(dp["ec"][ix] / ds["ec"][ix] - 1)))
            outs = [os.path.join(a.work, "segmatch", arm, pipe, name) for pipe in ("primary", arm)]
            for c, n in matched(p, q, *outs).items():
                counts.append(dict(arm=arm, cond=c, n_primary=n[0], n_clean=n[1], n_both=n[2]))
            jobs += [(outs[0], "primary", arm), (outs[1], arm, arm)]
    v = np.array(list(check.values()))
    print(f"re-extracted vs stored primary, EC ROI: largest relative difference {v.max():.2g}"
          f" over {len(v)} participants")
    C = pd.DataFrame(counts)
    print("kept segments, median per arm and condition:")
    print(C[C.n_both >= 0].groupby(["arm", "cond"])[["n_primary", "n_clean", "n_both"]]
          .median().to_string())
    print(f"segmentations that differ: {(C.n_both < 0).sum()}; conditions under 4 shared"
          f" segments: {(C.n_both.between(0, 3)).sum()}")
    with Pool(a.workers) as pool:
        rs = pool.imap(fit_one, [(p, pipe) for p, pipe, _ in jobs], chunksize=4)
        F = pd.DataFrame([dict(r, arm=arm) for (_, _, arm), rr in zip(jobs, rs) for r in rr])
    F = F[F.n_seg > 0]
    F.to_csv(os.path.join(a.work, "segmatch", "fits.csv"), index=False)
    O = pd.concat([summarise(G, ["primary", arm], a.nboot).assign(arm=arm)
                   for arm, G in F.groupby("arm", sort=False)])
    O = O[["arm"] + [c for c in O.columns if c != "arm"]]
    O.to_csv(os.path.join(RES, "hbn_clean_check_segmatched.csv"), index=False)
    report(O, ("arm", "model", "cond"))


def summarise(F, order, nboot):
    """Per model and condition, on the participants fitted in every pipeline
    of ORDER: the age slopes, lambda*, and the paired differences from the
    primary pipeline."""
    from lambda_curve import effect_curve
    rows = []
    for (model, cond), G in F.groupby(["model", "cond"]):
        W = G.pivot_table(index=["subject", "age"], columns="pipeline", values=["a", "b", "tot"])
        pipes = [p for p in order if ("a", p) in W.columns]
        ok = ((W["a"][pipes] > 0).all(axis=1)) & W.notna().all(axis=1)
        W = W[ok]
        x = W.index.get_level_values("age").to_numpy()
        for pipe in pipes:
            la, lb = np.log(W[("a", pipe)].to_numpy()), np.log(W[("b", pipe)].to_numpy())
            r = effect_curve(la, lb, x=x, nboot=nboot, rng=np.random.default_rng(0))
            for meas, y in (("lambda=0", la), ("background ln b", lb), ("lambda=1", la - lb),
                            ("total ln T", np.log(W[("tot", pipe)].to_numpy()))):
                s, lo, hi = slope(y, x)
                rows.append(dict(model=model, cond=cond, pipeline=pipe, n=len(W), measure=meas,
                                 est=s, lo=lo, hi=hi))
            rows.append(dict(model=model, cond=cond, pipeline=pipe, n=len(W),
                             measure="crossover lambda*", est=r["lam_star"], lo=r["hdi"][0],
                             hi=r["hdi"][1], p_cross_in_01=r["p_cross_in_01"]))
        # paired: age slope and median of the per-participant difference
        for arm in [p for p in pipes if p != "primary"]:
            for meas, c in (("diff ln a", "a"), ("diff ln b", "b"), ("diff ln T", "tot")):
                dy = np.log(W[(c, arm)].to_numpy()) - np.log(W[(c, "primary")].to_numpy())
                s, lo, hi = slope(dy, x)
                rows.append(dict(model=model, cond=cond, pipeline=f"{arm} - primary", n=len(W),
                                 measure=meas + " (age slope)", est=s, lo=lo, hi=hi))
                rows.append(dict(model=model, cond=cond, pipeline=f"{arm} - primary", n=len(W),
                                 measure=meas + " (median)", est=np.median(dy), lo=np.nan,
                                 hi=np.nan))
    return pd.DataFrame(rows)


def report(O, keys=("model", "cond")):
    for k, G in O.groupby(list(keys), sort=False):
        head = ", ".join(v.upper() if key == "cond" else v for key, v in zip(keys, k))
        print(f"\n=== {head}, n = {G.n.iloc[0]}")
        for _, r in G.iterrows():
            raw = "lambda*" in r.measure or "median" in r.measure
            f = (lambda v: v) if raw else (lambda v: 100 * (np.exp(v) - 1))
            extra = (f"  P(in [0,1]) {r.p_cross_in_01:.2f}"
                     if "lambda*" in r.measure else "")
            print(f"  {r.pipeline:18s} {r.measure:26s} {f(r.est):+8.3f}"
                  + ("" if raw and "lambda*" not in r.measure else
                     f" [{f(r.lo):+.3f}, {f(r.hi):+.3f}]") + ("" if raw else " %/y") + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "download", "extract", "compare", "segmatch"])
    ap.add_argument("--work", default=os.environ.get("HBN_CLEAN_WORK", "hbn_clean_check"))
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--arm", choices=ARMS + ("primary",), default="gedai")
    ap.add_argument("--per-bin", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    dict(select=select, download=download, extract=extract, compare=compare,
         segmatch=segmatch)[a.step](a)


if __name__ == "__main__":
    main()
