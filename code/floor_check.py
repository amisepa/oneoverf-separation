"""Recording-floor bracket for the re-tested claims of breadth_summary.py.

Why. A fitted aperiodic background b contains every aperiodic component of
the recording, neural or not. A floor (amplifier and electrode noise) that
is a large share of b in the band and does not change between the spectra
compared makes lambda = 1 look robust: dividing by b then mostly divides by
a constant (Trujillo organoids: floor 92-95% of b, lambda* 5.7-8.2 with it
in b, 0.85-1.9 with it modelled; trujillo_floor_model.py). The scalp, BIS
and Cz recordings here have no measured
floor reference, so nothing is subtracted: each claim is run with the
fitted plateau out of b and in b, and the plateau's share of the background
in the claim's band is reported.

Backgrounds, all with the samples, covariates and bootstrap seeds of
breadth_summary.py:
  (i)   primary   the background of results/breadth_summary.csv: a power law
                  (HBN, Dortmund, LEMON, VitalDB, Chennu), and for Brake the
                  eq6 model of Brake et al., whose flat term 10^mag e^offset
                  is a floor inside b
  (ii)  floor out knee+plateau, b = 10^b / (k + f^chi) without the plateau p,
                  a = tot - b - p (lambdacurve.band_power); Brake: eq6
                  without its flat term, a unchanged
  (iii) floor in  the same fit with b + p, a unchanged; for Brake (iii) = (i)
In (ii) and (iii) a is the same, so the samples (a > 0) are the same; only
the regressor ln b differs.

What the existing tables hold. code/lambda_curve.band_power (used by
aging_lambda.py, vitaldb_lambda.py) and hbn_kp.py take b from
ap_models.ap_band_power, which evaluates the whole model, plateau included,
and set a = tot - b. So in aging_bandpower.csv and vitaldb_cases.csv
tot - b - a = 0 to rounding and p cannot be read from them, and the
knee + plateau rows of breadth_summary_alt.csv (HBN, Dortmund, LEMON) are
background (iii), not (ii). Sources of p here:
  HBN       hbn_kp_fits.csv, column plateau; b without it = b - plateau,
            exact because the plateau is flat (no refit)
  Dortmund, LEMON, VitalDB
            knee+plateau refitted from the local PSD files with the band,
            fit range and censor of aging_lambda.py / vitaldb_lambda.py;
            b_neural + p must reproduce the table's b (printed)
  Chennu    knee+plateau fitted to the preprocessed .set files (spectra()
            of chennu_analysis.py, 1-40 Hz, 6-16 Hz censored, 8-15 Hz);
            the power-law b is refitted too, to check the spectra
  Brake     eq6 refitted with brake_analysis.fit_patient (same starts and
            seeds, so brake_bins.csv is reproduced), keeping each bin's
            parameters; about 5 min per patient
The claims themselves are run by breadth_summary's own code (local_curves,
hbn_p01) on substitute copies of its input tables in which a and b hold
the background in question.

Plateau share p / (b + p) in the claim's band, per condition, over the
quality-controlled spectra entering the claim (before the a > 0 cut):
median, 90th percentile, share of spectra above 10% (flagged when that is
more than 10% of them), and share in which the neural background is below
10% of the plateau (p / (b + p) > 0.91), the regime where the organoid
split-floor calibration could not estimate ln b (error s.d. > 0.3).
kp_knee_in_band: share of knee+plateau fits with exponent >= 5.9 or the
knee at or above the band's low edge (the knee may be absorbing the
rhythm, which inflates b and deflates the share). n_of_primary: sample
(a > 0) relative to (i); below one half the row is noted as not
interpretable (Chennu: the knee sits on the alpha band).

A fitted plateau is not a measured floor: it mixes amplifier noise, muscle
and spiking broadband, and its share is a bracket, not a measurement.

Usage: python floor_check.py [--jobs 14] [--cache DIR] [--skip brake,chennu]
  --cache  keep the refitted per-spectrum tables in DIR and reuse them
Writes results/floor_check.csv (one row per claim and background).
"""
import argparse
import contextlib
import glob
import os
import shutil
import sys
import tempfile
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lambdacurve", "src"))
import breadth_summary as BS
from lambdacurve import band_power as kp_band_power

warnings.filterwarnings("ignore")
EEG = os.environ.get("EEG_DATA", os.path.join(os.path.expanduser("~"), "eeg_data"))
KP = "knee_plateau"
SHARE_FLAG = 0.10          # plateau above 10% of the background in the band ...
FLAG_FRAC = 0.10           # ... in more than this share of the spectra
NEURAL_MIN = 0.10          # neural background below 10% of the plateau
MIN_KEPT = 0.5             # a > 0 in fewer than half the primary sample: no answer
CHENNU_FIT, CHENNU_CENSOR = (1.0, 40.0), ((6.0, 16.0),)    # as chennu_analysis.main
BG_LABEL = {"i": "(i) primary", "ii": "(ii) floor out", "iii": "(iii) floor in"}
SPEC = {"i": "power law", "ii": "knee+plateau, plateau out of b",
        "iii": "knee+plateau, plateau in b"}
SPEC_BRAKE = {"i": "eq6 incl. its flat term", "ii": "eq6 without its flat term",
              "iii": "eq6 incl. its flat term (= i)"}


def claim_list():
    """(claim, short label, dataset, band) in the order of breadth_summary.csv."""
    import brake_analysis as BA
    import chennu_analysis as CA
    out = [(f"alpha changes with age, {lab}", f"HBN {c.upper()}, age", "hbn", "IAF+/-2")
           for c, lab in (("ec", "eyes closed"), ("eo", "eyes open"))]
    out += [(a[1], s, "aging", "IAF+/-2") for a, s in
            zip(BS.AGING, ("Dortmund EC, age", "Dortmund EO, age", "LEMON EC, old-young"))]
    out += [(f"frontal alpha falls with age under {g}", f"VitalDB {g[:4]}, age", "vitaldb",
             "IAF+/-2") for g in ("propofol", "sevoflurane")]
    band = "{:g}-{:g}".format(*CA.ALPHA)
    out += [(c[2], s, "chennu", band) for c, s in
            zip(BS.CHENNU, ("Chennu anterior., drowsy", "Chennu frontal", "Chennu posterior"))]
    out += [(c[1], f"Brake {b}, pre-LOC", "brake", "{:g}-{:g}".format(*BA.BANDS[b]))
            for b, c in zip(("delta", "alpha", "beta"), BS.BRAKE)]
    return out


# ---------------------------------------------------------------- refits

def kp_fit(P, f, band, fit_range, censor):
    """Knee+plateau band power with the plateau split out, or None."""
    r = kp_band_power(P, f, band, fit_range, censor, model=KP)
    if r is None:
        return None
    return dict(tot=r["tot"], b_neural=r["b"], plateau=r["plateau"], a_kp=r["a"],
                exponent=r["exponent"], knee_freq=float(r["knee_freq"]), band_lo=band[0])


_IAF = {}


def _set_iaf(iaf):
    _IAF.update(iaf)


def _aging_file(args):
    """Knee+plateau refit of the claim conditions of one Dortmund/LEMON file,
    band IAF +/- 2 Hz with the IAF of aging_bandpower.csv (10 Hz if none)."""
    import aging_lambda as AL
    path, study, conds = args
    out = []
    for meta, cond, split, P, f, *_ in AL.records(path, study):
        key = (meta["study"], meta["subject"], meta["session"], cond)
        if split != "full" or key not in _IAF or cond not in conds:
            continue
        cf = _IAF[key] if np.isfinite(_IAF[key]) else 10.0
        r = kp_fit(P, f, (cf - 2, cf + 2), AL.FIT, AL.CENSOR)
        if r is not None:
            out.append(dict(study=key[0], subject=key[1], session=key[2], cond=cond, **r))
    return out


def _vitaldb_case(args):
    """Knee+plateau refit of one VitalDB case (mean of the two channels, as
    vitaldb_lambda.one), band IAF +/- 2 Hz."""
    import vitaldb_lambda as VL
    path, caseid, iaf = args
    d = np.load(path, allow_pickle=True)
    f, P = d["freqs"].astype(float), d["full"].astype(float).mean(0)
    cf = iaf if np.isfinite(iaf) else 10.0
    r = kp_fit(P, f, (cf - 2, cf + 2), VL.FIT, VL.CENSOR)
    return None if r is None else dict(caseid=caseid, **r)


def _chennu_file(args):
    """Knee+plateau and power-law fits of one Chennu file, both regions."""
    import chennu_analysis as CA
    from lambda_curve import band_power
    path, subject, level = args
    f, S, _ = CA.spectra(path)
    out = []
    for roi, parts in S.items():
        P = parts["full"]
        r = kp_fit(P, f, CA.ALPHA, CHENNU_FIT, CHENNU_CENSOR)
        fx = band_power(P, f, CA.ALPHA, CHENNU_FIT, CHENNU_CENSOR, "fixed")
        if r is not None:
            out.append(dict(subject=subject, level=level, roi=roi, b_fixed=fx["b"], **r))
    return out


def _brake_patient(args):
    """eq6 refit of one Brake patient through brake_analysis.fit_patient, with
    fit_bin wrapped to keep each bin's parameters. eq6 is
    L = 10^mag (e^offset + x2(f)): the flat term 10^mag e^offset is the floor."""
    import brake_analysis as BA
    fit_bin, params = BA.fit_bin, []

    def keep(*a, **k):
        r = fit_bin(*a, **k)
        params.append(r[0])
        return r

    BA.fit_bin = keep
    try:
        rows, _ = BA.fit_patient(args)
    finally:
        BA.fit_bin = fit_bin
    return [dict(patient=row["patient"], t=row["t"], floor=10.0 ** x[3] * np.exp(x[2]),
                 **{f"{b}_b_refit": row[f"{b}_b"] for b in BA.BANDS})
            for row, x in zip(rows, params)]


def pooled(fn, jobs, n, **kw):
    with Pool(n, **kw) as pool:
        out = pool.map(fn, jobs, chunksize=1)
    return [r for rs in out if rs is not None for r in (rs if isinstance(rs, list) else [rs])]


def cached(name, cache, make):
    """Refit table `name`, from the cache directory when it is there."""
    path = os.path.join(cache, f"floor_refit_{name}.csv") if cache else None
    if path and os.path.exists(path):
        return pd.read_csv(path)
    T = make()
    if path and T is not None:
        os.makedirs(cache, exist_ok=True)
        T.to_csv(path, index=False)
    return T


def refit_aging(n):
    A = pd.read_csv(os.path.join(RES, "aging_bandpower.csv"))
    K = A[(A.model == KP) & (A.split == "full")
          & (((A.study == "dortmund") & (A.session == 1) & A.cond.isin(["ec_pre", "eo_pre"]))
             | ((A.study == "lemon") & (A.cond == "ec")))]
    iaf = {(r.study, r.subject, r.session, r.cond): r.iaf for r in K.itertuples()}
    jobs = []
    for study, env, conds in (("dortmund", "DORTMUND_OUT", ("ec_pre", "eo_pre")),
                              ("lemon", "LEMON_OUT", ("ec",))):
        d = os.environ.get(env, os.path.join(EEG, f"{study}_psd"))
        jobs += [(p, study, conds) for p in sorted(glob.glob(os.path.join(d, "*.npz")))]
    if not jobs:
        return None
    return pd.DataFrame(pooled(_aging_file, jobs, n, initializer=_set_iaf, initargs=(iaf,)))


def refit_vitaldb(n):
    C = pd.read_csv(os.path.join(RES, "vitaldb_cases.csv"))
    d = os.environ.get("VITALDB_OUT", os.path.join(EEG, "vitaldb_psd"))
    jobs = [(os.path.join(d, f"case{int(r.caseid):04d}.npz"), r.caseid, r.iaf)
            for r in C[C.keep].itertuples()]
    jobs = [j for j in jobs if os.path.exists(j[0])]
    return pd.DataFrame(pooled(_vitaldb_case, jobs, n)) if jobs else None


def refit_chennu(n):
    import scipy.io as sio
    d = os.path.join(EEG, "chennu2016", "Sedation-RestingState")
    if not os.path.exists(os.path.join(d, "datainfo.mat")):
        return None
    m = sio.loadmat(os.path.join(d, "datainfo.mat"), squeeze_me=True)["datainfo"]
    info = pd.DataFrame(m, columns=["file", "level", "conc", "rt", "hits"])
    jobs = [(os.path.join(d, r.file + ".set"), int(r.file[:2]), int(r.level))
            for r in info.itertuples() if int(r.level) in (1, 3)]
    return pd.DataFrame(pooled(_chennu_file, jobs, n))


def refit_brake(n):
    z = os.path.join(EEG, "brake2024", "spectrogram_Cz_all_subjects.zip")
    if not os.path.exists(z):
        return None
    return pd.DataFrame(pooled(_brake_patient, [(pt, z, 5.0) for pt in range(1, 15)], n))


# ------------------------------------------------- claims on substitute tables

@contextlib.contextmanager
def inputs(subs):
    """Point breadth_summary at a directory holding its input tables, with
    the tables in subs ({file name: DataFrame}) replaced."""
    if not subs:
        yield
        return
    tmp = tempfile.mkdtemp(prefix="floor_check_")
    try:
        for name in ("aging_bandpower.csv", "vitaldb_cases.csv", "chennu_spectra.csv",
                     "brake_bins.csv", "hbn_kp_fits.csv", "hbn_qc_flags.csv"):
            if name in subs:
                subs[name].to_csv(os.path.join(tmp, name), index=False)
            elif os.path.exists(os.path.join(RES, name)):
                shutil.copy(os.path.join(RES, name), tmp)
        if os.path.exists(os.path.join(RES, "hbn_roi.csv")):     # ages (hbn_ages)
            pd.read_csv(os.path.join(RES, "hbn_roi.csv"), usecols=["subject", "age"]) \
                .drop_duplicates("subject").to_csv(os.path.join(tmp, "hbn_roi.csv"), index=False)
        res, BS.RES = BS.RES, tmp
        try:
            yield
        finally:
            BS.RES = res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def curves(model, subs):
    """{claim: effect_curve result} from breadth_summary's code, HBN, Dortmund
    and LEMON under `model`, on the substitute tables."""
    with inputs(subs):
        out = BS.local_curves(dict(model=model, vitaldb_band="iaf", chennu_window="censor 6-16"))
        calls, effect_curve = [], BS.effect_curve
        BS.effect_curve = lambda *a, **k: calls.append(effect_curve(*a, **k)) or calls[-1]
        try:
            BS.hbn_p01(model)                  # eyes closed, then eyes open
        finally:
            BS.effect_curve = effect_curve
    out.update(zip([f"alpha changes with age, {c}" for c in ("eyes closed", "eyes open")], calls))
    return out


def summary(r):
    c = r["curve"]
    ok = r["lam_star_bounded"]
    return dict(n=r["n"], lam0=c[0][0], lam0_lo=c[0][1], lam0_hi=c[0][2], lam1=c[-1][0],
                lam1_lo=c[-1][1], lam1_hi=c[-1][2], s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
                lam_star_bounded=ok, lam_star=r["lam_star"],
                hdi_lo=r["hdi"][0] if ok else np.nan, hdi_hi=r["hdi"][1] if ok else np.nan,
                p_cross_in_01=r["p_cross_in_01"])


# ------------------------------------------------------------ plateau shares

def shares(H, A, R_aging, V, R_vit, R_ch, Ch, B, R_br):
    """{claim: DataFrame(cond, share, kp_bad)} over the QC'd spectra entering
    each claim, for the datasets whose plateau is available."""
    out = {}
    bad = lambda R: (R.exponent >= 5.9) | (R.knee_freq >= R.band_lo)
    if H is not None:
        Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
        k = H[(H.model == KP) & (H.window == "censor 6-16") & (H.band == "alpha")
              & (H.split == "full") & H.subject.isin(Q.index[Q.qc_ok])]
        for c, lab in (("ec", "eyes closed"), ("eo", "eyes open")):
            g = k[k.cond == c]
            out[f"alpha changes with age, {lab}"] = pd.DataFrame(dict(
                cond=c, share=g.plateau / g.b,
                kp_bad=(g.exponent >= 5.9) | (g.knee_freq >= g.iaf - 2)))
    if R_aging is not None:
        key = ["study", "subject", "session", "cond"]
        F = A[(A.model == KP) & (A.split == "full") & ~A.bad][key + ["age"]].merge(R_aging, on=key)
        F["share"], F["kp_bad"] = F.plateau / (F.b_neural + F.plateau), bad(F)
        for (contrast, claim, *_), cond in zip(BS.AGING, ("ec_pre", "eo_pre", "ec")):
            if contrast.startswith("dortmund"):
                g = F[(F.study == "dortmund") & (F.cond == cond)].assign(cond=cond)
            else:
                g = F[(F.study == "lemon") & (F.cond == cond) & np.isfinite(F.age)]
                g = g.assign(cond=np.where(g.age > 45, "older", "younger"))
            out[claim] = g[["cond", "share", "kp_bad"]]
    if R_vit is not None:
        F = V[V.keep][["caseid", "agent"]].merge(R_vit, on="caseid")
        for agent in ("propofol", "sevoflurane"):
            g = F[F.agent == agent]
            out[f"frontal alpha falls with age under {agent}"] = pd.DataFrame(dict(
                cond=agent, share=g.plateau / (g.b_neural + g.plateau), kp_bad=bad(g)))
    if R_ch is not None:
        drowsy = Ch.groupby("subject").drowsy.first()
        F = R_ch.assign(share=R_ch.plateau / (R_ch.b_neural + R_ch.plateau), kp_bad=bad(R_ch),
                        cond=R_ch.level.map({1: "baseline", 3: "moderate"}),
                        drowsy=R_ch.subject.map(drowsy))
        for grp, meas, claim in BS.CHENNU:
            g = F[F.drowsy] if grp == "drowsy" else F
            if not meas.startswith("anteriorization"):
                g = g[g.roi == meas.split()[0]]
            out[claim] = g[["cond", "share", "kp_bad"]]
    if R_br is not None:
        # windows of breadth_summary.local_curves: pre-infusion minute, -60 to -10 s
        T = pd.read_csv(os.path.join(EEG, "brake2024", "source", "_data", "EEG_data",
                                     "data_time_information.csv"))
        T.index = np.arange(1, len(T) + 1)
        F = B.merge(R_br[["patient", "t", "floor"]], on=["patient", "t"])
        onset = (T.infusion_onset - T.object_drop).reindex(F.patient).to_numpy()
        win = {"baseline": (F.t >= onset - 60) & (F.t < onset - 5),
               "pre-LOC": (F.t >= -60.0) & (F.t < -10.0)}
        for band, claim in BS.BRAKE:
            rows = []
            for w, m in win.items():
                g = F[m].groupby("patient")[["floor", f"{band}_b"]].mean()
                rows.append(pd.DataFrame(dict(cond=w, share=g.floor / g[f"{band}_b"])))
            out[claim] = pd.concat(rows).assign(kp_bad=np.nan)
    return out


def share_stats(S):
    if S is None or not len(S):
        return dict(share_med=np.nan, share_p90=np.nan, share_by_condition="",
                    frac_share_gt_10pct=np.nan, frac_neural_lt_10pct_of_plateau=np.nan,
                    frac_kp_knee_in_band=np.nan, n_spectra=0)
    by = "; ".join(f"{c} {100 * g.share.median():.1f}/{100 * g.share.quantile(0.9):.1f}%"
                   for c, g in S.groupby("cond", sort=False))
    return dict(share_med=S.share.median(), share_p90=S.share.quantile(0.9),
                share_by_condition=by, frac_share_gt_10pct=float((S.share > SHARE_FLAG).mean()),
                frac_neural_lt_10pct_of_plateau=float((S.share > 1 / (1 + NEURAL_MIN)).mean()),
                frac_kp_knee_in_band=float(S.kp_bad.astype(float).mean()), n_spectra=len(S))


# -------------------------------------------------------------------- main

def check(label, x, y):
    """Largest relative difference between a refit and the table."""
    d = np.nanmax(np.abs(np.asarray(x, float) / np.asarray(y, float) - 1))
    print(f"  {label}: max relative difference {d:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(14, os.cpu_count() or 1))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--skip", default="", help="comma list of hbn,aging,vitaldb,chennu,brake")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    read = lambda name: (pd.read_csv(os.path.join(RES, name))
                         if os.path.exists(os.path.join(RES, name)) else None)
    notes = {}
    avail = {d: {"i"} for d in ("hbn", "aging", "vitaldb", "chennu", "brake")}
    subs2, subs3 = {}, {}

    H = read("hbn_kp_fits.csv") if "hbn" not in skip else None
    if H is not None and "plateau" in H:
        H2 = H[(H.model == KP) & (H.window == "censor 6-16") & (H.band == "alpha")
               & (H.split == "full")].copy()
        H2["b"] = H2.b - H2.plateau
        subs2["hbn_kp_fits.csv"] = H2
        avail["hbn"] |= {"ii", "iii"}
    else:
        H, notes["hbn"] = None, "hbn_kp_fits.csv missing or skipped"

    A, R_aging = read("aging_bandpower.csv"), None
    if A is not None:
        avail["aging"].add("iii")                   # the table's b already holds b + p
        if "aging" not in skip:
            print("refitting Dortmund and LEMON (knee+plateau)", flush=True)
            R_aging = cached("aging", a.cache, lambda: refit_aging(a.jobs))
        if R_aging is not None and len(R_aging):
            key = ["study", "subject", "session", "cond"]
            M = A[(A.model == KP) & (A.split == "full")].merge(R_aging, on=key)
            check("Dortmund/LEMON b_neural + p vs table b", M.b_neural + M.plateau, M.b)
            check("Dortmund/LEMON a vs table a", M.a_kp, M.a)
            A2 = A.merge(R_aging[key + ["plateau"]], on=key, how="left")
            m = (A2.model == KP) & (A2.split == "full")
            need = m & ~A2.bad & (A2.a > 0) & (
                ((A2.study == "dortmund") & (A2.session == 1) & A2.cond.isin(["ec_pre", "eo_pre"]))
                | ((A2.study == "lemon") & (A2.cond == "ec") & np.isfinite(A2.age)))
            assert A2.plateau[need].notna().all(), "claim rows without a refit"
            A2.loc[m, "b"] = A2.b[m] - A2.plateau[m]
            subs2["aging_bandpower.csv"] = A2.drop(columns="plateau")
            avail["aging"].add("ii")
        else:
            R_aging, notes["aging"] = None, "Dortmund/LEMON PSD files missing or skipped"

    V, R_vit = read("vitaldb_cases.csv"), None
    if V is not None:
        V3 = V.copy()
        V3["a_iaf_fixed"], V3["b_iaf_fixed"] = V.a_iaf_knee_plateau, V.b_iaf_knee_plateau
        subs3["vitaldb_cases.csv"] = V3
        avail["vitaldb"].add("iii")
        if "vitaldb" not in skip:
            print("refitting VitalDB (knee+plateau)", flush=True)
            R_vit = cached("vitaldb", a.cache, lambda: refit_vitaldb(a.jobs))
        if R_vit is not None and len(R_vit):
            M = V.merge(R_vit, on="caseid")
            check("VitalDB b_neural + p vs table b", M.b_neural + M.plateau, M.b_iaf_knee_plateau)
            check("VitalDB a vs table a", M.a_kp, M.a_iaf_knee_plateau)
            V2 = V3.merge(R_vit[["caseid", "plateau"]], on="caseid", how="left")
            assert V2.plateau[V2.keep].notna().all(), "kept cases without a refit"
            V2["b_iaf_fixed"] = V2.b_iaf_fixed - V2.plateau
            subs2["vitaldb_cases.csv"] = V2.drop(columns="plateau")
            avail["vitaldb"].add("ii")
        else:
            R_vit, notes["vitaldb"] = None, "VitalDB PSD files missing or skipped"

    Ch, R_ch = read("chennu_spectra.csv"), None
    if Ch is not None and "chennu" not in skip:
        print("fitting Chennu (knee+plateau, from the .set files)", flush=True)
        R_ch = cached("chennu", a.cache, lambda: refit_chennu(a.jobs))
    if R_ch is not None and len(R_ch):
        F = Ch[(Ch.split == "full") & (Ch.window == "censor 6-16")]
        M = F.merge(R_ch, on=["subject", "level", "roi"], suffixes=("", "_kp"))
        check("Chennu power-law b refit vs table b", M.b_fixed, M.b)
        base = F[F.level.isin([1, 3])].drop(columns=["tot", "b", "a"]).merge(
            R_ch[["subject", "level", "roi", "tot", "b_neural", "plateau", "a_kp"]],
            on=["subject", "level", "roi"])
        subs2["chennu_spectra.csv"] = base.assign(a=base.a_kp, b=base.b_neural)
        subs3["chennu_spectra.csv"] = base.assign(a=base.a_kp, b=base.b_neural + base.plateau)
        avail["chennu"] |= {"ii", "iii"}
    else:
        R_ch, notes["chennu"] = None, "Chennu .set files missing or skipped"

    B, R_br = read("brake_bins.csv"), None
    if B is not None:
        avail["brake"].add("iii")                   # eq6's flat term is already in b
        if "brake" not in skip:
            print("refitting Brake (eq6, about 5 min per patient)", flush=True)
            R_br = cached("brake", a.cache, lambda: refit_brake(a.jobs))
        if R_br is not None and len(R_br):
            import brake_analysis as BA
            M = B.merge(R_br, on=["patient", "t"])
            for band in BA.BANDS:
                check(f"Brake {band} b refit vs table b", M[f"{band}_b_refit"], M[f"{band}_b"])
            B2 = B.merge(R_br[["patient", "t", "floor"]], on=["patient", "t"], how="left")
            assert B2.floor.notna().all(), "bins without a refit"
            for band in BA.BANDS:
                B2[f"{band}_b"] = B2[f"{band}_b"] - B2.floor
            subs2["brake_bins.csv"] = B2.drop(columns="floor")
            avail["brake"].add("ii")
        else:
            R_br, notes["brake"] = None, "Brake spectrogram zip missing or skipped"

    print("running the claims under (i), (ii), (iii)", flush=True)
    C = {"i": curves("fixed", {}), "iii": curves(KP, subs3), "ii": curves(KP, subs2)}
    S = shares(H, A, R_aging, V, R_vit, R_ch, Ch, B, R_br)
    tab = {False: read("breadth_summary.csv"), True: read("breadth_summary_alt.csv")}

    rows = []
    for claim, short, ds, band in claim_list():
        st = share_stats(S.get(claim))
        for bg in ("i", "ii", "iii"):
            if claim not in C[bg] or bg not in avail[ds]:
                continue
            # the tabulated crossover of the same specification, where there is one:
            # (i) is breadth_summary.csv; (iii) is breadth_summary_alt.csv for HBN,
            # Dortmund, LEMON (knee + plateau, plateau in b) and Brake (repeated)
            t = tab[bg == "iii"]
            t = t[t.claim == claim] if t is not None and bg != "ii" else []
            same = len(t) and (bg == "i" or ds == "brake"
                               or (ds in ("hbn", "aging") and "knee" in str(t.specification.iloc[0])))
            rows.append(dict(claim=claim, label=short, dataset=ds, band=band,
                             background=BG_LABEL[bg],
                             specification=(SPEC_BRAKE if ds == "brake" else SPEC)[bg],
                             **summary(C[bg][claim]),
                             table_lam_star=t.lam_star.iloc[0] if same else np.nan,
                             **st, note=notes.get(ds, "")))
    O = BS.classify(pd.DataFrame(rows))
    O["flag_plateau_gt_10pct"] = O.frac_share_gt_10pct > FLAG_FRAC
    # a knee that absorbs the rhythm leaves few spectra with a > 0
    O["n_of_primary"] = O.n / O.claim.map(O[O.background == BG_LABEL["i"]].set_index("claim").n)
    few = O.n_of_primary < MIN_KEPT
    O.loc[few, "note"] = (O.note[few] + " a > 0 in " + O.n[few].astype(str) + " of "
                          + (O.n[few] / O.n_of_primary[few]).round().astype(int).astype(str)
                          + " under knee+plateau: not interpretable").str.strip()
    v = O.pivot(index="claim", columns="background", values="verdict")
    if BG_LABEL["ii"] in v and BG_LABEL["iii"] in v:
        O["verdict_changes_ii_iii"] = O.claim.map(v[BG_LABEL["ii"]] != v[BG_LABEL["iii"]])
    O.to_csv(os.path.join(RES, "floor_check.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    first = O.drop_duplicates("claim")
    print("\nplateau share p / (b + p) in the band, median/90th pct per condition")
    print(first[["label", "band", "share_by_condition", "frac_share_gt_10pct",
                 "frac_neural_lt_10pct_of_plateau", "frac_kp_knee_in_band",
                 "flag_plateau_gt_10pct"]].round(3).to_string(index=False))
    W = O.assign(cell=lambda d: d.verdict.str.replace(" under both", "").str.replace(
        " on lambda", "") + " " + d.lam_star.map("{:.2f}".format) + " (P01 "
        + d.p_cross_in_01.map("{:.2f}".format) + ")")
    W = W.pivot(index="label", columns="background", values="cell")
    print("\nverdict, lambda* and P(lambda* in [0, 1]) per background")
    print(W.reindex(first.label).to_string())
    print("\n" + "\n".join(f"  {k}: {v}" for k, v in notes.items()))
    print(f"wrote {os.path.normpath(os.path.join(RES, 'floor_check.csv'))}")


if __name__ == "__main__":
    main()
