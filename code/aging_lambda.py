"""Adult aging and alpha power under each separation rule: Dortmund Vital
Study (ds005385) and MPI-Leipzig LEMON.

Input: the .npz files of dortmund_extract_psd.py and lemon_extract_psd.py.
Posterior ROI (O1, Oz, O2, PO3, POz, PO4) spectra per recording; the
aperiodic background fitted over 2-45 Hz (50 Hz mains) with 6-16 Hz
censored, as a power law or with a knee and plateau (ap_models, Whittle);
alpha band = individual alpha frequency +/- 2 Hz (from the eyes-closed
spectrum); a = total - b.

Quality control as for HBN: recordings with fewer than 10 clean segments,
alpha-band power more than 300 times from the sample median, or no alpha
peak (IAF at the edge of 6-14 Hz) are excluded.

Reported, as a function of lambda with the crossover lambda* (lambda_curve):
  dortmund-cross   age slope, session 1, before the task battery (+ sex)
  dortmund-long    5-year change within person (session 2 - session 1)
  lemon-group      older minus younger group
and, for the published specparam results, the same contrasts on specparam's
flattened alpha (mean of log10 P - log10 L over the band; lambda = 1) and on
total power.

Usage: python aging_lambda.py [--dortmund DIR] [--lemon DIR] [--workers 12] [--tag TAG]
Writes results/aging_bandpower.csv (per recording; local) and
results/aging_lambda.csv; --tag is appended to the names (_iclabel for the
cleaned spectra).
"""
import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ap_models import ap_eval, fit_aperiodic
from lambda_curve import band_power, effect_curve, peak_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROI = ["O1", "Oz", "O2", "PO3", "POz", "PO4"]
FIT = (2.0, 45.0)
CENSOR = ((6.0, 16.0),)
warnings.filterwarnings("ignore")


def records(path, study):
    """Yield (recording id, meta, cond, split, spectrum, freqs, n_seg, flag)."""
    d = np.load(path, allow_pickle=True)
    f = d["freqs"].astype(float)
    ch = [str(c) for c in d["ch_names"]]
    ix = [ch.index(c) for c in ROI if c in ch]
    if len(ix) < 3:
        return
    if study == "dortmund":
        meta = dict(study=study, subject=str(d["subject"]), session=int(d["session"]),
                    age=float(d["age"]), sex=str(d["sex"]), late=str(d["late"]))
        conds = [k for k in ("ec_pre", "eo_pre", "ec_post", "eo_post") if f"{k}_full" in d]
    else:
        meta = dict(study=study, subject=str(d["subject"]), session=1, age=float(d["age"]),
                    sex=str(d["sex"]), late="")
        conds = [k for k in ("ec", "eo") if f"{k}_full" in d]
    for c in conds:
        for split in ("full", "odd", "even"):
            yield meta, c, split, d[f"{c}_{split}"][ix].mean(0).astype(float), f, \
                int(d[f"{c}_n_seg"]), bool(d[f"{c}_flag_few"])


def one(args):
    path, study = args
    out = []
    recs = list(records(path, study))
    if not recs:
        return out
    # IAF from the eyes-closed full spectrum of this file
    ec = [r for r in recs if r[1].startswith("ec") and r[2] == "full"]
    iaf = peak_frequency(ec[0][3], ec[0][4]) if ec else np.nan
    if not np.isfinite(iaf):
        iaf_use, edge = 10.0, True
    else:
        iaf_use, edge = iaf, (iaf <= 6.25 or iaf >= 13.75)
    for meta, cond, split, P, f, nseg, few in recs:
        for model in ("fixed", "knee_plateau"):
            bp = band_power(P, f, (iaf_use - 2, iaf_use + 2), FIT, CENSOR, model)
            if bp is None:
                continue
            row = dict(meta, cond=cond, split=split, model=model, iaf=iaf, iaf_edge=edge,
                       n_seg=nseg, few=few, **bp)
            if model == "fixed" and split == "full":
                # specparam-like flattened alpha and a fixed high-alpha band (Yang 2025)
                sel = (f >= FIT[0]) & (f <= FIT[1])
                keep = np.ones(sel.sum(), bool)
                for lo, hi in CENSOR:
                    keep &= ~((f[sel] >= lo) & (f[sel] <= hi))
                fit = fit_aperiodic(np.clip(P[sel], 1e-12, None), f[sel], keep, "fixed")
                L = ap_eval(fit["theta"], f[sel], "fixed")
                flat = np.log10(np.clip(P[sel], 1e-12, None)) - np.log10(L)
                ff = f[sel]
                row["flat_alpha"] = float(np.mean(flat[(ff >= iaf_use - 2) & (ff <= iaf_use + 2)]))
                hb = (ff >= 10) & (ff <= 13)
                row["tot_hi"], row["b_hi"] = float(np.mean(P[sel][hb])), float(np.mean(L[hb]))
                row["sp_peak"] = specparam_alpha(P, f)
            out.append(row)
    return out


def specparam_alpha(P, f):
    """Power (log10 height over the aperiodic fit) of the largest 7-14 Hz peak,
    specparam fixed mode 3-40 Hz, peak widths 1-8 Hz, threshold 2 (the
    settings of Politanskaia et al. 2026); nan if there is no alpha peak."""
    from specparam import SpectralModel
    try:
        sm = SpectralModel(peak_width_limits=[1, 8], peak_threshold=2,
                           aperiodic_mode="fixed", verbose=False)
        sm.fit(f, P, [3, 40])
        pk = np.atleast_2d(sm.get_params("peak"))
        if pk.size == 0 or not np.all(np.isfinite(pk)):
            return np.nan
        j = np.where((pk[:, 0] >= 7) & (pk[:, 0] <= 14))[0]
        return float(pk[j[np.argmax(pk[j, 1])], 1]) if j.size else np.nan
    except Exception:
        return np.nan


def qc(T):
    F = T[(T.split == "full") & (T.model == "fixed")]
    key = ["study", "subject", "session", "cond"]
    med = F.groupby(["study", "cond"]).tot.transform("median")
    bad = (F.tot < med / 300) | (F.tot > med * 300) | F.few | F.iaf_edge
    flags = F.assign(bad=bad)[key + ["bad"]]
    return T.merge(flags, on=key, how="left")


def contrast(D, name, y_a, y_b, x=None, cov=None, extra=None):
    r = effect_curve(y_a, y_b, x=x, covariates=cov, rng=np.random.default_rng(0))
    row = dict(contrast=name, n=r["n"], s_a=r["s_a"], s_b=r["s_b"], s_b_lo=r["s_b_ci"][0],
               s_b_hi=r["s_b_ci"][1], lam_star_bounded=r["lam_star_bounded"],
               lam_star=r["lam_star"],
               ci_lo=r["ci"][0], ci_hi=r["ci"][1], hdi_lo=r["hdi"][0], hdi_hi=r["hdi"][1],
               p_cross_in_01=r["p_cross_in_01"],
               lam0=r["curve"][0][0], lam0_lo=r["curve"][0][1], lam0_hi=r["curve"][0][2],
               lam1=r["curve"][-1][0], lam1_lo=r["curve"][-1][1], lam1_hi=r["curve"][-1][2])
    row.update(extra or {})
    return row


def slope_ci(y, x, cov=None):
    import statsmodels.api as sm
    X = np.column_stack([x] + ([cov] if cov is not None else []))
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    r = sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type="HC3")
    return float(r.params[1]), float(r.conf_int()[1][0]), float(r.conf_int()[1][1]), \
        float(r.pvalues[1]), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dortmund", default=os.environ.get("DORTMUND_OUT", "dortmund_psd"))
    ap.add_argument("--lemon", default=os.environ.get("LEMON_OUT", "lemon_psd"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    jobs = [(p, "dortmund") for p in sorted(glob.glob(os.path.join(a.dortmund, "*.npz")))] + \
           [(p, "lemon") for p in sorted(glob.glob(os.path.join(a.lemon, "*.npz")))]
    with Pool(a.workers) as pool:
        T = pd.DataFrame([r for rs in pool.imap(one, jobs, chunksize=4) for r in rs])
    T = qc(T)
    T.to_csv(os.path.join(RES, f"aging_bandpower{a.tag}.csv"), index=False)
    out = []
    for model in ("fixed", "knee_plateau"):
        F = T[(T.split == "full") & (T.model == model) & ~T.bad & (T.a > 0)]
        # Dortmund cross-sectional, session 1 before the task battery
        for cond in ("ec_pre", "eo_pre"):
            D = F[(F.study == "dortmund") & (F.session == 1) & (F.cond == cond)]
            if len(D) > 30:
                sex = (D.sex == "M").astype(float).to_numpy()
                out.append(contrast(D, f"dortmund-cross {cond}", np.log(D.a), np.log(D.b),
                                    x=D.age.to_numpy(), cov=sex,
                                    extra=dict(model=model, per="year")))
        # Dortmund longitudinal 5-year change, EC pre
        D = F[(F.study == "dortmund") & (F.cond == "ec_pre")]
        w = D.pivot_table(index="subject", columns="session", values=["a", "b"]).dropna()
        if len(w) > 20:
            out.append(contrast(w, "dortmund-long ec_pre (ses2 - ses1)",
                                np.log(w[("a", 2)]) - np.log(w[("a", 1)]),
                                np.log(w[("b", 2)]) - np.log(w[("b", 1)]),
                                extra=dict(model=model, per="5 years")))
        # LEMON older minus younger
        for cond in ("ec", "eo"):
            D = F[(F.study == "lemon") & (F.cond == cond) & np.isfinite(F.age)]
            if len(D) > 30:
                old = (D.age > 45).astype(float).to_numpy()
                sex = (D.sex == "M").astype(float).to_numpy()
                out.append(contrast(D, f"lemon-group {cond} (older - younger)", np.log(D.a),
                                    np.log(D.b), x=old, cov=sex, extra=dict(model=model, per="group")))
    # published-style measures (power law only)
    F = T[(T.split == "full") & (T.model == "fixed") & ~T.bad]
    pub = []
    for study, cond, xname in (("dortmund", "ec_pre", "age"), ("lemon", "ec", "older")):
        D = F[(F.study == study) & (F.cond == cond)]
        if study == "dortmund":
            D = D[D.session == 1]
            x = D.age.to_numpy()
        else:
            D = D[np.isfinite(D.age)]
            x = (D.age > 45).astype(float).to_numpy()
        if len(D) < 30:
            continue
        for meas, y in (("total alpha (ln)", np.log(D.tot)),
                        ("specparam flattened alpha (log10)", D.flat_alpha),
                        ("specparam alpha peak power (log10)", D.sp_peak),
                        ("high alpha 10-13 Hz, lambda = 0 (ln)",
                         np.log((D.tot_hi - D.b_hi).where(D.tot_hi > D.b_hi))),
                        ("high alpha 10-13 Hz, lambda = 1 (ln)",
                         np.log(((D.tot_hi - D.b_hi) / D.b_hi).where(D.tot_hi > D.b_hi)))):
            s, lo, hi, p, n = slope_ci(np.asarray(y, float), x)
            pub.append(dict(study=study, cond=cond, predictor=xname, measure=meas,
                            est=s, lo=lo, hi=hi, p=p, n=n))
    # Dortmund 5-year change within person (Politanskaia et al. 2026)
    from scipy import stats
    D = F[(F.study == "dortmund") & (F.cond == "ec_pre")]
    for meas in ("sp_peak", "flat_alpha", "tot", "iaf", "exponent"):
        w = D.pivot_table(index="subject", columns="session", values=meas).dropna()
        if len(w) > 20:
            y = (np.log(w[2]) - np.log(w[1])) if meas == "tot" else (w[2] - w[1])
            t = stats.ttest_1samp(y, 0)
            pub.append(dict(study="dortmund", cond="ec_pre", predictor="session 2 - 1",
                            measure=meas, est=float(y.mean()), lo=float(y.mean() - 1.96 * y.std(ddof=1) / np.sqrt(len(y))),
                            hi=float(y.mean() + 1.96 * y.std(ddof=1) / np.sqrt(len(y))),
                            p=float(t.pvalue), n=len(y)))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(RES, f"aging_lambda{a.tag}.csv"), index=False)
    pd.DataFrame(pub).to_csv(os.path.join(RES, f"aging_published{a.tag}.csv"), index=False)
    pd.set_option("display.width", 220)
    print(T.groupby(["study", "cond"]).apply(lambda g: pd.Series(dict(
        n=g.subject.nunique(), bad=int(g[(g.split == "full") & (g.model == "fixed")].bad.sum())))))
    print(O.round(3).to_string(index=False))
    print(pd.DataFrame(pub).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
