"""Collect the re-tested published claims into one table.

Reads the group-level outputs of hbn_age_alpha_robust.py, aging_lambda.py,
vitaldb_lambda.py, chennu_analysis.py and brake_analysis.py and writes
results/breadth_summary.csv: for each claim, the effect under lambda = 0 and
lambda = 1 with 95% intervals, the crossover lambda* with its
Bayesian-bootstrap HDI, the posterior share of crossovers inside [0, 1] and
inside [0, 2], and two classifications.

verdict, by significance at the two endpoints: "holds under both" (both
intervals exclude zero, same sign), "reverses" (both exclude zero, opposite
signs), "depends on lambda" (only one excludes zero) or "null under both".

verdict_crossover, by where the crossover lies: "inside" (its 95% HDI lies
within [0, 1], so the sign of the effect depends on lambda), "outside" (the
HDI lies wholly outside [0, 1], so it does not), "overlaps" (the HDI straddles
0 or 1) or "unbounded" (the interval of s_b includes 0 and the crossover has
no interval).

The share inside [0, 2] is not in the group-level tables. It is recomputed
from the local per-participant files with the seed the analysis scripts use,
and kept only where the recomputed crossover equals the tabulated one.

results/breadth_summary_alt.csv holds the same claims under the alternative
background specification of each dataset: knee+plateau instead of the power
law (HBN, Dortmund, LEMON), a fixed 8-12 Hz band instead of the individual
one (VitalDB), and a 6-25 Hz instead of a 6-16 Hz censor window (Chennu).
The propofol-induction rows (Brake) have a single specification and are
repeated.

Usage: python breadth_summary.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
from lambda_curve import effect_curve

# background specification of each dataset in the primary and alternative tables
SPECS = {False: dict(model="fixed", vitaldb_band="iaf", chennu_window="censor 6-16"),
         True: dict(model="knee_plateau", vitaldb_band="8-12", chennu_window="censor 6-25")}
SPEC_LABEL = {"fixed": "power law", "knee_plateau": "knee + plateau",
              "iaf": "power law, IAF +/- 2 Hz", "8-12": "power law, 8-12 Hz",
              "censor 6-16": "power law, 6-16 Hz censored",
              "censor 6-25": "power law, 6-25 Hz censored"}
AGING = (("dortmund-cross ec_pre", "alpha falls with adult age, eyes closed",
          "Politanskaia et al. 2026; Yang et al. 2025", "Dortmund, 20-70 y", "ln per year"),
         ("dortmund-cross eo_pre", "alpha falls with adult age, eyes open", "",
          "Dortmund, 20-70 y", "ln per year"),
         ("lemon-group ec (older - younger)", "alpha lower in older adults, eyes closed",
          "Tröndle et al. 2023; Wilson et al. 2022", "LEMON, 20-35 vs 59-77 y", "ln"))
CHENNU = (("drowsy", "anteriorization (frontal - posterior)",
           "alpha shifts frontally in drowsy participants"),
          ("all", "frontal alpha", "frontal alpha rises with propofol sedation"),
          ("all", "posterior alpha", "posterior alpha falls with propofol sedation"))
BRAKE = (("delta", "delta does not rise before loss of consciousness"),
         ("alpha", "alpha rises before loss of consciousness"),
         ("beta", "beta rises before loss of consciousness"))


def row(domain, claim, source, dataset, n, r, unit):
    """One claim. lam_star_bounded is False when the interval of s_b, the effect
    on ln b, includes 0: the crossover is then unbounded and its HDI is not
    reported."""
    bounded = bool(r.get("lam_star_bounded", True))
    return dict(domain=domain, claim=claim, source=source, dataset=dataset, n=int(n),
                lam0=r["lam0"], lam0_lo=r["lam0_lo"], lam0_hi=r["lam0_hi"],
                lam1=r["lam1"], lam1_lo=r["lam1_lo"], lam1_hi=r["lam1_hi"],
                s_b_lo=r.get("s_b_lo", np.nan), s_b_hi=r.get("s_b_hi", np.nan),
                lam_star_bounded=bounded, lam_star=r["lam_star"],
                hdi_lo=r["hdi_lo"] if bounded else np.nan,
                hdi_hi=r["hdi_hi"] if bounded else np.nan,
                p_cross_in_01=r["p_cross_in_01"],
                p_cross_in_02=r.get("p_cross_in_02", np.nan), unit=unit)


def p_in(r, lo, hi):
    """Share of the Bayesian-bootstrap crossovers of an effect_curve result in [lo, hi]."""
    d = r["lam_star_draws"]
    return float(np.mean((d >= lo) & (d <= hi)))


def hbn_ages():
    """Age per participant from the PSD files in $HBN_OUT or, when they are
    not available, from results/hbn_roi.csv."""
    import glob
    ages = {}
    for p in glob.glob(os.path.join(os.environ.get("HBN_OUT", "hbn_psd"), "*.npz")):
        d = np.load(p, allow_pickle=True)
        ages[str(d["subject"])] = float(d["age"])
    if not ages and os.path.exists(os.path.join(RES, "hbn_roi.csv")):
        R = pd.read_csv(os.path.join(RES, "hbn_roi.csv"), usecols=["subject", "age"])
        ages = R.drop_duplicates("subject").set_index("subject").age.to_dict()
    return ages


def hbn_p01(model="fixed"):
    """Shares of Bayesian-bootstrap crossovers inside [0, 1] and [0, 2] for
    the HBN age slopes, recomputed from the local per-participant files (not
    in the repo; needs results/hbn_kp_fits.csv and the participants' ages)."""
    try:
        k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
        Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    except FileNotFoundError:
        return {}
    ages = hbn_ages()
    if not ages:
        return {}
    k = k[(k.model == model) & (k.window == "censor 6-16") & (k.band == "alpha") & (k.split == "full")]
    w = k.pivot_table(index="subject", columns="cond", values=["a", "b"])
    w.columns = [f"{v}_{c}" for v, c in w.columns]
    w["age"] = pd.Series(ages)
    w = w.loc[w.index.intersection(Q.index[Q.qc_ok])]
    out = {}
    for cond in ("ec", "eo"):
        d = w[w[f"a_{cond}"] > 0].dropna(subset=["age"])
        r = effect_curve(np.log(d[f"a_{cond}"]), np.log(d[f"b_{cond}"]), x=d.age,
                         rng=np.random.default_rng(0))
        out[cond] = dict(p_cross_in_01=r["p_cross_in_01"], p_cross_in_02=p_in(r, 0, 2),
                         s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
                         lam_star_bounded=r["lam_star_bounded"])
    return out


def local_curves(spec):
    """{claim: effect_curve result} for the claims outside HBN, rebuilt from
    the local per-participant files as aging_lambda.py, vitaldb_lambda.py,
    chennu_analysis.py and brake_analysis.py build them. A dataset whose file
    is missing is left out."""
    out = {}
    path = lambda name: os.path.join(RES, name)
    male = lambda D: (D.sex == "M").astype(float).to_numpy()
    if os.path.exists(path("aging_bandpower.csv")):
        T = pd.read_csv(path("aging_bandpower.csv"))
        F = T[(T.split == "full") & (T.model == spec["model"]) & ~T.bad & (T.a > 0)]
        for (contrast, claim, *_), cond in zip(AGING, ("ec_pre", "eo_pre", "ec")):
            if contrast.startswith("dortmund"):
                D = F[(F.study == "dortmund") & (F.session == 1) & (F.cond == cond)]
                x = D.age.to_numpy()
            else:
                D = F[(F.study == "lemon") & (F.cond == cond) & np.isfinite(F.age)]
                x = (D.age > 45).astype(float).to_numpy()
            out[claim] = effect_curve(np.log(D.a), np.log(D.b), x=x, covariates=male(D),
                                      rng=np.random.default_rng(0))
    if os.path.exists(path("vitaldb_cases.csv")):
        C = pd.read_csv(path("vitaldb_cases.csv"))
        for agent, dose in (("propofol", "ppf_ce"), ("sevoflurane", "sevo_et")):
            D = C[C.keep & (C.agent == agent)]
            cov = np.column_stack([male(D), D.rftn_ce.to_numpy(), D[dose].to_numpy()])
            A, B = (D[f"{v}_{spec['vitaldb_band']}_fixed"] for v in ("a", "b"))
            ok = (A > 0).to_numpy()
            out[f"frontal alpha falls with age under {agent}"] = effect_curve(
                np.log(A[ok]), np.log(B[ok]), x=D.age.to_numpy()[ok] / 10, covariates=cov[ok],
                rng=np.random.default_rng(0))
    if os.path.exists(path("chennu_spectra.csv")):
        T = pd.read_csv(path("chennu_spectra.csv"))
        F = T[(T.split == "full") & (T.window == spec["chennu_window"])]
        w = F.pivot_table(index=["subject", "drowsy"], columns=["roi", "level"],
                          values=["b", "a"]).reset_index()
        for grp, meas, claim in CHENNU:
            W = w if grp == "all" else w.loc[w.drowsy.to_numpy()]
            col = lambda v, roi, lev: W[(v, roi, lev)].to_numpy(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                la = {(r, l): np.log(np.where(col("a", r, l) > 0, col("a", r, l), np.nan))
                      for r in ("frontal", "posterior") for l in (1, 3)}
                lb = {(r, l): np.log(col("b", r, l))
                      for r in ("frontal", "posterior") for l in (1, 3)}
            if meas.startswith("anteriorization"):
                d = [(v[("frontal", 3)] - v[("posterior", 3)])
                     - (v[("frontal", 1)] - v[("posterior", 1)]) for v in (la, lb)]
            else:
                roi = meas.split()[0]
                d = [v[(roi, 3)] - v[(roi, 1)] for v in (la, lb)]
            out[claim] = effect_curve(*d, rng=np.random.default_rng(0))
    timing = os.path.join(
        os.environ.get("EEG_DATA", os.path.join(os.path.expanduser("~"), "eeg_data")),
        "brake2024", "source", "_data", "EEG_data", "data_time_information.csv")
    if os.path.exists(path("brake_bins.csv")) and os.path.exists(timing):
        # baseline: the minute before each patient's infusion onset
        B = pd.read_csv(path("brake_bins.csv"))
        T = pd.read_csv(timing)
        T.index = np.arange(1, len(T) + 1)
        onset = (T.infusion_onset - T.object_drop).reindex(B.patient).to_numpy()
        base = B[(B.t >= onset - 60) & (B.t < onset - 5)].groupby("patient").mean(numeric_only=True)
        pre = B[(B.t >= -60.0) & (B.t < -10.0)].groupby("patient").mean(numeric_only=True)
        d = pre.join(base, rsuffix="_0", how="inner")
        for band, claim in BRAKE:
            with np.errstate(invalid="ignore", divide="ignore"):
                dA = np.log(d[f"{band}_a"].where(d[f"{band}_a"] > 0)) - \
                    np.log(d[f"{band}_a_0"].where(d[f"{band}_a_0"] > 0))
            dB = np.log(d[f"{band}_b"]) - np.log(d[f"{band}_b_0"])
            out[claim] = effect_curve(dA.to_numpy(), dB.to_numpy(), rng=np.random.default_rng(0))
    return out


def classify(O):
    """Add the two classifications to a table of claims."""
    sig0 = np.sign(O.lam0_lo) == np.sign(O.lam0_hi)
    sig1 = np.sign(O.lam1_lo) == np.sign(O.lam1_hi)
    same = np.sign(O.lam0) == np.sign(O.lam1)
    O["verdict"] = np.where(sig0 & sig1 & same, "holds under both",
                            np.where(sig0 & sig1 & ~same, "reverses",
                                     np.where(sig0 | sig1, "depends on lambda", "null under both")))
    inside = (O.hdi_lo >= 0) & (O.hdi_hi <= 1)
    outside = (O.hdi_hi < 0) | (O.hdi_lo > 1)
    O["verdict_crossover"] = np.where(~O.lam_star_bounded, "unbounded",
                                      np.where(inside, "inside",
                                               np.where(outside, "outside", "overlaps")))
    return O


def build(alt=False):
    """The table of claims under the primary or the alternative specification."""
    spec = SPECS[alt]
    out, specs = [], []
    # HBN children (hbn_age_alpha_robust.py): curve endpoints from the robust table
    R = pd.read_csv(os.path.join(RES, "hbn_age_alpha_robust.csv"))
    C = pd.read_csv(os.path.join(RES, "hbn_age_alpha_crossover.csv"))
    p01s = hbn_p01(spec["model"])
    for cond, label in (("ec", "eyes closed"), ("eo", "eyes open")):
        g = R[(R["sample"] == "qc") & (R.model == spec["model"]) & (R.cond == cond)
              & (R.estimator == "ols")]
        c = C[(C["sample"] == "qc") & (C.model == spec["model"]) & (C.cond == cond)].iloc[0]
        r0, r1 = g[g.lam == 0].iloc[0], g[g.lam == 1].iloc[0]
        extra = p01s.get(cond, dict(p_cross_in_01=np.nan))
        out.append(row("development", f"alpha changes with age, {label}", "Tröndle et al. 2022",
                       "HBN, 5-22 y", r0.n,
                       dict(lam0=r0.est, lam0_lo=r0.lo, lam0_hi=r0.hi, lam1=r1.est, lam1_lo=r1.lo,
                            lam1_hi=r1.hi, lam_star=c.lam_star, hdi_lo=c.hdi_lo, hdi_hi=c.hdi_hi,
                            **extra), "ln per year"))
        specs.append(SPEC_LABEL[spec["model"]])
    A = pd.read_csv(os.path.join(RES, "aging_lambda.csv"))
    F = A[A.model == spec["model"]]
    for contrast, claim, source, dataset, unit in AGING:
        r = F[F.contrast == contrast]
        if len(r):
            out.append(row("adult aging", claim, source, dataset, r.n.iloc[0], r.iloc[0], unit))
            specs.append(SPEC_LABEL[spec["model"]])
    V = pd.read_csv(os.path.join(RES, "vitaldb_lambda.csv"))
    for agent in ("propofol", "sevoflurane"):
        r = V[(V.agent == agent) & (V.model == "fixed") & (V.band == spec["vitaldb_band"])
              & (V.adjust == "+ dose")]
        out.append(row("anaesthesia", f"frontal alpha falls with age under {agent}",
                       "Purdon et al. 2015; Boncompte et al. 2024", "VitalDB, 18-89 y",
                       r.n.iloc[0], r.iloc[0], "ln per decade"))
        specs.append(SPEC_LABEL[spec["vitaldb_band"]])
    X = pd.read_csv(os.path.join(RES, "chennu_lambda.csv"))
    X = X[X.window == spec["chennu_window"]]
    for grp, meas, claim in CHENNU:
        r = X[(X.group == grp) & (X.measure == meas)]
        out.append(row("sedation", claim, "Chennu et al. 2016", "propofol, baseline vs moderate",
                       r.n.iloc[0], r.iloc[0], "ln"))
        specs.append(SPEC_LABEL[spec["chennu_window"]])
    B = pd.read_csv(os.path.join(RES, "brake_lambda.csv"))
    if "baseline" in B:
        B = B[B.baseline == ("pre-infusion" if (B.baseline == "pre-infusion").any()
                             else "first 60 s")]
    for band, claim in BRAKE:
        r = B[(B.band == band) & (B.window == "pre-LOC")]
        out.append(row("anaesthesia", claim, "Brake et al. 2024", "propofol induction, Cz",
                       r.n.iloc[0], r.iloc[0], "ln"))
        specs.append("as in the primary table")
    O = pd.DataFrame(out)
    # share of crossovers in [0, 2], where the local files reproduce the tabulated crossover
    for claim, r in local_curves(spec).items():
        m = (O.claim == claim) & np.isclose(O.lam_star, r["lam_star"], rtol=1e-9, atol=0)
        O.loc[m, "p_cross_in_02"] = p_in(r, 0, 2)
    O = classify(O)
    if alt:
        O.insert(4, "specification", specs)
    return O


def main():
    pd.set_option("display.width", 250)
    for alt, name in ((False, "breadth_summary.csv"), (True, "breadth_summary_alt.csv")):
        O = build(alt)
        O.to_csv(os.path.join(RES, name), index=False)
        print(f"\n{name}")
        print(O[["claim", "dataset", "n", "lam0", "lam1", "lam_star", "hdi_lo", "hdi_hi",
                 "p_cross_in_01", "p_cross_in_02", "verdict", "verdict_crossover"]]
              .round(3).to_string(index=False))
        for col in ("verdict", "verdict_crossover"):
            print(f"  {col}: " + ", ".join(f"{k} {v}" for k, v in O[col].value_counts().items()))


if __name__ == "__main__":
    main()
