"""The package against the research scripts in ../code and the published
tables in ../results. Skipped when the package is used outside the
repository, and the table test also when the local per-participant files
(not distributed) are missing."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import lambdacurve as L

REPO = Path(__file__).resolve().parents[2]
CODE, RES = REPO / "code", REPO / "results"
LOCAL = [RES / f for f in ("aging_bandpower.csv", "vitaldb_cases.csv", "chennu_spectra.csv",
                           "brake_bins.csv")]
TIMING = Path(os.environ.get("EEG_DATA", Path.home() / "eeg_data")) / "brake2024" / "source" /     "_data" / "EEG_data" / "data_time_information.csv"
LOCAL.append(TIMING)
pytestmark = pytest.mark.skipif(not (CODE / "lambda_gmm.py").exists(),
                                reason="research code not found")


@pytest.fixture(scope="module")
def research():
    if str(CODE) not in sys.path:
        sys.path.append(str(CODE))
    import ap_models
    import lambda_curve
    import lambda_gmm
    return lambda_curve, ap_models, lambda_gmm


def test_lambda_curve_matches_effect_curve(research):
    rc = research[0]
    rng = np.random.default_rng(5)
    n = 120
    x = rng.uniform(5, 20, n)
    cov = np.column_stack([rng.integers(0, 2, n), rng.normal(size=n)])
    lb = 1 - 0.08 * x + rng.normal(0, 0.4, n)
    la = 0.5 - 0.04 * x + rng.normal(0, 0.5, n)
    for args, kw in (((la, lb), dict(x=x, covariates=cov)), ((la - lb * 0.3, lb[::-1]), {})):
        r0 = rc.effect_curve(*args, nboot=300, rng=np.random.default_rng(0), **kw)
        r1 = L.lambda_curve(*args, nboot=300, seed=0, **kw)
        assert (r0["n"], r0["s_a"], r0["s_b"], r0["lam_star"]) == (r1.n, r1.s_a, r1.s_b, r1.lam_star)
        assert tuple(r0["ci"]) == r1.ci and tuple(r0["hdi"]) == r1.hdi
        assert r0["p_cross_in_01"] == r1.p_cross_in_01
        assert np.array_equal(r0["curve"], r1.curve)
        assert tuple(r0["curve"][0]) == r1.lam0 and tuple(r0["curve"][-1]) == r1.lam1


def test_aperiodic_and_band_power_match(research):
    rc, ra = research[0], research[1]
    rng = np.random.default_rng(2)
    f = np.arange(1, 60.01, 0.25)
    S = 10 ** 2.0 / (4.0 ** 2.5 + f ** 2.5) + 0.01 + 1.5 * np.exp(-0.5 * ((f - 10) / 1.5) ** 2)
    P = S * rng.gamma(40, 1 / 40, f.size)
    keep = L.fit_mask(f, [(6, 16)])
    for model in L.MODELS:
        assert np.array_equal(ra.fit_aperiodic(P, f, keep, model)["theta"],
                              L.fit_aperiodic(P, f, keep, model)["theta"])
        b0 = rc.band_power(P, f, (8, 12), model=model)
        b1 = L.band_power(P, f, (8, 12), model=model)
        assert (b0["tot"], b0["a"], b0["exponent"]) == (b1["tot"], b1["a"], b1["exponent"])
        # the research b includes the plateau; here it is reported separately
        assert b0["b"] == pytest.approx(b1["b"] + b1["plateau"], rel=1e-12)


def test_log_free_estimators_match(research):
    rg = research[2]
    rng = np.random.default_rng(3)
    n = 80
    lb1 = rng.normal(0, 0.7, n)
    b1, b2 = np.exp(lb1), np.exp(lb1 + rng.normal(0.3, 0.3, n))
    c = np.exp(rng.normal(0, 0.3, n))
    noisy = lambda v: v[:, None] * rng.gamma(100, 0.01, (n, 2))
    t1, t2 = noisy(c * b1 ** 0.5 + b1), noisy(2 * c * b2 ** 0.5 + b2)
    B1, B2 = noisy(b1), noisy(b2)
    g0 = rg.bootstrap(t1, t2, B1, B2, nboot=40, rng=np.random.default_rng(0))
    g1 = L.coupling_two_conditions(t1, t2, B1, B2, nboot=40, seed=0)
    assert np.isfinite(g1.lam)
    assert (g0["lam"], g0["delta"], tuple(g0["ci"]), g0["boot_fail"]) == \
        (g1.lam, g1.delta, g1.ci, g1.boot_fail)
    grp = np.repeat(np.arange(10), 30)
    b = np.exp(rng.normal(0, 0.7, 10)[grp] + rng.normal(0, 0.3, grp.size))
    a = np.exp(rng.normal(0, 0.3, 10))[grp] * b ** 0.5
    T = (a + b) * rng.gamma(100, 0.01, b.size)
    B, Bz = b * rng.gamma(400, 1 / 400, b.size), b * rng.gamma(400, 1 / 400, b.size)
    e0 = rg.estimate_levels(T, B, Bz, grp)
    e1 = L.coupling_levels(T, B, Bz, grp)
    assert np.isfinite(e1.lam)
    assert e0["lam"] == e1.lam and e0["roots"] == e1.roots


# ---- rows of results/breadth_summary.csv from the local per-participant files

def rebuild_breadth_rows():
    """{claim: LambdaCurve} for every non-HBN row of breadth_summary.csv,
    rebuilt as aging_lambda.py, vitaldb_lambda.py, chennu_analysis.py and
    brake_analysis.py build them."""
    import pandas as pd
    out = {}
    T = pd.read_csv(RES / "aging_bandpower.csv")
    F = T[(T.split == "full") & (T.model == "fixed") & ~T.bad & (T.a > 0)]
    for cond, claim in (("ec_pre", "alpha falls with adult age, eyes closed"),
                        ("eo_pre", "alpha falls with adult age, eyes open")):
        D = F[(F.study == "dortmund") & (F.session == 1) & (F.cond == cond)]
        out[claim] = L.lambda_curve(np.log(D.a), np.log(D.b), x=D.age.to_numpy(),
                                    covariates=(D.sex == "M").astype(float).to_numpy())
    D = F[(F.study == "lemon") & (F.cond == "ec") & np.isfinite(F.age)]
    out["alpha lower in older adults, eyes closed"] = L.lambda_curve(
        np.log(D.a), np.log(D.b), x=(D.age > 45).astype(float).to_numpy(),
        covariates=(D.sex == "M").astype(float).to_numpy())

    C = pd.read_csv(RES / "vitaldb_cases.csv")
    for agent, dose in (("propofol", "ppf_ce"), ("sevoflurane", "sevo_et")):
        D = C[C.keep & (C.agent == agent)]
        cov = np.column_stack([(D.sex == "M").astype(float).to_numpy(), D.rftn_ce.to_numpy(),
                               D[dose].to_numpy()])
        A, B = D["a_iaf_fixed"], D["b_iaf_fixed"]
        ok = (A > 0).to_numpy()
        out[f"frontal alpha falls with age under {agent}"] = L.lambda_curve(
            np.log(A[ok]), np.log(B[ok]), x=D.age.to_numpy()[ok] / 10, covariates=cov[ok])

    T = pd.read_csv(RES / "chennu_spectra.csv")
    F = T[(T.split == "full") & (T.window == "censor 6-16")]
    w = F.pivot_table(index=["subject", "drowsy"], columns=["roi", "level"],
                      values=["tot", "b", "a", "rel_alpha"]).reset_index()
    for grp, claim, kind in (("drowsy", "alpha shifts frontally in drowsy participants", "ant"),
                             ("all", "frontal alpha rises with propofol sedation", "frontal"),
                             ("all", "posterior alpha falls with propofol sedation", "posterior")):
        W = w if grp == "all" else w.loc[w.drowsy.to_numpy()]
        col = lambda v, roi, lev: W[(v, roi, lev)].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            la = {(r, l): np.log(np.where(col("a", r, l) > 0, col("a", r, l), np.nan))
                  for r in ("frontal", "posterior") for l in (1, 3)}
            lb = {(r, l): np.log(col("b", r, l)) for r in ("frontal", "posterior") for l in (1, 3)}
        if kind == "ant":
            d = [(v[("frontal", 3)] - v[("posterior", 3)]) - (v[("frontal", 1)] - v[("posterior", 1)])
                 for v in (la, lb)]
        else:
            d = [v[(kind, 3)] - v[(kind, 1)] for v in (la, lb)]
        out[claim] = L.lambda_curve(*d, dropna=True)

    B = pd.read_csv(RES / "brake_bins.csv")
    # baseline: the minute before each patient's infusion onset (brake_analysis.py)
    T = pd.read_csv(TIMING)
    T.index = np.arange(1, len(T) + 1)
    onset = (T.infusion_onset - T.object_drop).reindex(B.patient).to_numpy()
    base = B[(B.t >= onset - 60) & (B.t < onset - 5)].groupby("patient").mean(numeric_only=True)
    pre = B[(B.t >= -60.0) & (B.t < -10.0)].groupby("patient").mean(numeric_only=True)
    d = pre.join(base, rsuffix="_0", how="inner")
    for band, claim in (("delta", "delta does not rise before loss of responsiveness"),
                        ("alpha", "alpha rises before loss of responsiveness"),
                        ("beta", "beta rises before loss of responsiveness")):
        with np.errstate(invalid="ignore", divide="ignore"):
            dA = np.log(d[f"{band}_a"].where(d[f"{band}_a"] > 0)) - \
                np.log(d[f"{band}_a_0"].where(d[f"{band}_a_0"] > 0))
        dB = np.log(d[f"{band}_b"]) - np.log(d[f"{band}_b_0"])
        out[claim] = L.lambda_curve(dA.to_numpy(), dB.to_numpy(), dropna=True)
    return out


COLS = ("lam0", "lam0_lo", "lam0_hi", "lam1", "lam1_lo", "lam1_hi", "lam_star", "hdi_lo",
        "hdi_hi", "p_cross_in_01")


@pytest.mark.skipif(not all(p.exists() for p in LOCAL), reason="local per-participant files missing")
def test_breadth_summary_rows():
    pd = pytest.importorskip("pandas")
    S = pd.read_csv(RES / "breadth_summary.csv").set_index("claim")
    rows = rebuild_breadth_rows()
    assert len(rows) == 11
    for claim, r in rows.items():
        want, got = S.loc[claim], r.to_dict()
        assert got["n"] == want["n"], claim
        assert got["verdict"] == want["verdict"], claim
        # breadth_summary leaves the HDI blank when the crossover is unbounded
        assert bool(r.lam_star_bounded) == bool(want["lam_star_bounded"]), claim
        cols = COLS if want["lam_star_bounded"] else tuple(c for c in COLS
                                                          if c not in ("hdi_lo", "hdi_hi"))
        np.testing.assert_allclose([got[c] for c in cols], [want[c] for c in cols],
                                   rtol=1e-12, atol=0, err_msg=claim)
