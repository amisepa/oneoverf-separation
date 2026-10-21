"""The organoid maturation claim as a row of the re-tested-claims table.

Claim (Trujillo et al. 2019, Fig. 2H): 2.5-4.5 Hz oscillatory power of
cortical organoids rises up to the 25th week in culture. Window: weeks 9-17
since the first recording (17-25 weeks in culture; see trujillo_maturation.py).
Effect: the mean over wells of each well's least-squares slope over weeks of
ln a and ln b (ln per week). The row has the columns of
results/breadth_summary.csv (breadth_summary.py), so it can be appended there.

Resampling. Wells share each recording session (their week-to-week
fluctuations correlate at 0.3-0.9), so wells and recording days are resampled
together: intervals of the effects at lambda = 0 and 1 and of s_b from a
two-way pairs bootstrap, the crossover's HDI and the shares of crossovers in
[0, 1] and [0, 2] from a two-way Bayesian bootstrap (Dirichlet weights on wells,
and on days as weights of each well's regression).

Specifications: primary knee + plateau on the flanks (kp_flanks: the one
that passes the empty-band checks and recovers injected rhythms without
bias), alternative the paper's own per-channel robust line ("published").

The 8 wells are four experiments in duplicate, with the pairing undocumented:
results/trujillo_breadth_pairings.csv repeats the row for each of the 105
pairings, with the 4 pair means as units. With 4 units a percentile
bootstrap is too narrow, so t intervals on the unit slopes (n - 1 degrees of
freedom) and verdict_t are reported next to the bootstrap verdict.

The period after the turning point (weeks 17-33, 25-41 weeks in culture; the
paper: "tapering off slightly ... plateauing") is computed as a second row for
the Supplementary Note; it is a claim of no change and is not meant for the
table.

Writes results/trujillo_breadth.csv and results/trujillo_breadth_pairings.csv.

Usage: python trujillo_breadth.py [--nboot 4000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trujillo_maturation import RES, hdi, pairings  # noqa: E402

CLAIMS = (("9-17", "2.5-4.5 Hz oscillatory power rises to the 25th week in culture",
           "cortical organoids, 17-25 weeks in culture"),
          ("17-33", "2.5-4.5 Hz oscillatory power plateaus after the 25th week",
           "cortical organoids, 25-41 weeks in culture"))
SPEC_LABEL = {"kp_flanks": "knee + plateau", "published": "published robust line, per channel"}


def arrays(t):
    """Wells x days arrays of ln a, ln b (nan where missing or a <= 0)."""
    wells = np.array(sorted(t.well.unique()))
    days = np.array(sorted(t.week.unique()))
    la = np.full((wells.size, days.size), np.nan)
    lb = np.full_like(la, np.nan)
    wi = {w: i for i, w in enumerate(wells)}
    di = {d: j for j, d in enumerate(days)}
    for w, d, a, b in zip(t.well, t.week, t.a, t.b):
        if a > 0:
            la[wi[w], di[d]] = np.log(a)
            lb[wi[w], di[d]] = np.log(b)
    return wells, days, la, lb


def well_slopes(days, la, lb, w_day=None):
    """Weighted least-squares slope of ln a and ln b on weeks, per well."""
    out = np.full((la.shape[0], 2), np.nan)
    for i in range(la.shape[0]):
        ok = np.isfinite(la[i])
        if ok.sum() < 4:
            continue
        x, w = days[ok], (np.ones(ok.sum()) if w_day is None else w_day[ok])
        if np.sum(w > 0) < 3:
            continue
        xm = np.sum(w * x) / np.sum(w)
        sxx = np.sum(w * (x - xm) ** 2)
        for k, y in enumerate((la[i, ok], lb[i, ok])):
            ym = np.sum(w * y) / np.sum(w)
            out[i, k] = np.sum(w * (x - xm) * (y - ym)) / sxx
    return out


def unit_slopes(S, groups):
    """Mean slope per unit (a well, or a pair of wells)."""
    return np.array([np.nanmean(S[list(g)], 0) for g in groups])


def claim_row(t, groups, nboot, seed):
    wells, days, la, lb = arrays(t)
    idx = {w: i for i, w in enumerate(wells)}
    G = [[idx[w] for w in g if w in idx] for g in groups]
    U = unit_slopes(well_slopes(days, la, lb), G)
    s_a, s_b = np.nanmean(U, 0)
    rng = np.random.default_rng(seed)
    pairs, bayes = [], []
    for _ in range(nboot):
        # two-way pairs bootstrap: units and days with replacement
        gi = rng.integers(0, len(G), len(G))
        di = rng.integers(0, days.size, days.size)
        S = well_slopes(days[di], la[:, di], lb[:, di])
        pairs.append(np.nanmean(unit_slopes(S, [G[k] for k in gi]), 0))
        # two-way Bayesian bootstrap: Dirichlet weights on units and on days
        wd = rng.dirichlet(np.ones(days.size)) * days.size
        S = unit_slopes(well_slopes(days, la, lb, wd), G)
        wu = rng.dirichlet(np.ones(len(G)))
        ok = np.all(np.isfinite(S), 1)
        bayes.append((wu[ok] / wu[ok].sum()) @ S[ok])
    P, B = np.array(pairs), np.array(bayes)
    pct = lambda v: tuple(float(x) for x in np.nanpercentile(v, [2.5, 97.5]))
    lam0_ci, lam1_ci, sb_ci = pct(P[:, 0]), pct(P[:, 0] - P[:, 1]), pct(P[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = B[:, 0] / B[:, 1]
    bounded = bool(sb_ci[0] > 0 or sb_ci[1] < 0)
    h = hdi(cross)
    # t intervals on the unit slopes: with 4 units a percentile bootstrap is too narrow
    tq = stats.t.ppf(0.975, len(G) - 1)
    tci = {}
    for lam in (0, 1):
        v = U[:, 0] - lam * U[:, 1]
        v = v[np.isfinite(v)]
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
        tci[f"lam{lam}_t_lo"], tci[f"lam{lam}_t_hi"] = float(m - tq * se), float(m + tq * se)
    return dict(n=len(G), **tci, lam0=float(s_a), lam0_lo=lam0_ci[0], lam0_hi=lam0_ci[1],
                lam1=float(s_a - s_b), lam1_lo=lam1_ci[0], lam1_hi=lam1_ci[1],
                s_b=float(s_b), s_b_lo=sb_ci[0], s_b_hi=sb_ci[1], lam_star_bounded=bounded,
                lam_star=float(s_a / s_b), hdi_lo=h[0] if bounded else np.nan,
                hdi_hi=h[1] if bounded else np.nan,
                p_cross_in_01=float(np.mean((cross >= 0) & (cross <= 1))),
                p_cross_in_02=float(np.mean((cross >= 0) & (cross <= 2))))


def verdict(r, kind=""):
    sig0 = np.sign(r[f"lam0{kind}_lo"]) == np.sign(r[f"lam0{kind}_hi"])
    sig1 = np.sign(r[f"lam1{kind}_lo"]) == np.sign(r[f"lam1{kind}_hi"])
    same = np.sign(r["lam0"]) == np.sign(r["lam1"])
    if sig0 and sig1:
        return "holds under both" if same else "reverses"
    return "depends on lambda" if (sig0 or sig1) else "null under both"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--nboot-pairings", type=int, default=500)
    a = ap.parse_args()
    bp = pd.read_csv(os.path.join(RES, "trujillo_bandpower_2.5-4.5.csv"))
    rows, prow = [], []
    for spec in ("kp_flanks", "published"):
        T = bp[bp.spec == spec]
        wells = sorted(T.well.unique())
        for k, (win, claim, dataset) in enumerate(CLAIMS):
            w0, w1 = (float(v) for v in win.split("-"))
            t = T[(T.week >= w0) & (T.week <= w1)]
            r = claim_row(t, [[w] for w in wells], a.nboot, seed=k)
            rows.append(dict(domain="organoid maturation", claim=claim, source="Trujillo et al. 2019",
                             dataset=dataset, spec=SPEC_LABEL[spec], window=win, **r,
                             unit="ln per week", verdict=verdict(r), verdict_t=verdict(r, "_t"),
                             share_a_nonpos=float((t.a <= 0).mean())))
            for j, pr in enumerate(pairings(wells)):
                q = claim_row(t, pr, a.nboot_pairings, seed=1000 + j)
                prow.append(dict(spec=spec, window=win, pairing=j, pairs=str(pr), **q,
                                 verdict=verdict(q), verdict_t=verdict(q, "_t")))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "trujillo_breadth.csv"), index=False)
    P = pd.DataFrame(prow)
    P.to_csv(os.path.join(RES, "trujillo_breadth_pairings.csv"), index=False)
    cols = ["spec", "window", "n", "lam0", "lam0_lo", "lam0_hi", "lam1", "lam1_lo", "lam1_hi",
            "s_b_lo", "s_b_hi", "lam_star", "hdi_lo", "hdi_hi", "p_cross_in_01", "p_cross_in_02",
            "verdict", "verdict_t"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[cols].round(3).to_string(index=False))
        print("\nAcross the 105 pairings (4 units each):")
        g = P.groupby(["spec", "window"])
        print(pd.concat([g.p_cross_in_01.median().rename("p01_median"),
                         g.p_cross_in_01.min().rename("p01_min"),
                         g.p_cross_in_01.max().rename("p01_max"),
                         g.lam_star.median().rename("lam_star_median")], axis=1).round(3).to_string())
        print(pd.crosstab([P.spec, P.window], P.verdict).to_string())
        print(pd.crosstab([P.spec, P.window], P.verdict_t).to_string())


if __name__ == "__main__":
    main()
