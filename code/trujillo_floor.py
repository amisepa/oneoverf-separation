"""Separation against the neural background, with the recording floor removed.

In the Trujillo et al. (2019) organoids the spectrum of quiet windows (no
network activity) equals the spectrum of the same well after its spiking was
silenced pharmacologically (trujillo_drugs.py): between events there is only
the recording floor (electrode and amplifier noise, and whatever the drugs do
not touch). A fitted aperiodic background b (+ plateau p) therefore contains
that floor N, and N does not change with maturation, so a lambda-curve on b
mostly tests the floor, and "holds under lambda = 1" becomes nearly automatic.
A rhythm cannot couple to electrode noise: the background that matters for
lambda is the neural part,

    b_neural = b + p - N,   N = band power of the quiet windows (3-s spectra)

per well and recording (the quiet windows of the same recording). Periodic
power a = tot - b - p is unchanged.

Reported:
  - the floor's share of the fitted background, N / (b + p), by phase, for the
    time-averaged spectra (4-s Welch, kp_flanks and published) and for the
    event spectra (3-s, kp_flanks and robust_line)
  - lambda-curves of the maturation slope with b_neural, as in
    trujillo_maturation.py (per-well slopes, two-way bootstrap, pairings);
    recordings with b_neural <= 0 have no ln b and are dropped (share reported)
  - the re-tested-claims row (trujillo_breadth.claim_row) with b_neural

Windows: 9-17 (the rise, before any drug session) and 20.5-33 (after the first
drug session, which treated all 8 wells on 2016-12-17, week 19.9; see
trujillo_drug_days.py).

Writes results/trujillo_floor.csv and results/trujillo_floor_breadth.csv.

Usage: python trujillo_floor.py [--nboot 4000] [--windows 9-17,20.5-33]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trujillo_maturation import RES, analyse  # noqa: E402
from trujillo_breadth import claim_row, verdict as row_verdict  # noqa: E402


def floors():
    ev = pd.read_csv(os.path.join(RES, "trujillo_events_bandpower.csv"))
    q = ev[(ev.source == "Q") & (ev.spec == "kp_flanks")][["date", "well", "tot", "ctl_tot"]]
    return q.rename(columns={"tot": "N", "ctl_tot": "N_ctl"}), ev


def with_neural(t, q):
    t = t.copy()
    t["date"] = pd.to_datetime(t.date).dt.strftime("%Y-%m-%d")
    q = q.assign(date=pd.to_datetime(q.date).dt.strftime("%Y-%m-%d"))
    m = t.merge(q, on=["date", "well"], how="inner")
    m["floor_share"] = m.N / (m.b + m.p)
    m["b"] = m.b + m.p - m.N
    m["ctl_b"] = m.ctl_b + m.ctl_p - m.N_ctl
    m["p"] = 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--windows", default="9-17,20.5-33")
    a = ap.parse_args()
    q, ev = floors()
    bp = pd.read_csv(os.path.join(RES, "trujillo_bandpower_2.5-4.5.csv"))
    sources = {("timeavg", "kp_flanks"): bp[bp.spec == "kp_flanks"],
               ("timeavg", "published"): bp[bp.spec == "published"],
               ("event", "kp_flanks"): ev[(ev.source == "E") & (ev.spec == "kp_flanks")],
               ("event", "robust_line"): ev[(ev.source == "E") & (ev.spec == "robust_line")]}
    rows, brows = [], []
    print("Floor share of the fitted background, N / (b + p), median by phase:")
    for (src, spec), t in sources.items():
        m = with_neural(t, q)
        ph = pd.cut(m.week, [-1, 9, 17, 20.5, 34], labels=["0-9", "9-17", "17-20.5", "20.5-33"])
        print(f"  {src:8s} {spec:12s}",
              m.groupby(ph, observed=True).floor_share.median().round(2).to_dict())
        for win in a.windows.split(","):
            w0, w1 = (float(v) for v in win.split("-"))
            x = m[(m.week >= w0) & (m.week <= w1)]
            nonpos = float((x.b <= 0).mean())
            xb = x[x.b > 0]
            for r in analyse(xb, xb.iloc[:0], spec, win, 5, a.nboot):
                r.update(source=src, b_neural_nonpos=nonpos)
                rows.append(r)
            if src == "timeavg":
                wells = sorted(xb.well.unique())
                r = claim_row(xb, [[w] for w in wells], a.nboot, seed=7)
                brows.append(dict(source=src, spec=spec, window=win, b_neural_nonpos=nonpos, **r,
                                  verdict=row_verdict(r), verdict_t=row_verdict(r, "_t")))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "trujillo_floor.csv"), index=False)
    B = pd.DataFrame(brows)
    B.to_csv(os.path.join(RES, "trujillo_floor_breadth.csv"), index=False)
    cols = ["source", "spec", "window", "method", "n_wells", "b_neural_nonpos", "s_b", "lam0",
            "lam0_2w_lo", "lam0_2w_hi", "lam1", "lam1_2w_lo", "lam1_2w_hi", "lam_star",
            "lam1_pairs_excl0", "verdict_2w"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[[c for c in cols if c in out]].round(3).to_string(index=False))
        print("\nRe-tested-claims row with the neural background:")
        print(B[["spec", "window", "b_neural_nonpos", "lam0", "lam0_lo", "lam0_hi", "lam1",
                 "lam1_lo", "lam1_hi", "lam1_t_lo", "lam1_t_hi", "s_b_lo", "s_b_hi", "lam_star",
                 "hdi_lo", "hdi_hi", "p_cross_in_01", "verdict", "verdict_t"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
