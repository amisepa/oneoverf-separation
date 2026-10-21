"""Do the drug sessions break the Trujillo et al. (2019) maturation series?

All 8 wells were treated on each drug-session day (2016-12-17, 2017-02-07,
2017-03-03; weeks 19.9, 27.3, 30.7 since the first recording), and the
regular recording of those days may be a baseline or a washout. If a session
changed the cultures, the series after it is no longer maturation alone.

For each measure, per well, the change between consecutive regular
recordings (the drug-day regular recordings left out) is computed in log
units; the change across each drug session (last recording before the drug
day to the first one after) is ranked among all changes of that well, and
its size compared with the median absolute change elsewhere. Measures: event
rate, spike count, rhythm per event a_E / b_E (knee + plateau on event
spectra, recordings with a_E > 0), time-averaged band power and fitted
background (knee + plateau), the published oscillatory measure, and the
quiet-window (floor) band power.

Writes results/trujillo_drug_days.csv.

Usage: python trujillo_drug_days.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trujillo_maturation import DRUG_DAYS, RES  # noqa: E402


def table():
    ct = pd.read_csv(os.path.join(RES, "trujillo_events_counts.csv"))
    ev = pd.read_csv(os.path.join(RES, "trujillo_events_bandpower.csv"))
    bp = pd.read_csv(os.path.join(RES, "trujillo_bandpower_2.5-4.5.csv"))
    key = ["date", "well"]
    for d in (ct, ev, bp):
        d["date"] = pd.to_datetime(d.date).dt.strftime("%Y-%m-%d")
    e = ev[(ev.source == "E") & (ev.spec == "kp_flanks")]
    e = e.assign(ab_event=np.where(e.a > 0, e.a / e.b, np.nan))[key + ["ab_event"]]
    q = ev[(ev.source == "Q") & (ev.spec == "kp_flanks")][key + ["tot"]].rename(columns={"tot": "floor"})
    k = bp[bp.spec == "kp_flanks"][key + ["week", "tot", "b"]].rename(columns={"tot": "band_tot", "b": "bg"})
    pub = bp[bp.spec == "published"][key + ["osc_pub"]]
    t = ct[key + ["event_rate", "n_spikes"]]
    for x in (e, q, k, pub):
        t = t.merge(x, on=key, how="left")
    return t


def main():
    t = table()
    measures = {"event_rate": True, "n_spikes": True, "ab_event": True, "band_tot": True,
                "bg": True, "floor": True, "osc_pub": False}
    reg = t[~t.date.isin(DRUG_DAYS)].sort_values(["well", "week"])
    drug_weeks = sorted(t[t.date.isin(DRUG_DAYS)].week.unique())
    rows = []
    for (w, g) in reg.groupby("well"):
        g = g.reset_index(drop=True)
        for col, log in measures.items():
            y = g[col].to_numpy(float)
            y = np.log(np.where(y > 0, y, np.nan)) if log else y
            d = np.diff(y)
            mid0, mid1 = g.week.to_numpy()[:-1], g.week.to_numpy()[1:]
            for dw in drug_weeks:
                i = np.where((mid0 < dw) & (mid1 > dw))[0]
                if i.size == 0 or not np.isfinite(d[i[0]]):
                    continue
                other = d[np.isfinite(d) & (np.arange(d.size) != i[0])]
                rows.append(dict(well=w, measure=col, drug_week=round(dw, 1), change=d[i[0]],
                                 rank=float(np.mean(np.abs(other) < abs(d[i[0]]))),
                                 ratio_to_median=abs(d[i[0]]) / np.median(np.abs(other))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "trujillo_drug_days.csv"), index=False)
    s = out.groupby(["drug_week", "measure"]).agg(
        median_change=("change", "median"), wells_up=("change", lambda v: int((v > 0).sum())),
        n=("change", "size"), median_rank=("rank", "median"),
        median_ratio=("ratio_to_median", "median"))
    print("Change across each drug session (log units; osc_pub in log10 units), against each well's")
    print("other recording-to-recording changes (rank: share of other changes smaller in size):")
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
