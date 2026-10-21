"""Drug sessions of the Trujillo et al. (2019) organoids as controls.

Three recordings (2016-12-17, 2017-02-07, 2017-03-03; files *_Drugs.mat) were
made after drugs were applied to wells 5-12; the regular recording of the same
day is taken as the pre-treatment baseline (assumed: the files do not say).
The paper lists bicuculline, picrotoxin, muscimol, CNQX + AP5, baclofen and TTX
(n = 4 wells each). Which well received which drug is not documented, so the
wells are grouped by what the drug did, not by its name:

    silenced     spikes after < 5% of before: TTX, or a blocker that stops the
                 network (the paper: CNQX + AP5, baclofen and muscimol extinguish
                 network events, TTX abolishes all activity)
    events up    at least 1.5 times as many network events after (and 4 or
                 more): the paper's signature of GABA-A block (bicuculline and
                 picrotoxin increase network events; bicuculline abolishes the
                 nested oscillation but not the events)
    events kept  2 or more network events after, not "events up"
    reduced      the rest

Two questions:
  1. Is the high-frequency floor neural? In silenced wells the spectrum after
     the drug is the floor without spiking. Compare 100-200 Hz and 30-45 Hz
     power (4-s Welch, median over channels) before and after.
  2. Is the rhythm-per-event measure specific? In wells whose events survive,
     a_E / b_E (knee + plateau on the event spectrum, as in trujillo_events.py)
     before and after; the nested oscillation should be gone where
     bicuculline was applied. Event-spectrum fits whose knee absorbed the
     rhythm (exponent at its bound or knee frequency >= 2.5 Hz) give no a_E / b_E;
     two measures that need no knee are reported for every well: a / b over the
     paper's robust line, and a model-free peak ratio (band power over the
     geometric mean of 1.5-2 and 5-6.5 Hz).
     Caveats: the groups are defined by an outcome (event counts) of the same
     recordings; drug identities are unknown; the paper ran its pharmacology
     on 6-month organoids, and the 2017 sessions are at 8-9 months.

  3. Is the quiet-window spectrum neural? In silenced wells, the spectrum
     after the drug (3-s windows, all of them) over the same well's quiet
     windows before, per band: about 1 means quiet periods carry nothing above
     what remains without spiking.

Inputs: results/trujillo_psd/, results/trujillo_events/ (regular recordings)
and results/trujillo_psd_drugs/, results/trujillo_events_drugs/ (drug files;
trujillo_extract_psd.py and trujillo_extract_events.py with --out).
Writes results/trujillo_drugs.csv.

Usage: python trujillo_drugs.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trujillo_events import SPECS as EV_SPECS, classify, robust_line_terms  # noqa: E402
from trujillo_maturation import RES, SPECS, ap_eval, band_terms, fit_aperiodic, good_channels  # noqa: E402

DATES = ("161217", "170207", "170303")
QUIET_BANDS = {"0.5-2": (0.5, 2.0), "2.5-4.5": (2.5, 4.5), "6-20": (6.0, 20.0),
               "30-45": (30.0, 45.0), "61-100": (61.0, 100.0)}
BAND = (2.5, 4.5)


def band_mean(P, f, lo, hi):
    s = (f >= lo) & (f <= hi)
    return float(P[..., s].mean(-1)) if P.ndim == 1 else P[:, s].mean(1)


def kp_ratio(P, f, spec):
    """a / b in BAND, b, and whether the knee absorbed the rhythm (exponent at
    its bound or knee inside the band; then a / b is not returned)."""
    keep = np.zeros(f.size, bool)
    for lo, hi in spec["bins"]:
        keep |= (f >= lo) & (f <= hi)
    fit = fit_aperiodic(P[keep], f[keep], None, spec["model"])
    tot, b, p, a = band_terms(P, f, fit, BAND, spec["model"])
    absorbed = bool(fit["exponent"] >= 5.9 or fit["knee_freq"] >= BAND[0])
    return (np.nan if absorbed else a / b), b, absorbed


def well_row(date, w, psd, ev, tag):
    f4 = psd["f"]
    chans = psd[f"w{w}"][good_channels(psd, w)]
    P4 = chans.mean(0)
    pop, S, starts = ev[f"pop{w}"], ev[f"S{w}"], ev[f"win{w}"]
    f3 = ev["f"]
    events, quiet, thr, _ = classify(pop, starts)
    ab_avg, b_avg, _ = kp_ratio(P4, f4, SPECS["kp_flanks"])
    row = {f"{tag}_nspk": int(ev[f"nspk{w}"]), f"{tag}_events": len(events),
           f"{tag}_quiet": float(quiet.mean()),
           f"{tag}_hf_100_200": float(np.median(band_mean(chans, f4, 100, 200))),
           f"{tag}_hf_30_45": float(np.median(band_mean(chans, f4, 30, 45))),
           f"{tag}_band_tot": band_mean(P4, f4, *BAND), f"{tag}_ab_avg": ab_avg}
    if len(events) >= 2:
        E = S[events].mean(0)
        r, b, absorbed = kp_ratio(E, f3, EV_SPECS["kp_flanks"])
        (tot_r, b_r, _, a_r), _ = robust_line_terms(E, f3, BAND)
        flank = np.sqrt(band_mean(E, f3, 1.5, 2.0) * band_mean(E, f3, 5.0, 6.5))
        row.update({f"{tag}_ab_event": r, f"{tag}_b_event": b, f"{tag}_absorbed": absorbed,
                    f"{tag}_ab_event_robust": a_r / b_r,
                    f"{tag}_peak_ratio": band_mean(E, f3, *BAND) / flank})
    if quiet.sum() >= 10:
        row[f"{tag}_q_band"] = band_mean(S[quiet].mean(0), f3, *BAND)
    spectra = {"all": S.mean(0)}
    if quiet.sum() >= 10:
        spectra["quiet"] = S[quiet].mean(0)
    for k, P in spectra.items():
        for lab, (lo, hi) in QUIET_BANDS.items():
            row[f"{tag}_{k}_{lab}"] = band_mean(P, f3, lo, hi)
    return row


def main():
    rows = []
    for date in DATES:
        need = [os.path.join(RES, d, f"{date}{x}.npz") for d, x in
                (("trujillo_psd", ""), ("trujillo_events", ""), ("trujillo_psd_drugs", "_Drugs"),
                 ("trujillo_events_drugs", "_Drugs"))]
        if not all(os.path.exists(n) for n in need):
            print(f"{date}: not all extracted, skipped")
            continue
        pre = (np.load(os.path.join(RES, "trujillo_psd", f"{date}.npz")),
               np.load(os.path.join(RES, "trujillo_events", f"{date}.npz")))
        post = (np.load(os.path.join(RES, "trujillo_psd_drugs", f"{date}_Drugs.npz")),
                np.load(os.path.join(RES, "trujillo_events_drugs", f"{date}_Drugs.npz")))
        for w in range(4, 12):
            r = dict(date=date, well=w + 1)
            r.update(well_row(date, w, *pre, "pre"))
            r.update(well_row(date, w, *post, "post"))
            rows.append(r)
    t = pd.DataFrame(rows)
    ratio = t.post_nspk / t.pre_nspk.clip(lower=1)
    t["spike_ratio"] = ratio
    up = t.post_events >= np.maximum(4, 1.5 * t.pre_events)
    t["group"] = np.where(ratio < 0.05, "silenced",
                          np.where(up, "events up", np.where(t.post_events >= 2, "events kept",
                                                             "reduced")))
    t.to_csv(os.path.join(RES, "trujillo_drugs.csv"), index=False)
    cols = ["date", "well", "group", "pre_nspk", "post_nspk", "pre_events", "post_events",
            "pre_hf_100_200", "post_hf_100_200", "pre_hf_30_45", "post_hf_30_45",
            "pre_ab_avg", "post_ab_avg", "pre_ab_event", "post_ab_event"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t[[c for c in cols if c in t]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nBy group (medians): post / pre")
    for g, x in t.groupby("group"):
        msg = [f"{g}: n = {len(x)}", f"spikes {np.median(x.post_nspk / x.pre_nspk):.3g}",
               f"100-200 Hz {np.median(x.post_hf_100_200 / x.pre_hf_100_200):.3g}",
               f"30-45 Hz {np.median(x.post_hf_30_45 / x.pre_hf_30_45):.3g}",
               f"time-averaged a/b {np.median(x.pre_ab_avg):.3g} -> {np.median(x.post_ab_avg):.3g}"]
        for col, lab in (("ab_event", "a_E/b_E knee"), ("ab_event_robust", "robust line"),
                         ("peak_ratio", "peak ratio")):
            if f"post_{col}" in x and x[f"post_{col}"].notna().any():
                y = x.dropna(subset=[f"pre_{col}", f"post_{col}"])
                if len(y):
                    msg.append(f"{lab} {np.median(y[f'pre_{col}']):.3g} -> "
                               f"{np.median(y[f'post_{col}']):.3g} (n = {len(y)})")
        print("  " + "; ".join(msg))
    sil = t[t.group == "silenced"]
    if len(sil):
        print()
        print("Silenced wells: spectrum after / quiet windows before, per band (median, range)")
        for lab in QUIET_BANDS:
            r = sil[f"post_all_{lab}"] / sil[f"pre_quiet_{lab}"]
            print(f"  {lab:8s} Hz  {r.median():.2f}  [{r.min():.2f}, {r.max():.2f}]")


if __name__ == "__main__":
    main()
