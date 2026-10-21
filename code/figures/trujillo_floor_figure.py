"""The recording floor in the Trujillo et al. (2019) organoid spectra.

  a  one well during the rise: time-averaged spectrum, its knee + plateau
     background, and the quiet-window spectrum (the recording floor)
  b  the floor's share of the fitted background over weeks, time-averaged and
     inside events (median over wells)
  c  the rise (weeks 9-17), time-averaged: slope of ln a - lambda ln b against
     lambda with the fitted background (floor included), the background minus
     the quiet floor, and the floor modelled (trujillo_floor_model.py)
  d  the same inside events
  e  split-floor calibration: robust s.d. of the error of ln b against the
     neural background as % of the floor, with the observed values in weeks
     9-17 marked (median b / floor: 3.6-4.5% time-averaged, 24-32% in events)
  f  change across each drug session against each well's other
     recording-to-recording changes (median ratio of sizes), per measure

Inputs: results/trujillo_maturation.csv, results/trujillo_floor.csv,
results/trujillo_floor_model_maturation.csv, _calibration.csv,
results/trujillo_events_maturation.csv,
results/trujillo_drug_days.csv, results/trujillo_bandpower_2.5-4.5.csv,
results/trujillo_events_bandpower.csv, results/trujillo_psd/ and
results/trujillo_events/ (local).

Usage: python trujillo_floor_figure.py OUTDIR [--date 161118] [--well 7]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from make_figures import C0, C1, GREY, INK2, MM, RES, W2, panel, save  # noqa: E402
from trujillo_events import classify  # noqa: E402
from trujillo_maturation import SPECS, ap_eval, fit_aperiodic, well_spectrum  # noqa: E402
from trujillo_floor import floors, with_neural  # noqa: E402

SPEC_COL = {"kp_flanks": "#2a78d6", "published": "#0b0b0b", "robust_line": "#0b0b0b"}
SPEC_LABEL = {"kp_flanks": "knee + plateau", "published": "published robust line"}


def spectrum_panel(ax, date, well):
    d = np.load(os.path.join(RES, "trujillo_psd", f"{date}.npz"))
    f = d["f"]
    P, _ = well_spectrum(d, well - 1)
    sp = SPECS["kp_flanks"]
    keep = np.zeros(f.size, bool)
    for lo, hi in sp["bins"]:
        keep |= (f >= lo) & (f <= hi)
    fit = fit_aperiodic(P[keep], f[keep], None, sp["model"])
    e = np.load(os.path.join(RES, "trujillo_events", f"{date}.npz"))
    _, quiet, _, _ = classify(e[f"pop{well - 1}"], e[f"win{well - 1}"])
    Q = e[f"S{well - 1}"][quiet].mean(0)
    s = (f >= 0.5) & (f <= 45)
    s3 = (e["f"] >= 0.5) & (e["f"] <= 45)
    ax.loglog(f[s], P[s] * 1e12, color=INK2, lw=1.0, label="time-averaged")
    ax.loglog(f[s], ap_eval(fit["theta"], f[s], sp["model"]) * 1e12, color=SPEC_COL["kp_flanks"],
              lw=0.9, ls="--", label="fitted background")
    ax.loglog(e["f"][s3], Q[s3] * 1e12, color=GREY, lw=1.0, label="quiet windows (floor)")
    ax.axvspan(2.5, 4.5, color=GREY, alpha=0.15, lw=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz)")
    ax.legend(frameon=False, loc="lower left", fontsize=5.5)
    ax.set_title(f"Well {well}, 20{date[:2]}-{date[2:4]}-{date[4:]}", loc="left", color=INK2)


def share_panel(ax):
    q, ev = floors()
    bp = pd.read_csv(os.path.join(RES, "trujillo_bandpower_2.5-4.5.csv"))
    for t, col, lab in ((bp[bp.spec == "kp_flanks"], INK2, "time-averaged"),
                        (ev[(ev.source == "E") & (ev.spec == "kp_flanks")], "#e34948",
                         "inside events")):
        m = with_neural(t, q)
        g = m.groupby("week").floor_share.median()
        ax.plot(g.index, g.values, "o-", color=col, ms=2, lw=0.9, label=lab)
    ax.axhline(1, color=INK2, lw=0.6, ls=":")
    ax.axvspan(9, 17, color=C0, alpha=0.07, lw=0)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Weeks since first recording")
    ax.set_ylabel("Floor / fitted background, 2.5-4.5 Hz")
    ax.legend(frameon=False, loc="lower left", fontsize=5.5)
    ax.set_title("The background is mostly floor", loc="left", color=INK2)


def curve_panel(ax, source):
    """Rise, weeks 9-17: lines per background definition (ols slopes)."""
    grid = np.linspace(0, 1, 21)
    fl = pd.read_csv(os.path.join(RES, "trujillo_floor.csv"))
    fm = pd.read_csv(os.path.join(RES, "trujillo_floor_model_maturation.csv"))
    if source == "timeavg":
        mat = pd.read_csv(os.path.join(RES, "trujillo_maturation.csv"))
        fitted = mat[(mat.spec == "kp_flanks") & (mat.window == "9-17") & (mat.method == "ols")]
        sub = fl[(fl.source == "timeavg") & (fl.spec == "kp_flanks")]
        src = "A"
    else:
        ev = pd.read_csv(os.path.join(RES, "trujillo_events_maturation.csv"))
        fitted = ev[(ev.spec == "kp_flanks") & (ev.window == "9-17") & (ev.method == "ols")
                    & (ev.variant == "event_vs_event_bg")]
        sub = fl[(fl.source == "event") & (fl.spec == "kp_flanks")]
        src = "E"
    sub = sub[(sub.window == "9-17") & (sub.method == "ols")]
    lines = [(fitted, GREY, "-", "fitted background (floor in)"),
             (sub, "#2a78d6", "--", "minus quiet floor")]
    for model, col in (("knee_plateau", "#1baf7a"), ("fixed_plateau", "#eb6834")):
        r = fm[(fm.source == src) & (fm.model == model) & (fm.floor_scale == "fixed")
               & (fm.window == "9-17") & (fm.method == "ols")]
        lines.append((r, col, "-", f"floor modelled, {'knee' if model.startswith('knee') else 'power law'}"))
    for t, col, ls, lab in lines:
        if t.empty:
            continue
        r = t.iloc[0]
        ax.plot(grid, r.s_a - grid * r.s_b, color=col, ls=ls, lw=1.1, label=lab)
    ax.axhline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("Assumed λ")
    ax.set_ylabel("Slope of ln a − λ ln b per week")
    ax.legend(frameon=False, loc="lower left", fontsize=5)
    ax.set_title(f"The rise (weeks 9-17), {'time-averaged' if source == 'timeavg' else 'inside events'}",
                 loc="left", color=INK2)


def calibration_panel(ax):
    c = pd.read_csv(os.path.join(RES, "trujillo_floor_model_calibration.csv"))
    g = c.groupby("r_L").lnb_err.agg(lambda v: float(v.quantile(0.75) - v.quantile(0.25)) / 1.349)
    ax.loglog(g.index * 100, g.values, "o-", color=INK2, ms=3, lw=1.0)
    ax.axvspan(3.6, 4.5, color=GREY, alpha=0.25, lw=0)
    ax.axvspan(24, 32, color="#e34948", alpha=0.18, lw=0)
    ax.text(4.0, 0.06, "time-\naveraged", ha="center", fontsize=5.5, color=INK2)
    ax.text(27.7, 0.06, "events", ha="center", fontsize=5.5, color="#e34948")
    ax.set_xlabel("Neural background, % of floor in the band")
    ax.set_ylabel("Error of ln b (robust s.d.)")
    ax.set_title("Split-floor calibration", loc="left", color=INK2)


def drug_panel(ax):
    t = pd.read_csv(os.path.join(RES, "trujillo_drug_days.csv"))
    order = ["event_rate", "n_spikes", "bg", "band_tot", "ab_event", "osc_pub", "floor"]
    labels = ["event rate", "spikes", "background", "band power", "a_E / b_E", "published", "floor"]
    weeks = sorted(t.drug_week.unique())
    cols = ["#e34948", "#eda100", GREY]
    for k, (wk, c) in enumerate(zip(weeks, cols)):
        g = t[t.drug_week == wk].groupby("measure").ratio_to_median.median().reindex(order)
        ax.bar(np.arange(len(order)) + (k - 1) * 0.27, g.values, width=0.25, color=c,
               label=f"session at week {wk:g}")
    ax.axhline(1, color=INK2, lw=0.6, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=5.5)
    ax.set_ylabel("Change across session / typical change")
    ax.legend(frameon=False, loc="upper right", fontsize=5.5)
    ax.set_title("Drug sessions break the series", loc="left", color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--date", default="161118")
    ap.add_argument("--well", type=int, default=7)
    a = ap.parse_args()
    fig, ax = plt.subplots(2, 3, figsize=(W2, 115 * MM))
    spectrum_panel(ax[0, 0], a.date, a.well)
    share_panel(ax[0, 1])
    curve_panel(ax[0, 2], "timeavg")
    curve_panel(ax[1, 0], "events")
    calibration_panel(ax[1, 1])
    drug_panel(ax[1, 2])
    for k, x in zip("abcdef", ax.ravel()):
        panel(x, k)
    fig.tight_layout()
    save(fig, a.outdir, "figS_organoid_floor")


if __name__ == "__main__":
    main()
