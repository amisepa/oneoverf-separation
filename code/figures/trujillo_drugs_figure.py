"""Drug sessions of the Trujillo et al. (2019) organoids as controls.

  a  a silenced well: spectrum before and after (4-s Welch, mean over channels)
  b  100-200 Hz power after / before against spikes after / before, all wells
  c  a well with the GABA-A-block signature (events up): event spectra before
     and after (3-s windows)
  d  rhythm per event (a_E / b_E, knee + plateau) before and after, per well,
     by group (wells without events after, or with a knee fit that absorbed
     the rhythm, are left out)

Inputs: results/trujillo_drugs.csv and the extracted spectra (local).

Usage: python trujillo_drugs_figure.py OUTDIR [--silenced 170303,10] [--up 170207,8]
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
from make_figures import GREY, INK2, MM, RES, W2, panel, save  # noqa: E402
from trujillo_events import classify  # noqa: E402
from trujillo_maturation import good_channels  # noqa: E402

GROUP_COL = {"silenced": "#2a78d6", "reduced": "#1baf7a", "events kept": GREY,
             "events up": "#e34948"}
PRE, POST = "#9fc3ee", "#1d5aa6"


def load(date, kind, drug):
    folder = f"trujillo_{kind}" + ("_drugs" if drug else "")
    return np.load(os.path.join(RES, folder, f"{date}{'_Drugs' if drug else ''}.npz"))


def silenced_panel(ax, date, well):
    for drug, col, lab in ((False, PRE, "before"), (True, POST, "after")):
        d = load(date, "psd", drug)
        w = well - 1
        P = d[f"w{w}"][good_channels(d, w)].mean(0)
        f = d["f"]
        s = (f >= 0.5) & (f <= 200)
        ax.loglog(f[s], P[s] * 1e12, color=col, lw=1.0, label=lab)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"Silenced well ({date}, well {well})", loc="left", color=INK2)


def floor_panel(ax, t):
    for g, x in t.groupby("group"):
        ax.loglog(x.spike_ratio.clip(lower=1e-3), x.post_hf_100_200 / x.pre_hf_100_200, "o",
                  color=GROUP_COL[g], ms=3.5, label=g)
    ax.axhline(1, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("Spikes after / before")
    ax.set_ylabel("100-200 Hz power after / before")
    ax.legend(frameon=False, loc="lower right", fontsize=5.5)
    ax.set_title("High-frequency floor", loc="left", color=INK2)


def event_panel(ax, date, well):
    for drug, col, lab in ((False, PRE, "before"), (True, POST, "after")):
        d = load(date, "events", drug)
        w = well - 1
        ev, _, _, _ = classify(d[f"pop{w}"], d[f"win{w}"])
        f = d["f"]
        E = d[f"S{w}"][ev].mean(0)
        s = (f >= 0.5) & (f <= 45)
        ax.loglog(f[s], E[s] * 1e12, color=col, lw=1.0, label=f"{lab} ({len(ev)} events)")
    ax.axvspan(2.5, 4.5, color=GREY, alpha=0.15, lw=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Event spectrum (µV²/Hz)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"Events up ({date}, well {well})", loc="left", color=INK2)


def paired_panel(ax, t):
    x = t.dropna(subset=["pre_ab_event", "post_ab_event"])
    order = ["events up", "events kept"]
    for i, g in enumerate(order):
        y = x[x.group == g]
        for _, r in y.iterrows():
            ax.plot([i - 0.15, i + 0.15], [r.pre_ab_event, r.post_ab_event], "-o",
                    color=GROUP_COL[g], ms=2.5, lw=0.8, alpha=0.8)
    ax.axhline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}\n(before → after)" for g in order], fontsize=6)
    ax.set_ylim(-0.6, 3)
    ax.set_ylabel("Rhythm per event, a_E / b_E")
    ax.set_title("Is the per-event measure specific?", loc="left", color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--silenced", default="170303,10")
    ap.add_argument("--up", default="170207,8")
    a = ap.parse_args()
    t = pd.read_csv(os.path.join(RES, "trujillo_drugs.csv"), dtype={"date": str})
    fig, ax = plt.subplots(2, 2, figsize=(W2 * 0.75, 110 * MM))
    sd, sw = a.silenced.split(",")
    ud, uw = a.up.split(",")
    silenced_panel(ax[0, 0], sd, int(sw))
    floor_panel(ax[0, 1], t)
    event_panel(ax[1, 0], ud, int(uw))
    paired_panel(ax[1, 1], t)
    for k, x in zip("abcd", ax.ravel()):
        panel(x, k)
    fig.tight_layout()
    save(fig, a.outdir, "figS_organoid_drugs")


if __name__ == "__main__":
    main()
