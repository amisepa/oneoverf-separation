"""Inside versus between network events (Trujillo et al. 2019 organoids).

  a  population spiking of one well and recording, with the event peaks and
     the quiet windows
  b  that recording's spectra inside events (E), in quiet windows (Q) and
     time-averaged (A)
  c  E, Q and A band power (2.5-4.5 Hz) over weeks, geometric mean over wells
  d  event rate over weeks (mean over wells)
  e  quiet windows as an empty band: a / b in 2.5-4.5 Hz over weeks, per
     background specification (median over wells)
  f  rhythm per event over weeks: a_E / b_E (modelled event background) and
     a_E / Q (quiet background), knee + plateau, median over wells (open
     markers: median a_E <= 0, no rhythm)

Inputs: results/trujillo_events_bandpower.csv, results/trujillo_events_counts.csv,
results/trujillo_events/*.npz (local).

Usage: python trujillo_events_figure.py OUTDIR [--date 161216] [--well 5]
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

SRC = {"E": ("inside events", "#e34948"), "Q": ("quiet", "#2a78d6"), "A": ("time-averaged", INK2)}
SPEC_LABEL = {"kp_flanks": "knee + plateau", "plateau_flanks": "power law + plateau",
              "fixed_0.5-20": "power law 0.5-20 Hz", "robust_line": "robust line (published)"}
SPEC_COL = {"kp_flanks": "#2a78d6", "plateau_flanks": "#eb6834", "fixed_0.5-20": "#4a3aa7",
            "robust_line": "#0b0b0b"}


def phases(ax):
    ax.axvspan(9, 17, color=C0, alpha=0.07, lw=0)
    ax.axvspan(17, 33, color=C1, alpha=0.07, lw=0)


def trace_panel(ax, d, w):
    pop, starts = d[f"pop{w}"], d[f"win{w}"]
    ev, quiet, thr, _ = classify(pop, starts)
    t = np.arange(pop.size) / 100.0
    ax.semilogy(t, pop + 1, color=INK2, lw=0.4)
    ax.axhline(thr, color=GREY, lw=0.6, ls="--")
    pk = (starts[ev] + 500) // 10
    ax.plot(pk / 100.0, pop[pk] + 1, "v", color=SRC["E"][1], ms=4)
    qs = starts[quiet] / 1000.0
    ax.plot(qs + 1.5, np.full(qs.size, 1.5), "|", color=SRC["Q"][1], ms=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population spiking (spikes/s)")
    ax.set_title("Events (▼) and quiet windows (|)", loc="left", color=INK2)
    return ev, quiet


def spectra_panel(ax, d, w, ev, quiet):
    f, S = d["f"], d[f"S{w}"]
    s = (f >= 0.5) & (f <= 45)
    for key, P in (("E", S[ev].mean(0)), ("Q", S[quiet].mean(0)), ("A", S.mean(0))):
        ax.loglog(f[s], P[s] * 1e12, color=SRC[key][1], lw=1.0, label=SRC[key][0])
    ax.axvspan(2.5, 4.5, color=GREY, alpha=0.15, lw=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz)")
    ax.legend(frameon=False, loc="lower left")


def band_panel(ax, bp):
    t = bp[bp.spec == "kp_flanks"]
    for key in ("E", "Q", "A"):
        g = t[t.source == key].groupby("week").tot.apply(lambda v: np.exp(np.log(v).mean()) * 1e12)
        ax.semilogy(g.index, g.values, "o-", color=SRC[key][1], ms=2, lw=0.9, label=SRC[key][0])
    phases(ax)
    ax.set_xlabel("Weeks since first recording")
    ax.set_ylabel("Band power, 2.5-4.5 Hz (µV²/Hz)")
    ax.legend(frameon=False, loc="upper left")


def rate_panel(ax, ct):
    g = ct.groupby("week").event_rate
    m, se = g.mean(), g.sem()
    ax.fill_between(m.index, m - se, m + se, color=GREY, alpha=0.3, lw=0)
    ax.plot(m.index, m, "o-", color=INK2, ms=2, lw=0.9)
    phases(ax)
    ax.set_xlabel("Weeks since first recording")
    ax.set_ylabel("Events per minute (mean ± s.e.)")


def empty_panel(ax, bp):
    q = bp[bp.source == "Q"].assign(ab=lambda x: x.a / x.b)
    for spec, col in SPEC_COL.items():
        g = q[q.spec == spec].groupby("week").ab.median()
        ax.plot(g.index, g.values, "o-", color=col, ms=2, lw=0.9, label=SPEC_LABEL[spec])
    ax.axhline(0, color=INK2, lw=0.6, ls=":")
    phases(ax)
    ax.set_xlabel("Weeks since first recording")
    ax.set_ylabel("a / b in quiet windows")
    ax.set_ylim(-0.1, 0.45)
    ax.legend(frameon=False, loc="upper left", fontsize=5, ncol=2)
    ax.set_title("Quiet windows: should be 0 at every age", loc="left", color=INK2)


def per_event_panel(ax, bp):
    e = bp[(bp.source == "E") & (bp.spec == "kp_flanks")]
    q = bp[(bp.source == "Q") & (bp.spec == "kp_flanks")][["date", "well", "tot"]]
    m = e.merge(q.rename(columns={"tot": "q_tot"}), on=["date", "well"])
    for y, col, lab in ((m.a / m.b, SRC["E"][1], "a_E / b_E (event background)"),
                        (m.a / m.q_tot, SRC["Q"][1], "a_E / Q (quiet background)")):
        g = y.groupby(m.week).median()
        pos = g > 0
        ax.semilogy(g.index[pos], g.values[pos], "o-", color=col, ms=2, lw=0.9, label=lab)
        ax.plot(g.index[~pos], np.full((~pos).sum(), 1.2e-2), "o", mfc="white", color=col, ms=2.5)
    ax.set_ylim(1e-2, 20)
    phases(ax)
    ax.set_xlabel("Weeks since first recording")
    ax.set_ylabel("Rhythm per event (a / b)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    ax.set_title("Open markers: median a_E ≤ 0", loc="left", color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--date", default="161216")
    ap.add_argument("--well", type=int, default=5)
    a = ap.parse_args()
    bp = pd.read_csv(os.path.join(RES, "trujillo_events_bandpower.csv"))
    ct = pd.read_csv(os.path.join(RES, "trujillo_events_counts.csv"))
    d = np.load(os.path.join(RES, "trujillo_events", f"{a.date}.npz"))
    fig, ax = plt.subplots(2, 3, figsize=(W2, 115 * MM))
    ev, quiet = trace_panel(ax[0, 0], d, a.well - 1)
    spectra_panel(ax[0, 1], d, a.well - 1, ev, quiet)
    ax[0, 1].set_title(f"Well {a.well}, 20{a.date[:2]}-{a.date[2:4]}-{a.date[4:]}", loc="left", color=INK2)
    band_panel(ax[0, 2], bp)
    rate_panel(ax[1, 0], ct)
    empty_panel(ax[1, 1], bp)
    per_event_panel(ax[1, 2], bp)
    for k, x in zip("abcdef", ax.ravel()):
        panel(x, k)
    fig.tight_layout()
    save(fig, a.outdir, "figS_organoid_events")


if __name__ == "__main__":
    main()
