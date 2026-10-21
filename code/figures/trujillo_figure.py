"""Organoid maturation figure (Trujillo et al. 2019 data).

  a  one well's spectrum before the rhythm, as it appears, and later, with
     the knee+plateau background fitted on the flanks
  b  empty-band check: a / b in 2.5-4.5 Hz before the rhythm appears, per
     background specification (one dot per well-recording)
  c  band total and background over weeks (geometric mean over wells,
     knee+plateau background), with the two phases shaded; top axis in
     weeks of culture (first recording = week 8)
  d-f maturation slope of ln a - lambda ln b against lambda (ols, lines;
     log-free estimates at lambda = 0 and 1, dots), per specification, in
     weeks 9-17 (rise), 17-33 (after the paper's turning point) and 9-33
  g  injected-rhythm recovery: estimated over injected periodic power for a
     broad rhythm added to the pre-rhythm spectra, against its size relative to
     the band power (plateau included)
  h  end to end, "rise" regime: recovered slopes at lambda = 0 and 1 against
     the truth (dashed), median over random orders
  i  one well after the turning point: the 2.5-4.5 Hz peak slides below the
     band and the knee background follows the low-frequency hump

Inputs: results/trujillo_bandpower_2.5-4.5.csv, results/trujillo_maturation.csv,
results/trujillo_injection_recovery.csv, results/trujillo_injection_e2e.csv,
results/trujillo_psd/*.npz (local).

Usage: python trujillo_figure.py OUTDIR [--well 7] [--dates 160810,161124,170120]
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
from trujillo_maturation import SPECS, ap_eval, fit_aperiodic, well_spectrum  # noqa: E402

LABEL = {"kp_flanks": "knee + plateau", "kp_flanks_h": "knee + plateau, 6-9 Hz out",
         "plateau_flanks": "power law + plateau", "plateau_flanks_h": "power law + plateau, 6-9 Hz out",
         "fixed_0.5-20": "power law 0.5-20 Hz, Whittle",
         "published": "published (robust line, per channel)"}
ORDER = ["kp_flanks", "kp_flanks_h", "plateau_flanks", "plateau_flanks_h", "fixed_0.5-20", "published"]
COLS = ["#2a78d6", "#1baf7a", "#eb6834", "#e87ba4", "#4a3aa7", "#0b0b0b"]


def spectra_panel(ax, well, dates):
    shades = ["#9fc3ee", "#5a9ae2", "#1d5aa6"]
    for date, col in zip(dates, shades):
        d = np.load(os.path.join(RES, "trujillo_psd", f"{date}.npz"))
        f = d["f"]
        P, _ = well_spectrum(d, well - 1)
        s = (f >= 0.25) & (f <= 45)
        sp = SPECS["kp_flanks"]
        keep = np.zeros(f.size, bool)
        for lo, hi in sp["bins"]:
            keep |= (f >= lo) & (f <= hi)
        fit = fit_aperiodic(P[keep], f[keep], None, sp["model"])
        ax.loglog(f[s], P[s] * 1e12, color=col, lw=1.0,
                  label=f"20{date[:2]}-{date[2:4]}-{date[4:]}")
        ax.loglog(f[s], ap_eval(fit["theta"], f[s], sp["model"]) * 1e12, color=col, lw=0.7, ls="--")
    ax.axvspan(2.5, 4.5, color=GREY, alpha=0.15, lw=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"Well {well}: dashed = background", loc="left", color=INK2)


def empty_band_panel(ax, bp, pre):
    t = bp[bp.week < pre]
    for i, spec in enumerate(ORDER):
        v = (t[t.spec == spec].a / t[t.spec == spec].b).to_numpy()
        x = i + (np.random.default_rng(i).random(v.size) - 0.5) * 0.35
        ax.plot(x, v, "o", color=GREY, ms=2.2, mec="none", alpha=0.7)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK2, lw=1.2)
    ax.axhline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[s] for s in ORDER], rotation=30, ha="right", fontsize=5.5)
    ax.set_ylabel("a / b, 2.5-4.5 Hz")
    ax.set_title(f"Before the rhythm (weeks < {pre:g}): should be 0", loc="left", color=INK2)


def trajectory_panel(ax, bp, spec):
    t = bp[bp.spec == spec]
    g = t.groupby("week")[["tot", "b"]].agg(lambda v: np.exp(np.log(v).mean()) * 1e12)
    ax.axvspan(9, 17, color=C0, alpha=0.08, lw=0)
    ax.axvspan(17, 33, color=C1, alpha=0.08, lw=0)
    ax.semilogy(g.index, g.tot, "o-", color=INK2, ms=2.5, lw=0.9, label="total, 2.5-4.5 Hz")
    ax.semilogy(g.index, g.b, "o-", color=GREY, ms=2.5, lw=0.9, mfc="white", label="background b")
    ax.set_xlabel("Weeks since first recording")
    ax.set_ylabel("Power (µV²/Hz)")
    ax.legend(frameon=False, loc="upper left")
    top = ax.secondary_xaxis("top", functions=(lambda x: x + 8, lambda x: x - 8))
    top.tick_params(labelsize=5.5)
    ax.set_title("8-well mean (top axis: weeks in culture)", loc="left", color=INK2, pad=10)


def recovery_panel(ax, rec):
    for spec, col in zip(ORDER, COLS):
        r = rec[(rec.spec == spec) & (rec["shape"] == "broad")]
        ax.semilogx(r.r, r.a_hat_over_a_true, "o-", color=col, ms=2.5, lw=1.0)
    ax.axhline(1, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("Injected a / band power")
    ax.set_ylabel("Estimated / injected a")
    ax.set_title("Injected rhythm, pre-rhythm spectra (weeks < 6)", loc="left", color=INK2)


def e2e_panel(ax, e2e):
    r = e2e[e2e.regime == "rise"].groupby("spec")[["lf_lam0", "lf_lam1", "s_a_true", "s_b_true"]].median()
    for k, (spec, col) in enumerate(zip(ORDER, COLS)):
        if spec in r.index:
            ax.plot([0, 1], [r.loc[spec, "lf_lam0"], r.loc[spec, "lf_lam1"]], "o-", color=col,
                    ms=3, lw=1.0)
    t = r.iloc[0]
    ax.plot([0, 1], [t.s_a_true, t.s_a_true - t.s_b_true], color=INK2, ls="--", lw=1.0)
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.2, 1.2)
    ax.set_xlabel("Assumed λ")
    ax.set_ylabel("Slope per week")
    ax.set_title("Synthetic rise: recovered vs true (dashed)", loc="left", color=INK2)


def late_panel(ax, well, dates):
    shades = ["#9fc3ee", "#5a9ae2", "#1d5aa6"]
    for date, col in zip(dates, shades):
        d = np.load(os.path.join(RES, "trujillo_psd", f"{date}.npz"))
        f = d["f"]
        P, _ = well_spectrum(d, well - 1)
        s = (f >= 0.25) & (f <= 45)
        sp = SPECS["kp_flanks"]
        keep = np.zeros(f.size, bool)
        for lo, hi in sp["bins"]:
            keep |= (f >= lo) & (f <= hi)
        fit = fit_aperiodic(P[keep], f[keep], None, sp["model"])
        ax.loglog(f[s], P[s] * 1e12, color=col, lw=1.0,
                  label=f"20{date[:2]}-{date[2:4]}-{date[4:]}")
        ax.loglog(f[s], ap_eval(fit["theta"], f[s], sp["model"]) * 1e12, color=col, lw=0.7, ls="--")
    ax.axvspan(2.5, 4.5, color=GREY, alpha=0.15, lw=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"Well {well} after the turning point", loc="left", color=INK2)


def curve_panel(ax, mat, title):
    grid = np.linspace(0, 1, 21)
    for k, (spec, col) in enumerate(zip(ORDER, COLS)):
        r = mat[(mat.spec == spec) & (mat.method == "ols")]
        lf = mat[(mat.spec == spec) & (mat.method == "logfree")]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.plot(grid, r.s_a - grid * r.s_b, color=col, lw=1.2, label=LABEL[spec])
        ax.plot([0, 1], [r.lam0_lo, r.lam1_lo], color=col, lw=0.5, ls="--")
        ax.plot([0, 1], [r.lam0_hi, r.lam1_hi], color=col, lw=0.5, ls="--")
        if not lf.empty:
            dx = (k - 2.5) * 0.012
            ax.plot([0 + dx, 1 + dx], [lf.iloc[0].lam0, lf.iloc[0].lam1], "o", color=col, ms=2.5,
                    mec="white", mew=0.4, zorder=4)
    ax.axhline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("Assumed λ")
    ax.set_title(title, loc="left", color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--well", type=int, default=7)
    ap.add_argument("--pre", type=float, default=6.0)
    ap.add_argument("--dates", default="160810,161124,170120")
    a = ap.parse_args()
    bp = pd.read_csv(os.path.join(RES, "trujillo_bandpower_2.5-4.5.csv"))
    fig, ax = plt.subplots(3, 3, figsize=(W2, 170 * MM))
    spectra_panel(ax[0, 0], a.well, a.dates.split(","))
    empty_band_panel(ax[0, 1], bp, a.pre)
    trajectory_panel(ax[0, 2], bp, "kp_flanks")
    mat = pd.read_csv(os.path.join(RES, "trujillo_maturation.csv"))
    for x, (win, what) in zip(ax[1], (("9-17", "rhythm appears"), ("17-33", "after the turning point"),
                                      ("9-33", "one line through both"))):
        curve_panel(x, mat[mat.window == win], f"Weeks {win}: {what}")
    ax[1, 0].set_ylabel("Slope of ln a − λ ln b per week\n(dashed: 95% limits at λ = 0 and 1)")
    ax[1, 1].legend(frameon=False, loc="upper right", fontsize=5.5)
    ymin = min(x.get_ylim()[0] for x in ax[1])
    ymax = max(x.get_ylim()[1] for x in ax[1])
    for x in ax[1]:
        x.set_ylim(ymin, ymax)
    recovery_panel(ax[2, 0], pd.read_csv(os.path.join(RES, "trujillo_injection_recovery.csv")))
    e2e_panel(ax[2, 1], pd.read_csv(os.path.join(RES, "trujillo_injection_e2e.csv")))
    late_panel(ax[2, 2], 10, ["161124", "170113", "170310"])
    for k, x in zip("abcdefghi", ax.ravel()):
        panel(x, k)
    fig.tight_layout()
    save(fig, a.outdir, "figS_organoid_maturation")


if __name__ == "__main__":
    main()
