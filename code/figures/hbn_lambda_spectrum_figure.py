"""Supplementary figure: the HBN age effect and its crossover lambda* by frequency.

Columns: eyes closed, eyes open. Power-law background solid (filled
markers), knee+plateau dashed (open markers).
  a, b  share of participants with periodic power a = T - b > 0 in each
        1-Hz band; dotted line at 80%, the share needed to read lambda*.
        The bar under the axis marks the 6-16 Hz window left out of the
        background fit (the background is interpolated across it).
  c, d  age slope of ln a - lambda ln b at lambda = 0 (subtract) and
        lambda = 1 (divide), % per year, pairs-bootstrap 95% intervals for
        the power law. Hatched: fewer than 80% of participants have a > 0
        under either background, so no lambda* is read there.
  e, f  crossover lambda* = s_a / s_b with its Bayesian-bootstrap 95% HDI,
        shown only where at least 80% of participants have a > 0 under that
        marker's background (elsewhere a ~ 0 and lambda* is noise over
        noise). At every frequency shown the background falls with age
        (s_b < 0), so below 0 alpha rises with age under both rules, between
        0 and 1 it falls when the background is subtracted and rises when it
        is divided out, and above 1 it falls under both.
  g, h  the same in 1-Hz bands centred on each participant's alpha peak
        frequency (IAF + k), with the whole IAF +/- 2 Hz band (star) used in
        the paper.

Input: results/hbn_lambda_spectrum.csv (code/hbn_lambda_spectrum.py).

Usage: python hbn_lambda_spectrum_figure.py OUTDIR
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figures import C0, C1, GREY, INK, INK2, MM, RES, W2, panel, save  # noqa: E402

LAM_LIM = (-2.1, 2.6)
MODELS = (("fixed", "-", "o", "power-law background"),
          ("knee_plateau", "--", "s", "knee + plateau background"))
COND = (("ec", "Eyes closed"), ("eo", "Eyes open"))
RELIABLE = 0.8
ZONE = "#efece4"                    # lambda* in [0, 1]: the sign depends on the rule
HATCH = dict(facecolor="none", edgecolor="#c9c7c1", hatch="////", lw=0)
BOX = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1)


def pct(s):
    return 100 * (np.exp(s) - 1)


def hatch_unreliable(ax, D, cond):
    """Hatch the bands where fewer than RELIABLE of participants have a > 0
    under either background, so that no lambda* can be read there."""
    g = D[D.cond == cond].groupby(["f_lo", "f_hi"]).reliable.any().reset_index()
    for _, r in g[~g.reliable.astype(bool)].iterrows():
        ax.axvspan(r.f_lo, r.f_hi, **HATCH)


def zones(ax, right_labels):
    ax.axhspan(0, 1, color=ZONE, lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.axhline(1, color=INK2, lw=0.5)
    if right_labels:
        for y, s in ((1.8, "falls under\nboth rules"), (0.5, "sign depends\non the rule"),
                     (-1.3, "rises under\nboth rules")):
            ax.text(1.01, y, s, transform=ax.get_yaxis_transform(), va="center",
                    fontsize=5.5, color=INK2, linespacing=1.1)


def crossover_points(ax, g, x, marker, filled, dx=0.0):
    """lambda* with its HDI, only where the marker's own background leaves at
    least RELIABLE of participants with a > 0."""
    for xi, (_, r) in zip(x, g.iterrows()):
        if not r.reliable:
            continue
        if r.lam_star_bounded:
            ax.plot([xi + dx] * 2, [r.hdi_lo, r.hdi_hi], color=INK, lw=0.7,
                    solid_capstyle="butt")
        ax.plot(xi + dx, r.lam_star, marker, ms=2.8 if filled else 2.6, color=INK,
                mfc=INK if filled else "white", mew=0.6, zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    a = ap.parse_args()
    O = pd.read_csv(os.path.join(RES, "hbn_lambda_spectrum.csv"))
    A = O[O.axis == "absolute"]
    R = O[O.axis == "iaf_rel"]
    I = O[O.axis == "iaf_band"]
    for df in (A, R, I):
        assert (df[df.reliable.astype(bool)].s_b < 0).all(), "zone labels assume s_b < 0"

    fig, axes = plt.subplots(4, 2, figsize=(W2, 195 * MM),
                             gridspec_kw=dict(height_ratios=[0.7, 1, 1, 1], hspace=0.62,
                                              wspace=0.36))
    for j, (cond, title) in enumerate(COND):
        ax = axes[0, j]
        for model, ls, _, lab in MODELS:
            g = A[(A.cond == cond) & (A.model == model)]
            ax.plot(g.centre, 100 * g.share_pos, color=INK, ls=ls, lw=1.0, label=lab)
        ax.axhline(100 * RELIABLE, color=INK2, lw=0.6, ls=":")
        ax.text(40.4, 100 * RELIABLE - 3, "80%: enough to read λ*", ha="right", va="top",
                fontsize=5.5, color=INK2)
        ax.set_ylim(0, 128)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel("Children with power\nabove background (%)")
        ax.set_title(title, fontweight="bold")
        ax.axvspan(6, 16, ymin=0, ymax=0.035, color=GREY, alpha=0.6, lw=0)
        if j == 0:
            ax.text(16.4, 3, "6–16 Hz: not fitted", fontsize=5.2, color=INK2, va="bottom")
        ax.legend(handles=[Line2D([], [], color=INK, ls=ls, lw=1.0, label=lab)
                           for _, ls, _, lab in MODELS],
                  loc="upper right", ncol=2, handlelength=2.2, columnspacing=1.0)

        ax = axes[1, j]
        hatch_unreliable(ax, A, cond)
        for model, ls, _, _ in MODELS:
            g = A[(A.cond == cond) & (A.model == model)]
            for lam, col in ((0, C0), (1, C1)):
                ax.plot(g.centre, pct(g[f"s{lam}"]), color=col, ls=ls, lw=1.0)
                if model == "fixed":
                    ax.fill_between(g.centre, pct(g[f"s{lam}_lo"]), pct(g[f"s{lam}_hi"]),
                                    color=col, alpha=0.18, lw=0)
        ax.axhline(0, color=INK2, lw=0.6)
        if j == 0:
            ax.set_ylim(bottom=-32)
        ax.set_ylabel("Age slope (% per year)")
        if j == 0:
            ax.legend(handles=[
                Line2D([], [], color=C0, lw=1.2, label="λ = 0: background subtracted"),
                Line2D([], [], color=C1, lw=1.2, label="λ = 1: background divided out"),
                Patch(**HATCH, label="< 80% of children above background\n(under either model)")],
                loc="lower right", fontsize=5.5, handlelength=1.6, **BOX)

        ax = axes[2, j]
        hatch_unreliable(ax, A, cond)
        zones(ax, right_labels=j == 1)
        for model, _, mk, _ in MODELS:
            g = A[(A.cond == cond) & (A.model == model)]
            dx = -0.17 if model == "fixed" else 0.17
            crossover_points(ax, g, g.centre.to_numpy(), mk, model == "fixed", dx)
        ax.set_ylim(*LAM_LIM)
        ax.set_ylabel("Crossover λ*")
        ax.set_xlabel("Frequency (Hz), fixed 1-Hz bands")

        ax = axes[3, j]
        hatch_unreliable(ax, R, cond)
        zones(ax, right_labels=j == 1)
        for model, _, mk, _ in MODELS:
            g = R[(R.cond == cond) & (R.model == model)]
            dx = -0.12 if model == "fixed" else 0.12
            crossover_points(ax, g, g.centre.to_numpy(), mk, model == "fixed", dx)
            r = I[(I.cond == cond) & (I.model == model)].iloc[0]
            if r.reliable:
                xs = 5.0 + (0 if model == "fixed" else 0.35)
                ax.plot([xs] * 2, [r.hdi_lo, r.hdi_hi], color=INK, lw=0.7, clip_on=False)
                ax.plot(xs, r.lam_star, "*", ms=5.5, color=INK,
                        mfc=INK if model == "fixed" else "white", mew=0.6, clip_on=False,
                        zorder=3)
        ax.axvline(4.6, color=GREY, lw=0.4)
        ax.set_ylim(*LAM_LIM)
        ax.set_xlim(-4.6, 5.6)
        ax.set_xticks(list(range(-4, 5)) + [5.17])
        ax.set_xticklabels([str(k) for k in range(-4, 5)] + ["band\n±2"])
        ax.set_ylabel("Crossover λ*")
        ax.set_xlabel("Frequency relative to each child's alpha peak (Hz)")
        if j == 0:
            ax.legend(handles=[
                Line2D([], [], ls="none", marker="o", ms=3, color=INK, label="power law"),
                Line2D([], [], ls="none", marker="s", ms=3, color=INK, mfc="white",
                       mew=0.6, label="knee + plateau"),
                Line2D([], [], ls="none", marker="*", ms=5.5, color=INK,
                       label="peak ± 2 Hz (paper)"),
                Patch(color=ZONE, label="sign depends on rule")],
                loc="lower left", ncol=2, fontsize=5.5, handletextpad=0.3,
                columnspacing=0.8, **BOX)

        for ax in axes[:3, j]:
            ax.set_xlim(1.6, 40.4)
            ax.set_xticks([2, 5, 10, 15, 20, 25, 30, 35, 40])
    for k, ax in enumerate(axes.ravel()):
        panel(ax, "abcdefgh"[k])
    save(fig, a.outdir, "figS_hbn_lambda_spectrum")


if __name__ == "__main__":
    main()
