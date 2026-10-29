"""Supplementary figure: the HBN alpha-vs-age crossover lambda* at every channel,
and how it depends on which channels are averaged and in what order.

  a, b  lambda* = s(0) / s_b at each channel (power-law background, band
        IAF +/- 2 Hz of the posterior ROI), eyes closed and eyes open. In
        this sample the background under alpha falls with age at every
        usable channel (s_b < 0), so the three colour zones read directly:
        below 0 alpha rises with age under both rules, between 0 and 1 it
        falls when the background is subtracted and rises when it is
        divided out, above 1 it falls under both. Contours at 0 and 1.
        The colour field is interpolated from usable channels only (a > 0
        in at least 80% of participants) and clipped to the head. Black
        dots: the channel's zone is certain (Bayesian-bootstrap probability
        >= 0.95); grey dots: uncertain; crosses: too few participants with
        alpha above the background. Rings: the six-channel posterior ROI.
  c, d  lambda* with its 95% HDI for the ROI combined in three orders
        (average the spectra, then fit; fit each channel, then average
        ln a and ln b; fit each channel, average a and b, then log), for
        the 19 channels nearest the 10-20 positions and all 129 channels
        (average the spectra, then fit), and for three single channels.
        Filled circles: power-law background; open squares: knee+plateau.
        Grey: fewer than 80% of participants usable.

Input: results/hbn_age_crossover_channels.csv and
results/hbn_age_crossover_averaging.csv (code/hbn_age_crossover_channels.py).

Usage: python hbn_age_crossover_channels_figure.py OUTDIR
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.channels.layout import _find_topomap_coords
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figures import GREY, INK, INK2, MM, RES, W2, interval, panel, save  # noqa: E402

CH = [f"E{i}" for i in range(1, 129)] + ["Cz"]
ROI = ["E62", "E65", "E70", "E75", "E83", "E90"]
VLIM = (-0.5, 1.5)
LAM_LIM = (-1.0, 2.0)
RELIABLE = 0.8
P_MARK = 0.95
COND = (("ec", "Eyes closed"), ("eo", "Eyes open"))
ROWS = (("roi", "spectrum_mean", "Posterior ROI (6 ch): average spectra, then fit *"),
        ("roi", "mean_log", "Posterior ROI: fit each channel, average log power"),
        ("roi", "log_mean", "Posterior ROI: fit each channel, average power"),
        ("10-20", "spectrum_mean", "19 standard 10-20 sites: average spectra, then fit"),
        ("all", "spectrum_mean", "All 129 channels: average spectra, then fit"),
        ("Fz (E11)", "single", "Fz alone"),
        ("Pz (E62)", "single", "Pz alone"),
        ("Oz (E75)", "single", "Oz alone"))
GROUP_BREAKS = (3, 5)               # rule lines after these many rows
MODELS = (("fixed", "o", "power-law background"), ("knee_plateau", "s", "knee + plateau background"))
# lambda* < 0 green (rises with age under both rules), 0-1 pale (the sign
# depends on the rule), > 1 purple (falls under both); colour steps at 0 and 1
GREEN, PURPLE, MID = "#1f7a3f", "#9b3f97", "#f0efec"
ZONE = "#efece4"                    # the [0, 1] band in the interval panels


def cmap():
    lo, hi = VLIM
    z0, z1 = (0 - lo) / (hi - lo), (1 - lo) / (hi - lo)
    return LinearSegmentedColormap.from_list("lamstar", [
        (0.0, GREEN), (z0 - 1e-6, "#8fc29d"), (z0, "#dfece2"), (0.5, MID),
        (z1, "#ecdfeb"), (z1 + 1e-6, "#c592c2"), (1.0, PURPLE)])


def info():
    i = mne.create_info(CH, 250.0, "eeg")
    i.set_montage("GSN-HydroCel-129", verbose="ERROR")
    return i


def head_circle(ax):
    """Centre and radius of the head outline MNE drew (its longest line)."""
    xy = max((ln.get_xydata() for ln in ax.lines), key=len)
    c = xy.mean(0)
    return c, np.median(np.hypot(*(xy - c).T))


def topo(ax, g, pos):
    g = g.set_index("channel").loc[CH]
    lam = np.clip(g.lam_star.to_numpy(float), *VLIM)
    rel = (g.share_pos >= RELIABLE).to_numpy()
    im, cn = mne.viz.plot_topomap(lam[rel], pos[rel], axes=ax, cmap=cmap(), vlim=VLIM,
                                  contours=[0.0, 1.0], sensors=False, show=False,
                                  outlines="head", image_interp="cubic", res=256,
                                  extrapolate="local")
    c, r = head_circle(ax)
    clip = Circle(c, r, transform=ax.transData)
    im.set_clip_path(clip)
    if cn is not None:
        cn.set_clip_path(clip)
        cn.set_linewidth(0.7)
        cn.set_edgecolor(INK2)
    sure = np.zeros(len(CH), bool)
    for col in ("p01", "p_above1", "p_below0"):
        sure |= (g[col] >= P_MARK).to_numpy()
    sure &= rel
    ax.plot(*pos[rel & ~sure].T, "o", color=GREY, ms=1.6, mec="none", zorder=5)
    ax.plot(*pos[sure].T, "o", color=INK, ms=2.0, mec="none", zorder=6)
    ax.plot(*pos[~rel].T, "x", color=INK2, ms=2.2, mew=0.5, zorder=6)
    ax.plot(*pos[np.isin(CH, ROI)].T, "o", mfc="none", mec=INK, mew=0.6, ms=4.4, zorder=7)
    m = 0.012
    ax.set_xlim(pos[:, 0].min() - m, pos[:, 0].max() + m)
    ax.set_ylim(pos[:, 1].min() - m, pos[:, 1].max() + 2 * m)
    ax.set_aspect("equal")
    return im


def forest(ax, V, cond, show_labels):
    y = np.arange(len(ROWS))[::-1].astype(float)
    ax.axvspan(0, 1, color=ZONE, lw=0)
    for v in (0, 1):
        ax.axvline(v, color=INK2, lw=0.5)
    top = y[0] + 0.75
    for x, s, ha in ((-0.05, "rises under\nboth rules", "right"),
                     (0.5, "falls if subtracted,\nrises if divided", "center"),
                     (1.05, "falls under\nboth rules", "left")):
        ax.text(x, top, s, ha=ha, va="bottom", fontsize=5.5, color=INK2, linespacing=1.1)
    for (mname, mk, _), dy in zip(MODELS, (0.15, -0.15)):
        for yi, (sname, oname, _) in zip(y, ROWS):
            r = V[(V.model == mname) & (V.cond == cond) & (V.set == sname) & (V.order == oname)]
            if r.empty or not np.isfinite(r.lam_star.iloc[0]):
                continue
            r = r.iloc[0]
            col = INK if r.share_pos >= RELIABLE else GREY
            interval(ax, r.hdi_lo, r.hdi_hi, yi + dy, *LAM_LIM, col, lw=0.8)
            if LAM_LIM[0] <= r.lam_star <= LAM_LIM[1]:
                ax.plot(r.lam_star, yi + dy, mk, ms=3.2, mew=0.7, mec=col,
                        mfc=col if mname == "fixed" else "white", zorder=5)
    for k in GROUP_BREAKS:
        ax.axhline(y[0] - k + 0.5, color=GREY, lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for *_, lab in ROWS] if show_labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*LAM_LIM)
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.set_xlabel("Crossover λ* (95% interval)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    a = ap.parse_args()
    O = pd.read_csv(os.path.join(RES, "hbn_age_crossover_channels.csv"))
    V = pd.read_csv(os.path.join(RES, "hbn_age_crossover_averaging.csv"))
    O = O[(O.model == "fixed") & (O.band == "iaf_roi")]
    pos = _find_topomap_coords(info(), None)

    fig = plt.figure(figsize=(W2, 165 * MM))
    H = 165 * MM / W2
    side = 0.29
    for k, (cond, title) in enumerate(COND):
        ax = fig.add_axes([0.05 + 0.33 * k, 0.60, side, side / H])
        im = topo(ax, O[O.cond == cond], pos)
        ax.set_title(title, pad=2, fontweight="bold")
        panel(ax, "ab"[k])
    cax = fig.add_axes([0.72, 0.64, 0.012, 0.26])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([-0.5, 0, 1, 1.5])
    cb.outline.set_linewidth(0.4)
    cax.set_title("Crossover λ*", fontsize=6.5, pad=4, loc="left")
    for v, s in ((-0.25, "alpha rises with age\nunder both rules"),
                 (0.5, "alpha falls if the background\nis subtracted, rises if\nit is divided out"),
                 (1.25, "alpha falls with age\nunder both rules")):
        cax.text(3.6, v, s, transform=cax.get_yaxis_transform(), va="center", fontsize=5.8,
                 color=INK, linespacing=1.15)
    hm = [Line2D([], [], ls="none", marker="o", color=INK, ms=2.0, label="zone certain (P ≥ 0.95)"),
          Line2D([], [], ls="none", marker="o", color=GREY, ms=1.6, label="zone uncertain"),
          Line2D([], [], ls="none", marker="x", color=INK2, ms=2.2, mew=0.5,
                 label="too little alpha (< 80% of children)"),
          Line2D([], [], ls="none", marker="o", mfc="none", mec=INK, mew=0.6, ms=4.4,
                 label="posterior ROI used in the paper")]
    fig.legend(handles=hm, loc="center", ncol=4, bbox_to_anchor=(0.45, 0.565),
               handletextpad=0.3, columnspacing=1.2)
    fig.text(0.05, 0.535, "Maps: power-law background; colour interpolated between usable "
             "channels only.", fontsize=5.5, color=INK2)
    for k, (cond, title) in enumerate(COND):
        ax = fig.add_axes([0.37 + 0.34 * k, 0.07, 0.27, 0.36])
        forest(ax, V, cond, show_labels=k == 0)
        ax.set_title(title, pad=16, fontweight="bold")
        panel(ax, "cd"[k])
    hf = [Line2D([], [], ls="none", marker="o", color=INK, ms=3.2, label=MODELS[0][2]),
          Line2D([], [], ls="none", marker="s", mfc="white", mec=INK, mew=0.7, ms=3.2,
                 label=MODELS[1][2]),
          Line2D([], [], ls="none", marker="o", color=GREY, ms=3.2,
                 label="< 80% of children usable"),
          Patch(color=ZONE, label="sign depends on the rule")]
    fig.legend(handles=hf, loc="center", ncol=4, bbox_to_anchor=(0.55, 0.0),
               handletextpad=0.3, columnspacing=1.2)
    fig.text(0.01, 0.005, "* as in the paper", fontsize=5.5, color=INK2)
    save(fig, a.outdir, "figS_hbn_age_crossover_channels")


if __name__ == "__main__":
    main()
