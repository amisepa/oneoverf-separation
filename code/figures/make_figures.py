"""Figures of the paper.

Main text
  1  the coupling family and what specparam and IRASA assume
  2  alpha power against age under each separation rule (HBN)
  3  published claims re-tested: crossover lambda* per claim
     (results/breadth_summary.csv)
  4  every estimate of lambda, by design (results/hbn_lambda_gmm.csv,
     identification_summary.csv)
  5  what sets lambda in a synaptic model (results/sim_mechanisms.csv)
Supplementary (--only 11-15)
  S1 identifiability: single spectrum vs across spectra
  S2 consequences for condition contrasts (sim02, steepening scenario)
  S3 spatial test against its null (fitted and fit-free)
  S4 robustness of the age slopes across estimators
  S5 gain tipping point

Inputs: results/sim01.mat, results/sim02_steepen.mat, results/sim_topography_null_flanks.csv,
results/hbn_topography_flanks.csv,
results/hbn_lambda_gmm.csv, results/sim_topography_null.csv,
results/hbn_age_alpha_robust.csv, results/hbn_age_alpha_crossover.csv,
results/hbn_age_alpha_phases.csv, results/hbn_age_alpha_spline.csv, and the local
per-subject files results/hbn_kp_fits.csv and results/hbn_topography*.npz
(regenerable with the scripts in code/). Ages come from the PSD files in
$HBN_OUT, or from results/hbn_roi.csv (local) when those are absent.

Usage: python make_figures.py OUTDIR [--only 1,2,11]
"""
import argparse
import glob
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(HERE, ".."))

# ---- style -------------------------------------------------------------
C0 = "#2a78d6"     # lambda = 0 (additive, IRASA-like)
C1 = "#eb6834"     # lambda = 1 (multiplicative, specparam-like)
CH = "#1baf7a"     # lambda = 0.5
GREY = "#8a8984"   # total power, background, reference quantities
INK = "#0b0b0b"
INK2 = "#52514e"
MM = 1 / 25.4
W2 = 183 * MM      # Nature double column

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "legend.frameon": False,
    "lines.linewidth": 1.4, "savefig.dpi": 300,
})


def panel(ax, letter):
    ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def null_bar(ax, x, lo, hi, min_h=0.02):
    """Range of a null simulation as a grey bar, visible even when lo == hi."""
    pad = max(0.0, (min_h - (hi - lo)) / 2)
    ax.plot([x, x], [lo - pad, hi + pad], color=GREY, lw=6,
            solid_capstyle="butt", alpha=0.6)


def interval(ax, lo, hi, y, lo_x, hi_x, color, ends=(True, True), ms=3.5, **kw):
    """Horizontal interval at y, cut at lo_x and hi_x. A cut end gets an
    arrowhead, so an interval that runs past the axis is not read as ending
    there; ends=(False, True) leaves the lower end unmarked."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo_x or lo > hi_x:
        return
    ax.plot([max(lo, lo_x), min(hi, hi_x)], [y, y], color=color, **kw)
    for cut, x, mk, show in ((lo < lo_x, lo_x, "<", ends[0]), (hi > hi_x, hi_x, ">", ends[1])):
        if cut and show:
            ax.plot(x, y, mk, color=color, ms=ms, mec="none", clip_on=False, zorder=4)


# output names in the current numbering, keyed by the name each function saves
RENAME = {"fig4_age_reversal": "fig2_age_reversal",
          "figS3_mechanisms": "fig5_mechanisms",
          "fig2_identifiability": "figS1_identifiability",
          "fig3_condition_contrasts": "figS2_condition_contrasts",
          "figS1_age_robustness": "figS4_age_robustness",
          "figS2_gain_tipping": "figS5_gain_tipping"}


def save(fig, outdir, name):
    name = RENAME.get(name, name)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def gauss(f, cf, bw):
    return np.exp(-0.5 * ((f - cf) / bw) ** 2)


# ---- Figure 1 -----------------------------------------------------------
def fig1(outdir):
    f = np.linspace(1, 40, 800)
    chi, cf, bw = 1.4, 10.0, 1.5
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.72, 2.3))

    # (a) same intrinsic peak on a low and a high background, lambda 0 vs 1
    ax = axs[0]
    ref = 10 ** 1.0 / cf ** chi
    for off, ls in ((1.0, "-"), (1.6, "--")):
        L = 10 ** off / f ** chi
        for lam, col in ((0, C0), (1, C1)):
            amp = 1.2 * ref / ref ** lam          # same size at the low background
            ax.loglog(f, L + amp * gauss(f, cf, bw) * L ** lam, color=col, ls=ls,
                      lw=1.1)
        ax.loglog(f, L, color=GREY, ls=ls, lw=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title("Same rhythm on a low and a high background",
                 loc="left", color=INK2)
    ax.text(0.03, 0.05, "solid: low background\ndashed: high background",
            color=INK2, transform=ax.transAxes, fontsize=6)
    ax.text(0.97, 0.95, "λ = 0: peak added", color=C0, transform=ax.transAxes,
            ha="right", va="top")
    ax.text(0.97, 0.86, "λ = 1: peak scales", color=C1, transform=ax.transAxes,
            ha="right", va="top")
    ax.set_xlim(1, 40)
    panel(ax, "a")

    # (b) what each tool reports as the background level changes
    ax = axs[1]
    offs = np.linspace(0.6, 2.0, 50)
    for truth, ls in ((0, "-"), (1, "--")):
        rel, absd = [], []
        for off in offs:
            Lcf = 10 ** off / cf ** chi
            amp = 1.2 * ref / ref ** truth
            peak = amp * Lcf ** truth
            rel.append(np.log10((Lcf + peak) / Lcf))     # specparam: log ratio
            absd.append(peak)                            # IRASA: difference
        rel, absd = np.array(rel), np.array(absd)
        ax.plot(offs, rel / rel[0], color=C1, ls=ls)
        ax.plot(offs, absd / absd[0], color=C0, ls=ls)
    ax.set_yscale("log")
    ax.set_xlabel("Background offset (log$_{10}$ power)")
    ax.set_ylabel("Reported peak (relative to first point)")
    ax.set_title("What each tool reports as the background rises",
                 loc="left", color=INK2)
    ax.text(0.03, 0.95, "IRASA-style (difference)", color=C0,
            transform=ax.transAxes, va="top")
    ax.text(0.03, 0.86, "specparam-style (log ratio)", color=C1,
            transform=ax.transAxes, va="top")
    ax.text(0.03, 0.05, "solid: additive truth\ndashed: multiplicative truth",
            color=INK2, transform=ax.transAxes, va="bottom", fontsize=6)
    panel(ax, "b")

    fig.tight_layout(w_pad=2.2)
    save(fig, outdir, "fig1_coupling_family")


# ---- Figure 2 -----------------------------------------------------------
def whittle_fit(P, f, K, lam):
    """Fit aperiodic + one peak with lambda fixed; return the deviance."""
    def mu(th):
        off, chi, la, cf, lbw = th
        L = 10 ** off / f ** chi
        return L + 10 ** la * gauss(f, cf, 10 ** lbw) * L ** lam
    def dev(th):
        m = np.maximum(mu(th), 1e-300)
        r = P / m
        return 2 * K * np.sum(r - np.log(r) - 1)
    lp = np.log10(P)
    keep = (f < 6) | (f > 16)
    b = np.polyfit(np.log10(f[keep]), lp[keep], 1)
    off0, chi0 = b[1], -b[0]
    L0 = 10 ** off0 / f ** chi0
    i = np.argmax(P / L0 * ((f > 6) & (f < 14)))
    pk = max(P[i] - L0[i], 1e-6)
    best = None
    for la0 in (np.log10(pk / L0[i] ** lam),):
        r = minimize(dev, [off0, chi0, la0, f[i], np.log10(1.5)],
                     method="Nelder-Mead",
                     options=dict(maxiter=6000, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.fun


def fig2(outdir):
    from sim_control_band import synth_segments
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))

    # (a) lambda = 1 peak vs its best Gaussian (lambda = 0) approximation
    ax = axs[0]
    f = np.linspace(2, 30, 2000)
    L = f ** -1.4
    pk1 = gauss(f, 10, 1.5) * L
    def res(th):
        return np.sum((th[0] * gauss(f, th[1], th[2]) - pk1) ** 2)
    th = minimize(res, [pk1.max(), 9.7, 1.5], method="Nelder-Mead",
                  options=dict(xatol=1e-10, fatol=1e-16, maxiter=20000)).x
    g0 = th[0] * gauss(f, th[1], th[2])
    s = 1 / pk1.max()
    ax.plot(f, pk1 * s, color=C1, label="λ = 1 peak")
    ax.plot(f, g0 * s, color=C0, ls="--", label="best λ = 0 peak")
    ax.plot(f, (pk1 - g0) * s * 100, color=GREY, lw=0.9,
            label="difference × 100")
    ax.set_xlim(4, 16)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Periodic power (peak = 1)")
    ax.legend(loc="upper right")
    ax.set_title(f"Max difference {np.abs(pk1 - g0).max() * s * 100:.2f}% of peak",
                 loc="left", color=INK2)
    panel(ax, "a")

    # (b) single-spectrum Whittle profiles over lambda: flat
    ax = axs[1]
    rng = np.random.default_rng(7)
    grid = np.linspace(0, 1, 11)
    for lam_true, col in ((0.0, C0), (1.0, C1)):
        for rep in range(3):
            off = rng.normal(1.0, 0.3)
            ref = 10 ** 1.0 / 10 ** 1.2
            amp = 1.0 * ref / ref ** lam_true
            acc, ff = synth_segments(off, 1.2, 10.0, 1.5, amp, lam_true, rng)
            m = (ff >= 2) & (ff <= 30)
            P, fm, K = acc.mean(0)[m], ff[m], acc.shape[0]
            d = np.array([whittle_fit(P, fm, K, lam) for lam in grid])
            ax.plot(grid, d - d.min(), color=col, lw=1.3, alpha=0.9)
    ax.axhline(3.84, color=INK2, lw=0.6, ls=":")
    ax.text(0.02, 3.84, " 95% threshold", color=INK2, va="bottom", fontsize=6)
    ax.set_xlabel("λ assumed in the fit")
    ax.set_ylabel("Δ deviance (one spectrum)")
    ax.text(0.97, 0.55, "true λ = 0", color=C0, transform=ax.transAxes,
            ha="right", va="top")
    ax.text(0.97, 0.46, "true λ = 1", color=C1, transform=ax.transAxes,
            ha="right", va="top")
    ax.set_ylim(-0.2, 4.6)
    ax.set_title("A single spectrum does not identify λ", loc="left", color=INK2)
    panel(ax, "b")

    # (c) across spectra: the GLM coefficient recovers lambda
    ax = axs[2]
    fh = h5py.File(os.path.join(RES, "sim01.mat"), "r")
    R = fh["R"]
    names = ["".join(chr(c) for c in fh[n[0]][:].flatten()) for n in fh["names"]]
    lt, est = [], {}
    oracle = []
    for i in range(3):
        lt.append(np.array(fh[R["lambda_true"][i][0]]).item())
        res_ = fh[R["res"][i][0]]
        lam = [np.array(fh[r]).item() for r in np.array(res_["lambda"]).ravel()]
        se = [np.array(fh[r]).item() for r in np.array(res_["lambda_se"]).ravel()]
        for n_, l_, s_ in zip(names, lam, se):
            est.setdefault(n_, []).append((l_, s_))
        tr = fh[R["truth"][i][0]]
        a_, b_ = np.array(tr["a_true"]).ravel(), np.array(tr["b_true"]).ravel()
        X = np.c_[np.ones(a_.size), np.log(b_)]
        oracle.append(np.linalg.lstsq(X, np.log(a_), rcond=None)[0][1])
    lt = np.array(lt)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
    ax.plot(lt, oracle, "o", color=GREY, ms=5, label="true background")
    for n_, col, dx, lab in (("Censored 6-16 Hz", C0, -0.025, "censored regression"),
                             ("specparam 3 peaks", C1, 0.025, "specparam, 3 peaks")):
        v = np.array(est[n_])
        ax.errorbar(lt + dx, v[:, 0], yerr=1.96 * v[:, 1], fmt="o", color=col,
                    ms=4, lw=1, capsize=0, label=lab)
    ax.set_xlabel("True λ")
    ax.set_ylabel("Estimated λ (GLM across 120 spectra)")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="upper left")
    ax.set_title("Across spectra, λ is identified", loc="left", color=INK2)
    panel(ax, "c")
    fig.tight_layout(w_pad=2.2)
    save(fig, outdir, "fig2_identifiability")


# ---- Figure 3 -----------------------------------------------------------
def fig3(outdir):
    # post-stimulus steepening, as reported by Gyurkovics et al. 2022
    fh = h5py.File(os.path.join(RES, "sim02_steepen.mat"), "r")
    S = fh["S"]

    def get(i, k):
        a = np.array(fh[S[k][i][0]]).squeeze()
        return a["real"] if a.dtype.names else a

    def getg(i, k):
        g = fh[S["g2"][i][0]]
        a = np.array(g[k]).squeeze()
        return a["real"] if a.dtype.names else a

    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.78, 2.3),
                            gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    cols = ["#c9c8c3", C0, C1, INK2]
    truth_names = {0: "additive truth", 1: "multiplicative truth"}
    for gi, i in enumerate(range(2)):
        lam_t = float(get(i, "lambda_true"))
        t = float(get(i, "dlogc_true"))
        ok = get(i, "ok").astype(bool)
        dla, dlb = get(i, "dloga")[ok], get(i, "dlogb")[ok]
        dtot = get(i, "ddB")[ok] * np.log(10) / 10      # dB -> natural log
        vals = [dtot, dla, dla - dlb]
        x0 = gi * 5
        for j, v in enumerate(vals):
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
            ax.bar(x0 + j, m, color=cols[j], width=0.72)
            ax.errorbar(x0 + j, m, yerr=1.96 * se, color=INK, lw=0.8, capsize=0)
        d = float(getg(i, "delta"))
        ci = getg(i, "delta_ci").ravel()
        ax.bar(x0 + 3, d, color=cols[3], width=0.72)
        ax.errorbar(x0 + 3, d, yerr=[[d - ci[0]], [ci[1] - d]], color=INK,
                    lw=0.8, capsize=0)
        ax.hlines(t, x0 - 0.5, x0 + 3.5, color=INK, lw=0.8, ls="--")
        ax.text(x0 + 1.5, -0.33, truth_names[int(round(lam_t))], ha="center",
                color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([0, 1, 2, 3, 5, 6, 7, 8])
    ax.set_xticklabels(["total", "λ = 0", "λ = 1", "λ̂"] * 2)
    ax.set_ylim(-0.4, 1.0)
    ax.set_ylabel("Recovered change in alpha (Δ log)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=INK, lw=0.8, ls="--",
                              label="true change")], loc="upper right")
    ax.set_title("One true change, four ways to measure it", loc="left", color=INK2)
    panel(ax, "a")

    # (b) per-trial dependence on background for raw power vs the GLM
    # (additive truth)
    ax = axs[1]
    i = 0
    ok = get(i, "ok").astype(bool)
    lb0 = get(i, "lb0")[ok]
    raw = get(i, "dPow")[ok]
    glm = get(i, "dlogc_hat")          # already restricted to ok trials
    z = lambda v: (v - v.mean()) / v.std()
    ax.plot(lb0 / np.log(10), z(raw), "o", ms=3, color=GREY, alpha=0.8,
            mec="none", label=f"raw band power  r = {np.corrcoef(lb0, raw)[0, 1]:.2f}")
    ax.plot(lb0 / np.log(10), z(glm), "o", ms=3, color=INK, alpha=0.8,
            mec="none", label=f"λ̂-corrected  r = {np.corrcoef(lb0, glm)[0, 1]:.2f}")
    ax.set_xlabel("Baseline background level (log$_{10}$ power)")
    ax.set_ylabel("Estimated change per trial (z)")
    ax.legend(loc="upper left")
    ax.set_title("Dependence on background (additive truth)", loc="left",
                 color=INK2)
    panel(ax, "b")
    fig.tight_layout(w_pad=2.2)
    save(fig, outdir, "fig3_condition_contrasts")


# ---- Figure 4 -----------------------------------------------------------
def load_age_table(model, qc=True):
    k = pd.read_csv(os.path.join(RES, "hbn_kp_fits.csv"))
    k = k[(k.model == model) & (k.window == "censor 6-16") & (k.band == "alpha")
          & (k.split == "full")]
    w = k.pivot_table(index="subject", columns="cond", values=["a", "b", "tot"])
    w.columns = [f"{v}_{c}" for v, c in w.columns]
    rows = []
    for p in glob.glob(os.path.join(os.environ.get("HBN_OUT", "hbn_psd"), "*.npz")):
        d = np.load(p, allow_pickle=True)
        rows.append((str(d["subject"]), float(d["age"])))
    if not rows:
        # no PSD files here: the ages copied into the per-subject ROI table
        roi = pd.read_csv(os.path.join(RES, "hbn_roi.csv"), usecols=["subject", "age"])
        rows = list(roi.drop_duplicates("subject").itertuples(index=False, name=None))
    ages = pd.DataFrame(rows, columns=["subject", "age"]).drop_duplicates("subject")
    w = w.join(ages.set_index("subject"), how="inner").dropna()
    if qc:
        q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
        w = w.loc[w.index.intersection(q.index[q.qc_ok])]
    return w


def slope_curve(x, la, lb, grid, rng, nboot=2000):
    """Age slope of la - g * lb for every g in grid, with its 95% pairs-bootstrap
    interval (the same resamples at every g)."""
    X = np.c_[np.ones_like(x), x]
    Y = np.c_[la, lb]
    sa, sb = np.linalg.lstsq(X, Y, rcond=None)[0][1]
    n = x.size
    bs = np.array([np.linalg.lstsq(X[i], Y[i], rcond=None)[0][1]
                   for i in (rng.integers(0, n, n) for _ in range(nboot))])
    lo, hi = np.percentile(bs[:, :1] - grid * bs[:, 1:], [2.5, 97.5], axis=0)
    return sa - grid * sb, lo, hi


def spline_trend(x, y, n=100):
    """Natural cubic spline of y on x (4 df, fitted between the 1st and 99th
    percentiles of x, as in hbn_age_alpha_robust.spline_derivative)."""
    import patsy
    lo, hi = np.percentile(x, [1, 99])
    D = patsy.dmatrix("cr(x, df=4, lower_bound=lb, upper_bound=ub)",
                      {"x": np.clip(x, lo, hi), "lb": lo, "ub": hi})
    beta = np.linalg.lstsq(np.asarray(D), y, rcond=None)[0]
    xx = np.linspace(lo, hi, n)
    return xx, np.asarray(patsy.build_design_matrices(
        [D.design_info], {"x": xx, "lb": lo, "ub": hi})[0]) @ beta


def fig4(outdir):
    rng = np.random.default_rng(0)
    wf = load_age_table("fixed")
    wk = load_age_table("knee_plateau")
    cut = 12.0                      # division between the two age ranges (years)
    YOUNG, OLD = "#b07cc6", "#4f2582"
    pct = lambda s: 100 * (np.exp(s) - 1)
    fig = plt.figure(figsize=(W2, 4.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.55, wspace=1.3)

    # (a) eyes-closed alpha vs age under three rules
    w = wf[wf.a_ec > 0]
    series = [("total power", wf.age, np.log(wf.tot_ec), GREY),
              ("λ = 0 (subtract)", w.age, np.log(w.a_ec), C0),
              ("λ = 1 (divide)", w.age, np.log(w.a_ec / w.b_ec), C1)]
    box = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8)
    for j, (lab, age, y, col) in enumerate(series):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        age, y = age.to_numpy(), y.to_numpy()
        ax.plot(age, y / np.log(10), "o", ms=1.2, color=col, alpha=0.25, mec="none")
        ax.axvline(cut, color=INK2, lw=0.5, ls=":")
        # straight line over all ages, and the spline trend
        b = np.polyfit(age, y, 1)
        xx = np.array([age.min(), age.max()])
        ax.plot(xx, np.polyval(b, xx) / np.log(10), color=INK2, lw=0.8, ls="--")
        xs, ys = spline_trend(age, y)
        ax.plot(xs, ys / np.log(10), color=INK, lw=1.4)
        lo, hi = np.percentile(y / np.log(10), [1, 99])
        ax.set_ylim(lo - 0.14 * (hi - lo), hi + 0.3 * (hi - lo))
        ax.set_xlabel("Age (years)")
        if j == 0:
            ax.set_ylabel("Eyes-closed alpha (log$_{10}$)")
            panel(ax, "a")
        ax.set_title(lab, loc="left", color=col if j else INK2)
        # least-squares slope within each age range
        for m, x0, ha, name in ((age < cut, cut - 0.4, "right", f"under {cut:.0f} y"),
                                (age >= cut, cut + 0.4, "left", f"{cut:.0f} y and over")):
            sl = np.polyfit(age[m], y[m], 1)[0]
            ax.text(x0, 0.97, f"{name}\n{pct(sl):+.1f}% per year",
                    transform=ax.get_xaxis_transform(), ha=ha, va="top", color=INK,
                    fontsize=6, linespacing=1.15, bbox=box)
        ax.text(0.98, 0.03, f"all ages {pct(b[0]):+.1f}% per year", transform=ax.transAxes,
                ha="right", color=INK2, fontsize=6, bbox=box)

    # (b) slope per year as a function of the assumed lambda: all ages (left)
    # and the two age ranges, eyes closed (right)
    grid = np.round(np.arange(-0.3, 1.5001, 0.1), 2)
    lo_x, hi_x = -0.35, 1.55

    def curve(ax, w_, cond, col, ls, lab, band=True, lw=1.2):
        w2 = w_[w_[f"a_{cond}"] > 0]
        x = w2.age.to_numpy()
        la, lb = np.log(w2[f"a_{cond}"].to_numpy()), np.log(w2[f"b_{cond}"].to_numpy())
        s, lo, hi = slope_curve(x, la, lb, grid, rng)
        ax.plot(grid, pct(s), color=col, ls=ls, lw=lw, label=lab)
        if band:
            ax.fill_between(grid, pct(lo), pct(hi), color=col, alpha=0.12, lw=0)

    def cross(ax, r, col, dx, dy, ha):
        interval(ax, r.hdi_lo, r.hdi_hi, 0, lo_x, hi_x, col, ms=4.5, lw=3.5, alpha=0.5,
                 solid_capstyle="butt")
        ax.plot(r.lam_star, 0, "o", color=col, ms=4)
        ax.annotate(f"λ* = {r.lam_star:.2f}", (r.lam_star, 0), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, color=col, fontsize=6,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    axl = fig.add_subplot(gs[1, 0:2])
    axr = fig.add_subplot(gs[1, 2:4], sharey=axl)
    curve(axl, wf, "ec", INK, "-", "eyes closed")
    curve(axl, wf, "eo", INK2, "-", "eyes open")
    curve(axl, wk, "ec", INK, "--", "eyes closed, knee+plateau", band=False)
    cr = pd.read_csv(os.path.join(RES, "hbn_age_alpha_crossover.csv"))
    for cond, col, where in (("ec", INK, (-4, 6, "right")), ("eo", INK2, (5, -11, "left"))):
        r = cr[(cr["sample"] == "qc") & (cr.model == "fixed") & (cr.cond == cond)].iloc[0]
        cross(axl, r, col, *where)
    curve(axr, wf[wf.age < cut], "ec", YOUNG, "-", f"under {cut:.0f} y")
    curve(axr, wf[wf.age >= cut], "ec", OLD, "-", f"{cut:.0f} y and over")
    curve(axr, wf, "ec", INK, "-", "all ages", band=False, lw=0.7)
    ph = pd.read_csv(os.path.join(RES, "hbn_age_alpha_phases.csv"))
    ph = ph[(ph.model == "fixed") & (ph.cond == "ec") & (ph.cut == cut)].set_index("range")
    cross(axr, ph.loc["younger"], YOUNG, -4, 6, "right")
    cross(axr, ph.loc["older"], OLD, 5, -11, "left")
    for ax, title, loc in ((axl, "All ages", "upper left"),
                           (axr, "Eyes closed, by age range", "lower right")):
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlim(lo_x, hi_x)
        ax.set_xticks([0, 0.5, 1, 1.5])
        ax.set_xticklabels(["0\nIRASA-like", "0.5", "1\nspecparam-like", "1.5"])
        for t, col in ((ax.get_xticklabels()[0], C0), (ax.get_xticklabels()[2], C1)):
            t.set_color(col)
        ax.set_xlabel("Assumed coupling λ")
        ax.set_title(title, loc="left", color=INK2)
        ax.legend(loc=loc, handlelength=1.6, borderaxespad=0.2)
    axl.set_ylim(-19, 22)
    axl.set_ylabel("Change in alpha with age (% per year)")
    panel(axl, "b")

    # (c) age-resolved slopes, eyes closed
    ax = fig.add_subplot(gs[1, 4:6])
    sp = pd.read_csv(os.path.join(RES, "hbn_age_alpha_spline.csv"))
    for lam, col, lab in ((0.0, C0, "λ = 0"), (1.0, C1, "λ = 1")):
        g = sp[(sp.cond == "ec") & (sp.lam == lam)]
        ax.plot(g.age, pct(g.deriv), "-o", color=col, ms=2.5, lw=1.2, label=lab)
        ax.fill_between(g.age, pct(g.lo), pct(g.hi), color=col, alpha=0.15, lw=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(cut, color=INK2, lw=0.5, ls=":")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Eyes-closed alpha (% per year)")
    ax.legend(loc="lower left")
    ax.set_title("Slope by age (spline derivative)", loc="left", color=INK2)
    panel(ax, "c")
    save(fig, outdir, "fig4_age_reversal")


# ---- Figure 5 -----------------------------------------------------------
# ---- Supplementary Figures 1 and 2 ------------------------------------
def figs1(outdir):
    """Age slope of alpha under lambda = 0 and 1 across estimators and samples."""
    R = pd.read_csv(os.path.join(RES, "hbn_age_alpha_robust.csv"))
    labels = {"ols": "OLS (HC3)", "boot": "OLS, bootstrap CI", "huber": "Huber M",
              "theilsen": "Theil-Sen", "wls": "WLS (split-half weights)",
              "covar": "+ sex, release, data quantity",
              "release_re": "random effects over releases",
              "age5_16": "ages 5-16 only", "quad": "quadratic, at median age"}
    ests = list(labels)
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.85, 3.0), sharey=True)
    for ax, cond, title in ((axs[0], "ec", "Eyes closed"), (axs[1], "eo", "Eyes open")):
        for lam, col, dy in ((0.0, C0, 0.17), (1.0, C1, -0.17)):
            for sample, mk in (("qc", "o"), ("all", "o")):
                d = R[(R["sample"] == sample) & (R.model == "fixed") & (R.cond == cond)
                      & (R.lam == lam)]
                for j, e in enumerate(ests):
                    r = d[d.estimator == e]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    y = j + dy + (-0.07 if sample == "qc" else 0.07)
                    f = lambda v: 100 * (np.exp(v) - 1)
                    ax.plot([f(r.lo), f(r.hi)], [y, y], color=col, lw=0.9,
                            alpha=1 if sample == "qc" else 0.45)
                    ax.plot(f(r.est), y, mk, color=col, ms=3,
                            mfc=col if sample == "qc" else "white")
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(range(len(ests)))
        ax.set_yticklabels([labels[e] for e in ests])
        ax.set_xlabel("Age slope of alpha power (% per year)")
        ax.set_title(title, loc="left", color=INK2)
    axs[0].invert_yaxis()          # shared y axis: invert once
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=C0, marker="o", ls="-", ms=3, label="λ = 0"),
        Line2D([], [], color=C1, marker="o", ls="-", ms=3, label="λ = 1"),
        Line2D([], [], color=INK2, marker="o", ls="none", ms=3,
               label=f"quality-controlled (n = {int(Q.qc_ok.sum()):,})"),
        Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", ms=3,
               label=f"all (n = {len(Q):,})")], loc="lower center", ncol=4,
               bbox_to_anchor=(0.55, -0.05))
    panel(axs[0], "a"); panel(axs[1], "b")
    fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1))
    save(fig, outdir, "figS1_age_robustness")


def figs2(outdir):
    """Sign of the intrinsic age trend over true coupling and gain share."""
    cr = pd.read_csv(os.path.join(RES, "hbn_age_alpha_crossover.csv"))
    bands = pd.read_csv(os.path.join(RES, "hbn_age_alpha_bands.csv"))
    lam = np.linspace(0, 1, 201)
    phi = np.linspace(0, 1, 201)
    L, F = np.meshgrid(lam, phi)
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.7, 2.4), sharey=True)
    for ax, cond, title in ((axs[0], "ec", "Eyes closed"), (axs[1], "eo", "Eyes open")):
        r = cr[(cr["sample"] == "qc") & (cr.model == "fixed") & (cr.cond == cond)].iloc[0]
        declines = (1 - L) * (1 - F) > 1 - r.lam_star
        ax.contourf(L, F, declines.astype(float), levels=[-0.5, 0.5, 1.5],
                    colors=["#f6d7c9", "#cfe0f5"])
        ax.contour(L, F, (1 - L) * (1 - F), levels=[1 - r.lam_star], colors=[INK],
                   linewidths=1)
        pr = bands[(bands.measure.str.startswith("gain reference")) &
                   (bands.cond == cond)].est.iloc[0]
        ax.axhline(pr, color=INK2, lw=0.8, ls="--")
        ax.text(0.98, pr + 0.02, f"30-45 Hz reference φ = {pr:.2f}", ha="right",
                va="bottom", fontsize=6, color=INK2)
        ax.text(0.04, 0.05, "intrinsic alpha\ndeclines", color=C0, fontsize=6)
        if not declines.all():         # lambda* >= 1 leaves no region where it rises
            ax.text(0.96, 0.9, "rises", color=C1, fontsize=6, ha="right")
        ax.set_xlabel("True coupling λ")
        ax.set_title(f"{title} (λ* = {r.lam_star:.2f})", loc="left", color=INK2)
    axs[0].set_ylabel("Share of background age slope\nthat is scalp gain (φ)")
    panel(axs[0], "a"); panel(axs[1], "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, outdir, "figS2_gain_tipping")


# ---- Figure 6 -----------------------------------------------------------
def fig_claims(outdir):
    """Published claims re-tested under both rules: crossover lambda* per claim."""
    from matplotlib.lines import Line2D
    S = pd.read_csv(os.path.join(RES, "breadth_summary.csv"))
    fig, ax = plt.subplots(figsize=(W2 * 0.72, 4.0))
    lo_x, hi_x = -1.0, 4.0
    ax.axvspan(0, 1, color=CH, alpha=0.12, lw=0)
    ax.text(0.5, len(S) - 1.0, "assumption\ndecides", ha="center", va="center", fontsize=5.5,
            color=INK2)
    # a claim is placed by the probability that its crossover lies in [0, 1]
    place = lambda p: "inside" if p >= 0.95 else ("outside" if p <= 0.05 else "undetermined")
    col = {"inside": C1, "undetermined": INK, "outside": GREY}
    for y, r in enumerate(S.itertuples()):
        cls = place(r.p_cross_in_01)
        c = col[cls]
        # an end is marked unless the point itself sits there as a triangle
        interval(ax, r.hdi_lo, r.hdi_hi, y, lo_x, hi_x, c, lw=1,
                 ends=(r.lam_star >= lo_x, r.lam_star <= hi_x))
        x = np.clip(r.lam_star, lo_x, hi_x)
        mk = ">" if r.lam_star > hi_x else ("<" if r.lam_star < lo_x else "o")
        ax.plot(x, y, mk, color=c, ms=4 if mk == "o" else 5,
                mfc="white" if cls == "outside" else c)
        if not getattr(r, "lam_star_bounded", True):
            # effect on ln b indistinguishable from 0: no finite crossover interval
            ax.text(x + (-0.12 if mk == ">" else 0.12), y, "unbounded", fontsize=5,
                    color=INK2, va="center", ha="right" if mk == ">" else "left")
    ax.set_yticks(range(len(S)))
    ax.set_yticklabels([f"{r.claim}\n{r.dataset}, n = {r.n:,}" for r in S.itertuples()],
                       fontsize=5.5)
    ax.set_xlim(lo_x - 0.1, hi_x + 0.1)
    ax.set_ylim(len(S) - 0.4, -0.6)
    ax.set_xlabel("Crossover λ* (assumed λ at which the effect changes sign)")
    ax.legend(handles=[Line2D([], [], color=C1, marker="o", ls="-", ms=4, label="P(0 ≤ λ* ≤ 1) ≥ 0.95"),
                       Line2D([], [], color=INK, marker="o", ls="-", ms=4, label="0.05 to 0.95"),
                       Line2D([], [], color=GREY, marker="o", mfc="white", ls="-", ms=4,
                              label="≤ 0.05"),
                       Line2D([], [], color=GREY, marker=">", ls="", ms=3.5, mec="none",
                              label="continues past the axis")],
              loc="upper center", bbox_to_anchor=(0.3, -0.09), ncol=4, fontsize=6,
              columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    save(fig, outdir, "fig3_published_claims")


def fig_estimates(outdir):
    """Every attempt to estimate lambda, grouped by design.

    Everything drawn comes from results/identification_summary.csv. A row
    that carries lam_alt (the two-condition estimate with the instrument from
    the other half of the recording) is drawn as a pair: both estimates with
    their intervals, joined by a band that spans the range between them.
    """
    from matplotlib.lines import Line2D
    I = pd.read_csv(os.path.join(RES, "identification_summary.csv"))
    for c in ("lam_alt", "lo_alt", "hi_alt", "n_patients"):
        if c not in I:
            I[c] = np.nan
    lo_x, hi_x, dy = -0.9, 2.2, 0.19
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 5.0))
    ax.axvline(0, color=C0, lw=0.8, ls="--")
    ax.axvline(1, color=C1, lw=0.8, ls="--")
    order = ["between conditions: eyes closed vs open",
             "within session: epoch fluctuations, eyes open",
             "within session: segment fluctuations, eyes closed",
             "within session: segment fluctuations, eyes open",
             "within session: epoch fluctuations, intracranial",
             "between sessions: test-retest",
             "between doses: propofol baseline vs moderate"]
    headers = {order[0]: "HBN, eyes closed vs open (2,396 children)",
               order[1]: "ds003690, eyes open (within session)",
               order[2]: "Dortmund, eyes closed (within session)",
               order[3]: "Dortmund, eyes open (within session)",
               order[4]: "Intracranial rest (within session)",
               order[5]: "Test-retest (between sessions)",
               order[6]: "Propofol, baseline vs moderate (20 volunteers)"}
    y, ticks, labels, paired = 0, [], [], False
    for g in order:
        G_ = I[I.design == g]
        if G_.empty:
            continue
        ax.text(lo_x + 0.05, y, headers[g], fontsize=6, color=INK, fontweight="bold", va="center",
                bbox=dict(fc="white", ec="none", pad=0.4))
        y += 1
        for r in G_.itertuples():
            spec = (r.spec.replace("before 2-h tasks", "before tasks")
                    .replace("after 2-h tasks", "after tasks")
                    .replace("covariates: ", "").replace("censor ", "")
                    .replace("flanks 3-6, 26-36", "flanks"))
            if np.isfinite(r.n_patients):
                # channels of this row come from this many patients
                spec = spec.replace(" channels", f" channels, {int(r.n_patients)} patients")
            if np.isfinite(r.lam_alt):
                paired = True
                ax.plot([r.lam_alt, r.lam], [y, y], color=GREY, lw=5, alpha=0.4,
                        solid_capstyle="butt", zorder=1)
                interval(ax, r.lo, r.hi, y - dy, lo_x, hi_x, INK, ms=3, lw=1)
                ax.plot(r.lam, y - dy, "o", color=INK, ms=3)
                interval(ax, r.lo_alt, r.hi_alt, y + dy, lo_x, hi_x, INK, ms=3, lw=1)
                ax.plot(r.lam_alt, y + dy, "o", color=INK, mfc="white", ms=3, mew=0.8)
            else:
                interval(ax, r.lo, r.hi, y, lo_x, hi_x, INK, lw=1)
                ax.plot(r.lam, y, "o", color=INK, ms=3.5)
            ticks.append(y)
            labels.append(spec)
            y += 1
        y += 0.4
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_ylim(y - 0.6, -0.8)
    ax.set_xlim(lo_x, hi_x)
    ax.set_xlabel("Estimated λ (95% interval; 0 additive, 1 multiplicative)")
    handles = [Line2D([], [], color=INK, marker=">", ls="-", lw=1, ms=3, mec="none",
                      markevery=[1], label="interval continues past the axis")]
    if paired:
        handles = [Line2D([], [], color=INK, marker="o", ls="-", lw=1, ms=3,
                          label="instrument from the half of the band total"),
                   Line2D([], [], color=INK, marker="o", mfc="white", mew=0.8, ls="-", lw=1,
                          ms=3, label="instrument from the half of the background"),
                   Line2D([], [], color=GREY, lw=5, alpha=0.4, solid_capstyle="butt",
                          label="range between the two")] + handles
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, -0.075), ncol=1,
              fontsize=5.5, handlelength=2.2)
    fig.tight_layout()
    save(fig, outdir, "fig4_estimates_by_design")


def figs_spatial(outdir):
    """The spatial test and its null, fitted and fit-free (HBN eyes closed vs open)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.7, 2.3))
    ax = axs[0]
    specs = [("2-40", ""), ("4-40", "_f4-40"), ("5-40", "_f5-40"), ("2-30", "_f2-30")]
    null = pd.read_csv(os.path.join(RES, "sim_topography_null.csv"))
    for j, (lab, tag) in enumerate(specs):
        lo_, hi_ = (float(v) for v in lab.split("-"))
        nn = null[(null.fit_lo == lo_) & (null.fit_hi == hi_)].r_exp_alpha
        null_bar(ax, j, nn.min(), nn.max())
        d = np.load(os.path.join(RES, f"hbn_topography{tag}.npz"))
        mE, mA = np.nanmean(d["d_exponent"], 0), np.nanmean(d["d_log_a"], 0)
        g = np.isfinite(mE) & np.isfinite(mA)
        ax.plot(j, np.corrcoef(mE[g], mA[g])[0, 1], "o", color=C0, ms=5)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels([s_[0] + " Hz" for s_ in specs])
    ax.set_xlabel("Aperiodic fit range")
    ax.set_ylabel("Spatial r, Δexponent vs Δalpha maps")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=C0, ms=5, label="HBN"),
                       Patch(color=GREY, alpha=0.6,
                             label="simulations with no\nbackground change")],
              loc="lower left")
    ax.set_title("Fitted-exponent test vs its null", loc="left", color=INK2)
    panel(ax, "a")
    ax = axs[1]
    fr = pd.read_csv(os.path.join(RES, "hbn_topography_flanks.csv"))
    fn = pd.read_csv(os.path.join(RES, "sim_topography_null_flanks.csv"))
    for j, (fl, lab) in enumerate((("low", "2-4 Hz"), ("high", "30-40 Hz"))):
        nn = fn[fn.flank == fl].r_flank_alpha
        null_bar(ax, j, nn.min(), nn.max())
        r = fr[(fr.flank == fl) & (fr.group == "all")].iloc[0]
        ax.errorbar(j, r.r, yerr=[[r.r - r.lo], [r.hi - r.r]], fmt="o", color=C0,
                    ms=5, lw=1, capsize=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2-4 Hz", "30-40 Hz"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_xlabel("Flank band (no fitting)")
    ax.set_ylabel("Spatial r, Δflank vs Δalpha maps")
    ax.set_title("Fit-free test vs its null", loc="left", color=INK2)
    panel(ax, "b")
    fig.tight_layout(w_pad=3.0)
    save(fig, outdir, "figS3_spatial_test")


# ---- Figure 5 ------------------------------------------------
MECH_LABELS = {
    "gain": "Gain (skull, electrodes)",
    "synaptic_gain": "Synaptic gain (all currents)",
    "tau_i": "Slower GABA$_A$ decay, rhythm in I",
    "gi_rhythm_i": "Stronger inhibition, rhythm in I",
    "gi_rhythm_e": "Stronger inhibition, rhythm in E",
    "drive_relative": "More drive, rhythm a fixed fraction",
    "drive_absolute": "More drive, rhythm of fixed size",
    "separate_common_gain": "Separate generator, common gain",
    "separate_background_drive": "Separate generator, more background drive",
}


def figs3(outdir):
    """What sets lambda in a synaptic model of the EEG (results/sim_mechanisms.csv)."""
    S = pd.read_csv(os.path.join(RES, "sim_mechanisms.csv")).set_index("scenario")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    from matplotlib.lines import Line2D

    ax = axs[0]
    for x, c in ((0, C0), (1, C1)):
        ax.axvline(x, color=c, lw=0.8, ls="--")
    ax.axvline(2, color=GREY, lw=0.6, ls=":")
    est = (("lf_true", "o", INK, "true background", -0.2),
           ("lf_fixed", "s", INK2, "fitted, power law", 0.0),
           ("lf_knee", "^", GREY, "fitted, knee", 0.2))
    names = [n for n in MECH_LABELS if n in S.index]
    for i, n in enumerate(names):
        r = S.loc[n]
        ax.plot([r.analytic] * 2, [i - 0.36, i + 0.36], color=INK, lw=1.6,
                solid_capstyle="butt", alpha=0.35, zorder=1)
        for col, mk, c, _, dy in est:
            ax.plot([r[col + "_lo"], r[col + "_hi"]], [i + dy, i + dy], color=c, lw=0.8)
            ax.plot(r[col], i + dy, mk, color=c, ms=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([MECH_LABELS[n] for n in names], fontsize=6)
    ax.set_ylim(len(names) - 0.5, -0.6)
    ax.set_xlim(-0.4, 2.5)
    ax.set_xlabel("Coupling exponent of the change (λ)")
    ax.set_title("One parameter differs between conditions", loc="left", color=INK2)
    ax.legend(handles=[Line2D([], [], marker="|", ls="", color=INK, alpha=0.35, ms=7, mew=1.6,
                              label="predicted")] +
              [Line2D([], [], marker=mk, color=c, ls="-", lw=0.8, ms=3, label=lab)
               for _, mk, c, lab, _ in est],
              loc="lower right", fontsize=5.5, handlelength=1.4)
    panel(ax, "a")

    ax = axs[1]
    ax.plot([0, 2], [0, 2], color=GREY, lw=0.6)
    groups = ((["mix_gain0.25", "mix_gain0.5", "mix_gain0.75"], "o",
               "gain + drive: share of\nbackground change from gain", "{:.0%}"),
              (["sources_phi0.2", "sources_phi0.5", "sources_phi0.9"], "s",
               "two sources: share of the rhythm\nin the changing source", "{:.0%}"))
    for keys, mk, lab, fmt in groups:
        keys = [k for k in keys if k in S.index]
        for k in keys:
            r = S.loc[k]
            ax.plot([r.ols_true, r.ols_true], [r.lf_true_lo, r.lf_true_hi], color=INK, lw=0.8)
            ax.plot(r.ols_true, r.lf_true, mk, color=INK, ms=3.5, mfc="white" if mk == "s" else INK)
            mix = k.startswith("mix")
            share = float(k.split("gain")[-1]) if mix else float(k.split("phi")[-1])
            # mixture labels to the right of the points, source labels to the left
            ax.text(r.ols_true + (0.06 if mix else -0.06), r.lf_true, fmt.format(share),
                    fontsize=5.5, color=INK2, va="center", ha="left" if mix else "right")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Expected λ (slope of true Δln a on Δln b)")
    ax.set_ylabel("Estimated λ (log-free, 95% interval)")
    ax.set_title("Mechanisms or sources mixed", loc="left", color=INK2)
    ax.legend(handles=[Line2D([], [], marker="o", color=INK, ls="", ms=3.5, label=groups[0][2]),
                       Line2D([], [], marker="s", color=INK, mfc="white", ls="", ms=3.5,
                              label=groups[1][2])],
              loc="upper left", fontsize=5.5, handletextpad=0.3)
    panel(ax, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, outdir, "figS3_mechanisms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--only", default="1,2,3,4,5,11,12,13,14,15")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    todo = {int(x) for x in a.only.split(",")}
    for k, fn in ((1, fig1), (2, fig4), (3, fig_claims), (4, fig_estimates), (5, figs3),
                  (11, fig2), (12, fig3), (13, figs_spatial), (14, figs1), (15, figs2)):
        if k in todo:
            fn(a.outdir)


if __name__ == "__main__":
    main()
