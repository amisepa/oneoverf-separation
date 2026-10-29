"""Alpha-vs-age crossover lambda* at every scalp channel, and the order of
channel averaging and background fitting.

The primary developmental analysis (hbn_age_alpha_robust.py) averages the
spectra of six posterior channels and fits one background to the mean. Here
the same analysis is run at each of the 129 channels, and three channel sets
are combined in different orders.

For each participant passing quality control (results/hbn_qc_flags.csv),
eyes closed and eyes open, and each channel: background fitted by Whittle
deviance over 2-55 Hz with 6-16 Hz censored (power law; knee+plateau as the
alternative), as in hbn_kp.py; total power T and background b in the band
IAF +/- 2 Hz; a = T - b. IAF is the eyes-closed individual alpha frequency
of the ROI mean spectrum (hbn_kp.find_iaf), the same band at every channel
(band "iaf_roi", the primary band), or of the channel's own eyes-closed
spectrum (band "iaf_chan").

Over the participants with a > 0 at a channel: s(0) = OLS age slope of ln a,
s(1) of ln a - ln b, s_b of ln b, with HC3 intervals; lambda* = s(0) / s_b,
its Bayesian-bootstrap 95% HDI, and the shares of draws in [0, 1] (p01),
above 1 and below 0. lambda* has no interval when the HC3 interval of s_b
includes 0 (bounded False). A channel is reliable when a > 0 in at least 80%
of participants; elsewhere a is mostly a fit residual.

The Dirichlet weights of each draw are shared by all channels and
configurations, so the draws are joint and support simultaneous statements
over the reliable channels:
  frac01      share of channels whose lambda* lies in [0, 1] (point
              estimate; median and 95% interval over draws)
  m_eff       effective number of independent channels (Li & Ji 2005) from
              the rank correlation of the per-channel lambda* draws
  clusters    spatially contiguous groups (Delaunay neighbours of the
              GSN-HydroCel-129 positions) of channels with p01 >= 0.95, and
              of channels with P(lambda* > 1) >= 0.95

Averaging orders (band iaf_roi), for the posterior ROI (E62 E65 E70 E75 E83
E90), the 19 channels nearest the 10-20 positions, and all 129 channels:
  spectrum_mean    average the channel spectra, then fit (the primary order)
  mean_log         fit each channel, average ln a and ln b over the channels
                   (participants with a > 0 at every channel of the set)
  log_mean         fit each channel, average a and b over the channels,
                   then take logs
  median_lam_star  lambda* at each reliable channel of the set, median over
                   those channels
plus the single channels nearest Fz, Pz and Oz. Each configuration's
difference from the primary (ROI, spectrum_mean) is computed from the same
draws: diff_primary with its HDI (diff_hdi_lo, diff_hdi_hi) and
p_diff_pos = P(difference > 0). p01_all_roi (summary, band iaf_roi) is the
share of draws with lambda* in [0, 1] at all six ROI channels at once.

Slopes are in natural-log units per year. Writes
results/hbn_age_crossover_channels.csv (per channel),
results/hbn_age_crossover_averaging.csv (averaging orders),
results/hbn_age_crossover_channels_summary.csv (joint statements) and the
per-participant band powers results/hbn_age_crossover_channels_fits_<model>.npz
(kept out of the repository). With --channels sets only the ROI and 10-20
channels are fitted (used for knee+plateau, which is ~7 times slower per
fit); the fit-then-average orders for all 129 channels and the joint
statements over channels then need the power law.

Usage: python hbn_age_crossover_channels.py [--psd-dir DIR] [--workers 12]
           [--models fixed,knee_plateau] [--nboot 2000] [--from-fits]
           [--limit N] [--channels all|sets]
"""
import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ap_models import ap_eval, fit_aperiodic
from hbn_kp import CENSOR, FIT_RANGE, ROI, find_iaf, mask_for
from lambda_curve import hdi

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CH = [f"E{i}" for i in range(1, 129)] + ["Cz"]
# GSN-HydroCel-129 channels nearest the 10-20 positions
TEN_TWENTY = {"Fp1": "E22", "Fp2": "E9", "F7": "E33", "F3": "E24", "Fz": "E11",
              "F4": "E124", "F8": "E122", "T7": "E45", "C3": "E36", "Cz": "Cz",
              "C4": "E104", "T8": "E108", "P7": "E58", "P3": "E52", "Pz": "E62",
              "P4": "E92", "P8": "E96", "O1": "E70", "O2": "E83"}
SETS = {"roi": sorted(ROI, key=CH.index), "10-20": list(TEN_TWENTY.values()), "all": CH}
SINGLE = {"Fz": "E11", "Pz": "E62", "Oz": "E75"}
CONDS = ("ec", "eo")
RELIABLE = 0.8
P_MARK = 0.95
warnings.filterwarnings("ignore")


def fits_path(model):
    return os.path.join(RES, f"hbn_age_crossover_channels_fits_{model}.npz")


def subject_fit(args):
    """Band totals and backgrounds of one participant at every channel and for
    every channel-set mean spectrum; (dict or None, error or None)."""
    path, models, fit_ch = args
    try:
        d = np.load(path, allow_pickle=True)
        f0 = d["freqs"].astype(float)
        sel = (f0 >= FIT_RANGE[0]) & (f0 <= FIT_RANGE[1])
        f = f0[sel]
        pos = {str(c): i for i, c in enumerate(d["ch_names"])}
        if any(c not in pos for c in CH):
            return None, f"{os.path.basename(path)}: missing channels"
        ix = np.array([pos[c] for c in CH])
        raw = {c: d[c][ix][:, sel].astype(float) for c in CONDS}
        sets = {s: np.array([CH.index(c) for c in chans]) for s, chans in SETS.items()}
        iaf = find_iaf(raw["ec"][sets["roi"]].mean(0), f)
        if not np.isfinite(iaf):
            return None, None
        iaf_ch = np.array([find_iaf(raw["ec"][k], f) for k in range(len(CH))])
        keep = mask_for(f, "censor", [CENSOR])
        b_roi = (f >= iaf - 2) & (f <= iaf + 2)
        out = dict(subject=str(d["subject"]), age=float(d["age"]), iaf_roi=iaf,
                   iaf_chan=iaf_ch)
        nan = lambda *s: np.full(s, np.nan)
        for model in models:
            r = {k: nan(len(CONDS), len(CH)) for k in ("tot_roi", "b_roi", "tot_chan", "b_chan")}
            r["set_tot"], r["set_b"] = nan(len(CONDS), len(SETS)), nan(len(CONDS), len(SETS))
            for j, cond in enumerate(CONDS):
                P = np.clip(np.nan_to_num(raw[cond], nan=1e-12), 1e-12, None)
                for k in fit_ch:
                    fit = fit_aperiodic(P[k], f, keep, model)
                    if fit is None:
                        continue
                    L = ap_eval(fit["theta"], f, model)
                    r["tot_roi"][j, k], r["b_roi"][j, k] = P[k, b_roi].mean(), L[b_roi].mean()
                    if np.isfinite(iaf_ch[k]):
                        m = (f >= iaf_ch[k] - 2) & (f <= iaf_ch[k] + 2)
                        r["tot_chan"][j, k], r["b_chan"][j, k] = P[k, m].mean(), L[m].mean()
                for s, (name, idx) in enumerate(sets.items()):
                    Pm = np.clip(np.nan_to_num(raw[cond][idx].mean(0), nan=1e-12), 1e-12, None)
                    fit = fit_aperiodic(Pm, f, keep, model)
                    if fit is None:
                        continue
                    L = ap_eval(fit["theta"], f, model)
                    r["set_tot"][j, s], r["set_b"][j, s] = Pm[b_roi].mean(), L[b_roi].mean()
            out[model] = r
        return out, None
    except Exception as e:
        return None, f"{os.path.basename(path)}: {type(e).__name__}: {e}"


def run_fits(psd_dir, models, workers, ok_ids, limit=0, channels="all"):
    fit_ch = (list(range(len(CH))) if channels == "all" else
              sorted({CH.index(c) for c in SETS["roi"] + SETS["10-20"]}))
    files = sorted(glob.glob(os.path.join(psd_dir, "*.npz")))
    jobs = [(p, models, fit_ch) for p in files
            if os.path.basename(p).split("_", 1)[1].replace(".npz", "") in ok_ids]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} PSD files of participants passing QC, models {models}, "
          f"{len(fit_ch)} channels", flush=True)
    rows = []
    with Pool(workers) as pool:
        for k, (r, err) in enumerate(pool.imap(subject_fit, jobs, chunksize=4)):
            if err:
                print(f"  {err}", flush=True)
            if r is not None:
                rows.append(r)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)}", flush=True)
    seen, uniq = set(), []
    for r in rows:
        if r["subject"] not in seen:
            seen.add(r["subject"])
            uniq.append(r)
    for model in models:
        z = dict(subject=np.array([r["subject"] for r in uniq]),
                 age=np.array([r["age"] for r in uniq]),
                 iaf_roi=np.array([r["iaf_roi"] for r in uniq]),
                 iaf_chan=np.stack([r["iaf_chan"] for r in uniq]))
        for key in uniq[0][model]:
            z[key] = np.stack([r[model][key] for r in uniq])
        np.savez_compressed(fits_path(model), channels=np.array(CH),
                            sets=np.array(list(SETS)), conds=np.array(CONDS), **z)
        print(f"wrote {fits_path(model)} ({len(uniq)} participants)", flush=True)


def wslope(x, Y, M, w):
    """Weighted least-squares slope of each column of Y on x over the rows
    where M is true; Y must be finite wherever M is."""
    W = M * w[:, None]
    Y = np.where(M, Y, 0.0)
    Sw, Sx, Sxx = W.sum(0), x @ W, (x * x) @ W
    WY = W * Y
    Sy, Sxy = WY.sum(0), x @ WY
    return (Sw * Sxy - Sx * Sy) / (Sw * Sxx - Sx ** 2)


def hc3(x, y):
    r = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    lo, hi = r.conf_int()[1]
    return r.params[1], lo, hi, r.pvalues[1]


def crossover(x, A, B, G, base=None):
    """Point estimates and joint Bayesian-bootstrap draws of lambda* for each
    column of A (periodic) and B (background) power.

    x: age (n,); A, B: (n, C); G: Dirichlet weights (nboot, n); base: (n, C)
    mask of participants with a usable fit (default: finite A and B).
    Returns per-column rows and the draws (nboot, C).
    """
    base = np.isfinite(A) & np.isfinite(B) & (B > 0) if base is None else base
    M = base & (A > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        LA, LB = np.log(np.where(M, A, 1.0)), np.log(np.where(M, B, 1.0))
    one = np.ones(x.size)
    s_a, s_b = wslope(x, LA, M, one), wslope(x, LB, M, one)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.array([wslope(x, LA, M, g) / wslope(x, LB, M, g) for g in G])
    rows = []
    for k in range(A.shape[1]):
        m = M[:, k]
        n_all, n = int(base[:, k].sum()), int(m.sum())
        row = dict(n_all=n_all, n=n, share_pos=n / n_all if n_all else np.nan)
        if n < 20:
            rows.append(row)
            continue
        xa, la, lb = x[m], LA[m, k], LB[m, k]
        ok = base[:, k]
        row["median_a_over_b"] = float(np.median(A[ok, k] / B[ok, k]))
        for key, y in (("s0", la), ("s1", la - lb), ("s_b", lb)):
            row[key], row[f"{key}_lo"], row[f"{key}_hi"], row[f"p_{key}"] = hc3(xa, y)
        row["bounded"] = bool(np.sign(row["s_b_lo"]) == np.sign(row["s_b_hi"]))
        row["lam_star"] = s_a[k] / s_b[k]
        dk = draws[:, k]
        row["hdi_lo"], row["hdi_hi"] = hdi(dk)
        row["p01"] = float(np.mean((dk >= 0) & (dk <= 1)))
        row["p_above1"] = float(np.mean(dk > 1))
        row["p_below0"] = float(np.mean(dk < 0))
        rows.append(row)
    return rows, draws


def li_ji(C):
    """Effective number of independent tests from a correlation matrix
    (Li & Ji 2005)."""
    ev = np.abs(np.linalg.eigvalsh(C))
    return float(np.sum((ev >= 1) + (ev - np.floor(ev))))


def adjacency():
    import mne
    info = mne.create_info(CH, 250.0, "eeg")
    info.set_montage("GSN-HydroCel-129", verbose="ERROR")
    with mne.utils.use_log_level("error"):
        adj, names = mne.channels.find_ch_adjacency(info, "eeg")
    assert list(names) == CH
    return adj.toarray().astype(bool)


def clusters(mask, adj):
    """Connected groups of the channels in mask; list of channel-name lists,
    largest first."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    _, lab = connected_components(adj[np.ix_(idx, idx)], directed=False)
    groups = [[CH[idx[i]] for i in np.where(lab == g)[0]] for g in np.unique(lab)]
    return sorted(groups, key=len, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-dir", default=os.environ.get("HBN_OUT", "hbn_psd"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--models", default="fixed,knee_plateau")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--from-fits", action="store_true",
                    help="read the per-participant band powers instead of refitting")
    ap.add_argument("--limit", type=int, default=0, help="fit only the first N files")
    ap.add_argument("--channels", choices=("all", "sets"), default="all",
                    help="fit every channel, or only the ROI and 10-20 channels "
                         "(set-mean spectra are always fitted)")
    a = ap.parse_args()
    models = a.models.split(",")
    Q = pd.read_csv(os.path.join(RES, "hbn_qc_flags.csv"), index_col=0)
    ok_ids = set(Q.index[Q.qc_ok])
    if not a.from_fits:
        run_fits(a.psd_dir, models, a.workers, ok_ids, a.limit, a.channels)
    adj = adjacency()

    chan_rows, avg_rows, sum_rows = [], [], []
    for model in models:
        z = np.load(fits_path(model), allow_pickle=True)
        keep = np.isin(z["subject"], list(ok_ids))
        x = z["age"][keep]
        G = np.random.default_rng(0).dirichlet(np.ones(x.size), size=a.nboot)
        print(f"\n##### {model}: {x.size} participants #####")
        for j, cond in enumerate(CONDS):
            draws_by_band = {}
            for band in ("iaf_roi", "iaf_chan"):
                T, B = z[f"tot_{band[4:]}"][keep, j], z[f"b_{band[4:]}"][keep, j]
                rows, draws = crossover(x, T - B, B, G)
                draws_by_band[band] = (rows, draws, T, B)
                for ch, r in zip(CH, rows):
                    chan_rows.append(dict(model=model, band=band, cond=cond, channel=ch, **r))
                R = pd.DataFrame(rows)
                rel = (R.share_pos >= RELIABLE).to_numpy()
                s = dict(model=model, band=band, cond=cond,
                         n_fitted=int((R.n_all > 0).sum()), n_reliable=int(rel.sum()))
                if not rel.any() or s["n_fitted"] < len(CH):
                    # joint statements over channels need every channel fitted
                    sum_rows.append(s)
                    continue
                lam = R.lam_star.to_numpy()
                in01 = (draws >= 0) & (draws <= 1)
                frac = in01[:, rel].mean(1)
                rk = stats.rankdata(draws[:, rel], axis=0)
                meff = li_ji(np.corrcoef(rk, rowvar=False))
                cl01 = clusters(rel & (R.p01 >= P_MARK).to_numpy(), adj)
                cl1 = clusters(rel & (R.p_above1 >= P_MARK).to_numpy(), adj)
                s.update(n_bounded=int((rel & R.bounded.fillna(False).to_numpy()).sum()),
                         frac01_point=float(np.mean((lam[rel] >= 0) & (lam[rel] <= 1))),
                         frac01_median=float(np.median(frac)),
                         frac01_lo=float(np.percentile(frac, 2.5)),
                         frac01_hi=float(np.percentile(frac, 97.5)),
                         n_p01_95=int((rel & (R.p01 >= P_MARK).to_numpy()).sum()),
                         n_above1_95=int((rel & (R.p_above1 >= P_MARK).to_numpy()).sum()),
                         n_below0_95=int((rel & (R.p_below0 >= P_MARK).to_numpy()).sum()),
                         lam_median=float(np.median(lam[rel])),
                         lam_q25=float(np.percentile(lam[rel], 25)),
                         lam_q75=float(np.percentile(lam[rel], 75)),
                         m_eff=meff,
                         clusters_p01=" | ".join(" ".join(c) for c in cl01),
                         cluster_sizes_p01=" ".join(str(len(c)) for c in cl01),
                         clusters_above1=" | ".join(" ".join(c) for c in cl1),
                         cluster_sizes_above1=" ".join(str(len(c)) for c in cl1))
                sum_rows.append(s)
                print(f"\n=== {model}, {band}, {cond.upper()} ===")
                print(f"  reliable channels {s['n_reliable']} (bounded {s['n_bounded']}); "
                      f"lambda* median {s['lam_median']:.2f} [IQR {s['lam_q25']:.2f}, "
                      f"{s['lam_q75']:.2f}]; in [0,1]: {s['frac01_point']:.2f} of channels, "
                      f"joint draws {s['frac01_median']:.2f} [{s['frac01_lo']:.2f}, "
                      f"{s['frac01_hi']:.2f}]; m_eff {meff:.1f}")
                print(f"  p01 >= {P_MARK}: {s['n_p01_95']} channels in clusters of "
                      f"{s['cluster_sizes_p01'] or '-'}; P(>1) >= {P_MARK}: "
                      f"{s['n_above1_95']} in clusters of {s['cluster_sizes_above1'] or '-'}; "
                      f"P(<0) >= {P_MARK}: {s['n_below0_95']}")

            # averaging orders and single channels, primary band
            rows, draws, T, B = draws_by_band["iaf_roi"]
            A = T - B
            Rch = pd.DataFrame(rows, index=CH)
            cfg = []                            # (row, draws) of each configuration
            for sname, chans in SETS.items():
                idx = np.array([CH.index(c) for c in chans])
                s = list(SETS).index(sname)
                St, Sb = z["set_tot"][keep, j, s], z["set_b"][keep, j, s]
                fitted = not np.isnan(B[:, idx]).all(0).any()
                ok = np.isfinite(A[:, idx]).all(1) & np.isfinite(B[:, idx]).all(1)
                allpos = (A[:, idx] > 0).all(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ml_a = np.exp(np.log(np.where(allpos[:, None], A[:, idx], np.nan)).mean(1))
                    ml_b = np.exp(np.log(B[:, idx]).mean(1))
                configs = {"spectrum_mean": (St - Sb, Sb, None),
                           "mean_log": (np.where(allpos, ml_a, -1.0), ml_b, ok),
                           "log_mean": (A[:, idx].mean(1), B[:, idx].mean(1), ok)}
                for cname, (ca, cb, cbase) in configs.items():
                    if cname != "spectrum_mean" and not fitted:
                        continue
                    base = None if cbase is None else cbase[:, None]
                    r, dr = crossover(x, ca[:, None], cb[:, None], G, base)
                    cfg.append((dict(model=model, cond=cond, set=sname,
                                     n_channels=len(chans), order=cname, **r[0]), dr[:, 0]))
                if not fitted:
                    continue
                ridx = idx[(Rch.share_pos.iloc[idx] >= RELIABLE).to_numpy()]
                if ridx.size == 0:
                    continue
                med = np.median(draws[:, ridx], axis=1)
                cfg.append((dict(model=model, cond=cond, set=sname,
                                 n_channels=len(ridx), order="median_lam_star",
                                 lam_star=float(np.median(Rch.lam_star.iloc[ridx])),
                                 hdi_lo=hdi(med)[0], hdi_hi=hdi(med)[1],
                                 p01=float(np.mean((med >= 0) & (med <= 1))),
                                 n=int(Rch.n.iloc[ridx].min())), med))
            for lab, ch in SINGLE.items():
                cfg.append((dict(model=model, cond=cond, set=f"{lab} ({ch})",
                                 n_channels=1, order="single", **Rch.loc[ch].to_dict()),
                            draws[:, CH.index(ch)]))
            # difference from the primary configuration, from the same draws
            ref_row, ref = cfg[0]
            for r, dr in cfg:
                dd = dr - ref
                r["diff_primary"] = r["lam_star"] - ref_row["lam_star"]
                r["diff_hdi_lo"], r["diff_hdi_hi"] = hdi(dd)
                r["p_diff_pos"] = float(np.mean(dd[np.isfinite(dd)] > 0))
                avg_rows.append(r)
            # all six ROI channels inside [0, 1] in the same draw
            roi_idx = [CH.index(c) for c in SETS["roi"]]
            dr = draws[:, roi_idx]
            sum_rows[-2]["p01_all_roi"] = float(np.mean(np.all((dr >= 0) & (dr <= 1), 1)))

    O = pd.DataFrame(chan_rows)
    O.to_csv(os.path.join(RES, "hbn_age_crossover_channels.csv"), index=False)
    S = pd.DataFrame(sum_rows)
    S.to_csv(os.path.join(RES, "hbn_age_crossover_channels_summary.csv"), index=False)
    V = pd.DataFrame(avg_rows)
    V.to_csv(os.path.join(RES, "hbn_age_crossover_averaging.csv"), index=False)

    pct = lambda s: 100 * (np.exp(s) - 1)
    print("\n=== averaging order (band IAF +/- 2 Hz of the ROI) ===")
    print(f"{'model':13s} {'cond':4s} {'set':10s} {'order':16s} {'s(0)':>7s} {'s(1)':>7s} "
          f"{'s_b':>7s} {'lam*':>6s} {'HDI':>15s} {'P01':>5s} {'n':>5s}  "
          f"{'diff from primary [HDI]':>24s}")
    for _, r in V.iterrows():
        sl = (f"{pct(r.s0):+7.2f} {pct(r.s1):+7.2f} {pct(r.s_b):+7.2f}"
              if np.isfinite(r.get("s0", np.nan)) else f"{'':7s} {'':7s} {'':7s}")
        print(f"{r.model:13s} {r.cond:4s} {r.set:10s} {r.order:16s} {sl} {r.lam_star:6.2f} "
              f"[{r.hdi_lo:5.2f},{r.hdi_hi:5.2f}] {r.p01:5.2f} {int(r.n):5d}  "
              f"{r.diff_primary:+6.2f} [{r.diff_hdi_lo:+5.2f},{r.diff_hdi_hi:+5.2f}]")
    print("\n=== ROI channels, band IAF +/- 2 Hz of the ROI ===")
    roi = O[(O.band == "iaf_roi") & O.channel.isin(SETS["roi"])]
    for _, r in roi.iterrows():
        print(f"  {r.model:13s} {r.cond} {r.channel:4s} lam* {r.lam_star:5.2f} "
              f"[{r.hdi_lo:5.2f},{r.hdi_hi:5.2f}] P01 {r.p01:.2f} n {r.n} "
              f"s(0) {pct(r.s0):+.2f} s(1) {pct(r.s1):+.2f} s_b {pct(r.s_b):+.2f} %/y")
    print(f"\nwrote {os.path.join(RES, 'hbn_age_crossover_channels.csv')}, "
          f"hbn_age_crossover_averaging.csv, hbn_age_crossover_channels_summary.csv")


if __name__ == "__main__":
    main()
