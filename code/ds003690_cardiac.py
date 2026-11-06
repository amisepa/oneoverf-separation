"""Cardiac share of the posterior background and what it does to lambda (ds003690).

Schmidt et al. (eLife 100605) report that the ECG has its own aperiodic
component, that ICA leaves it in EEG, and that it explains part of the age
effect on the "cortical" exponent. Our ICA + ICLabel cleaning removes only eye
and muscle components, so a cardiac part of b is untested. ds003690 has an ECG
channel and young and older adults.

extract: per participant, the EEG preprocessed exactly as ds003690_epochs.py
(0.4-100 Hz, 250 Hz, average reference) and the ECG filtered and resampled the
same way. Each posterior ROI channel is regressed on the ECG at lags -0.1 to
+0.5 s (a temporal response function; one least-squares solve for the six
channels), cross-fitted: the weights from one half of the recording predict the
other half, so only what generalises is removed. Null: the same with the ECG
shifted circularly by half the recording. The 2-s epochs, three DPSS tapers and
covariates of ds003690_epochs.py are kept; per epoch the ROI spectrum of the
original EEG ("post", identical to ds003690_epochs.py), of the EEG minus the
cardiac prediction ("post_clean"), of the prediction itself ("post_card"), the
same two for the null ("post_nclean", "post_ncard") and the ECG spectrum.
The ECG lead can also pick up EEG, so the same regression is repeated on an
impulse train at the R peaks (MNE's detector; a detection within 0.3 s of the
previous one dropped), lags -0.25 to +0.75 s: this fits the average
heartbeat-locked waveform, overlaps included, and contains no EEG
("post_rclean", "post_rcard"; null "post_nrclean", "post_nrcard").
The out-of-sample prediction carries its own estimation noise, which is
independent of the held-out EEG, so |Y - pred|^2 overstates the cleaned power
(by 2% at 2-6 Hz in sub-AB10). The cardiac power is therefore estimated from
the cross-spectrum, Re(Y conj(pred)) = (post + post_card - post_clean) / 2,
which that noise does not bias, and the cleaned spectrum as post minus it
("dclean"; "ndclean" for the null). That estimate is unbiased on average but
noisy in a single epoch and taper, where it can exceed the total power, so it
is used only for mean spectra; per-epoch analyses use the residual spectra
|Y - pred|^2 ("dres", "rres", "nrres"), which stay positive at the price of the
small prediction noise.

analyse: (1) cardiac share (cross-spectral) of the background in the alpha
band and in its flanks, young vs older; (2) the within-session estimate of
lambda (ds003690_lambda.run, flank background) on the original and the
residual spectra (ECG, R peaks, R-peak null), all on the participants and band
of the original; (3) the older-vs-younger crossover lambda* on the mean
spectrum per participant, original and cross-spectrally cleaned
(aging_lambda's band power: 2-45 Hz, 6-16 Hz censored, power law and knee +
plateau; sex as covariate); (4) the ECG's own exponent, young vs older.

Usage: python ds003690_cardiac.py extract [--data DIR] [--workers 8]
       python ds003690_cardiac.py analyse [--nboot 200]
Writes $DS003690_CARDIAC (default ./ds003690_cardiac) and
results/ds003690_cardiac.csv.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import glob
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal.windows import dpss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.environ.get("DS003690_CARDIAC", "ds003690_cardiac")
POST = ["O1", "Oz", "O2", "PO3", "POz", "PO4"]
TEMP = ["T7", "T8", "TP7", "TP8", "FT7", "FT8"]
EPOCH = 2.0
GAP = 1.5
LAGS = (-0.1, 0.5)                                   # s; positive: EEG after ECG
RLAGS = (-0.25, 0.75)                                # s around each R peak
SIGNALS = ("post", "post_clean", "post_card", "post_nclean", "post_ncard",
           "post_rclean", "post_rcard", "post_nrclean", "post_nrcard")


def cardiac_part(Y, ecg, sr, shift=0, lags=LAGS):
    """Cross-fitted prediction of Y (n x k) from the lagged regressor (n,)."""
    lo, hi = int(round(lags[0] * sr)), int(round(lags[1] * sr))
    e = np.roll(ecg, shift)
    E = np.r_[np.zeros(hi), e, np.zeros(-lo)]
    X = sliding_window_view(E, hi - lo + 1)                     # n x lags, X[t, m] = ecg[t + m - hi]
    n = Y.shape[0]
    halves = (slice(0, n // 2), slice(n // 2, n))
    pred = np.empty_like(Y)
    for fit, out in (halves, halves[::-1]):
        Xf = np.column_stack([X[fit], np.ones(X[fit].shape[0])])
        w = np.linalg.lstsq(Xf, Y[fit], rcond=None)[0]           # (lags + 1) x k
        pred[out] = X[out] @ w[:-1] + w[-1]
    return pred


def one(args):
    path, meta = args
    sub = os.path.basename(path).split("_")[0]
    outfile = os.path.join(OUT, f"{sub}.npz")
    if os.path.exists(outfile):
        return sub, "cached"
    import mne
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        types = {}
        for c in raw.ch_names:
            u = c.upper()
            if u.startswith("VEO") or u.startswith("HEO"):
                types[c] = "eog"
            elif u.startswith("EKG") or u.startswith("ECG"):
                types[c] = "ecg"
            elif "DIA" in u:
                types[c] = "misc"
        raw.set_channel_types(types)
        if "ecg" not in raw:
            return sub, "no-ecg"
        pupil = raw.copy().pick("misc").get_data() if "misc" in raw else None
        sr0 = raw.info["sfreq"]
        ecg_name = [c for c, t in types.items() if t == "ecg"][0]
        rp = mne.preprocessing.find_ecg_events(raw, ch_name=ecg_name, verbose="ERROR")[0][:, 0]
        rp = rp - raw.first_samp
        rp = rp[np.r_[True, np.diff(rp) >= 0.3 * sr0]]
        rr = np.diff(rp) / sr0
        # ECG: the EEG's filter and resampling (psd_utils.preprocess), no reference
        re_ = raw.copy().pick("ecg")
        dec = max(1, int(round(sr0 / U.SRATE_TARGET)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            re_.filter(0.4, min(sr0 / dec / 2 - 5, 100.0), picks="all", fir_design="firwin",
                       verbose="ERROR")
            if dec > 1:
                re_.resample(sr0 / dec, verbose="ERROR")
        ecg = re_.get_data()[0] * 1e6
        ev = pd.read_csv(path.replace("_eeg.set", "_events.tsv"), sep="\t")
        cues = ev.loc[ev.trial_type == "cue", "onset"].to_numpy(float)
        raw.pick(["eeg", "eog"])
        x, sr, names = U.preprocess(raw)
        ipost = [names.index(c) for c in POST if c in names]
        itemp = [names.index(c) for c in TEMP if c in names]
        ieog = [names.index(c) for c in names if c.upper().startswith(("VEO", "HEO"))]
        m = min(x.shape[1], ecg.size)
        x, ecg = x[:, :m], ecg[:m]
        Y = x[ipost].T                                            # n x 6
        card = cardiac_part(Y, ecg, sr)
        ncard = cardiac_part(Y, ecg, sr, shift=m // 2)
        imp = np.zeros(m)
        ir = np.round(rp / dec).astype(int)
        imp[ir[ir < m]] = 1.0
        rcard = cardiac_part(Y, imp, sr, lags=RLAGS)
        nrcard = cardiac_part(Y, imp, sr, shift=m // 2, lags=RLAGS)
        # channels: ROI original, cleaned, cardiac, null-cleaned, null-cardiac,
        # the same four from the R peaks, ECG
        S = np.concatenate([Y, Y - card, card, Y - ncard, ncard,
                            Y - rcard, rcard, Y - nrcard, nrcard, ecg[:, None]], 1).T
        k = len(ipost)
        n = int(EPOCH * sr)
        tapers = dpss(n, 2.0, 3)
        f = np.fft.rfftfreq(n, 1 / sr)
        bounds = list(zip(cues + GAP, np.r_[cues[1:], cues[-1] + 8.0]))
        rows, spec, ecgs = [], [], []
        for kk, (a, b) in enumerate(bounds):
            for s in np.arange(a, b - EPOCH + 1e-9, EPOCH):
                i0 = int(s * sr)
                if i0 + n > m:
                    continue
                seg = x[:, i0:i0 + n]
                seg = seg - seg.mean(1, keepdims=True)
                P = np.abs(np.fft.rfft(seg[None, :, :] * tapers[:, None, :], axis=2)) ** 2 / sr
                P[:, :, 1:-1] *= 2
                eogp = np.log(P.mean(0)[ieog][:, (f >= 0.5) & (f <= 4)].mean(1))
                emg = np.log(P.mean(0)[itemp][:, (f >= 60) & (f <= 95)].mean())
                sg = S[:, i0:i0 + n]
                sg = sg - sg.mean(1, keepdims=True)
                Q = np.abs(np.fft.rfft(sg[None, :, :] * tapers[:, None, :], axis=2)) ** 2 / sr
                Q[:, :, 1:-1] *= 2                                # taper x channel x f
                spec.append(np.stack([Q[:, j * k:(j + 1) * k].mean(1) for j in range(len(SIGNALS))]))
                ecgs.append(Q[:, -1].mean(0))
                pu = np.nan
                if pupil is not None:
                    j0, j1 = int(s * sr0), int((s + EPOCH) * sr0)
                    v = pupil[:, j0:j1]
                    v = v[np.isfinite(v) & (v > 0)]
                    pu = float(np.mean(v)) if v.size > 0.5 * (j1 - j0) else np.nan
                rows.append(dict(t=s, interval=kk, pupil=pu, emg=emg,
                                 veog=eogp[0] if len(eogp) > 0 else np.nan,
                                 heog=eogp[1] if len(eogp) > 1 else np.nan))
        if not rows:
            return sub, "no-epochs"
        R = pd.DataFrame(rows)
        spec = np.stack(spec).astype(np.float32)                  # epochs x signal x taper x f
        r2 = lambda p: float(1 - np.sum((Y - p) ** 2) / np.sum(Y ** 2))
        np.savez_compressed(outfile, freqs=f.astype(np.float32),
                            **{nm: spec[:, j] for j, nm in enumerate(SIGNALS)},
                            ecg=np.stack(ecgs).astype(np.float32),
                            r2=r2(card), r2_null=r2(ncard), r2_r=r2(rcard), r2_rnull=r2(nrcard),
                            n_rpeaks=len(rp), rr_median=float(np.median(rr)),
                            rr_odd=float(np.mean((rr < 0.4) | (rr > 1.5))),
                            subject=sub, age=float(meta["age"]), sex=meta["sex"],
                            group=meta["group"], **{c: R[c].to_numpy() for c in R.columns})
        return sub, f"ok {len(R)}"
    except Exception as e:
        return sub, f"error {type(e).__name__}: {e}"


def extract(a):
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv(os.path.join(a.data, "participants.tsv"), sep="\t").set_index("participant_id")
    files = sorted(glob.glob(os.path.join(a.data, "sub-*", "eeg", "*task-passive*_eeg.set")))
    jobs = [(p, meta.loc[os.path.basename(p).split("_")[0]].to_dict()) for p in files]
    print(f"{len(jobs)} recordings", flush=True)
    if a.workers == 1:                       # serial (a spawned pool can fail in some shells)
        results = map(one, jobs)
    else:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"           # inherited by the spawned workers
        ex = ProcessPoolExecutor(max_workers=a.workers)
        results = (fu.result() for fu in as_completed([ex.submit(one, j) for j in jobs]))
    for sub, st in results:
        if not st.startswith(("ok", "cached")):
            print(sub, st, flush=True)
    print("done", flush=True)


def load_units(signal):
    """Units in ds003690_lambda.load's format. signal: "post" (original);
    "dclean" (ECG-predicted cross-spectrum removed), "rclean" (R-peak-locked
    cross-spectrum removed), "ndclean"/"nrclean" (the same with the shifted
    regressor), which can be negative in single epochs and serve mean spectra
    only; or the residual spectra "dres", "rres", "nrres" for per-epoch work.
    Pcard, Pncard, Prcard, Pnrcard: the cross-spectral powers."""
    units = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        P0 = d["post"].astype(float)
        card = (P0 + d["post_card"] - d["post_clean"]) / 2
        ncard = (P0 + d["post_ncard"] - d["post_nclean"]) / 2
        rcard = (P0 + d["post_rcard"] - d["post_rclean"]) / 2
        nrcard = (P0 + d["post_nrcard"] - d["post_nrclean"]) / 2
        P = {"post": P0, "dclean": P0 - card, "ndclean": P0 - ncard,
             "rclean": P0 - rcard, "nrclean": P0 - nrcard,
             "dres": d["post_clean"].astype(float), "rres": d["post_rclean"].astype(float),
             "nrres": d["post_nrclean"].astype(float)}[signal]
        units.append(dict(subject=str(d["subject"]), group=str(d["group"]), age=float(d["age"]),
                          sex=str(d["sex"]), f=d["freqs"].astype(float),
                          P=P, Pcard=card, Pncard=ncard,
                          Prcard=rcard, Pnrcard=nrcard, ecg=d["ecg"].astype(float),
                          r2=float(d["r2"]), r2_null=float(d["r2_null"]),
                          r2_r=float(d["r2_r"]), r2_rnull=float(d["r2_rnull"]),
                          rr_odd=float(d["rr_odd"]), n_rpeaks=int(d["n_rpeaks"]),
                          cov=np.column_stack([d["pupil"], d["veog"], d["heog"], d["emg"]])))
    return units


def analyse(a):
    import ds003690_lambda as L
    from ap_models import fit_aperiodic, flank_bins, flank_level
    from lambda_curve import band_power, effect_curve
    rng = np.random.default_rng(0)
    out = []
    base = load_units("post")
    f = base[0]["f"]
    # (1) cardiac share of the background: alpha band (IAF +/- 2) and the flanks
    sh = []
    bins = np.where(flank_bins(f, L.FLANKS))[0]
    for u in base:
        Pm = u["P"].mean(axis=(0, 1))
        cf = L.iaf(Pm, f)
        if not np.isfinite(cf):
            continue
        band = (f >= cf - 2) & (f <= cf + 2)
        B = flank_level(u["P"].mean(1), f, bins, band, L.FLANKS, Pm).mean()
        fl = flank_bins(f, L.FLANKS)
        row = dict(subject=u["subject"], group=u["group"], iaf=cf, r2=u["r2"], r2_null=u["r2_null"],
                   r2_r=u["r2_r"], r2_rnull=u["r2_rnull"], rr_odd=u["rr_odd"])
        for tag, key in (("", "Pcard"), ("_null", "Pncard"), ("_r", "Prcard"), ("_rnull", "Pnrcard")):
            Pc = u[key].mean(axis=(0, 1))
            row["share_band" + tag] = Pc[band].mean() / B
            row["share_flanks" + tag] = Pc[fl].mean() / Pm[fl].mean()
        sh.append(row)
    SH = pd.DataFrame(sh)
    print("cardiac share (median [IQR]) by group:")
    for g, G in SH.groupby("group"):
        q = lambda c: f"{G[c].median():.4f} [{G[c].quantile(.25):.4f}, {G[c].quantile(.75):.4f}]"
        print(f"  {g:6s} n {len(G)}  ECG: band {q('share_band')} null {q('share_band_null')}  "
              f"flanks {q('share_flanks')} null {q('share_flanks_null')}  ROI R2 {q('r2')} null {q('r2_null')}")
        print(f"  {'':6s} R peaks: band {q('share_band_r')} null {q('share_band_rnull')}  "
              f"flanks {q('share_flanks_r')} null {q('share_flanks_rnull')}  ROI R2 {q('r2_r')} null {q('r2_rnull')}")
        cols = [c for c in SH.columns if c.startswith(("share", "r2"))]
        for c in cols:
            out.append(dict(analysis="cardiac share", group=g, measure=c, est=G[c].median(),
                            lo=G[c].quantile(.25), hi=G[c].quantile(.75), n=len(G)))
    # (2) within-session lambda, flank background, no covariates; every signal on
    # the participants and band (IAF +/- 2) of the original spectra, since
    # cleaning can push a weak peak under iaf's threshold or move it by a bin
    iafs = {u["subject"]: L.iaf(u["P"].mean(axis=(0, 1)), f) for u in base}

    def band_rows(units):
        rows = []
        for u in units:
            cf = iafs[u["subject"]]
            if not np.isfinite(cf):
                continue
            P = u["P"]
            band = (f >= cf - 2) & (f <= cf + 2)
            B = np.column_stack([flank_level(P[:, j], f, bins, band, L.FLANKS, P.mean(axis=(0, 1)))
                                 for j in range(P.shape[1])])
            rows.append(dict(subject=u["subject"], group=u["group"], T=P[..., band].mean(-1), B=B,
                             cov=u["cov"], iaf=cf))
        return rows

    for signal in ("post", "dres", "rres", "nrres"):
        units = base if signal == "post" else load_units(signal)
        rows = band_rows(units)
        for grp in ("all", "Young", "older"):
            rr = rows if grp == "all" else [r for r in rows if r["group"].lower() == grp.lower()]
            for cname in ("none",):
                r = L.run(rr, L.COVSETS[cname], a.nboot, rng)
                print(f"  within {signal:11s} {grp:6s} cov={cname:6s} lambda {r['lam']:+.3f} "
                      f"[{r['lo']:+.2f}, {r['hi']:+.2f}] fail {r['fail']:.2f} n {r['n_units']}", flush=True)
                out.append(dict(analysis="within-session lambda", signal=signal, group=grp,
                                covariates=cname, est=r["lam"], lo=r["lo"], hi=r["hi"],
                                fail=r["fail"], n=r["n_units"]))
    # (3) older vs younger crossover on the mean spectrum per participant
    for signal in ("post", "dclean", "rclean"):
        units = base if signal == "post" else load_units(signal)
        for model in ("fixed", "knee_plateau"):
            rows = []
            for u in units:
                cf = iafs[u["subject"]]
                if not np.isfinite(cf):
                    continue
                bp = band_power(u["P"].mean(axis=(0, 1)), f, (cf - 2, cf + 2), (2.0, 45.0),
                                ((6.0, 16.0),), model)
                if bp is not None and bp["a"] > 0:
                    rows.append(dict(subject=u["subject"], older=float(u["group"] == "Older"),
                                     male=float(u["sex"] == "M"), a=bp["a"], b=bp["b"]))
            D = pd.DataFrame(rows)
            r = effect_curve(np.log(D.a), np.log(D.b), x=D.older.to_numpy(), covariates=D.male.to_numpy(),
                             rng=np.random.default_rng(0))
            print(f"  age {signal:11s} {model:12s} n {r['n']} s_a {r['s_a']:+.3f} s_b {r['s_b']:+.3f} "
                  f"[{r['s_b_ci'][0]:+.3f}, {r['s_b_ci'][1]:+.3f}] lambda* {r['lam_star']:.2f} "
                  f"HDI [{r['hdi'][0]:.2f}, {r['hdi'][1]:.2f}] P(in [0,1]) {r['p_cross_in_01']:.2f}", flush=True)
            out.append(dict(analysis="older-younger lambda*", signal=signal, model=model, n=r["n"],
                            s_a=r["s_a"], s_b=r["s_b"], s_b_lo=r["s_b_ci"][0], s_b_hi=r["s_b_ci"][1],
                            bounded=r["lam_star_bounded"], est=r["lam_star"], lo=r["hdi"][0],
                            hi=r["hdi"][1], p_in_01=r["p_cross_in_01"]))
    # (4) the ECG's own exponent (power law, 2-45 Hz), older minus younger
    ex = []
    for u in base:
        Pe = u["ecg"].mean(0)
        sel = (f >= 2) & (f <= 45)
        fit = fit_aperiodic(np.clip(Pe[sel], 1e-12, None), f[sel], np.ones(sel.sum(), bool), "fixed")
        if fit is not None:
            ex.append(dict(group=u["group"], exponent=fit["exponent"]))
    E = pd.DataFrame(ex)
    for g, G in E.groupby("group"):
        print(f"  ECG exponent {g:6s} median {G.exponent.median():.2f} (n {len(G)})")
        out.append(dict(analysis="ECG exponent", group=g, est=G.exponent.median(),
                        lo=G.exponent.quantile(.25), hi=G.exponent.quantile(.75), n=len(G)))
    pd.DataFrame(out).to_csv(os.path.join(RES, "ds003690_cardiac.csv"), index=False)
    SH.to_csv(os.path.join(OUT, "cardiac_share_per_participant.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("extract", "analyse"))
    ap.add_argument("--data", default=os.path.join(os.environ.get("EEG_DATA", "eeg_data"), "ds003690"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nboot", type=int, default=200)
    a = ap.parse_args()
    extract(a) if a.step == "extract" else analyse(a)


if __name__ == "__main__":
    main()
