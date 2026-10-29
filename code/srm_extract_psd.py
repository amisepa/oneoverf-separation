"""Reduce the SRM resting-state EEG (OpenNeuro ds003775) to Welch spectra.

111 adults aged 17-71, 4 min eyes closed, 64 BioSemi channels at 1024 Hz;
42 were recorded again at a second session (ses-t2). Each EDF is
preprocessed as the other datasets (psd_utils: 0.4-100 Hz, 256 Hz, average
reference; the first 2 s dropped) and cut into 4-s Hann segments with
outlier rejection. One .npz per participant and session holds the mean
spectrum of every channel over kept segments and over odd/even halves, and
per-segment spectra for the posterior channel group.

Output: $SRM_OUT (default ./srm_psd). Finished files are skipped.

Usage: python srm_extract_psd.py [--root DIR] [--workers 6]
"""
import argparse
import csv
import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import psd_utils as U

OUT = os.environ.get("SRM_OUT", "srm_psd")
GROUPS = {"post": ["O1", "Oz", "O2", "PO3", "POz", "PO4"]}
SKIP_SEC = 2.0
# cleaned recordings (clean_adult.py: <EDF stem>.set) read instead of the EDFs
CLEANED = os.environ.get("SRM_CLEANED", "")


def read_edf(path):
    """MNE Raw; a header with an impossible start time (one file has second
    60) is read from a temporary copy with the time set to 00.00.00."""
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        except ValueError as e:
            if "must be in" not in str(e):
                raise
            tmp = os.path.join(OUT, "_fixed_" + os.path.basename(path))
            with open(path, "rb") as src, open(tmp, "wb") as dst:
                head = bytearray(src.read(256))
                head[176:184] = b"00.00.00"
                dst.write(head)
                dst.write(src.read())
            try:
                return mne.io.read_raw_edf(tmp, preload=True, verbose="ERROR")
            finally:
                os.remove(tmp)


def one(args):
    path, meta = args
    sub = os.path.basename(path).split("_")[0]
    ses = os.path.basename(path).split("_")[1].replace("ses-", "")
    outfile = os.path.join(OUT, f"{sub}_ses-{ses}.npz")
    if os.path.exists(outfile):
        return sub, ses, "cached"
    try:
        if CLEANED:
            import mne
            raw = mne.io.read_raw_eeglab(os.path.join(CLEANED, os.path.basename(path)[:-4] + ".set"),
                                         preload=True, verbose="ERROR")
        else:
            raw = read_edf(path)
        raw.pick("eeg")
        x, srate, names = U.preprocess(raw)
        a = int(SKIP_SEC * srate)
        P, f, _ = U.welch_segments(x, srate, [(a, x.shape[1])])
        if P is None:
            return sub, ses, "short"
        s, fr = U.summarise(P, f, U.reject(P), GROUPS, names)
    except Exception as e:
        return sub, ses, f"error {type(e).__name__}: {e}"
    np.savez_compressed(outfile, freqs=fr.astype(np.float32), ch_names=np.array(names),
                        subject=sub, session=ses, age=float(meta.get("age", "nan")),
                        sex=meta.get("sex", ""), dur=x.shape[1] / srate,
                        **{f"ec_{k}": v for k, v in s.items()})
    return sub, ses, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("EEG_DATA", "eeg_data"),
                                                   "ds003775"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(a.root, "participants.tsv")) as fh:
        meta = {r["participant_id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    files = sorted(glob.glob(os.path.join(a.root, "sub-*", "ses-*", "eeg", "*_eeg.edf")))
    jobs = [(p, meta.get(os.path.basename(p).split("_")[0], {})) for p in files]
    print(f"{len(jobs)} recordings", flush=True)
    counts = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for sub, ses, st in ex.map(one, jobs):
            counts[st if st in ("ok", "cached") else "failed"] = \
                counts.get(st if st in ("ok", "cached") else "failed", 0) + 1
            if st not in ("ok", "cached"):
                print(sub, ses, st, flush=True)
    print(counts, flush=True)


if __name__ == "__main__":
    main()
