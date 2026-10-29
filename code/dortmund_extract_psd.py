"""Download the Dortmund Vital Study resting EEG (OpenNeuro ds005385) and
reduce each recording to Welch spectra.

608 adults aged 20-70 (208 re-tested about 5 years later), 64 channels at
1000 Hz referenced to FCz, 3 min eyes closed and 3 min eyes open, recorded
before and after a 2-hour task battery. Each EDF is fetched, preprocessed
(psd_utils: 0.4-100 Hz, 250 Hz, average reference; the first 2 s dropped),
cut into 4-s Hann segments with outlier rejection, and deleted. One .npz per
participant and session holds, for each of ec_pre, eo_pre, ec_post, eo_post,
the mean spectrum of every channel over kept segments and over odd/even
halves, and per-segment spectra for posterior, frontopolar and temporal
channel groups (for epoch-level analyses and eye/muscle proxies).

Output: $DORTMUND_OUT (default ./dortmund_psd). Finished files are skipped.

Usage: python dortmund_extract_psd.py [--workers 6] [--limit N]
"""
import argparse
import csv
import io
import os
import time
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import psd_utils as U

S3 = "https://s3.amazonaws.com/openneuro.org/ds005385"
OUT = os.environ.get("DORTMUND_OUT", "dortmund_psd")
TMP = os.environ.get("DORTMUND_TMP", "dortmund_tmp")
GROUPS = {"post": ["O1", "Oz", "O2", "PO3", "POz", "PO4"],
          "fp": ["Fp1", "Fp2"],
          "temp": ["T7", "T8", "FT7", "FT8", "TP7", "TP8"]}
RECS = [("ec", "EyesClosed"), ("eo", "EyesOpen")]
SKIP_SEC = 2.0
# cleaned recordings (clean_adult.py: <EDF stem>.set, the FCz reference
# restored) read instead of the downloads; FCz is dropped again so the
# average reference is the uncleaned one's
CLEANED = os.environ.get("DORTMUND_CLEANED", "")
REF = "FCz"


def file_name(sub, ses, task, acq):
    return f"{sub}_ses-{ses}_task-{task}_acq-{acq}_eeg.edf"


def fetch(url, dest, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404 or k == tries - 1:
                return False
            time.sleep(2 ** k)
        except Exception:
            if k == tries - 1:
                return False
            time.sleep(2 ** k)
    return False


def participants():
    txt = urllib.request.urlopen(f"{S3}/participants.tsv", timeout=120).read().decode()
    return list(csv.DictReader(io.StringIO(txt), delimiter="\t"))


def one(args):
    sub, ses, meta = args
    outfile = os.path.join(OUT, f"{sub}_ses-{ses}.npz")
    if os.path.exists(outfile):
        return sub, ses, "cached"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    import mne
    save = {}
    status = []
    freqs = ch = None
    for acq in ("pre", "post"):
        for key, task in RECS:
            name = file_name(sub, ses, task, acq)
            if CLEANED:
                path = os.path.join(CLEANED, name[:-4] + ".set")
                if not os.path.exists(path):
                    status.append(f"{key}_{acq}:missing")
                    continue
            else:
                path = os.path.join(TMP, name)
                if not fetch(f"{S3}/{sub}/ses-{ses}/eeg/{name}", path):
                    status.append(f"{key}_{acq}:missing")
                    continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if CLEANED:
                        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
                        raw.drop_channels([REF])
                    else:
                        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
                    raw.pick("eeg")
                x, srate, names = U.preprocess(raw)
                a = int(SKIP_SEC * srate)
                P, f, _ = U.welch_segments(x, srate, [(a, x.shape[1])])
                if P is None:
                    status.append(f"{key}_{acq}:short")
                    continue
                s, fr = U.summarise(P, f, U.reject(P), GROUPS, names)
                for k2, v in s.items():
                    save[f"{key}_{acq}_{k2}"] = v
                save[f"{key}_{acq}_dur"] = x.shape[1] / srate
                freqs, ch = fr, names
                status.append(f"{key}_{acq}:ok")
            except Exception as e:
                status.append(f"{key}_{acq}:error {type(e).__name__}")
            finally:
                if not CLEANED:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
    if freqs is None:
        return sub, ses, ";".join(status)
    np.savez_compressed(outfile, freqs=freqs.astype(np.float32), ch_names=np.array(ch),
                        subject=sub, session=ses, age=float(meta["age"]),
                        sex=meta["sex"], handedness=meta["handedness"],
                        late=meta.get(f"late_ses{ses}", ""), status=";".join(status),
                        **save)
    return sub, ses, ";".join(status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    P = participants()
    if a.limit:
        P = P[:a.limit]
    jobs = [(p["participant_id"], s, p) for p in P for s in (1, 2)
            if p.get(f"session{s}") == "yes"]
    print(f"{len(jobs)} participant-sessions queued", flush=True)
    t0 = time.time()
    counts = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs)):
            sub, ses, st = fu.result()
            ok = st == "cached" or st.count(":ok") == 4
            counts["complete" if ok else "partial"] = counts.get("complete" if ok else "partial", 0) + 1
            if (k + 1) % 20 == 0 or not ok:
                print(f"[{k + 1}/{len(jobs)}] {(time.time() - t0) / 60:6.1f} min "
                      f"{sub} ses-{ses} {st} {counts}", flush=True)
    print(f"FINISHED {len(jobs)} in {(time.time() - t0) / 60:.1f} min {counts}", flush=True)


if __name__ == "__main__":
    main()
